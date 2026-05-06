"""Stage 0a — build the global galaxy registry from LVDB v1.0.5.

Implements docs/plan/data_sources.md §"How this changes Stage 0 of the pipeline" /
"Filtering rules at ingest" / "Transformations we apply".

Run:
    python -m data_ingest.stage0a_registry

Outputs:
    data/registry/galaxies.ecsv
    data/registry/build_log.txt

Failures are loud: a checksum mismatch, an unknown LVDB key from the study sample,
a host that doesn't transitively resolve to MW, a missing required field, or a
sphericalized-radius cross-check failure all abort the build.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from astropy import units as u

from data_ingest.staging import verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[1]
LVDB_DIR = REPO_ROOT / "data" / "lvdb_v1.0.5"
LVDB_CSV = LVDB_DIR / "comb_all.csv"
REGISTRY_DIR = REPO_ROOT / "data" / "registry"
STUDY_SAMPLE_YAML = REPO_ROOT / "data_ingest" / "config" / "study_sample.yaml"
SPATIAL_OVERRIDES_YAML = REPO_ROOT / "data_ingest" / "config" / "spatial_model_overrides.yaml"

ARCMIN_TO_RAD = math.pi / (180.0 * 60.0)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _walk_host(host: str, host_map: dict[str, str], visited: set[str]) -> list[str]:
    """Return the host chain for a given host, terminating at 'mw' or a non-host."""
    chain = [host]
    cur = host
    while cur != "mw":
        if cur in visited:
            raise RuntimeError(f"Host cycle detected: {chain}")
        visited.add(cur)
        if cur not in host_map:
            return chain  # terminates at non-MW root (e.g. m31, lg, ...)
        cur = host_map[cur]
        chain.append(cur)
    return chain


def _load_study_sample() -> list[dict]:
    raw = yaml.safe_load(STUDY_SAMPLE_YAML.read_text())["galaxies"]
    if len({g["lvdb_key"] for g in raw}) != len(raw):
        raise RuntimeError("Duplicate lvdb_key in study_sample.yaml")
    return raw


def _load_spatial_overrides() -> tuple[str, dict[str, str]]:
    cfg = yaml.safe_load(SPATIAL_OVERRIDES_YAML.read_text())
    return cfg["default"], cfg["overrides"] or {}


def _resolve_profile(lvdb_key: str, default: str, overrides: dict[str, str]) -> str:
    return overrides.get(lvdb_key, default)


def _plummer_radius_pc(profile: str, r_half_2d_pc: float, lvdb_row: pd.Series) -> float:
    """Per-profile Plummer scale radius in pc.

    For non-Plummer profiles, the conversion uses the LVDB structural columns
    that match the profile (rad_sersic, rcore, etc.). For now only the Plummer
    case is exercised — overrides YAML is empty by default — but the dispatch
    is here so adding an override doesn't require code changes downstream.
    """
    if profile == "plummer":
        return r_half_2d_pc
    if profile == "exponential":
        # P&S 2018 §3 footnote 4: r_p = 1.68 * r_exp.
        # LVDB does not currently expose a separate r_exponential column for our
        # systems; if an override is added, the per-galaxy r_exp must be sourced
        # explicitly. Fail loudly until this is resolved.
        raise NotImplementedError(
            f"Exponential profile override needs an explicit r_exponential source "
            f"for LVDB key {lvdb_row['key']!r}; not implemented."
        )
    if profile in ("sersic", "king"):
        # Non-P&S extension per data_sources.md: r_p = r_1/2.
        return r_half_2d_pc
    raise ValueError(f"Unknown profile {profile!r} for {lvdb_row['key']!r}")


def build_registry() -> Table:
    verify_checksums(LVDB_DIR)
    df = pd.read_csv(LVDB_CSV)
    df["key"] = df["key"].astype(str)

    study = _load_study_sample()
    study_keys = [g["lvdb_key"] for g in study]
    missing = [k for k in study_keys if k not in set(df["key"])]
    if missing:
        raise RuntimeError(f"Study-sample LVDB keys not found in comb_all.csv: {missing}")

    sub = df[df["key"].isin(study_keys)].copy()
    if len(sub) != len(study_keys):
        raise RuntimeError(
            f"Expected {len(study_keys)} rows after study-sample filter, got {len(sub)}"
        )

    # Confirmed-dwarf filter (column is float 1.0/0.0 in v1.0.5).
    not_dwarf = sub[sub["confirmed_dwarf"].fillna(0.0) < 0.5]
    if len(not_dwarf):
        raise RuntimeError(
            f"Study-sample galaxies failing confirmed_dwarf filter: {list(not_dwarf['key'])}"
        )

    # Required-fields filter — abort if any study-sample row is missing them.
    for col in ("rhalf", "distance_modulus", "apparent_magnitude_v"):
        miss = sub[sub[col].isna()]
        if len(miss):
            raise RuntimeError(f"Required column {col!r} missing for: {list(miss['key'])}")

    # Host-walking (transitive to MW). The host map is built from the full LVDB,
    # so subhalo hosts (e.g., lmc -> mw) resolve correctly even though the
    # subhalo row itself may not be in the study sample.
    host_map = dict(zip(df["key"], df["host"]))
    host_chains: dict[str, list[str]] = {}
    for k, h in zip(sub["key"], sub["host"]):
        chain = _walk_host(h, host_map, visited=set())
        if chain[-1] != "mw":
            raise RuntimeError(f"{k}: host chain does not resolve to mw: {chain}")
        host_chains[k] = [h] + chain[1:] if h != chain[0] else chain  # noqa

    # Spatial-model resolution.
    spatial_default, spatial_overrides = _load_spatial_overrides()
    profile_for = {k: _resolve_profile(k, spatial_default, spatial_overrides) for k in sub["key"]}

    # Reorder rows to match study-sample order, and pull in study metadata.
    sub = sub.set_index("key").loc[study_keys].reset_index()
    study_by_key = {g["lvdb_key"]: g for g in study}

    rows = []
    log_lines = []
    for _, r in sub.iterrows():
        key = r["key"]
        meta = study_by_key[key]

        mu = float(r["distance_modulus"])
        d_kpc = 10.0 ** ((mu - 10.0) / 5.0)  # plan §"Transformations we apply" eq. 1
        rhalf_arcmin = float(r["rhalf"])
        eps_raw = r["ellipticity"]
        eps = float(eps_raw) if pd.notna(eps_raw) else math.nan
        # Per pipeline_overview.md Stage 0a: when ellipticity is missing or only an
        # upper limit (e.g. crater_2 has only ellipticity_ul=0.10), treat ε as
        # missing — the registry stores ellipticity_missing=True and Stage 2 will
        # substitute a uniform U(0, 0.5) prior. For the deterministic registry
        # median we use ε_for_calc = 0 (no sphericalization), which is the right
        # central value under that prior. The upper-limit value itself is preserved
        # in `ellipticity_ul` for diagnostic / Stage-2 visibility.
        eps_for_calc = eps if not math.isnan(eps) else 0.0
        ellipticity_missing = bool(math.isnan(eps))

        rhalf_major_pc = rhalf_arcmin * ARCMIN_TO_RAD * d_kpc * 1000.0
        r_half_2d_pc = rhalf_major_pc * math.sqrt(max(0.0, 1.0 - eps_for_calc))

        profile = profile_for[key]
        r_p_pc = _plummer_radius_pc(profile, r_half_2d_pc, r)

        # Cross-checks vs LVDB derived columns. The LVDB's rhalf_*_physical central
        # values are MC medians over the joint (rhalf, ε, distance) errors, not the
        # deterministic combination of central inputs, so a small offset is expected
        # even when our transformation is bug-free. Empirically across the 39-galaxy
        # sample (verified 2026-05-05) the offsets are bounded by:
        #   - rhalf_major_pc vs LVDB rhalf_physical:        worst 1.6% (tucana_5)
        #   - r_half_2d_pc   vs LVDB rhalf_sph_physical:    worst 5.5% (leo_5)
        # The major-axis check is much tighter because no concave √(1−ε) is involved;
        # the sphericalized check is loose because Jensen's inequality on √(1−ε)
        # systematically lowers the LVDB MC median when ε has appreciable error.
        # Bounds set to 3% and 7% — both still catch unit / sphericalization / sign
        # errors comfortably (any such error would be O(60×) or larger).
        lvdb_major = r["rhalf_physical"]
        if pd.notna(lvdb_major) and lvdb_major > 0:
            rel = abs(rhalf_major_pc - float(lvdb_major)) / float(lvdb_major)
            if rel >= 3e-2:
                raise RuntimeError(
                    f"{key}: rhalf_major cross-check failed: ours={rhalf_major_pc:.6g} pc, "
                    f"LVDB={float(lvdb_major):.6g} pc, rel diff={rel:.3e}"
                )
        lvdb_sph = r["rhalf_sph_physical"]
        if pd.notna(lvdb_sph) and lvdb_sph > 0:
            rel = abs(r_half_2d_pc - float(lvdb_sph)) / float(lvdb_sph)
            if rel >= 7e-2:
                raise RuntimeError(
                    f"{key}: r_1/2 cross-check failed: ours={r_half_2d_pc:.6g} pc, "
                    f"LVDB={float(lvdb_sph):.6g} pc, rel diff={rel:.3e}"
                )

        vlos_sigma_missing = bool(pd.isna(r["vlos_sigma"]))

        rows.append({
            "lvdb_key": key,
            "study_name": meta["study_name"],
            "path": meta["path"],
            "geha_galaxy": meta.get("geha_galaxy") or "",
            "host": str(r["host"]),
            "host_chain": "->".join(host_chains[key]),
            "ra_deg": float(r["ra"]),
            "dec_deg": float(r["dec"]),
            "rhalf_arcmin": rhalf_arcmin,
            "rhalf_arcmin_em": float(r["rhalf_em"]) if pd.notna(r["rhalf_em"]) else math.nan,
            "rhalf_arcmin_ep": float(r["rhalf_ep"]) if pd.notna(r["rhalf_ep"]) else math.nan,
            "ellipticity": eps,
            "ellipticity_em": float(r["ellipticity_em"]) if pd.notna(r["ellipticity_em"]) else math.nan,
            "ellipticity_ep": float(r["ellipticity_ep"]) if pd.notna(r["ellipticity_ep"]) else math.nan,
            "ellipticity_ul": float(r["ellipticity_ul"]) if pd.notna(r["ellipticity_ul"]) else math.nan,
            "ellipticity_missing": ellipticity_missing,
            "distance_modulus": mu,
            "distance_modulus_em": float(r["distance_modulus_em"]) if pd.notna(r["distance_modulus_em"]) else math.nan,
            "distance_modulus_ep": float(r["distance_modulus_ep"]) if pd.notna(r["distance_modulus_ep"]) else math.nan,
            "distance_kpc": d_kpc,
            "rhalf_major_pc": rhalf_major_pc,
            "r_half_2d_pc": r_half_2d_pc,
            "plummer_radius_pc": r_p_pc,
            "spatial_model": profile,
            "vlos_systemic_kms": float(r["vlos_systemic"]),
            "vlos_systemic_em": float(r["vlos_systemic_em"]) if pd.notna(r["vlos_systemic_em"]) else math.nan,
            "vlos_systemic_ep": float(r["vlos_systemic_ep"]) if pd.notna(r["vlos_systemic_ep"]) else math.nan,
            "vlos_sigma_kms": float(r["vlos_sigma"]) if pd.notna(r["vlos_sigma"]) else math.nan,
            "vlos_sigma_unresolved": vlos_sigma_missing,
            "M_V": float(r["M_V"]) if pd.notna(r["M_V"]) else math.nan,
            "ref_vlos": str(r["ref_vlos"]) if pd.notna(r["ref_vlos"]) else "",
            "ref_structure": str(r["ref_structure"]) if pd.notna(r["ref_structure"]) else "",
        })
        log_lines.append(
            f"{key}: host_chain={'->'.join(host_chains[key])}, profile={profile}, "
            f"d={d_kpc:.2f} kpc, r_1/2={r_half_2d_pc:.2f} pc, "
            f"ellipticity_missing={ellipticity_missing}, vlos_sigma_unresolved={vlos_sigma_missing}"
        )

    tab = Table(rows=rows)
    units = {
        "ra_deg": u.deg, "dec_deg": u.deg,
        "rhalf_arcmin": u.arcmin, "rhalf_arcmin_em": u.arcmin, "rhalf_arcmin_ep": u.arcmin,
        "distance_modulus": u.mag, "distance_modulus_em": u.mag, "distance_modulus_ep": u.mag,
        "distance_kpc": u.kpc,
        "rhalf_major_pc": u.pc, "r_half_2d_pc": u.pc, "plummer_radius_pc": u.pc,
        "vlos_systemic_kms": u.km / u.s, "vlos_systemic_em": u.km / u.s, "vlos_systemic_ep": u.km / u.s,
        "vlos_sigma_kms": u.km / u.s,
        "M_V": u.mag,
    }
    for col, unit in units.items():
        tab[col].unit = unit

    tab.meta["lvdb_version"] = "v1.0.5"
    tab.meta["lvdb_zenodo_doi"] = "10.5281/zenodo.15476348"
    tab.meta["spatial_model_default"] = spatial_default
    tab.meta["spatial_model_overrides"] = json.dumps(spatial_overrides)
    tab.meta["git_commit"] = _git_commit()
    tab.meta["build_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    out = REGISTRY_DIR / "galaxies.ecsv"
    tab.write(out, format="ascii.ecsv", overwrite=True)
    (REGISTRY_DIR / "build_log.txt").write_text("\n".join(log_lines) + "\n")

    return tab


def main() -> int:
    tab = build_registry()
    print(f"Wrote {REGISTRY_DIR / 'galaxies.ecsv'} with {len(tab)} rows.")
    print("Path tally:", dict(pd.Series(list(tab["path"])).value_counts()))
    print("Spatial model tally:", dict(pd.Series(list(tab["spatial_model"])).value_counts()))
    n_eps_miss = int(np.sum(tab["ellipticity_missing"]))
    n_sigma_unres = int(np.sum(tab["vlos_sigma_unresolved"]))
    print(f"Ellipticity missing/UL: {n_eps_miss} galaxies")
    print(f"vlos_sigma unresolved:  {n_sigma_unres} galaxies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
