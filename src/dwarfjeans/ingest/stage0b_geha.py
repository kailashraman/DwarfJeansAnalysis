"""Stage 0b Path A — ingest Geha 2026 Paper I Table 3A into per-galaxy npz files.

Implements docs/plan/data_sources.md §"Path A — Geha DEIMOS ingest".

Run:
    python -m dwarfjeans.ingest.stage0b_geha

Outputs:
    data/star_catalogs/<lvdb_key>.npz   (one per Path A galaxy in study_sample.yaml)

Each archive carries the canonical columns (R, V, sigma_eps, p, star_id, RA_star,
Dec_star) plus auxiliary Geha-specific columns ([Fe/H], [Fe/H]_err, MV, Var,
nmask, t_exp, SN, gr) and a JSON-serialized `_meta` dict with provenance.

Stage 0b is raw-data-only: no thresholds, no membership cuts. Sample selection
lives in a downstream stage (out of scope here).
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

from dwarfjeans.ingest.staging import projected_radius_kpc, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent / "config"
GEHA_DIR = REPO_ROOT / "data" / "geha2026"
GEHA_CSV = GEHA_DIR / "table3A_20260110.csv"
REGISTRY_ECSV = REPO_ROOT / "data" / "registry" / "galaxies.ecsv"
STAR_CATALOG_DIR = REPO_ROOT / "data" / "star_catalogs"
STUDY_SAMPLE_YAML = CONFIG_DIR / "study_sample.yaml"

GEHA_BIBCODE_PAPER1 = "2026arXiv260210200G"  # placeholder until ADS resolves arXiv:2602.10200
GEHA_BIBCODE_PAPER2 = "2026arXiv260210202G"  # placeholder until ADS resolves arXiv:2602.10202

# Geha Table 3A header → canonical name. Verified 2026-05-05.
COLUMN_MAPPING = {
    "v": "V",
    "verr": "sigma_eps",
    "Pmem": "p",
    "RA": "RA_star",
    "DEC": "Dec_star",
    "FeH": "FeH",
    "FeH_err": "FeH_err",
    "Var": "Var",
    "r": "r_mag",      # r-band apparent magnitude (Paper I §); NOT absolute V — see _meta
    "gr": "gr",
    "nmask": "nmask",
    "t_exp": "t_exp",
    "SN": "SN",
    "CaT": "CaT",
    "CaTerr": "CaTerr",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def ingest_one(geha_df: pd.DataFrame, registry_row, *, build_utc: str, git_commit: str) -> Path:
    key = registry_row["lvdb_key"]
    geha_galaxy = registry_row["geha_galaxy"]
    sub = geha_df[geha_df["Galaxy"] == geha_galaxy].copy()
    if len(sub) == 0:
        raise RuntimeError(f"{key}: 0 rows for Galaxy={geha_galaxy!r} in Table 3A")

    # star_id: per-row positional index within the per-galaxy slice. Required to
    # exist (per the plan) so downstream consumers can verify per-star granularity.
    sub = sub.reset_index(drop=True)
    star_id = np.arange(len(sub), dtype=np.int64)
    if len(np.unique(star_id)) != len(sub):
        raise RuntimeError(f"{key}: per-star granularity check failed (n_unique != n_rows)")

    ra_star = sub["RA"].to_numpy(dtype=float)
    dec_star = sub["DEC"].to_numpy(dtype=float)
    R = projected_radius_kpc(
        ra_star, dec_star,
        float(registry_row["ra_deg"]), float(registry_row["dec_deg"]),
        float(registry_row["distance_kpc"]),
    )

    arrays = {
        "R": R,                                          # kpc, projected radius from LVDB center
        "V": sub["v"].to_numpy(dtype=float),             # km/s, heliocentric
        "sigma_eps": sub["verr"].to_numpy(dtype=float),  # km/s
        "p": sub["Pmem"].to_numpy(dtype=float),          # membership probability (verbatim)
        "star_id": star_id,
        "RA_star": ra_star,
        "Dec_star": dec_star,
        # Auxiliary columns
        "FeH": sub["FeH"].to_numpy(dtype=float),
        "FeH_err": sub["FeH_err"].to_numpy(dtype=float),
        "Var": sub["Var"].to_numpy(dtype=float),
        "r_mag": sub["r"].to_numpy(dtype=float),  # r-band apparent magnitude
        "gr": sub["gr"].to_numpy(dtype=float),
        "nmask": sub["nmask"].to_numpy(dtype=float),
        "t_exp": sub["t_exp"].to_numpy(dtype=float),
        "SN": sub["SN"].to_numpy(dtype=float),
        "CaT": sub["CaT"].to_numpy(dtype=float),
        "CaTerr": sub["CaTerr"].to_numpy(dtype=float),
    }

    # Apply missing-probability default per data_sources.md "Settled conventions".
    n_p_missing = int(np.sum(np.isnan(arrays["p"])))
    if n_p_missing:
        arrays["p"] = np.where(np.isnan(arrays["p"]), 1.0, arrays["p"])

    meta = {
        "lvdb_key": key,
        "study_name": str(registry_row["study_name"]),
        "source_path": "Path A: Geha 2026",
        "source_paper_bibcode_paper1": GEHA_BIBCODE_PAPER1,
        "source_paper_bibcode_paper2": GEHA_BIBCODE_PAPER2,
        "source_table": "table3A_20260110",
        "system_name_in_table3A": geha_galaxy,
        "n_rows": int(len(sub)),
        "catalog_granularity": "per_star",
        "star_id_source_column": "row_index",
        "column_mapping": COLUMN_MAPPING,
        "lvdb_center_ra_deg": float(registry_row["ra_deg"]),
        "lvdb_center_dec_deg": float(registry_row["dec_deg"]),
        "lvdb_distance_kpc": float(registry_row["distance_kpc"]),
        "missing_probability_default_applied": n_p_missing,
        "geha_csv_sha_relpath": "data/geha2026/table3A_20260110.csv",
        "lvdb_version": "v1.0.5",
        "git_commit": git_commit,
        "build_utc": build_utc,
        "notes": "p_i ← Pmem (Var carried as auxiliary). Earlier plan referenced "
                 "Pmem_novar, which does not exist in table3A_20260110.csv; resolved "
                 "in data_sources.md changelog 2026-05-05.",
    }

    STAR_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    out = STAR_CATALOG_DIR / f"{key}.npz"
    np.savez(
        out,
        _meta=np.array(json.dumps(meta), dtype=object),
        **arrays,
    )
    return out


def main() -> int:
    verify_checksums(GEHA_DIR)

    registry = Table.read(REGISTRY_ECSV, format="ascii.ecsv")
    path_a = registry[registry["path"] == "A"]
    if len(path_a) != 22:
        raise RuntimeError(f"Expected 22 Path A galaxies, got {len(path_a)}")

    geha_df = pd.read_csv(GEHA_CSV)
    build_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git_commit = _git_commit()

    for row in path_a:
        out = ingest_one(geha_df, row, build_utc=build_utc, git_commit=git_commit)
        n = np.load(out)["V"].shape[0]
        print(f"  {row['lvdb_key']:24s} → {out.relative_to(REPO_ROOT)}   ({n} rows)")
    print(f"Wrote {len(path_a)} per-galaxy catalogs to {STAR_CATALOG_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
