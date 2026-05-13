"""Stage 0b Path A — ingest Geha 2026 Paper I Table 5A into per-galaxy npz files.

Source: ``data/geha2026/table5A_20260110.csv`` (Geha+2026 Paper I per-star
catalog, full precision; arXiv:2602.10200). Carries the binary ``Pmem_novar``
column that the Paper II §3.1 analysis uses to compute Table A1 N* counts.

The earlier ``table3A_20260110.csv`` release only carries a graded ``Pmem``
and a separate ``Var`` flag; combining them as ``Pmem > 0.5 & Var != 1`` does
not exactly reproduce Paper II's N*, so we use Table 5A instead.

Run:
    python -m dwarfjeans.ingest.stage0b_geha

Outputs:
    data/star_catalogs/<lvdb_key>.npz   (one per Path A galaxy in study_sample.yaml)

Each archive carries the canonical columns (R, V, sigma_eps, p, star_id,
RA_star, Dec_star) plus auxiliary Geha-specific columns and a JSON-serialized
``_meta`` dict with provenance. ``p`` is set from ``Pmem_novar`` (0/1 binary).

Stage 0b is raw-data-only: no thresholds, no membership cuts. Sample selection
lives in ``dwarfjeans.jeans.selection``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from dwarfjeans.ingest.staging import (
    pm_meta_from_registry_row,
    projected_radius_kpc,
    verify_checksums,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GEHA_DIR = REPO_ROOT / "data" / "geha2026"
GEHA_CSV = GEHA_DIR / "table5A_20260110.csv"
REGISTRY_ECSV = REPO_ROOT / "data" / "registry" / "galaxies.ecsv"
STAR_CATALOG_DIR = REPO_ROOT / "data" / "star_catalogs"

GEHA_BIBCODE_PAPER1 = "2026arXiv260210200G"  # placeholder until ADS resolves arXiv:2602.10200
GEHA_BIBCODE_PAPER2 = "2026arXiv260210202G"  # placeholder until ADS resolves arXiv:2602.10202

# Table 5A column → canonical npz column. "Pmem_novar" carries the binary
# membership-and-no-velocity-variability flag; map directly to ``p``.
COLUMN_MAPPING = {
    "v":           "V",
    "v_err":       "sigma_eps",
    "Pmem_novar":  "p",
    "RA":          "RA_star",
    "DEC":         "Dec_star",
    "ew_feh":      "FeH",
    "ew_feh_err":  "FeH_err",
    "flag_var":    "Var",
    "Pmem":        "Pmem",  # graded probability, kept as auxiliary
    "gmag_o":      "g_mag",
    "rmag_o":      "r_mag",
    "MV_o":        "MV",
    "nmask":       "nmask",
    "t_exp":       "t_exp",
    "SN":          "SN",
    "ew_cat":      "CaT",
    "ew_cat_err":  "CaTerr",
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
    sub = geha_df[geha_df["system_name"] == geha_galaxy].reset_index(drop=True)
    if len(sub) == 0:
        raise RuntimeError(f"{key}: 0 rows for system_name={geha_galaxy!r} in Table 5A")

    star_id = np.arange(len(sub), dtype=np.int64)

    ra_star = sub["RA"].to_numpy(dtype=float)
    dec_star = sub["DEC"].to_numpy(dtype=float)
    R = projected_radius_kpc(
        ra_star, dec_star,
        float(registry_row["ra_deg"]), float(registry_row["dec_deg"]),
        float(registry_row["distance_kpc"]),
    )

    arrays = {
        "R": R,
        "V": sub["v"].to_numpy(dtype=float),
        "sigma_eps": sub["v_err"].to_numpy(dtype=float),
        "p": sub["Pmem_novar"].to_numpy(dtype=float),
        "star_id": star_id,
        "RA_star": ra_star,
        "Dec_star": dec_star,
        "FeH": sub["ew_feh"].to_numpy(dtype=float),
        "FeH_err": sub["ew_feh_err"].to_numpy(dtype=float),
        "Var": sub["flag_var"].to_numpy(dtype=float),
        "Pmem": sub["Pmem"].to_numpy(dtype=float),
        "g_mag": sub["gmag_o"].to_numpy(dtype=float),
        "r_mag": sub["rmag_o"].to_numpy(dtype=float),
        "MV": sub["MV_o"].to_numpy(dtype=float),
        "nmask": sub["nmask"].to_numpy(dtype=float),
        "t_exp": sub["t_exp"].to_numpy(dtype=float),
        "SN": sub["SN"].to_numpy(dtype=float),
        "CaT": sub["ew_cat"].to_numpy(dtype=float),
        "CaTerr": sub["ew_cat_err"].to_numpy(dtype=float),
    }

    # Pmem_novar is binary 0/1, no NaNs expected. Defend the invariant
    # explicitly so a future release that drifts from this convention
    # surfaces immediately rather than silently propagating.
    p = arrays["p"]
    n_p_missing = int(np.sum(np.isnan(p)))
    if n_p_missing:
        raise RuntimeError(f"{key}: {n_p_missing} stars have NaN Pmem_novar")
    bad = np.setdiff1d(np.unique(p), [0.0, 1.0])
    if bad.size:
        raise RuntimeError(f"{key}: Pmem_novar carries non-binary values {bad}")

    meta = {
        "lvdb_key": key,
        "study_name": str(registry_row["study_name"]),
        "source_path": "Path A: Geha 2026",
        "source_paper_bibcode_paper1": GEHA_BIBCODE_PAPER1,
        "source_paper_bibcode_paper2": GEHA_BIBCODE_PAPER2,
        "source_table": "table5A_20260110",
        "source_file": "data/geha2026/table5A_20260110.csv",
        "system_name_in_table5A": geha_galaxy,
        "n_rows": int(len(sub)),
        "catalog_granularity": "per_star",
        "R_unit": "kpc",
        "rhalf_unit_in_registry": "rhalf_major_pc (semi-major axis, pc)",
        "star_id_source_column": "row_index",
        "column_mapping": COLUMN_MAPPING,
        "lvdb_center_ra_deg": float(registry_row["ra_deg"]),
        "lvdb_center_dec_deg": float(registry_row["dec_deg"]),
        "lvdb_distance_kpc": float(registry_row["distance_kpc"]),
        "p_source_column": "Pmem_novar (binary; velocity variables already removed)",
        "lvdb_version": "v1.0.5",
        "git_commit": git_commit,
        "build_utc": build_utc,
        "notes": "p_i ← Pmem_novar from Paper I Table 5A. Reproduces Paper II "
                 "Table A1 N* counts. Selection downstream applies the §3.1 "
                 "R<2*r_½ cut using the sphericalized 3D Plummer half-mass "
                 "radius.",
        **pm_meta_from_registry_row(registry_row),
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
