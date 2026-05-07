"""Stage 0b Path B — driver for the per-paper adapters.

Implements docs/plan/data_sources.md §"Path B — LVDB `ref_vlos` paper ingest".

Run:
    # All Path B galaxies for which an adapter exists
    python -m dwarfjeans.ingest.stage0b_pathb

    # Just one galaxy (useful when iterating one paper at a time)
    python -m dwarfjeans.ingest.stage0b_pathb --lvdb-key carina_1

The driver looks up the adapter for each Path B registry row by deriving the
`<bibkey>` (`<lastname><year>` lowercased) from the `ref_vlos` column. A
missing adapter is logged and that galaxy skipped — the iteration loop adds
adapters one-by-one per data/path_b_status.md, and an unfinished sweep is
expected during the rollout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from dwarfjeans.ingest.staging import projected_radius_kpc, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_ECSV = REPO_ROOT / "data" / "registry" / "galaxies.ecsv"
STAR_CATALOG_DIR = REPO_ROOT / "data" / "star_catalogs"

REQUIRED_KEYS = ("V", "sigma_eps", "p", "star_id", "RA_star", "Dec_star")
BIBCODE_LEN = 19


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def derive_bibkey(ref_vlos: str) -> str:
    """`<LastName><19-char-bibcode>` -> `<lastname><year>` lowercased."""
    if len(ref_vlos) < BIBCODE_LEN:
        raise ValueError(f"ref_vlos too short: {ref_vlos!r}")
    bibcode = ref_vlos[-BIBCODE_LEN:]
    last_name = ref_vlos[:-BIBCODE_LEN]
    year = re.match(r"^(\d{4})", bibcode).group(1)
    return (last_name + year).lower().replace(" ", "").replace(".", "")


def ingest_one(registry_row, *, build_utc: str, git_commit: str) -> Path:
    lvdb_key = registry_row["lvdb_key"]
    ref_vlos = str(registry_row["ref_vlos"])
    bibkey = derive_bibkey(ref_vlos)
    staged_dir = REPO_ROOT / "data" / bibkey

    verify_checksums(staged_dir)

    try:
        adapter = importlib.import_module(f"dwarfjeans.ingest.path_b_adapters.{bibkey}")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"{lvdb_key}: no adapter at dwarfjeans/ingest/path_b_adapters/{bibkey}.py "
            f"({e}). Add the adapter and retry."
        )

    arrays, meta_extra = adapter.load(staged_dir, registry_row)
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise RuntimeError(f"{lvdb_key}: adapter missing required keys {missing}")
    n = arrays["V"].shape[0]
    if n == 0:
        raise RuntimeError(f"{lvdb_key}: adapter returned 0 rows")
    for k, v in arrays.items():
        if v.shape[0] != n:
            raise RuntimeError(f"{lvdb_key}: column {k!r} has length {v.shape[0]} != {n}")

    # star_id uniqueness (per-star granularity) — papers with per-epoch tables
    # would relax this; ours is per-star here.
    granularity = meta_extra.get("catalog_granularity", "per_star")
    if granularity == "per_star" and len(np.unique(arrays["star_id"])) != n:
        raise RuntimeError(
            f"{lvdb_key}: per-star granularity check failed (n_unique star_id != n_rows)"
        )

    # Projected radius from LVDB-tabulated center.
    arrays["R"] = projected_radius_kpc(
        arrays["RA_star"], arrays["Dec_star"],
        float(registry_row["ra_deg"]), float(registry_row["dec_deg"]),
        float(registry_row["distance_kpc"]),
    )

    # Missing-probability default per data_sources.md "Settled conventions".
    n_p_missing = int(np.sum(np.isnan(arrays["p"])))
    if n_p_missing:
        arrays["p"] = np.where(np.isnan(arrays["p"]), 1.0, arrays["p"])

    meta = {
        "lvdb_key": lvdb_key,
        "study_name": str(registry_row["study_name"]),
        "source_path": "Path B: LVDB ref_vlos",
        "lvdb_ref_vlos": ref_vlos,
        "source_paper_bibcode": ref_vlos[-BIBCODE_LEN:],
        "bibkey": bibkey,
        "staged_dir": str(staged_dir.relative_to(REPO_ROOT)),
        "n_rows": int(n),
        "R_unit": "kpc",
        "rhalf_unit_in_registry": "rhalf_major_pc (semi-major axis, pc)",
        "lvdb_center_ra_deg": float(registry_row["ra_deg"]),
        "lvdb_center_dec_deg": float(registry_row["dec_deg"]),
        "lvdb_distance_kpc": float(registry_row["distance_kpc"]),
        "missing_probability_default_applied": n_p_missing,
        "lvdb_version": "v1.0.5",
        "git_commit": git_commit,
        "build_utc": build_utc,
    }
    meta.update(meta_extra)

    STAR_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    out = STAR_CATALOG_DIR / f"{lvdb_key}.npz"
    np.savez(
        out,
        _meta=np.array(json.dumps(meta), dtype=object),
        **arrays,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lvdb-key", default=None,
                    help="If set, ingest only this one galaxy.")
    args = ap.parse_args()

    registry = Table.read(REGISTRY_ECSV, format="ascii.ecsv")
    path_b = registry[registry["path"] == "B"]
    if args.lvdb_key:
        path_b = path_b[path_b["lvdb_key"] == args.lvdb_key]
        if len(path_b) == 0:
            raise SystemExit(f"lvdb_key={args.lvdb_key!r} not found among Path B galaxies")

    build_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git_commit = _git_commit()

    n_done, n_skip = 0, 0
    for row in path_b:
        try:
            out = ingest_one(row, build_utc=build_utc, git_commit=git_commit)
            n = np.load(out)["V"].shape[0]
            print(f"  {row['lvdb_key']:24s} -> {out.relative_to(REPO_ROOT)}   ({n} rows)")
            n_done += 1
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  {row['lvdb_key']:24s} SKIP: {e}")
            n_skip += 1

    print(f"Path B summary: {n_done} ingested, {n_skip} skipped (missing adapter or stage).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
