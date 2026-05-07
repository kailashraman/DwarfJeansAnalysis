"""Li et al. 2018 (ApJ 857, 145) — adapter for Carina II + Carina III.

Source: VizieR `J/ApJ/857/145/table4`, 407 rows / 283 unique stars.
Per-epoch granularity (`MJD` column shows multi-epoch observations of
the same `MagLiteS` star). Membership is a tri-state `Mm`:
    0 = non-member of both galaxies
    2 = Carina II member
    3 = Carina III member

Per data_sources.md "Multi-measurement handling" / "raw-data-only", we
ingest all 407 rows into each per-galaxy npz with `p` encoded according
to the target galaxy:
    - carina_2.npz: p = 1 iff Mm == 2; else 0
    - carina_3.npz: p = 1 iff Mm == 3; else 0

R_i in each file is computed (by the driver) from the target galaxy's
LVDB center. Stars that are members of the *other* galaxy or
non-members of both end up at p = 0 and contribute to the background
component of the Stage 1 likelihood.

Velocity frame: heliocentric (column `HRV`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from data_ingest.staging import per_star_indices

_LVDB_KEY_TO_MEMBER_CODE = {
    "carina_2": 2,
    "carina_3": 3,
}

COLUMN_MAPPING = {
    "HRV": "V",
    "e_HRV": "sigma_eps",
    "Mm == <code>": "p (binary, encoded per target galaxy)",
    "RAJ2000": "RA_star",
    "DEJ2000": "Dec_star",
    "MagLiteS": "star_id_source",
    "MJD": "epoch",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    lvdb_key = registry_row["lvdb_key"]
    if lvdb_key not in _LVDB_KEY_TO_MEMBER_CODE:
        raise ValueError(
            f"li2018 adapter does not serve lvdb_key={lvdb_key!r} (only carina_2, carina_3)"
        )
    target_code = _LVDB_KEY_TO_MEMBER_CODE[lvdb_key]
    t = Table.read(staged_dir / "table4.csv", format="ascii.ecsv")
    n = len(t)
    if n == 0:
        raise RuntimeError("li2018: 0 rows in table4")

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    mm = np.asarray(t["Mm"], dtype=int)
    p = (mm == target_code).astype(float)

    arrays = {
        "V": _col("HRV"),
        "sigma_eps": _col("e_HRV"),
        "p": p,
        "star_id": per_star_indices(np.asarray(t["MagLiteS"], dtype=str)),
        "RA_star": _col("RAJ2000"),
        "Dec_star": _col("DEJ2000"),
        # Auxiliary
        "MagLiteS_source_id": np.asarray(t["MagLiteS"], dtype=str),
        "MJD": _col("MJD"),
        "Inst": np.asarray(t["Inst"], dtype=str),
        "Mm": mm.astype(float),  # carry full tri-state for traceability
        "FeH": _col("[Fe/H]"),
        "FeH_err": _col("e_[Fe/H]"),
        "EW": _col("EW"),
        "EW_err": _col("e_EW"),
        "g0mag": _col("g0mag"),
        "r0mag": _col("r0mag"),
        "SN": _col("S/N"),
    }

    meta_extra = {
        "vizier_catalog": "J/ApJ/857/145",
        "vizier_table": "table4",
        "membership_rule": (
            "tri_state_Mm: 0=non-member, 2=Carina II, 3=Carina III. "
            f"For target {lvdb_key}, p=1 iff Mm=={target_code}, else 0."
        ),
        "membership_target_code": int(target_code),
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_epoch",
        "star_id_source_column": "MagLiteS",
        "epoch_column": "MJD",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Per-epoch table: 407 rows from 283 unique MagLiteS stars. "
            "All rows are written into each per-galaxy npz (carina_2 and "
            "carina_3 both contain all 407); only the `p` encoding differs."
        ),
    }
    return arrays, meta_extra
