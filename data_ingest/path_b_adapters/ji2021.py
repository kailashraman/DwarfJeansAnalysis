"""Ji et al. 2021 (ApJ 921, 32) — adapter for Antlia II + Crater II.

Source: VizieR `J/ApJ/921/32`. Two per-star tables, one per galaxy:
    table4: Antlia II  (508 rows, RA ~143°, Dec ~-37°)
    table5: Crater II  (207 rows, RA ~177°, Dec ~-18°)

Membership: continuous `Mm` ∈ [0, 1] from Ji+21's Bayesian mixture
analysis. Velocity frame: heliocentric (`HRV`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

_LVDB_KEY_TO_FILE = {
    "antlia_2": "table4.csv",
    "crater_2": "table5.csv",
}

COLUMN_MAPPING = {
    "HRV": "V",
    "e_HRV": "sigma_eps",
    "Mm": "p",
    "RA_ICRS": "RA_star",
    "DE_ICRS": "Dec_star",
    "Gaia": "star_id_source",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    lvdb_key = registry_row["lvdb_key"]
    if lvdb_key not in _LVDB_KEY_TO_FILE:
        raise ValueError(f"ji2021 adapter does not serve {lvdb_key!r}")
    fname = _LVDB_KEY_TO_FILE[lvdb_key]
    t = Table.read(staged_dir / fname, format="ascii.ecsv")
    n = len(t)
    if n == 0:
        raise RuntimeError(f"ji2021: 0 rows in {fname}")

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    arrays = {
        "V": _col("HRV"),
        "sigma_eps": _col("e_HRV"),
        "p": _col("Mm"),  # continuous probability verbatim
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": _col("RA_ICRS"),
        "Dec_star": _col("DE_ICRS"),
        "Gaia_source_id": np.asarray(t["Gaia"], dtype=str),
        "Gmag": _col("Gmag"),
        "pmRA": _col("pmRA"),
        "pmDE": _col("pmDE"),
        "FeH_method1": _col("[Fe/H]1"),
        "FeH_method1_err": _col("e_[Fe/H]1"),
        "FeH_method2": _col("[Fe/H]2"),
        "FeH_method2_err": _col("e_[Fe/H]2"),
        "Bin": np.asarray(t["Bin"], dtype=str),
        "SN": _col("S/N"),
    }
    meta_extra = {
        "vizier_catalog": "J/ApJ/921/32",
        "vizier_table": fname.replace(".csv", ""),
        "membership_rule": "continuous",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "row_index",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Two galaxies served via separate per-galaxy tables (table4 -> "
            "Antlia II, table5 -> Crater II). Mm is a continuous Bayesian "
            "mixture probability; stored verbatim. Bin column flags binary "
            "candidates and is carried as auxiliary."
        ),
    }
    return arrays, meta_extra
