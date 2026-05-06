"""Walker et al. 2015 (ApJ 808, 108) — adapter for Reticulum II.

Source: VizieR `J/ApJ/808/108/table1`, 38 rows, per-star M2FS spectroscopy.
Membership rule: binary `Mm?` Y/N flag from Walker+15's Bayesian mixture
analysis (encode Y -> p=1, N -> p=0).
Velocity frame: heliocentric ("solar rest frame", paper §2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

COLUMN_MAPPING = {
    "Vlos": "V",
    "e_Vlos": "sigma_eps",
    "Mm?": "p",
    "RAJ2000": "RA_star",
    "DEJ2000": "Dec_star",
    "Ret2": "star_id_source",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "reticulum_2":
        raise ValueError(
            f"walker2015 adapter only serves reticulum_2; got {registry_row['lvdb_key']!r}"
        )
    t = Table.read(staged_dir / "table1.csv", format="ascii.ecsv")
    n = len(t)
    if n == 0:
        raise RuntimeError("walker2015: 0 rows")

    sky = SkyCoord(
        ra=np.asarray(t["RAJ2000"], dtype=str),
        dec=np.asarray(t["DEJ2000"], dtype=str),
        unit=(u.hourangle, u.deg),
    )
    p = np.array([1.0 if str(m) == "Y" else 0.0 for m in t["Mm?"]], dtype=float)

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    arrays = {
        "V": _col("Vlos"),
        "sigma_eps": _col("e_Vlos"),
        "p": p,
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": sky.ra.deg,
        "Dec_star": sky.dec.deg,
        "Ret2_source_id": np.asarray(t["Ret2"], dtype=str),
        "FeH": _col("[Fe/H]"),
        "FeH_err": _col("e_[Fe/H]"),
        "Teff": _col("Teff"),
        "Teff_err": _col("e_Teff"),
        "logg": _col("logg"),
        "logg_err": _col("e_logg"),
        "gmag": _col("gmag"),
        "rmag": _col("rmag"),
        "SN": _col("S/N"),
    }

    meta_extra = {
        "vizier_catalog": "J/ApJ/808/108",
        "vizier_table": "table1",
        "membership_rule": "binary_flag",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "row_index",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Mm? (Y/N) encoded as p=1/0 verbatim. Walker+15 paper text reports "
            "17 confirmed members; the VizieR table flags 18 with Mm?=Y."
        ),
    }
    return arrays, meta_extra
