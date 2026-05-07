"""Simon et al. 2020 (ApJ 892, 137) — adapter for Tucana IV.

Source: VizieR `J/ApJ/892/137/table3`, per-epoch catalog covering Grus II,
Tucana IV, and Tucana V. We filter to `Gal == 'TucIV'` (223 rows / 132
unique DES stars) — the other two galaxies have different `ref_vlos`
papers in our registry.

Membership: binary `Mm` ∈ {0, 1}.
Velocity frame: heliocentric (`HRV`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from dwarfjeans.ingest.staging import per_star_indices

_LVDB_KEY_TO_GAL = {"tucana_4": "TucIV"}

COLUMN_MAPPING = {
    "HRV": "V",
    "e_HRV": "sigma_eps",
    "Mm": "p",
    "RAJ2000": "RA_star",
    "DEJ2000": "Dec_star",
    "DES": "star_id_source",
    "MJD": "epoch",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    lvdb_key = registry_row["lvdb_key"]
    if lvdb_key not in _LVDB_KEY_TO_GAL:
        raise ValueError(f"simon2020 adapter does not serve {lvdb_key!r}")
    gal = _LVDB_KEY_TO_GAL[lvdb_key]
    t = Table.read(staged_dir / "table3.csv", format="ascii.ecsv")
    sub = t[np.asarray(t["Gal"], dtype=str) == gal]
    if len(sub) == 0:
        raise RuntimeError(f"simon2020: 0 rows for Gal={gal!r}")

    def _col(name: str) -> np.ndarray:
        c = sub[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    n = len(sub)
    arrays = {
        "V": _col("HRV"),
        "sigma_eps": _col("e_HRV"),
        "p": _col("Mm"),  # 0/1 verbatim
        "star_id": per_star_indices(np.asarray(sub["DES"], dtype=str)),
        "RA_star": _col("RAJ2000"),
        "Dec_star": _col("DEJ2000"),
        "DES_source_id": np.asarray(sub["DES"], dtype=str),
        "MJD": _col("MJD"),
        "Nrv": _col("Nrv"),
        "FeH": _col("[Fe/H]"),
        "FeH_err": _col("e_[Fe/H]"),
        "EW": _col("EW"),
        "EW_err": _col("e_EW"),
        "gmag": _col("gmag"),
        "rmag": _col("rmag"),
        "SN": _col("S/N"),
    }
    meta_extra = {
        "vizier_catalog": "J/ApJ/892/137",
        "vizier_table": "table3",
        "system_gal_value": gal,
        "membership_rule": "binary_flag",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_epoch",
        "star_id_source_column": "DES",
        "epoch_column": "MJD",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Per-epoch table covering 3 galaxies; we filter to Gal='TucIV'. "
            "Mm binary flag encoded verbatim as p∈{0,1}."
        ),
    }
    return arrays, meta_extra
