"""Bruce et al. 2023 (ApJ 950, 167) — adapter for Aquarius II.

Source: CDS MRT Table 3 ("Aquarius II Observed Spectra"), downloaded
manually from iopscience (no VizieR mirror). 12 stars, combined-epoch
heliocentric velocities with `Mem ∈ {M, NM}`.

The astropy CDS reader handles the fixed-width parsing; the adapter just
maps Mem -> p and copies columns through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import ascii

MRT_FILE = "apjacc943t3_mrt.txt"

COLUMN_MAPPING = {
    "Gaia": "star_id_source (Gaia DR3 source_id)",
    "RAdeg": "RA_star",
    "DEdeg": "Dec_star",
    "Vhelio": "V",
    "e_Vhelio": "sigma_eps",
    "Mem": "p (M->1, NM->0)",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "aquarius_2":
        raise ValueError(
            f"bruce2023 adapter only serves aquarius_2; got {registry_row['lvdb_key']!r}"
        )
    t = ascii.read(staged_dir / MRT_FILE, format="cds")
    n = len(t)

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    mem = np.asarray(t["Mem"], dtype=str)
    p = np.where(mem == "M", 1.0, 0.0)

    arrays = {
        "V": _col("Vhelio"),
        "sigma_eps": _col("e_Vhelio"),
        "p": p,
        "star_id": np.asarray(t["Gaia"], dtype=np.int64),
        "RA_star": _col("RAdeg"),
        "Dec_star": _col("DEdeg"),
        "Name_source_id": np.asarray(t["Gaia"], dtype=str),
        "member_flag": mem,
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": MRT_FILE,
        "source_table_label": "Bruce+2023 Table 3 (Aquarius II Observed Spectra)",
        "membership_rule": "paper M/NM flag (M->p=1, NM->p=0)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "Gaia",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "12 rows, 8 confirmed members and 4 non-members. Combined "
            "two-epoch heliocentric velocity (Vhelio) per star; per-epoch "
            "SNR/EW columns retained in the source table but not propagated "
            "to the npz."
        ),
    }
    return arrays, meta_extra
