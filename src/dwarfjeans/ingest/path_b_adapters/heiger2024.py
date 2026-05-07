"""Heiger et al. 2024 (ApJ 961, 234) — adapter for Centaurus I.

Source: CDS MRT Table 4. The paper covers Eri IV + Cen I; Table 4 is the
Cen I spectroscopic sample (62 rows). Combined-fit `vhel`/`e_vhel`,
heliocentric, with a `Member ∈ {0,1}` flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import ascii

MRT_FILE = "apjad0cf7t4_mrt.txt"

COLUMN_MAPPING = {
    "ID": "star_id_source (DELVE quick_object_id or Gaia source_id)",
    "RAdeg": "RA_star",
    "DEdeg": "Dec_star",
    "vhel": "V (combined-fit heliocentric)",
    "e_vhel": "sigma_eps",
    "Member": "p (1->1, 0->0)",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "centaurus_1":
        raise ValueError(
            f"heiger2024 adapter only serves centaurus_1; got {registry_row['lvdb_key']!r}"
        )
    t = ascii.read(staged_dir / MRT_FILE, format="cds")
    n = len(t)

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    member = np.asarray(t["Member"], dtype=int)
    p = member.astype(float)

    arrays = {
        "V": _col("vhel"),
        "sigma_eps": _col("e_vhel"),
        "p": p,
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": _col("RAdeg"),
        "Dec_star": _col("DEdeg"),
        "Name_source_id": np.asarray(t["ID"], dtype=str),
        "member_flag": member,
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": MRT_FILE,
        "source_table_label": "Heiger+2024 Table 4 (Cen I spectroscopic sample)",
        "membership_rule": "paper Member flag (1->p=1, 0->p=0)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "ID",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "62 rows, 34 confirmed members. Combined-fit `vhel` is the "
            "canonical column; per-mask velocities (`vhel-N-MJD`) live in "
            "the source table but are not propagated to the npz."
        ),
    }
    return arrays, meta_extra
