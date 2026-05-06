"""Chiti et al. 2023 (AJ 165, 55) — adapter for Tucana II.

Source: VizieR `J/AJ/165/55/table6`, 60 rows / 19 unique Tuc II member
stars. Per-epoch kinematic compilation across multiple instruments
(M2FS, IMACS, MIKE, MagE) with provenance flags (`Ref ∈ {1..5}`).

Membership: the paper publishes a *member list* (no probability column);
per the data_sources.md missing-probability default, we assign p_i = 1
to every row. The `f_RVel == 'b'` binary-candidate flag is carried as
auxiliary so downstream sample-selection can demote suspected binaries.

Velocity frame: heliocentric (column documented).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

COLUMN_MAPPING = {
    "RVel": "V",
    "e_RVel": "sigma_eps",
    "(member-list -> p=1)": "p",
    "RAJ2000": "RA_star",
    "DEJ2000": "Dec_star",
    "Name": "star_id_source",
    "MJD": "epoch",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "tucana_2":
        raise ValueError(
            f"chiti2023 adapter only serves tucana_2; got {registry_row['lvdb_key']!r}"
        )
    t = Table.read(staged_dir / "table6.csv", format="ascii.ecsv")
    n = len(t)

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    arrays = {
        "V": _col("RVel"),
        "sigma_eps": _col("e_RVel"),
        "p": np.ones(n, dtype=float),  # member list -> all p=1
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": _col("RAJ2000"),
        "Dec_star": _col("DEJ2000"),
        "Name_source_id": np.asarray(t["Name"], dtype=str),
        "MJD": _col("MJD"),
        "Inst": np.asarray(t["Inst"], dtype=str),
        "f_RVel": np.asarray(t["f_RVel"], dtype=str),
        "Ref": _col("Ref"),
    }
    meta_extra = {
        "vizier_catalog": "J/AJ/165/55",
        "vizier_table": "table6",
        "membership_rule": "missing_default (member list, p=1 verbatim)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_epoch",
        "star_id_source_column": "Name",
        "epoch_column": "MJD",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "60 epochs from 19 unique Tuc II member stars across "
            "M2FS/IMACS/MIKE/MagE (Refs 1-5). Member list, no published "
            "probability column -> p=1 for every row per the missing-prob "
            "default. f_RVel='b' is a binary-candidate flag, carried as "
            "auxiliary for downstream binary-aware sample selection."
        ),
    }
    return arrays, meta_extra
