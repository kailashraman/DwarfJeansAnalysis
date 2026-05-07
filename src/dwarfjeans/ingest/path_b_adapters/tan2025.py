"""Tan et al. 2025 (ApJ 979, 176) — adapter for Leo VI.

Source: ASCII Table 2, downloaded manually from iopscience (no VizieR
mirror). 13 rows: 9 confirmed members (ε_vhel < 10 km/s) followed by 4
candidate members (ε_vhel > 10 km/s). The paper's own footnote defines
the split; the file has no section break.

Adapter encodes confirmed -> p=1, candidate -> p=0, and stores
`member_flag` ∈ {M, C} for downstream re-inclusion.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ASCII_FILE = "apjad9b0ct2_ascii.txt"
EPS_CONFIRMED_KMS = 10.0  # paper's own threshold

COLUMN_MAPPING = {
    "Star Name": "Name_source_id",
    "R.A.": "RA_star",
    "Decl.": "Dec_star",
    "v_hel": "V (+ sigma_eps from `+or-` split)",
    "(eps_vhel < 10 km/s)": "p (confirmed -> 1, candidate -> 0)",
}


def _parse_velocity_cell(cell: str) -> tuple[float, float]:
    s = cell.strip()
    if s in ("...", "cdots", ""):
        return (float("nan"), float("nan"))
    m = re.match(r"^([-+]?\d+\.?\d*)\s*\+or-\s*(\d+\.?\d*)$", s)
    if not m:
        raise ValueError(f"unparseable v_hel cell: {cell!r}")
    return (float(m.group(1)), float(m.group(2)))


def _data_rows(text: str) -> list[list[str]]:
    """Skip the title + column-header lines (lines 1-6 of the file) and the
    trailing 'Note.' block. Data rows are tab-separated and start with a
    non-empty `Star Name` cell."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("Note") or line.startswith("Table"):
            continue
        cells = line.split("\t")
        if len(cells) < 10:
            continue
        first = cells[0].strip()
        # column-header / unit / title rows
        if not first:
            continue  # unit row begins with an empty leading cell
        if first == "Star Name":
            continue
        if first.startswith("Properties"):
            continue
        rows.append(cells)
    return rows


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "leo_6":
        raise ValueError(
            f"tan2025 adapter only serves leo_6; got {registry_row['lvdb_key']!r}"
        )
    text = (staged_dir / ASCII_FILE).read_text()
    rows = _data_rows(text)
    n = len(rows)
    if n != 13:
        raise RuntimeError(f"expected 13 rows in Tan+2025 Table 2; got {n}")

    names = np.array([r[0].strip() for r in rows], dtype=str)
    ra = np.array([float(r[1]) for r in rows], dtype=float)
    dec = np.array([float(r[2]) for r in rows], dtype=float)

    V = np.empty(n, dtype=float)
    sigma_eps = np.empty(n, dtype=float)
    for i, r in enumerate(rows):
        V[i], sigma_eps[i] = _parse_velocity_cell(r[6])

    is_confirmed = sigma_eps < EPS_CONFIRMED_KMS
    p = is_confirmed.astype(float)
    flag = np.where(is_confirmed, "M", "C")

    arrays = {
        "V": V,
        "sigma_eps": sigma_eps,
        "p": p,
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": ra,
        "Dec_star": dec,
        "Name_source_id": names,
        "member_flag": flag,
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": ASCII_FILE,
        "source_table_label": "Tan+2025 Table 2 (Leo VI confirmed + candidate members)",
        "membership_rule": (
            f"paper's split: ε_vhel < {EPS_CONFIRMED_KMS} km/s -> "
            "confirmed (p=1); else candidate (p=0)"
        ),
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "Star Name",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "13 rows: 9 confirmed + 4 candidates per the paper's own ε_vhel "
            "threshold (Note in Table 2). Candidates carry p=0 and "
            "member_flag='C' so downstream sample selection can opt back in."
        ),
    }
    return arrays, meta_extra
