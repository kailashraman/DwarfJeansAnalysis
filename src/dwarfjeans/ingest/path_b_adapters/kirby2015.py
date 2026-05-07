"""Kirby, Simon & Cohen 2015 (ApJ 810, 56) — adapter for Pisces II.

Source: ASCII Table 2 ("Target List"), downloaded manually from iopscience
(no VizieR mirror). The file stacks three systems (Hydra II / Pisces II /
Laevens 1) in one table separated by section header rows; this adapter
slices the Pisces II block and ingests it.

Format quirks handled here:
- Sexagesimal RA (`hh mm ss.ss`) and Dec (`±dd mm ss.s`) -> decimal degrees.
- `v_helio` is one tab-cell formatted `<v> +or- <err>` -> split into V, sigma_eps.
- Missing values written as ` ... `.
- `Member?` is `Y`/`N` -> p = 1/0.

Velocity frame: heliocentric.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

ASCII_FILE = "apj518514t2_ascii.txt"
SECTION_HEADER = "Pisces II"
NEXT_HEADER = "Laevens 1"

COLUMN_MAPPING = {
    "ID": "star_id_source",
    "R.A. (J2000)": "RA_star",
    "decl. (J2000)": "Dec_star",
    "v_helio": "V (+ sigma_eps from `+or-` split)",
    "Member?": "p (Y->1, N->0)",
}


def _parse_velocity_cell(cell: str) -> tuple[float, float]:
    """`'-224.9 +or- 1.6'` -> (-224.9, 1.6); `' ... '` -> (nan, nan)."""
    s = cell.strip()
    if s == "..." or s == "":
        return (float("nan"), float("nan"))
    m = re.match(r"^([-+]?\d+\.?\d*)\s*\+or-\s*(\d+\.?\d*)$", s)
    if not m:
        raise ValueError(f"unparseable v_helio cell: {cell!r}")
    return (float(m.group(1)), float(m.group(2)))


def _slice_pisces_ii(text: str) -> list[list[str]]:
    """Return Pisces II rows as list-of-cells (tab-split). Skips blanks."""
    in_block = False
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == SECTION_HEADER:
            in_block = True
            continue
        if in_block and stripped == NEXT_HEADER:
            break
        if not in_block:
            continue
        if not stripped:
            continue
        rows.append(line.split("\t"))
    if not rows:
        raise RuntimeError("Pisces II section not found in Kirby 2015 Table 2.")
    return rows


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "pisces_2":
        raise ValueError(
            f"kirby2015 adapter only serves pisces_2; got {registry_row['lvdb_key']!r}"
        )
    text = (staged_dir / ASCII_FILE).read_text()
    rows = _slice_pisces_ii(text)
    n = len(rows)

    ids = np.array([r[0].strip().rstrip("^b") for r in rows], dtype=str)
    ra_sex = [r[1].strip() for r in rows]
    dec_sex = [r[2].strip() for r in rows]
    v_cells = [r[7] for r in rows]
    member_flag = np.array([r[8].strip() for r in rows], dtype=str)

    coords = SkyCoord(
        ra=ra_sex, dec=dec_sex, unit=(u.hourangle, u.deg), frame="icrs"
    )
    ra_deg = np.asarray(coords.ra.deg, dtype=float)
    dec_deg = np.asarray(coords.dec.deg, dtype=float)

    V = np.empty(n, dtype=float)
    sigma_eps = np.empty(n, dtype=float)
    for i, cell in enumerate(v_cells):
        V[i], sigma_eps[i] = _parse_velocity_cell(cell)

    p = np.where(member_flag == "Y", 1.0, 0.0)

    arrays = {
        "V": V,
        "sigma_eps": sigma_eps,
        "p": p,
        "star_id": np.array([int(s) for s in ids], dtype=np.int64),
        "RA_star": ra_deg,
        "Dec_star": dec_deg,
        "Name_source_id": ids,
        "member_flag": member_flag,
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": ASCII_FILE,
        "source_table_label": "Kirby+2015 Table 2 (Pisces II block)",
        "membership_rule": "paper Y/N flag (Y->p=1, N->p=0)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "ID",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Table 2 stacks Hydra II / Pisces II / Laevens 1; adapter slices "
            "the Pisces II block (13 stars, 7 Y-flagged members). Sexagesimal "
            "coords converted to decimal degrees via astropy.SkyCoord. "
            "`v_helio` cell parsed by regex on `<v> +or- <err>`. ID 191385 "
            "carries a `^b` footnote in the paper (CMD non-member); the "
            "marker is stripped, the Y/N flag (N) is preserved verbatim."
        ),
    }
    return arrays, meta_extra
