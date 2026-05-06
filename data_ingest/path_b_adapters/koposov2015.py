"""Koposov et al. 2015 (ApJ 811, 62) — adapter for Horologium I.

Source: ASCII Table 2 stacking Reticulum 2 + Horologium 1 sub-tables.
The adapter slices the Hor 1 block (between the `Horologium 1` header
and end-of-file).

Format quirks:
- `delta(J2000)` cells are wrapped in `$…$` LaTeX math mode.
- `V_hel` cells appear in two forms: ASCII `<v> +or- <err>` and LaTeX
  `$<v>\\pm <err>$` (used on a few RGB-MW rows in the Reti 2 block).
- `Member?` values: `Yes` / `Yes?` -> p=1; `cdots` / blank / other -> p=0.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ASCII_FILE = "apj519209t2_ascii.txt"
SECTION_HEADER = "Horologium 1"

COLUMN_MAPPING = {
    "Object": "Name_source_id",
    "alpha(J2000)": "RA_star",
    "delta(J2000)": "Dec_star (LaTeX `$…$` stripped)",
    "V_hel": "V (+ sigma_eps from `+or-` or `\\pm` split)",
    "Member?": "p (Yes/Yes? -> 1; else 0)",
}

_V_RE = re.compile(
    r"^\$?\s*([-+]?\d+\.?\d*)\s*(?:\+or-|\\pm)\s*(\d+\.?\d*)\s*\$?$"
)


def _strip_dollars(s: str) -> str:
    return s.strip().lstrip("$").rstrip("$").strip()


def _parse_velocity_cell(cell: str) -> tuple[float, float]:
    s = cell.strip()
    if s in ("...", "cdots", ""):
        return (float("nan"), float("nan"))
    m = _V_RE.match(s)
    if not m:
        raise ValueError(f"unparseable V_hel cell: {cell!r}")
    return (float(m.group(1)), float(m.group(2)))


def _slice_horologium(text: str) -> list[list[str]]:
    in_block = False
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == SECTION_HEADER:
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            continue
        rows.append(line.split("\t"))
    if not rows:
        raise RuntimeError("Horologium 1 section not found in Koposov+2015 Table 2.")
    return rows


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "horologium_1":
        raise ValueError(
            f"koposov2015 adapter only serves horologium_1; got {registry_row['lvdb_key']!r}"
        )
    text = (staged_dir / ASCII_FILE).read_text()
    rows = _slice_horologium(text)
    n = len(rows)

    names = np.array([r[0].strip() for r in rows], dtype=str)
    ra = np.array([float(r[1]) for r in rows], dtype=float)
    dec = np.array([float(_strip_dollars(r[2])) for r in rows], dtype=float)

    V = np.empty(n, dtype=float)
    sigma_eps = np.empty(n, dtype=float)
    for i, r in enumerate(rows):
        V[i], sigma_eps[i] = _parse_velocity_cell(r[4])

    member_flag = np.array([r[10].strip() for r in rows], dtype=str)
    p = np.where(np.isin(member_flag, ["Yes", "Yes?"]), 1.0, 0.0)

    arrays = {
        "V": V,
        "sigma_eps": sigma_eps,
        "p": p,
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": ra,
        "Dec_star": dec,
        "Name_source_id": names,
        "member_flag": member_flag,
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": ASCII_FILE,
        "source_table_label": "Koposov+2015 Table 2 (Horologium 1 block)",
        "membership_rule": "paper Member? flag (Yes / Yes? -> p=1; else 0)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "Object",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Table 2 stacks Reticulum 2 + Horologium 1; adapter slices the "
            "Horologium 1 block (18 stars, 5 Yes-flagged members). Dec "
            "values are unwrapped from `$…$` LaTeX. V_hel cells accept "
            "both `+or-` and `\\pm` forms."
        ),
    }
    return arrays, meta_extra
