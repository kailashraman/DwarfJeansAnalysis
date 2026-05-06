"""Li et al. 2017 (ApJ 838, 8) — adapter for Eridanus II.

Source: ASCII Table 2, per-epoch with continuation rows. The first row of
each star carries the `ID`, coords, mags, and `MEM`; subsequent epochs of
the same star are rows with an empty `ID` cell — only `MJD`, `S/N`, `v`,
`EW`, `[Fe/H]` are repopulated. The adapter inherits ID / RA / Dec / MEM
from the most recent header row.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ASCII_FILE = "apjaa6113t2_ascii.txt"

COLUMN_MAPPING = {
    "ID": "Name_source_id (DES J... designation)",
    "MJD": "MJD (epoch)",
    "RA": "RA_star",
    "Decl.": "Dec_star",
    "v": "V (+ sigma_eps from `+or-` split)",
    "MEM": "p (1->1, 0->0) — inherited across epochs of the same star",
}

_V_RE = re.compile(r"^([-+]?\d+\.?\d*)\s*\+or-\s*(\d+\.?\d*)$")


def _parse_velocity_cell(cell: str) -> tuple[float, float]:
    s = cell.strip()
    if s in ("...", ""):
        return (float("nan"), float("nan"))
    m = _V_RE.match(s)
    if not m:
        raise ValueError(f"unparseable v cell: {cell!r}")
    return (float(m.group(1)), float(m.group(2)))


def _per_epoch_rows(text: str) -> list[dict]:
    """Walk lines; each output dict is one epoch with ID/coords/MEM
    inherited from the most recent header row."""
    out: list[dict] = []
    cur_id: str | None = None
    cur_ra = float("nan")
    cur_dec = float("nan")
    cur_mem: str | None = None
    cur_star_idx: int = -1
    name_to_idx: dict[str, int] = {}

    for line in text.splitlines():
        if not line.strip() or line.startswith("Note") or line.startswith("Table"):
            continue
        cells = line.split("\t")
        if len(cells) < 11:
            continue
        first = cells[0].strip()
        if first == "ID":
            continue  # column header
        if first.startswith("("):
            continue  # unit row

        # Skip rows whose MJD cell is not numeric — catches the unit row
        # (which has empty first cell + non-numeric content) and any other
        # non-data lines that slip through.
        mjd_cell = cells[1].strip()
        try:
            float(mjd_cell)
        except ValueError:
            continue

        if first.startswith("DES J"):
            cur_id = first
            cur_ra = float(cells[2])
            cur_dec = float(cells[3])
            cur_mem = cells[10].strip()
            if cur_id not in name_to_idx:
                name_to_idx[cur_id] = len(name_to_idx)
            cur_star_idx = name_to_idx[cur_id]
        elif first == "":
            if cur_id is None:
                raise RuntimeError("continuation row before any header row")
        else:
            continue  # unrecognized line

        v, ev = _parse_velocity_cell(cells[7])
        if not np.isfinite(v):
            continue
        out.append({
            "name": cur_id,
            "star_idx": cur_star_idx,
            "mjd": float(mjd_cell),
            "ra": cur_ra,
            "dec": cur_dec,
            "v": v,
            "e_v": ev,
            "mem": cur_mem,
        })
    return out


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "eridanus_2":
        raise ValueError(
            f"li2017 adapter only serves eridanus_2; got {registry_row['lvdb_key']!r}"
        )
    text = (staged_dir / ASCII_FILE).read_text()
    rows = _per_epoch_rows(text)
    n = len(rows)
    if n == 0:
        raise RuntimeError("li2017: no per-epoch rows extracted from Table 2.")

    p = np.array([1.0 if r["mem"] == "1" else 0.0 for r in rows], dtype=float)
    arrays = {
        "V": np.array([r["v"] for r in rows], dtype=float),
        "sigma_eps": np.array([r["e_v"] for r in rows], dtype=float),
        "p": p,
        "star_id": np.array([r["star_idx"] for r in rows], dtype=np.int64),
        "RA_star": np.array([r["ra"] for r in rows], dtype=float),
        "Dec_star": np.array([r["dec"] for r in rows], dtype=float),
        "Name_source_id": np.array([r["name"] for r in rows], dtype=str),
        "MJD": np.array([r["mjd"] for r in rows], dtype=float),
        "member_flag": np.array([r["mem"] for r in rows], dtype=str),
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": ASCII_FILE,
        "source_table_label": "Li+2017 Table 2 (Eridanus II velocity + metallicity)",
        "membership_rule": "paper MEM flag (1->p=1, 0->p=0); inherited across continuation rows",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_epoch",
        "star_id_source_column": "ID (per-star index assigned in encounter order)",
        "epoch_column": "MJD",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "54 unique stars (28 members), 93 per-epoch rows. Most stars "
            "have two MJD epochs (Nov 2015 / Oct 2015 IMACS runs); "
            "continuation rows are recognized by an empty `ID` cell and "
            "inherit ID, coords, and MEM from the preceding header row. "
            "star_id is the unique-name index in encounter order."
        ),
    }
    return arrays, meta_extra
