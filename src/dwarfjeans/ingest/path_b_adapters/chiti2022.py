"""Chiti et al. 2022 (ApJ 939, 41) — adapter for Grus I.

Source: ASCII Table 2, per-epoch with continuation rows. Same shape as
Li+2017 (Eri II) but with a tri-state MEM flag (M / NM / CM); CM rows
have consistent radial velocity but no [Fe/H] (paper Note ^c).

Encoding: M -> p=1, NM/CM -> p=0; original MEM flag preserved in
`member_flag`.

Instrument provenance: Table 2 contains ONLY the new Magellan/IMACS
observations from Chiti+2022 (3 campaigns: 2015, 2019, 2021;
MJDs 57229.33, 58762.03, 59471.02). The IMACS−M2FS = −2.6 km/s offset
discussed in §3.4.1 is a cross-paper calibration against EXTERNAL
M2FS velocities from Walker+2016 — those M2FS data are NOT in this
table and are not ingested by this adapter. Every emitted row is
therefore tagged `Inst="IMACS"`; the framework's
`CombinePolicy.zero_point_offsets_kms` hook is unused for Grus I as
long as we only ingest Chiti+2022. (If a future commit merges
Walker+2016 M2FS observations into the Grus I per-epoch table, the
IMACS rows already carry the right tag and the offset would be wired
in `combiners/chiti2022.py`.)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ASCII_FILE = "apjac96edt2_ascii.txt"

COLUMN_MAPPING = {
    "ID": "Name_source_id (DES J... designation)",
    "MJD": "MJD (epoch)",
    "R.A.": "RA_star",
    "Decl.": "Dec_star",
    "v": "V (+ sigma_eps from `+or-` split)",
    "MEM": "p (M->1; NM/CM->0)",
    "(constant)": "Inst='IMACS' (Table 2 is IMACS-only; see docstring)",
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
    out: list[dict] = []
    cur_id: str | None = None
    cur_ra = float("nan")
    cur_dec = float("nan")
    cur_mem: str | None = None
    cur_star_idx: int = -1
    name_to_idx: dict[str, int] = {}

    for line in text.splitlines():
        if not line.strip() or line.startswith(("Note", "Table")):
            continue
        cells = line.split("\t")
        if len(cells) < 9:
            continue
        first = cells[0].strip()
        if first == "ID" or first.startswith("("):
            continue
        try:
            float(cells[1].strip())
        except ValueError:
            continue

        if first.startswith("DES J"):
            cur_id = first
            cur_ra = float(cells[2])
            cur_dec = float(cells[3])
            cur_mem = cells[8].strip()
            if cur_id not in name_to_idx:
                name_to_idx[cur_id] = len(name_to_idx)
            cur_star_idx = name_to_idx[cur_id]
        elif first == "":
            if cur_id is None:
                raise RuntimeError("continuation row before any header row")
        else:
            continue

        v, ev = _parse_velocity_cell(cells[7])
        if not np.isfinite(v):
            continue
        out.append({
            "name": cur_id,
            "star_idx": cur_star_idx,
            "mjd": float(cells[1].strip()),
            "ra": cur_ra,
            "dec": cur_dec,
            "v": v,
            "e_v": ev,
            "mem": cur_mem,
        })
    return out


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "grus_1":
        raise ValueError(
            f"chiti2022 adapter only serves grus_1; got {registry_row['lvdb_key']!r}"
        )
    text = (staged_dir / ASCII_FILE).read_text()
    rows = _per_epoch_rows(text)
    n = len(rows)
    if n == 0:
        raise RuntimeError("chiti2022: no per-epoch rows extracted from Table 2.")

    p = np.array([1.0 if r["mem"] == "M" else 0.0 for r in rows], dtype=float)
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
        # Single-instrument provenance: Table 2 is IMACS-only (see
        # module docstring). Stamped explicitly so downstream policy
        # hooks (zero_point_offsets_kms) are usable should Walker+2016
        # M2FS data ever be merged in.
        "Inst": np.array(["IMACS"] * n, dtype=str),
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": ASCII_FILE,
        "source_table_label": "Chiti+2022 Table 2 (Grus I velocity measurements)",
        "membership_rule": (
            "paper MEM tri-state: M -> p=1, NM/CM -> p=0; original flag "
            "preserved in `member_flag` for downstream re-inclusion of CM"
        ),
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_epoch",
        "star_id_source_column": "ID (per-star index assigned in encounter order)",
        "epoch_column": "MJD",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "70 unique stars (8 M, 4 CM, 58 NM), 80 per-epoch rows. "
            "Continuation rows for additional epochs of the same star are "
            "recognized by an empty `ID` cell and inherit ID/coords/MEM "
            "from the preceding header row."
        ),
    }
    return arrays, meta_extra
