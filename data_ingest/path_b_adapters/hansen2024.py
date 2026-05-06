"""Hansen et al. 2024 (ApJ 968, 21) — adapter for Tucana V.

Source: ASCII Table 1 ("Observing Log"), per-epoch heliocentric velocity
across 3 Tuc V member stars and ~17 epochs (MIKE + IMACS). The first row
of each star carries the Gaia DR3 ID and sexagesimal coords; continuation
rows have an empty or `(Tuc V-N)` first cell and `cdots` placeholders for
coords/mags/SNR.

The paper publishes a 3-star member list; all 17 epochs are assigned p=1
per the data_sources.md missing-probability default.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

ASCII_FILE = "apjad3a52t1_ascii.txt"

COLUMN_MAPPING = {
    "Gaia DR3 ID": "Name_source_id",
    "R.A.": "RA_star (sexagesimal hh:mm:ss[.ss] -> decimal deg)",
    "Decl.": "Dec_star (sexagesimal -> decimal deg)",
    "MJD": "MJD (epoch)",
    "V_hel": "V (+ sigma_eps from `+or-` split)",
    "(member-list -> p=1)": "p",
}

_V_RE = re.compile(r"^([-+]?\d+\.?\d*)\s*\+or-\s*(\d+\.?\d*)$")
_RA_FOURCOLON_RE = re.compile(r"^(\d+):(\d+):(\d+):(\d+)$")


def _normalize_ra(s: str) -> str:
    """Convert `hh:mm:ss:ff` typo to `hh:mm:ss.ff`. Pass-through otherwise."""
    s = s.strip()
    m = _RA_FOURCOLON_RE.match(s)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}.{m.group(4)}"
    return s


def _parse_velocity_cell(cell: str) -> tuple[float, float]:
    s = cell.strip()
    if s in ("...", "cdots", ""):
        return (float("nan"), float("nan"))
    m = _V_RE.match(s)
    if not m:
        raise ValueError(f"unparseable V_hel cell: {cell!r}")
    return (float(m.group(1)), float(m.group(2)))


def _per_epoch_rows(text: str) -> list[dict]:
    """Header row = first cell is a 19-digit Gaia DR3 source_id.
    Continuation row = first cell is empty or `(Tuc V-N)`.
    Unit / Note / column-header rows are skipped."""
    out: list[dict] = []
    cur_gaia: str | None = None
    cur_ra = float("nan")
    cur_dec = float("nan")
    cur_star_idx: int = -1
    name_to_idx: dict[str, int] = {}

    for line in text.splitlines():
        if not line.strip() or line.startswith(("Note", "Table")):
            continue
        cells = line.split("\t")
        if len(cells) < 13:
            continue
        first = cells[0].strip()
        if first == "Gaia DR3 ID" or first.startswith("("):
            # column header or unit row; skip — but `(Tuc V-N)` must NOT
            # be filtered here, so use a tighter check
            if first.startswith("(Tuc"):
                pass  # continuation row labelled with internal name
            else:
                continue

        # Distinguish header from continuation
        is_header = bool(re.match(r"^\d{15,19}$", first))
        if is_header:
            cur_gaia = first
            ra_raw = _normalize_ra(cells[1])
            dec_raw = cells[2].strip()
            coord = SkyCoord(ra=ra_raw, dec=dec_raw,
                             unit=(u.hourangle, u.deg), frame="icrs")
            cur_ra = float(coord.ra.deg)
            cur_dec = float(coord.dec.deg)
            if cur_gaia not in name_to_idx:
                name_to_idx[cur_gaia] = len(name_to_idx)
            cur_star_idx = name_to_idx[cur_gaia]
        else:
            if cur_gaia is None:
                continue  # haven't hit a header yet — skip pre-data rows

        # Each data row (header or continuation) carries a real epoch:
        # cells[7]=MJD, cells[11]=V_hel, cells[12]=Instrument
        try:
            mjd = float(cells[7].strip())
        except ValueError:
            continue
        v, ev = _parse_velocity_cell(cells[11])
        if not np.isfinite(v):
            continue
        out.append({
            "gaia": cur_gaia,
            "star_idx": cur_star_idx,
            "mjd": mjd,
            "ra": cur_ra,
            "dec": cur_dec,
            "v": v,
            "e_v": ev,
            "instrument": cells[12].strip(),
        })
    return out


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "tucana_5":
        raise ValueError(
            f"hansen2024 adapter only serves tucana_5; got {registry_row['lvdb_key']!r}"
        )
    text = (staged_dir / ASCII_FILE).read_text()
    rows = _per_epoch_rows(text)
    n = len(rows)
    if n == 0:
        raise RuntimeError("hansen2024: no per-epoch rows extracted from Table 1.")

    arrays = {
        "V": np.array([r["v"] for r in rows], dtype=float),
        "sigma_eps": np.array([r["e_v"] for r in rows], dtype=float),
        "p": np.ones(n, dtype=float),
        "star_id": np.array([r["star_idx"] for r in rows], dtype=np.int64),
        "RA_star": np.array([r["ra"] for r in rows], dtype=float),
        "Dec_star": np.array([r["dec"] for r in rows], dtype=float),
        "Name_source_id": np.array([r["gaia"] for r in rows], dtype=str),
        "MJD": np.array([r["mjd"] for r in rows], dtype=float),
        "Inst": np.array([r["instrument"] for r in rows], dtype=str),
    }
    meta_extra = {
        "vizier_catalog": None,
        "source_table_file": ASCII_FILE,
        "source_table_label": "Hansen+2024 Table 1 (Tuc V Observing Log)",
        "membership_rule": "missing_default (member list, p=1 verbatim)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_epoch",
        "star_id_source_column": "Gaia DR3 ID (per-star index assigned in encounter order)",
        "epoch_column": "MJD",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "3 unique members across 17 epochs (MIKE + IMACS). Tuc V-1 "
            "shows ~21 km/s peak-to-peak velocity variation -> likely "
            "binary; binary-aware aggregation is downstream of Stage 0b. "
            "RA cells written as `hh:mm:ss:ff` on rows 1-2 are normalized "
            "to `hh:mm:ss.ff` before SkyCoord parsing."
        ),
    }
    return arrays, meta_extra
