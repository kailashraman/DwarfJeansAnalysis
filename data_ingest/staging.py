"""Shared helpers for verifying staged data folders and computing canonical
per-star quantities used by both Stage 0b Path A (Geha) and Path B ingests."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


def verify_checksums(folder: Path) -> None:
    """Verify every file listed in `folder/checksums.sha256`. Raise on mismatch."""
    chk = folder / "checksums.sha256"
    if not chk.exists():
        raise FileNotFoundError(
            f"Missing {chk}. Stage the folder per data_sources.md "
            "(copy + PROVENANCE.md + checksums.sha256) before re-running."
        )
    for line in chk.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")  # `sha256sum -b` prefix
        path = folder / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"SHA256 mismatch for {path}: expected {expected}, got {actual}. "
                "Re-stage from the upstream source."
            )


def per_star_indices(names: np.ndarray) -> np.ndarray:
    """Map each per-epoch row to a stable per-star integer index.

    Two rows that share the same `names[i]` value receive the same int.
    Indices are assigned in encounter order so the ordering is stable
    across runs.
    """
    seen: dict = {}
    out = np.empty(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        idx = seen.setdefault(name, len(seen))
        out[i] = idx
    return out


def per_star_indices(names: np.ndarray) -> np.ndarray:
    """Map per-epoch row names to a stable per-star integer index.

    For per-epoch catalogs, `data_sources.md` requires `star_id` to be a
    grouping key that lets downstream consumers fold epochs back to
    their parent star. This helper assigns indices in encounter order so
    two rows that share a name receive the same int.
    """
    seen: dict = {}
    out = np.empty(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        idx = seen.setdefault(name, len(seen))
        out[i] = idx
    return out


def projected_radius_kpc(ra_deg: np.ndarray, dec_deg: np.ndarray,
                         ra_center_deg: float, dec_center_deg: float,
                         distance_kpc: float) -> np.ndarray:
    """Small-angle projected radius (kpc), measured from the LVDB-tabulated center.

    R = distance × sin(angular_separation) ≈ distance × angular_separation
    for the sub-degree separations relevant to MW dwarf members. Uses the
    flat-sky (Δα·cos δ_c, Δδ) form, matching `Segue1_test/run_segue1.py:179–183`.
    """
    cos_d = math.cos(math.radians(dec_center_deg))
    dRA = (ra_deg - ra_center_deg) * cos_d
    dDec = dec_deg - dec_center_deg
    sep_deg = np.sqrt(dRA * dRA + dDec * dDec)
    sep_rad = np.deg2rad(sep_deg)
    return distance_kpc * sep_rad
