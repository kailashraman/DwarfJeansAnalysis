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


def pm_meta_from_registry_row(registry_row) -> dict:
    """Pull the seven PM-related columns out of a registry row into ``_meta``.

    Returned keys are the per-galaxy proper-motion fields plus a precomputed
    ``perspective_correction_applicable`` flag, set ``True`` iff all six
    numeric PM fields (μ_α*, μ_δ, and both asymmetric errors per axis) are
    finite. NaN registry values are stored as JSON-safe ``None`` so the
    metadata round-trips through the ``_meta`` JSON in the per-galaxy npz.

    LVDB ``pmra`` is the Gaia convention μ_α* = μ_α cos δ; do not re-apply
    a cos δ factor downstream.
    """
    def _f(name: str) -> float | None:
        try:
            f = float(registry_row[name])
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    pmra = _f("pmra_mas_yr")
    pmra_em = _f("pmra_em_mas_yr")
    pmra_ep = _f("pmra_ep_mas_yr")
    pmdec = _f("pmdec_mas_yr")
    pmdec_em = _f("pmdec_em_mas_yr")
    pmdec_ep = _f("pmdec_ep_mas_yr")
    ref_raw = registry_row["ref_proper_motion"]
    ref = str(ref_raw) if ref_raw not in (None, "") else None

    applicable = all(v is not None for v in (pmra, pmra_em, pmra_ep, pmdec, pmdec_em, pmdec_ep))
    return {
        "lvdb_pmra_mas_yr": pmra,
        "lvdb_pmra_em_mas_yr": pmra_em,
        "lvdb_pmra_ep_mas_yr": pmra_ep,
        "lvdb_pmdec_mas_yr": pmdec,
        "lvdb_pmdec_em_mas_yr": pmdec_em,
        "lvdb_pmdec_ep_mas_yr": pmdec_ep,
        "lvdb_ref_proper_motion": ref,
        "perspective_correction_applicable": applicable,
    }


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
