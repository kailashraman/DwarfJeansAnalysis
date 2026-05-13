"""Kaplinghat–Strigari perspective-motion correction for per-star v_los.

For a satellite with measured bulk proper motion the observed line-of-sight
velocity of a star at angular offset (Δα·cos δ_0, Δδ) from the galaxy center
is shifted relative to the galaxy systemic by the projection of the bulk
plane-of-sky velocity onto the star's line of sight (Walker et al.\\ 2008,
Appendix; Kaplinghat & Strigari 2008). In the small-angle limit relevant to
dwarf-galaxy fields (ρ ≪ 1 rad):

    Δv_persp(R) = A · d · (μ_α* · Δα* + μ_δ · Δδ)         [km/s]

with A = 4.7404 km/s / (mas/yr · kpc), d in kpc, μ_α* = μ_α cos δ in mas/yr
(Gaia convention), and Δα* = (RA − RA_0) cos δ_0 in radians, Δδ in radians.
The dropped quadratic term enters K&S eq. 1 with a minus sign and is below
0.05 km/s for ρ < 0.5° at |v_sys| ≲ 300 km/s; the helper
``perspective_correction_full`` keeps it for cross-checks.

Corrected v_los: ``v_corr = v_los − Δv_persp``. This module computes
Δv_persp only; subtraction happens in the Stage 2 caller after the
per-galaxy systemic v_sys is fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

A_KMS_PER_MASYR_KPC = 4.74047  # km/s per (mas/yr · kpc); Allen 1973 / Gaia DPAC


def perspective_correction(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra_center_deg: float,
    dec_center_deg: float,
    distance_kpc: float,
    pm_alpha_star_masyr: float,
    pm_delta_masyr: float,
) -> np.ndarray:
    """Per-star perspective shift Δv_persp (km/s), small-angle limit.

    Parameters
    ----------
    ra_deg, dec_deg :
        Per-star RA/Dec (degrees). Arrays of equal length.
    ra_center_deg, dec_center_deg :
        Galaxy center RA/Dec (degrees).
    distance_kpc :
        Heliocentric distance (kpc).
    pm_alpha_star_masyr :
        Galaxy bulk μ_α* = μ_α cos δ (mas/yr; Gaia convention).
    pm_delta_masyr :
        Galaxy bulk μ_δ (mas/yr).

    Returns
    -------
    dv_persp_kms :
        Per-star perspective shift in km/s. Add to the modelled mean to get
        the expected observed v_los, or subtract from the observed v_los
        before forming the Jeans likelihood residual.
    """
    cos_d0 = math.cos(math.radians(dec_center_deg))
    dRA_rad = np.deg2rad(np.asarray(ra_deg, dtype=float) - ra_center_deg) * cos_d0
    dDec_rad = np.deg2rad(np.asarray(dec_deg, dtype=float) - dec_center_deg)
    coeff = A_KMS_PER_MASYR_KPC * float(distance_kpc)
    return coeff * (pm_alpha_star_masyr * dRA_rad + pm_delta_masyr * dDec_rad)


def perspective_correction_full(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra_center_deg: float,
    dec_center_deg: float,
    distance_kpc: float,
    pm_alpha_star_masyr: float,
    pm_delta_masyr: float,
    v_sys_kms: float,
) -> np.ndarray:
    """Full Kaplinghat & Strigari (2008) eq. 1, keeping the ½ v_sys ρ² term.

    Used for cross-checking ``perspective_correction``. Returns the same
    "subtract from v_los" quantity. ``v_sys_kms`` must be in the same
    frame as the per-star ``v_los`` it will be applied to (heliocentric
    for the catalogs ingested here).
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    ra0 = math.radians(ra_center_deg)
    dec0 = math.radians(dec_center_deg)

    cos_rho = (np.sin(dec) * math.sin(dec0)
               + np.cos(dec) * math.cos(dec0) * np.cos(ra - ra0))
    cos_rho = np.clip(cos_rho, -1.0, 1.0)

    # Spherical-trig decomposition of the star's unit LOS vector into the
    # tangent-plane east/north components at the galaxy center. Both vanish
    # at the center and reduce to Δα* and Δδ (radians) at small ρ.
    east  = np.cos(dec) * np.sin(ra - ra0)
    north = (np.sin(dec) * math.cos(dec0)
             - np.cos(dec) * math.sin(dec0) * np.cos(ra - ra0))

    v_east  = A_KMS_PER_MASYR_KPC * float(distance_kpc) * pm_alpha_star_masyr
    v_north = A_KMS_PER_MASYR_KPC * float(distance_kpc) * pm_delta_masyr

    # K&S eq. 1, expressed with v_sys = +v_los at ρ=0 (the conventional
    # systemic). Δv relative to v_sys: linear plane-of-sky projection minus
    # the v_sys(1 − cos ρ) "perspective contraction" term.
    dv_relative_to_sys = (v_east * east
                          + v_north * north
                          - float(v_sys_kms) * (1.0 - cos_rho))
    return dv_relative_to_sys


@dataclass
class SanityReport:
    n_stars: int
    pm_available: bool
    rms_kms: float
    max_abs_kms: float
    quadratic_term_max_abs_kms: float
    small_vs_full_residual_kms: float

    def __str__(self) -> str:
        if not self.pm_available:
            return f"perspective: N={self.n_stars}, no published PM — nothing to compute."
        return (
            f"perspective: N={self.n_stars}, RMS={self.rms_kms:.4f} km/s, "
            f"|Δv|_max={self.max_abs_kms:.4f} km/s, "
            f"dropped-quad |term|_max={self.quadratic_term_max_abs_kms:.4f} km/s, "
            f"|small−full|_max={self.small_vs_full_residual_kms:.4f} km/s."
        )


def sanity_check(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra_center_deg: float,
    dec_center_deg: float,
    distance_kpc: float,
    pm_alpha_star_masyr: float | None,
    pm_delta_masyr: float | None,
    v_sys_kms: float,
) -> SanityReport:
    """Diagnostics for the perspective correction on a single galaxy.

    Returns RMS and peak |Δv_persp|, the worst-case dropped ½ v_sys ρ²
    quadratic term, and the residual between the small-angle form and
    the full K&S eq. 1. No apply/skip recommendation — the caller
    decides whether to subtract Δv_persp from v_los. The only hard skip
    is the no-PM case (``pm_available = False``), in which there is
    nothing to compute.
    """
    if pm_alpha_star_masyr is None or pm_delta_masyr is None:
        return SanityReport(
            n_stars=int(np.asarray(ra_deg).size),
            pm_available=False,
            rms_kms=0.0,
            max_abs_kms=0.0,
            quadratic_term_max_abs_kms=0.0,
            small_vs_full_residual_kms=0.0,
        )

    dv = perspective_correction(
        ra_deg, dec_deg, ra_center_deg, dec_center_deg, distance_kpc,
        pm_alpha_star_masyr, pm_delta_masyr,
    )
    rms = float(np.sqrt(np.mean(dv * dv)))
    max_abs = float(np.max(np.abs(dv)))

    cos_d0 = math.cos(math.radians(dec_center_deg))
    dRA = np.deg2rad(np.asarray(ra_deg, dtype=float) - ra_center_deg) * cos_d0
    dDec = np.deg2rad(np.asarray(dec_deg, dtype=float) - dec_center_deg)
    rho2 = dRA * dRA + dDec * dDec
    quad_max = float(0.5 * abs(v_sys_kms) * np.max(rho2))

    dv_full = perspective_correction_full(
        ra_deg, dec_deg, ra_center_deg, dec_center_deg, distance_kpc,
        pm_alpha_star_masyr, pm_delta_masyr, v_sys_kms,
    )
    full_residual = float(np.max(np.abs(dv - dv_full)))

    return SanityReport(
        n_stars=int(np.asarray(ra_deg).size),
        pm_available=True,
        rms_kms=rms,
        max_abs_kms=max_abs,
        quadratic_term_max_abs_kms=quad_max,
        small_vs_full_residual_kms=full_residual,
    )
