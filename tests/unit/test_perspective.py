"""Tests for the Kaplinghat–Strigari perspective-motion correction."""

import math

import numpy as np
import pytest

from dwarfjeans.jeans.perspective import (
    A_KMS_PER_MASYR_KPC,
    perspective_correction,
    perspective_correction_full,
    sanity_check,
)


# Canonical test config: Sculptor-like (Pace+ 2022 PM, McConnachie+ 2012 distance).
RA0, DEC0 = 15.0183, -33.7186
D_KPC = 86.0
PMRA, PMDEC = 0.10, -0.158
V_SYS = 111.4


def test_constant_value():
    """A independent of compilation: 4.74047 km/s per (mas/yr · kpc)."""
    # 1 mas/yr at 1 kpc → 4.74047 km/s.
    mas = math.radians(1e-3 / 3600.0)
    kpc_km = 3.0856775814913673e16
    yr_s = 365.25 * 86400.0
    expected = mas * kpc_km / yr_s
    assert A_KMS_PER_MASYR_KPC == pytest.approx(expected, rel=1e-4)


def test_zero_offset_yields_zero():
    dv = perspective_correction(
        ra_deg=np.array([RA0]), dec_deg=np.array([DEC0]),
        ra_center_deg=RA0, dec_center_deg=DEC0,
        distance_kpc=D_KPC,
        pm_alpha_star_masyr=PMRA, pm_delta_masyr=PMDEC,
    )
    assert dv.shape == (1,)
    assert dv[0] == 0.0


def test_pure_east_offset_matches_formula():
    """Star 0.5° east of center, μ_δ = 0 → Δv = A·d·μ_α*·Δα*_rad."""
    dRA_deg = 0.5
    ra = np.array([RA0 + dRA_deg / math.cos(math.radians(DEC0))])
    dec = np.array([DEC0])
    dv = perspective_correction(
        ra_deg=ra, dec_deg=dec,
        ra_center_deg=RA0, dec_center_deg=DEC0,
        distance_kpc=D_KPC,
        pm_alpha_star_masyr=PMRA, pm_delta_masyr=0.0,
    )
    expected = A_KMS_PER_MASYR_KPC * D_KPC * PMRA * math.radians(dRA_deg)
    assert dv[0] == pytest.approx(expected, rel=1e-12)


def test_pure_north_offset_matches_formula():
    """Star 0.5° north of center, μ_α* = 0 → Δv = A·d·μ_δ·Δδ_rad."""
    dDec_deg = 0.5
    ra = np.array([RA0])
    dec = np.array([DEC0 + dDec_deg])
    dv = perspective_correction(
        ra_deg=ra, dec_deg=dec,
        ra_center_deg=RA0, dec_center_deg=DEC0,
        distance_kpc=D_KPC,
        pm_alpha_star_masyr=0.0, pm_delta_masyr=PMDEC,
    )
    expected = A_KMS_PER_MASYR_KPC * D_KPC * PMDEC * math.radians(dDec_deg)
    assert dv[0] == pytest.approx(expected, rel=1e-12)


def test_sign_eastward_motion_eastward_star():
    """μ_α* > 0 ⇒ galaxy moves east; star east of center should see +Δv."""
    ra = np.array([RA0 + 0.1 / math.cos(math.radians(DEC0))])
    dec = np.array([DEC0])
    dv = perspective_correction(
        ra_deg=ra, dec_deg=dec,
        ra_center_deg=RA0, dec_center_deg=DEC0,
        distance_kpc=D_KPC,
        pm_alpha_star_masyr=+1.0, pm_delta_masyr=0.0,
    )
    assert dv[0] > 0


def test_antisymmetry_under_offset_flip():
    """Δv(−offset) = −Δv(+offset) in the small-angle limit."""
    ra_off = np.array([RA0 + 0.1 / math.cos(math.radians(DEC0))])
    ra_on  = np.array([RA0 - 0.1 / math.cos(math.radians(DEC0))])
    dec    = np.array([DEC0])
    dv_plus = perspective_correction(ra_off, dec, RA0, DEC0, D_KPC, PMRA, PMDEC)
    dv_minus = perspective_correction(ra_on,  dec, RA0, DEC0, D_KPC, PMRA, PMDEC)
    assert dv_plus[0] == pytest.approx(-dv_minus[0], rel=1e-12)


def test_small_angle_agrees_with_full_to_quadratic_order():
    """At ρ < 0.5° the small-angle formula matches K&S eq. 1 to <0.01 km/s."""
    rng = np.random.default_rng(42)
    # ~30 arcmin field, larger than any classical dwarf's R_h.
    ra = RA0 + rng.uniform(-0.5, 0.5, 200) / math.cos(math.radians(DEC0))
    dec = DEC0 + rng.uniform(-0.5, 0.5, 200)
    dv_small = perspective_correction(ra, dec, RA0, DEC0, D_KPC, PMRA, PMDEC)
    dv_full = perspective_correction_full(
        ra, dec, RA0, DEC0, D_KPC, PMRA, PMDEC, V_SYS,
    )
    # Worst-case residual is the dropped (1/2) v_sys ρ² term plus a smaller
    # O(A·d·μ·sin δ_0 · ρ²) cross-term from Δα·cos δ_0 vs cos δ · sin Δα.
    rho2_max = 2.0 * (math.radians(0.5)) ** 2
    bound = (0.5 * abs(V_SYS)
             + A_KMS_PER_MASYR_KPC * D_KPC * max(abs(PMRA), abs(PMDEC)) * abs(math.sin(math.radians(DEC0)))
             ) * rho2_max + 5e-3
    assert np.max(np.abs(dv_small - dv_full)) < bound


def test_full_formula_zero_at_center():
    dv = perspective_correction_full(
        np.array([RA0]), np.array([DEC0]), RA0, DEC0, D_KPC,
        PMRA, PMDEC, V_SYS,
    )
    assert abs(dv[0]) < 1e-12


def test_sanity_reports_no_pm():
    rep = sanity_check(
        ra_deg=np.array([RA0, RA0 + 0.1]),
        dec_deg=np.array([DEC0, DEC0 + 0.1]),
        ra_center_deg=RA0, dec_center_deg=DEC0,
        distance_kpc=D_KPC,
        pm_alpha_star_masyr=None, pm_delta_masyr=None,
        v_sys_kms=V_SYS,
    )
    assert rep.pm_available is False
    assert rep.max_abs_kms == 0.0
    assert "no published PM" in str(rep)


def test_sanity_reports_diagnostics_for_classical():
    """Sculptor-like config produces non-zero RMS / max and a small full-vs-small residual."""
    rng = np.random.default_rng(7)
    ra = RA0 + rng.uniform(-0.3, 0.3, 500) / math.cos(math.radians(DEC0))
    dec = DEC0 + rng.uniform(-0.3, 0.3, 500)
    rep = sanity_check(
        ra_deg=ra, dec_deg=dec,
        ra_center_deg=RA0, dec_center_deg=DEC0,
        distance_kpc=D_KPC,
        pm_alpha_star_masyr=PMRA, pm_delta_masyr=PMDEC,
        v_sys_kms=V_SYS,
    )
    assert rep.pm_available is True
    assert rep.max_abs_kms > 0.05
    assert rep.rms_kms > 0.0
    assert rep.quadratic_term_max_abs_kms < 0.05
    assert rep.small_vs_full_residual_kms < 0.05


def test_vector_shape_preserved():
    ra = np.array([RA0, RA0 + 0.1, RA0 + 0.2])
    dec = np.array([DEC0, DEC0 + 0.05, DEC0 - 0.05])
    dv = perspective_correction(ra, dec, RA0, DEC0, D_KPC, PMRA, PMDEC)
    assert dv.shape == ra.shape
