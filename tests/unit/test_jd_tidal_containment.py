"""Tidal radius + exact-geometry J-factor angular containment.

Covers src/dwarfjeans/jd/tidal.py (Tormen/Springel r_t, galactocentric
distance) and the ported containment in src/dwarfjeans/jd/factors.py
(los_radius, j_containment, j_aperture_and_containment).
"""

import numpy as np
import pytest
from scipy.integrate import dblquad

from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jd import factors as jdf
from dwarfjeans.jd import tidal

# A representative resolved UFD-ish NFW.
RS, RHOS, D = 0.3, 3.0e8, 30.0


# --------------------------------------------------------------------------- #
# Tidal radius
# --------------------------------------------------------------------------- #

def test_tidal_radius_satisfies_implicit_equation():
    host = tidal.SATGEN_M12_HOST
    Dgc = 50.0
    rt = tidal.tidal_radius(RS, RHOS, Dgc)
    assert np.isfinite(rt) and rt > 0.0
    lhs = rt ** 3 * (2.0 - host.dlnM_dlnr(Dgc)) * host.M_enc(Dgc)
    rhs = Dgc ** 3 * jeans.nfw_M(rt, RS, RHOS)
    assert abs(lhs - rhs) / rhs < 1e-6


def test_tidal_radius_hill_limit_point_mass():
    """factor=3 with a point-mass host (gamma=0) -> Hill radius D(m/3M)^(1/3)."""
    class PointMass:
        def M_enc(self, r):
            return 1.0e12
        def dlnM_dlnr(self, r):
            return 0.0

    rt = tidal.tidal_radius(RS, RHOS, 50.0, host=PointMass(), factor=3.0)
    m_sub = jeans.nfw_M(rt, RS, RHOS)
    hill = 50.0 * (m_sub / (3.0 * 1.0e12)) ** (1.0 / 3.0)
    assert rt == pytest.approx(hill, rel=1e-6)


def test_tidal_radius_monotonic_in_subhalo_density():
    # Denser subhalo (larger rho_s) is harder to strip -> larger r_t.
    rts = [tidal.tidal_radius(RS, rhos, 50.0) for rhos in (1e8, 3e8, 1e9)]
    assert rts[0] < rts[1] < rts[2]


def test_tidal_radius_factor3_smaller_than_factor2():
    # Jacobi (3 - gamma) denominator is larger -> smaller r_t than radial (2 - gamma).
    rt2 = tidal.tidal_radius(RS, RHOS, 50.0, factor=2.0)
    rt3 = tidal.tidal_radius(RS, RHOS, 50.0, factor=3.0)
    assert rt3 < rt2


def test_host_normalisation():
    host = tidal.SATGEN_M12_HOST
    assert host.M_enc(host.Rvir_kpc) == pytest.approx(host.Mvir_Msun, rel=1e-9)


def test_galactocentric_distance_toward_gc_is_small():
    # Galactic-centre direction (ICRS ra/dec of the GC) at the Sun-GC distance
    # lands essentially at the Galactic centre.
    d_sun_gc = 8.122
    Dgc = tidal.galactocentric_distance(266.4051, -28.936175, d_sun_gc)
    assert Dgc < 0.1  # only the ~21 pc z_sun offset remains


def test_galactocentric_distance_vectorized():
    d = np.array([20.0, 30.0, 40.0])
    Dgc = tidal.galactocentric_distance(100.0, -60.0, d)
    assert Dgc.shape == (3,)
    assert np.all(np.isfinite(Dgc)) and np.all(Dgc > 0)


# --------------------------------------------------------------------------- #
# Exact-geometry J + containment
# --------------------------------------------------------------------------- #

def _rt_here(Dgc=None):
    return tidal.tidal_radius(RS, RHOS, Dgc if Dgc is not None else D)


def test_exact_J_matches_small_angle_at_half_degree():
    rt = _rt_here()
    Jx, _ = jdf.j_aperture_and_containment(D, RS, RHOS, rt, aperture_rad=0.5 * jdf.DEG)
    Jsa, _ = jdf.J_D_factors(0.5 * jdf.DEG, D, RS, RHOS, rt)
    log_x = np.log10(Jx)
    log_sa = np.log10(Jsa) + jdf.LOG10_J_FAC
    assert abs(log_x - log_sa) < 0.01   # < 0.01 dex


def test_containment_recovers_fraction():
    """J(<theta95) / J_full == frac by construction (cross-checked via the API)."""
    rt = _rt_here()
    _, th95 = jdf.j_aperture_and_containment(D, RS, RHOS, rt, frac=0.95)
    # J_full: aperture beyond the halo's angular extent is clamped to theta_max.
    J_full, _ = jdf.j_aperture_and_containment(D, RS, RHOS, rt, aperture_rad=np.pi / 2)
    J_95, _ = jdf.j_aperture_and_containment(D, RS, RHOS, rt, aperture_rad=th95)
    # ~0.5% tolerance: J_full and J_95 come from independent quadratures whose
    # log-spaced theta grids differ (each forces a different aperture node).
    assert J_95 / J_full == pytest.approx(0.95, abs=5e-3)


def test_containment_angle_within_geometric_bound():
    rt = _rt_here()
    _, th95 = jdf.j_aperture_and_containment(D, RS, RHOS, rt)
    theta_geom = np.arcsin(min(rt / D, 1.0))
    assert 0.0 < th95 < theta_geom


def test_containment_shrinks_with_distance():
    # More distant dwarf is more point-like -> smaller containment angle.
    angles = []
    for dd in (20.0, 40.0, 80.0):
        rt = _rt_here(tidal.galactocentric_distance(0.0, -60.0, dd))
        _, th = jdf.j_aperture_and_containment(dd, RS, RHOS, rt)
        angles.append(th)
    assert angles[0] > angles[1] > angles[2]


def test_full_J_matches_scipy_dblquad():
    """Gold-standard: J_full vs an independent scipy dblquad on the truncated NFW."""
    rt = _rt_here()

    def rho2(r):
        return np.where(r <= rt, jdf.nfw_rho(r, RS, RHOS), 0.0) ** 2

    theta_max = np.arcsin(min(rt / D, 1.0))

    def integrand(L, theta):
        r = jdf.los_radius(L, D, theta)
        return rho2(r) * 2.0 * np.pi * np.sin(theta)

    ref, _ = dblquad(integrand, 0.0, theta_max, lambda t: 0.0, lambda t: 2.0 * D,
                     epsabs=0.0, epsrel=1e-6)
    ref_gev = ref * 10.0 ** jdf.LOG10_J_FAC

    J_full, _ = jdf.j_aperture_and_containment(D, RS, RHOS, rt, aperture_rad=np.pi / 2)
    assert abs(np.log10(J_full) - np.log10(ref_gev)) < 0.01   # < 0.01 dex


def test_theta95_matches_independent_quad_brentq():
    """Gold-standard for the containment ANGLE itself (not just J_full): solve
    theta95 with an independent scipy quad+brentq pipeline that integrates
    dJ/dtheta directly (no Gauss-Legendre sec-substitution, no log-theta grid,
    no cusp w[0]=w[1] patch). The suite's J_95/J_full==0.95 check is
    self-consistent and cannot catch a cdf-shape bias shared by numerator and
    denominator; this can."""
    import warnings
    from scipy.integrate import quad, IntegrationWarning
    from scipy.optimize import brentq

    rt = _rt_here()
    theta_max = float(np.arcsin(min(rt / D, 1.0)))

    def dJ_dtheta(theta):
        # Integrate only the supported LOS segment: rho>0 where r<=rt, i.e.
        # L in [L0-half, L0+half] with L0=d cos(theta) (closest approach),
        # half=sqrt(rt^2 - rmin^2), rmin=d sin(theta). Flag the cusp peak at L0
        # so scipy quad needn't hunt for the jump/peak (fast + no roundoff warns).
        rmin = D * np.sin(theta)
        if rmin >= rt:
            return 0.0
        L0 = D * np.cos(theta)
        half = np.sqrt(rt ** 2 - rmin ** 2)
        La, Lb = max(0.0, L0 - half), min(2.0 * D, L0 + half)
        if not (Lb > La):
            return 0.0
        pts = [L0] if La < L0 < Lb else None
        inner, _ = quad(lambda L: jdf.nfw_rho(jdf.los_radius(L, D, theta), RS, RHOS) ** 2,
                        La, Lb, points=pts, limit=200)
        return 2.0 * np.pi * np.sin(theta) * inner

    def cumJ(theta):
        val, _ = quad(dJ_dtheta, 0.0, theta, limit=200)
        return val

    # scipy quad emits convergence chatter on the tall, narrow rho^2 cusp; the
    # <1% cross-check below is the real correctness gate, so silence the noise.
    # Scope the suppression to the outer theta-integral only — let any genuine
    # convergence failure during the brentq search surface rather than pass mute.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        target = 0.95 * cumJ(theta_max)
    th95_ref = brentq(lambda t: cumJ(t) - target, 1e-6, theta_max, xtol=1e-9)

    _, th95 = jdf.j_aperture_and_containment(D, RS, RHOS, rt, frac=0.95)
    assert abs(th95 - th95_ref) / th95_ref < 0.01   # < 1%


# --------------------------------------------------------------------------- #
# Regression: the physical r_t rewire changes D, barely moves J
# --------------------------------------------------------------------------- #

def test_physical_rt_changes_D_more_than_J():
    """Guards the rewire: switching r_t=1 kpc -> physical r_t leaves J(0.5°)
    essentially unchanged but materially raises the more-extended D(0.5°)."""
    rt = _rt_here()                      # ~3 kpc for this UFD at D=30 kpc
    assert rt > 1.5                      # physical r_t is well above the old 1 kpc
    J_old, D_old = jdf.J_D_factors(0.5 * jdf.DEG, D, RS, RHOS, 1.0)
    J_new, D_new = jdf.J_D_factors(0.5 * jdf.DEG, D, RS, RHOS, rt)
    dlog_J = abs(np.log10(J_new) - np.log10(J_old))
    dlog_D = np.log10(D_new) - np.log10(D_old)
    assert dlog_J < 0.005                # J ~ rho^2: r_t essentially irrelevant
    assert dlog_D > 0.015                # D ~ rho: r_t materially raises D (~4%)
    assert dlog_D > 3.0 * dlog_J         # and the D shift dominates the J shift
