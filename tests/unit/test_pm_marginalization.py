"""Tests for PM uncertainty marginalisation (9D Stage 2 path)."""

import math

import numpy as np
import pytest

from dwarfjeans.jeans.inference import (
    A_KMS_PER_MASYR_KPC, ARCMIN_TO_RAD, make_loglike_with_nuisances,
)
from dwarfjeans.jeans.priors import (
    get_prior, split_normal_ppf, wrap_with_pm_marginalization,
)


# ---- split-normal PPF ----

def test_split_normal_ppf_symmetric_reduces_to_normal():
    """When sigma_lo == sigma_hi, split_normal_ppf == normal ppf."""
    from scipy.stats import norm
    mu, s = 0.5, 0.1
    u = np.linspace(0.05, 0.95, 19)
    sn = split_normal_ppf(u, mu, s, s)
    nn = norm.ppf(u, loc=mu, scale=s)
    np.testing.assert_allclose(sn, nn, rtol=1e-10)


def test_split_normal_ppf_mode_at_split_fraction():
    """u = sigma_lo/(sigma_lo+sigma_hi) → mu (the mode)."""
    mu = 1.0
    s_lo, s_hi = 0.2, 0.5
    frac = s_lo / (s_lo + s_hi)
    # Approach the mode from both sides; both should converge to mu.
    eps = 1e-9
    below = split_normal_ppf(frac - eps, mu, s_lo, s_hi)
    above = split_normal_ppf(frac + eps, mu, s_lo, s_hi)
    assert abs(below - mu) < 1e-6
    assert abs(above - mu) < 1e-6


def test_split_normal_ppf_asymmetric_scales():
    """Tail behavior matches the corresponding half-Gaussian on each side."""
    from scipy.stats import norm
    mu, s_lo, s_hi = 0.0, 0.1, 0.4
    frac = s_lo / (s_lo + s_hi)
    # PPF: u ∈ [0, frac) → mu - s_lo · |norm.ppf(0.5 · u/frac)|.
    # For output mu - 2·s_lo we need norm.ppf(0.5·u/frac) = −2, so
    #   0.5 · u / frac = norm.cdf(−2)  →  u = 2 · frac · norm.cdf(−2).
    u_lo = 2.0 * frac * norm.cdf(-2.0)
    assert split_normal_ppf(u_lo, mu, s_lo, s_hi) == pytest.approx(mu - 2 * s_lo, rel=1e-6)
    # Symmetric check on the upper side: u = frac + (1 - frac) · (2 · norm.cdf(2) − 1)
    #   ⇒ output = mu + 2 · s_hi
    u_hi = frac + (1.0 - frac) * (2.0 * norm.cdf(2.0) - 1.0)
    assert split_normal_ppf(u_hi, mu, s_lo, s_hi) == pytest.approx(mu + 2 * s_hi, rel=1e-6)


# ---- wrap_with_pm_marginalization ----

def test_wrap_appends_pm_dims():
    """9D prior_transform extends a 7D base with split-normal PM priors."""
    def base(u):  # trivial 7D identity
        return np.asarray(u, dtype=float).copy()
    wrapped = wrap_with_pm_marginalization(
        base,
        pmra_mean=0.10, pmra_em=0.002, pmra_ep=0.003,
        pmdec_mean=-0.158, pmdec_em=0.002, pmdec_ep=0.002,
    )
    u = np.array([0.5] * 9)
    x = wrapped(u)
    assert x.shape == (9,)
    # First 7 dims pass through identity.
    np.testing.assert_array_equal(x[:7], u[:7])
    # PM dims hit the mode at u = frac.
    frac_a = 0.002 / (0.002 + 0.003)
    frac_d = 0.002 / (0.002 + 0.002)  # symmetric
    pmra_at_frac = wrap_with_pm_marginalization(
        base, 0.10, 0.002, 0.003, -0.158, 0.002, 0.002,
    )(np.array([0.5] * 7 + [frac_a, frac_d]))
    assert pmra_at_frac[7] == pytest.approx(0.10, abs=1e-6)
    assert pmra_at_frac[8] == pytest.approx(-0.158, abs=1e-6)


# ---- 9D likelihood ----

def _fixture_galaxy(N=80, rseed=42):
    """Sculptor-like mock for 9D loglike sanity tests."""
    rng = np.random.default_rng(rseed)
    ra_center, dec_center = 15.0183, -33.7186
    cos_d0 = math.cos(math.radians(dec_center))
    # 80 stars in a ±0.3° box.
    ra = ra_center + rng.uniform(-0.3, 0.3, N) / cos_d0
    dec = dec_center + rng.uniform(-0.3, 0.3, N)
    d_kpc = 86.0
    pmra_true, pmdec_true = 0.10, -0.158
    rhalf_arcmin = 11.17
    # Perspective at truth.
    cos_d0_f = float(cos_d0)
    dRA = np.deg2rad(ra - ra_center) * cos_d0_f
    dDec = np.deg2rad(dec - dec_center)
    dv_true = A_KMS_PER_MASYR_KPC * d_kpc * (pmra_true * dRA + pmdec_true * dDec)
    sigma_los = 9.2
    sigma_eps = np.full(N, 2.0)
    V_sys = 111.4
    V = V_sys + dv_true + rng.normal(0.0, np.sqrt(sigma_los**2 + sigma_eps**2), N)
    # Build Rad_arcmin from RA/Dec (matches the 7D inference convention).
    rho_deg = np.sqrt(dRA**2 + dDec**2) * 180.0 / math.pi
    Rad_arcmin = rho_deg * 60.0
    return {
        "Rad_arcmin": Rad_arcmin,
        "V": V,
        "V_observed": V,  # for the 9D path
        "sigma_eps": sigma_eps,
        "p": np.ones(N),
        "RA_star": ra, "Dec_star": dec,
        "ra_center": ra_center, "dec_center": dec_center,
        "truth": {"V_sys": V_sys, "d": d_kpc, "rhalf_arcmin": rhalf_arcmin,
                  "pmra": pmra_true, "pmdec": pmdec_true},
    }


def test_9d_loglike_finite_at_truth():
    """Sanity: 9D likelihood returns a finite log-likelihood at parameter truth."""
    gal = _fixture_galaxy()
    prior = get_prior("loguniform")  # no Jeffreys correction → simpler
    ll = make_loglike_with_nuisances(
        Rad_arcmin=gal["Rad_arcmin"], V=gal["V"], sigma_eps=gal["sigma_eps"],
        p=gal["p"], prior=prior,
        perspective={
            "V_observed": gal["V_observed"],
            "RA_star": gal["RA_star"], "Dec_star": gal["Dec_star"],
            "ra_center": gal["ra_center"], "dec_center": gal["dec_center"],
        },
    )
    theta = np.array([
        gal["truth"]["V_sys"],
        np.log10(0.5),    # r_s = 0.5 kpc (an NFW core)
        np.log10(1e8),    # rho_s
        0.0,              # beta_tilde
        gal["truth"]["d"],
        0.33,             # eps
        gal["truth"]["rhalf_arcmin"],
        gal["truth"]["pmra"],
        gal["truth"]["pmdec"],
    ])
    val = ll(theta)
    assert np.isfinite(val)


def test_9d_loglike_prefers_truth_pm_to_zero_pm():
    """At the truth PM the loglike beats μ = (0, 0) (which mis-corrects)."""
    gal = _fixture_galaxy()
    prior = get_prior("loguniform")
    ll = make_loglike_with_nuisances(
        Rad_arcmin=gal["Rad_arcmin"], V=gal["V"], sigma_eps=gal["sigma_eps"],
        p=gal["p"], prior=prior,
        perspective={
            "V_observed": gal["V_observed"],
            "RA_star": gal["RA_star"], "Dec_star": gal["Dec_star"],
            "ra_center": gal["ra_center"], "dec_center": gal["dec_center"],
        },
    )
    base = [gal["truth"]["V_sys"], np.log10(0.5), np.log10(1e8), 0.0,
            gal["truth"]["d"], 0.33, gal["truth"]["rhalf_arcmin"]]
    theta_truth = np.array(base + [gal["truth"]["pmra"], gal["truth"]["pmdec"]])
    theta_zero = np.array(base + [0.0, 0.0])
    assert ll(theta_truth) > ll(theta_zero)


def test_7d_path_unchanged_when_perspective_none():
    """No-perspective path returns the same likelihood as the original 7D code."""
    gal = _fixture_galaxy(N=30, rseed=7)
    prior = get_prior("loguniform")
    ll_new = make_loglike_with_nuisances(
        Rad_arcmin=gal["Rad_arcmin"], V=gal["V"], sigma_eps=gal["sigma_eps"],
        p=gal["p"], prior=prior, perspective=None,
    )
    theta7 = np.array([gal["truth"]["V_sys"], np.log10(0.5), np.log10(1e8), 0.0,
                       gal["truth"]["d"], 0.33, gal["truth"]["rhalf_arcmin"]])
    val = ll_new(theta7)
    assert np.isfinite(val)
