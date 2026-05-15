"""Unit tests for dwarfjeans.jeans.priors — the three-entry registry."""

import numpy as np
import pytest

from dwarfjeans.jeans.priors import (
    BETA_TILDE_BOUNDS,
    LOG10_RHOS_BOUNDS,
    LOG10_RS_BOUNDS,
    PRIOR_REGISTRY,
    V_HALFWIDTH,
    get_prior,
    jeffreys_log_term,
    make_loguniform_prior_transform,
    make_uniform_prior_transform,
)


def test_registry_keys():
    assert set(PRIOR_REGISTRY.keys()) == {"uniform", "loguniform", "jeffreys", "satgen"}


def test_get_prior_unknown():
    with pytest.raises(KeyError):
        get_prior("not_a_real_prior")


def test_loguniform_matches_historical_formula():
    """The relocated builder must reproduce the historical prior_transform
    byte-for-byte (the math is unchanged)."""
    rng = np.random.default_rng(42)
    U = rng.random((1000, 4))
    pt = make_loguniform_prior_transform(V_center=200.0)
    out = np.array([pt(u) for u in U])

    V_lo = 200.0 - V_HALFWIDTH
    V_hi = 200.0 + V_HALFWIDTH
    expected = np.empty_like(U)
    expected[:, 0] = V_lo + (V_hi - V_lo) * U[:, 0]
    expected[:, 1] = LOG10_RS_BOUNDS[0] + (LOG10_RS_BOUNDS[1] - LOG10_RS_BOUNDS[0]) * U[:, 1]
    expected[:, 2] = LOG10_RHOS_BOUNDS[0] + (LOG10_RHOS_BOUNDS[1] - LOG10_RHOS_BOUNDS[0]) * U[:, 2]
    expected[:, 3] = BETA_TILDE_BOUNDS[0] + (BETA_TILDE_BOUNDS[1] - BETA_TILDE_BOUNDS[0]) * U[:, 3]
    assert np.array_equal(out, expected)


def test_loguniform_rs_min_override():
    pt = make_loguniform_prior_transform(V_center=0.0, log10_rs_min=0.0)
    # u[1]=0 should map to log10_rs = 0.0 (i.e. r_s = 1 kpc), not -2.0.
    out = pt(np.array([0.5, 0.0, 0.5, 0.5]))
    assert out[1] == 0.0


def test_uniform_linear_in_rs_rhos():
    """Linear-uniform must produce r_s and rho_s uniform in linear space."""
    rng = np.random.default_rng(0)
    U = rng.random((10000, 4))
    pt = make_uniform_prior_transform(V_center=0.0)
    out = np.array([pt(u) for u in U])
    r_s = 10.0 ** out[:, 1]
    rho_s = 10.0 ** out[:, 2]

    rs_lo, rs_hi = 10.0 ** LOG10_RS_BOUNDS[0], 10.0 ** LOG10_RS_BOUNDS[1]
    rhos_lo, rhos_hi = 10.0 ** LOG10_RHOS_BOUNDS[0], 10.0 ** LOG10_RHOS_BOUNDS[1]
    assert r_s.min() >= rs_lo - 1e-9
    assert r_s.max() <= rs_hi + 1e-9
    assert rho_s.min() >= rhos_lo - 1e-3
    assert rho_s.max() <= rhos_hi + 1e-3

    expected_rs = rs_lo + (rs_hi - rs_lo) * U[:, 1]
    expected_rhos = rhos_lo + (rhos_hi - rhos_lo) * U[:, 2]
    assert np.allclose(r_s, expected_rs, rtol=1e-12, atol=1e-15)
    assert np.allclose(rho_s, expected_rhos, rtol=1e-12, atol=1e-15)


def test_jeffreys_constant_T_returns_neg_inf():
    """Degenerate parameter point: identical T across stars → D=0 → -inf."""
    sigma_los_sq = np.array([10.0, 10.0, 10.0])
    T = np.array([2.0, 2.0, 2.0])  # constant
    sigma_eps_sq = np.array([1.0, 1.0, 1.0])
    p = np.array([1.0, 1.0, 1.0])
    val = jeffreys_log_term(sigma_los_sq, T, sigma_eps_sq, p)
    assert val == -np.inf


def test_jeffreys_finite_when_T_varies():
    sigma_los_sq = np.array([10.0, 10.0, 10.0])
    T = np.array([2.0, 3.0, 4.0])
    sigma_eps_sq = np.array([1.0, 1.0, 1.0])
    p = np.array([1.0, 1.0, 1.0])
    val = jeffreys_log_term(sigma_los_sq, T, sigma_eps_sq, p)
    assert np.isfinite(val)


def test_prior_objects_have_consistent_needs_T():
    assert get_prior("uniform").needs_T is False
    assert get_prior("loguniform").needs_T is False
    assert get_prior("jeffreys").needs_T is True


def test_zero_correction_is_zero():
    """uniform / loguniform must have zero log-correction regardless of args."""
    sigma_los_sq = np.array([5.0, 7.0])
    T = np.array([1.0, 2.0])
    sigma_eps_sq = np.array([0.5, 0.5])
    p = np.array([1.0, 0.8])
    for name in ("uniform", "loguniform"):
        prior = get_prior(name)
        assert prior.log_correction(sigma_los_sq, T, sigma_eps_sq, p) == 0.0
