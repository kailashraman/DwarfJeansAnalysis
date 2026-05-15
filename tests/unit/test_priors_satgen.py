"""Unit tests for the SatGen-conditioned (r_s, rho_s) prior."""

import numpy as np
import pytest

from dwarfjeans.jeans.priors import (
    BETA_TILDE_BOUNDS,
    LOG10_RHOS_BOUNDS,
    LOG10_RS_BOUNDS,
    SATGEN_PRIOR_TABLE,
    V_HALFWIDTH,
    _load_satgen_table,
    get_prior,
    make_satgen_prior_transform,
    make_satgen_prior_transform_with_nuisances,
)


pytestmark = pytest.mark.skipif(
    not SATGEN_PRIOR_TABLE.exists(),
    reason="SatGen prior table not built; run scripts/build_satgen_prior_table.py",
)


def test_registry_includes_satgen():
    p = get_prior("satgen")
    assert p.name == "satgen"
    assert p.needs_T is False


def test_loader_shapes_and_monotone_cdf():
    grid, cdf, centers, mu, sigma = _load_satgen_table(str(SATGEN_PRIOR_TABLE))
    assert grid.ndim == cdf.ndim == centers.ndim == mu.ndim == sigma.ndim == 1
    assert grid.size == cdf.size
    assert centers.size == mu.size == sigma.size
    assert np.all(np.isfinite(cdf))
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] == pytest.approx(0.0, abs=1e-12)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


def test_transform_midpoint_in_bounds():
    pt = make_satgen_prior_transform(V_center=200.0)
    x = pt(np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.isfinite(x).all()
    assert 200.0 - V_HALFWIDTH <= x[0] <= 200.0 + V_HALFWIDTH
    assert LOG10_RS_BOUNDS[0] <= x[1] <= LOG10_RS_BOUNDS[1]
    assert LOG10_RHOS_BOUNDS[0] <= x[2] <= LOG10_RHOS_BOUNDS[1]
    assert BETA_TILDE_BOUNDS[0] <= x[3] <= BETA_TILDE_BOUNDS[1]


def test_transform_marginal_matches_catalog():
    """Sampling the transform must reproduce SatGen's log10 r_s marginal."""
    from scipy.stats import ks_2samp
    grid, _, _, _, _ = _load_satgen_table(str(SATGEN_PRIOR_TABLE))
    pt = make_satgen_prior_transform(V_center=0.0)
    rng = np.random.default_rng(0)
    U = rng.random((20_000, 4))
    out = np.array([pt(u) for u in U])
    # Resample log10_rs from the tabulated marginal CDF directly.
    _, cdf, _, _, _ = _load_satgen_table(str(SATGEN_PRIOR_TABLE))
    u_ref = rng.random(20_000)
    ref = np.interp(u_ref, cdf, grid)
    ks = ks_2samp(out[:, 1], ref)
    assert ks.statistic < 0.02


def test_conditional_rhos_gaussian_at_fixed_rs():
    """Hold u[1] fixed and check that log10 rho_s draws follow N(mu, sigma)."""
    _, _, centers, mu, sigma = _load_satgen_table(str(SATGEN_PRIOR_TABLE))
    pt = make_satgen_prior_transform(V_center=0.0)
    # Pick a unit cube value that maps near the bulk; identify the implied
    # log10_rs and check the conditional draws against the implied mu, sigma.
    rng = np.random.default_rng(1)
    u1_fixed = 0.4
    U = np.column_stack((
        rng.random(20_000),
        np.full(20_000, u1_fixed),
        rng.random(20_000),
        rng.random(20_000),
    ))
    out = np.array([pt(u) for u in U])
    log10_rs_val = out[0, 1]
    mu_expected = float(np.interp(log10_rs_val, centers, mu))
    sigma_expected = float(np.interp(log10_rs_val, centers, sigma))
    sample = out[:, 2]
    assert abs(sample.mean() - mu_expected) < 0.02
    assert abs(sample.std(ddof=1) - sigma_expected) < 0.03
    # Guard against the boundary-pinning regression: no draw should
    # equal LOG10_RHOS_BOUNDS exactly. (Past bug: a hard clip to the
    # registry box pinned tail draws to {4, 14}; cf. review-checklist.)
    assert (sample == LOG10_RHOS_BOUNDS[0]).sum() == 0
    assert (sample == LOG10_RHOS_BOUNDS[1]).sum() == 0


def test_log10_rs_min_override_truncates():
    pt = make_satgen_prior_transform(V_center=0.0, log10_rs_min=-0.5)
    rng = np.random.default_rng(2)
    U = rng.random((5_000, 4))
    out = np.array([pt(u) for u in U])
    assert out[:, 1].min() >= -0.5 - 1e-9
    assert out[:, 1].max() <= LOG10_RS_BOUNDS[1] + 1e-9


def test_log10_rs_min_above_top_raises():
    with pytest.raises(ValueError):
        make_satgen_prior_transform(V_center=0.0, log10_rs_min=LOG10_RS_BOUNDS[1] + 0.5)


def test_with_nuisances_signature():
    pt = make_satgen_prior_transform_with_nuisances(
        V_center=0.0,
        d_mean=30.0, d_sigma=2.0,
        eps_mean=0.3, eps_sigma=0.05,
        rhalf_mean=4.0, rhalf_sigma=0.3,
    )
    x = pt(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    assert x.shape == (7,)
    assert np.isfinite(x).all()
    # Halo block matches the 4D variant at the midpoint.
    pt4 = make_satgen_prior_transform(V_center=0.0)
    x4 = pt4(np.array([0.5, 0.5, 0.5, 0.5]))
    np.testing.assert_allclose(x[:4], x4)
    # Nuisance midpoints.
    assert abs(x[4] - 30.0) < 1e-9
    assert abs(x[5] - 0.3) < 1e-2  # truncnorm shifts very slightly
    assert abs(x[6] - 4.0) < 1e-9


def test_zero_correction_for_satgen():
    p = get_prior("satgen")
    sigma_los_sq = np.array([5.0, 7.0])
    T = np.array([1.0, 2.0])
    sigma_eps_sq = np.array([0.5, 0.5])
    pp = np.array([1.0, 0.8])
    assert p.log_correction(sigma_los_sq, T, sigma_eps_sq, pp) == 0.0
