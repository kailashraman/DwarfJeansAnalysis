"""Unit tests for dwarfjeans.ingest.multi_epoch primitives."""

import numpy as np
import pytest

from dwarfjeans.ingest.multi_epoch import (
    combine_star,
    ivw_combine,
    variability_chi2,
)


def test_ivw_textbook_3epoch():
    v = np.array([100.0, 102.0, 101.0])
    sigma = np.array([1.0, 2.0, 1.5])
    v_bar, sigma_vbar = ivw_combine(v, sigma, sigma_sys=0.0)
    w = 1.0 / sigma ** 2
    expected_vbar = (w * v).sum() / w.sum()
    expected_sigma = (1.0 / w.sum()) ** 0.5
    assert v_bar == pytest.approx(expected_vbar, rel=1e-12)
    assert sigma_vbar == pytest.approx(expected_sigma, rel=1e-12)


def test_ivw_sigma_sys_added_in_quadrature_after():
    v = np.array([100.0, 102.0])
    sigma = np.array([1.0, 1.0])
    _, sigma_vbar = ivw_combine(v, sigma, sigma_sys=1.1)
    # Without floor: σ_post = 1/√2 ≈ 0.7071
    # With floor:    σ_post = √(0.5 + 1.21) ≈ 1.3076
    expected = np.sqrt(0.5 + 1.21)
    assert sigma_vbar == pytest.approx(expected, rel=1e-12)


def test_ivw_single_epoch_passthrough():
    v_bar, sigma_vbar = ivw_combine(np.array([100.0]), np.array([2.0]),
                                      sigma_sys=1.1)
    # Single-epoch IVW: v̄ = v, σ_v̄ = √(σ² + σ_sys²)
    assert v_bar == pytest.approx(100.0)
    assert sigma_vbar == pytest.approx(np.sqrt(4.0 + 1.21))


def test_ivw_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        ivw_combine(np.array([100.0, 101.0]), np.array([1.0, 0.0]))


def test_variability_constant_v_no_flag():
    v = np.array([100.0, 100.0, 100.0])
    sigma = np.array([1.0, 1.0, 1.0])
    chi2_stat, dof, p, flag = variability_chi2(v, sigma, v_bar=100.0)
    assert chi2_stat == 0.0
    assert dof == 2
    assert p == pytest.approx(1.0)
    assert flag is False


def test_variability_5sigma_outlier_flagged():
    v = np.array([100.0, 100.0, 105.0])  # last is 5σ out of σ=1
    sigma = np.array([1.0, 1.0, 1.0])
    v_bar, _ = ivw_combine(v, sigma)
    chi2_stat, dof, p, flag = variability_chi2(v, sigma, v_bar,
                                                 p_threshold=0.01)
    assert dof == 2
    assert p < 0.01
    assert flag is True


def test_variability_single_epoch_unflagged():
    chi2_stat, dof, p, flag = variability_chi2(
        np.array([100.0]), np.array([1.0]), v_bar=100.0
    )
    assert dof == 0
    assert np.isnan(chi2_stat)
    assert np.isnan(p)
    assert flag is False


def test_combine_star_returns_full_dict():
    v = np.array([100.0, 102.0, 101.0])
    sigma = np.array([1.0, 1.0, 1.0])
    out = combine_star(v, sigma, sigma_sys=0.5, p_threshold=0.01)
    assert set(out.keys()) == {
        "v_bar", "sigma_vbar", "n_epoch", "chi2", "dof", "p_value", "var_flag"
    }
    assert out["n_epoch"] == 3
    # Same input — IVW must agree with the standalone primitive
    v_bar_check, sigma_check = ivw_combine(v, sigma, sigma_sys=0.5)
    assert out["v_bar"] == pytest.approx(v_bar_check)
    assert out["sigma_vbar"] == pytest.approx(sigma_check)


# --- Strict-deconvolution path -----------------------------------------------

from dwarfjeans.ingest.multi_epoch import (  # noqa: E402
    combine_star_strict,
    ivw_combine_strict,
)


def test_ivw_strict_recovers_textbook_when_sys_is_zero():
    """sigma_sys=0 → strict path equals the as-statistical path exactly."""
    v = np.array([100.0, 102.0, 101.0])
    sigma = np.array([1.0, 2.0, 1.5])
    v_strict, s_strict = ivw_combine_strict(v, sigma, sigma_sys=0.0)
    v_stat, s_stat = ivw_combine(v, sigma, sigma_sys=0.0)
    assert v_strict == pytest.approx(v_stat, rel=1e-12)
    assert s_strict == pytest.approx(s_stat, rel=1e-12)


def test_ivw_strict_inflates_sigma_vbar_vs_as_statistical():
    """At fixed published total error, the strict path returns a LARGER
    sigma_vbar than the as-statistical path because the IVW averages
    only sigma_stat, not sigma_total."""
    sigma_total = np.array([1.0, 1.0, 1.0])
    sigma_sys = 0.7  # sigma_stat² = 1 - 0.49 = 0.51
    v = np.array([100.0, 100.0, 100.0])
    _, s_strict = ivw_combine_strict(v, sigma_total, sigma_sys=sigma_sys)
    # As-statistical path with the same total error and no extra floor:
    _, s_stat = ivw_combine(v, sigma_total, sigma_sys=0.0)
    assert s_strict > s_stat
    # Analytic expectation: sigma_stat=√0.51, IVW gives σ_stat/√3, then
    # add σ_sys in quadrature:  √(0.51/3 + 0.49) = √(0.17 + 0.49) = √0.66
    assert s_strict == pytest.approx(np.sqrt(0.51 / 3 + 0.49), rel=1e-9)


def test_ivw_strict_raises_when_total_below_sys():
    """Implied σ_stat would be imaginary — that's a paper-vs-policy
    inconsistency, not something to silently mask."""
    v = np.array([100.0, 100.0])
    sigma_total = np.array([1.0, 0.5])
    with pytest.raises(ValueError, match="imaginary"):
        ivw_combine_strict(v, sigma_total, sigma_sys=0.7)


def test_ivw_strict_single_epoch_returns_sigma_total():
    """N=1 sanity: σ_stat² = σ_total² − σ_sys²; 1/W = σ_stat²;
    σ_vbar = √(σ_stat² + σ_sys²) = σ_total."""
    v_bar, s = ivw_combine_strict(np.array([100.0]), np.array([1.5]), sigma_sys=0.7)
    assert v_bar == 100.0
    assert s == pytest.approx(1.5, rel=1e-12)


def test_combine_star_strict_chi2_uses_sigma_stat_not_sigma_total():
    """The strict χ² test must run on σ_stat (deconvolved), else it
    under-flags variables. Compare to a hand-computed expectation."""
    v = np.array([100.0, 105.0])  # 5 km/s scatter
    sigma_total = np.array([1.0, 1.0])
    sigma_sys = 0.7  # sigma_stat ≈ 0.714
    out = combine_star_strict(v, sigma_total, sigma_sys=sigma_sys, p_threshold=0.01)
    # Strict χ² should be larger than the σ_total-based χ² for the same data
    out_naive = combine_star(v, sigma_total, sigma_sys=0.0, p_threshold=0.01)
    assert out["chi2"] > out_naive["chi2"]
    assert out["var_flag"] is True
