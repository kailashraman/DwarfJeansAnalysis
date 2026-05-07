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
