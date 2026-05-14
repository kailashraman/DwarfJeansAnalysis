"""Tests for the optional perspective-motion injection in
``dwarfjeans.mocks.galaxy.make_mock_galaxy``."""

import math

import numpy as np
import pytest

from dwarfjeans.jeans.perspective import A_KMS_PER_MASYR_KPC
from dwarfjeans.mocks.galaxy import make_mock_galaxy


_BASE = dict(n_stars=200, r_s=0.5, rho_s=1e8, r_p=0.05,
             beta=0.0, V_sys=0.0, sigma_eps=2.0)
_PERSP = dict(d_kpc=124.0, ra_center_deg=143.8, dec_center_deg=-36.7,
              pmra_true_mas_yr=-0.10, pmdec_true_mas_yr=0.10)


def test_no_perspective_kwargs_is_default():
    """No perspective kwargs → no RA/Dec/V_observed in output."""
    g = make_mock_galaxy(**_BASE, rng=np.random.default_rng(0))
    assert "RA_star" not in g
    assert "Dec_star" not in g
    assert "V_observed" not in g
    assert "dv_persp_true" not in g


def test_partial_perspective_kwargs_raises():
    with pytest.raises(ValueError, match="perspective injection requires all"):
        make_mock_galaxy(**_BASE, rng=np.random.default_rng(0),
                         d_kpc=124.0, ra_center_deg=143.8)


def test_perspective_injects_dv_with_correct_rms():
    """Δv RMS ≈ A·d·|μ|·⟨ρ⟩/√2 with ⟨ρ⟩ ~ r_p/d (small-angle Plummer)."""
    g = make_mock_galaxy(**_BASE, **_PERSP, rng=np.random.default_rng(1))
    # Both arrays must be present and same length.
    assert g["RA_star"].shape == (200,)
    assert g["Dec_star"].shape == (200,)
    assert g["dv_persp_true"].shape == (200,)
    # V_observed = V + dv_persp_true.
    np.testing.assert_allclose(g["V_observed"], g["V"] + g["dv_persp_true"], rtol=1e-12)
    # RMS should be comparable to A·d·|μ|·⟨ρ⟩ with ⟨ρ⟩ ~ r_p/d.
    pm_mag = math.hypot(_PERSP["pmra_true_mas_yr"], _PERSP["pmdec_true_mas_yr"])
    rho_typical = _BASE["r_p"] / _PERSP["d_kpc"]  # radians
    scale = A_KMS_PER_MASYR_KPC * _PERSP["d_kpc"] * pm_mag * rho_typical
    rms = float(np.sqrt(np.mean(g["dv_persp_true"] ** 2)))
    # Loose envelope: same order of magnitude, factor of 3 either way.
    assert scale / 3 < rms < 3 * scale


def test_zero_pm_yields_zero_dv():
    """μ = (0, 0) → all stars get zero perspective shift."""
    g = make_mock_galaxy(**_BASE, rng=np.random.default_rng(2),
                        d_kpc=124.0, ra_center_deg=143.8, dec_center_deg=-36.7,
                        pmra_true_mas_yr=0.0, pmdec_true_mas_yr=0.0)
    np.testing.assert_array_equal(g["dv_persp_true"], np.zeros(_BASE["n_stars"]))
    np.testing.assert_array_equal(g["V_observed"], g["V"])


def test_ra_dec_match_projected_R():
    """Each star's tangent-plane separation matches its R / d."""
    g = make_mock_galaxy(**_BASE, **_PERSP, rng=np.random.default_rng(3))
    cos_d0 = math.cos(math.radians(_PERSP["dec_center_deg"]))
    dRA = np.deg2rad(g["RA_star"] - _PERSP["ra_center_deg"]) * cos_d0
    dDec = np.deg2rad(g["Dec_star"] - _PERSP["dec_center_deg"])
    rho = np.sqrt(dRA ** 2 + dDec ** 2)
    R_from_rho = rho * _PERSP["d_kpc"]
    np.testing.assert_allclose(R_from_rho, g["R"], rtol=1e-10)


def test_truth_dict_carries_pm():
    g = make_mock_galaxy(**_BASE, **_PERSP, rng=np.random.default_rng(4))
    assert g["truth"]["pmra"] == _PERSP["pmra_true_mas_yr"]
    assert g["truth"]["pmdec"] == _PERSP["pmdec_true_mas_yr"]
    assert g["truth"]["d_kpc"] == _PERSP["d_kpc"]
