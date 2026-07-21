"""Tests for the exploratory velocity-gradient diagnostic (scripts/velocity_gradients.py).

Covers the fit's three contracts: (1) it recovers a known injected linear
gradient and its position angle; (2) it correctly flags the sigma_int-railed
regime where the conditional |g| Wald error is unreliable (the willman_1 case
from review) while leaving the significance honest; (3) a pure-dispersion field
with no gradient reads back low significance with an interior (unrailed) sigma_int.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "velocity_gradients", _SCRIPTS / "velocity_gradients.py"
)
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)


def _field(rng, n=200, half_kpc=0.3):
    """Uniform-ish projected star field, coords in kpc."""
    xi = rng.uniform(-half_kpc, half_kpc, n)
    eta = rng.uniform(-half_kpc, half_kpc, n)
    return xi, eta


def test_recovers_injected_gradient_and_pa():
    rng = np.random.default_rng(0)
    xi, eta = _field(rng, n=400)
    g_xi, g_eta = 8.0, 0.0            # pure East gradient -> PA = 90 deg (East of N)
    verr = np.full(xi.size, 0.5)
    v = 100.0 + g_xi * xi + g_eta * eta + rng.normal(0.0, verr)

    out = vg.fit_gradient(xi, eta, v, verr)

    assert abs(out["g_mag"] - 8.0) < 1.5
    assert abs(out["pa_deg"] - 90.0) < 10.0
    assert out["sigma"] > 5.0        # a clear gradient is highly significant
    # No intrinsic scatter beyond verr -> sigma_int rails to the lower bound.
    assert out["sigma_int_railed"] is True


def test_no_gradient_is_insignificant_and_unrailed():
    rng = np.random.default_rng(1)
    xi, eta = _field(rng, n=400)
    sigma_int = 3.0                  # genuine dispersion, spatially flat
    verr = rng.uniform(0.5, 1.5, xi.size)
    v = 100.0 + rng.normal(0.0, np.hypot(sigma_int, verr))

    out = vg.fit_gradient(xi, eta, v, verr)

    assert out["sigma"] < 3.0        # no injected gradient
    assert out["sigma_int_railed"] is False   # dispersion is interior, not on a bound
    assert abs(out["sigma_int"] - sigma_int) < 1.5


def test_sky_offsets_match_projected_radius_convention():
    """sky_offsets_kpc must reproduce staging.projected_radius_kpc."""
    from dwarfjeans.ingest.staging import projected_radius_kpc

    ra0, dec0, dist = 260.0, 57.9, 76.0
    ra = np.array([260.05, 259.9, 260.0])
    dec = np.array([57.95, 57.85, 57.9])
    xi, eta = vg.sky_offsets_kpc(ra, dec, ra0, dec0, dist)
    r_ref = projected_radius_kpc(ra, dec, ra0, dec0, dist)
    np.testing.assert_allclose(np.hypot(xi, eta), r_ref, rtol=1e-10)
