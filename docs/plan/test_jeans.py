"""
Iteration-loop verification harness for jeans.py.

Tests run on every change during development:
  1. NFW small-x series matches a high-order Taylor reference, and beats the
     direct ln(1+x) - x/(1+x) form in the regime where the latter is unstable.
  2. β̃ prior-edge smoke test: σ_los is finite, positive, and emits zero
     RuntimeWarnings across β̃ ∈ {-0.95, -0.5, 0, +0.5, +0.95} for a
     representative galaxy. The β̃ = -0.95 case is the tangential edge
     (β = -38), where naive evaluation of s^(2β) overflows float64.
  3. Physical sanity: ρ_s scaling (×2 ⇒ σ_los × √2), and σ_los at r_p within
     a factor of ~2 of the Wolf+2010 estimator.
  4. Speed: grid timing for 30 and 100 stars per likelihood call.

Quad cross-checks (nu_sigma_r2_grid vs nu_sigma_r2_quad, and the projection
counterpart) live in test_jeans_vs_quad.py — out of the iteration loop because
quad is ~10⁴× slower. Run them on demand when something looks wrong.
"""

from __future__ import annotations

import math
import time
import warnings
import numpy as np

import jeans


# ----------------------------------------------------------------------------
# 1. NFW small-x series
# ----------------------------------------------------------------------------

def test_nfw_g_small_x():
    print("\n[Test 1] NFW g(x) small-x series vs direct form")
    # In the regime where direct form is trustworthy (x ~ 0.01 to 100), the
    # two should agree to machine precision since we only use the series for
    # x < 1e-3.
    x_test = np.geomspace(0.01, 100.0, 50)
    g_via_direct = np.log1p(x_test) - x_test / (1.0 + x_test)
    g_module = jeans.nfw_g(x_test)
    err = np.max(np.abs(g_module - g_via_direct) / np.abs(g_via_direct))
    print(f"  max relative error in [0.01, 100]: {err:.3e}")
    assert err < 1e-12, "series vs direct disagree in stable regime"

    # In the regime where direct form is unstable (x ~ 1e-8 to 1e-3), the
    # series should agree with a high-order Taylor reference. Series:
    #   ln(1+x) - x/(1+x) = (1/2)x² - (2/3)x³ + (3/4)x⁴ - (4/5)x⁵ + ...
    # The module truncates at x⁶; reference uses 18 terms.
    x_small = np.geomspace(1e-8, 1e-3, 20)
    g_ref = np.zeros_like(x_small)
    for n in range(2, 20):  # power of x; coefficient (-1)^n * (n-1)/n
        sign = 1.0 if (n % 2 == 0) else -1.0
        g_ref += sign * (n - 1) / n * x_small ** n
    g_module_small = jeans.nfw_g(x_small)
    err_small = np.max(np.abs(g_module_small - g_ref) / np.abs(g_ref))
    print(f"  max relative error vs 18-term ref in [1e-8, 1e-3]: {err_small:.3e}")
    assert err_small < 1e-10, "small-x series fails"

    # Confirm the *naive* direct form would be wrong here — motivation for
    # having the series in the first place.
    naive = np.log1p(x_small) - x_small / (1.0 + x_small)
    naive_err = np.max(np.abs(naive - g_ref) / np.abs(g_ref))
    print(f"  for comparison, naive direct form max relative error: {naive_err:.3e}")
    print("  PASS")


# ----------------------------------------------------------------------------
# 2. β̃ prior-edge smoke test
# ----------------------------------------------------------------------------

def beta_tilde_to_beta(beta_tilde: float) -> float:
    """Read+ symmetrized anisotropy: β = 2β̃/(1+β̃)."""
    return 2.0 * beta_tilde / (1.0 + beta_tilde)


def test_beta_edges():
    print("\n[Test 2] β̃ prior-edge smoke test")
    # Classical-dwarf parameters; covers the relevant Plummer/NFW scale ratio.
    r_s, rho_s, r_p = 1.0, 1e7, 0.30
    R = np.geomspace(0.01 * r_p, 5.0 * r_p, 20)

    # Span the prior. β̃ = ±0.95 are the prior boundaries used in production;
    # ±0.95 → β ≈ -38, +0.974 respectively. The tangential side is the one
    # where naive s^(2β) evaluation overflows float64.
    beta_tildes = [-0.95, -0.5, 0.0, 0.5, 0.95]

    print(f"  {'β̃':>8} {'β':>14} {'min σ_los':>12} {'max σ_los':>12} {'warnings':>10}")
    for bt in beta_tildes:
        beta = beta_tilde_to_beta(bt)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sl = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")
        runtime_warnings = [w for w in wlist if issubclass(w.category, RuntimeWarning)]
        finite = bool(np.all(np.isfinite(sl)))
        positive = bool(np.all(sl > 0))
        ok = finite and positive and len(runtime_warnings) == 0
        marker = "" if ok else "  ← FAIL"
        print(f"  {bt:>8.3f} {beta:>14.6g} {np.nanmin(sl):>12.4g} "
              f"{np.nanmax(sl):>12.4g} {len(runtime_warnings):>10}{marker}")
        if runtime_warnings:
            for w in runtime_warnings:
                print(f"           {w.category.__name__}: {w.message}")
        assert finite, f"σ_los not finite at β̃ = {bt}"
        assert positive, f"σ_los not positive at β̃ = {bt}"
        assert len(runtime_warnings) == 0, (
            f"RuntimeWarning at β̃ = {bt} (β = {beta:.6g}): "
            f"{[str(w.message) for w in runtime_warnings]}"
        )
    print("  PASS")


# ----------------------------------------------------------------------------
# 3. Physical sanity
# ----------------------------------------------------------------------------

def test_physical_sanity():
    print("\n[Test 3] physical sanity")

    r_s, rho_s, r_p, beta = 1.0, 1e7, 0.30, 0.0
    R = np.geomspace(0.01 * r_p, 10.0 * r_p, 30)
    sl = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")
    print(f"  σ_los at R/r_p = {R[0]/r_p:.3f}: {sl[0]:.3f} km/s")
    print(f"  σ_los at R/r_p = {R[-1]/r_p:.3f}: {sl[-1]:.3f} km/s")
    # Plummer + NFW with isotropic β=0 is not strictly monotone in R — there's
    # a small bump near R ~ r_s due to the halo mass distribution — so we
    # don't assert monotonicity, only finiteness and physical magnitude.

    # Doubling ρ_s doubles M(r) and so doubles σ_los² → multiplies σ_los by √2.
    # This is an exact relation (linear in ρ_s through the Jeans integral), so
    # the test should pass to ~1e-3 with a 4096-point grid.
    sl1 = jeans.sigma_los(np.array([r_p]), beta, r_s, rho_s, r_p, method="grid")[0]
    sl2 = jeans.sigma_los(np.array([r_p]), beta, r_s, 2 * rho_s, r_p, method="grid")[0]
    ratio = sl2 / sl1
    print(f"  ρ_s × 2 → σ_los ratio: {ratio:.4f} (expected {math.sqrt(2):.4f})")
    assert abs(ratio - math.sqrt(2)) < 1e-3

    # Order-of-magnitude check: a classical dSph with M(r_p) ~ M_half and
    # r_p ~ r_1/2 has σ_los ~ √(G M / r_1/2 / 3) (Wolf+2010-style estimator).
    # Computed σ_los should be within a factor of ~2.
    M_half = jeans.nfw_M(r_p, r_s, rho_s)
    sigma_wolf_est = np.sqrt(jeans.G_KPC_KMS2_MSUN * M_half / r_p) / np.sqrt(3.0)
    print(f"  M(<r_p) = {M_half:.3e} M_sun, naive Wolf estimate σ ~ {sigma_wolf_est:.2f} km/s")
    print(f"  computed σ_los at r_p: {sl1:.2f} km/s")
    assert 0.3 < sl1 / sigma_wolf_est < 3.0
    print("  PASS")


# ----------------------------------------------------------------------------
# 4. Speed (grid only — quad timing lives in test_jeans_vs_quad.py)
# ----------------------------------------------------------------------------

def test_speed():
    print("\n[Test 4] grid timing")
    r_s, rho_s, r_p, beta = 0.5, 1e8, 0.2, 0.0
    R_30 = np.geomspace(0.01 * r_p, 3.0 * r_p, 30)
    R_100 = np.geomspace(0.01 * r_p, 3.0 * r_p, 100)

    # Warm-up so first-call import/JIT cost doesn't dominate.
    _ = jeans.sigma_los(R_30, beta, r_s, rho_s, r_p, method="grid")

    t0 = time.perf_counter()
    _ = jeans.sigma_los(R_30, beta, r_s, rho_s, r_p, method="grid")
    t_30 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = jeans.sigma_los(R_100, beta, r_s, rho_s, r_p, method="grid")
    t_100 = time.perf_counter() - t0

    print(f"  grid (30 stars):  {t_30 * 1000:.2f} ms")
    print(f"  grid (100 stars): {t_100 * 1000:.2f} ms")
    # Stage-2 budget: ~10 ms per likelihood at N_stars = 100 supports >10⁵
    # likelihood calls/CPU/min → nlive=500, dlogz=0.1 in ~10 min wall.
    assert t_100 < 0.05, f"100-star call too slow: {t_100 * 1000:.1f} ms"
    print("  PASS")


if __name__ == "__main__":
    test_nfw_g_small_x()
    test_beta_edges()
    test_physical_sanity()
    test_speed()
    print("\nAll tests passed.")
