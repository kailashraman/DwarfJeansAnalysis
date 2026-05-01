"""
Quad cross-checks for jeans.py — out of the iteration loop.

These are the slow, trusted reference tests: nu_sigma_r2_grid vs
nu_sigma_r2_quad, and Sigma_sigma_los2_grid vs Sigma_sigma_los2_quad,
across a parameter sweep covering the MW-dwarf regime (UFD r_p ~ 0.03 kpc,
classical r_p ~ 0.3 kpc) and a range of β.

Quad evaluation is ~10⁴× slower than the production grid implementation —
on the order of a minute per parameter combination — so this script is
NOT run in the iteration loop. Run it on demand when:

  * touching anything in the inner-Jeans (`nu_sigma_r2_*`) or projection
    (`Sigma_sigma_los2_*`) code paths;
  * a specific galaxy's Stage-2 results look wrong and you want to bisect
    between "the model is bad" vs "the integration is bad";
  * before merging anything that changes integration tolerances, grid
    sizes, or the log-space scheme.

Usage: `python test_jeans_vs_quad.py`. Expect ~80 seconds end-to-end.
"""

from __future__ import annotations

import numpy as np

import jeans


# Cover ultra-faints (r_p ~ 0.03 kpc, r_s ~ 0.1 kpc), classicals (r_p ~ 0.3,
# r_s ~ 1), and a range of β (radial, isotropic, tangential).
PARAM_GRID = [
    # (label, r_s [kpc], rho_s [Msun/kpc^3], r_p [kpc], beta)
    ("UFD-isotropic",   0.10, 1e9,  0.03,  0.0),
    ("UFD-radial",      0.10, 1e9,  0.03,  0.5),
    ("UFD-tangential",  0.10, 1e9,  0.03, -1.0),
    ("classical-iso",   1.0,  1e7,  0.30,  0.0),
    ("classical-rad",   1.0,  1e7,  0.30,  0.3),
    ("classical-tan",   1.0,  1e7,  0.30, -2.0),
    ("small-rs",        0.05, 1e10, 0.05,  0.0),
    ("large-rs",        5.0,  1e6,  0.50,  0.0),
]


def test_jeans_inner():
    print("\n[Quad-check 1] nu_sigma_r2_grid vs nu_sigma_r2_quad")
    print(f"  {'label':<20} {'max rel err':>12} {'r at max err':>14}")
    worst_err_overall = 0.0
    for label, r_s, rho_s, r_p, beta in PARAM_GRID:
        # Test at radii spanning the relevant range — well below r_p, around
        # r_p, around r_s, well above.
        r_test = np.geomspace(0.01 * min(r_s, r_p), 10.0 * max(r_s, r_p), 30)
        ref = jeans.nu_sigma_r2_quad(r_test, beta, r_s, rho_s, r_p)
        fast = jeans.nu_sigma_r2_grid(r_test, beta, r_s, rho_s, r_p, n_grid=4096)
        rel_err = np.abs(fast - ref) / np.abs(ref)
        i_worst = np.argmax(rel_err)
        max_err = rel_err[i_worst]
        worst_err_overall = max(worst_err_overall, max_err)
        print(f"  {label:<20} {max_err:>12.3e} {r_test[i_worst]:>14.3e}")
    print(f"  worst rel err overall: {worst_err_overall:.3e}")
    # Trapezoidal on log-grid with 4096 points should reach ~1e-4 relative.
    assert worst_err_overall < 1e-3, "inner Jeans grid disagrees with quad"
    print("  PASS")


def test_jeans_projection():
    print("\n[Quad-check 2] Sigma_sigma_los2_grid vs Sigma_sigma_los2_quad")
    print(f"  {'label':<20} {'max rel err':>12} {'R at max err':>14}")
    worst_err_overall = 0.0
    for label, r_s, rho_s, r_p, beta in PARAM_GRID:
        # Only 5 R-values per case — quad is ~2 sec per R, so this stays
        # under ~10 sec per case, ~80 sec total for the sweep.
        R_test = np.geomspace(0.01 * r_p, 5.0 * r_p, 5)
        ref = jeans.Sigma_sigma_los2_quad(R_test, beta, r_s, rho_s, r_p)
        fast = jeans.Sigma_sigma_los2_grid(R_test, beta, r_s, rho_s, r_p,
                                           n_inner=4096, n_outer=2048)
        rel_err = np.abs(fast - ref) / np.abs(ref)
        i_worst = np.argmax(rel_err)
        max_err = rel_err[i_worst]
        worst_err_overall = max(worst_err_overall, max_err)
        print(f"  {label:<20} {max_err:>12.3e} {R_test[i_worst]:>14.3e}")
    print(f"  worst rel err overall: {worst_err_overall:.3e}")
    assert worst_err_overall < 1e-2, "projection grid disagrees with quad"
    print("  PASS")


if __name__ == "__main__":
    test_jeans_inner()
    test_jeans_projection()
    print("\nAll quad cross-checks passed.")
