"""Bench: parity + wall-time of (n_inner, n_outer) = (4096, 1024) vs (2048, 512).

Item 2 of the speedup plan. Read-only on solver — passes grid sizes explicitly
to compare without changing defaults. Exercises Sigma_sigma_los2_grid and
Sigma_sigma_los2_grid_pair on UFD + classical parameter regimes.

Run: python scripts/bench_grid_halving.py
"""
from __future__ import annotations

import time

import numpy as np

from dwarfjeans.jeans import solver as jeans

PARAMS = [
    # (label, r_s [kpc], rho_s [Msun/kpc^3], r_p [kpc], beta)
    ("UFD-iso",      0.10, 1e9,  0.03,  0.0),
    ("UFD-rad",      0.10, 1e9,  0.03,  0.5),
    ("UFD-tan",      0.10, 1e9,  0.03, -1.0),
    ("classical",    1.0,  1e7,  0.30,  0.0),
    ("classical-r",  1.0,  1e7,  0.30,  0.3),
]

# 30 R-values per case, spanning 0.01·r_p .. 5·r_p (typical star-radius range).
def R_for(r_p):
    return np.geomspace(0.01 * r_p, 5.0 * r_p, 30)


def bench(fn_name, n_inner_a, n_outer_a, n_inner_b, n_outer_b, n_repeat=20):
    print(f"\n=== {fn_name}: ({n_inner_a},{n_outer_a}) vs ({n_inner_b},{n_outer_b}) ===")
    print(f"  {'label':<14} {'max rel err':>12} {'t_old [ms]':>12} {'t_new [ms]':>12} {'speedup':>8}")
    fn = getattr(jeans, fn_name)
    worst = 0.0
    sum_old = 0.0
    sum_new = 0.0
    for label, r_s, rho_s, r_p, beta in PARAMS:
        R = R_for(r_p)

        # Warmup
        a = fn(R, beta, r_s, rho_s, r_p, n_inner=n_inner_a, n_outer=n_outer_a)
        b = fn(R, beta, r_s, rho_s, r_p, n_inner=n_inner_b, n_outer=n_outer_b)

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            a = fn(R, beta, r_s, rho_s, r_p, n_inner=n_inner_a, n_outer=n_outer_a)
        t_old = (time.perf_counter() - t0) / n_repeat

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            b = fn(R, beta, r_s, rho_s, r_p, n_inner=n_inner_b, n_outer=n_outer_b)
        t_new = (time.perf_counter() - t0) / n_repeat

        if isinstance(a, tuple):
            ra = np.concatenate(a); rb = np.concatenate(b)
        else:
            ra = a; rb = b
        rel = np.max(np.abs(rb - ra) / np.abs(ra))
        worst = max(worst, rel)
        sum_old += t_old
        sum_new += t_new
        print(f"  {label:<14} {rel:>12.3e} {t_old*1e3:>12.3f} {t_new*1e3:>12.3f} {t_old/t_new:>8.2f}x")
    print(f"  worst rel err: {worst:.3e}    aggregate speedup: {sum_old/sum_new:.2f}x")
    return worst, sum_old / sum_new


if __name__ == "__main__":
    print("Solver-grid halving benchmark: (4096, 1024) -> (2048, 512)")
    w1, s1 = bench("Sigma_sigma_los2_grid",      4096, 1024, 2048, 512)
    w2, s2 = bench("Sigma_sigma_los2_grid_pair", 4096, 1024, 2048, 512)
    print("\nSummary:")
    print(f"  Sigma_sigma_los2_grid:      worst rel err {w1:.2e}, speedup {s1:.2f}x")
    print(f"  Sigma_sigma_los2_grid_pair: worst rel err {w2:.2e}, speedup {s2:.2f}x")
