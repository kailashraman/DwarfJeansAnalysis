"""Bench: parity + wall-time of vectorized solver vs the previous Python loop.

Item 1 of the speedup plan. Reproduces the exact pre-vectorization per-R
loop body locally and compares its output (and wall-time) against the
production solver.

Run: python scripts/bench_vectorize_loop.py
"""
from __future__ import annotations

import time

import numpy as np

from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jeans.solver import nu_sigma_r2_grid, plummer_nu, nfw_g, nfw_h, G_KPC_KMS2_MSUN


def loop_grid(R, beta, r_s, rho_s, r_p, n_inner=2048, n_outer=512):
    """Pre-vectorization Sigma_sigma_los2_grid (per-R Python loop)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    s_inf = 100.0 * max(r_s, r_p)
    s_min = 1e-4 * min(r_s, r_p)
    log_r_tab = np.linspace(np.log(s_min), np.log(s_inf), n_inner)
    r_tab = np.exp(log_r_tab)
    nsr2_tab = nu_sigma_r2_grid(r_tab, beta, r_s, rho_s, r_p, n_grid=n_inner, s_inf=s_inf)
    u_max = 100.0 * max(r_s, r_p)
    out = np.empty_like(R)
    for i, Ri in enumerate(R):
        u_min = 1e-4 * Ri
        log_u = np.linspace(np.log(u_min), np.log(u_max), n_outer - 1)
        u = np.concatenate([[0.0], np.exp(log_u)])
        r_of_u = np.sqrt(Ri ** 2 + u ** 2)
        nsr2_at = np.interp(np.log(np.clip(r_of_u, s_min, s_inf)),
                            log_r_tab, nsr2_tab)
        integrand = (1.0 - beta * Ri ** 2 / r_of_u ** 2) * nsr2_at
        out[i] = 2.0 * np.trapezoid(integrand, u)
    return out


def loop_grid_pair(R, beta, r_s, rho_s, r_p, n_inner=2048, n_outer=512):
    """Pre-vectorization Sigma_sigma_los2_grid_pair (per-R Python loop)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    s_inf = 100.0 * max(r_s, r_p)
    s_min = 1e-4 * min(r_s, r_p)
    log_s = np.linspace(np.log(s_min), np.log(s_inf), n_inner)
    s = np.exp(log_s); x = s / r_s
    log_pref = np.log(G_KPC_KMS2_MSUN * 4.0 * np.pi * rho_s * r_s ** 3)
    log_nu = np.log(plummer_nu(s, r_p))
    common = (2.0 * beta - 1.0) * log_s + log_nu + log_pref
    log_int_g = common + np.log(nfw_g(x))
    log_int_h = common + np.log(nfw_h(x))
    dlog = np.diff(log_s); log_dlog = np.log(dlog); half_log2 = np.log(2.0)
    log_seg_g = np.logaddexp(log_int_g[:-1], log_int_g[1:]) - half_log2 + log_dlog
    log_seg_h = np.logaddexp(log_int_h[:-1], log_int_h[1:]) - half_log2 + log_dlog
    log_I_g = np.concatenate([np.logaddexp.accumulate(log_seg_g[::-1])[::-1], [-np.inf]])
    log_I_h = np.concatenate([np.logaddexp.accumulate(log_seg_h[::-1])[::-1], [-np.inf]])
    nsr2_g_tab = np.exp(log_I_g - 2.0 * beta * log_s)
    nsr2_h_tab = np.exp(log_I_h - 2.0 * beta * log_s)
    log_r_tab = log_s
    u_max = 100.0 * max(r_s, r_p)
    P = np.empty_like(R); Q = np.empty_like(R)
    for i, Ri in enumerate(R):
        u_min = 1e-4 * Ri
        log_u = np.linspace(np.log(u_min), np.log(u_max), n_outer - 1)
        u = np.concatenate([[0.0], np.exp(log_u)])
        r_of_u = np.sqrt(Ri ** 2 + u ** 2)
        log_r_of_u = np.log(np.clip(r_of_u, s_min, s_inf))
        gA = np.interp(log_r_of_u, log_r_tab, nsr2_g_tab)
        hA = np.interp(log_r_of_u, log_r_tab, nsr2_h_tab)
        kernel = 1.0 - beta * Ri ** 2 / r_of_u ** 2
        P[i] = 2.0 * np.trapezoid(kernel * gA, u)
        Q[i] = 2.0 * np.trapezoid(kernel * hA, u)
    return P, Q


PARAMS = [
    ("UFD-iso",     0.10, 1e9,  0.03,  0.0),
    ("UFD-rad",     0.10, 1e9,  0.03,  0.5),
    ("UFD-tan",     0.10, 1e9,  0.03, -1.0),
    ("classical",   1.0,  1e7,  0.30,  0.0),
    ("classical-r", 1.0,  1e7,  0.30,  0.3),
]

def R_for(r_p):
    return np.geomspace(0.01 * r_p, 5.0 * r_p, 30)


def bench(name, loop_fn, vec_fn, n_repeat=20):
    print(f"\n=== {name}: loop vs vectorized (defaults 2048/512) ===")
    print(f"  {'label':<14} {'max rel err':>12} {'t_loop [ms]':>12} {'t_vec [ms]':>12} {'speedup':>8}")
    worst = 0.0; sum_l = 0.0; sum_v = 0.0
    for label, r_s, rho_s, r_p, beta in PARAMS:
        R = R_for(r_p)
        a = loop_fn(R, beta, r_s, rho_s, r_p)
        b = vec_fn(R, beta, r_s, rho_s, r_p)
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            a = loop_fn(R, beta, r_s, rho_s, r_p)
        tl = (time.perf_counter() - t0) / n_repeat
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            b = vec_fn(R, beta, r_s, rho_s, r_p)
        tv = (time.perf_counter() - t0) / n_repeat
        if isinstance(a, tuple):
            ra = np.concatenate(a); rb = np.concatenate(b)
        else:
            ra = a; rb = b
        rel = np.max(np.abs(rb - ra) / np.abs(ra))
        worst = max(worst, rel); sum_l += tl; sum_v += tv
        print(f"  {label:<14} {rel:>12.3e} {tl*1e3:>12.3f} {tv*1e3:>12.3f} {tl/tv:>8.2f}x")
    print(f"  worst rel err: {worst:.3e}    aggregate speedup: {sum_l/sum_v:.2f}x")
    return worst, sum_l / sum_v


if __name__ == "__main__":
    print("Vectorize-per-R-loop benchmark (Item 1)")
    w1, s1 = bench("Sigma_sigma_los2_grid",      loop_grid,      jeans.Sigma_sigma_los2_grid)
    w2, s2 = bench("Sigma_sigma_los2_grid_pair", loop_grid_pair, jeans.Sigma_sigma_los2_grid_pair)
    print("\nSummary:")
    print(f"  grid:      worst rel err {w1:.2e}, speedup {s1:.2f}x")
    print(f"  grid_pair: worst rel err {w2:.2e}, speedup {s2:.2f}x")
