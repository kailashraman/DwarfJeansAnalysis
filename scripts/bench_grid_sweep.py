"""Sweep (n_inner, n_outer) below 2048/512 to find the accuracy floor."""
from __future__ import annotations
import time
import numpy as np
from dwarfjeans.jeans import solver as jeans

PARAMS = [
    ("UFD-iso",     0.10, 1e9,  0.03,  0.0),
    ("UFD-rad",     0.10, 1e9,  0.03,  0.5),
    ("UFD-tan",     0.10, 1e9,  0.03, -1.0),
    ("classical",   1.0,  1e7,  0.30,  0.0),
    ("classical-r", 1.0,  1e7,  0.30,  0.3),
]

# Reference: 4096 / 2048 (matches the gold-standard quad test setting).
REF_INNER, REF_OUTER = 4096, 2048

def R_for(r_p):
    return np.geomspace(0.01 * r_p, 5.0 * r_p, 30)

def measure(fn, n_inner, n_outer, n_repeat=20):
    worst = 0.0
    t_total = 0.0
    for _, r_s, rho_s, r_p, beta in PARAMS:
        R = R_for(r_p)
        ref = fn(R, beta, r_s, rho_s, r_p, n_inner=REF_INNER, n_outer=REF_OUTER)
        cur = fn(R, beta, r_s, rho_s, r_p, n_inner=n_inner, n_outer=n_outer)
        if isinstance(ref, tuple):
            ra = np.concatenate(ref); rb = np.concatenate(cur)
        else:
            ra = ref; rb = cur
        rel = np.max(np.abs(rb - ra) / np.abs(ra))
        worst = max(worst, rel)
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            fn(R, beta, r_s, rho_s, r_p, n_inner=n_inner, n_outer=n_outer)
        t_total += (time.perf_counter() - t0) / n_repeat
    return worst, t_total

GRIDS = [
    (4096, 2048),
    (4096, 1024),
    (2048, 512),
    (1024, 512),
    (1024, 256),
    (512,  256),
    (512,  128),
    (256,  128),
]

for fn_name in ["Sigma_sigma_los2_grid", "Sigma_sigma_los2_grid_pair"]:
    fn = getattr(jeans, fn_name)
    print(f"\n=== {fn_name}  (ref = {REF_INNER}/{REF_OUTER}) ===")
    print(f"  {'(n_in,n_out)':<14} {'worst rel err':>14} {'t_agg [ms]':>12} {'speedup vs 2048/512':>22}")
    base_t = None
    for n_in, n_out in GRIDS:
        worst, t = measure(fn, n_in, n_out)
        if (n_in, n_out) == (2048, 512):
            base_t = t
        suffix = "" if base_t is None else f"  {base_t/t:>5.2f}x"
        print(f"  ({n_in:>4},{n_out:>4}) {worst:>14.3e} {t*1e3:>12.3f}{suffix}")
