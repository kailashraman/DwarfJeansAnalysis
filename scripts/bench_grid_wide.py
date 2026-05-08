"""Wider parameter sweep validating the (2048, 512) default vs (4096, 2048) reference.

Covers (a) β edges of the Jeffreys prior support, (b) r_p/r_s ratio extremes,
(c) real-galaxy R-arrays (segue_1, draco_1), and (d) Q/P cancellation
(grid_pair).

Run: python scripts/bench_grid_wide.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dwarfjeans.jeans import solver as jeans

REPO = Path(__file__).resolve().parents[1]

CUR_INNER, CUR_OUTER = 2048, 512
REF_INNER, REF_OUTER = 4096, 2048

# Jeffreys prior on β runs ~[-9, 0.99]; explore both edges.
# r_p / r_s ratio in the priors can hit ~0.001 (UFD with diffuse halo) up to ~10.
PARAM_GRID = [
    # (label, r_s, rho_s, r_p, beta)
    # β edges
    ("UFD-iso",         0.10, 1e9,  0.03,   0.0),
    ("UFD-radial-extreme", 0.10, 1e9, 0.03,  0.95),
    ("UFD-tangential-extreme", 0.10, 1e9, 0.03, -9.0),
    ("classical-radial-extreme", 1.0, 1e7, 0.30, 0.95),
    ("classical-tangential-extreme", 1.0, 1e7, 0.30, -9.0),
    # r_p / r_s ratio extremes
    ("rp_rs_0p001",     1.0,  1e7,  0.001,  0.0),
    ("rp_rs_0p01",      1.0,  1e7,  0.01,   0.0),
    ("rp_rs_10",        0.05, 1e9,  0.5,    0.0),
    ("rp_rs_10_radial", 0.05, 1e9,  0.5,    0.5),
    # rho_s extremes (mostly a prefactor; should not affect rel err)
    ("low-rho",         1.0,  1e5,  0.30,  0.0),
    ("high-rho",        0.10, 1e11, 0.03,  0.0),
]


def measure(fn, R, beta, r_s, rho_s, r_p):
    ref = fn(R, beta, r_s, rho_s, r_p, n_inner=REF_INNER, n_outer=REF_OUTER)
    cur = fn(R, beta, r_s, rho_s, r_p, n_inner=CUR_INNER, n_outer=CUR_OUTER)
    if isinstance(ref, tuple):
        ra = np.concatenate(ref); rb = np.concatenate(cur)
    else:
        ra = ref; rb = cur
    rel = np.abs(rb - ra) / np.maximum(np.abs(ra), 1e-300)
    return float(rel.max()), float(np.median(rel)), int(rel.argmax())


def real_galaxy_R(key):
    p = REPO / "data" / "star_catalogs" / f"{key}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    R = d["R"]
    # Floor R the same way prepare_jeans_input does — see jeans/preprocess.py.
    # Use a conservative floor: 1e-5 kpc.
    return np.maximum(R, 1e-5)


def fmt(x):
    return f"{x:>10.3e}"


def main():
    print(f"Wide sweep: ({CUR_INNER},{CUR_OUTER}) vs ref ({REF_INNER},{REF_OUTER})\n")

    # Section A: synthetic R-arrays at edge parameter combinations
    print(f"=== Section A: synthetic R = geomspace(0.01·r_p, 5·r_p, 30) ===")
    for fn_name in ["Sigma_sigma_los2_grid", "Sigma_sigma_los2_grid_pair"]:
        fn = getattr(jeans, fn_name)
        print(f"\n  {fn_name}")
        print(f"  {'label':<32} {'max rel err':>12} {'median rel':>12}")
        worst = 0.0
        for label, r_s, rho_s, r_p, beta in PARAM_GRID:
            R = np.geomspace(0.01 * r_p, 5.0 * r_p, 30)
            mx, md, _ = measure(fn, R, beta, r_s, rho_s, r_p)
            worst = max(worst, mx)
            flag = "  ⚠" if mx > 1e-2 else ""
            print(f"  {label:<32} {mx:>12.3e} {md:>12.3e}{flag}")
        print(f"  worst: {worst:.3e}")

    # Section B: Q/P cancellation in T = 3 - Q/P
    print(f"\n=== Section B: Jeffreys T = 3 - Q/P at synthetic R ===")
    print(f"  {'label':<32} {'max rel err on T':>18} {'min T':>10} {'max T':>10}")
    worst_T = 0.0
    for label, r_s, rho_s, r_p, beta in PARAM_GRID:
        R = np.geomspace(0.01 * r_p, 5.0 * r_p, 30)
        Pr, Qr = jeans.Sigma_sigma_los2_grid_pair(R, beta, r_s, rho_s, r_p,
                                                  n_inner=REF_INNER, n_outer=REF_OUTER)
        Pc, Qc = jeans.Sigma_sigma_los2_grid_pair(R, beta, r_s, rho_s, r_p,
                                                  n_inner=CUR_INNER, n_outer=CUR_OUTER)
        Tr = 3.0 - Qr / Pr
        Tc = 3.0 - Qc / Pc
        rel = np.abs(Tc - Tr) / np.maximum(np.abs(Tr), 1e-300)
        worst_T = max(worst_T, float(rel.max()))
        flag = "  ⚠" if rel.max() > 1e-2 else ""
        print(f"  {label:<32} {rel.max():>18.3e} {Tr.min():>10.3f} {Tr.max():>10.3f}{flag}")
    print(f"  worst T rel err: {worst_T:.3e}")

    # Section C: real-galaxy R-arrays at posterior-median-ish params
    print(f"\n=== Section C: real-galaxy R-arrays (UFD + classical) ===")
    real = [("segue_1", 0.0194, 1.86e9, 0.0194, 0.13),    # post-cut R from earlier run
            ("draco_1", 0.20,   1e8,    0.20,   0.0)]      # rough Plummer/NFW for Draco
    for key, r_s, rho_s, r_p, beta in real:
        R = real_galaxy_R(key)
        if R is None:
            print(f"  {key}: catalog not found, skipping")
            continue
        # Floor at solver s_min so we don't trip the bounds check.
        s_min = 1e-4 * min(r_s, r_p)
        R = np.maximum(R, s_min)
        for fn_name in ["Sigma_sigma_los2_grid", "Sigma_sigma_los2_grid_pair"]:
            fn = getattr(jeans, fn_name)
            mx, md, imax = measure(fn, R, beta, r_s, rho_s, r_p)
            flag = "  ⚠" if mx > 1e-2 else ""
            print(f"  {key:<10} {fn_name:<30} N={len(R):>4}  max rel err = {mx:.3e}  median = {md:.3e}{flag}")

    print("\nGate: rel err < 1e-2 (matches gold-standard tolerance)")


if __name__ == "__main__":
    main()
