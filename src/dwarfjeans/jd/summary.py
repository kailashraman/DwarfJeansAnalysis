"""
Offline J- and D-factor post-processing of the cached UFD population chains.

Pushes each chain through Stage-3 J(θ) and D(θ) integrals at four reporting
angles. Does not re-run dynesty.

Mock-specific test choices (these are not in the existing chains; they are
fixed here for the purposes of this synthetic-data validation):
  * Distance d = 30 kpc — representative UFD scale (Reticulum II ~30 kpc,
    Coma Ber ~44, Segue 1 ~23). Held fixed at truth here; not marginalized.
  * Tidal radius r_t = 1 kpc — the stage2.md / pipeline_overview.md
    convention for unresolved-σ_los systems (independent of host). Our
    UFD mocks are technically resolved (σ_los ≈ 6.5 km/s), but the proper
    Springel+08 + Eadie & Harris 16 r_t computation needs a host distance
    and a 3D position which the synthetic data don't have. r_t = 1 kpc is
    the closest defensible default and is used as truth for the J/D integral
    (no r_t marginalization).

J angles: 0.1°, 0.2°, 0.5°, α_c
D angles: 0.1°, 0.2°, 0.5°, α_c/2
where α_c = 2 r_½,3D / d.

Outputs: a per-galaxy table of medians + z-scores, plus population-level
mean/std(z), coverage, and KS-vs-N(0,1) diagnostics.

Reads:  mc_results/compact_ufd_seed{0..14}.npz
Writes: mc_results/ufd_pop_jd.json (per-realization)
        mc_results/ufd_pop_jd_diagnostics.json (population)
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
from scipy import stats

from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jeans import inference as jeans_inference


OUT_DIR = "mc_results"
SEEDS = list(range(15))

D_KPC = 30.0      # mock distance, kpc
R_T_KPC = 1.0     # mock tidal radius, kpc

# Fast defaults: thin_to=200, n_R=48, n_u=96. Verified to reproduce the
# original (thin_to=500, n_R=64, n_u=128) population diagnostics to within
# MC noise at ~8× the speed (~30s vs ~4 min for 15 seeds).
THIN_TO_DEFAULT = 200
N_R_DEFAULT = 48
N_U_DEFAULT = 96

JD_KEYS = (
    "log10_J_0p1deg", "log10_J_0p2deg", "log10_J_0p5deg", "log10_J_alphac",
    "log10_D_0p1deg", "log10_D_0p2deg", "log10_D_0p5deg", "log10_D_alphacover2",
)


def reconstruct_truth(npz) -> dict:
    """As in extend_summary_M_half — recover full truth dict including
    half-light radii from older chain archives."""
    keys = list(npz["truth_keys"])
    vals = list(npz["truth_vals"])
    truth = {k: float(v) for k, v in zip(keys, vals)}
    if "R_half_2d" not in truth:
        r_p = truth["r_p"]; r_s = truth["r_s"]; rho_s = truth["rho_s"]
        truth["R_half_2d"] = r_p
        truth["r_half_3d"] = 1.30476740610256 * r_p
        truth["M_half_2d"] = float(jeans.nfw_M(truth["R_half_2d"], r_s, rho_s))
        truth["M_half_3d"] = float(jeans.nfw_M(truth["r_half_3d"], r_s, rho_s))
        truth["log10_M_half_2d"] = float(np.log10(truth["M_half_2d"]))
        truth["log10_M_half_3d"] = float(np.log10(truth["M_half_3d"]))
    return truth


def header() -> str:
    abbr = {
        "log10_J_0p1deg": "J(0.1°)", "log10_J_0p2deg": "J(0.2°)",
        "log10_J_0p5deg": "J(0.5°)", "log10_J_alphac": "J(α_c)",
        "log10_D_0p1deg": "D(0.1°)", "log10_D_0p2deg": "D(0.2°)",
        "log10_D_0p5deg": "D(0.5°)", "log10_D_alphacover2": "D(α_c/2)",
    }
    parts = [f"{'seed':>4s}"]
    for k in JD_KEYS:
        parts.append(f"{abbr[k]:>9s}")
        parts.append(f"{'z':>5s}")
    return "  ".join(parts)


def truth_row(jd: dict) -> str:
    parts = [f"{'TRUTH':>4s}"]
    for k in JD_KEYS:
        parts.append(f"{jd[k]['truth']:>+9.3f}")
        parts.append(f"{'':>5}")
    return "  ".join(parts)


def fmt_row(seed: int, jd: dict) -> str:
    parts = [f"{seed:>4d}"]
    for k in JD_KEYS:
        parts.append(f"{jd[k]['median']:>+9.3f}")
        parts.append(f"{jd[k]['z']:>+5.2f}")
    return "  ".join(parts)


def population_diagnostics(rows: list[dict]) -> dict:
    out = {}
    zs = {k: np.array([r["jd"][k]["z"] for r in rows]) for k in JD_KEYS}
    biases = {k: np.array([r["jd"][k]["median"] - r["jd"][k]["truth"]
                           for r in rows]) for k in JD_KEYS}
    sigmas = {k: np.array([0.5 * (r["jd"][k]["sigma_lo"] +
                                  r["jd"][k]["sigma_hi"]) for r in rows])
              for k in JD_KEYS}
    cov68 = {}
    for k in JD_KEYS:
        in_iv = [r["jd"][k]["q16"] <= r["jd"][k]["truth"] <= r["jd"][k]["q84"]
                 for r in rows]
        cov68[k] = float(np.mean(in_iv))
    for k in JD_KEYS:
        ks_stat, ks_p = stats.kstest(zs[k], "norm")
        out[k] = {
            "mean_z": float(np.mean(zs[k])),
            "std_z": float(np.std(zs[k], ddof=1)),
            "median_bias": float(np.median(biases[k])),
            "mean_bias": float(np.mean(biases[k])),
            "mean_sigma": float(np.mean(sigmas[k])),
            "cov68": cov68[k],
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        }
    return out


def print_diagnostics(diag: dict) -> None:
    print("\nPopulation J/D diagnostics (n = 15)")
    print(f"  {'param':<22} {'mean(z)':>9} {'std(z)':>8} "
          f"{'med bias':>10} {'<σ>':>8} {'cov68%':>8} {'KS p':>7}")
    for k in JD_KEYS:
        d = diag[k]
        print(f"  {k:<22} "
              f"{d['mean_z']:>+9.2f} "
              f"{d['std_z']:>8.2f} "
              f"{d['median_bias']:>+10.3f} "
              f"{d['mean_sigma']:>8.3f} "
              f"{d['cov68']:>7.0%} "
              f"{d['ks_p']:>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asimov", action="store_true",
                    help="Run J/D push on the single Asimov chain "
                         "(mc_results/compact_ufd_asimov.npz) instead of the "
                         "15-realization MC chains. Skips population diagnostics.")
    ap.add_argument("--thin-to", type=int, default=THIN_TO_DEFAULT,
                    help=f"Samples per chain pushed through J/D (default {THIN_TO_DEFAULT}).")
    ap.add_argument("--n-R", type=int, default=N_R_DEFAULT,
                    help=f"R-grid resolution in J/D integral (default {N_R_DEFAULT}).")
    ap.add_argument("--n-u", type=int, default=N_U_DEFAULT,
                    help=f"u-grid resolution in J/D integral (default {N_U_DEFAULT}).")
    args = ap.parse_args()

    if args.asimov:
        path = os.path.join(OUT_DIR, "compact_ufd_asimov.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run `python run_ufd_population.py --asimov` "
                f"first to produce the Asimov chain."
            )
        d = np.load(path, allow_pickle=True)
        truth = reconstruct_truth(d)
        jd = jeans_inference.summarize_jd(
            d["samples_eq"], truth,
            d_kpc=D_KPC, r_t_kpc=R_T_KPC,
            thin_to=args.thin_to, n_R=args.n_R, n_u=args.n_u,
        )
        print(header())
        print(truth_row(jd))
        print("-" * len(header()))
        # No z-row-with-realization-spread for Asimov; show the single posterior
        # truth-vs-median offset.
        parts = [f"{'ASIMOV':>4s}"]
        for k in JD_KEYS:
            parts.append(f"{jd[k]['median']:>+9.3f}")
            # 'z' column is now the per-quantity (truth - median)/sigma offset
            # — for Asimov this is posterior-asymmetry offset, not realization
            # spread; we report it for shape but not as a calibration metric.
            parts.append(f"{jd[k]['z']:>+5.2f}")
        print("  ".join(parts))
        meta = jd["_meta"]
        print(f"  [thin_to={meta['thin_to']}, "
              f"α_c={meta['alpha_c_deg']:.4f}°, "
              f"d={meta['d_kpc']} kpc, r_t={meta['r_t_kpc']} kpc, "
              f"n_R={args.n_R}, n_u={args.n_u}]")
        out_path = os.path.join(OUT_DIR, "ufd_asimov_jd.json")
        with open(out_path, "w") as f:
            json.dump({
                "d_kpc": D_KPC, "r_t_kpc": R_T_KPC,
                "thin_to": args.thin_to, "n_R": args.n_R, "n_u": args.n_u,
                "asimov": True,
                "jd": {k: v for k, v in jd.items() if k != "_meta"},
                "_meta": jd["_meta"],
            }, f, indent=2)
        print(f"\nWrote {out_path}")
        return

    rows = []
    for i, seed in enumerate(SEEDS):
        path = os.path.join(OUT_DIR, f"compact_ufd_seed{seed}.npz")
        d = np.load(path, allow_pickle=True)
        truth = reconstruct_truth(d)
        jd = jeans_inference.summarize_jd(
            d["samples_eq"], truth,
            d_kpc=D_KPC, r_t_kpc=R_T_KPC,
            thin_to=args.thin_to, n_R=args.n_R, n_u=args.n_u,
        )
        rows.append({"seed": seed, "jd": jd, "truth": truth})

        # Running table after each
        print()
        print(header())
        print(truth_row(rows[0]["jd"]))
        print("-" * len(header()))
        for r in rows:
            print(fmt_row(r["seed"], r["jd"]))
        meta = jd["_meta"]
        print(f"  [seed={seed}: thin_to={meta['thin_to']}, "
              f"α_c={meta['alpha_c_deg']:.4f}°, "
              f"d={meta['d_kpc']} kpc, r_t={meta['r_t_kpc']} kpc, "
              f"n_R={args.n_R}, n_u={args.n_u}]")

    diag = population_diagnostics(rows)
    print_diagnostics(diag)

    table_path = os.path.join(OUT_DIR, "ufd_pop_jd.json")
    with open(table_path, "w") as f:
        json.dump({"d_kpc": D_KPC, "r_t_kpc": R_T_KPC,
                   "thin_to": args.thin_to, "n_R": args.n_R, "n_u": args.n_u,
                   "rows":
                   [{"seed": r["seed"],
                     "jd": {k: v for k, v in r["jd"].items() if k != "_meta"},
                     "_meta": r["jd"]["_meta"]}
                    for r in rows]}, f, indent=2)
    diag_path = os.path.join(OUT_DIR, "ufd_pop_jd_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({"d_kpc": D_KPC, "r_t_kpc": R_T_KPC,
                   "thin_to": args.thin_to, "n_R": args.n_R, "n_u": args.n_u,
                   "diagnostics": diag},
                  f, indent=2)
    print(f"\nWrote {table_path}")
    print(f"Wrote {diag_path}")


if __name__ == "__main__":
    main()
