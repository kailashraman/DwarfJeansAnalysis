"""
Population study: 15 mock UFD galaxies with the same true halo (r_s = 0.3 kpc,
rho_s = 3e8 Msun/kpc^3, r_p = 0.05 kpc, N_stars = 30, beta = 0, V_sys = 0).
Each realization differs only in the random seed used to draw stellar
positions and velocity scatter.

For each run we record posterior 16/50/84 percentiles for V, log10 r_s,
log10 rho_s, beta_tilde, and the derived log10(rho_s · r_s^3), plus the
asymmetric z-score (truth - median) / sigma_{lower or upper}.

A running table is printed after every realization, with a 'truth' row
prepended so we can verify all 15 galaxies share the same input truth.

NOTE on parallelism: this box has 1 CPU (`nproc`), so multiprocessing only
adds overhead. Runs are sequential here. On a multi-core node the per-run
code is independent and the loop can be wrapped in concurrent.futures.
"""

from __future__ import annotations

import os
import time
import json
import argparse
import numpy as np
from scipy import stats

import mock_galaxy
import jeans_inference


# -- Configuration -----------------------------------------------------------
TRUTH = dict(
    n_stars=30, r_s=0.3, rho_s=3e8, r_p=0.05,
    beta=0.0, V_sys=0.0, sigma_eps=2.0,
)
N_REALIZATIONS = 15
SEEDS = list(range(N_REALIZATIONS))
NLIVE = 300
DLOGZ = 0.5
SAMPLE = "unif"
OUT_DIR = "mc_results"
TABLE_PATH = os.path.join(OUT_DIR, "ufd_pop_table.json")

PARAM_KEYS = ("V", "log10_rs", "log10_rhos", "beta_tilde", "log10_rhos_rs3")
# Truth values for each derived parameter, computed once from TRUTH:
TRUTH_VALS = {
    "V": TRUTH["V_sys"],
    "log10_rs": float(np.log10(TRUTH["r_s"])),
    "log10_rhos": float(np.log10(TRUTH["rho_s"])),
    "beta_tilde": TRUTH["beta"] / (2.0 - TRUTH["beta"]),
    "log10_rhos_rs3": float(np.log10(TRUTH["rho_s"] * TRUTH["r_s"] ** 3)),
}


def existing_chain(seed: int):
    p = os.path.join(OUT_DIR, f"compact_ufd_seed{seed}.npz")
    return p if os.path.exists(p) else None


ASIMOV_CHAIN_PATH = os.path.join(OUT_DIR, "compact_ufd_asimov.npz")


def run_or_load_asimov() -> dict:
    """Single Asimov realization — deterministic, no seed."""
    g = mock_galaxy.make_asimov_galaxy(**TRUTH)
    truth_dict = g["truth"]

    if os.path.exists(ASIMOV_CHAIN_PATH):
        d = np.load(ASIMOV_CHAIN_PATH, allow_pickle=True)
        samples_eq = d["samples_eq"]
        elapsed = float(d["elapsed"]) if "elapsed" in d.files else float("nan")
        n_iter = int(d["n_iter"]) if "n_iter" in d.files else -1
        logz = float(d["logz"]) if "logz" in d.files else float("nan")
        cached_flag = True
    else:
        t0 = time.perf_counter()
        res = jeans_inference.run_inference(
            g, V_center=TRUTH["V_sys"],
            nlive=NLIVE, dlogz=DLOGZ, rseed=0,
            sample=SAMPLE, print_progress=False,
            asimov=True,
        )
        elapsed = time.perf_counter() - t0
        samples_eq = res["samples_eq"]
        n_iter = res["n_iter"]
        logz = float(res["logz"])
        np.savez(
            ASIMOV_CHAIN_PATH,
            samples_eq=samples_eq,
            param_names=np.array(("V", "log10_rs", "log10_rhos", "beta_tilde")),
            R=g["R"], V=g["V"], sigma_eps=g["sigma_eps"], p=g["p"],
            sigma_los_true=g["sigma_los_true"],
            truth_keys=np.array(list(truth_dict.keys())),
            truth_vals=np.array([float(v) for v in truth_dict.values()]),
            logz=res["logz"], logz_err=res["logz_err"],
            n_iter=res["n_iter"],
            elapsed=elapsed,
            is_asimov=np.array(True),
        )
        cached_flag = False

    summary = jeans_inference.summarize_posterior(samples_eq, truth_dict, asimov=True)
    return {
        "asimov": True,
        "elapsed": float(elapsed), "n_iter": int(n_iter),
        "logz": float(logz), "cached": cached_flag,
        "sigma_los_true_med": float(np.median(g["sigma_los_true"])),
        "summary": {k: {sk: (float(sv) if not isinstance(sv, bool) else sv)
                        for sk, sv in s.items()}
                    for k, s in summary.items()},
    }


def run_or_load(seed: int) -> dict:
    """Run inference for the given seed, or reuse a cached chain on disk."""
    rng = np.random.default_rng(1000 * seed + abs(hash("compact_ufd")) % 1000)
    g = mock_galaxy.make_mock_galaxy(rng=rng, **TRUTH)
    truth_dict = g["truth"]

    cached_path = existing_chain(seed)
    if cached_path is not None:
        d = np.load(cached_path, allow_pickle=True)
        samples_eq = d["samples_eq"]
        elapsed = float(d["elapsed"]) if "elapsed" in d.files else float("nan")
        n_iter = int(d["n_iter"]) if "n_iter" in d.files else -1
        logz = float(d["logz"]) if "logz" in d.files else float("nan")
        cached_flag = True
    else:
        t0 = time.perf_counter()
        res = jeans_inference.run_inference(
            g, V_center=TRUTH["V_sys"],
            nlive=NLIVE, dlogz=DLOGZ, rseed=seed,
            sample=SAMPLE, print_progress=False,
        )
        elapsed = time.perf_counter() - t0
        samples_eq = res["samples_eq"]
        n_iter = res["n_iter"]
        logz = float(res["logz"])
        out_path = os.path.join(OUT_DIR, f"compact_ufd_seed{seed}.npz")
        np.savez(
            out_path,
            samples_eq=samples_eq,
            param_names=np.array(("V", "log10_rs", "log10_rhos", "beta_tilde")),
            R=g["R"], V=g["V"], sigma_eps=g["sigma_eps"], p=g["p"],
            sigma_los_true=g["sigma_los_true"],
            truth_keys=np.array(list(truth_dict.keys())),
            truth_vals=np.array([float(v) for v in truth_dict.values()]),
            logz=res["logz"], logz_err=res["logz_err"],
            n_iter=res["n_iter"],
            elapsed=elapsed,
        )
        cached_flag = False

    summary = jeans_inference.summarize_posterior(samples_eq, truth_dict)
    # Capture this run's truth values too, so the running table can verify
    # they are identical across realizations.
    run_truth = {k: float(summary[k]["truth"]) for k in PARAM_KEYS}
    return {
        "seed": seed, "elapsed": float(elapsed), "n_iter": int(n_iter),
        "logz": float(logz), "cached": cached_flag,
        "V_std_obs": float(g["V"].std()),
        "sigma_los_true_med": float(np.median(g["sigma_los_true"])),
        "truth_per_run": run_truth,
        "summary": {k: {sk: float(sv) for sk, sv in s.items()}
                    for k, s in summary.items()},
    }


# -- Table formatting --------------------------------------------------------

def header() -> str:
    cols = ["seed"]
    for k in PARAM_KEYS:
        cols.append(f"{'med  '+k:>22}")
        cols.append(f"{'  z':>6}")
    cols.append(f"{'time':>6}")
    return "  ".join(cols)


def truth_row() -> str:
    parts = [f"{'TRUTH':>4s}"]
    for k in PARAM_KEYS:
        parts.append(f"{TRUTH_VALS[k]:>+22.4f}")
        parts.append(f"{'':>6}")
    parts.append(f"{'':>6}")
    return "  ".join(parts)


def fmt_row(r: dict, mismatch: bool) -> str:
    s = r["summary"]
    tag = "*" if mismatch else (" " if not r["cached"] else "c")
    parts = [f"{r['seed']:>3d}{tag}"]
    for k in PARAM_KEYS:
        med = s[k]["median"]
        z = s[k]["z"]
        parts.append(f"{med:>+22.4f}")
        parts.append(f"{z:>+6.2f}")
    parts.append(f"{r['elapsed']:>5.0f}s")
    return "  ".join(parts)


def truth_matches(r: dict) -> bool:
    """All per-run truth values agree with the global TRUTH_VALS to 1e-9?"""
    for k in PARAM_KEYS:
        if abs(r["truth_per_run"][k] - TRUTH_VALS[k]) > 1e-9:
            return False
    return True


def print_running_table(rows: list[dict]) -> None:
    print()
    print(header())
    print(truth_row())
    print("-" * len(header()))
    for r in rows:
        print(fmt_row(r, mismatch=not truth_matches(r)))


# -- Population diagnostics --------------------------------------------------

def population_diagnostics(rows: list[dict]) -> dict:
    """
    For each parameter:
      * mean / std of the asymmetric z-score (target: 0, 1)
      * coverage at the 68% credible interval (truth in [16, 84])
      * coverage at the 95% credible interval (truth in [2.5, 97.5])
        — this requires the raw equal-weight chain, so reload from disk
      * median bias (median - truth, raw units)  — sign indicates direction
      * Kolmogorov-Smirnov p-value vs N(0,1) (sanity check on z distribution)
    """
    out = {}
    n = len(rows)

    # Pull all z-scores into per-parameter arrays
    zs = {k: np.array([r["summary"][k]["z"] for r in rows]) for k in PARAM_KEYS}
    biases = {k: np.array([r["summary"][k]["median"] - r["summary"][k]["truth"]
                           for r in rows]) for k in PARAM_KEYS}

    # 68% coverage: truth in [q16, q84]?
    cov68 = {}
    for k in PARAM_KEYS:
        in_iv = []
        for r in rows:
            t = r["summary"][k]["truth"]
            in_iv.append(r["summary"][k]["q16"] <= t <= r["summary"][k]["q84"])
        cov68[k] = float(np.mean(in_iv))

    # 95% coverage requires the raw chains — load and compute q2.5, q97.5
    cov95 = {}
    for k in PARAM_KEYS:
        in_iv = []
        for r in rows:
            seed = r["seed"]
            d = np.load(os.path.join(OUT_DIR, f"compact_ufd_seed{seed}.npz"))
            samples = d["samples_eq"]
            V, lr, lp, bt = samples.T
            chain = {
                "V": V, "log10_rs": lr, "log10_rhos": lp, "beta_tilde": bt,
                "log10_rhos_rs3": lp + 3.0 * lr,
            }[k]
            q2p5, q97p5 = np.percentile(chain, [2.5, 97.5])
            t = r["summary"][k]["truth"]
            in_iv.append(q2p5 <= t <= q97p5)
        cov95[k] = float(np.mean(in_iv))

    for k in PARAM_KEYS:
        # KS test: is the empirical z distribution consistent with N(0, 1)?
        # With n=15 this has weak power, but a tiny p-value would still flag
        # gross miscalibration.
        ks_stat, ks_p = stats.kstest(zs[k], "norm")
        out[k] = {
            "mean_z": float(np.mean(zs[k])),
            "std_z": float(np.std(zs[k], ddof=1)),
            "median_bias": float(np.median(biases[k])),
            "mean_bias": float(np.mean(biases[k])),
            "cov68": cov68[k],
            "cov95": cov95[k],
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        }
    return out


def print_diagnostics(diag: dict) -> None:
    print("\nPopulation diagnostics (n = 15 realizations)")
    print(f"  {'param':<18} {'mean(z)':>9} {'std(z)':>8} "
          f"{'med bias':>10} {'cov68%':>8} {'cov95%':>8} {'KS p':>7}")
    for k in PARAM_KEYS:
        d = diag[k]
        print(f"  {k:<18} "
              f"{d['mean_z']:>+9.2f} "
              f"{d['std_z']:>8.2f} "
              f"{d['median_bias']:>+10.3f} "
              f"{d['cov68']:>7.0%} "
              f"{d['cov95']:>7.0%} "
              f"{d['ks_p']:>7.3f}")
    print("\n  Targets for an unbiased, well-calibrated procedure:")
    print("    mean(z) ≈ 0, std(z) ≈ 1, cov68 ≈ 68%, cov95 ≈ 95%, KS p > ~0.05")
    print("  Asymmetric z's are not strictly N(0,1) even for a perfect")
    print("  procedure (median is biased relative to one-sided σ when the")
    print("  posterior is skewed), but |mean(z)| << 1 and reasonable coverage")
    print("  rule out gross procedural bias.")


# -- Main loop ---------------------------------------------------------------

def print_asimov_summary(r: dict) -> None:
    s = r["summary"]
    print(f"\nAsimov realization (single, deterministic).")
    print(f"  σ_los_true median = {r['sigma_los_true_med']:.3f} km/s; "
          f"wall {r['elapsed']:.0f}s; n_eq {r['n_iter']}; "
          f"{'cached' if r['cached'] else 'fresh'}")
    print()
    print(f"  {'param':<22} {'truth':>12} {'median':>12} {'σ_lo':>8} {'σ_hi':>8} "
          f"{'med-truth':>11}  notes")
    for k in PARAM_KEYS + ("log10_M_half_2d", "log10_M_half_3d"):
        e = s[k]
        bias = e["median"] - e["truth"]
        prior_only = e.get("prior_only", False)
        note = "[prior-only]" if prior_only else ""
        print(f"  {k:<22} {e['truth']:>+12.4f} {e['median']:>+12.4f} "
              f"{e['sigma_lo']:>8.4f} {e['sigma_hi']:>8.4f} "
              f"{bias:>+11.4f}  {note}")
    print("\n  Targets for a procedure with no implementation bug:")
    print("    Asimov MLE at truth (verified separately by scipy.optimize)")
    print("    Asimov medians offset from truth by less than the MC realization")
    print("    spread (the offset is posterior asymmetry, not bias)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asimov", action="store_true",
                    help="Run a single Asimov realization (fast dev-loop check) "
                         "instead of the 15-realization MC.")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.asimov:
        print("Asimov dev-loop run (single deterministic realization).")
        print("Truth:")
        for k, v in TRUTH.items():
            print(f"  {k}: {v}")
        r = run_or_load_asimov()
        print_asimov_summary(r)
        out_path = os.path.join(OUT_DIR, "ufd_asimov_table.json")
        with open(out_path, "w") as f:
            json.dump({"truth": TRUTH_VALS, "asimov": r}, f, indent=2)
        print(f"\nWrote {out_path}")
        return

    rows: list[dict] = []
    print(f"Truth (shared by all {N_REALIZATIONS} realizations):")
    for k, v in TRUTH.items():
        print(f"  {k}: {v}")
    print(f"  → log10(rho_s · r_s^3) = {TRUTH_VALS['log10_rhos_rs3']:.3f}")
    print()
    t_start = time.perf_counter()
    for i, seed in enumerate(SEEDS):
        t_run = time.perf_counter()
        r = run_or_load(seed)
        rows.append(r)
        elapsed_total = time.perf_counter() - t_start
        print(f"\n[{i+1}/{N_REALIZATIONS}] seed={seed} done in {r['elapsed']:.0f}s "
              f"(cumulative wall {elapsed_total:.0f}s); "
              f"σ_los_med={r['sigma_los_true_med']:.2f} V_std={r['V_std_obs']:.2f}"
              + ("  [cached]" if r["cached"] else ""))
        print_running_table(rows)

        # Persist the running table
        with open(TABLE_PATH, "w") as f:
            json.dump({"truth": TRUTH_VALS, "rows": rows}, f, indent=2)

    # End-of-run population diagnostics
    diag = population_diagnostics(rows)
    print_diagnostics(diag)
    with open(os.path.join(OUT_DIR, "ufd_pop_diagnostics.json"), "w") as f:
        json.dump({"truth": TRUTH_VALS, "diagnostics": diag}, f, indent=2)


if __name__ == "__main__":
    main()
