"""Mock-data calibration for perspective + PM marginalisation.

Generates N realisations of an Antlia-II-like mock galaxy (large angular
field → non-trivial perspective signal), pushes each one through the 9D
Stage 2 likelihood with PM marginalisation enabled, and tabulates bias
and dispersion in σ_los(R_h), log10 J(α_c), and the recovered (μ_α*, μ_δ).

Truth halo + Plummer chosen to give Antlia-II-like dispersion (~6 km/s)
at a similar angular extent (R_h ≈ 1.4° on the sky). Truth PM and PM
priors taken from Pace+2022 for Antlia II.

Run:
    python -m tests.integration.run_perspective_calibration --n-realisations 12

Each realisation writes its dynesty chain to
``results/tests/perspective_calibration/seed{NNN}.npz`` so the script can
resume after interruption.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from dwarfjeans.jeans import inference as jeans_inference
from dwarfjeans.jeans import solver as jeans
from dwarfjeans.mocks import galaxy as mock_galaxy


# -- Truth configuration -----------------------------------------------------
# Smaller-N variant: 50 stars per realisation. Per-realisation σ_los
# precision is loose (~14%) but the population-mean bias estimate still
# converges as 1/√N_realisations.
TRUTH = dict(
    n_stars=50,
    r_s=1.0,           # kpc
    rho_s=5.0e7,       # M_sun / kpc^3 → σ_los at R_h ≈ 6 km/s
    r_p=3.0,           # kpc; r_p/d ≈ 0.024 rad ≈ 1.4° (Antlia-II-like)
    beta=0.0,
    V_sys=300.0,       # km/s (Antlia II vlos_systemic_kms = 288.8)
    sigma_eps=2.0,     # km/s
    d_kpc=124.0,
    ra_center_deg=143.8,
    dec_center_deg=-36.7,
    pmra_true_mas_yr=-0.093,
    pmdec_true_mas_yr=+0.100,
)

# PM measurement priors during inference (Pace+2022 Antlia II).
PM_PRIOR = dict(
    pmra_mean=-0.093, pmra_em=0.008, pmra_ep=0.008,
    pmdec_mean=+0.100, pmdec_em=0.009, pmdec_ep=0.009,
)

# Nuisance priors: tight Gaussians at truth — we're calibrating the
# perspective+PM part of the pipeline, not the distance/eps/rhalf priors.
NUISANCE_PRIORS = dict(
    d_mean=TRUTH["d_kpc"], d_sigma=2.0,
    eps_mean=0.0, eps_sigma=0.05,    # truncnorm clips to [0, 1)
    rhalf_mean=math.degrees(TRUTH["r_p"] / TRUTH["d_kpc"]) * 60.0,  # arcmin
    rhalf_sigma=0.05 * math.degrees(TRUTH["r_p"] / TRUTH["d_kpc"]) * 60.0,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "tests" / "perspective_calibration"
ARCMIN_TO_RAD = math.pi / (180.0 * 60.0)


def _alpha_c_deg(rho_s: float, r_s: float, R_max_kpc: float, d_kpc: float) -> float:
    """Critical angle α_c (degrees) at the truth — geometric, for log10 J(α_c).

    Definition matches scripts/run_production.py: α_c is the angular
    radius enclosing the outermost member, in radians. We return degrees.
    """
    return math.degrees(R_max_kpc / d_kpc)


def _truth_summary() -> dict:
    """Pre-compute truth values for σ_los(R_h) and log10 J at α_c."""
    r_p = TRUTH["r_p"]
    r_s = TRUTH["r_s"]
    rho_s = TRUTH["rho_s"]
    beta = TRUTH["beta"]
    sigma_at_Rhalf = float(jeans.sigma_los(np.array([r_p]), beta, r_s, rho_s, r_p,
                                            method="grid")[0])
    return {
        "sigma_los_Rhalf_kms": sigma_at_Rhalf,
        "log10_rs":            math.log10(r_s),
        "log10_rhos":          math.log10(rho_s),
        "R_half_2d_kpc":       r_p,
        "pmra_mas_yr":         TRUTH["pmra_true_mas_yr"],
        "pmdec_mas_yr":        TRUTH["pmdec_true_mas_yr"],
        "V_sys_kms":           TRUTH["V_sys"],
        "d_kpc":               TRUTH["d_kpc"],
    }


def _run_one(seed: int, *, nlive: int, dlogz: float, npool: int) -> dict:
    """Run a single realisation. Returns a dict of per-realisation summaries."""
    out_path = OUT_DIR / f"seed{seed:03d}.npz"
    if out_path.exists():
        return _load(seed)

    rng = np.random.default_rng(10_000 + seed)
    # R_max_factor=2 to mimic the production selection cut (R < 2 R_h).
    g = mock_galaxy.make_mock_galaxy(rng=rng, R_max_factor=2.0, **TRUTH)

    # Build the inputs Stage 2 expects.
    R_kpc = np.clip(g["R"], 1e-5, None)
    Rad_arcmin = R_kpc / TRUTH["d_kpc"] / ARCMIN_TO_RAD

    galaxy = {
        "R": R_kpc,
        "Rad_arcmin": Rad_arcmin,
        "V": g["V_observed"],    # 9D path consumes V_observed directly
        "sigma_eps": g["sigma_eps"],
        "p": g["p"],
        "truth": {"r_p": TRUTH["r_p"]},
    }
    perspective = {
        "V_observed": g["V_observed"],
        "RA_star": g["RA_star"], "Dec_star": g["Dec_star"],
        "ra_center": TRUTH["ra_center_deg"],
        "dec_center": TRUTH["dec_center_deg"],
    }

    t0 = time.perf_counter()
    res = jeans_inference.run_inference(
        galaxy,
        V_center=TRUTH["V_sys"], V_halfwidth=10.0,
        nlive=nlive, dlogz=dlogz, rseed=seed,
        print_progress=False,
        marginalize_nuisances=True,
        nuisance_priors=NUISANCE_PRIORS,
        prior_name="jeffreys",
        perspective=perspective,
        pm_prior=PM_PRIOR,
        npool=npool,
    )
    elapsed = time.perf_counter() - t0
    samples_eq = res["samples_eq"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        samples_eq=samples_eq,
        logz=res["logz"], logz_err=res["logz_err"],
        elapsed=elapsed, n_eq=res["n_eq"],
        seed=seed, nlive=nlive, dlogz=dlogz,
        sigma_los_true=g["sigma_los_true"],
        dv_persp_true=g["dv_persp_true"],
        R=R_kpc, V_observed=g["V_observed"], V_no_persp=g["V"],
        sigma_eps=g["sigma_eps"],
    )
    return _load(seed)


def _load(seed: int) -> dict:
    """Compute summaries from a cached chain."""
    out_path = OUT_DIR / f"seed{seed:03d}.npz"
    d = np.load(out_path, allow_pickle=True)
    samples_eq = d["samples_eq"]
    V, lr, lp, bt, dist, eps, rh, pmra, pmdec = samples_eq.T

    # Posterior σ_los at R_h (= r_p truth). Median + 1σ via per-sample
    # evaluation on a thinned subset (50 draws plenty given low variance
    # for this calibration).
    rng = np.random.default_rng(42 + seed)
    idx = rng.choice(samples_eq.shape[0], size=min(50, samples_eq.shape[0]),
                     replace=False)
    r_s_chain = 10.0 ** lr[idx]
    rho_s_chain = 10.0 ** lp[idx]
    beta_chain = jeans_inference.beta_tilde_to_beta(bt[idx])
    r_p_chain = dist[idx] * rh[idx] * ARCMIN_TO_RAD * np.sqrt(np.clip(1.0 - eps[idx], 0.0, None))
    sigma_at_R = np.full(idx.size, np.nan)
    for j, (rs, rhs, b, rp) in enumerate(zip(r_s_chain, rho_s_chain, beta_chain, r_p_chain)):
        try:
            sigma_at_R[j] = jeans.sigma_los(np.array([TRUTH["r_p"]]), b, rs, rhs, rp,
                                              method="grid")[0]
        except Exception:
            sigma_at_R[j] = np.nan

    def _q(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        return tuple(np.percentile(x, [16, 50, 84]).tolist())

    return {
        "seed": int(seed),
        "elapsed_s": float(d["elapsed"]),
        "n_eq": int(d["n_eq"]),
        "logz": float(d["logz"]),
        "sigma_los_Rhalf": dict(zip(("q16", "q50", "q84"), _q(sigma_at_R))),
        "pmra":            dict(zip(("q16", "q50", "q84"), _q(pmra))),
        "pmdec":           dict(zip(("q16", "q50", "q84"), _q(pmdec))),
        "V_sys":           dict(zip(("q16", "q50", "q84"), _q(V))),
        "d_kpc":           dict(zip(("q16", "q50", "q84"), _q(dist))),
        "median_dv_persp_kms": float(np.median(np.abs(d["dv_persp_true"]))),
        "max_dv_persp_kms":    float(np.max(np.abs(d["dv_persp_true"]))),
    }


def _aggregate(rows: list[dict], truth: dict) -> dict:
    """Bias and dispersion of (σ_los, μ_α*, μ_δ) across realisations.

    Acceptance: median bias well below the per-realisation 1σ, and ~68% of
    realisations have their 68% CI containing truth.
    """
    def _arr(key, q):
        return np.asarray([r[key][q] for r in rows], dtype=float)

    sigma_truth = truth["sigma_los_Rhalf_kms"]
    sigma_meds = _arr("sigma_los_Rhalf", "q50")
    sigma_q16s = _arr("sigma_los_Rhalf", "q16")
    sigma_q84s = _arr("sigma_los_Rhalf", "q84")
    sigma_cov = float(np.mean((sigma_q16s <= sigma_truth) & (sigma_truth <= sigma_q84s)))

    pmra_truth = truth["pmra_mas_yr"]
    pmra_meds = _arr("pmra", "q50")
    pmra_q16s = _arr("pmra", "q16")
    pmra_q84s = _arr("pmra", "q84")
    pmra_cov = float(np.mean((pmra_q16s <= pmra_truth) & (pmra_truth <= pmra_q84s)))

    pmdec_truth = truth["pmdec_mas_yr"]
    pmdec_meds = _arr("pmdec", "q50")
    pmdec_q16s = _arr("pmdec", "q16")
    pmdec_q84s = _arr("pmdec", "q84")
    pmdec_cov = float(np.mean((pmdec_q16s <= pmdec_truth) & (pmdec_truth <= pmdec_q84s)))

    return {
        "n_realisations": len(rows),
        "sigma_los_Rhalf": {
            "truth": sigma_truth,
            "median_of_medians": float(np.median(sigma_meds)),
            "bias_kms": float(np.median(sigma_meds) - sigma_truth),
            "bias_pct": float(100 * (np.median(sigma_meds) - sigma_truth) / sigma_truth),
            "dispersion_kms": float(np.std(sigma_meds, ddof=1)),
            "coverage_68": sigma_cov,
        },
        "pmra": {
            "truth": pmra_truth,
            "median_of_medians": float(np.median(pmra_meds)),
            "bias_mas_yr": float(np.median(pmra_meds) - pmra_truth),
            "coverage_68": pmra_cov,
        },
        "pmdec": {
            "truth": pmdec_truth,
            "median_of_medians": float(np.median(pmdec_meds)),
            "bias_mas_yr": float(np.median(pmdec_meds) - pmdec_truth),
            "coverage_68": pmdec_cov,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-realisations", type=int, default=8)
    ap.add_argument("--nlive", type=int, default=100)
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--npool", type=int, default=4)
    args = ap.parse_args()

    truth = _truth_summary()
    print(f"=== Perspective+PM calibration on Antlia-II-like mocks ===")
    print(f"  truth σ_los(R_h)  = {truth['sigma_los_Rhalf_kms']:.3f} km/s")
    print(f"  truth (μ_α*, μ_δ) = ({truth['pmra_mas_yr']:+.3f}, "
          f"{truth['pmdec_mas_yr']:+.3f}) mas/yr")
    print(f"  N realisations    = {args.n_realisations}, nlive={args.nlive}, "
          f"dlogz={args.dlogz}")
    print()

    rows = []
    for k in range(args.n_realisations):
        seed = args.seed_base + k
        t0 = time.perf_counter()
        row = _run_one(seed, nlive=args.nlive, dlogz=args.dlogz, npool=args.npool)
        rows.append(row)
        dt = time.perf_counter() - t0
        sigma = row["sigma_los_Rhalf"]
        pmra = row["pmra"]
        print(f"  seed={seed:3d}  σ_los={sigma['q50']:.3f} "
              f"[{sigma['q16']:.3f}, {sigma['q84']:.3f}]  "
              f"μ_α*={pmra['q50']:+.4f} [{pmra['q16']:+.4f}, {pmra['q84']:+.4f}]  "
              f"max|Δv|={row['max_dv_persp_kms']:.2f}  ({dt:.1f}s)")

    agg = _aggregate(rows, truth)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "aggregate.json").write_text(json.dumps(
        {"truth": truth, "aggregate": agg, "rows": rows}, indent=2,
    ))

    s = agg["sigma_los_Rhalf"]
    print()
    print(f"=== Aggregate (N={agg['n_realisations']}) ===")
    print(f"  σ_los(R_h):  truth={s['truth']:.3f}, median(med)={s['median_of_medians']:.3f} "
          f"({s['bias_pct']:+.2f}%), dispersion={s['dispersion_kms']:.3f} km/s, "
          f"coverage_68={s['coverage_68']:.0%}")
    pr = agg["pmra"]
    pd_ = agg["pmdec"]
    print(f"  μ_α*:        truth={pr['truth']:+.4f}, median(med)={pr['median_of_medians']:+.4f}, "
          f"coverage_68={pr['coverage_68']:.0%}")
    print(f"  μ_δ:         truth={pd_['truth']:+.4f}, median(med)={pd_['median_of_medians']:+.4f}, "
          f"coverage_68={pd_['coverage_68']:.0%}")
    print()
    print(f"Wrote {OUT_DIR / 'aggregate.json'}")


if __name__ == "__main__":
    main()
