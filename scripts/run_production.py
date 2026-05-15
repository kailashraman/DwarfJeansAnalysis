"""Single-galaxy production driver.

Given an LVDB key, this script runs the full Stage-1 inference:

  1. Load the staged per-star or per-epoch catalog from
     ``data/star_catalogs/<lvdb_key>.npz`` and route it through
     ``prepare_jeans_input`` (which dispatches to the per-paper
     combiner via ``_meta["source_paper_bibcode"]`` and applies
     ``SelectionPolicy``).
  2. Build the 7D nuisance-marginalized galaxy dict (V, sigma_eps, p,
     R, Rad_arcmin) plus nuisance priors derived directly from
     ``data/registry/galaxies.ecsv`` (the registry is the single
     source of truth for distance, rhalf, ellipticity and their
     uncertainties).
  3. Run dynesty with the requested base prior (default: Jeffreys
     conditional on (ln ρ_s, ln r_s)).
  4. Derive per-sample chains for σ_los(R_½,2D), M(R_½,2D),
     M(r_½,3D), J(α_c) + J(0.1°/0.2°/0.5°), D(α_c/2) + D at the same
     fixed angles.
  5. Dump samples + a summary CSV + the selection/combine audit JSON
     to ``results/production/<lvdb_key>/<prior>/`` (overwrites each run).

Usage:
    python scripts/run_production.py --lvdb-key tucana_2

Run ``--help`` for the full argparse interface.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jeans import inference as jeans_inference
from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.jeans.selection import SelectionPolicy
from dwarfjeans.jd import factors as jdf
from dwarfjeans.jeans.constant_sigma import constant_sigma_inference

ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
PLUMMER_3D_OVER_2D = 1.30477  # r_½(3D) / r_½(2D) for Plummer


# ----------------------------------------------------------------------------
# Registry helpers
# ----------------------------------------------------------------------------

def _read_registry_row(lvdb_key: str) -> dict:
    """Read a single row from data/registry/galaxies.ecsv. Returns a
    dict keyed by column name; values are floats for numeric columns,
    strings otherwise. Raises KeyError if the lvdb_key is not present.
    """
    import shlex
    ecsv = REPO / "data" / "registry" / "galaxies.ecsv"
    header: list[str] | None = None
    for line in ecsv.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        toks = shlex.split(line)
        if header is None:
            header = toks
            continue
        if toks[0] == lvdb_key:
            row = dict(zip(header, toks))
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            return row
    raise KeyError(f"{lvdb_key!r} not found in {ecsv}")


def _registry_nuisance_priors(row: dict) -> dict:
    """Build nuisance priors (Gaussian mean/sigma for d_kpc, ε,
    rhalf_arcmin) directly from the registry row. Distance-modulus
    uncertainty propagates to d_kpc as σ_d = (ln10/5) · d · σ_μ.
    Falls back to 5% of the central value if em/ep are missing.
    """
    def sym(val: float, em_key: str, ep_key: str) -> float:
        em = row.get(em_key)
        ep = row.get(ep_key)
        if not isinstance(em, (int, float)) or not isinstance(ep, (int, float)) \
                or np.isnan(em) or np.isnan(ep):
            return 0.05 * abs(val) if val != 0 else 0.05
        return (abs(float(em)) + abs(float(ep))) / 2.0

    d_kpc = float(row["distance_kpc"])
    mu = float(row["distance_modulus"])
    mu_sig = sym(mu, "distance_modulus_em", "distance_modulus_ep")
    d_sig = (np.log(10.0) / 5.0) * d_kpc * mu_sig

    rh_arcmin = float(row["rhalf_arcmin"])
    rh_sig = sym(rh_arcmin, "rhalf_arcmin_em", "rhalf_arcmin_ep")

    eps_raw = row.get("ellipticity", np.nan)
    eps_missing = isinstance(eps_raw, float) and np.isnan(eps_raw)
    eps = 0.0 if eps_missing else float(eps_raw)
    # eps_sigma is Gaussian σ even though the prior is truncated to [0,1).
    # For galaxies with only an upper-limit constraint (`ellipticity_ul`,
    # `ellipticity_missing=True`) the symmetric em/ep are NaN; widen the
    # fallback to ellipticity_ul/2 so the prior actually spans the
    # allowed range instead of collapsing to 5% of zero.
    eps_ul = row.get("ellipticity_ul", np.nan)
    if eps_missing and isinstance(eps_ul, float) and np.isfinite(eps_ul):
        eps_sig = float(eps_ul) / 2.0
    else:
        eps_sig = sym(eps if eps != 0.0 else 0.1,
                       "ellipticity_em", "ellipticity_ep")

    return {
        "d_mean":      d_kpc,    "d_sigma":     d_sig,
        "eps_mean":    eps,      "eps_sigma":   eps_sig,
        "rhalf_mean":  rh_arcmin, "rhalf_sigma": rh_sig,
    }


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def run(lvdb_key: str,
        *,
        prior_name: str = "jeffreys",
        nlive: int = 500,
        dlogz: float = 0.1,
        rseed: int = 0,
        p_min: float = 0.5,
        rmax_over_rhalf: float = 2.0,
        drop_variable: bool = True,
        use_p_weights: bool = False,
        thin_sigma: int = 2000,
        thin_jd: int = 500,
        thin_profile: int = 300,
        output_base: Path | None = None,
        npool: int = 1,
        ) -> Path:
    """Run the full pipeline for one galaxy. Returns the output dir."""
    if output_base is None:
        output_base = REPO / "results" / "production"
    # Canonical, single-output-per-(galaxy, prior). Each run overwrites
    # the previous one — wrong results are not preserved for provenance,
    # to keep the central results tree from ballooning over re-runs.
    out_dir = output_base / lvdb_key / prior_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    log: list[str] = []

    def logp(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.append(msg)

    t_total0 = time.time()
    logp(f"=== Production run: {lvdb_key} | prior={prior_name} ===")
    logp(f"out: {out_dir}")

    # ----- Registry row + nuisance priors -----
    row = _read_registry_row(lvdb_key)
    nuisance_priors = _registry_nuisance_priors(row)
    d_kpc = float(row["distance_kpc"])
    rhalf_major_pc = float(row["rhalf_major_pc"])
    V_center = float(row.get("vlos_systemic_kms", 0.0))
    if np.isnan(V_center):
        V_center = 0.0
        logp("WARN: registry vlos_systemic_kms NaN; using 0 km/s as V_center")

    # Per-galaxy override on the V prior halfwidth. Defaults to V_HALFWIDTH
    # (10 km/s) from priors.py; can be widened in the registry for small
    # samples or unusual velocity distributions (P&S 2018 footnote 6).
    from dwarfjeans.jeans.priors import V_HALFWIDTH as _V_HW_DEFAULT
    _vhw_raw = row.get("vlos_prior_halfwidth_kms", _V_HW_DEFAULT)
    V_halfwidth = (_V_HW_DEFAULT if (isinstance(_vhw_raw, float) and np.isnan(_vhw_raw))
                   else float(_vhw_raw))

    logp(f"\n=== Registry row ({lvdb_key}) ===")
    logp(f"  distance_kpc:    {d_kpc:.3f}")
    logp(f"  rhalf_major_pc:  {rhalf_major_pc:.3f}")
    logp(f"  V_halfwidth (km/s): {V_halfwidth:.2f}"
         + ("" if V_halfwidth == _V_HW_DEFAULT else " (registry override)"))
    logp(f"  spatial_model:   {row.get('spatial_model', 'plummer')!r}")
    logp(f"  source_paper:    vlos={row.get('ref_vlos', '?')}, "
         f"struct={row.get('ref_structure', '?')}")
    logp(f"  nuisance priors:")
    for k in ("d_mean", "d_sigma", "eps_mean", "eps_sigma",
              "rhalf_mean", "rhalf_sigma"):
        logp(f"    {k}: {nuisance_priors[k]:.4f}")

    # ----- Load + select -----
    catalog_path = REPO / "data" / "star_catalogs" / f"{lvdb_key}.npz"
    if not catalog_path.exists():
        raise FileNotFoundError(f"staged catalog missing: {catalog_path}")
    cat = np.load(catalog_path, allow_pickle=True)
    arrays, audit = prepare_jeans_input(
        cat,
        row,
        selection_policy=SelectionPolicy(
            p_min=p_min,
            R_over_rhalf_max=rmax_over_rhalf,
            drop_variable=drop_variable,
        ),
    )
    sel = audit.get("selection", {})
    logp(f"\n=== Selection ({catalog_path.name}) ===")
    logp(f"  policy: p_min={p_min}, R/rhalf_major<={rmax_over_rhalf}, "
         f"drop_variable={drop_variable}")
    logp(f"  n_input={sel.get('n_input', '?')} -> "
         f"n_after_p={sel.get('n_after_p', '?')} -> "
         f"n_after_R={sel.get('n_after_R', '?')} -> "
         f"n_after_var={sel.get('n_after_var', '?')} -> "
         f"n_final={sel.get('n_final', '?')}")
    if audit.get("combine"):
        cd = audit["combine"]
        logp(f"  combine: n_input_rows={cd.get('n_input_rows')}, "
             f"n_stars={cd.get('n_stars')}, n_variable={cd.get('n_variable')}, "
             f"sigma_sys={cd.get('sigma_sys_kms')}, "
             f"treatment={cd.get('sigma_sys_treatment', 'as_statistical')}, "
             f"offsets={cd.get('zero_point_offsets_kms', {})}")

    R_kpc = np.asarray(arrays["R"], dtype=float)
    R_kpc = np.clip(R_kpc, 1e-5, None)  # grid integrator floors u_min = 1e-4·R
    Rad_arcmin = R_kpc / d_kpc / ARCMIN_TO_RAD
    V = np.asarray(arrays["V"], dtype=float)
    sigma_eps = np.asarray(arrays["sigma_eps"], dtype=float)
    p = np.asarray(arrays["p"], dtype=float)
    if R_kpc.size == 0:
        raise RuntimeError(f"selection retained 0 stars for {lvdb_key}")
    logp(f"  N final = {R_kpc.size};  R/kpc range=[{R_kpc.min():.4f}, "
         f"{R_kpc.max():.4f}];  V mean/std={V.mean():+.2f}/{V.std():.2f} km/s")

    # Re-center the V_sys prior on the IVW mean of the post-selection
    # velocities. Registry vlos_systemic_kms can be stale / based on a
    # different sample (e.g. Pegasus III posterior railed against a
    # ±10 km/s registry-centered prior). The ±V_HALFWIDTH window is
    # set in priors.py and unchanged. Hard-floor σ_eps at 0.05 km/s:
    # spectroscopic σ_eps is realistically ≥ 0.1 km/s, so anything
    # smaller is a data bug and would dominate the weighted mean.
    if not np.all(sigma_eps > 0.05):
        raise ValueError(
            f"sigma_eps for {lvdb_key} contains values <= 0.05 km/s "
            f"(min={sigma_eps.min():.4g}); refusing to compute IVW "
            "V_center — investigate the catalog."
        )
    _w = 1.0 / sigma_eps ** 2
    V_center_ivw = float(np.sum(_w * V) / np.sum(_w))
    logp(f"  V_center (km/s): registry={V_center:+.2f} -> "
         f"post-selection IVW mean={V_center_ivw:+.2f} (used for prior)")
    V_center = V_center_ivw

    # Post-cut membership convention. The default (use_p_weights=False)
    # treats every survivor as a confirmed member (p_i := 1) — the
    # standard convention used by tests/integration/run_segue1.py and the
    # P&S 2018 reference numbers. Setting use_p_weights=True propagates
    # the catalog's continuous p_i into the likelihood (Walker+2006).
    p_raw = p.copy()
    if not use_p_weights:
        p = np.ones_like(p_raw)
        logp(f"  use_p_weights=False: post-cut p_i replaced by 1.0 in the likelihood")
    else:
        logp(f"  use_p_weights=True: continuous p_i propagated into the likelihood")

    # r_p(2D) at registry-fiducial geometry. The 7D nuisance-marginalized
    # likelihood does NOT consume `truth.r_p` (per_sample r_p is built
    # internally from d × rhalf_arcmin × √(1−ε) at each draw). It is
    # passed in only as a defensive fallback for the 4D / Asimov code
    # paths in run_inference; harmless for the 7D path used here.
    _eps_raw = row.get("ellipticity", np.nan)
    eps_fid = 0.0 if (isinstance(_eps_raw, float) and np.isnan(_eps_raw)) \
        else float(_eps_raw)
    r_p_kpc_fid = (d_kpc * float(row["rhalf_arcmin"]) * ARCMIN_TO_RAD
                    * np.sqrt(max(0.0, 1.0 - eps_fid)))

    galaxy = {
        "R": R_kpc,
        "Rad_arcmin": Rad_arcmin,
        "V": V,
        "sigma_eps": sigma_eps,
        "p": p,
        "truth": {"r_p": r_p_kpc_fid},
    }

    # ----- Perspective-motion marginalisation setup -----
    # If prepare_jeans_input already applied the central-value correction
    # (audit["perspective"]["applied"] == True), we hand V_observed + RA/Dec
    # to the 9D likelihood so it re-samples Δv at each (μ_α*, μ_δ) draw.
    # PM split-normal priors come from the LVDB registry columns.
    persp_meta = audit.get("perspective", {})
    perspective_kwargs = {}
    if persp_meta.get("applied"):
        # Fail loudly if the invariant breaks: prepare_jeans_input must
        # stash V_observed whenever it sets applied=True. Falling through
        # to the 7D path silently here would drop PM marginalisation
        # without warning.
        if "V_observed" not in arrays:
            raise RuntimeError(
                "prepare_jeans_input audit reports perspective.applied=True "
                "but arrays['V_observed'] is missing; refusing to silently "
                "drop PM marginalisation."
            )
        V_observed = np.asarray(arrays["V_observed"], dtype=float)
        ra_star = np.asarray(arrays["RA_star"], dtype=float)
        dec_star = np.asarray(arrays["Dec_star"], dtype=float)
        perspective_kwargs = dict(
            perspective={
                "V_observed": V_observed,
                "RA_star": ra_star, "Dec_star": dec_star,
                "ra_center": float(row["ra_deg"]),
                "dec_center": float(row["dec_deg"]),
            },
            pm_prior={
                "pmra_mean": float(row["pmra_mas_yr"]),
                "pmra_em":   float(row["pmra_em_mas_yr"]),
                "pmra_ep":   float(row["pmra_ep_mas_yr"]),
                "pmdec_mean": float(row["pmdec_mas_yr"]),
                "pmdec_em":   float(row["pmdec_em_mas_yr"]),
                "pmdec_ep":   float(row["pmdec_ep_mas_yr"]),
            },
        )
        logp(f"  PM marginalisation: μ_α* = {row['pmra_mas_yr']:+.3f} "
             f"+{row['pmra_ep_mas_yr']:.3f}/-{row['pmra_em_mas_yr']:.3f}, "
             f"μ_δ = {row['pmdec_mas_yr']:+.3f} "
             f"+{row['pmdec_ep_mas_yr']:.3f}/-{row['pmdec_em_mas_yr']:.3f} mas/yr")
    else:
        logp(f"  PM marginalisation: skipped ({persp_meta.get('reason', 'no perspective audit')})")

    # ----- Walker+2006 constant-σ dispersion (data-only, model-free) -----
    # The σ_los Walker baseline accepts {uniform, loguniform, jeffreys};
    # the (r_s, ρ_s) `satgen` prior has no σ_los counterpart, so fall
    # back to the production-default `jeffreys` σ prior in that case.
    sigma_prior_name = "jeffreys" if prior_name == "satgen" else prior_name
    cs = constant_sigma_inference(V, sigma_eps, p, V_center=V_center,
                                   V_halfwidth=V_halfwidth,
                                   prior=sigma_prior_name)
    sigma_los_walker = cs["sigma_int"]                 # (V̄, σ) joint marginal
    logp(f"\n=== Constant-σ inference (Walker+2006, radius-independent, "
         f"prior={sigma_prior_name}) ===")
    logp(f"  σ_los (Bayes,  median): {sigma_los_walker['median']:.3f} "
         f"[{sigma_los_walker['q16']:.3f}, {sigma_los_walker['q84']:.3f}] km/s")

    # ----- Inference -----
    ndim = 9 if perspective_kwargs else 7
    logp(f"\n=== dynesty ({ndim}D, prior={prior_name}, nlive={nlive}, "
         f"dlogz={dlogz}, npool={npool}) ===")
    t_inf = time.time()
    result = jeans_inference.run_inference(
        galaxy,
        V_center=V_center,
        V_halfwidth=V_halfwidth,
        nlive=nlive,
        dlogz=dlogz,
        rseed=rseed,
        print_progress=False,
        marginalize_nuisances=True,
        nuisance_priors=nuisance_priors,
        prior_name=prior_name,
        npool=npool,
        **perspective_kwargs,
    )
    dt_inf = time.time() - t_inf
    logp(f"  done in {dt_inf:.1f}s")
    logp(f"  logZ = {result['logz']:.3f} ± {result['logz_err']:.3f}")
    logp(f"  n_eq = {result['n_eq']}")

    samples_eq = result["samples_eq"]
    if perspective_kwargs:
        (V_chain, lr_chain, lp_chain, btilde_chain, d_chain, eps_chain, rhalf_arcmin_chain,
         pmra_chain, pmdec_chain) = samples_eq.T
    else:
        V_chain, lr_chain, lp_chain, btilde_chain, d_chain, eps_chain, rhalf_arcmin_chain = (
            samples_eq.T)
        pmra_chain = pmdec_chain = None
    r_s_chain = 10.0 ** lr_chain
    rho_s_chain = 10.0 ** lp_chain
    beta_chain = jeans_inference.beta_tilde_to_beta(btilde_chain)
    r_p_chain = (d_chain * rhalf_arcmin_chain * ARCMIN_TO_RAD
                  * np.sqrt(np.clip(1.0 - eps_chain, 0.0, None)))
    R_half_2d_chain = r_p_chain   # Plummer
    r_half_3d_chain = PLUMMER_3D_OVER_2D * r_p_chain

    # ----- Derived posteriors -----
    log10_M_2d = np.log10(jeans.nfw_M(R_half_2d_chain, r_s_chain, rho_s_chain))
    log10_M_3d = np.log10(jeans.nfw_M(r_half_3d_chain, r_s_chain, rho_s_chain))

    rng = np.random.default_rng(rseed)
    N_chain = samples_eq.shape[0]

    def _thin(n: int) -> np.ndarray:
        if N_chain > n:
            return rng.choice(N_chain, size=n, replace=False)
        return np.arange(N_chain)

    # σ_los at R_½,2D — per-sample (loop, sigma_los takes scalars)
    idx_sig = _thin(thin_sigma)
    sigma_at_Rhalf = np.full(idx_sig.size, np.nan)
    for j, i in enumerate(idx_sig):
        try:
            s = jeans.sigma_los(np.array([R_half_2d_chain[i]]),
                                  beta_chain[i], r_s_chain[i],
                                  rho_s_chain[i], r_p_chain[i],
                                  method="grid")
            sigma_at_Rhalf[j] = float(s[0])
        except Exception:
            pass

    # σ_los radial profile (16/50/84 percentile bands across samples)
    d_med = float(np.median(d_chain))
    R_kpc_med = d_med * Rad_arcmin * ARCMIN_TO_RAD
    R_grid = np.geomspace(max(R_kpc_med.min(), 1e-3), R_kpc_med.max(), 30)
    idx_prof = _thin(thin_profile)
    sigma_profile = np.full((idx_prof.size, R_grid.size), np.nan)
    for j, i in enumerate(idx_prof):
        try:
            sigma_profile[j] = jeans.sigma_los(R_grid, beta_chain[i],
                                                  r_s_chain[i], rho_s_chain[i],
                                                  r_p_chain[i], method="grid")
        except Exception:
            pass
    sig_q16 = np.nanpercentile(sigma_profile, 16, axis=0)
    sig_q50 = np.nanpercentile(sigma_profile, 50, axis=0)
    sig_q84 = np.nanpercentile(sigma_profile, 84, axis=0)
    n_sigma_nan = int(np.isnan(sigma_at_Rhalf).sum())
    n_profile_nan = int(np.isnan(sigma_profile).all(axis=1).sum())
    if n_sigma_nan or n_profile_nan:
        logp(f"  sigma_los failures: at-R_½ {n_sigma_nan}/{idx_sig.size}, "
             f"profile {n_profile_nan}/{idx_prof.size} (filled NaN)")

    # J/D chains at fixed angles + α_c (resp. α_c/2 for D)
    r_t_kpc = 1.0  # truncation; matches stage3.md mock-test default
    alpha_c_chain = jdf.alpha_c_radians(r_half_3d_chain, d_chain)
    fixed_J_angles = {"0p1deg": 0.1 * jdf.DEG,
                       "0p2deg": 0.2 * jdf.DEG,
                       "0p5deg": 0.5 * jdf.DEG}
    fixed_D_angles = dict(fixed_J_angles)
    idx_jd = _thin(thin_jd)
    log10_J = {tag: np.full(idx_jd.size, np.nan)
                for tag in (*fixed_J_angles, "alphac")}
    log10_D = {tag: np.full(idx_jd.size, np.nan)
                for tag in (*fixed_D_angles, "alphacover2")}
    logp(f"\n=== J/D integrals (thin {idx_jd.size}/{N_chain}, r_t={r_t_kpc} kpc) ===")
    logp(f"  alpha_c (median): {float(np.median(alpha_c_chain)):.5f} rad "
         f"({float(np.median(alpha_c_chain))/jdf.DEG:.4f} deg)")
    t_jd = time.time()
    for j, i in enumerate(idx_jd):
        rs_i, rhos_i = float(r_s_chain[i]), float(rho_s_chain[i])
        d_i = float(d_chain[i])
        ac_i = float(alpha_c_chain[i])
        for tag, th in fixed_J_angles.items():
            J, _ = jdf.J_D_factors(th, d_i, rs_i, rhos_i, r_t_kpc)
            log10_J[tag][j] = (np.log10(J) + jdf.LOG10_J_FAC) if J > 0 else np.nan
        J_ac, _ = jdf.J_D_factors(ac_i, d_i, rs_i, rhos_i, r_t_kpc)
        log10_J["alphac"][j] = (np.log10(J_ac) + jdf.LOG10_J_FAC) if J_ac > 0 else np.nan
        for tag, th in fixed_D_angles.items():
            _, D = jdf.J_D_factors(th, d_i, rs_i, rhos_i, r_t_kpc)
            log10_D[tag][j] = (np.log10(D) + jdf.LOG10_D_FAC) if D > 0 else np.nan
        _, D_aco2 = jdf.J_D_factors(0.5 * ac_i, d_i, rs_i, rhos_i, r_t_kpc)
        log10_D["alphacover2"][j] = (np.log10(D_aco2) + jdf.LOG10_D_FAC) if D_aco2 > 0 else np.nan
    logp(f"  done in {time.time()-t_jd:.1f}s")

    # ----- Save outputs -----
    np.savez(
        out_dir / "posterior_samples.npz",
        samples_eq=samples_eq,
        V=V_chain, log10_rs=lr_chain, log10_rhos=lp_chain,
        beta_tilde=btilde_chain, beta=beta_chain,
        d_kpc_chain=d_chain, eps_chain=eps_chain,
        rhalf_arcmin_chain=rhalf_arcmin_chain,
        r_p_chain=r_p_chain, R_half_2d_chain=R_half_2d_chain,
        r_half_3d_chain=r_half_3d_chain, alpha_c_chain=alpha_c_chain,
        log10_M_half_2d=log10_M_2d, log10_M_half_3d=log10_M_3d,
        sigma_los_at_Rhalf2d=sigma_at_Rhalf,
        R_grid_sigma_profile=R_grid,
        sigma_profile_q16=sig_q16, sigma_profile_q50=sig_q50,
        sigma_profile_q84=sig_q84,
        **{f"log10_J_{tag}": v for tag, v in log10_J.items()},
        **{f"log10_D_{tag}": v for tag, v in log10_D.items()},
        r_t_kpc=r_t_kpc,
        idx_jd=idx_jd, idx_sig=idx_sig, idx_prof=idx_prof,
    )

    # Summary CSV — one quantity per row, q16/q50/q84
    def _q(arr: np.ndarray) -> tuple[float, float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan)
        return tuple(np.percentile(a, [16, 50, 84]).tolist())

    summary_rows = [
        ("V_sys_kms",            *_q(V_chain)),
        ("log10_rs_kpc",         *_q(lr_chain)),
        ("log10_rhos_Msun_kpc3", *_q(lp_chain)),
        ("beta",                 *_q(beta_chain)),
        ("d_kpc",                *_q(d_chain)),
        ("eps",                  *_q(eps_chain)),
        ("rhalf_arcmin",         *_q(rhalf_arcmin_chain)),
        ("r_p_kpc",              *_q(r_p_chain)),
        ("R_half_2d_kpc",        *_q(R_half_2d_chain)),
        ("r_half_3d_kpc",        *_q(r_half_3d_chain)),
        ("log10_M_half_2d_Msun", *_q(log10_M_2d)),
        ("log10_M_half_3d_Msun", *_q(log10_M_3d)),
        ("sigma_los_at_Rhalf2d_kms", *_q(sigma_at_Rhalf)),
        ("sigma_los_walker_kms",
            sigma_los_walker["q16"], sigma_los_walker["median"], sigma_los_walker["q84"]),
        ("alpha_c_rad",          *_q(alpha_c_chain)),
    ]
    if pmra_chain is not None:
        summary_rows.append(("pmra_mas_yr",  *_q(pmra_chain)))
        summary_rows.append(("pmdec_mas_yr", *_q(pmdec_chain)))
    for tag in (*fixed_J_angles, "alphac"):
        summary_rows.append((f"log10_J_{tag}_GeV2_cm5", *_q(log10_J[tag])))
    for tag in (*fixed_D_angles, "alphacover2"):
        summary_rows.append((f"log10_D_{tag}_GeV_cm2", *_q(log10_D[tag])))

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w") as f:
        f.write("quantity,q16,q50,q84\n")
        for name, q16, q50, q84 in summary_rows:
            f.write(f"{name},{q16:.6g},{q50:.6g},{q84:.6g}\n")

    audit_payload = {
        "lvdb_key": lvdb_key,
        "prior_name": prior_name,
        "timestamp_utc": timestamp,
        "registry_row": {k: (v if not isinstance(v, float)
                              or np.isfinite(v) else None)
                          for k, v in row.items()},
        "selection_policy": {
            "p_min": p_min,
            "R_over_rhalf_max": rmax_over_rhalf,
            "drop_variable": drop_variable,
        },
        "nuisance_priors": nuisance_priors,
        "prepare_jeans_input_audit": audit,
        "dynesty": {
            "nlive": nlive, "dlogz": dlogz, "rseed": rseed,
            "logZ": float(result["logz"]),
            "logZ_err": float(result["logz_err"]),
            "n_eq": int(result["n_eq"]),
            "wallclock_s": dt_inf,
        },
        "thinning": {"sigma": int(idx_sig.size),
                      "profile": int(idx_prof.size),
                      "jd": int(idx_jd.size)},
        "r_t_kpc": r_t_kpc,
    }
    (out_dir / "audit.json").write_text(json.dumps(audit_payload,
                                                       indent=2,
                                                       default=str))

    (out_dir / "run.log").write_text("\n".join(log) + "\n")
    logp(f"\n=== Wrote outputs to {out_dir} (total {time.time()-t_total0:.1f}s) ===")
    return out_dir


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--lvdb-key", required=True,
                   help="Galaxy key in data/registry/galaxies.ecsv")
    p.add_argument("--prior", default="jeffreys",
                   choices=("uniform", "loguniform", "jeffreys", "satgen", "satgen_box"),
                   help="Base halo prior on (ln ρ_s, ln r_s)")
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--rseed", type=int, default=0)
    p.add_argument("--p-min", type=float, default=0.5)
    p.add_argument("--rmax-over-rhalf", type=float, default=2.0)
    p.add_argument("--use-p-weights", action="store_true",
                   help="Propagate continuous p_i into the likelihood instead of "
                        "replacing post-cut survivors with p=1 (default).")
    p.add_argument("--keep-variable", action="store_true",
                   help="Disable the variability/χ² drop in selection")
    p.add_argument("--thin-sigma", type=int, default=2000,
                   help="Posterior thin for σ_los at R_½,2D")
    p.add_argument("--thin-profile", type=int, default=300,
                   help="Posterior thin for σ_los radial profile bands")
    p.add_argument("--thin-jd", type=int, default=500,
                   help="Posterior thin for J/D integrals")
    p.add_argument("--output-base", default=None,
                   help="Override results/production")
    p.add_argument("--npool", type=int, default=1,
                   help="Multiprocessing pool size for dynesty likelihood "
                        "evaluations (default 1 = serial). Set to match "
                        "--cpus-per-task on SLURM. Pool order is non-deterministic, "
                        "so posterior medians shift at the ~1%% level vs --npool 1.")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    out = run(
        args.lvdb_key,
        prior_name=args.prior,
        nlive=args.nlive,
        dlogz=args.dlogz,
        rseed=args.rseed,
        p_min=args.p_min,
        rmax_over_rhalf=args.rmax_over_rhalf,
        drop_variable=not args.keep_variable,
        use_p_weights=args.use_p_weights,
        thin_sigma=args.thin_sigma,
        thin_profile=args.thin_profile,
        thin_jd=args.thin_jd,
        output_base=Path(args.output_base) if args.output_base else None,
        npool=args.npool,
    )
    print(out)
