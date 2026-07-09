"""Post-inference derived quantities for the dwarf Jeans pipeline.

Everything downstream of dynesty lives here: the (sampler-free) data
preparation -- star selection, V-centering, the Walker+2006 constant-sigma
baseline -- and the chain-derived quantities -- M(r_half), sigma_los, the
J/D-factors, the tidal radius r_t, and the 95% J-factor containment angle
theta95.

Design: the equal-weight posterior chain is the single source of truth.
Derived quantities are NOT persisted; they are recomputed on demand from
(chain + the frozen star catalog + the registry). A change to any derived
quantity therefore backfills onto existing runs without re-running the
sampler. Two entry points share this module: scripts/run_production.py (after
inference) and scripts/reprocess.py (regenerate views from a saved chain).

Thinning of the per-draw loops (J/D, sigma) is reproduced deterministically
from the stored `rseed`, so the thinned subsets are identical across
re-derivations without persisting any index arrays.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dwarfjeans.jeans import inference as jeans_inference
from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jeans.constant_sigma import constant_sigma_inference
from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.jeans.selection import SelectionPolicy
from dwarfjeans.jd import factors as jdf
from dwarfjeans.jd import tidal as jdtidal

# src/dwarfjeans/postprocess.py -> repo root
REPO = Path(__file__).resolve().parents[2]
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
PLUMMER_3D_OVER_2D = 1.30477  # r_half(3D) / r_half(2D) for Plummer

# Equal-weight chain column layout (the npz `samples_eq` columns).
PARAM_NAMES_7 = ["V", "log10_rs", "log10_rhos", "beta_tilde",
                 "d_kpc", "eps", "rhalf_arcmin"]
PARAM_NAMES_9 = PARAM_NAMES_7 + ["pmra", "pmdec"]


def _noop(*_a):
    pass


# ----------------------------------------------------------------------------
# Registry helpers (single home; run_production imports these)
# ----------------------------------------------------------------------------

def read_registry_row(lvdb_key: str) -> dict:
    """Read a single row from data/registry/galaxies.ecsv. Returns a dict
    keyed by column name; values are floats for numeric columns, strings
    otherwise. Raises KeyError if the lvdb_key is not present.
    """
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


def registry_nuisance_priors(row: dict) -> dict:
    """Build nuisance priors (Gaussian mean/sigma for d_kpc, eps,
    rhalf_arcmin) directly from the registry row. Distance-modulus
    uncertainty propagates to d_kpc as sigma_d = (ln10/5) * d * sigma_mu.
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
    eps_ul = row.get("ellipticity_ul", np.nan)
    if eps_missing and isinstance(eps_ul, float) and np.isfinite(eps_ul):
        eps_sig = float(eps_ul) / 2.0
    else:
        eps_sig = sym(eps if eps != 0.0 else 0.1,
                      "ellipticity_em", "ellipticity_ep")

    return {
        "d_mean":      d_kpc,     "d_sigma":     d_sig,
        "eps_mean":    eps,       "eps_sigma":   eps_sig,
        "rhalf_mean":  rh_arcmin, "rhalf_sigma": rh_sig,
    }


# ----------------------------------------------------------------------------
# Galaxy context (sampler-free data prep, shared by inference + derivation)
# ----------------------------------------------------------------------------

@dataclass
class GalaxyContext:
    """Everything downstream of inference needs that does NOT depend on the
    posterior chain: the registry row, the selected stars + their geometry,
    the V prior centering, the Walker baseline, and the PM-marginalisation
    setup. Built by :func:`prepare` from (lvdb_key + selection config)."""

    lvdb_key: str
    row: dict
    nuisance_priors: dict
    d_kpc: float
    V_center: float
    V_halfwidth: float
    galaxy: dict
    sigma_los_walker: dict
    perspective_kwargs: dict
    selection_audit: dict
    Rad_arcmin: np.ndarray

    @property
    def has_pm(self) -> bool:
        return bool(self.perspective_kwargs)

    @property
    def ra_deg(self) -> float:
        return float(self.row["ra_deg"])

    @property
    def dec_deg(self) -> float:
        return float(self.row["dec_deg"])


def prepare(lvdb_key: str, *,
            prior_name: str = "jeffreys",
            p_min: float = 0.5,
            rmax_over_rhalf: float = 2.0,
            drop_variable: bool = True,
            use_p_weights: bool = False,
            logp=_noop) -> GalaxyContext:
    """Sampler-free preparation: registry row, star selection, V-centering,
    the Walker baseline, and PM-marginalisation setup. Returns a
    :class:`GalaxyContext` consumed by both inference and :func:`derive`.
    """
    row = read_registry_row(lvdb_key)
    nuisance_priors = registry_nuisance_priors(row)
    d_kpc = float(row["distance_kpc"])
    rhalf_major_pc = float(row["rhalf_major_pc"])
    V_center = float(row.get("vlos_systemic_kms", 0.0))
    if np.isnan(V_center):
        V_center = 0.0
        logp("WARN: registry vlos_systemic_kms NaN; using 0 km/s as V_center")

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

    R_kpc = np.asarray(arrays["R"], dtype=float)
    R_kpc = np.clip(R_kpc, 1e-5, None)
    Rad_arcmin = R_kpc / d_kpc / ARCMIN_TO_RAD
    V = np.asarray(arrays["V"], dtype=float)
    sigma_eps = np.asarray(arrays["sigma_eps"], dtype=float)
    p = np.asarray(arrays["p"], dtype=float)
    if R_kpc.size == 0:
        raise RuntimeError(f"selection retained 0 stars for {lvdb_key}")
    logp(f"  N final = {R_kpc.size};  R/kpc range=[{R_kpc.min():.4f}, "
         f"{R_kpc.max():.4f}];  V mean/std={V.mean():+.2f}/{V.std():.2f} km/s")

    # Re-center the V_sys prior on the IVW mean of the post-selection velocities.
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

    # Post-cut membership convention.
    p_raw = p.copy()
    if not use_p_weights:
        p = np.ones_like(p_raw)
        logp("  use_p_weights=False: post-cut p_i replaced by 1.0 in the likelihood")
    else:
        logp("  use_p_weights=True: continuous p_i propagated into the likelihood")

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
    persp_meta = audit.get("perspective", {})
    perspective_kwargs: dict = {}
    if persp_meta.get("applied"):
        if "V_observed" not in arrays:
            raise RuntimeError(
                "prepare_jeans_input audit reports perspective.applied=True "
                "but arrays['V_observed'] is missing; refusing to silently "
                "drop PM marginalisation."
            )
        perspective_kwargs = dict(
            perspective={
                "V_observed": np.asarray(arrays["V_observed"], dtype=float),
                "RA_star": np.asarray(arrays["RA_star"], dtype=float),
                "Dec_star": np.asarray(arrays["Dec_star"], dtype=float),
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

    # ----- Walker+2006 constant-sigma baseline (data-only, sampler-free) -----
    sigma_prior_name = (
        "jeffreys" if prior_name in ("satgen", "satgen_box", "satgen_shmr")
        else prior_name
    )
    cs = constant_sigma_inference(V, sigma_eps, p, V_center=V_center,
                                  V_halfwidth=V_halfwidth,
                                  prior=sigma_prior_name)
    sigma_los_walker = cs["sigma_int"]
    logp(f"\n=== Constant-σ inference (Walker+2006, radius-independent, "
         f"prior={sigma_prior_name}) ===")
    logp(f"  σ_los (Bayes,  median): {sigma_los_walker['median']:.3f} "
         f"[{sigma_los_walker['q16']:.3f}, {sigma_los_walker['q84']:.3f}] km/s")

    return GalaxyContext(
        lvdb_key=lvdb_key, row=row, nuisance_priors=nuisance_priors,
        d_kpc=d_kpc, V_center=V_center, V_halfwidth=V_halfwidth,
        galaxy=galaxy, sigma_los_walker=sigma_los_walker,
        perspective_kwargs=perspective_kwargs, selection_audit=audit,
        Rad_arcmin=Rad_arcmin,
    )


# ----------------------------------------------------------------------------
# Chain-derived quantities
# ----------------------------------------------------------------------------

def derive(samples_eq, ctx: GalaxyContext, *,
           rseed: int,
           thin_sigma: int = 2000,
           thin_jd: int = 500,
           thin_profile: int = 300,
           host=jdtidal.DEFAULT_HOST,
           tidal_factor: float = 2.0,
           logp=_noop) -> dict:
    """Compute all chain-derived posteriors from the equal-weight chain.

    Returns a dict of arrays/values: the unpacked parameter chains, M(r_half),
    sigma_los at R_half and its radial-profile bands, the J/D-factor chains at
    the reporting angles, the per-draw tidal radius and galactocentric
    distance, and the 95% J containment angle. Per-draw loops (sigma, J/D) are
    thinned with a fresh ``default_rng(rseed)`` so the subsets are reproducible.
    """
    samples_eq = np.asarray(samples_eq, dtype=float)
    n_expected = len(PARAM_NAMES_9 if ctx.has_pm else PARAM_NAMES_7)
    if samples_eq.shape[1] != n_expected:
        raise ValueError(
            f"chain has {samples_eq.shape[1]} columns but ctx.has_pm="
            f"{ctx.has_pm} expects {n_expected}. The prepared context "
            f"(PM={ctx.has_pm}) disagrees with the saved chain — the "
            "registry PM columns likely changed since the run."
        )
    if ctx.has_pm:
        (V_chain, lr_chain, lp_chain, btilde_chain, d_chain, eps_chain,
         rhalf_arcmin_chain, pmra_chain, pmdec_chain) = samples_eq.T
    else:
        (V_chain, lr_chain, lp_chain, btilde_chain, d_chain, eps_chain,
         rhalf_arcmin_chain) = samples_eq.T
        pmra_chain = pmdec_chain = None

    r_s_chain = 10.0 ** lr_chain
    rho_s_chain = 10.0 ** lp_chain
    beta_chain = jeans_inference.beta_tilde_to_beta(btilde_chain)
    r_p_chain = (d_chain * rhalf_arcmin_chain * ARCMIN_TO_RAD
                 * np.sqrt(np.clip(1.0 - eps_chain, 0.0, None)))
    R_half_2d_chain = r_p_chain   # Plummer
    r_half_3d_chain = PLUMMER_3D_OVER_2D * r_p_chain

    log10_M_2d = np.log10(jeans.nfw_M(R_half_2d_chain, r_s_chain, rho_s_chain))
    log10_M_3d = np.log10(jeans.nfw_M(r_half_3d_chain, r_s_chain, rho_s_chain))

    rng = np.random.default_rng(rseed)
    N_chain = samples_eq.shape[0]

    def _thin(n: int) -> np.ndarray:
        if N_chain > n:
            return rng.choice(N_chain, size=n, replace=False)
        return np.arange(N_chain)

    # sigma_los at R_half,2D
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

    # sigma_los radial profile (16/50/84 bands across samples)
    d_med = float(np.median(d_chain))
    R_kpc_med = d_med * ctx.Rad_arcmin * ARCMIN_TO_RAD
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

    # J/D chains + per-draw tidal radius + 95% J containment angle.
    alpha_c_chain = jdf.alpha_c_radians(r_half_3d_chain, d_chain)
    idx_jd = _thin(thin_jd)
    R_GC_chain = jdtidal.galactocentric_distance(
        ctx.ra_deg, ctx.dec_deg, d_chain[idx_jd])
    fixed_J_angles = {"0p1deg": 0.1 * jdf.DEG,
                      "0p2deg": 0.2 * jdf.DEG,
                      "0p5deg": 0.5 * jdf.DEG}
    fixed_D_angles = dict(fixed_J_angles)
    log10_J = {tag: np.full(idx_jd.size, np.nan)
               for tag in (*fixed_J_angles, "alphac")}
    log10_D = {tag: np.full(idx_jd.size, np.nan)
               for tag in (*fixed_D_angles, "alphacover2")}
    r_t_chain = np.full(idx_jd.size, np.nan)
    theta95_J = np.full(idx_jd.size, np.nan)
    theta95_D = np.full(idx_jd.size, np.nan)
    # Exact-geometry TOTAL factors within r_t (= J/D at theta=pi/2; the small-angle
    # approximation is invalid at the full angular extent). j_/d_containment
    # already return these in GeV units, so log10 is taken directly -- NOT with
    # a +LOG10_*_FAC offset like the small-angle fixed-angle log10_J/log10_D.
    log10_J_pi2 = np.full(idx_jd.size, np.nan)
    log10_D_pi2 = np.full(idx_jd.size, np.nan)
    logp(f"\n=== J/D integrals (thin {idx_jd.size}/{N_chain}, "
         f"per-draw r_t in {type(host).__name__} host) ===")
    logp(f"  alpha_c (median): {float(np.median(alpha_c_chain)):.5f} rad "
         f"({float(np.median(alpha_c_chain))/jdf.DEG:.4f} deg)")
    logp(f"  D_GC (median): {float(np.median(R_GC_chain)):.2f} kpc")
    for j, i in enumerate(idx_jd):
        rs_i, rhos_i = float(r_s_chain[i]), float(rho_s_chain[i])
        d_i = float(d_chain[i])
        ac_i = float(alpha_c_chain[i])
        r_t_i = jdtidal.tidal_radius(rs_i, rhos_i, float(R_GC_chain[j]),
                                     host=host, factor=tidal_factor)
        r_t_chain[j] = r_t_i
        if not np.isfinite(r_t_i):
            continue
        for tag, th in fixed_J_angles.items():
            J, _ = jdf.J_D_factors(th, d_i, rs_i, rhos_i, r_t_i)
            log10_J[tag][j] = (np.log10(J) + jdf.LOG10_J_FAC) if J > 0 else np.nan
        J_ac, _ = jdf.J_D_factors(ac_i, d_i, rs_i, rhos_i, r_t_i)
        log10_J["alphac"][j] = (np.log10(J_ac) + jdf.LOG10_J_FAC) if J_ac > 0 else np.nan
        for tag, th in fixed_D_angles.items():
            _, D = jdf.J_D_factors(th, d_i, rs_i, rhos_i, r_t_i)
            log10_D[tag][j] = (np.log10(D) + jdf.LOG10_D_FAC) if D > 0 else np.nan
        _, D_aco2 = jdf.J_D_factors(0.5 * ac_i, d_i, rs_i, rhos_i, r_t_i)
        log10_D["alphacover2"][j] = (np.log10(D_aco2) + jdf.LOG10_D_FAC) if D_aco2 > 0 else np.nan
        _, J_pi2_i, th95 = jdf.j_aperture_and_containment(
            d_i, rs_i, rhos_i, r_t_i, aperture_rad=0.5 * jdf.DEG, frac=0.95)
        theta95_J[j] = th95
        log10_J_pi2[j] = np.log10(J_pi2_i) if (np.isfinite(J_pi2_i) and J_pi2_i > 0) else np.nan
        _, D_pi2_i, th95_D = jdf.d_aperture_and_containment(
            d_i, rs_i, rhos_i, r_t_i, aperture_rad=0.5 * jdf.DEG, frac=0.95)
        theta95_D[j] = th95_D
        log10_D_pi2[j] = np.log10(D_pi2_i) if (np.isfinite(D_pi2_i) and D_pi2_i > 0) else np.nan
    n_rt_nan = int(np.isnan(r_t_chain).sum())
    if n_rt_nan:
        logp(f"  r_t failures: {n_rt_nan}/{idx_jd.size} (filled NaN)")
    logp(f"  r_t (median): {float(np.nanmedian(r_t_chain)):.3f} kpc; "
         f"theta95_J (median): {float(np.nanmedian(theta95_J))/jdf.DEG:.4f} deg; "
         f"theta95_D (median): {float(np.nanmedian(theta95_D))/jdf.DEG:.4f} deg")

    return {
        "V_chain": V_chain, "lr_chain": lr_chain, "lp_chain": lp_chain,
        "btilde_chain": btilde_chain, "beta_chain": beta_chain,
        "d_chain": d_chain, "eps_chain": eps_chain,
        "rhalf_arcmin_chain": rhalf_arcmin_chain,
        "pmra_chain": pmra_chain, "pmdec_chain": pmdec_chain,
        "r_s_chain": r_s_chain, "rho_s_chain": rho_s_chain,
        "r_p_chain": r_p_chain, "R_half_2d_chain": R_half_2d_chain,
        "r_half_3d_chain": r_half_3d_chain, "alpha_c_chain": alpha_c_chain,
        "log10_M_2d": log10_M_2d, "log10_M_3d": log10_M_3d,
        "sigma_at_Rhalf": sigma_at_Rhalf,
        "R_grid": R_grid, "sigma_profile_q16": sig_q16,
        "sigma_profile_q50": sig_q50, "sigma_profile_q84": sig_q84,
        "log10_J": log10_J, "log10_D": log10_D,
        "log10_J_pi2": log10_J_pi2, "log10_D_pi2": log10_D_pi2,
        "r_t_chain": r_t_chain, "R_GC_chain": R_GC_chain,
        "theta95_J": theta95_J, "theta95_D": theta95_D,
        "n_jd": int(idx_jd.size), "n_sigma": int(idx_sig.size),
        "n_profile": int(idx_prof.size),
        "idx_jd": idx_jd, "idx_sig": idx_sig, "idx_prof": idx_prof,
    }


# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------

def _q(arr) -> tuple[float, float, float]:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(a, [16, 50, 84]).tolist())


def summary_rows(d: dict, ctx: GalaxyContext) -> list[tuple]:
    """Build the (quantity, q16, q50, q84) rows for summary.csv."""
    fixed_J = ("0p1deg", "0p2deg", "0p5deg", "alphac")
    fixed_D = ("0p1deg", "0p2deg", "0p5deg", "alphacover2")
    walker = ctx.sigma_los_walker
    rows = [
        ("V_sys_kms",            *_q(d["V_chain"])),
        ("log10_rs_kpc",         *_q(d["lr_chain"])),
        ("log10_rhos_Msun_kpc3", *_q(d["lp_chain"])),
        ("beta",                 *_q(d["beta_chain"])),
        ("d_kpc",                *_q(d["d_chain"])),
        ("eps",                  *_q(d["eps_chain"])),
        ("rhalf_arcmin",         *_q(d["rhalf_arcmin_chain"])),
        ("r_p_kpc",              *_q(d["r_p_chain"])),
        ("R_half_2d_kpc",        *_q(d["R_half_2d_chain"])),
        ("r_half_3d_kpc",        *_q(d["r_half_3d_chain"])),
        ("log10_M_half_2d_Msun", *_q(d["log10_M_2d"])),
        ("log10_M_half_3d_Msun", *_q(d["log10_M_3d"])),
        ("sigma_los_at_Rhalf2d_kms", *_q(d["sigma_at_Rhalf"])),
        ("sigma_los_walker_kms", walker["q16"], walker["median"], walker["q84"]),
        ("alpha_c_rad",          *_q(d["alpha_c_chain"])),
        ("R_GC_kpc",             *_q(d["R_GC_chain"])),
        ("r_t_kpc",              *_q(d["r_t_chain"])),
        ("theta95_J_deg",        *_q(d["theta95_J"] / jdf.DEG)),
        ("theta95_D_deg",        *_q(d["theta95_D"] / jdf.DEG)),
    ]
    if d["pmra_chain"] is not None:
        rows.append(("pmra_mas_yr",  *_q(d["pmra_chain"])))
        rows.append(("pmdec_mas_yr", *_q(d["pmdec_chain"])))
    for tag in fixed_J:
        rows.append((f"log10_J_{tag}_GeV2_cm5", *_q(d["log10_J"][tag])))
    # Exact-geometry total J within r_t (= J at theta=pi/2), large-angle correct.
    rows.append(("log10_J_pi2_GeV2_cm5", *_q(d["log10_J_pi2"])))
    for tag in fixed_D:
        rows.append((f"log10_D_{tag}_GeV_cm2", *_q(d["log10_D"][tag])))
    rows.append(("log10_D_pi2_GeV_cm2", *_q(d["log10_D_pi2"])))
    return rows


def write_summary_csv(path: Path, rows: list[tuple]) -> None:
    with Path(path).open("w") as f:
        f.write("quantity,q16,q50,q84\n")
        for name, q16, q50, q84 in rows:
            f.write(f"{name},{q16:.6g},{q50:.6g},{q84:.6g}\n")


# ----------------------------------------------------------------------------
# Derived-quantity export (for external consumers, e.g. SatGen_Dwarf/Jdata.py)
# ----------------------------------------------------------------------------

def save_derived(path: Path, d: dict, ctx: GalaxyContext) -> None:
    """Write the full per-draw derived arrays from :func:`derive` to a npz.

    This is a regenerable side-output, NOT part of the chain-only contract:
    the equal-weight chain in posterior_samples.npz stays the single source of
    truth, and this file is rebuilt from it by scripts/reprocess.py. Key names
    mirror the pre-refactor posterior_samples.npz layout so prior consumers are
    a one-line path change, plus the newer ``theta95_J_deg``.

    Array lengths: V/log10_rs/log10_rhos/beta(_tilde)/d/eps/rhalf/r_p/
    R_half_2d/r_half_3d/alpha_c and log10_M_half_* are full-chain length N.
    The J/D block (log10_J_*, log10_D_*, r_t_kpc, R_GC_kpc, theta95_J_deg) is
    length len(idx_jd) — map back to the chain with idx_jd. sigma_los_at_Rhalf2d
    is length len(idx_sig) (map back with idx_sig); sigma_profile_q* have length
    len(R_grid_sigma_profile) (their own idx_prof thin).
    """
    out = {
        "V":                    d["V_chain"],
        "log10_rs":             d["lr_chain"],
        "log10_rhos":           d["lp_chain"],
        "beta_tilde":           d["btilde_chain"],
        "beta":                 d["beta_chain"],
        "d_kpc_chain":          d["d_chain"],
        "eps_chain":            d["eps_chain"],
        "rhalf_arcmin_chain":   d["rhalf_arcmin_chain"],
        "r_p_chain":            d["r_p_chain"],
        "R_half_2d_chain":      d["R_half_2d_chain"],
        "r_half_3d_chain":      d["r_half_3d_chain"],
        "alpha_c_chain":        d["alpha_c_chain"],
        "log10_M_half_2d":      d["log10_M_2d"],
        "log10_M_half_3d":      d["log10_M_3d"],
        "sigma_los_at_Rhalf2d": d["sigma_at_Rhalf"],
        "R_grid_sigma_profile": d["R_grid"],
        "sigma_profile_q16":    d["sigma_profile_q16"],
        "sigma_profile_q50":    d["sigma_profile_q50"],
        "sigma_profile_q84":    d["sigma_profile_q84"],
        "r_t_kpc":              d["r_t_chain"],
        "R_GC_kpc":             d["R_GC_chain"],
        "theta95_J_deg":        d["theta95_J"] / jdf.DEG,
        "theta95_D_deg":        d["theta95_D"] / jdf.DEG,
        "log10_J_pi2":          d["log10_J_pi2"],
        "log10_D_pi2":          d["log10_D_pi2"],
        "idx_jd":               d["idx_jd"],
        "idx_sig":              d["idx_sig"],
        "idx_prof":             d["idx_prof"],
    }
    for tag, arr in d["log10_J"].items():
        out[f"log10_J_{tag}"] = arr
    for tag, arr in d["log10_D"].items():
        out[f"log10_D_{tag}"] = arr
    if d["pmra_chain"] is not None:
        out["pmra"] = d["pmra_chain"]
        out["pmdec"] = d["pmdec_chain"]
    np.savez(Path(path), **out)


# ----------------------------------------------------------------------------
# Chain persistence (the npz contract: chain + reproducibility metadata only)
# ----------------------------------------------------------------------------

def save_chain(path: Path, samples_eq, ctx: GalaxyContext, *,
               rseed: int, prior_name: str, shmr: str | None,
               thin_sigma: int, thin_jd: int, thin_profile: int,
               host=jdtidal.DEFAULT_HOST,
               tidal_factor: float = 2.0,
               p_min: float, rmax_over_rhalf: float,
               drop_variable: bool, use_p_weights: bool) -> None:
    """Persist the equal-weight chain plus the metadata needed to reproduce
    every derived view. No derived quantities are stored."""
    samples_eq = np.asarray(samples_eq, dtype=float)
    names = PARAM_NAMES_9 if ctx.has_pm else PARAM_NAMES_7
    np.savez(
        Path(path),
        samples_eq=samples_eq,
        param_names=np.array(names),
        rseed=int(rseed),
        lvdb_key=ctx.lvdb_key,
        prior_name=prior_name,
        shmr=("" if shmr is None else shmr),
        thin_sigma=int(thin_sigma), thin_jd=int(thin_jd),
        thin_profile=int(thin_profile),
        selection_p_min=float(p_min),
        selection_rmax_over_rhalf=float(rmax_over_rhalf),
        selection_drop_variable=bool(drop_variable),
        selection_use_p_weights=bool(use_p_weights),
        tidal_factor=float(tidal_factor),
        **jdtidal.host_save_fields(host),
    )


def load_chain(path: Path) -> tuple[np.ndarray, dict]:
    """Load (samples_eq, meta) from a chain npz. ``meta`` carries the
    reproducibility config (rseed, thinning, selection policy, host). Works on
    both new chain-only npz files and legacy npz files that also stored
    derived arrays (the extra keys are ignored)."""
    z = np.load(Path(path), allow_pickle=True)
    samples_eq = z["samples_eq"]
    meta: dict = {}
    for k in ("rseed", "thin_sigma", "thin_jd", "thin_profile"):
        if k in z.files:
            meta[k] = int(z[k])
    for k in ("lvdb_key", "prior_name", "shmr"):
        if k in z.files:
            meta[k] = str(z[k])
    if "selection_p_min" in z.files:
        meta["selection"] = dict(
            p_min=float(z["selection_p_min"]),
            rmax_over_rhalf=float(z["selection_rmax_over_rhalf"]),
            drop_variable=bool(z["selection_drop_variable"]),
            use_p_weights=bool(z["selection_use_p_weights"]),
        )
    if "tidal_factor" in z.files:
        meta["tidal_factor"] = float(z["tidal_factor"])
    host = jdtidal.host_from_npz(z)
    if host is not None:
        meta["host"] = host
    if "param_names" in z.files:
        meta["param_names"] = [str(s) for s in z["param_names"]]
    return samples_eq, meta


def host_from_meta(meta: dict):
    """Tidal host for a run whose ``meta`` came from ``load_chain``/``resolve_run_meta``.

    A chain that recorded its host deserialises to exactly that host. A chain
    that recorded none gets ``jdtidal.DEFAULT_HOST`` (MW2022) — the same default
    ``derive`` uses — so a recompute reproduces the run's stored derived arrays
    instead of silently substituting a different host.
    """
    return meta.get("host", jdtidal.DEFAULT_HOST)


def legacy_meta_from_audit(run_dir: Path) -> dict | None:
    """Reconstruct reproducibility metadata from a legacy run's audit.json.

    Pre-contract npz files don't carry the selection policy / rseed / thinning,
    but the sibling audit.json does. Returns a meta dict shaped like
    :func:`load_chain`'s, or None if audit.json is absent or lacks the policy.
    """
    import json
    ap = Path(run_dir) / "audit.json"
    if not ap.exists():
        return None
    a = json.loads(ap.read_text())
    sp = a.get("selection_policy") or {}
    if "p_min" not in sp:
        return None
    thinning = a.get("thinning") or {}
    dyn = a.get("dynesty") or {}
    return {
        "lvdb_key": a.get("lvdb_key") or Path(run_dir).parent.name,
        "prior_name": a.get("prior_name", "jeffreys"),
        "selection": {
            "p_min": float(sp["p_min"]),
            "rmax_over_rhalf": float(sp.get("R_over_rhalf_max", 2.0)),
            "drop_variable": bool(sp.get("drop_variable", True)),
            "use_p_weights": bool(sp.get("use_p_weights", False)),
        },
        "rseed": int(dyn.get("rseed", 0)),
        "thin_sigma": int(thinning.get("sigma", 2000)),
        "thin_jd": int(thinning.get("jd", 500)),
        "thin_profile": int(thinning.get("profile", 300)),
    }


def resolve_run_meta(run_dir: Path) -> tuple[np.ndarray, dict]:
    """Load (samples_eq, meta) for a run, backfilling legacy npz metadata from
    audit.json. The single meta-resolution path shared by reprocess.py and the
    plotting tool so every regenerated view uses the SAME prior / selection /
    rseed. Raises if a legacy npz has no usable audit.json."""
    run_dir = Path(run_dir)
    samples_eq, meta = load_chain(run_dir / "posterior_samples.npz")
    if "selection" not in meta or meta.get("lvdb_key") is None:
        legacy = legacy_meta_from_audit(run_dir)
        if legacy is None:
            raise RuntimeError(
                f"{run_dir}: legacy npz with no usable audit.json "
                "selection_policy; cannot reproduce derived views."
            )
        # audit.json records no host, so a host the chain *did* record must
        # survive the swap — dropping it would silently fall back to the
        # MW2022 default and recompute r_t against the wrong potential.
        if "host" in meta:
            legacy["host"] = meta["host"]
        meta = legacy
    return samples_eq, meta
