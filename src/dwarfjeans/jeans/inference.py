"""
Stage 2 inference for a single mock galaxy (NFW + Plummer + constant β).

Free parameters (matching stage2.md, modulo nuisances):
  V         — systemic velocity (km/s)            uniform [V_lit-10, V_lit+10]
  log10 r_s — NFW scale radius (kpc)              uniform [-2, 1] (Jeffreys)
  log10 rho_s — NFW scale density (M_sun/kpc^3)   uniform [4, 14] (Jeffreys)
  beta_tilde — symmetrized anisotropy              uniform [-0.95, 1)

For these mock-recovery tests we KEEP THE NUISANCES FIXED at the truth:
  r_p, d, eps, mu_alpha, mu_delta — the goal is to test halo-parameter recovery,
  not nuisance-marginalization, since nuisance handling is documented separately
  (uncertainty_conventions.md). Adding nuisance priors is a follow-up experiment.

The r_s > r_1/2 constraint is omitted here — for a mock galaxy where we're
generating data from a known truth, we want to see whether the prior alone
can recover the parameters; introducing a stage-2-specific deterministic
lower bound on r_s would prejudge the recovery in a test of the procedure.
This is documented as a deviation from stage2.md for these synthetic tests.

Likelihood (post-membership-cut convention with all p_i = 1):
  ln L = sum_i  -1/2 * ln[2π (sigma_los²(R_i) + sigma_eps_i²)]
                -1/2 * (V_i - V)² / (sigma_los²(R_i) + sigma_eps_i²)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, truncnorm
import dynesty

from dwarfjeans.jeans import solver as jeans


# Prior bounds (matching stage2.md)
LOG10_RS_BOUNDS = (-2.0, 1.0)        # log10(r_s / kpc)
LOG10_RHOS_BOUNDS = (4.0, 14.0)      # log10(rho_s / [M_sun/kpc^3])
BETA_TILDE_BOUNDS = (-0.95, 1.0)     # uniform symmetrized anisotropy
V_HALFWIDTH = 10.0                   # km/s, default per-galaxy override
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


def beta_tilde_to_beta(beta_tilde: float) -> float:
    return 2.0 * beta_tilde / (1.0 + beta_tilde)


def make_prior_transform(V_center: float, log10_rs_min: float | None = None):
    """Closed-form unit-cube to physical parameter map. Order: (V, log_rs, log_rhos, btilde).

    log10_rs_min overrides the default lower bound on log10(r_s), e.g. to enforce r_s > r_p.
    """
    V_lo = V_center - V_HALFWIDTH
    V_hi = V_center + V_HALFWIDTH
    rs_lo = LOG10_RS_BOUNDS[0] if log10_rs_min is None else log10_rs_min
    rs_hi = LOG10_RS_BOUNDS[1]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        x = np.empty_like(u)
        x[0] = V_lo + (V_hi - V_lo) * u[0]
        x[1] = rs_lo + (rs_hi - rs_lo) * u[1]
        x[2] = LOG10_RHOS_BOUNDS[0] + (LOG10_RHOS_BOUNDS[1] - LOG10_RHOS_BOUNDS[0]) * u[2]
        x[3] = BETA_TILDE_BOUNDS[0] + (BETA_TILDE_BOUNDS[1] - BETA_TILDE_BOUNDS[0]) * u[3]
        return x

    return prior_transform


def make_prior_transform_with_nuisances(
    V_center: float,
    d_mean: float, d_sigma: float,
    eps_mean: float, eps_sigma: float,
    rhalf_mean: float, rhalf_sigma: float,
):
    """
    7D prior transform: (V, log_rs, log_rhos, btilde, d_kpc, eps, rhalf_arcmin).

    - V, log_rs, log_rhos, btilde: same uniform priors as 4D (LOG10_RS_BOUNDS lower
      bound — the rs > rp constraint is enforced inside the likelihood since the
      sampled r_p varies per draw).
    - d_kpc: Normal(d_mean, d_sigma).
    - eps: Normal(eps_mean, eps_sigma) truncated to [0, 1).
    - rhalf_arcmin: Normal(rhalf_mean, rhalf_sigma).
    """
    V_lo = V_center - V_HALFWIDTH
    V_hi = V_center + V_HALFWIDTH
    rs_lo, rs_hi = LOG10_RS_BOUNDS
    eps_a = (0.0 - eps_mean) / eps_sigma
    eps_b = (1.0 - eps_mean) / eps_sigma  # truncnorm a, b are in std-normal units

    def prior_transform(u: np.ndarray) -> np.ndarray:
        x = np.empty_like(u)
        x[0] = V_lo + (V_hi - V_lo) * u[0]
        x[1] = rs_lo + (rs_hi - rs_lo) * u[1]
        x[2] = LOG10_RHOS_BOUNDS[0] + (LOG10_RHOS_BOUNDS[1] - LOG10_RHOS_BOUNDS[0]) * u[2]
        x[3] = BETA_TILDE_BOUNDS[0] + (BETA_TILDE_BOUNDS[1] - BETA_TILDE_BOUNDS[0]) * u[3]
        x[4] = norm.ppf(u[4], loc=d_mean, scale=d_sigma)
        x[5] = truncnorm.ppf(u[5], eps_a, eps_b, loc=eps_mean, scale=eps_sigma)
        x[6] = norm.ppf(u[6], loc=rhalf_mean, scale=rhalf_sigma)
        return x

    return prior_transform


def _jeffreys_log_term(sigma_los_sq: np.ndarray, T: np.ndarray,
                        sigma_eps_sq: np.ndarray, p: np.ndarray) -> float:
    """
    ½ ln D where D is the (membership-weighted) Fisher determinant in
    (ln ρ_s, ln r_s) at fixed β:

        w̃_i  = A_i² / (A_i + ε_i²)²,        A_i = σ_los²(R_i)
        S0   = Σ p_i w̃_i
        T̄   = Σ p_i w̃_i T_i / S0
        D    = S0 · Σ p_i w̃_i (T_i − T̄)²

    The variance-form is mathematically equivalent to D = S0·S2 − S1² but
    avoids catastrophic cancellation when Var(T) → 0.

    Returns the additive contribution `0.5 * ln D` to the log-likelihood,
    or -inf if D is non-finite or ≤ 0 (degenerate parameter point — the
    stars do not span a range of R/r_s and r_s is unidentifiable).
    """
    A = sigma_los_sq
    s_tot_sq = A + sigma_eps_sq
    w_tilde = (A / s_tot_sq) ** 2  # = A² / s_tot⁴, computed without overflow
    pw = p * w_tilde
    S0 = pw.sum()
    if not np.isfinite(S0) or S0 <= 0.0:
        return -np.inf
    T_bar = (pw * T).sum() / S0
    D = S0 * (pw * (T - T_bar) ** 2).sum()
    if not np.isfinite(D) or D <= 0.0:
        return -np.inf
    return 0.5 * np.log(D)


def make_loglike_with_nuisances(
    Rad_arcmin: np.ndarray, V: np.ndarray, sigma_eps: np.ndarray, p: np.ndarray,
    use_jeffreys_prior: bool = True,
    fix_r_p_arcmin: bool = False,
):
    """
    7D Stage-2 log-likelihood with d, ε, rhalf_arcmin as nuisance parameters.

    Per-draw derived quantities:
        r_p   = d · rhalf_arcmin · ARCMIN_TO_RAD · √(1 − ε)        [kpc]
        R_kpc = d · Rad_arcmin   · ARCMIN_TO_RAD                    [kpc, per star]

    fix_r_p_arcmin=True reinterprets the 7th parameter (`rhalf_arcmin`) as the
    angular Plummer scale `r_p_arcmin` directly: r_p = d · r_p_arcmin · ARCMIN_TO_RAD,
    skipping the √(1 − ε) factor. ε is still sampled and reported but not used
    for the geometry. The caller is responsible for passing a Normal prior on
    that 7th slot centered on the desired r_p_arcmin.

    The rs > r_p constraint is enforced here (not in the prior) because the sampled
    r_p varies per draw; rejection returns -1e300 to match the existing failure path.

    use_jeffreys_prior=True (default): add ½ ln D to the log-likelihood, where D is
    the conditional Fisher determinant in (ln ρ_s, ln r_s) at fixed β
    (jeffreys_jeans_derivation.md). The σ_los solve is replaced with
    `jeans.sigma_los_with_T` to obtain σ_los and T = 3 − 𝒬/𝒫 in one pass.
    """
    Rad_arcmin = np.asarray(Rad_arcmin, dtype=float)
    V = np.asarray(V, dtype=float)
    sigma_eps = np.asarray(sigma_eps, dtype=float)
    p = np.asarray(p, dtype=float)
    sigma_eps_sq = sigma_eps ** 2

    LN_2PI = np.log(2.0 * np.pi)

    def loglike(theta: np.ndarray) -> float:
        V_sys, log_rs, log_rhos, btilde, d, eps, rhalf_arcmin = theta
        r_s = 10.0 ** log_rs
        rho_s = 10.0 ** log_rhos
        beta = beta_tilde_to_beta(btilde)
        # Derived per-draw observables.
        if fix_r_p_arcmin:
            r_p = d * rhalf_arcmin * ARCMIN_TO_RAD
        else:
            r_p = d * rhalf_arcmin * ARCMIN_TO_RAD * np.sqrt(1.0 - eps)
        if not (r_p > 0.0 and r_s > r_p):
            return -1e300
        R = d * Rad_arcmin * ARCMIN_TO_RAD
        # Match the R floor used in the fixed-d path (run_segue1.load_stars), to
        # protect the inner-Jeans u-grid against R=0.
        R = np.clip(R, 1e-5, None)
        try:
            if use_jeffreys_prior:
                sigma_los, T = jeans.sigma_los_with_T(R, beta, r_s, rho_s, r_p)
            else:
                sigma_los = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")
        except (ValueError, FloatingPointError):
            return -1e300
        if not np.all(np.isfinite(sigma_los)):
            return -1e300
        sigma2 = sigma_los ** 2 + sigma_eps_sq
        ln_li = -0.5 * (LN_2PI + np.log(sigma2)) - 0.5 * (V - V_sys) ** 2 / sigma2
        ll = float(np.sum(p * ln_li))
        if use_jeffreys_prior:
            log_det = _jeffreys_log_term(sigma_los ** 2, T, sigma_eps_sq, p)
            if not np.isfinite(log_det):
                return -1e300
            ll += log_det
        return ll

    return loglike


def make_loglike(R: np.ndarray, V: np.ndarray, sigma_eps: np.ndarray,
                  p: np.ndarray, r_p: float,
                  use_jeffreys_prior: bool = True):
    """Stage-2 log-likelihood. Captures the per-star data + fixed r_p.

    use_jeffreys_prior=True (default): add ½ ln D, the conditional Jeffreys
    determinant in (ln ρ_s, ln r_s) at fixed β.
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    sigma_eps = np.asarray(sigma_eps, dtype=float)
    p = np.asarray(p, dtype=float)
    sigma_eps_sq = sigma_eps ** 2

    LN_2PI = np.log(2.0 * np.pi)

    def loglike(theta: np.ndarray) -> float:
        V_sys, log_rs, log_rhos, btilde = theta
        r_s = 10.0 ** log_rs
        rho_s = 10.0 ** log_rhos
        beta = beta_tilde_to_beta(btilde)
        try:
            if use_jeffreys_prior:
                sigma_los, T = jeans.sigma_los_with_T(R, beta, r_s, rho_s, r_p)
            else:
                sigma_los = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")
        except (ValueError, FloatingPointError):
            return -1e300
        if not np.all(np.isfinite(sigma_los)):
            return -1e300
        sigma2 = sigma_los ** 2 + sigma_eps_sq
        # log-Gaussian, membership-weighted; p_i are Bpr probabilities (0.8–1.0 after hard cut)
        # ln L_i = -1/2 ln(2π σ²) - (V_i - V)² / (2 σ²)
        ln_li = -0.5 * (LN_2PI + np.log(sigma2)) - 0.5 * (V - V_sys) ** 2 / sigma2
        ll = float(np.sum(p * ln_li))
        if use_jeffreys_prior:
            log_det = _jeffreys_log_term(sigma_los ** 2, T, sigma_eps_sq, p)
            if not np.isfinite(log_det):
                return -1e300
            ll += log_det
        return ll

    return loglike


def make_loglike_asimov(R: np.ndarray, sigma_eps: np.ndarray, p: np.ndarray,
                         r_p: float, sigma_los_truth: np.ndarray):
    """
    Asimov log-likelihood: replaces each `(V_i - V_sys)²` in the Gaussian
    Stage-2 likelihood with its expectation under truth,
        σ_tot,truth²(R_i) = sigma_los_truth(R_i)² + sigma_eps_i².

    The V_i array is *not* consumed (no synthetic V dataset can make the
    standard Gaussian log-likelihood take its expectation value at truth),
    so this is a separate likelihood, not a transformation of the data.
    Per-star Asimov term:

        ln <L_i>_truth(θ) = -½ ln[2π σ_tot²(R_i; θ)]
                            -½ σ_tot,truth²(R_i) / σ_tot²(R_i; θ)

    where σ_tot²(R_i; θ) = sigma_los(R_i; r_s, ρ_s, β, r_p)² + σ_eps,i².

    Maximising over θ gives σ_tot²(R_i; θ) = σ_tot,truth²(R_i) for all i,
    placing the MLE at truth (modulo any exact-match degeneracy in
    σ_los profiles). V_sys does not appear and is therefore unconstrained
    by this likelihood — the V posterior is the V prior.

    Caller responsibility: pass a `sigma_los_truth` evaluated at the true
    (β, r_s, ρ_s, r_p) at the given R. `make_asimov_galaxy` precomputes this
    and stores it under `sigma_los_true` in its returned dict.
    """
    R = np.asarray(R, dtype=float)
    sigma_eps = np.asarray(sigma_eps, dtype=float)
    p = np.asarray(p, dtype=float)
    sigma_los_truth = np.asarray(sigma_los_truth, dtype=float)
    sigma_eps_sq = sigma_eps ** 2
    sigma_tot_truth_sq = sigma_los_truth ** 2 + sigma_eps_sq

    LN_2PI = np.log(2.0 * np.pi)

    def loglike(theta: np.ndarray) -> float:
        # V_sys appears in theta[0] for schema compatibility but does not
        # enter the likelihood. The prior on V is therefore reflected in the
        # posterior unchanged.
        _, log_rs, log_rhos, btilde = theta
        r_s = 10.0 ** log_rs
        rho_s = 10.0 ** log_rhos
        beta = beta_tilde_to_beta(btilde)
        try:
            sigma_los = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")
        except (ValueError, FloatingPointError):
            return -1e300
        if not np.all(np.isfinite(sigma_los)):
            return -1e300
        sigma2 = sigma_los ** 2 + sigma_eps_sq
        ln_li = -0.5 * (LN_2PI + np.log(sigma2)) - 0.5 * sigma_tot_truth_sq / sigma2
        return float(np.sum(p * ln_li))

    return loglike


def run_inference(
    galaxy: dict,
    V_center: float = 0.0,
    nlive: int = 500,
    dlogz: float = 0.1,
    rseed: int = 0,
    print_progress: bool = False,
    sample: str = "rwalk",
    bound: str = "multi",
    asimov: bool = False,
    log10_rs_min: float | None = None,
    marginalize_nuisances: bool = False,
    nuisance_priors: dict | None = None,
    use_jeffreys_prior: bool = True,
    fix_r_p_arcmin: bool = False,
) -> dict:
    """
    Run dynesty on the mock galaxy. Returns a dict with:
      samples_eq    — equal-weight samples (n_eq, 4)
      samples       — raw nested-sampling samples (n_iter, 4)
      logwt, logz, logz_err
      param_names   — ('V', 'log10_rs', 'log10_rhos', 'beta_tilde')
      sampler       — the dynesty sampler object (for diagnostics)

    asimov=True selects the Asimov likelihood (`make_loglike_asimov`), which
    replaces (V_i - V_sys)² with σ_tot,truth²(R_i) per star. V_sys is then
    unconstrained by the data and its posterior equals the prior.
    """
    rng = np.random.default_rng(rseed)

    if marginalize_nuisances:
        if asimov:
            raise ValueError("marginalize_nuisances + asimov not implemented")
        if nuisance_priors is None:
            raise ValueError("marginalize_nuisances=True requires nuisance_priors dict")
        loglike = make_loglike_with_nuisances(
            Rad_arcmin=galaxy["Rad_arcmin"],
            V=galaxy["V"],
            sigma_eps=galaxy["sigma_eps"],
            p=galaxy["p"],
            use_jeffreys_prior=use_jeffreys_prior,
            fix_r_p_arcmin=fix_r_p_arcmin,
        )
        prior_transform = make_prior_transform_with_nuisances(
            V_center,
            d_mean=nuisance_priors["d_mean"], d_sigma=nuisance_priors["d_sigma"],
            eps_mean=nuisance_priors["eps_mean"], eps_sigma=nuisance_priors["eps_sigma"],
            rhalf_mean=nuisance_priors["rhalf_mean"], rhalf_sigma=nuisance_priors["rhalf_sigma"],
        )
        ndim = 7
        param_names = ("V", "log10_rs", "log10_rhos", "beta_tilde",
                        "d_kpc", "eps", "rhalf_arcmin")
    elif asimov:
        loglike = make_loglike_asimov(
            R=galaxy["R"],
            sigma_eps=galaxy["sigma_eps"],
            p=galaxy["p"],
            r_p=galaxy["truth"]["r_p"],
            sigma_los_truth=galaxy["sigma_los_true"],
        )
        prior_transform = make_prior_transform(V_center, log10_rs_min=log10_rs_min)
        ndim = 4
        param_names = ("V", "log10_rs", "log10_rhos", "beta_tilde")
    else:
        loglike = make_loglike(
            R=galaxy["R"],
            V=galaxy["V"],
            sigma_eps=galaxy["sigma_eps"],
            p=galaxy["p"],
            r_p=galaxy["truth"]["r_p"],
            use_jeffreys_prior=use_jeffreys_prior,
        )
        prior_transform = make_prior_transform(V_center, log10_rs_min=log10_rs_min)
        ndim = 4
        param_names = ("V", "log10_rs", "log10_rhos", "beta_tilde")

    sampler = dynesty.NestedSampler(
        loglike,
        prior_transform,
        ndim=ndim,
        nlive=nlive,
        bound=bound,
        sample=sample,
        rstate=np.random.default_rng(rseed),
    )
    sampler.run_nested(dlogz=dlogz, print_progress=print_progress)

    res = sampler.results
    samples = res["samples"]
    logwt = res["logwt"] - res["logz"][-1]  # normalize log weights
    weights = np.exp(logwt)
    samples_eq = dynesty.utils.resample_equal(samples, weights)

    return {
        "samples": samples,
        "samples_eq": samples_eq,
        "logwt": logwt,
        "logz": res["logz"][-1],
        "logz_err": res["logzerr"][-1],
        "param_names": param_names,
        "n_iter": len(samples),
        "n_eq": len(samples_eq),
    }


def summarize_posterior(samples_eq: np.ndarray, truth: dict,
                         asimov: bool = False) -> dict:
    """
    For each parameter and several derived quantities, return median,
    16/84 percentiles, and an asymmetric z-score (truth vs median, scaled
    by the appropriate one-sided σ).

    Derived quantities:
      - log10(rho_s · r_s^3): the user's heuristic degeneracy axis.
      - log10 M(R_half_2d): NFW enclosed mass at the 2D projected half-light
        radius (= r_p for Plummer). The pipeline docs use 2D for `r_1/2`.
      - log10 M(r_half_3d): NFW enclosed mass at the 3D half-mass radius
        (= 1.30477 r_p for Plummer). This is Wolf+2010's prediction for
        the best-constrained quantity.

    Both M(r_1/2) chains are computed by pushing each posterior sample of
    (r_s, rho_s) through the NFW M(r) at the truth r_p / r_½ — r_p is
    held fixed in these recovery tests, so this is a clean derivation
    from the already-stored chains.

    asimov=True flags V as 'prior-only' (the Asimov likelihood does not
    constrain V_sys; reporting a z-score on V would describe the V prior,
    not data recovery). The V row is still populated for schema continuity.
    """
    V, lr, lp, bt = samples_eq.T
    r_s_chain = 10.0 ** lr
    rho_s_chain = 10.0 ** lp
    log_rho_rs3 = lp + 3.0 * lr

    R_half_2d = truth["R_half_2d"]
    r_half_3d = truth["r_half_3d"]
    M_chain_2d = jeans.nfw_M(R_half_2d, r_s_chain, rho_s_chain)
    M_chain_3d = jeans.nfw_M(r_half_3d, r_s_chain, rho_s_chain)
    log_M_2d = np.log10(M_chain_2d)
    log_M_3d = np.log10(M_chain_3d)

    derived = {
        "V": V,
        "log10_rs": lr,
        "log10_rhos": lp,
        "beta_tilde": bt,
        "log10_rhos_rs3": log_rho_rs3,
        "log10_M_half_2d": log_M_2d,
        "log10_M_half_3d": log_M_3d,
    }
    truth_map = {
        "V": truth["V_sys"],
        "log10_rs": truth["log10_rs"],
        "log10_rhos": truth["log10_rhos"],
        "beta_tilde": truth["beta_tilde"],
        "log10_rhos_rs3": truth["log10_rhos_rs3"],
        "log10_M_half_2d": truth["log10_M_half_2d"],
        "log10_M_half_3d": truth["log10_M_half_3d"],
    }

    out = {}
    for k, arr in derived.items():
        q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
        sigma_lo = q50 - q16
        sigma_hi = q84 - q50
        true_val = truth_map[k]
        if true_val < q50:
            z = (true_val - q50) / max(sigma_lo, 1e-12)
        else:
            z = (true_val - q50) / max(sigma_hi, 1e-12)
        entry = {
            "truth": true_val,
            "median": q50,
            "q16": q16, "q84": q84,
            "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
            "z": z,
        }
        if asimov and k == "V":
            entry["prior_only"] = True
            entry["z"] = float("nan")  # avoid mistaken inclusion in pop diagnostics
        out[k] = entry
    return out


def summarize_jd(samples_eq: np.ndarray, truth: dict,
                  d_kpc: float, r_t_kpc: float,
                  thin_to: int = 500,
                  n_R: int = 64, n_u: int = 128) -> dict:
    """
    Push the chain through Stage-3 J(θ) and D(θ) integrals.

    Returns the same per-quantity dict as summarize_posterior (truth, median,
    q16/q84, sigma_lo/hi, z) for:

      log10_J_0p1deg, log10_J_0p2deg, log10_J_0p5deg, log10_J_alphac
      log10_D_0p1deg, log10_D_0p2deg, log10_D_0p5deg, log10_D_alphacover2

    where:
      * J angles follow pipeline_overview Stage 3: 0.1°, 0.2°, 0.5°, α_c
        with α_c = 2 r_½,3D / d.
      * D angles follow Stage 3: 0.1°, 0.2°, 0.5°, α_c/2.

    Truth values are computed at the same (r_s, ρ_s, d, r_t) used by the mock
    (r_s, ρ_s from `truth`; d, r_t passed in by the caller — these are not
    in the existing chains since the mock generation didn't fix them).

    All values are reported in P&S 2018 units: log10(J / [GeV²/cm⁵]) and
    log10(D / [GeV/cm²]).

    Cost: thin_to samples × 8 angles × ~4 ms ≈ 16 s for thin_to=500.
    """
    # Lazy import so this module doesn't pull in j_d_factors unless requested.
    import j_d_factors as jdf

    # Chain samples
    V, lr, lp, bt = samples_eq.T
    r_s_chain = 10.0 ** lr
    rho_s_chain = 10.0 ** lp

    # Thin to manage cost (percentiles converge quickly with N>>1)
    N = samples_eq.shape[0]
    if N > thin_to:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=thin_to, replace=False)
        r_s_chain = r_s_chain[idx]
        rho_s_chain = rho_s_chain[idx]

    # Angle list
    alpha_c = jdf.alpha_c_radians(truth["r_half_3d"], d_kpc)
    angles_J = {
        "0p1deg": 0.1 * jdf.DEG,
        "0p2deg": 0.2 * jdf.DEG,
        "0p5deg": 0.5 * jdf.DEG,
        "alphac": alpha_c,
    }
    angles_D = {
        "0p1deg": 0.1 * jdf.DEG,
        "0p2deg": 0.2 * jdf.DEG,
        "0p5deg": 0.5 * jdf.DEG,
        "alphacover2": 0.5 * alpha_c,
    }

    # Truth values
    truth_log_J = {}
    truth_log_D = {}
    for tag, th in angles_J.items():
        J, _ = jdf.J_D_factors(th, d_kpc, truth["r_s"], truth["rho_s"], r_t_kpc,
                                 n_R=n_R, n_u=n_u)
        truth_log_J[tag] = float(np.log10(J)) + jdf.LOG10_J_FAC
    for tag, th in angles_D.items():
        _, D = jdf.J_D_factors(th, d_kpc, truth["r_s"], truth["rho_s"], r_t_kpc,
                                 n_R=n_R, n_u=n_u)
        truth_log_D[tag] = float(np.log10(D)) + jdf.LOG10_D_FAC

    # Chain values
    M = r_s_chain.size
    chain_log_J = {tag: np.empty(M) for tag in angles_J}
    chain_log_D = {tag: np.empty(M) for tag in angles_D}

    for i in range(M):
        rs_i = float(r_s_chain[i])
        rhos_i = float(rho_s_chain[i])
        for tag, th in angles_J.items():
            J, _ = jdf.J_D_factors(th, d_kpc, rs_i, rhos_i, r_t_kpc,
                                     n_R=n_R, n_u=n_u)
            chain_log_J[tag][i] = np.log10(J) + jdf.LOG10_J_FAC if J > 0 else -np.inf
        for tag, th in angles_D.items():
            _, D = jdf.J_D_factors(th, d_kpc, rs_i, rhos_i, r_t_kpc,
                                     n_R=n_R, n_u=n_u)
            chain_log_D[tag][i] = np.log10(D) + jdf.LOG10_D_FAC if D > 0 else -np.inf

    # Build the standard summary entries
    out = {}
    def _fill(key, chain, truth_val):
        chain = chain[np.isfinite(chain)]
        q16, q50, q84 = np.percentile(chain, [16.0, 50.0, 84.0])
        sigma_lo = q50 - q16
        sigma_hi = q84 - q50
        if truth_val < q50:
            z = (truth_val - q50) / max(sigma_lo, 1e-12)
        else:
            z = (truth_val - q50) / max(sigma_hi, 1e-12)
        out[key] = {
            "truth": float(truth_val),
            "median": float(q50),
            "q16": float(q16), "q84": float(q84),
            "sigma_lo": float(sigma_lo), "sigma_hi": float(sigma_hi),
            "z": float(z),
        }

    for tag in angles_J:
        _fill(f"log10_J_{tag}", chain_log_J[tag], truth_log_J[tag])
    for tag in angles_D:
        _fill(f"log10_D_{tag}", chain_log_D[tag], truth_log_D[tag])

    out["_meta"] = {
        "d_kpc": float(d_kpc),
        "r_t_kpc": float(r_t_kpc),
        "alpha_c_rad": float(alpha_c),
        "alpha_c_deg": float(alpha_c / jdf.DEG),
        "thin_to": int(thin_to),
        "M_used": int(M),
        "log10_J_unit": "log10(J / [GeV^2 cm^-5])",
        "log10_D_unit": "log10(D / [GeV cm^-2])",
    }
    return out
