"""Walker+2006 radius-independent constant-σ dispersion inference.

This is a model-free counterpart to the spherical-Jeans posterior
``σ_los(R)``: a single global dispersion fit to all members under a
Gaussian likelihood. It is the canonical cross-paper comparison
quantity (P&S 2018, Walker 2006).
"""
from __future__ import annotations

import numpy as np

from dwarfjeans.jeans.priors import V_HALFWIDTH


def constant_sigma_inference(V_obs, sigma_eps, p, V_center,
                             V_halfwidth=V_HALFWIDTH,
                             log10_sigma_min=-2.0, log10_sigma_max=2.0,
                             n_V=400, n_sigma=400,
                             prior="jeffreys"):
    """Walker+2006 membership-weighted Gaussian likelihood for (V_sys, σ_los).

    Likelihood per star:
        ln L_i = p_i [ -½ ln(2π(σ² + ε_i²)) - ½ (V_i - V_sys)² / (σ² + ε_i²) ]

    Prior on V_sys is uniform on [V_center ± V_halfwidth]. Prior on σ is
    selected by ``prior``, truncated to log₁₀σ ∈ [log10_sigma_min,
    log10_sigma_max]:

      * ``"jeffreys"`` (default): proper Fisher-determinant prior. Per-star
        Fisher info is diagonal in (V̄, σ) with I_V̄V̄ = 1/σ_i² and
        I_σσ = 2σ²/σ_i⁴ where σ_i² = σ² + ε_i², so
            p_J(σ) ∝ σ · √([Σ_i p_i/σ_i²][Σ_i p_i/σ_i⁴])    (V̄-independent).
      * ``"loguniform"``: uniform in log₁₀σ over the truncation range,
        i.e. dP/dσ ∝ 1/σ.
      * ``"uniform"``: uniform in σ over the truncation range, i.e.
        dP/dσ = const.
    """
    V_obs = np.asarray(V_obs, dtype=float)
    sigma_eps_sq = np.asarray(sigma_eps, dtype=float) ** 2
    p = np.asarray(p, dtype=float)

    V_grid = np.linspace(V_center - V_halfwidth, V_center + V_halfwidth, n_V)
    log10_sigma_grid = np.linspace(log10_sigma_min, log10_sigma_max, n_sigma)
    sigma_grid = 10.0 ** log10_sigma_grid

    sigma2 = sigma_grid[:, None] ** 2 + sigma_eps_sq[None, :]
    log_norm = np.log(2.0 * np.pi * sigma2) @ p
    dV2 = (V_obs[None, :] - V_grid[:, None]) ** 2
    chi2 = dV2 @ (p[:, None] / sigma2.T)

    log_lik = -0.5 * (log_norm[None, :] + chi2)

    inv_sigma2 = 1.0 / sigma2
    sum1 = inv_sigma2 @ p
    sum2 = (inv_sigma2 ** 2) @ p
    # log_prior_J retained for diagnostic export below; selected prior
    # drives the actual posterior.
    log_prior_J = np.log(sigma_grid) + 0.5 * (np.log(sum1) + np.log(sum2))

    if prior == "jeffreys":
        log_prior = log_prior_J
    elif prior == "loguniform":
        # dP/dσ ∝ 1/σ → log_prior = -ln σ (any additive constant absorbed
        # by the global normalization below).
        log_prior = -np.log(sigma_grid)
    elif prior == "uniform":
        log_prior = np.zeros_like(sigma_grid)
    else:
        raise ValueError(
            f"prior must be one of 'jeffreys', 'loguniform', 'uniform'; "
            f"got {prior!r}"
        )

    log_post = log_lik + log_prior[None, :]
    log_post -= log_post.max()
    log_lik  -= log_lik.max()
    post = np.exp(log_post)

    marg_V = np.trapezoid(post, sigma_grid, axis=1)
    marg_V /= np.trapezoid(marg_V, V_grid)
    marg_sigma = np.trapezoid(post, V_grid, axis=0)
    marg_sigma /= np.trapezoid(marg_sigma, sigma_grid)
    marg_log10_sigma = marg_sigma * sigma_grid * np.log(10.0)

    def _pct(x, pdf):
        dx = np.diff(x)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)])
        cdf /= cdf[-1]
        return [float(np.interp(q, cdf, x)) for q in (0.16, 0.50, 0.84)]

    q16_V, q50_V, q84_V = _pct(V_grid, marg_V)
    q16_s, q50_s, q84_s = _pct(sigma_grid, marg_sigma)

    return {
        "V_sys":     {"median": q50_V, "q16": q16_V, "q84": q84_V,
                      "sigma_lo": q50_V - q16_V, "sigma_hi": q84_V - q50_V},
        "sigma_int": {"median": q50_s, "q16": q16_s, "q84": q84_s,
                      "sigma_lo": q50_s - q16_s, "sigma_hi": q84_s - q50_s},
        "V_grid": V_grid, "sigma_grid": sigma_grid,
        "log10_sigma_grid": log10_sigma_grid,
        "prior": prior,
        "log_post": log_post, "log_lik": log_lik, "log_prior_J": log_prior_J,
        "log_prior": log_prior,
        "marg_V": marg_V,
        "marg_sigma": marg_sigma, "marg_log10_sigma": marg_log10_sigma,
    }
