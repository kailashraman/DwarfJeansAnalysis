"""
Prior families for the Stage-2 NFW Jeans inference.

Three options are exposed via ``PRIOR_REGISTRY`` / ``get_prior``:

  uniform     — linear-uniform on (r_s, rho_s) over the same physical
                bounds as ``loguniform``. The chain still records
                ``log10_rs, log10_rhos`` so existing summary code is
                unchanged.
  loguniform  — uniform on (log10 r_s, log10 rho_s). This is the default.
  jeffreys    — loguniform base + the data-dependent Jeffreys
                log-determinant correction added to the log-posterior.
                The Fisher information comes from the per-star Gaussian
                pseudo-likelihood built on the σ_los Jeans solution
                (see docs/plan/jeffreys_jeans_derivation.md).

Each family is encoded as a ``Prior`` dataclass holding:
  - ``make_transform``: builder returning a unit-cube → physical map
    that dynesty consumes.
  - ``log_correction``: data-dependent additive term applied inside
    the likelihood. Returns 0.0 for non-Jeffreys priors.

V is always a uniform prior on [V_center - V_HALFWIDTH, V_center +
V_HALFWIDTH]; β̃ is always uniform on BETA_TILDE_BOUNDS. The "prior
family" only affects (r_s, rho_s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm, truncnorm


# Prior bounds (matching stage2.md).
LOG10_RS_BOUNDS = (-2.0, 1.0)        # log10(r_s / kpc)
LOG10_RHOS_BOUNDS = (4.0, 14.0)      # log10(rho_s / [M_sun/kpc^3])
BETA_TILDE_BOUNDS = (-0.95, 1.0)     # uniform symmetrized anisotropy
V_HALFWIDTH = 10.0                   # km/s, default per-galaxy override


# ---------------------------------------------------------------------------
# Loguniform prior transforms (the historical default)
# ---------------------------------------------------------------------------

def make_loguniform_prior_transform(V_center: float,
                                     log10_rs_min: float | None = None):
    """4D unit-cube → (V, log10 r_s, log10 rho_s, β̃) with uniform priors
    on the log-space halo parameters.

    log10_rs_min overrides the default lower bound on log10(r_s), e.g. to
    enforce r_s > r_p.
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


def make_loguniform_prior_transform_with_nuisances(
    V_center: float,
    d_mean: float, d_sigma: float,
    eps_mean: float, eps_sigma: float,
    rhalf_mean: float, rhalf_sigma: float,
):
    """7D unit-cube → (V, log10 r_s, log10 rho_s, β̃, d, ε, rhalf_arcmin).

    - V, log_rs, log_rhos, btilde: same uniform-on-log priors as 4D.
    - d_kpc:       Normal(d_mean, d_sigma).
    - eps:         Normal(eps_mean, eps_sigma) truncated to [0, 1).
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


# ---------------------------------------------------------------------------
# Linear-uniform prior transforms
# ---------------------------------------------------------------------------

def make_uniform_prior_transform(V_center: float,
                                  log10_rs_min: float | None = None):
    """4D unit-cube → (V, log10 r_s, log10 rho_s, β̃) with linear-uniform
    priors on r_s and rho_s over the same physical span as ``loguniform``.

    The unit cube maps to *linear* (r_s, rho_s); chain entries for slots
    1 and 2 are returned as ``log10`` of those values to keep the chain
    schema (and ``summarize_posterior``) unchanged across prior choices.

    log10_rs_min overrides the default lower bound on r_s — interpreted
    in log-space so the constraint matches the loguniform variant.
    """
    V_lo = V_center - V_HALFWIDTH
    V_hi = V_center + V_HALFWIDTH
    rs_lo_log = LOG10_RS_BOUNDS[0] if log10_rs_min is None else log10_rs_min
    rs_lo = 10.0 ** rs_lo_log
    rs_hi = 10.0 ** LOG10_RS_BOUNDS[1]
    rhos_lo = 10.0 ** LOG10_RHOS_BOUNDS[0]
    rhos_hi = 10.0 ** LOG10_RHOS_BOUNDS[1]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        x = np.empty_like(u)
        x[0] = V_lo + (V_hi - V_lo) * u[0]
        r_s = rs_lo + (rs_hi - rs_lo) * u[1]
        rho_s = rhos_lo + (rhos_hi - rhos_lo) * u[2]
        x[1] = np.log10(r_s)
        x[2] = np.log10(rho_s)
        x[3] = BETA_TILDE_BOUNDS[0] + (BETA_TILDE_BOUNDS[1] - BETA_TILDE_BOUNDS[0]) * u[3]
        return x

    return prior_transform


def make_uniform_prior_transform_with_nuisances(
    V_center: float,
    d_mean: float, d_sigma: float,
    eps_mean: float, eps_sigma: float,
    rhalf_mean: float, rhalf_sigma: float,
):
    """7D analogue of make_uniform_prior_transform with the same nuisance
    block as the loguniform variant.
    """
    V_lo = V_center - V_HALFWIDTH
    V_hi = V_center + V_HALFWIDTH
    rs_lo = 10.0 ** LOG10_RS_BOUNDS[0]
    rs_hi = 10.0 ** LOG10_RS_BOUNDS[1]
    rhos_lo = 10.0 ** LOG10_RHOS_BOUNDS[0]
    rhos_hi = 10.0 ** LOG10_RHOS_BOUNDS[1]
    eps_a = (0.0 - eps_mean) / eps_sigma
    eps_b = (1.0 - eps_mean) / eps_sigma

    def prior_transform(u: np.ndarray) -> np.ndarray:
        x = np.empty_like(u)
        x[0] = V_lo + (V_hi - V_lo) * u[0]
        r_s = rs_lo + (rs_hi - rs_lo) * u[1]
        rho_s = rhos_lo + (rhos_hi - rhos_lo) * u[2]
        x[1] = np.log10(r_s)
        x[2] = np.log10(rho_s)
        x[3] = BETA_TILDE_BOUNDS[0] + (BETA_TILDE_BOUNDS[1] - BETA_TILDE_BOUNDS[0]) * u[3]
        x[4] = norm.ppf(u[4], loc=d_mean, scale=d_sigma)
        x[5] = truncnorm.ppf(u[5], eps_a, eps_b, loc=eps_mean, scale=eps_sigma)
        x[6] = norm.ppf(u[6], loc=rhalf_mean, scale=rhalf_sigma)
        return x

    return prior_transform


# ---------------------------------------------------------------------------
# Jeffreys log-determinant correction
# ---------------------------------------------------------------------------

def jeffreys_log_term(sigma_los_sq: np.ndarray, T: np.ndarray,
                       sigma_eps_sq: np.ndarray, p: np.ndarray) -> float:
    """½ ln D where D is the (membership-weighted) Fisher determinant
    in (ln ρ_s, ln r_s) at fixed β:

        w̃_i  = A_i² / (A_i + ε_i²)²,        A_i = σ_los²(R_i)
        S0   = Σ p_i w̃_i
        T̄   = Σ p_i w̃_i T_i / S0
        D    = S0 · Σ p_i w̃_i (T_i − T̄)²

    The variance-form is mathematically equivalent to D = S0·S2 − S1²
    but avoids catastrophic cancellation when Var(T) → 0.

    Returns 0.5 * ln D, or -inf if D is non-finite or ≤ 0 (degenerate
    parameter point — the stars do not span a range of R/r_s and r_s
    is unidentifiable).
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


def _zero_correction(sigma_los_sq: np.ndarray, T: np.ndarray,
                      sigma_eps_sq: np.ndarray, p: np.ndarray) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LogCorrection = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Prior:
    """A prior family for the Stage-2 NFW Jeans inference.

    Attributes
    ----------
    name
        Registry key.
    make_transform
        4D builder: (V_center, log10_rs_min=None) -> prior_transform(u).
    make_transform_with_nuisances
        7D builder: (V_center, d_mean, d_sigma, eps_mean, eps_sigma,
        rhalf_mean, rhalf_sigma) -> prior_transform(u).
    log_correction
        (sigma_los_sq, T, sigma_eps_sq, p) -> float. Added to the
        log-likelihood inside the inference driver. Identically zero
        for ``uniform`` and ``loguniform``; the Fisher term for
        ``jeffreys``.
    needs_T
        True if the likelihood must compute the σ_los shape factor T
        on every draw (because ``log_correction`` consumes it).
    """

    name: str
    make_transform: Callable
    make_transform_with_nuisances: Callable
    log_correction: LogCorrection
    needs_T: bool


PRIOR_REGISTRY: dict[str, Prior] = {
    "uniform": Prior(
        name="uniform",
        make_transform=make_uniform_prior_transform,
        make_transform_with_nuisances=make_uniform_prior_transform_with_nuisances,
        log_correction=_zero_correction,
        needs_T=False,
    ),
    "loguniform": Prior(
        name="loguniform",
        make_transform=make_loguniform_prior_transform,
        make_transform_with_nuisances=make_loguniform_prior_transform_with_nuisances,
        log_correction=_zero_correction,
        needs_T=False,
    ),
    "jeffreys": Prior(
        name="jeffreys",
        make_transform=make_loguniform_prior_transform,
        make_transform_with_nuisances=make_loguniform_prior_transform_with_nuisances,
        log_correction=jeffreys_log_term,
        needs_T=True,
    ),
}


def get_prior(name: str) -> Prior:
    if name not in PRIOR_REGISTRY:
        valid = ", ".join(sorted(PRIOR_REGISTRY.keys()))
        raise KeyError(f"unknown prior {name!r}; valid: {valid}")
    return PRIOR_REGISTRY[name]
