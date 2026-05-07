"""Dataset-agnostic primitives for combining per-epoch line-of-sight
velocity measurements into per-star arrays.

Two operations:

  ``ivw_combine``        — inverse-variance-weighted mean velocity and
                            its uncertainty, with an optional systematic
                            error floor added in quadrature *after* the
                            IVW.

  ``variability_chi2``    — χ² test for velocity variability against the
                            IVW mean, returning a binary flag at a
                            user-specified p-value threshold.

Per-instrument zero-point offsets must be applied to the input
velocities by the caller (typically the per-dataset combiner in
``ingest/combiners/<dataset>.py``) *before* invoking ``ivw_combine``.
The systematic floor passed in here is the post-combine instrument
floor (e.g. 1.1 km/s for DEIMOS multi-slit), added in quadrature to
the IVW uncertainty.

Single-epoch input is handled gracefully: ``ivw_combine`` returns the
single value with σ floored as ``sqrt(σ² + σ_sys²)``;
``variability_chi2`` returns ``var_flag=False`` and ``p_value=NaN`` so
the star is left unflagged.

----

**Convention: σ_sys treated as statistical** (decided 2026-05-07).

Every UFD spectroscopy paper we ingest (Li+2017, Li+2018, Chiti+2022,
Chiti+2023, Simon+2020, ...) reports per-epoch errors that already
have a systematic floor σ_sys added in quadrature with the statistical
error: ``e_RVel² = σ_stat² + σ_sys²``. Strictly, σ_sys is an
*instrument-correlated* term and should not be averaged down by 1/√N
along with σ_stat in the IVW; the textbook treatment is to deconvolve
σ_sys from each epoch's error, IVW the σ_stat values, then re-add σ_sys
in quadrature to the combined uncertainty.

By default this module does **not** do that deconvolution: with
``CombinePolicy.sigma_sys_kms = 0`` (the convention used by every
per-paper handler today) we feed published ``e_RVel`` straight into
``ivw_combine`` and treat σ_sys as if it were statistical. Strict
deconvolution is opt-in: setting ``CombinePolicy.sigma_sys_kms`` to
the published σ_sys value routes ``default.combine`` through
``combine_star_strict`` (this module), which deconvolves σ_stat from
σ_total per-epoch, IVWs σ_stat, and re-adds σ_sys post-combine. The
χ² variability test in the strict path also runs on σ_stat (not
σ_total) so it is not under-powered against epoch-to-epoch scatter.

Consequence of the *default* path: the combined uncertainty
``σ_vbar`` is biased *low* by a factor approaching
``σ_stat / sqrt(σ_stat² + σ_sys²)`` at large N (typical bias ~10–30%
for N=2–5 epochs at σ_sys ~ σ_stat). χ² / p-values for variability
flagging are biased correspondingly *high* (too few stars flagged
variable). The bias is conservative-low on σ_los — inferred velocity
dispersions absorb it as scatter — and small for the multi-epoch
counts we currently see (median N=2–3 across the seven per-epoch
catalogs).

Why we accept this approximation as the default:

  - Per-paper σ_sys values are well-bounded (0.5–1.2 km/s) and
    documented in each handler's docstring + ``docs/plan/per_paper_combiners.md``.
  - For the seven per-epoch catalogs we currently ingest, the strict
    treatment moves σ_vbar by less than the chain noise on σ_los at
    Stage-1 precision.
  - For galaxies where it would matter (small-N; see caveat below)
    the per-paper handler can opt in by setting
    ``CombinePolicy.sigma_sys_kms`` to the published σ_sys.

If a future analysis is sensitive to the σ_vbar bias (e.g. precision
σ_los measurements at the 0.1 km/s level), revisit this. For the
current Jeans inference at σ_los ~ a few km/s, the bias is well below
the chain-noise floor.

**Caveat — small-sample galaxies.** The "below chain noise" claim
assumes ≳5 surviving members per galaxy, where per-star σ_vbar errors
enter the σ_los likelihood quadratically alongside the dispersion
itself and a 10–30% per-star error inflation produces a much smaller
σ_los bias. For galaxies with ≤3 selected members (e.g. Tucana V
after dropping the binary Tuc V-1, leaving 2 stars), per-star σ_vbar
is a leading contributor to σ_los and the bound does not apply.
For these galaxies either swap in the strict deconvolution path or
propagate the bias explicitly into the systematic-error budget.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as scipy_chi2


def ivw_combine_strict(v: np.ndarray,
                        sigma_total: np.ndarray,
                        sigma_sys: float) -> tuple[float, float]:
    """Strict-deconvolution IVW: treat sigma_sys as a fully-correlated
    instrument floor rather than a statistical error.

    Decomposes each per-epoch error as
    ``sigma_stat² = sigma_total² − sigma_sys²``, IVWs on ``sigma_stat``,
    then re-adds ``sigma_sys`` in quadrature post-combine. This is the
    textbook treatment of a correlated systematic; the default
    σ_sys-as-statistical pipeline (see ``ivw_combine`` and the module
    docstring) is biased low for σ_vbar by 10–30% at typical N=2–5
    epochs because it averages σ_sys down by 1/√N along with σ_stat.

    Use this when per-star σ_vbar is a leading contributor to the
    σ_los inference (small-N galaxies; e.g. Tuc V with 2–3 surviving
    members).

    Parameters
    ----------
    v
        Per-epoch velocities (km/s).
    sigma_total
        Per-epoch total uncertainties (km/s) AS-PUBLISHED — i.e.
        sqrt(stat² + sys²). Strictly positive and strictly larger
        than ``sigma_sys`` (else the implied stat error is imaginary
        and we raise — that itself is a flag against the input).
    sigma_sys
        Published systematic floor (km/s).

    Returns
    -------
    v_bar : float
    sigma_vbar : float
    """
    v = np.asarray(v, dtype=float)
    sigma_total = np.asarray(sigma_total, dtype=float)
    if v.shape != sigma_total.shape:
        raise ValueError(
            f"v and sigma_total shape mismatch: {v.shape} vs {sigma_total.shape}"
        )
    if v.size == 0:
        raise ValueError("ivw_combine_strict: empty input")
    if sigma_sys < 0:
        raise ValueError(f"ivw_combine_strict: sigma_sys must be ≥0, got {sigma_sys}")
    sigma_stat_sq = sigma_total ** 2 - sigma_sys ** 2
    if not np.all(sigma_stat_sq > 0.0):
        bad = np.where(sigma_stat_sq <= 0.0)[0]
        raise ValueError(
            f"ivw_combine_strict: sigma_total² must exceed sigma_sys² for "
            f"every epoch (else implied σ_stat is imaginary). Failed at "
            f"indices {bad.tolist()}: total²={sigma_total[bad]**2}, "
            f"sys²={sigma_sys**2}. Either the published σ_sys is wrong or "
            f"the per-epoch σ_total is."
        )
    w = 1.0 / sigma_stat_sq
    W = w.sum()
    v_bar = float((w * v).sum() / W)
    sigma_vbar = float(np.sqrt(1.0 / W + sigma_sys ** 2))
    return v_bar, sigma_vbar


def ivw_combine(v: np.ndarray,
                 sigma: np.ndarray,
                 sigma_sys: float = 0.0) -> tuple[float, float]:
    """Inverse-variance-weighted mean of per-epoch velocities.

    .. math::
        \\bar v = \\frac{\\sum_i v_i / \\sigma_i^2}{\\sum_i 1 / \\sigma_i^2},
        \\quad \\sigma_{\\bar v} = \\sqrt{\\frac{1}{\\sum_i 1/\\sigma_i^2}
                                         + \\sigma_\\text{sys}^2}.

    Parameters
    ----------
    v
        Per-epoch velocities (km/s). Caller must apply per-instrument
        zero-point offsets before passing in.
    sigma
        Per-epoch 1σ uncertainties (km/s), strictly positive.
    sigma_sys
        Systematic error floor (km/s), added in quadrature to the
        IVW-combined uncertainty *post*-combine — assumes the supplied
        ``sigma`` is σ_stat (i.e. σ_sys already removed). For
        deconvolution from σ_total (the published per-epoch error in
        every catalog we ingest), use ``ivw_combine_strict`` instead.

    Returns
    -------
    v_bar : float
    sigma_vbar : float
    """
    v = np.asarray(v, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if v.shape != sigma.shape:
        raise ValueError(f"v and sigma shape mismatch: {v.shape} vs {sigma.shape}")
    if v.size == 0:
        raise ValueError("ivw_combine: empty input")
    if not np.all(sigma > 0.0):
        raise ValueError("ivw_combine: sigma must be strictly positive")

    w = 1.0 / sigma ** 2
    W = w.sum()
    v_bar = float((w * v).sum() / W)
    sigma_vbar = float(np.sqrt(1.0 / W + sigma_sys ** 2))
    return v_bar, sigma_vbar


def variability_chi2(v: np.ndarray,
                      sigma: np.ndarray,
                      v_bar: float,
                      p_threshold: float = 0.01,
                      ) -> tuple[float, int, float, bool]:
    """χ² test for velocity variability against the IVW mean.

    Compares observed scatter to the per-epoch errors:

    .. math::
        \\chi^2 = \\sum_i \\frac{(v_i - \\bar v)^2}{\\sigma_i^2},
        \\quad \\text{dof} = N - 1.

    Parameters
    ----------
    v
        Per-epoch velocities (km/s). Should already have the same
        zero-point convention used to compute ``v_bar``.
    sigma
        Per-epoch 1σ uncertainties (km/s). The systematic floor
        passed to ``ivw_combine`` should *not* be applied here — the
        χ² test is on the raw per-epoch scatter.
    v_bar
        IVW mean (typically from ``ivw_combine``).
    p_threshold
        Reject-the-null threshold. ``var_flag`` is True iff
        ``p_value < p_threshold``.

    Returns
    -------
    chi2 : float
        Test statistic (NaN for single-epoch input).
    dof : int
        Degrees of freedom (0 for single-epoch).
    p_value : float
        Upper-tail p-value (NaN for single-epoch).
    var_flag : bool
        True iff the star is flagged as variable. Single-epoch stars
        are left unflagged (``var_flag=False``).
    """
    v = np.asarray(v, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = v.size
    if n <= 1:
        return float("nan"), 0, float("nan"), False
    chi2_stat = float(np.sum((v - v_bar) ** 2 / sigma ** 2))
    dof = n - 1
    p_value = float(scipy_chi2.sf(chi2_stat, dof))
    var_flag = bool(p_value < p_threshold)
    return chi2_stat, dof, p_value, var_flag


def combine_star(v: np.ndarray,
                  sigma: np.ndarray,
                  sigma_sys: float = 0.0,
                  p_threshold: float = 0.01,
                  ) -> dict:
    """Convenience: run ``ivw_combine`` + ``variability_chi2`` for one
    star's per-epoch arrays. Returns a per-star dict with keys

        v_bar, sigma_vbar, n_epoch, chi2, dof, p_value, var_flag.
    """
    v_bar, sigma_vbar = ivw_combine(v, sigma, sigma_sys=sigma_sys)
    chi2_stat, dof, p_value, var_flag = variability_chi2(
        v, sigma, v_bar, p_threshold=p_threshold
    )
    return {
        "v_bar": v_bar,
        "sigma_vbar": sigma_vbar,
        "n_epoch": int(v.size),
        "chi2": chi2_stat,
        "dof": dof,
        "p_value": p_value,
        "var_flag": var_flag,
    }


def combine_star_strict(v: np.ndarray,
                         sigma_total: np.ndarray,
                         sigma_sys: float,
                         p_threshold: float = 0.01,
                         ) -> dict:
    """Strict-deconvolution variant of ``combine_star``.

    Treats σ_sys as a fully-correlated instrument floor: deconvolves
    σ_stat = sqrt(σ_total² − σ_sys²), runs IVW on σ_stat, re-adds
    σ_sys post-combine. The χ² variability test runs on σ_stat (NOT
    σ_total), since the test is on epoch-to-epoch scatter relative to
    the *statistical* error — adding σ_sys in quadrature there would
    under-flag variables.

    Use when σ_vbar bias from the σ_sys-as-statistical approximation
    is unacceptable (typically small-N galaxies where per-star σ_vbar
    leads the σ_los inference). See ``ivw_combine_strict`` for math
    and the module docstring for the convention discussion.
    """
    v_bar, sigma_vbar = ivw_combine_strict(v, sigma_total, sigma_sys=sigma_sys)
    sigma_stat = np.sqrt(np.asarray(sigma_total, dtype=float) ** 2 - sigma_sys ** 2)
    chi2_stat, dof, p_value, var_flag = variability_chi2(
        v, sigma_stat, v_bar, p_threshold=p_threshold
    )
    return {
        "v_bar": v_bar,
        "sigma_vbar": sigma_vbar,
        "n_epoch": int(np.asarray(v).size),
        "chi2": chi2_stat,
        "dof": dof,
        "p_value": p_value,
        "var_flag": var_flag,
    }
