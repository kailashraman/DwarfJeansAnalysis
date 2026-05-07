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
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as scipy_chi2


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
        IVW-combined uncertainty.

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
