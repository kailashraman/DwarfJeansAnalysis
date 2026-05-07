"""Default per-epoch → per-star combiner.

Groups input per-epoch arrays by ``star_id`` (same convention used by
``staging.per_star_indices``), runs IVW + χ² per group, and emits the
canonical per-star schema. The caller's ``CombinePolicy`` carries the
σ_sys floor, the variability p-threshold, and (if non-empty) a
per-instrument zero-point offset table applied before the IVW.

Dataset-specific behavior beyond per-instrument offsets (binary-aware
σ-deconvolution, etc.) goes in a per-paper module that preprocesses the
per_epoch dict and then calls ``combine``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dwarfjeans.ingest.multi_epoch import combine_star, combine_star_strict


PER_STAR_REQUIRED = ("star_id", "V", "sigma_eps", "p")


def combine(per_epoch: dict, registry_row: Any, policy) -> tuple[dict, dict]:
    """Generic IVW+χ² combiner.

    Parameters
    ----------
    per_epoch
        Dict of equal-length arrays (the adapter's output). Required
        keys: ``star_id``, ``V``, ``sigma_eps``, ``p``. All other
        equal-length columns are passed through using the *first*
        epoch's value per star.
    registry_row
        Unused by the default handler. Reserved for handlers that
        need geometric or distance-dependent corrections.
    policy
        ``CombinePolicy`` instance.

    Returns
    -------
    per_star : dict of arrays
        New schema with ``v_bar``-derived ``V``/``sigma_eps`` plus the
        ``n_epoch`` and ``var_flag`` columns. Length == n_unique_stars.
    diagnostics : dict
        Per-call summary: number of stars, number of variable flags,
        median n_epoch.
    """
    for k in PER_STAR_REQUIRED:
        if k not in per_epoch:
            raise KeyError(f"per_epoch input missing required column {k!r}")

    star_id = np.asarray(per_epoch["star_id"])
    n_rows = star_id.size
    if n_rows == 0:
        raise ValueError("default combiner: empty input")

    unique_ids, first_idx, inverse = np.unique(
        star_id, return_index=True, return_inverse=True
    )
    n_stars = unique_ids.size

    # Apply per-instrument zero-point offsets *before* the IVW. The
    # policy's offsets dict is the only authoritative source — silent
    # zero-offset fallback on missing tags is the bug this guard exists
    # to prevent, so we raise rather than .get(tag, 0.0).
    V_in = np.asarray(per_epoch["V"], dtype=float).copy()
    # None / empty mapping → no offsets applied (pre-patch back-compat).
    offsets = getattr(policy, "zero_point_offsets_kms", None) or {}
    if offsets:
        if "Inst" not in per_epoch:
            raise KeyError(
                "policy.zero_point_offsets_kms is non-empty but per_epoch "
                "lacks an 'Inst' column; per-instrument offsets cannot be "
                "applied. Either drop the offsets from the policy or "
                "extend the adapter to emit Inst per epoch."
            )
        inst = np.asarray(per_epoch["Inst"])
        unknown = set(np.unique(inst).tolist()) - set(offsets.keys())
        if unknown:
            raise KeyError(
                f"policy.zero_point_offsets_kms missing entries for "
                f"instruments {sorted(unknown)!r}; refusing to apply a "
                f"silent zero offset. Add them explicitly (use 0.0 for "
                f"the reference instrument)."
            )
        shifts = np.array([offsets[t] for t in inst], dtype=float)
        V_in = V_in + shifts

    # Output buffers
    V_out = np.empty(n_stars, dtype=float)
    sigma_out = np.empty(n_stars, dtype=float)
    n_epoch_out = np.empty(n_stars, dtype=int)
    var_flag_out = np.zeros(n_stars, dtype=bool)
    chi2_out = np.full(n_stars, np.nan, dtype=float)
    p_value_out = np.full(n_stars, np.nan, dtype=float)

    # Group rows by unique star
    sigma_sys = float(getattr(policy, "sigma_sys_kms", 0.0))
    p_thresh = float(getattr(policy, "p_threshold", 0.01))
    # Routing: sigma_sys > 0 opts the handler into the strict
    # σ_sys-deconvolution path (see multi_epoch.combine_star_strict).
    # Default (sigma_sys = 0) is the σ_sys-as-statistical convention.
    # NB: this overloads sigma_sys_kms — there is intentionally no way
    # to request "as-statistical convention with a post-combine
    # quadrature add" through default.combine; that combination would
    # double-count under our σ_sys-already-in-published-e_RVel
    # premise. If a future caller legitimately needs it, add an
    # explicit `sigma_sys_treatment` field to CombinePolicy.
    use_strict = sigma_sys > 0.0

    for k in range(n_stars):
        rows = np.where(inverse == k)[0]
        v = V_in[rows]
        sigma = np.asarray(per_epoch["sigma_eps"], dtype=float)[rows]
        # Drop rows with non-finite v / sigma (defensive: shouldn't happen
        # post-adapter, but guard anyway)
        mask = np.isfinite(v) & np.isfinite(sigma) & (sigma > 0.0)
        if not mask.any():
            raise ValueError(
                f"star_id={unique_ids[k]!r}: no valid (V, sigma_eps) rows"
            )
        v = v[mask]
        sigma = sigma[mask]
        if use_strict:
            out = combine_star_strict(
                v, sigma, sigma_sys=sigma_sys, p_threshold=p_thresh
            )
        else:
            out = combine_star(v, sigma, sigma_sys=sigma_sys, p_threshold=p_thresh)
        V_out[k] = out["v_bar"]
        sigma_out[k] = out["sigma_vbar"]
        n_epoch_out[k] = out["n_epoch"]
        var_flag_out[k] = out["var_flag"]
        chi2_out[k] = out["chi2"]
        p_value_out[k] = out["p_value"]

    # Pass-through columns: take the value from the *first* row of each star.
    # This works because per-star quantities (RA/Dec/star_id_source/p) are
    # constant within a group; per-epoch quantities (MJD/Inst/SN) are arbitrary
    # and the diagnostic value is in the first row.
    per_star = {
        "star_id": unique_ids,
        "V": V_out,
        "sigma_eps": sigma_out,
        "n_epoch": n_epoch_out,
        "var_flag": var_flag_out,
        "chi2": chi2_out,
        "chi2_p_value": p_value_out,
    }
    for col, arr in per_epoch.items():
        if col in per_star or col in ("V", "sigma_eps"):
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 1 and arr.shape[0] == n_rows:
            per_star[col] = arr[first_idx]
        # Skip non-row-aligned entries (shouldn't appear in adapter output).

    diagnostics = {
        "n_input_rows": n_rows,
        "n_stars": n_stars,
        "n_variable": int(var_flag_out.sum()),
        "median_n_epoch": int(np.median(n_epoch_out)),
        "max_n_epoch": int(n_epoch_out.max()),
        "sigma_sys_kms": sigma_sys,
        "sigma_sys_treatment": "strict_deconvolution" if use_strict else "as_statistical",
        "p_threshold": p_thresh,
        "zero_point_offsets_kms": dict(offsets),
    }
    return per_star, diagnostics
