"""Analysis-time preprocessing: raw catalog → Stage-1 inference input.

`prepare_jeans_input` is the single entry point. It:

  1. Reads the catalog's ``_meta["catalog_granularity"]`` flag.
  2. If ``per_epoch``, dispatches to the per-paper combiner via
     ``ingest.combiners.get_combiner`` (keyed by
     ``_meta["source_paper_bibcode"]``); per-star catalogs skip this
     step — the paper's published combination is part of the
     provenance and we don't redo it.
  3. Applies ``select_jeans_stars`` with the supplied
     ``SelectionPolicy``.
  4. Returns the filtered per-star arrays plus an audit dict
     recording both policies and the per-stage row counts. The audit
     dict is the canonical artifact to drop next to a chain so a run
     is reproducible from the raw ``.npz`` plus the recorded
     policies.

The combiner-side and selection-side knobs are exposed as separate
policies on purpose: someone running an ablation on the variability
threshold should not also have to think about the membership cut.

Per-star catalogs do not consume the ``CombinePolicy`` argument; it
is accepted unconditionally so callers can write the same code path
for either granularity.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from dwarfjeans.ingest.combiners import CombinePolicy, get_combiner
from dwarfjeans.jeans.perspective import perspective_correction, sanity_check
from dwarfjeans.jeans.selection import SelectionPolicy, select_jeans_stars


def _as_array_dict(catalog) -> dict[str, np.ndarray]:
    if hasattr(catalog, "files"):
        return {k: catalog[k] for k in catalog.files}
    return dict(catalog)


def _read_meta(catalog: dict) -> dict:
    """Return the parsed ``_meta`` dict, or {} if absent/unparseable."""
    blob = catalog.get("_meta")
    if blob is None:
        return {}
    try:
        payload = blob.item() if hasattr(blob, "item") else blob
        if isinstance(payload, dict):
            return payload
        return json.loads(str(payload))
    except (ValueError, TypeError):
        return {}


def prepare_jeans_input(
    catalog: Any,
    registry_row: Any,
    selection_policy: SelectionPolicy = SelectionPolicy(),
    combine_policy: CombinePolicy | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Combine (if per-epoch) and select stars for Stage-1 inference.

    Parameters
    ----------
    catalog
        ``np.lib.npyio.NpzFile`` (typically from
        ``np.load("data/star_catalogs/<key>.npz")``) or a plain dict
        of equal-length arrays plus an optional ``_meta`` blob.
    registry_row
        Mapping with at least ``rhalf_major_pc``. Passed through to
        ``select_jeans_stars`` and to the combiner.
    selection_policy
        Cuts applied after combination. Defaults to the Geha+2026
        recipe.
    combine_policy
        Per-call combiner tunables (σ_sys floor, χ² p-threshold).
        If None, the registry's default policy for the catalog's
        source paper is used. Ignored entirely for per-star catalogs.

    Returns
    -------
    arrays : dict
        Per-star arrays after combination (if any) and selection.
    audit : dict
        ``{
            "granularity":      str,                 # 'per_star' or 'per_epoch'
            "n_input_rows":     int,
            "combine":          dict | None,         # combiner diagnostics
            "selection":        dict,                # selection report
            "selection_policy": dict,
            "combine_policy":   dict | None,
            "source_paper_bibcode": str | None,
          }``
    """
    cat = _as_array_dict(catalog)
    meta = _read_meta(cat)
    granularity = meta.get("catalog_granularity", "per_star")
    bibcode = meta.get("source_paper_bibcode")

    n_input_rows = int(len(cat["R"])) if "R" in cat else int(
        len(next(iter(v for v in cat.values()
                     if hasattr(v, "shape") and v.ndim >= 1), []))
    )

    combine_diag: dict | None = None
    combine_policy_used: CombinePolicy | None = None

    if granularity == "per_epoch":
        if bibcode is None:
            raise ValueError(
                "per-epoch catalog missing _meta['source_paper_bibcode']; "
                "combiner dispatch requires it."
            )
        combine_fn, default_policy = get_combiner(bibcode)
        combine_policy_used = combine_policy if combine_policy is not None else default_policy
        # Drop _meta from per-epoch input so the combiner sees only data columns.
        per_epoch_arrays = {k: v for k, v in cat.items() if k != "_meta"}
        per_star, combine_diag = combine_fn(
            per_epoch_arrays, registry_row, combine_policy_used
        )
        # Carry _meta through with the granularity flipped to per_star so
        # downstream selection accepts it (R_unit and other annotations
        # are preserved).
        if "_meta" in cat:
            new_meta = dict(meta)
            new_meta["catalog_granularity"] = "per_star"
            new_meta["combined_from"] = "per_epoch"
            per_star["_meta"] = np.array(json.dumps(new_meta), dtype=object)
        # The combiner does not recompute R per star; per-epoch R values
        # for one star are identical (R is computed from RA/Dec and a
        # fixed registry center), so the first-epoch passthrough in the
        # default combiner produces the correct per-star R.
        cat_for_selection = per_star
    elif granularity == "per_star":
        cat_for_selection = cat
    else:
        raise ValueError(
            f"unknown catalog_granularity {granularity!r} (expected 'per_star' or 'per_epoch')"
        )

    filtered, sel_report = select_jeans_stars(
        cat_for_selection, registry_row, selection_policy
    )

    persp_audit = _apply_perspective(filtered, meta, registry_row)

    audit = {
        "granularity": granularity,
        "n_input_rows": n_input_rows,
        "combine": combine_diag,
        "selection": sel_report,
        "selection_policy": {
            "p_min": selection_policy.p_min,
            "R_over_rhalf_max": selection_policy.R_over_rhalf_max,
            "drop_variable": selection_policy.drop_variable,
        },
        "combine_policy": (
            None if combine_policy_used is None else {
                "sigma_sys_kms": combine_policy_used.sigma_sys_kms,
                "p_threshold": combine_policy_used.p_threshold,
                "zero_point_offsets_kms": combine_policy_used.zero_point_offsets_kms,
            }
        ),
        "source_paper_bibcode": bibcode,
        "perspective": persp_audit,
    }
    return filtered, audit


def _registry_get(registry_row, key: str, default=None):
    """Read a key from either a dict or an astropy Table row."""
    try:
        v = registry_row[key]
    except (KeyError, IndexError):
        return default
    return v if v is not None else default


def _apply_perspective(filtered: dict, meta: dict, registry_row) -> dict:
    """Subtract Δv_persp from filtered['V'] in place; return audit sub-dict.

    Skips silently (records `applied=False` with a reason) when PM
    metadata is absent, when ``perspective_correction_applicable`` is
    False, or when per-star RA/Dec are missing — so legacy npz files
    (pre-PM-ingest) continue to flow through unmodified.
    """
    base = {
        "applied": False,
        "reason": "",
        "pm_alpha_star_masyr": None,
        "pm_delta_masyr": None,
        "ref_proper_motion": None,
        "rms_kms": 0.0,
        "max_abs_kms": 0.0,
        "quadratic_term_max_abs_kms": 0.0,
        "small_vs_full_residual_kms": 0.0,
    }

    if not meta.get("perspective_correction_applicable", False):
        base["reason"] = "_meta.perspective_correction_applicable is False or absent"
        return base
    if "RA_star" not in filtered or "Dec_star" not in filtered:
        base["reason"] = "per-star RA_star/Dec_star not in catalog"
        return base
    if filtered["V"].size == 0:
        base["reason"] = "empty post-selection sample"
        return base

    pmra = meta.get("lvdb_pmra_mas_yr")
    pmdec = meta.get("lvdb_pmdec_mas_yr")
    if pmra is None or pmdec is None:
        base["reason"] = "_meta missing lvdb_pmra/pmdec"
        return base

    ra0 = _registry_get(registry_row, "ra_deg")
    dec0 = _registry_get(registry_row, "dec_deg")
    d_kpc = _registry_get(registry_row, "distance_kpc")
    if ra0 is None or dec0 is None or d_kpc is None:
        base["reason"] = "registry_row missing ra_deg/dec_deg/distance_kpc"
        return base
    ra0, dec0, d_kpc = float(ra0), float(dec0), float(d_kpc)
    v_sys_raw = _registry_get(registry_row, "vlos_systemic_kms", 0.0)
    v_sys = float(v_sys_raw) if v_sys_raw is not None else 0.0
    if not np.isfinite(v_sys):  # NaN systemic → diagnostic only; don't propagate NaN
        v_sys = 0.0

    dv = perspective_correction(
        ra_deg=filtered["RA_star"], dec_deg=filtered["Dec_star"],
        ra_center_deg=ra0, dec_center_deg=dec0,
        distance_kpc=d_kpc,
        pm_alpha_star_masyr=float(pmra), pm_delta_masyr=float(pmdec),
    )
    filtered["V_observed"] = filtered["V"].copy()
    filtered["V"] = filtered["V_observed"] - dv

    rep = sanity_check(
        ra_deg=filtered["RA_star"], dec_deg=filtered["Dec_star"],
        ra_center_deg=ra0, dec_center_deg=dec0,
        distance_kpc=d_kpc,
        pm_alpha_star_masyr=float(pmra), pm_delta_masyr=float(pmdec),
        v_sys_kms=v_sys,
    )

    return {
        "applied": True,
        "reason": "",
        "pm_alpha_star_masyr": float(pmra),
        "pm_delta_masyr": float(pmdec),
        "ref_proper_motion": meta.get("lvdb_ref_proper_motion"),
        "rms_kms": rep.rms_kms,
        "max_abs_kms": rep.max_abs_kms,
        "quadratic_term_max_abs_kms": rep.quadratic_term_max_abs_kms,
        "small_vs_full_residual_kms": rep.small_vs_full_residual_kms,
    }
