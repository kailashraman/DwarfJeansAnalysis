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
    }
    return filtered, audit
