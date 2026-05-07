"""Principled per-star selection for Stage-1 Jeans inference.

Three cuts, applied in order:

  1. Membership.  ``p > policy.p_min`` if the catalog has graded
     membership probabilities. Catalogs whose ``p`` is uniformly 1.0
     (already hard-cut upstream) or absent skip this cut.
  2. Radial.      ``R < policy.R_over_rhalf_max · r_½`` where ``r_½``
     is the **semi-major axis** half-light radius taken from the
     registry's ``rhalf_major_pc``. ``R`` is the projected radius in
     kpc (`staging.projected_radius_kpc`).
  3. Variability. Drop velocity-variable stars: ``var_flag=True`` from
     the multi-epoch combiner, or ``Var == 1`` for Geha Path A
     catalogs that pre-flag with the integer convention.

Inputs are catalog dicts (or ``np.load(.npz)`` outputs) plus a
registry row exposing ``rhalf_major_pc``. Returns ``(filtered_catalog,
selection_report)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _read_meta_field(meta_obj, key: str) -> str | None:
    """Extract a string field from a catalog's ``_meta`` blob if present
    and parseable. Returns None on missing/unparseable input — the
    caller decides whether to enforce."""
    if meta_obj is None:
        return None
    try:
        if hasattr(meta_obj, "item"):
            payload = meta_obj.item()
        else:
            payload = meta_obj
        if isinstance(payload, dict):
            meta = payload
        else:
            meta = json.loads(str(payload))
    except (ValueError, TypeError):
        return None
    val = meta.get(key) if isinstance(meta, dict) else None
    return val if isinstance(val, str) else None


@dataclass(frozen=True)
class SelectionPolicy:
    """Star-selection thresholds. Defaults match the Geha+2026 recipe."""

    p_min: float = 0.5
    R_over_rhalf_max: float = 2.0
    drop_variable: bool = True


def _as_array_dict(catalog) -> dict[str, np.ndarray]:
    """Accept either an ``np.lib.npyio.NpzFile`` or a plain dict."""
    if hasattr(catalog, "files"):
        return {k: catalog[k] for k in catalog.files}
    return dict(catalog)


def _has_graded_membership(p: np.ndarray) -> bool:
    """True iff ``p`` carries information beyond a hard 0/1 cut."""
    if p is None or len(p) == 0:
        return False
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return False
    # Strictly between 0 and 1 means a graded probability is present.
    return bool(np.any((finite > 0.0) & (finite < 1.0)))


def _normalize_variability(catalog: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """Return a boolean ``var_flag`` array unifying the two conventions.

    - ``var_flag``  (preferred): boolean produced by the combiner.
    - ``Var``       (Geha Path A): float; 1.0 means variable.

    Returns None if neither column is present.
    """
    if "var_flag" in catalog:
        return np.asarray(catalog["var_flag"], dtype=bool)
    if "Var" in catalog:
        Var = np.asarray(catalog["Var"], dtype=float)
        return Var == 1.0
    return None


def select_jeans_stars(
    catalog: Any,
    registry_row: Mapping[str, Any],
    policy: SelectionPolicy = SelectionPolicy(),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the principled selection cuts. Returns (catalog, report).

    Parameters
    ----------
    catalog
        Either an ``np.lib.npyio.NpzFile`` (e.g. result of
        ``np.load("data/star_catalogs/<key>.npz")``) or a dict of
        equal-length 1D arrays. Required keys: ``R``. Optional but
        consumed: ``p``, ``var_flag`` / ``Var``.
    registry_row
        Mapping with at least ``rhalf_major_pc`` (semi-major axis
        half-light radius in pc).
    policy
        Cut thresholds.
    """
    cat = _as_array_dict(catalog)
    if "R" not in cat:
        raise KeyError("catalog missing required 'R' column")
    n_input = len(cat["R"])

    # Granularity guard: selection treats every row as one star. Running
    # it on a per-epoch catalog would silently apply the radial cut to
    # individual epochs and double-count stars in the survivor set. The
    # caller must combine first (see jeans.preprocess.prepare_jeans_input).
    granularity = _read_meta_field(cat.get("_meta"), "catalog_granularity")
    if granularity == "per_epoch":
        raise ValueError(
            "select_jeans_stars on a per-epoch catalog would treat each "
            "epoch as a separate star. Combine first via "
            "jeans.preprocess.prepare_jeans_input, or use the combiner "
            "directly from ingest.combiners."
        )

    # Unit-consistency check: if the catalog declares an R unit in its
    # _meta dict, assert it is kpc (the only convention this module
    # supports). Catalogs without the annotation pass through.
    r_unit = _read_meta_field(cat.get("_meta"), "R_unit")
    if r_unit is not None and r_unit != "kpc":
        raise ValueError(
            f"catalog has R_unit={r_unit!r}; selection requires kpc. "
            "Re-ingest to standardize."
        )

    keep = np.ones(n_input, dtype=bool)
    cuts_applied: list[str] = []
    n_after_p = n_input
    n_after_R = n_input
    n_after_var = n_input
    membership_noop = True

    p_arr = np.asarray(cat["p"], dtype=float) if "p" in cat else None
    if p_arr is not None and _has_graded_membership(p_arr):
        keep &= p_arr > policy.p_min
        n_after_p = int(keep.sum())
        cuts_applied.append(f"p > {policy.p_min}")
        membership_noop = False
    else:
        cuts_applied.append("p: no-op (uniform/absent)")

    rhalf_major_pc = float(registry_row["rhalf_major_pc"])
    if not np.isfinite(rhalf_major_pc) or rhalf_major_pc <= 0.0:
        raise ValueError(
            f"registry rhalf_major_pc must be positive finite, got {rhalf_major_pc!r}"
        )
    R_max_kpc = policy.R_over_rhalf_max * rhalf_major_pc / 1000.0
    R_kpc = np.asarray(cat["R"], dtype=float)
    keep &= R_kpc < R_max_kpc
    n_after_R = int(keep.sum())
    cuts_applied.append(
        f"R_kpc < {policy.R_over_rhalf_max} * rhalf_major_pc/1000 = {R_max_kpc:.4g} kpc"
    )

    var_flag = _normalize_variability(cat)
    if policy.drop_variable and var_flag is not None:
        keep &= ~var_flag
        n_after_var = int(keep.sum())
        cuts_applied.append("var_flag=False (or Var != 1)")
    else:
        cuts_applied.append("variability: no-op")

    def _filter(v):
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_input:
            return arr[keep]
        return v
    filtered = {k: _filter(v) for k, v in cat.items()}

    report = {
        "n_input": n_input,
        "n_after_p": n_after_p,
        "n_after_R": n_after_R,
        "n_after_var": n_after_var,
        "n_final": int(keep.sum()),
        "cuts_applied": cuts_applied,
        "membership_noop": membership_noop,
        "policy": {
            "p_min": policy.p_min,
            "R_over_rhalf_max": policy.R_over_rhalf_max,
            "drop_variable": policy.drop_variable,
        },
        "rhalf_major_pc": rhalf_major_pc,
        "R_max_kpc": R_max_kpc,
    }
    return filtered, report
