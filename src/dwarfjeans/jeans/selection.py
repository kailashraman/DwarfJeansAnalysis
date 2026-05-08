"""Principled per-star selection for Stage-1 Jeans inference.

Three cuts, applied in order:

  1. Membership.  ``p > policy.p_min`` whenever any star has
     ``p < 1.0`` — covers both graded probabilities and 0/1 hard
     flags where the catalog retains non-members tagged ``p == 0``.
     Catalogs whose ``p`` is uniformly 1.0 (already hard-cut
     upstream) or absent skip this cut.
  2. Radial.      ``R < policy.R_over_rhalf_max · r_½`` where ``r_½``
     is the **sphericalized 3D Plummer half-mass radius**:
     ``r_½ = rhalf_major · √(1−ε) · 4/3``. Empirically reproduces
     Paper II Table A1 N* to within ±2 stars for 17 of 22 Path A
     satellites; the 5 outliers (Leo I +15, Leo II +26, Sextans +7,
     UMi −9, Herc −5) trace to Pmem-novar snapshot drift, not the cut.
     ``R`` is the circular projected radius in kpc.
  3. Variability. Drop velocity-variable stars: ``var_flag=True`` from
     the multi-epoch combiner, or ``Var == 1`` for Geha Path A
     catalogs that pre-flag with the integer convention.

Inputs are catalog dicts (or ``np.load(.npz)`` outputs) plus a
registry row exposing ``rhalf_major_pc`` and ``ellipticity``. Returns
``(filtered_catalog, selection_report)``.
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


# Selection cut uses 4/3 (the conventional round approximation to the
# Plummer 3D/2D ratio) per the empirically-tuned recipe that reproduces
# Paper II Table A1 N*. The exact Plummer ratio used by the *derived*
# chains (M(r_½,3D), J/D factors) in scripts/run_production.py is
# 1.30477; the two values differ by 2.2%. They are kept distinct
# deliberately: this constant is a cut-radius multiplier whose value
# was chosen empirically, not derived analytically. Do not unify with
# the derived-chain Plummer factor.
SELECTION_R_CUT_3D_OVER_2D = 4.0 / 3.0


@dataclass(frozen=True)
class SelectionPolicy:
    """Star-selection thresholds. Defaults match the Geha+2026 recipe.

    The radial cut uses ``r_½ = rhalf_major · √(1−ε) · 4/3``, which
    empirically reproduces Paper II Table A1 N* across the Path A sample
    (17 of 22 within ±2 stars; see ``docs/plan/data_sources.md``).
    """

    p_min: float = 0.5
    R_over_rhalf_max: float = 2.0
    drop_variable: bool = True


def _as_array_dict(catalog) -> dict[str, np.ndarray]:
    """Accept either an ``np.lib.npyio.NpzFile`` or a plain dict."""
    if hasattr(catalog, "files"):
        return {k: catalog[k] for k in catalog.files}
    return dict(catalog)


def _membership_carries_info(p: np.ndarray) -> bool:
    """True iff ``p`` carries usable membership information.

    The cut is a no-op only when every star already has ``p == 1.0`` —
    i.e., the catalog has been hard-cut upstream and only members
    remain. Any other case (graded probabilities, or 0/1 hard flags
    where non-members are retained tagged ``p == 0``) means the cut
    must be applied: graded catalogs filter on ``p > policy.p_min``,
    and 0/1 catalogs drop ``p == 0`` for any sensible ``p_min < 1``.
    """
    if p is None or len(p) == 0:
        return False
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return False
    return bool(np.any(finite < 1.0))


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
    if p_arr is not None and _membership_carries_info(p_arr):
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
    # ellipticity must be present in the registry row (sentinel: NaN means
    # "no measurement", treated as eps=0 for sphericalization). A missing
    # KEY is a caller bug — selection.py is the gatekeeper for the cut
    # radius and must never silently default to eps=0 when the caller
    # forgot to plumb the column through. Defends against the silent-
    # fallback failure mode flagged in docs/review-checklist.md.
    if "ellipticity" not in registry_row:
        raise KeyError(
            "registry_row missing 'ellipticity'; selection requires "
            "explicit ellipticity (NaN = unmeasured is OK). Pass the full "
            "registry row, not a stripped dict."
        )
    eps_raw = registry_row["ellipticity"]
    eps_missing = eps_raw is None or (
        isinstance(eps_raw, float) and np.isnan(eps_raw))
    eps = 0.0 if eps_missing else float(eps_raw)
    if not (0.0 <= eps < 1.0):
        raise ValueError(
            f"registry ellipticity must be in [0,1), got {eps_raw!r}"
        )
    rhalf_sph_3d_pc = rhalf_major_pc * np.sqrt(1.0 - eps) * SELECTION_R_CUT_3D_OVER_2D
    R_max_kpc = policy.R_over_rhalf_max * rhalf_sph_3d_pc / 1000.0
    R_kpc = np.asarray(cat["R"], dtype=float)
    keep &= R_kpc < R_max_kpc
    n_after_R = int(keep.sum())
    cuts_applied.append(
        f"R_kpc < {policy.R_over_rhalf_max} * rhalf_sph_3d_pc/1000 = {R_max_kpc:.4g} kpc "
        f"(rhalf_major={rhalf_major_pc:.3g} pc, eps={eps:.3g})"
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
        "ellipticity": eps,
        "ellipticity_missing": bool(eps_missing),
        "rhalf_sph_3d_pc": rhalf_sph_3d_pc,
        "R_max_kpc": R_max_kpc,
    }
    return filtered, report
