"""Per-dataset multi-epoch combiners.

Each combiner produces the per-star schema expected by the downstream
Jeans analysis:

    star_id, RA_star, Dec_star, V, sigma_eps, p, R, n_epoch, var_flag

plus dataset-specific auxiliary columns. Combiners are organized
**per dataset, not per dwarf**: the procedure (zero-point offsets,
σ_sys floor, p-threshold) is a property of the survey/instrument, not
of the target galaxy.

Dispatch is keyed by ``source_paper_bibcode`` (already stamped into
each adapter's ``meta_extra``). The ``default`` handler does IVW + χ²
with no zero-point offsets and zero systematic floor — adequate for
papers whose published velocities are already on a single instrument's
zero-point and whose error bars include the survey's systematic
budget. Per-dataset handlers override either when needed.
"""

from __future__ import annotations

from typing import Any, Callable

from . import default as _default

CombineFn = Callable[[dict, Any, "CombinePolicy"], tuple[dict, dict]]


from dataclasses import dataclass


@dataclass(frozen=True)
class CombinePolicy:
    """Tunables shared across handlers. Dataset-specific overrides go in
    the per-dataset module."""

    sigma_sys_kms: float = 0.0     # added in quadrature *after* the IVW
    p_threshold: float = 0.01      # variability χ² p-value threshold
    zero_point_offsets_kms: dict = None  # per-instrument additive offsets


# Registry: source_paper_bibcode → (combine_fn, default_policy)
COMBINER_REGISTRY: dict[str, tuple[CombineFn, CombinePolicy]] = {
    # No paper-specific entries yet — default handler covers all 7
    # currently per-epoch catalogs with IVW + χ² and zero σ_sys.
}


def get_combiner(bibcode: str) -> tuple[CombineFn, CombinePolicy]:
    """Return ``(combine_fn, default_policy)`` for the given source paper.

    Falls back to ``default`` if the paper is unregistered.
    """
    if bibcode in COMBINER_REGISTRY:
        return COMBINER_REGISTRY[bibcode]
    return _default.combine, CombinePolicy()
