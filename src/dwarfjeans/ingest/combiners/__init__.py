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


from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


_EMPTY_OFFSETS: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class CombinePolicy:
    """Tunables shared across handlers. Dataset-specific overrides go in
    the per-dataset module.

    .. note::
       ``sigma_sys_kms`` is the systematic floor that ``default.combine``
       adds in quadrature *after* the IVW. Under the current
       "σ_sys-as-statistical" convention (see ``multi_epoch.py``
       module docstring) every per-paper handler sets this to ``0.0``
       because the source paper has already added its σ_sys into the
       published per-epoch ``e_RVel``; setting it to a non-zero value
       here would double-count. A handler that opts into the strict-
       deconvolution path (also documented in ``multi_epoch.py``) may
       set ``sigma_sys_kms`` here explicitly to the published value.
       See ``docs/plan/per_paper_combiners.md`` for the per-paper
       review table.

    .. note::
       ``zero_point_offsets_kms`` maps instrument tag (the value of the
       per-epoch ``Inst`` column) → additive shift in km/s applied to
       ``V`` *before* the IVW. The reference instrument has offset 0.
       When non-empty, ``default.combine`` requires the per_epoch dict
       to carry an ``Inst`` column and raises otherwise. Unknown
       instrument tags raise — silent zero-offset fallback is the bug
       this hook exists to prevent. Sign convention follows the
       paper's published shift: e.g. Chiti+2022 §3.4.1 reports
       v_IMACS − v_M2FS = −2.6 km/s, so to bring IMACS onto the M2FS
       zero-point pass ``{"IMACS": +2.6, "M2FS": 0.0}``.
    """

    sigma_sys_kms: float = 0.0
    p_threshold: float = 0.01
    zero_point_offsets_kms: Mapping[str, float] = field(
        default_factory=lambda: _EMPTY_OFFSETS
    )


# Per-paper handlers. Each module wraps default.combine and exposes a
# DEFAULT_POLICY whose σ_sys / p_threshold / zero-point choices are
# justified in the module docstring against a specific paper section.
# See docs/plan/per_paper_combiners.md for the cross-paper review table.
from . import li2017, li2018, chiti2022, chiti2023, simon2020, hansen2024  # noqa: E402

# Registry: source_paper_bibcode → (combine_fn, default_policy)
COMBINER_REGISTRY: dict[str, tuple[CombineFn, CombinePolicy]] = {
    "2017ApJ...838....8L": (li2017.combine,    li2017.DEFAULT_POLICY),
    "2018ApJ...857..145L": (li2018.combine,    li2018.DEFAULT_POLICY),
    "2022ApJ...939...41C": (chiti2022.combine, chiti2022.DEFAULT_POLICY),
    "2023AJ....165...55C": (chiti2023.combine, chiti2023.DEFAULT_POLICY),
    "2020ApJ...892..137S": (simon2020.combine, simon2020.DEFAULT_POLICY),
    "2024ApJ...968...21H": (hansen2024.combine, hansen2024.DEFAULT_POLICY),
}


def get_combiner(bibcode: str) -> tuple[CombineFn, CombinePolicy]:
    """Return ``(combine_fn, default_policy)`` for the given source paper.

    Falls back to ``default`` if the paper is unregistered.
    """
    if bibcode in COMBINER_REGISTRY:
        return COMBINER_REGISTRY[bibcode]
    return _default.combine, CombinePolicy()
