"""Simon+2020 (2020ApJ...892..137S) per-epoch → per-star — Tucana IV.

Source:        https://ui.adsabs.harvard.edu/abs/2020ApJ...892..137S
Galaxy:        tucana_4
Instruments:   M2FS
σ_sys floor:   0.0 km/s   — not explicitly published; framework default
                            (TODO QA-sweep #4 §refs)
p_threshold:   0.01       — framework default χ² variability cut
Zero-point offsets: None  — single instrument
Variability flagging in source: not propagated; we recompute via the
                                 χ² test in default.combine.
Notes: See docs/plan/per_paper_combiners.md for the review table.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
