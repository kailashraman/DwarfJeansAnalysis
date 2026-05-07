"""Li+2017 (2017ApJ...838....8L) per-epoch → per-star — Eridanus II.

Source:        https://ui.adsabs.harvard.edu/abs/2017ApJ...838....8L
Galaxy:        eridanus_2
Instruments:   M2FS
σ_sys floor:   0.0 km/s   — not explicitly published; framework default
                            (TODO QA-sweep #4 §refs)
p_threshold:   0.01       — framework default χ² variability cut
Zero-point offsets: None  — single instrument
Variability flagging in source: not propagated; we recompute via the
                                 χ² test in default.combine.
Notes: 28 epochs of 16 unique members. See docs/plan/per_paper_combiners.md
       for the review table that tracks calibration status.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
