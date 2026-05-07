"""Chiti+2022 (2022ApJ...939...41C) per-epoch → per-star — Grus I.

Source:        https://ui.adsabs.harvard.edu/abs/2022ApJ...939...41C
Galaxy:        grus_1
Instruments:   MIKE, M2FS (multi-epoch compilation)
σ_sys floor:   0.0 km/s   — not explicitly published; framework default
                            (TODO QA-sweep #4 §refs)
p_threshold:   0.01       — framework default χ² variability cut
Zero-point offsets: None  — TODO confirm: multi-instrument compilation
                            may carry per-instrument zero-points.
Variability flagging in source: paper flags individual stars; we recompute
                                 via default.combine.
Notes: See docs/plan/per_paper_combiners.md for the review table that
       tracks calibration status across all per-paper handlers.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
