"""Hansen+2024 (2024ApJ...968...21H) per-epoch → per-star — Tucana V.

Source:        https://ui.adsabs.harvard.edu/abs/2024ApJ...968...21H
Galaxy:        tucana_5
Instruments:   M2FS
σ_sys floor:   0.0 km/s   — not explicitly published; framework default
                            (TODO QA-sweep #4 §refs)
p_threshold:   0.01       — framework default χ² variability cut
Zero-point offsets: None  — single instrument
Variability flagging in source: paper discusses binarity; we recompute
                                 multi-epoch χ² via default.combine.
                                 Binary σ-deconvolution is a separate
                                 QA-sweep #3 track.
Notes: See docs/plan/per_paper_combiners.md for the review table.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
