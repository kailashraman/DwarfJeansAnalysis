"""Chiti+2023 (2023AJ....165...55C) per-epoch → per-star — Tucana II.

Source:        https://ui.adsabs.harvard.edu/abs/2023AJ....165...55C
Galaxy:        tucana_2
Instruments:   M2FS, IMACS, MIKE, MagE (multi-instrument compilation,
                                          60 epochs / 19 unique stars)
σ_sys floor:   0.0 km/s   — not explicitly published; framework default
                            (TODO QA-sweep #4 §refs)
p_threshold:   0.01       — framework default χ² variability cut
Zero-point offsets: None  — TODO confirm: a 4-instrument compilation
                            very likely carries per-instrument shifts;
                            check the velocity-construction section.
Variability flagging in source: paper carries f_RVel='b' (binary
                                 candidate) in the auxiliary column;
                                 binarity ≠ variability for our χ² test.
                                 We recompute var_flag via default.combine.
Notes: See docs/plan/per_paper_combiners.md for the review table.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
