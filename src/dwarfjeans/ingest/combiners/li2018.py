"""Li+2018 (2018ApJ...857..145L) per-epoch → per-star — Carina II/III.

Source:        https://ui.adsabs.harvard.edu/abs/2018ApJ...857..145L
Galaxies:      carina_2, carina_3
Instruments:   M2FS, IMACS
σ_sys floor:   0.0 km/s   — not explicitly published; framework default
                            (TODO QA-sweep #4 §refs)
p_threshold:   0.01       — framework default χ² variability cut
Zero-point offsets: None  — TODO confirm: paper may quote per-instrument
                            zero-points between M2FS and IMACS.
Variability flagging in source: paper discusses binarity (e.g. f_RVel='b'
                                 flag in derived tables); we recompute the
                                 multi-epoch χ² via default.combine. Binary
                                 σ-deconvolution is a separate QA-sweep #3
                                 track.
Notes: One paper covers two galaxies; the carina_2 / carina_3 split is
       preserved by combining each .npz independently — the published
       407-row table is duplicated into both keys with only the `p`
       column distinguishing membership (Mm==2 vs Mm==3 in the source).
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
