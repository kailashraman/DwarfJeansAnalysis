"""Li+2017 (2017ApJ...838....8L) per-epoch → per-star — Eridanus II.

Source:        https://ui.adsabs.harvard.edu/abs/2017ApJ...838....8L
arXiv:         https://arxiv.org/abs/1611.05052
Galaxy:        eridanus_2
Instruments:   IMACS (Magellan/Baade) — single instrument
σ_sys floor:   1.2 km/s (October 2015 run) / 1.0 km/s (November 2015 run)
                — paper §3.1: "σ_v,sys = 1.2 km s^-1 for the October
                observations and σ_v,sys = 1.0 km s^-1 for the November
                observations. The slight difference … is mainly because
                the new Kr lamp included in November improved the
                wavelength solution at the blue end."
                Already added in quadrature into published e_RVel.
                Per the framework convention (multi_epoch.py docstring),
                CombinePolicy.sigma_sys_kms = 0.0 to avoid double-counting.
p_threshold:   0.01 (framework default). The staged eridanus_2.npz
                has 92 epochs across 54 unique stars (38 with 2 epochs
                Nov+Oct 2015, 16 with 1 epoch). Paper §3.2 reports no
                binary detections over the ~1-month baseline ("we are
                not able to detect any binary stars in Eri II based on
                the one month baseline"); our χ² test is run on the
                two-epoch stars and inherits the same conclusion at
                p<0.01 unless residual scatter exceeds the σ_sys floor.
Zero-point offsets: None — single-instrument observations. The Oct
                σ_sys=1.2 vs Nov σ_sys=1.0 difference is a per-run
                wavelength-solution improvement, NOT a velocity
                zero-point shift between runs.
Variability flagging in source: not propagated; var_flag is set by our
                                 χ² test in default.combine. Two-epoch
                                 stars get a real test; single-epoch
                                 stars are left unflagged (var_flag=False,
                                 p_value=NaN).
Velocity error includes systematics: yes (σ_sys absorbed in published
                                       e_RVel per §3.1).
Notes: Multi-epoch IVW with σ_sys folded into per-epoch errors. The
       2.0 km/s sys quoted on the *systemic velocity* (paper §4) is
       from the RV template zero-point, separate from per-star σ_sys;
       not relevant to per-star combining.
       See docs/plan/per_paper_combiners.md for the cross-paper review.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
