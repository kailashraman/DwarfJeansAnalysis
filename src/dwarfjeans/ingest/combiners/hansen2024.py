"""Hansen+2024 (2024ApJ...968...21H) per-epoch → per-star — Tucana V.

Paper:         "Chemical Diversity on Small Scales — Abundance Analysis
                of the Tucana V Ultra-Faint Dwarf Galaxy"
Source:        https://ui.adsabs.harvard.edu/abs/2024ApJ...968...21H
arXiv:         https://arxiv.org/abs/2403.13060
Galaxy:        tucana_5
Instruments:   MIKE (Magellan/Clay) — primary, multiple epochs per star
               IMACS (Magellan/Baade) — one additional epoch per star
σ_sys floor:   NOT EXPLICITLY PUBLISHED. Paper §2 reports per-epoch
                ``Vhel ± σ`` from χ² fits against an HD122563 template
                (radial-velocity standard) — see Table 1. The ±σ values
                listed (e.g. 0.6, 1.2, 1.6 km/s) appear to be template-
                fit statistical uncertainties only; we have no §-quote
                pinning whether a systematic floor was added.
                Per framework convention, CombinePolicy.sigma_sys_kms=0.
                If a Stage 1 run on Tuc V shows underestimated σ_vbar,
                this is a candidate cause — flag for QA-sweep #5.
p_threshold:   0.01 (framework default). Paper does not use a formal
                χ² threshold; binarity for Tuc V-1 is established by
                a multi-epoch orbital fit (TheJoker rejection sampling,
                §5.2.1: P=381 d, e=0.10, K=11.0 km/s). Tuc V-2 and Tuc
                V-3 show "no evidence of velocity variability" over the
                ~1 yr baseline. Our default χ² test should comfortably
                flag Tuc V-1 (Δv ~21 km/s peak-to-peak); var_flag for
                V-2 and V-3 should be False.
Zero-point offsets: None published. §2.3 notes "slit centering has not
                resulted in any meaningful offsets on the measured
                radial velocities as confirmed by the good agreement
                between individual measurements of Tuc V-2 and Tuc V-3"
                — i.e., no MIKE−IMACS offset reported.
Variability flagging in source: text-only. Tuc V-1 is the binary; this
                                 should match our recomputed var_flag.
Membership convention: member-list only (3 stars, all confirmed
                       members). Adapter maps to p=1 verbatim.
Velocity error includes systematics: AMBIGUOUS — published ±σ values
                                       look like template-fit stat-only
                                       errors. No quadrature-sum
                                       statement found in §-search.
Notes: Smallest sample of the six (3 stars, 17 epochs). Tuc V-1's
       binarity is the clearest variability detection in our seven
       per-epoch catalogs. Earlier IMACS spectroscopy of Tuc V appeared
       in Simon+2020; the Hansen+2024 dataset adds ~2 yr of MIKE
       monitoring.
       See docs/plan/per_paper_combiners.md.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
