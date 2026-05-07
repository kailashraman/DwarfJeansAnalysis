"""Hansen+2024 (2024ApJ...968...21H) per-epoch → per-star — Tucana V.

Paper:         "Chemical Diversity on Small Scales — Abundance Analysis
                of the Tucana V Ultra-Faint Dwarf Galaxy"
Source:        https://ui.adsabs.harvard.edu/abs/2024ApJ...968...21H
arXiv:         https://arxiv.org/abs/2403.13060
Galaxy:        tucana_5
Instruments:   MIKE (Magellan/Clay) — primary, multiple epochs per star
               IMACS (Magellan/Baade) — one additional epoch per star
σ_sys floor:   NOT EXPLICITLY PUBLISHED, but **inferred sys-included
                from empirical magnitudes** (verified 2026-05-07).
                Per-epoch sigma_eps in the staged tucana_5.npz (17
                epochs / 3 stars):
                    MIKE  (n=14): min 0.6, max 1.7, mean 1.14 km/s
                    IMACS (n=3):  min 1.1, max 1.7, mean 1.33 km/s
                MIKE template-fit stat-only errors are typically
                ~0.2–0.4 km/s for high-S/N giants; the floor at 0.6
                km/s and the 1.0–1.2 km/s typical value are
                inconsistent with stat-only and consistent with the
                published MIKE σ_sys range (~0.5–1.0 km/s) added in
                quadrature with stat. We therefore treat Hansen+2024
                ±σ as already sys-included and keep
                CombinePolicy.sigma_sys_kms=0 (framework convention).
                Caveat: this is an empirical inference, not a paper
                statement; if a future deep-reviewer or contact with
                the authors yields a contradictory answer, revisit.
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
Velocity error includes systematics: empirically YES (see σ_sys floor
                                       row above for the magnitude
                                       argument). No paper §-quote
                                       confirms this directly.
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
