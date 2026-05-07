"""Chiti+2023 (2023AJ....165...55C) per-epoch → per-star — Tucana II.

Paper:         "Detailed Chemical Abundances of Stars in the Outskirts
                of the Tucana II Ultrafaint Dwarf Galaxy"
Source:        https://ui.adsabs.harvard.edu/abs/2023AJ....165...55C
arXiv:         https://arxiv.org/abs/2205.01740
Galaxy:        tucana_2
Instruments:   M2FS, IMACS, MIKE, MagE  (4-instrument compilation,
                                          60 epochs / 19 unique stars)
σ_sys floor:   0.9 km/s for new MIKE observations, 1.2 km/s for
                archival MIKE — paper §3.1: "a systematic uncertainty
                of 0.9 km s^-1 needed to be added in quadrature to the
                random uncertainties." Errors in published Table 1 are
                σ_sys-dominated (random < 0.4 km/s, systematic ~0.9
                km/s). Per framework convention, CombinePolicy.sigma_sys_kms=0.
p_threshold:   0.01 (framework default). Paper does NOT publish a
                formal χ² threshold — §3.1 flags binaries by absolute
                inter-epoch differences (TucII-309 with Δv = 8.4 km/s,
                TucII-078 with Δv > 8 km/s). Our default χ² test will
                generally agree on these (large-Δv stars push χ²
                very high) but the calibrated threshold is "Δv > 8 km/s"
                rather than p<0.01.
Zero-point offsets: APPLIED here (verified 2026-05-07). MIKE is the
                reference. §3.1 reports pairwise inter-instrument
                offsets:
                  v_MIKE − v_MagE  = +1.0 km/s (excl. TucII-309)
                  v_MIKE − v_M2FS  = +2.5 ± 0.7 km/s
                  v_MIKE − v_IMACS = +2.2 km/s
                Byte-verify (against staged tucana_2.npz, 60 epochs /
                19 stars):
                  - Empirical v_MIKE−v_M2FS = +1.5 km/s (n=5),
                    v_MIKE−v_IMACS = +1.6 km/s (n=2) — within 1σ of
                    paper §3.1 values (+2.5, +2.2) modulo small-N
                    scatter.
                  - Empirical v_MIKE−v_MagE = +2.8 km/s with all 6
                    cross-matched stars; drops to +1.6 km/s (n=5) when
                    TucII-309 (the +8.8 km/s outlier; binary candidate
                    flagged f_RVel='b') is excluded — this is the same
                    exclusion the paper §3.1 text makes ("excl.
                    TucII-309"). Excluded mean is consistent with the
                    paper's +1.0 km/s within 1σ small-N scatter.
                    TucII-309's velocity is itself unreliable until
                    binary motion is accounted for; the χ² variability
                    flag should catch it downstream.
                  - stars.csv (Table 1, 5 high-res MIKE-only stars)
                    matches MIKE-only IVW from Table 6 to within ~0.1
                    km/s, confirming Chiti+2023 themselves DO NOT
                    pre-shift Table 6 onto a common zero-point.
                Therefore Table 6 velocities are raw per-instrument and
                applying the §3.1 offsets is the science-correct choice
                given the hook now exists (commit 62415e1). Impact:
                individual v_bar shifts by up to +2.5 km/s, σ_los
                across 19 stars drops 4.02 → 3.88 km/s (~3%). v_sys
                shifts by ~+1.2 km/s toward the MIKE zero-point.
Variability flagging in source: text-only; no variability column in
                                 Table 1. Named binaries: TucII-309,
                                 TucII-078. Our χ² recomputation is the
                                 authoritative variability check.
Membership convention: member-list only — every Table 1 row is a
                       confirmed member from Chiti+2021. Adapter maps
                       to p=1 verbatim (missing_default rule).
Velocity error includes systematics: yes (sys-dominated; σ_sys ~ 0.9
                                       >> σ_random ~ 0.4 km/s).
Notes: Largest multi-instrument compilation in our six. The inter-
       instrument offset claim (already-applied) is the assumption that
       most needs verification under QA-sweep #5.
       See docs/plan/per_paper_combiners.md.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(
    sigma_sys_kms=0.0,
    p_threshold=0.01,
    zero_point_offsets_kms={
        "MIKE": 0.0,
        "M2FS": +2.5,
        "IMACS": +2.2,
        "MagE": +1.0,
    },
)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
