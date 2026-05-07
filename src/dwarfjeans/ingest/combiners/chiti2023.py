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
Zero-point offsets: documented in §3.1 but their handling in Table 1
                is **NOT BYTE-VERIFIED** (open issue #2 in
                docs/plan/per_paper_combiners.md). §3.1 reports
                pairwise inter-instrument offsets observed in the data:
                  v_MIKE − v_MagE = +1.0 km/s (excl. TucII-309)
                  v_MIKE − v_M2FS = +2.5 ± 0.7 km/s
                  v_MIKE − v_IMACS = +2.2 km/s
                These are *diagnostic measurements* of the input
                catalogs. Whether Table 1's per-star velocities are
                published on a common zero-point (offsets absorbed by
                the per-star weighted average) or carry instrument-tagged
                raw values is not byte-verified yet — the handler today
                does not apply offsets, which is correct only under the
                "common zero-point in Table 1" interpretation. Until
                verified, treat Tuc II σ_los as TODO and re-check this
                interpretation if a Stage 1 run shows ~1–2 km/s extra
                dispersion attributable to inter-instrument scatter.
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

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
