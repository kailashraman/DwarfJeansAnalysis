"""Chiti+2022 (2022ApJ...939...41C) per-epoch → per-star — Grus I.

Paper:         "Detailed Properties of the Ultrafaint Dwarf Galaxy Grus I"
Source:        https://ui.adsabs.harvard.edu/abs/2022ApJ...939...41C
arXiv:         https://arxiv.org/abs/2206.04580
Galaxy:        grus_1
Instruments:   Magellan/IMACS — single-instrument compilation in the
                published Table 2. (Three IMACS campaigns: 2015, 2019,
                2021 → MJDs 57229.33, 58762.03, 59471.02.)
σ_sys floor:   1.1 km/s — paper §3.1: "a systematic velocity uncertainty
                of 1.1 km s^-1 on our velocity measurements, based on
                repeat observations of stars … We computed final
                velocity uncertainties by adding in quadrature the
                random velocity uncertainties and the systematic
                velocity uncertainty."
                Per framework convention, CombinePolicy.sigma_sys_kms=0.
p_threshold:   0.01 (framework default). Paper does NOT use a uniform
                χ² threshold — §3.5 reports per-star p-values and makes
                ad hoc calls: "strong evidence (p = 0.01) of binarity
                for Gru1-003 …, and marginal evidence (p = 0.04) for
                Gru1-022." Our default 0.01 cut would flag Gru1-003 as
                variable; Gru1-022 would survive — that matches the
                paper's "marginal" call (which they keep in the kinematic
                analysis).
Zero-point offsets: NONE applied within Chiti+2022.
                §3.4.1 reports "a small, but statistically significant,
                average velocity offset of v_IMACS − v_M2FS = −2.6 ±
                0.8 km/s" (refined to −2.8 +1.0/−0.9 km/s in the §4.1
                MCMC fit). This is a *cross-paper* calibration check
                against EXTERNAL M2FS velocities from Walker+2016 —
                those M2FS data are not in Table 2 and are not ingested
                by our chiti2022 path-B adapter. Adapter stamps every
                row Inst="IMACS" so the framework hook
                (CombinePolicy.zero_point_offsets_kms) is wired but
                empty here. Should a future commit merge Walker+2016
                M2FS observations into the Grus I per-epoch table,
                replace the DEFAULT_POLICY constructor below with
                ``CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01,
                zero_point_offsets_kms={"IMACS": +2.6, "M2FS": 0.0})``
                (CombinePolicy is frozen, so reassign — don't mutate).
                Sign: M2FS as reference; +2.6 added to IMACS rows
                brings them onto the M2FS zero-point.
Variability flagging in source: paper has no "variability" column. The
                                 χ² recomputation in default.combine is
                                 the authoritative variability check;
                                 the paper's per-star p-value examples
                                 (Gru1-003, Gru1-022) are the calibration
                                 reference.
Membership convention: paper Table 2 retains all observed targets with
                       MEM column ∈ {M, CM, NM}; our adapter maps M→p=1,
                       CM/NM→p=0. CM (candidate members) are dropped at
                       selection time.
Velocity error includes systematics: yes (per §3.1, quadrature sum).
Notes: Bibcode-to-galaxy: serves grus_1 only.
       See docs/plan/per_paper_combiners.md.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
