"""Chiti+2022 (2022ApJ...939...41C) per-epoch → per-star — Grus I.

Paper:         "Magellan/IMACS Spectroscopy of Grus I: A Low Metallicity
                Ultra-faint Dwarf Galaxy"
Source:        https://ui.adsabs.harvard.edu/abs/2022ApJ...939...41C
arXiv:         https://arxiv.org/abs/2206.04580
Galaxy:        grus_1
Instruments:   IMACS (Magellan/Baade) — primary
               M2FS  (Magellan/Clay) — secondary observations of a
                                        subset of stars
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
Zero-point offsets: TODO (REAL OFFSET, NOT YET APPLIED)
                §3.4.1: "a small, but statistically significant, average
                velocity offset of v_IMACS - v_M2FS = -2.6 ± 0.8 km/s"
                (refined to -2.8 +1.0/-0.9 km/s in the §4.1 MCMC fit).
                Whether the published Table 2 velocities have this shift
                already applied is unclear from the §refs the researcher
                pulled — likely NOT, because §4.1 sets it as a free
                parameter in the kinematic fit. **Our adapter does not
                carry an instrument-tag column** (see
                ``src/dwarfjeans/ingest/path_b_adapters/chiti2022.py`` —
                the per-epoch arrays it builds drop the source
                spectrograph), so the per-paper handler cannot apply
                the shift today.
                Action: re-stage grus_1 with an instrument column, then
                wire the offset application into this handler. Until
                then, the σ_los inferred for Grus I will absorb the
                ~2.6 km/s inter-instrument scatter as extra dispersion.
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
Notes: Bibcode-to-galaxy: also serves grus_1 only; no other galaxies in
       this paper.
       See docs/plan/per_paper_combiners.md.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
