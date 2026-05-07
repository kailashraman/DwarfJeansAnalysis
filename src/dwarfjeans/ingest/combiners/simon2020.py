"""Simon+2020 (2020ApJ...892..137S) per-epoch → per-star — Tucana IV.

Paper:         "Birds of a Feather? Magellan/IMACS Spectroscopy of the
                Ultra-faint Satellites Grus II, Tucana IV, and Tucana V"
Source:        https://ui.adsabs.harvard.edu/abs/2020ApJ...892..137S
arXiv:         https://arxiv.org/abs/1911.08493
Galaxy:        tucana_4
Instruments:   IMACS (Magellan/Baade) — primary; AAOmega/AAT used for
                wide-field RV survey, NOT used in the kinematic analysis.
σ_sys floor:   1.0 km/s (observations from 2015 November onward)
               1.2 km/s (observations through 2015 October)
                — paper §3.1: "because of the inferior wavelength
                solutions obtained without the Kr comparison lamp the
                systematic velocity uncertainty is 1.2 km/s for
                observations obtained through 2015 October and 1.0 km/s
                beginning in 2015 November." Total per-epoch
                uncertainties are stat ⊕ sys in quadrature (§3.1: "The
                total velocity uncertainties consist of the quadrature
                sum of the statistical and systematic uncertainties.").
                Per framework convention, CombinePolicy.sigma_sys_kms=0.
p_threshold:   0.01 — paper §3.4: "For member stars that were observed
                on at least two separate observing runs, we check for
                velocity variations between measurements using a χ²
                test." Two confirmed binaries (DES J000228.19−604814.3
                with Δv=15.7±3.1 km/s; DES J000119.59−604439.2 with
                Δv=13.0±2.8 km/s) and one marginal (p=0.03). Our
                default 0.01 cut matches the paper's strong-evidence
                threshold; the marginal case (p=0.03) is kept by both
                the paper and our default — consistent.
Zero-point offsets: None. Single-instrument (IMACS) for the kinematic
                analysis. Different comparison-lamp configurations over
                time (Ne/Ar/He through Oct 2015; Ne/Ar/Kr from Nov 2015
                onward) but no inter-config offset is reported.
Variability flagging in source: paper Table 3 carries Memb=0/1; binaries
                                 are identified in §3.4 text but not
                                 separately marked in the table. Our
                                 χ² recomputation is the authoritative
                                 var_flag.
Membership convention: Members + non-members both retained in published
                       Table 3 with Memb column.
Velocity error includes systematics: yes (per §3.1).
Notes: This paper covers Grus II + Tuc IV + Tuc V; we dispatch on it
       only for tuc_4. (Tuc V's kinematics are revisited in
       Hansen+2024.) Paper concludes Tuc IV is a spectroscopically
       confirmed dwarf with σ_los = 4.3 +1.7/−1.0 km/s (§5).
       See docs/plan/per_paper_combiners.md.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
