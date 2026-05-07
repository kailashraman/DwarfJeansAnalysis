"""Li+2018 (2018ApJ...857..145L) per-epoch → per-star — Carina II/III.

Source:        https://ui.adsabs.harvard.edu/abs/2018ApJ...857..145L
arXiv:         https://arxiv.org/abs/1802.06810
Galaxies:      carina_2, carina_3
Instruments:   IMACS (Magellan/Baade) + AAOmega/2dF (AAT) + GIRAFFE+FLAMES (VLT)
σ_sys floor:   IMACS    : 1.0 km/s (adopted from Simon+2017)
               AAT      : 0.5 km/s (this paper, repeat measurements of 18
                                    bright stars S/N>8 across Jan/May runs)
               VLT      : 0.9 km/s (adopted from Li+ in prep on Horologium I,
                                    same VLT/GIRAFFE instrument setup)
                — paper §3.1: "We adopted the systematic floor of 1.0 km
                s^-1 for IMACS from Simon et al. (2017). For AAT, we
                determine the systematic floor to be 0.5 km s^-1 …
                Since only one exposure was taken with VLT, we were not
                able to derive a systematic floor with this dataset. We
                adopted a systematic floor of 0.9 km s^-1 from the VLT
                observations of Horologium I (Li et al., in prep) …
                We added these systematic uncertainties in quadrature
                with the statistical uncertainties to obtain the final
                reported velocity uncertainties δv."
                Per framework convention, CombinePolicy.sigma_sys_kms = 0
                because σ_sys is already in published e_RVel.
                NOTE: σ_sys is *instrument-dependent* in this paper. The
                framework treats σ_sys as statistical and pools all
                epochs in a single IVW, which is the right thing under
                that approximation regardless of per-instrument values.
p_threshold:   0.01 (framework default) — paper does not publish a
                formal χ² threshold. Binaries (2 in Car II) and RR Lyrae
                (2 in Car II) are identified by inspection of large
                inter-epoch velocity differences (~25 km/s for the
                binaries) — much larger than what our default χ² test
                at p<0.01 would flag.
Zero-point offsets: None to apply. §3.1 figure caption: "There are no
                obvious zero-point shifts between three instruments";
                paper §3.1: "we conclude that there is no significant
                zero-point shift between the various spectrographs, and
                that combining the three datasets will not introduce
                additional velocity uncertainties."
Variability flagging in source: paper Table footnote (c): "There are 18
                                 spectroscopic members but only the 14
                                 non-variable stars are used for
                                 kinematic analysis." Variability is
                                 hand-flagged in the published table
                                 (binary tag for binaries, RR Lyrae for
                                 the variables). Our χ² test is
                                 independent and may flag a different
                                 set if it ever fires.
Membership convention: Mm column — Mm==2 → Carina II member, Mm==3 →
                       Carina III member, 0 → non-member. Same 407-row
                       table is duplicated into both .npz files; only
                       the `p` column distinguishes (p=1 iff Mm==N).
Velocity error includes systematics: yes (per §3.1).
Notes: Multi-instrument combination. The 2 binaries + 2 RR Lyrae in
       Car II should be flagged variable; if our χ² test underflags
       them, the next QA-sweep should compare with the paper's
       hand-flag list.
       See docs/plan/per_paper_combiners.md.
"""
from . import CombinePolicy
from .default import combine as _default_combine

DEFAULT_POLICY = CombinePolicy(sigma_sys_kms=0.0, p_threshold=0.01)


def combine(per_epoch, registry_row, policy):
    return _default_combine(per_epoch, registry_row, policy)
