# Segue 1 Test Run

A single-galaxy end-to-end test of the Jeans inference pipeline on Segue 1, the
ultra-faint dwarf. Purpose: validate the pipeline on real data and compare against
Pace & Strigari 2018 (P&S 2018). Script: `Segue1_test/run_segue1.py`, importing
from `docs/plan/` (the live working copy of the pipeline modules).

---

## Observational parameters

Fixed to Pace & Strigari 2018 central values (not LVDB-derived). Hardcoded in
`run_segue1.py` as module-level constants:

| Parameter | P&S 2018 value | Uncertainty | LVDB v1.0.5 value | Source |
|---|---|---|---|---|
| Distance | 23.0 kpc | ± 2.0 kpc | 22.9 kpc (μ=16.8, Belokurov 2007) | P&S 2018 |
| r_half (azimuthally averaged) | 21 pc | ± 5 pc | 19.75 pc (rhalf=3.62′, ε=0.33) | P&S 2018 |

The LVDB values are loaded and logged at runtime for provenance only. Priors on `(d, ε, rhalf_arcmin)` are hardcoded P&S 2018 / Martin+2008 values (see "Inference setup" below); LVDB error bars are not used. Derived quantities at the fiducial `(d=23, ε=0.47)` (also the prior centers):
- `r_p = 0.021 kpc` (Plummer scale = azimuthally averaged r_½ for Plummer)
- `r_half_3d = 1.30477 × 0.021 = 0.02740 kpc`
- `α_c = 2 r_half_3d / d = 0.1365°` (J/D critical angle)

These derived quantities are now **per-draw posterior chains** in the 7D nuisance-marginalized run — the values above are just the fiducial central values. The `r_s > r_p` constraint is enforced per-draw inside the likelihood (no deterministic `log10_rs_min`); see `docs/plan/stage2.md` for the rationale.

---

## Membership

Simon & Geha 2011 per-star kinematic catalog (`segue1_kinematics_simon2011.csv`).
Bayesian membership probability cut Bpr > 0.8 → **62 member stars**.

Stars at exactly R = 0 are floored to R = 1e-5 kpc (comment in `load_stars`)
to avoid a division by zero in the Jeans projection u-grid.

---

## Inference setup

**Jeans inference (Stage 2):**
- Sampler: dynesty, nlive=500, dlogz=0.1, rwalk, multi
- Free parameters: V_sys, log10(r_s), log10(ρ_s), β̃
- Priors: uniform V_sys ∈ [208.5 ± 10] km/s; Jeffreys log10(r_s) ∈ [−1.678, 1]
  (r_s > r_p floor applied); Jeffreys log10(ρ_s) ∈ [4, 14]; uniform β̃ ∈ [−0.95, 1]
- Halo: NFW; tracer: Plummer with r_p = 0.021 kpc fixed
- Membership weights p_i = Bpr (0.8–1.0 after hard cut)
- Runtime: ~290 s on a single CPU

**Constant-σ block (Walker+2006 likelihood):**
- 2D grid posterior (400×400) on (V_sys, σ_los); σ-grid log-spaced over log10 σ ∈ [−2, 2].
- Likelihood: ∑_i p_i · [−½ ln(2π(σ² + ε_i²)) − ½(V_i − V_sys)²/(σ² + ε_i²)] (membership-weighted; p_i = Bpr inside the likelihood, not just a hard cut).
- **Prior on σ: proper Jeffreys (Fisher determinant)** for the (V̄, σ) Walker+2006 model, truncated to log10 σ ∈ [−2, 2]:

      p_J(σ) ∝ σ · √( [Σ_i p_i / σ_i²] · [Σ_i p_i / σ_i⁴] ),     σ_i² ≡ σ² + ε_i²

  V̄-independent. This prior was chosen after experimenting with two alternatives that proved inadequate:
  - *Uniform-on-σ* (the original code): no principled justification.
  - *Log-uniform on log10 σ* (often called "Jeffreys" but really not): produces a long lower tail because the Walker likelihood is σ-independent for σ ≪ ε_i, so the broad log-prior piles posterior mass at small σ.
  The proper Jeffreys prior is `∝ σ` for σ ≪ ε_i (kills the small-σ pile-up) and `∝ σ⁻²` for σ ≫ ε_i.

- **Profile-likelihood interval (prior-independent cross-check):** Δlnℒ = ½ from the MLE on the profile-LL `prof_lnL(σ) = max_V̄ lnL(V̄, σ)`, inverted by linear interpolation. Reported in `summary.csv` as `sigma_los_walker_profileLL_mle_dlnL_half`.
- Runtime: < 1 s; independent of the Jeans inference. Output plot: `sigma_los_walker.png` (Bayes marginal + profile-LL endpoints overlaid).

---

## Latest results (7D Jeans inference, nuisance-marginalized over d / ε / rhalf_arcmin)

Sampled parameters: 4 halo (V_sys, log10 r_s, log10 ρ_s, β̃) + 3 nuisances (d, ε, rhalf_arcmin). r_p, R_½,2D, r_½,3D, α_c are derived per draw. rs > r_p enforced inside the likelihood per draw.

| Quantity | Median | [16th, 84th] | Unit |
|---|---|---|---|
| V_sys (Walker, constant-σ block) | 209.4 | [208.5, 210.3] | km/s |
| **σ_los (Walker, proper Jeffreys)** | **3.96** | **[2.80, 5.13]** | km/s |
| σ_los profile-LL Δlnℒ=½ (prior-indep) | 3.95 | [2.81, 5.13] | km/s |
| V_sys (Jeans) | 209.1 | [208.2, 210.0] | km/s |
| σ_los at R_half (Jeans) | 2.84 | [0.65, 4.12] | km/s |
| log10 r_s | −0.60 | [−1.41, +0.44] | log10(kpc) |
| β | 0.52 | [−0.39, +0.83] | — |
| **d_kpc** (nuisance) | 23.0 | [21.0, 24.9] | kpc |
| **ε** (nuisance) | 0.47 | [0.36, 0.57] | — |
| **rhalf_arcmin** (nuisance) | 4.35 | [3.40, 5.38] | arcmin |
| **r_p (derived)** | 0.0207 | [0.0156, 0.0270] | kpc |
| **r_half_3d (derived)** | 0.0270 | [0.0204, 0.0352] | kpc |
| **α_c (derived)** | 0.137 | [0.104, 0.172] | deg |
| log10 M_half (2D) | 5.02 | [3.84, 5.36] | log10(M☉) |
| log10 M_half (3D) | 5.23 | [4.05, 5.56] | log10(M☉) |
| log10 J(0.5°) | 19.21 | [17.49, 19.88] | log10(GeV²/cm⁵) |
| log10 J(α_c≈0.137°) | 18.99 | [17.30, 19.65] | log10(GeV²/cm⁵) |
| log10 D(α_c/2) | 17.03 | [16.29, 17.38] | log10(GeV/cm²) |

logZ = −209.42 ± 0.11; n_eq = 4784 equal-weight samples. Wall time 288 s.

**Nuisance posteriors all sit on top of their priors** (data don't add information about d, ε, rhalf — expected, since they enter only through the dynamical scale `r_p` and the per-star R conversion). The implied marginal `r_p = 21⁺⁶₋₅ pc` matches P&S 2018's 21 ± 5 pc target. Compared to the previous fixed-nuisance run, σ_los at R_½ and log10 M_half broaden on the *lower* tail (extra uncertainty from r_½,3D variation) — the qualitatively expected effect.

**P&S 2018 Table A1 reference (Segue 1):** σ_los = 3.10 +0.90/−0.80 km/s. Our constant-σ result agrees within ~1σ on the median; errors are ~30% wider. The Bayesian credible interval (proper Jeffreys) and the prior-independent profile-LL interval coincide to two decimals — strong indicator the result is robust.

---

## Deviations from general pipeline plan

| Plan feature | Plan spec | Segue 1 test status |
|---|---|---|
| Observational inputs | LVDB v1.0.5 | **Hardcoded priors**: distance from P&S 2018 (`d ~ N(23.0, 2.0)` kpc); ellipticity from Martin+2008 (`ε ~ N(0.47, 0.11)` truncated to `[0, 1)`); `rhalf_arcmin` reverse-engineered to give `r_p = 21 ± 5 pc` (P&S 2018) at fiducial `(d, ε)`. LVDB v1.0.5 is loaded for diagnostic logging only. |
| Nuisance marginalization | d, r_p, ε marginalized via split-normal priors | **Implemented** as 7D dynesty with symmetric-Gaussian priors: d ~ N(23.0, 2.0) kpc, ε ~ N(0.47, 0.11) trunc [0,1), rhalf ~ N(4.31, 1.03) arcmin; r_p derived per draw |
| r_s > r_p constraint | Median-r_1/2 from LVDB (stage2.md §) | **Per-draw rejection** inside the likelihood (compares against the sampled r_p) — see `make_loglike_with_nuisances` in `jeans_inference.py` |
| Membership probabilities | Full p_i treatment | Implemented — Bpr used as soft weights |
| Perspective-motion correction | Applied where proper motions available | **Not applied** |
| Constant-σ cross-check | Not in plan | **Added** — Walker+2006 likelihood with proper Jeffreys (Fisher det) prior on σ, plus a profile-likelihood Δlnℒ=½ interval as prior-independent cross-check. Both match to 2 decimals. |
| Tidal radius | Computed from Springel+2008 / Eadie+2016 | **Placeholder** r_t = 1.0 kpc (negligible effect on J/D at these angles) |

---

## Next steps toward P&S 2018 replication

1. Compare J(0.5°) and D(α_c/2) posteriors directly to P&S 2018 Table 2 numbers.
2. (Population-pipeline groundwork) Generalize the symmetric-Gaussian nuisance priors to LVDB asymmetric (split-normal) error bars; the helper-function shape is in place — just swap `norm.ppf` / `truncnorm.ppf` for split-normal inverse-CDFs.
3. (Optional) Switch the rs > rp likelihood-rejection to a prior-transform-conditional draw if the rejection rate becomes a bottleneck for harder galaxies (apparently a non-issue for Segue 1 — 7D wall time 288 s ≈ 4D wall time, so rejections aren't dominating).
3. Compare J(0.5°) posterior directly against P&S 2018 Table 2
