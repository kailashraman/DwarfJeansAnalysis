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

Simon & Geha 2011 per-star kinematic catalog (`Segue1_test/data/segue1_kinematics_simon2011.csv`).
The bundled VizieR table has 522 rows / 393 unique stars; 97 of those stars have 2–3
epochs across 15 distinct MJDs, with `Bpr` filled on exactly one row per star and `NaN`
on the rest. `load_stars_simon` groups by SDSS id, propagates the single non-NaN `Bpr`,
and inverse-variance-combines V/eV across epochs:

    V_comb  = Σ_t V_t / e_V_t² / Σ_t 1/e_V_t²
    eV_comb = 1 / sqrt(Σ_t 1/e_V_t²)

The Bayesian membership cut Bpr > 0.8 then selects **62 unique stars (27 of which are
multi-epoch)**.

This combiner is intentionally bare: it does NOT inflate eV by sqrt(chi²/dof) when intra-star
scatter exceeds the formal errors. V matches Pace's `Bayes_0d8_binary.dat` to 0.000 km/s on
all 62 stars, but eV is underestimated relative to Pace on roughly half of the multi-epoch
stars (Pace runs a binary-aware estimator). This biases σ_los/β slightly relative to a Pace
run; the discrepancy is documented but not corrected here.

Pace's 0.8-membership combined-velocity file (`Segue1_test/data/Pace_Segue1_Bayes_0d8_binary.dat`)
is also bundled and selects the same 62 stars (verified by `compare_pace_vs_bpr08.py`). The
`SOURCE` constant in `run_segue1.py` toggles between the two inputs (`'simon'` vs `'pace'`);
Pace-source outputs are written with a `_pace` suffix.

Stars at exactly R = 0 are floored to R = 1e-5 kpc (comment in `load_stars`)
to avoid a division by zero in the Jeans projection u-grid.

---

## Inference setup

**Jeans inference (Stage 2):**
- Sampler: dynesty, nlive=500, dlogz=0.1, rwalk, multi
- Free parameters: V_sys, log10(r_s), log10(ρ_s), β̃
- Priors:
  - uniform V_sys ∈ [208.5 ± 10] km/s; uniform β̃ ∈ [−0.95, 1)
  - **Conditional Jeffreys prior on (ln ρ_s, ln r_s) at fixed β** (Fisher
    determinant for the Walker+2006 likelihood — see
    `jeffreys_jeans_derivation.md`), truncated by the box
    `log10 r_s ∈ [−2, 1]`, `log10 ρ_s ∈ [4, 14]`. Implementation: per-call,
    add `½ ln D` to the log-likelihood with
    `D = S0 · Σ p_i w̃_i (T_i − T̄)²`,
    `w̃_i = A_i² / (A_i + ε_i²)²`,
    `T_i = 3 − 𝒬_i / 𝒫_i` (one extra Jeans-style integral with `g(x) → x²/(1+x)²`).
  - r_s > r_p enforced inside the likelihood per draw (no deterministic floor).
- Halo: NFW; tracer: Plummer with r_p = 0.021 kpc fixed
- Membership weights p_i = Bpr (0.8–1.0 after hard cut)
- Runtime: ~290 s on a single CPU at default sampler settings

**Toggles in `run_segue1.py`** (top-level constants):

- `SOURCE` ∈ {`'simon'`, `'pace'`} — which catalog feeds the inference. Suffix `_pace` when Pace.
- `USE_P_WEIGHTS` (default `True`) — when `False`, post-cut p_i are collapsed to 1.0 in the likelihood (both Walker constant-σ and Jeans). Suffix `_nop`.
- `FIX_R_P_ARCMIN` ∈ `(mean, sigma) | None` (default `None`) — when set, the 7th nuisance parameter is reinterpreted as `r_p_arcmin` directly with a Normal(mean, sigma) prior, bypassing rhalf·√(1−ε) inside the likelihood. ε is still sampled and reported but unused for r_p geometry. Threaded into the library via the `fix_r_p_arcmin: bool` kwarg on `make_loglike_with_nuisances` / `run_inference`. Suffix `_fixrp`.
- `DYNESTY_NLIVE`, `DYNESTY_DLOGZ` (defaults 500, 0.1) — exposed at the top of the file so longer/tighter runs can be configured without touching the inference call.

Output filenames combine the active suffixes in order: e.g. `summary_pace_nop_fixrp.csv`.

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
| V_sys (Jeans) | 209.2 | [208.3, 210.1] | km/s |
| **σ_los at R_½ (Jeans)** | **3.86** | **[2.89, 4.92]** | km/s |
| log10 r_s | −1.22 | [−1.58, −0.58] | log10(kpc) |
| log10 ρ_s | +9.32 | [+8.36, +9.98] | log10(M☉ kpc⁻³) |
| β | 0.48 | [−0.08, +0.79] | — |
| **d_kpc** (nuisance) | 23.0 | [21.0, 25.1] | kpc |
| **ε** (nuisance) | 0.47 | [0.37, 0.58] | — |
| **rhalf_arcmin** (nuisance) | 4.27 | [3.22, 5.29] | arcmin |
| **r_p (derived)** | 0.0203 | [0.0153, 0.0261] | kpc |
| **r_half_3d (derived)** | 0.0265 | [0.0199, 0.0340] | kpc |
| **α_c (derived)** | 0.133 | [0.101, 0.167] | deg |
| log10 M_half (2D) | 5.33 | [5.05, 5.56] | log10(M☉) |
| log10 M_half (3D) | 5.52 | [5.25, 5.74] | log10(M☉) |
| log10 J(0.5°) | 19.63 | [19.15, 20.07] | log10(GeV²/cm⁵) |
| log10 J(α_c≈0.133°) | 19.49 | [19.03, 19.98] | log10(GeV²/cm⁵) |
| log10 D(α_c/2) | 17.18 | [16.93, 17.40] | log10(GeV/cm²) |

logZ = −209.61 ± 0.13; n_eq = 5227 equal-weight samples. Wall time 479 s.

**Nuisance posteriors all sit on top of their priors** (data don't add information about d, ε, rhalf — expected, since they enter only through the dynamical scale `r_p` and the per-star R conversion). The implied marginal `r_p = 20⁺⁶₋₅ pc` matches P&S 2018's 21 ± 5 pc target.

**Effect of switching from log-flat to the conditional Jeffreys prior on (ln ρ_s, ln r_s).** Held against the log-flat baseline (`Segue1_test/baseline_logflat/`):
- ρ_s posterior shifts up by ~1.4 dex and tightens — the low-ρ_s tail is correctly suppressed (the doc's main qualitative prediction: in the unresolved-σ regime, w̃_i ∝ ρ_s², so the prior decays as ρ_s² rather than being log-flat there).
- σ_los at R_½ shifts from 2.84 → 3.86 km/s, **now matching the prior-independent Walker constant-σ result (3.96 km/s) to within rounding**. The log-flat Jeans posterior was biased low against the data-driven Walker estimate; the Jeffreys prior corrects this.
- M_half (2D and 3D) shifts up by ~0.3 dex and tightens by ~3×.
- J/D at the small angles shift up by ~0.5 dex.

**P&S 2018 Table A1 reference (Segue 1):** σ_los = 3.10 +0.90/−0.80 km/s. Our Walker constant-σ result agrees within ~1σ on the median (errors ~30% wider). The Walker Bayesian credible interval (proper Jeffreys on σ) and the prior-independent profile-LL interval coincide to two decimals; the Jeans+Jeffreys σ_los at R_½ now agrees with both — strong indicator the result is robust.

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
