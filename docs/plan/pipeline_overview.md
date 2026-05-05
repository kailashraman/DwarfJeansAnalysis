# Jeans Analysis Pipeline — Running Design Doc

**Methodology reference:** Pace & Strigari 2018, *Scaling Relations for Dark Matter Annihilation and Decay Profiles in Dwarf Spheroidal Galaxies*, MNRAS 482, 3480 ([arXiv:1802.06811](https://arxiv.org/abs/1802.06811)).

**Goal:** A cluster-deployable pipeline that takes per-dwarf stellar spectroscopic data plus tabulated global properties from the Local Volume Database (distance, structural parameters, kinematics, photometry summaries), and produces posterior distributions on J- and D-factors at several integration angles per galaxy, following the dynamical methodology of P&S 2018.

This is a living document. Each section gets fleshed out as decisions are made.

---

## Pipeline Stages

### Stage 0 — Setup & data ingestion

Splits into two sub-stages by data source. Full details in [`data_sources.md`](./data_sources.md).

**Stage 0a — Global properties from the LVDB.**
- Source: Local Volume Database, pinned to **v1.0.5** ([Zenodo DOI 10.5281/zenodo.15476348](https://doi.org/10.5281/zenodo.15476348)).
- Download `comb_all.csv` once, commit as a versioned data artifact, verify checksum on load.
- Filter to `confirmed_dwarf == True`. (`confirmed_real` is implied by `confirmed_dwarf` for our sample; no additional filter needed.)
- Required LVDB fields per galaxy: `rhalf`, `distance_modulus`, `apparent_magnitude_v` (and their errors). These are present for every confirmed MW dwarf in v1.0.5; if any are missing for a future system, drop that galaxy. `vlos_systemic` and `vlos_sigma` are **not** required for ingest:
  - `vlos_systemic` is used only to *center* the V prior in Stages 1 and 2 (default ±10 km/s). For a galaxy without a tabulated systemic velocity, the per-galaxy `vlos_prior_halfwidth` override is widened in config and the prior center is taken from the source spectroscopic paper.
  - `vlos_sigma` is metadata only — used as a sanity cross-check against the Stage 1 fit, never as a prior. A new spectroscopic discovery without a published `vlos_sigma` is processed normally; Stage 1 produces the first σ_los measurement.
- `ellipticity` handling: if the LVDB has a measured value with errors, use it as a split-normal nuisance prior. If the LVDB reports an upper limit (e.g., `< 0.10` for some round dwarfs) or no value at all, we treat ellipticity as missing and substitute a uniform prior `ε ∼ U(0, 0.5)` for that galaxy. We log which galaxies fall into this case at registry-build time.
- Ingest **raw** structural quantities (`rhalf` in arcmin, `ellipticity`, `distance_modulus`) and apply all transformations ourselves — distance, major-axis-physical, sphericalized 2D `r_1/2 = R_half,maj × √(1−ε)`, and Plummer-radius substitutions. Keep all intermediates in the registry. Cross-check our derived `r_1/2` against the LVDB's `rhalf_sph_physical` as a sanity check, but never consume it as input. See `data_sources.md` for the full rationale.
- **Units convention.** Raw LVDB columns are kept in their native LVDB units (arcmin, mag, mas/yr, etc.) with the unit recorded in the registry column name. Derived quantities use the conventional units of P&S 2018 for direct comparability: `r_s` in kpc, `ρ_s` in `M_⊙ kpc⁻³`, `r_1/2` in pc, `d` in kpc, `J` in `GeV² cm⁻⁵`, `D` in `GeV cm⁻²`, σ_los in km/s.
- Generate the internal galaxy registry from LVDB columns.

**Stage 0b — Per-star spectroscopic catalogs.**
- Sourced separately per galaxy from original spectroscopic studies (LVDB does not redistribute per-star data).
- Uniform per-star schema: `(R_i, V_i, σ_ε,i, p_i)` — projected radius (arcmin), heliocentric line-of-sight velocity (km/s), velocity measurement error (km/s; use the values reported by the source paper, no inflation or homogenization), membership probability.
- **Projected-radius convention.** `R_i` is computed from the star's (RA, Dec) relative to the LVDB-tabulated galaxy center (`ra`, `dec` columns). Used uniformly for all galaxies regardless of any alternative center the source paper may quote.
- Per-galaxy membership selection rules (P&S 2018 §2):
  - `p > 0.95` for the Walker classicals (Carina, Fornax, Sculptor, Sextans, Draco, Ursa Minor)
  - 3σ-clipping for Leo I, Leo II
  - `p > 0.8` for Segue 1
  - "Best" sample of 37 members for Boötes I
  - Various per-galaxy modifications (RR Lyrae removal, binary removal, etc.) applied at ingest with mask + reason-code preservation; see `data_sources.md`.
- Membership rules live in config, not in model code.
- **Storage format.** Per-star data are stored as a NumPy `npz` archive per galaxy, keyed by quantity name (`R`, `V`, `sigma_eps`, `p`, plus the inclusion mask and reason codes). The registry is an Astropy Table written as ECSV (units-aware) for the global properties, with per-galaxy `npz` files referenced by path.
- Detailed ingestion plan TBD — see `data_sources.md`.

### Stage 1 — Velocity-dispersion diagnostic and classification

Full details: [`stage1.md`](./stage1.md).

A model-independent kinematic characterization of each galaxy, run before any halo modeling. Uses the Walker et al. 2006 constant-σ_los Gaussian likelihood (P&S eq. 8 with `σ_los(R_i) ≡ σ_los`), with `p_i` entering as a weight on the per-star log-likelihood contribution rather than as a multiplicative product-prefactor — see Differences section. Two free parameters (`V`, `σ_los`); log-uniform prior on `log_10 σ_los` over `[-2, 2]`; uniform prior on `V` centered on `V_lit` with default ±10 km/s half-width.

Each galaxy is classified by the σ_los posterior shape into one of four categories: **resolved**, **small zero-tail** (~5% boundary mass), **large zero-tail** (~40–60%), or **unresolved / upper limit**. The classification feeds Stage 2 prior choices (esp. the expanded `ρ_s` prior range for unresolved galaxies) and Stage 4 reporting style.

**Validation gate:** the launch blocker for Stage 2 is reproducing P&S Segue 1 (the most sensitive single-galaxy test of the Stage 0b → Stage 1 chain). The full Table A1 sweep across all dwarfs runs in parallel with Stage 2 as a broader audit, not a launch blocker. Per-galaxy disagreements found in the sweep are triaged individually. Validation runs use `pi_weighting: replicate_ps2018` to match P&S's operational `p_i = 1` convention. See `stage1.md` for full validation details.

### Stage 2 — Jeans modeling per galaxy (parallel core)

Full details: [`stage2.md`](./stage2.md).

The expensive step. One sampler run per galaxy, parallelized across the cluster.

**Free parameters:** `V`, `log r_s`, `log ρ_s`, symmetrized anisotropy `β̃ = β / (2 - β)`. **Nuisances:** `d`, `r_p`, `ε`, proper motions `(μ_α cosδ, μ_δ)` where available — all with split-normal priors from raw LVDB asymmetric errors (see `uncertainty_conventions.md`). **Likelihood:** P&S eq. 8 with `σ_los(R)` computed from the spherical Jeans projection (Plummer tracer, NFW halo, constant β), and `p_i` entering the log-likelihood as a per-star weight (matching Stage 1; see Differences section). **Halo profile** lives behind a `HaloProfile` interface so non-NFW alternatives can be added without touching Stage 2/3 callers.

The `r_s > r_1/2` constraint (P&S §3, after Bonnivard et al. 2015a) is enforced at the median `r_1/2` rather than stochastically per draw — see `stage2.md` for the rationale and Differences section in this doc.

**Sampler.** dynesty as sole production backend (`nlive=500`, `dlogz=0.1`). MultiNest available for one-off validation comparisons but not maintained as a runtime backend.

### Stage 3 — J/D-factor computation from posteriors

Full details: [`stage3.md`](./stage3.md).

A cheap post-processing step. Each Stage 2 posterior draw is pushed through the small-angle line-of-sight + solid-angle integrals to produce J(θ_max) and D(θ_max) chains. A single (J, D) evaluation at all four reporting angles takes a few ms; a 4000-sample chain pushes through in a few seconds.

**Reporting angles.** `θ_max ∈ {0.1°, 0.2°, 0.5°, α_c}` for J (where `α_c = 2 r_½,3D / d` is the Wolf+2010 critical angle); `θ_max ∈ {0.1°, 0.2°, 0.5°, α_c/2}` for D.

**Tidal-radius computation.** Per draw, `r_t` is computed from Springel et al. 2008 eq. 12 in the host's enclosed-mass profile: Eadie & Harris 2016 for MW satellites (default), the LMC/SMC profile at the appropriate level for satellites of those hosts (falls back to MW if not yet implemented), and a fixed `r_t = 1 kpc` for unresolved-σ_los systems. Implementation, knob calibrations, and the MC + Asimov validation results all live in `stage3.md`.

### Stage 4 — Outputs & diagnostics

- **Compilation table** (analog of P&S Table A2): for J and D at each θ_max, **report all three of: median, q16/q84, and 1D-KDE MAP**, per galaxy. The MAP is the procedure-faithful summary (sits at truth on the Asimov to ≤ 0.13 dex at every angle); the median + q16/q84 is the field convention; the median−MAP gap is a posterior-shape indicator (large on wide-and-curved posteriors). Reporting all three lets downstream consumers pick the summary they want and makes the transform-skew visible without resolving it. We are reproducing P&S 2018 — we do not adjust priors or reparametrize to shrink the gap; we report it.
- **Per-galaxy diagnostic plots:** corner plots of `(r_s, ρ_s, β̃)` posteriors, marginal posteriors on `log M(r_½, 3D)` (the well-constrained Stage 2 derived quantity; cross-check against the Wolf+2010 estimator `M ≈ 4 σ_los² r_½ / G`), posterior distributions of `log J(θ)` and `log D(θ)` at each angle (with median, q16/q84, and 1D-KDE MAP all marked), and posterior-predictive checks against the observed `σ_los(R)` profile.
- **Validation/diagnostic runs (MC + Asimov)** compare *both* median and 1D-KDE MAP to truth, separately, for every reported quantity at every angle. The Asimov verification (`stage3.md`) is the canonical demonstration that these two summaries can disagree by O(0.2 dex) on wide-and-curved log-J posteriors and that the disagreement is posterior-shape, not procedure error.

---

## Differences from Pace & Strigari 2018

This pipeline closely follows P&S 2018's methodology, but deliberately departs from it in a small number of places. Each departure is documented inline in the relevant stage; this section consolidates them for review and provides the rationale in one place.

| # | Departure | P&S 2018 convention | Our convention | Rationale |
|---|---|---|---|---|
| 1 | **Membership-probability weighting in the likelihood** | P&S eq. 8 carries `p_i` inside a product of per-star Gaussians as a multiplicative prefactor `p_i / √[2π(...)]`. As written, this rescales the likelihood by a global constant `∏ p_i` and does not affect the inference — the gradients, posterior shape, and MAP are identical to `p_i ≡ 1`. We treat this as a publication error in P&S 2018 eq. 8: to make `p_i` actually weight the contribution of each star to the inference, it must multiply terms in the *sum* of log-likelihood contributions. Operationally P&S avoid the issue by hard-cutting at ingest and setting all surviving stars to `p_i = 1`. | Members enter the log-likelihood as `∑_i p_i · ln L_i(V, σ_los)`, with `p_i` as a continuous weight on each star's log-contribution. The membership cut is still hard at ingest (per P&S §2 thresholds); the weighting acts on surviving stars, who may carry continuous `p_i` from the source paper. | The corrected form is mathematically what "weight by membership probability" actually means. In the P&S-replication regime where surviving `p_i ≡ 1`, the two forms agree exactly, so reproduction of Table A1 is unaffected. The corrected weighting only differs from P&S when membership probabilities are propagated as continuous values (notably Segue 1 with Simon+11 Bayesian probabilities). |
| 2 | **Sampler** | MultiNest (Feroz & Hobson 2008; Feroz, Hobson & Bridges 2009). | dynesty as sole production sampler; MultiNest available for one-off validation comparisons but not maintained as a runtime backend. | Modern dynamic nested sampling, actively maintained, MIT-licensed; competitive on the curving degeneracies typical of Jeans posteriors. See "Sampler Decisions" section. |
| 3 | **`r_s > r_1/2` constraint** | Imposed as a literal pointwise constraint where `r_1/2` is whatever the current draw of `(rhalf, ε)` produces. | Imposed at the median: `r_s > r_1/2(median)`, where `r_1/2(median) = median(rhalf) × √(1 − median(ε))` is computed once per galaxy at registry-build time and used as a deterministic prior bound on `r_s` across all draws. | Keeps the `r_s` prior transform a pure closed-form unit-cube map without rejection or conditional sampling; avoids dynesty-vs-MultiNest implementation specifics. The constraint is slightly weaker on draws where the sampled `r_1/2` exceeds its median, but the J-factor effect is dominated by the sampler-varied `rhalf` and `ε` priors directly, which are unchanged. |
| 4 | **Photometric tracer-profile conversions for non-Plummer / non-exponential fits** | P&S 2018 §3 footnote 4 documents only the Plummer ↔ exponential conversion (`r_p = 1.68 × r_exp`). | We extend to Sersic and King fits by setting `r_p = r_1/2` directly (treating the half-light radius as the equivalent Plummer scale), since the LVDB has a small number of dwarfs with these `spatial_model` values. The Plummer functional form (P&S eq. 5) is still assumed in the Jeans projection regardless of the photometric `spatial_model`. | The galaxies affected are bright classicals where any adequate scale-length proxy gives consistent J-factors; the Plummer-form assumption itself is the dominant approximation, and using `r_1/2` as the Plummer scale is consistent with the Plummer case. Logged as a non-P&S convention so it's visible in the registry. |
| 5 | **Ellipticity missing-data fallback** | P&S 2018 inherits whatever value (or upper-limit treatment) was in the source structural papers. | When the LVDB provides only an upper limit or no measurement, ε is replaced by a uniform `U(0, 0.5)` prior for that galaxy. | LVDB cleanly distinguishes "measured" from "upper-limit / unknown" via the `_em` / `_ep` / `_ul` columns. The uniform-prior fallback is the principled non-informative choice for a parameter on `[0, 0.5]` (most dwarfs); galaxies known to be more elongated would be flagged for a custom prior. |
| 6 | **Proper-motion error correlation** | P&S 2018 also ignores the correlation. | We ignore `pmra_pmdec_corr` and treat the two PM components as independent split-normal priors. | Identical to P&S; flagged here for completeness because the LVDB does provide the correlation column. Effect on log J is far below reported per-galaxy errors; full justification in `uncertainty_conventions.md`. |
| 7 | **Asymmetric-error prior shape** | P&S 2018 §3 averages `_em` and `_ep` to a single Gaussian σ ("To approximate Gaussianity, some parameter errors represent the average of the upper and lower error bars"). | We use **split-normal priors** that preserve the LVDB's raw asymmetric error structure without averaging. | Symmetrization throws away information that the LVDB provides for free. Split-normal reduces exactly to a Gaussian when `_em == _ep`, so it's strictly more general. Full convention in `uncertainty_conventions.md`. |
| 8 | **Pipeline scope** | P&S 2018 covers MW, M31, and Local Field dwarfs. | We restrict to MW satellites only (including LMC/SMC subsystems, walked via the LVDB host chain). | The Eadie & Harris 2016 host-mass profile we use for the tidal radius is MW-specific; M31 satellites would need the Sofue 2015 M31 profile, and Local Field dwarfs would need the fixed `r_t = 25 kpc` convention. None of these are blockers; they're deferred to a future iteration to keep initial scope contained. |
| 9 | **Halo profile** | Same — P&S uses NFW only. | NFW only for initial implementation, but lives behind a `HaloProfile` interface so Burkert / Einasto / generalized α-β-γ can be plugged in without touching Stage 2 / 3 callers. | Architectural, not methodological: matches P&S exactly for the initial run; bounds future-rework cost. |

A summary observation: most of these departures either (a) make explicit a convention that P&S 2018 also uses operationally but doesn't state cleanly (6), (b) are clean architectural or implementation choices that don't change inference (2, 3, 9), or (c) are strictly-more-general or strictly-more-defensible conventions (5, 7) where collapsing to P&S's choice is a config-flag away. Departures (4) and (8) are scope/coverage extensions that don't apply to galaxies P&S analyzed. Departure (1) — the `p_i`-as-log-likelihood-weight correction — is the only one whose default behavior produces *different inference* than running P&S's eq. 8 verbatim, and even then only on galaxies with continuous `p_i` (notably Segue 1). The Table A1 validation is run with a `pi_weighting: replicate_ps2018` config flag to collapse this departure for the validation gate; production runs use the corrected form.

---

## Cluster Orchestration

- **Workflow runner:** Snakemake (or equivalent DAG runner). One rule per stage, with Stage 2 fanning out across galaxies as independent SLURM jobs.
- **Per-galaxy run dirs** containing config, sampler output, and derived J/D chains, so reruns and provenance are clean.
- **Embarrassing parallelism** across galaxies (~30–40 dSphs) makes Stage 2 trivially distributable.

---

## Sampler Decisions

- **Stages 1 and 2:** `dynesty` is the sole production sampler. Settings and rationale in [`stage2.md`](./stage2.md) (Sampler section).
- `emcee` ruled out for these stages due to curving degeneracies, prior-dominated posteriors, evidence-computation needs, and convergence issues on small-N data.
- **MultiNest** is available for one-off validation comparisons but is not maintained as a runtime-swappable backend (see `stage2.md` sampler-comparison note).

---

## Open Questions

- **Per-star spectroscopic ingestion:** detailed plan TBD (see `data_sources.md`).
- **Snakemake vs. alternatives** (Nextflow, custom SLURM array scripts)?

---

## Decisions Log

| Date | Decision | Notes |
|---|---|---|
| 2026-04-30 | Follow P&S 2018 dynamical methodology | Spherical Jeans, NFW, Plummer, constant β |
| 2026-04-30 | dynesty as sole production sampler | MultiNest available for one-off validation comparisons; no runtime abstraction |
| 2026-04-30 | Scope limited to per-galaxy J/D-factor posteriors | Population-level scaling-relation fit dropped from pipeline |
| 2026-04-30 | LVDB v1.0.5 for global dwarf properties | See data_sources.md; per-star ingestion TBD |
| 2026-04-30 | Split-normal priors with raw asymmetric errors | See uncertainty_conventions.md |
| 2026-04-30 | Truncate priors at physical boundaries uniformly | See uncertainty_conventions.md |
| 2026-04-30 | Independent proper-motion priors (no `pmra_pmdec_corr`) | Effect on log J negligible vs reported errors; matches P&S 2018 |
| 2026-04-30 | Galaxy scope: MW satellites only, including LMC/SMC subsystems | Walk LVDB host chain; root must be `mw`. Excludes M31 and Local Field dwarfs. |
| 2026-04-30 | Halo profile: NFW only for initial implementation | See `stage2.md` |
| 2026-04-30 | Halo profile lives behind a single `HaloProfile` interface | Stage 2 and Stage 3 are profile-agnostic; bounds future rework cost when alternatives are added |
| 2026-04-30 | Plummer scale radius `r_p` derived from LVDB structural parameters via per-profile conversion factors | No separate photometric ingestion; matches P&S 2018 §3 footnote 4. See data_sources.md |
| 2026-04-30 | Required LVDB fields: `rhalf`, `distance_modulus`, `apparent_magnitude_v`; `vlos_systemic` / `vlos_sigma` not required (override or metadata-only); ellipticity gets uniform `U(0, 0.5)` prior if missing or upper limit | All confirmed MW dwarfs in v1.0.5 have the required fields |
| 2026-04-30 | Per-star velocity errors: use as reported by source paper, no inflation | Convention captured during Stage 0b ingest |
| 2026-04-30 | Projected radius computed from LVDB `(ra, dec)` galaxy center | Uniform across all galaxies |
| 2026-04-30 | Stage 1 classification by distinct posterior peak | Resolved iff σ_los posterior has a peak ≥1 decade from boundary; otherwise unresolved/tail |
| 2026-04-30 | Per-galaxy `vlos_prior_halfwidth` override field, default 10 km/s | Allows expansion for individual galaxies per P&S 2018 footnote 6 |
| 2026-04-30 | Storage: ECSV registry + per-galaxy `npz` archives | NumPy arrays keyed by quantity name for per-star data and Stage 1 chains |
| 2026-04-30 | Conventional units for derived/reported quantities | `r_s` in kpc, `ρ_s` in M⊙/kpc³, `J` in GeV²/cm⁵, `D` in GeV/cm² |
| 2026-05-01 | Membership probability `p_i` enters as a weight on the per-star log-likelihood (not as a multiplicative product-prefactor) | Departure from P&S 2018 eq. 8 as written; reproduces P&S exactly when surviving `p_i ≡ 1`. Validation runs use `pi_weighting: replicate_ps2018` flag. See Differences section. |
| 2026-05-01 | `r_s > r_1/2` constraint enforced at median `r_1/2`, not stochastically per draw | Keeps `r_s` prior transform deterministic; logged as departure from P&S |
| 2026-05-01 | Provenance metadata stored under `_meta` key in `npz` archives (JSON-serialized dict) | Uniform convention across Stage 0b, Stage 1, Stage 2; canonical record (registry columns are derived) |
| 2026-05-01 | Stage 1 validation: Segue 1 reproduction is the Stage 2 launch blocker; full Table A1 sweep runs in parallel as a broader audit, not a launch gate | Per-galaxy disagreements in the sweep are triaged individually. See `stage1.md`. |
| 2026-05-01 | Plummer functional form (P&S eq. 5) assumed in Jeans projection regardless of LVDB `spatial_model`; only `r_p` derivation is profile-aware | Matches P&S; flagged for visibility |
| 2026-05-01 | LVDB pin remains v1.0.5; v1.0.6 / v1.1.0 noted as upstream | Doc updated to reflect v1.1.0 as current latest |
| 2026-05-01 | Stage 2 likelihood + Jeans projection + dynesty validated on 15 UFD MC realizations at fixed truth | `M(r_½, 3D)` recovered unbiased to <0.01 dex with std(z) ≈ 1.1; individual halo parameters (`r_s`, `ρ_s`, `ρ_s · r_s³`) prior-limited at UFD-scale `N_stars`. Test omits nuisance marginalization, the `r_s > r_½` constraint, and uses non-production sampler settings; a full re-run is a Stage 2 prerequisite. See `stage2.md` MC recovery test. |
| 2026-05-01 | `M(r_½, 2D)` and `M(r_½, 3D)` chains added to Stage 2 outputs | Derived per draw from the `(r_s, ρ_s, r_p)` chain. The 3D quantity is the headline well-constrained number from Stage 2; the 2D quantity matches the pipeline's operational `r_1/2` convention. See `stage2.md` Outputs. |
| 2026-05-01 | Stage 3 J/D-factor integration validated on the same 15 UFD MC chains at fixed `(d, r_t)` | D-factors recovered cleanly at all four reporting angles (median bias ≤ 0.03 dex, std(z) ≤ 1.15). J-factors unbiased to ~0.1 dex with mildly under-dispersed posteriors (std(z) up to 1.22, +0.07 to +0.10 dex high-side bias from chain-median offset under the curved log J(r_s, ρ_s) transform; J/D bias ratio ≈ 2× across angles, consistent with the density-squared kernel in J). Realization z-scores at the Wolf+10 angle track `z(M(r_½, 3D))` tightly. Test holds `d` and `r_t` fixed at truth (no nuisance marginalization, no Springel+08 r_t); production J/D run will marginalize both. See `stage3.md` MC recovery test, and `stage3.md` Asimov verification of the J-bias source for the chain-MAP-vs-median attribution. |
| 2026-05-01 | Asimov dev-loop check added as fast smoke test | Single deterministic realization with stratified-CDF Plummer R_i and an Asimov log-likelihood that substitutes `(V_i − V_sys)² → σ_tot,truth²(R_i)` per star. MLE at truth (numerically verified); V_sys is unconstrained by construction and flagged `prior_only`. Halo + J/D end-to-end in ~30 s. Used during development; does not replace the 15-realization MC calibration gate. See `stage2.md` MC recovery test → Asimov dev-loop check. |
| 2026-05-01 | J/D push knobs `(thin_to, n_R, n_u) = (200, 48, 96)` adopted as defaults for the MC sweep | Calibrated against `(500, 64, 128)`; population diagnostics agree to within 15-realization MC noise; 8× speedup (~30 s vs ~4 min for the full sweep). Knobs CLI-exposed in `run_jd_summary.py` for stricter-resolution validation runs. |
| 2026-05-01 | Asimov J-bias source verified on `compact_ufd_asimov.npz`; "Jensen on wide log ρ_s" attribution corrected | Three findings: (1) `M(r_½, 3D)` recovery clean (median +0.038 dex on σ = 0.142 dex, near-symmetric, near-Gaussian, σ matches MC). (2) The 1D-KDE MAP of `log₁₀ J(α_c)` sits at +0.024 dex from truth while the median sits at +0.162 dex — the bias lives entirely in the median of a positively-skewed transformation distribution; the procedure puts the J-posterior peak at truth. Same pattern at all four angles. (3) The handoff's "small-x analytic, J ∝ ρ_s² r_s³" attribution is wrong: chain median of `2 log ρ_s + 3 log r_s` is +0.42 dex (vs actual +0.13–0.23 dex on J), and per-sample regression of Δlog J on the small-x prediction has slopes 0.29–0.53 (not 1) and r² of 0.08–0.52. Marginal-projection decomposition at α_c shows the bias is a small (+0.16 dex) residual between two large opposite-sign 1D offsets (+0.46 dex from `log r_s`, −0.32 dex from `log ρ_s`). The wide `log r_s` posterior crosses the small-x → large-x regime so `log J(r_s, ρ_s)` is curved, not log-linear, over the chain ridge. J/D bias ratio of 2.0–2.6× is consistent with this projection picture. See `stage3.md` Asimov verification of the J-bias source; reproduce with `analyze_asimov.py`. |
| 2026-05-01 | Stage 4 reporting strategy for J/D under curved-transform skew | The chain-median offset on `log J` under the wide `(log r_s, log ρ_s)` posterior is a posterior-shape property, not an estimator bias — the Bayesian posterior is what it is. Reporting strategy: report **all three** of median, q16/q84, and 1D-KDE MAP for `log J(θ)` and `log D(θ)` at every angle, per galaxy; the median−MAP gap is implicit in the report and visible to downstream consumers. Validation/diagnostic runs (MC + Asimov) compare *both* median and MAP to truth, separately. We are reproducing P&S 2018 — we make no attempt in the main pipeline to remove or shrink the median−MAP gap (no informative concentration-mass prior, no `(M_½, shape)` reparametrization, etc.). See `stage3.md` Asimov verification of the J-bias source. |
| 2026-05-04 | Conditional Jeffreys prior on `(ln ρ_s, ln r_s)` at fixed β as the default for Stage 2 | Replaces the previous log-flat-on-`(log r_s, log ρ_s)` truncated to a box. Fisher determinant of the Walker+2006 likelihood; `½ ln D` added to the log-likelihood per call. Derived in `jeffreys_jeans_derivation.md`. On Segue 1 the prior shifts ρ_s up by ~1.4 dex (low-ρ_s tail correctly suppressed) and Jeans-derived σ_los at R_½ now matches the prior-independent Walker constant-σ result (3.86 vs 3.96 km/s; log-flat fit was 2.84). Toggled by `use_jeffreys_prior` (default `True`); the existing 15-realization MC calibration is pinned to log-flat in `run_ufd_population.py` until re-validated under the new default. See `stage2.md` § Priors and `segue1_test.md` § Latest results. |
