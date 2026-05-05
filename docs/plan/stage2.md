# Stage 2 — Jeans modeling per galaxy (parallel core)

The expensive step. One sampler run per galaxy, parallelized across the cluster. This is the stage where the dark matter halo parameters (`r_s`, `ρ_s`) are inferred from the per-star kinematic data via the spherical Jeans equation, conditioned on the photometric and distance nuisances.

This document is the canonical Stage 2 reference. The high-level role in the pipeline is summarized in [`pipeline_overview.md`](./pipeline_overview.md); deviations from P&S 2018 are consolidated in the Differences section there.

---

## Free parameters

`V`, `log r_s`, `log ρ_s`, symmetrized anisotropy `β̃ = β / (2 - β)`.

The symmetrized anisotropy follows P&S 2018 §3 (after Read et al. 2006; Read & Steger 2017): `β̃` uniformly favors radial (`β̃ → 1`) and tangential (`β̃ → −1`) orbits, whereas a uniform prior in linear `β` would preferentially sample tangential orbits.

---

## Nuisance parameters

`d`, `r_p`, `ε`, and proper motions `(μ_α cosδ, μ_δ)` where available. These use **split-normal priors built from the LVDB's raw asymmetric error bars**, applied in native units (mag for μ, arcmin for `rhalf`, mas/yr for proper motions). See [`uncertainty_conventions.md`](./uncertainty_conventions.md) for the full convention. Priors are truncated at physical boundaries (`> 0` for positive-definite quantities, `[0, 1)` for ellipticity).

Proper motions are treated as independent (correlation `pmra_pmdec_corr` ignored — the resulting shift in log J is far below P&S's per-galaxy reported errors and matches P&S 2018's own treatment; full justification in `uncertainty_conventions.md`).

---

## Priors on free parameters

The halo parameters `(ln ρ_s, ln r_s)` use the **conditional Jeffreys prior at fixed β** for the Walker+2006 likelihood — i.e., the Fisher-information determinant in `(ln ρ_s, ln r_s)` evaluated at each likelihood call. The derivation and closed-form recipe are in [`jeffreys_jeans_derivation.md`](./jeffreys_jeans_derivation.md). Implementation: `½ ln D` is added to the log-likelihood per evaluation, with

```
D = S0 · Σ p_i w̃_i (T_i − T̄)²,    T̄ = Σ p_i w̃_i T_i / S0,    S0 = Σ p_i w̃_i,
w̃_i = A_i² / (A_i + ε_i²)²,    A_i = σ_los²(R_i),    T_i = 3 − 𝒬_i / 𝒫_i.
```

`𝒬_i` is computed via one extra Jeans-style integral with the NFW dimensionless mass function `g(x) = ln(1+x) − x/(1+x)` replaced by `h(x) = x²/(1+x)²` (`jeans.sigma_los_with_T` does both in one pass; ~2× the cost of `sigma_los`). Note that `(ρ_s, r_s)` enter the Fisher matrix only through `A_i` and `T_i`; the `8πGρ_s r_s³` prefactor and `Σ(R_i)` cancel in `𝒬/𝒫`.

The remaining priors are unchanged:
- `β̃ ∈ [-0.95, 1)` uniform (the conditional Jeffreys above is at fixed β; the joint prior is `p(ln ρ_s, ln r_s | β) · p(β̃)`).
- `V ∈ [V_lit - 10, V_lit + 10]` km/s (per-galaxy override on the half-width).
- The `(ρ_s, r_s)` Jeffreys prior is truncated to a uniform-in-log10 box: `−2 ≤ log_10(r_s / kpc) ≤ 1` (plus `r_s > r_½(median)` — see below) and `4 ≤ log_10(ρ_s / [M_⊙ kpc⁻³]) ≤ 14`, expanded to `0 ≤ log_10(ρ_s / [M_⊙ kpc⁻³]) ≤ 13` for unresolved galaxies (Stage 1 classification = "unresolved / upper limit"). The truncation bounds are tied to P&S 2018 unit conventions (kpc, M⊙ kpc⁻³); they would need to change if a future implementation switched to other units.

**Toggle.** A `use_jeffreys_prior` flag (default `True`) on `make_loglike`, `make_loglike_with_nuisances`, and `run_inference` falls back to log-flat priors on `(log_10 r_s, log_10 ρ_s)` when set to `False`. The Asimov dev-loop path (`make_loglike_asimov`) does not apply the Jeffreys term and is unaffected by the flag.

**Geometry toggle.** `make_loglike_with_nuisances` and `run_inference` also accept `fix_r_p_arcmin: bool = False`. When `True`, the 7th nuisance parameter (normally `rhalf_arcmin`) is reinterpreted as `r_p_arcmin` directly: `r_p = d · r_p_arcmin · ARCMIN_TO_RAD`, skipping the √(1−ε) factor. ε is still sampled and reported but unused for r_p geometry. The caller is responsible for centering the corresponding nuisance prior on the desired r_p value. Used by the Segue 1 test pipeline's `FIX_R_P_ARCMIN` toggle.

**Calibration status.** The 15-realization MC recovery test below was performed with the previous **log-flat** priors. `run_ufd_population.py` therefore explicitly pins `use_jeffreys_prior=False` for the MC-realization path so the documented calibration result remains reproducible. **TODO:** rerun the MC calibration with `use_jeffreys_prior=True` and update the recovery summary.

### Median-r_1/2 implementation of the `r_s > r_1/2` constraint

P&S 2018 §3 imposes `r_s > r_1/2` (originally from Bonnivard et al. 2015a). In our setup, `r_1/2` is not a free parameter — it's derived per draw from the nuisance parameters `rhalf` and `ε` via `r_1/2 = R_half,maj × √(1 − ε)`. A literal reading of the constraint would make it *stochastic*: for each `(rhalf, ε)` draw the allowed `r_s` range shifts, requiring either rejection in the prior transform or a conditional `r_s` transform. We avoid this by evaluating the constraint at a single fixed point estimate computed once per galaxy at registry-build time:

```
r_1/2(median) = median(rhalf) × √(1 − median(ε))
```

using the LVDB-tabulated central values for `rhalf` and `ε`. The Stage 2 prior on `r_s` is then `log_10(r_s / kpc) ≥ log_10(r_1/2(median) / kpc)`, a deterministic lower bound applied uniformly across all draws. This:

- Keeps the prior transform a pure closed-form unit-cube map (no rejection, no conditioning).
- Avoids the dynesty-vs-MultiNest implementation-specific choice of how to enforce stochastic constraints.
- Is a slightly weaker constraint than P&S's literal version on draws where the sampled `r_1/2` exceeds the median (the constraint would forbid some `r_s` values our prior allows). The effect on log J is dominated by the `rhalf`/`ε` *prior* width itself, which the sampler still varies via the nuisance parameters; this is captured in the J-factor posterior.
- Recorded in the registry as `r_half_2d_median_pc` for provenance.

**Segue 1 test status:** The `log10_rs_min` floor described above is the design for the population pipeline. The Segue 1 test now runs the *nuisance-marginalized* form (see "Segue 1 test (interim) vs full population pipeline" below), and enforces `r_s > r_p` via per-draw rejection inside the likelihood — not via a deterministic `log10_rs_min`. The 4D path with `log10_rs_min` is still in `make_prior_transform` / `run_inference` and is used by the mock-galaxy tests in `mc_test/`.

### Segue 1 test (interim) vs full population pipeline

The Segue 1 implementation in `Segue1_test/run_segue1.py` is a 7D nuisance-marginalized run, departing from the population-pipeline spec above in two ways:

1. **Symmetric Gaussian priors instead of split-normal.** Per-galaxy LVDB asymmetric error bars are not yet plumbed in. The Segue 1 priors are hardcoded to the P&S 2018 / Martin+2008 central values:
   - `d ~ N(23.0, 2.0)` kpc,
   - `ε ~ N(0.47, 0.11)` truncated to `[0, 1)`,
   - `rhalf_arcmin ~ N(4.31, 1.03)` (chosen so that, at the fiducial `(d=23, ε=0.47)`, the implied `r_p = 21 ± 5 pc` matches P&S 2018).

   The Plummer scale is *derived* per draw: `r_p = d · rhalf_arcmin · ARCMIN_TO_RAD · √(1−ε)`. Implementation: `make_prior_transform_with_nuisances` and `make_loglike_with_nuisances` in `docs/plan/jeans_inference.py`. Splicing in split-normal inverse-CDFs from `uncertainty_conventions.md` is a one-function-replacement when the population pipeline is staged.

2. **Per-draw `r_s > r_p` rejection inside the likelihood.** Because the sampled `r_p` varies per draw, the constraint is enforced inside `make_loglike_with_nuisances` by returning `−1e300` when `r_s ≤ r_p`. This is the literal, stochastic version of the P&S 2018 constraint (vs the LVDB-median floor described above). Rejection rate is low enough on Segue 1 that the 7D wall time matches the 4D run (~290 s).

The full population pipeline retains the median-r_1/2 floor and split-normal priors; the Segue 1 test is a single-galaxy demonstrator. See `docs/plan/segue1_test.md` for results.

---

## Likelihood

Unbinned sum of per-star log-Gaussian contributions, each weighted by membership probability `p_i` — i.e. P&S eq. 8 with `p_i` moved into the log-likelihood sum, matching the Stage 1 convention:

```
ln L({V_i, σ_ε,i, R_i, p_i} | V, log r_s, log ρ_s, β̃, d, r_p, ε, ...)
  = ∑_i p_i · {  −½ ln[2π (σ_los²(R_i) + σ_ε,i²)]
                 − (V_i − V)² / [2(σ_los²(R_i) + σ_ε,i²))]  }
```

`σ_los(R)` is computed from projecting the spherical Jeans solution: spherical Jeans equation (P&S eq. 3) → radial dispersion `σ_r(r)` → projection along the line of sight (P&S eq. 4) → `σ_los(R)` evaluated at each star's projected radius `R_i`.

The Jeans projection assumes:

- **Tracer profile:** Plummer (P&S 2018 eq. 5), the full 3D density `ν_Plummer(r)`, used regardless of the LVDB `spatial_model` flag for the galaxy. Only the derivation of `r_p` from the LVDB structural parameters depends on `spatial_model` — see `data_sources.md`. This is consistent with P&S §3, which assumes Plummer throughout.
- **Halo profile:** NFW (P&S 2018 eq. 7), parameterized by `(r_s, ρ_s)`.
- **Anisotropy:** constant β with radius (P&S §3, after Read et al. 2006), parameterized by `β̃` as above.

The `p_i` weighting is applied to whatever `p_i` values survive the Stage 0b membership cut; in the P&S-replication regime where survivors all have `p_i = 1`, the weighted form reduces to P&S's exactly. See the Differences from P&S 2018 section in `pipeline_overview.md`.

---

## Jeans projection — numerical implementation

Two integrals must be evaluated per likelihood call:

**Inner (Jeans equation, constant β closed-form solution):**
```
ν(r) σ_r²(r) = ∫_r^∞ (s/r)^(2β) · ν(s) · G M(s) / s² · ds
```

**Outer (line-of-sight projection, P&S eq. 4):**
```
Σ(R) σ_los²(R) = 2 ∫_R^∞ (1 − β R²/r²) · r · ν(r) σ_r²(r) / √(r² − R²) · dr
```

For the constant-β + Plummer + NFW combination, neither integral has a closed form, but both have smooth integrands amenable to tabulated trapezoidal quadrature. NFW enclosed mass `M(r) = 4π ρ_s r_s³ · g(r/r_s)` with `g(x) = ln(1+x) − x/(1+x)` is analytic but loses precision below `x ≈ 10⁻³` due to catastrophic cancellation; we use a Taylor series through `x⁶` below threshold (relative error < 1e-12). Plummer `ν(r)` and `Σ(R)` are analytic everywhere.

### Inner integral — log-space cumulative trapezoid

The natural form `r^(-2β) · ∫_r^∞ s^(2β) F(s) ds` (where `F(s) = ν(s) · G M(s) / s²`) is numerically unstable at extreme β. At the tangential prior edge (β̃ = -0.95 → β = -38), the prefactor `r^76` overflows and the integrand `s^(-76) F(s)` underflows for any `s` more than a factor of a few above `r_min`, while their product is a perfectly finite physical answer. With `s_min ~ 10⁻⁴ · min(r_s, r_p) ~ 3 × 10⁻⁵` kpc, naive evaluation of `s^(-76)` exceeds float64's `~1.8 × 10³⁰⁸` ceiling on hundreds of grid points.

We sidestep this by computing the integral entirely in log-space and combining with the prefactor by subtraction in log before the final exponentiation. Concretely: define `log g(s) = 2β · log s + log F(s) + log s` (where the trailing `log s` comes from converting `f(s) ds → f(s) s d(log s)` for trapezoidal integration on the log grid). Each `log g(s_j)` is finite — `F(s)` is positive and smooth on `s > 0`, and `log s` is bounded on the grid. The trapezoidal segment contributions in log are `log_seg[j] = logaddexp(log g[j], log g[j+1]) - log 2 + log dlog[j]`. A reverse cumulative log-sum-exp via `np.logaddexp.accumulate` on the reversed array gives `log I[i] = log ∫_{s_i}^{s_inf} s^(2β) F(s) ds` for all grid points in O(n_grid). Interpolating in `(log s, log I)` at `log r` and returning `exp(log I_at_r - 2β · log r)` recovers the cancellation between the huge prefactor and the tiny integral with full float64 precision.

This keeps the single-shared-log-grid + cumulative-from-the-right structure (so one inner solve serves all stars), at ~4× the per-call cost of the linear-space version (~0.4 ms vs ~0.1 ms at n_grid = 4096) — negligible against the ~10 ms outer-projection cost per likelihood at N_stars = 100.

The radial prior edge β̃ → 1 is the limit β → 1, **not** β → ∞ — the symmetrized parameterization compresses the unbounded radial side `β ∈ (-∞, 1]` into `β̃ ∈ (-∞, 1]`, with β = 1 (full radial) at β̃ = 1. So the radial side of the prior is mild numerically (`2β ≤ 2`); only the tangential edge requires the log-space evaluation, but doing it unconditionally costs nothing meaningful and removes a class of bugs.

### Outer integral — singularity removal and shared u-grid

The √(r² − R²) singularity at r = R is removed by substitution r² = R² + u², giving a smooth integrand on u ∈ [0, ∞). The integrand is sharply peaked at small u for small R (since the line of sight near the center samples the densest region of the tracer profile), so the u-grid must be log-spaced — uniform u-grids undersample the peak by orders of magnitude for R << r_p, producing >100% errors in σ_los.

We use a single shared log-spaced u-grid covering all stars in the galaxy: `u ∈ [u_min, u_max]` with `u_min = 10⁻⁴ · min(R_array)` and `u_max = 100 · max(r_s, r_p)`, plus an explicit u = 0 endpoint. This restores full vectorization across stars (one outer trapezoidal pass for all R values via a 2D `(n_R, n_outer)` array).

### Tabulation strategy

Per likelihood evaluation:

1. Evaluate `ν σ_r²` on a shared log-r tabulation grid (`n_inner = 4096` points) covering `r ∈ [s_min, s_inf]`.
2. For each star at `R_i`, evaluate `σ_los²(R_i)` via the outer trapezoidal integral on the shared u-grid (`n_outer = 2048` points), with `ν σ_r²(r(u))` obtained by linear interpolation in log-r on the tabulation from step 1.

Cost: O(n_inner · n_R) for the inner step + O(n_outer · n_R) for the outer step. Per-galaxy likelihood evaluation runs in single-digit milliseconds for typical N_stars (10–500), supporting >10⁵ likelihood calls per CPU per minute — adequate for `nlive = 500, dlogz = 0.1` runs to converge in ~10 minutes wall time per galaxy.

### Verification

Numerical correctness was verified against `scipy.integrate.quad` reference implementations across the parameter ranges relevant for MW dwarfs (UFD: r_p ~ 0.03 kpc, r_s ~ 0.1 kpc; classical: r_p ~ 0.3 kpc, r_s ~ 1 kpc) and across the β̃ prior range. Worst observed grid-vs-quad relative error is ~10⁻⁴ for both the inner Jeans integral and the full projection. Cross-checks with `quad` are not run as part of the iteration loop (they are 10⁴× slower than the production grid implementation); they live in `test_jeans_vs_quad.py` and are run on demand when something looks wrong. Iteration-loop tests in `test_jeans.py` instead cover: NFW small-x correctness, β̃ prior-boundary smoke tests (finite, positive σ_los, and zero RuntimeWarnings — the warnings assertion catches future regressions where the log-space scheme is silently lost), Wolf 2010 estimator order-of-magnitude consistency, ρ_s scaling exactly recovering √2, and a per-call timing budget.

### Tidal-radius cutoff

The σ_los integrals do not require a tidal-radius cutoff — both integrate to ∞ (or numerically to `s_inf = 100 · max(r_s, r_p)`, well into the Plummer r⁻⁵ tail). The tidal radius `r_t` (Springel et al. 2008 eq. 12; Eadie & Harris 2016 MW host profile) only enters the J- and D-factor integrals, computed in Stage 3 from the Stage 2 posterior chains. Stage 2 itself is `r_t`-independent.

### MC recovery test

End-to-end statistical validation of the inference (separate from the numerical-correctness checks above). 15 mock UFD realizations at fixed truth (`r_s = 0.3 kpc`, `ρ_s = 3 × 10⁸ M⊙ kpc⁻³`, `r_p = 0.05 kpc`, `β = 0`, `N_stars = 30`, `σ_ε = 2 km/s`), varied seed, run through the full likelihood + dynesty pipeline. Recovery is unbiased and within posterior widths on every realization (max `|z| = 2.3` across all 7 halo / `M(r_½)` quantities + 8 J/D-factor reporting angles × 15 realizations).

**Prior used for this calibration:** log-flat on `(log_10 r_s, log_10 ρ_s)` (the previous default). The conditional-Jeffreys prior introduced for the Segue 1 analysis has not yet been validated through this MC; `run_ufd_population.py` pins `use_jeffreys_prior=False` so the recovery numbers below remain reproducible. Re-running the MC under `use_jeffreys_prior=True` is on the TODO list — qualitatively, in the resolved-dispersion regime that this MC sits in (`A_i ≫ ε_i²`), the Jeffreys term becomes ρ_s-independent, so the recovery should be similar; this needs to be verified rather than assumed.

The well-constrained derived quantity is **`M(r_½, 3D)`**, with mean 1σ width ≈ 0.15 dex, std(z) ≈ 1.1, KS p ≈ 0.7 against N(0,1), and median bias < 0.01 dex. The individual halo parameters `log r_s`, `log ρ_s`, and the combination `log(ρ_s · r_s³)` have 1σ widths of ~0.9, ~1.2, and ~1.5 dex respectively — the data don't constrain them individually at UFD-scale `N_stars`, and the prior box drives 100% credible-interval coverage. For UFDs `r_½ / r_s ~ 0.2` puts us in the small-x NFW limit `M(r) ∝ ρ_s · r_s · r²`, so `ρ_s · r_s³` is *not* the well-constrained axis at this scale. The Stage 3 / Stage 4 reporting strategy follows from this: J/D-factors are the headline output (also tightly constrained because they integrate the same well-constrained mass profile), but `M(r_½)` is reported as a sanity check against the Wolf+2010 estimator.

The Stage 3 J/D-factor integrals are validated on the same 15 chains; the J/D-side recovery numbers, the J/D MC speedup-knobs calibration, and the Asimov verification of the J-bias source live in [`stage3.md`](./stage3.md) § MC recovery test and § Asimov verification of the J-bias source. The two pieces share the same chains and scripts; the split between the two docs is by what's being validated (halo + M(r_½) here, J/D there).

#### Asimov dev-loop check

A single Asimov realization is used as the fast smoke test during development. Construction:

- **R_i** — deterministic equal-probability-stratified midpoints of the truncated Plummer surface density on `R ∈ [0, 5 r_p]`: `u_i = (i − 0.5)/N`, `R_i = r_p √(u_i F_max / (1 − u_i F_max))` with `F_max = (5 r_p)² / ((5 r_p)² + r_p²)`. Reproduces the truncated-Plummer moments exactly to O(1/N²) for smooth integrands.
- **σ_los,truth(R_i)** — Jeans-projected at the true `(r_s, ρ_s, β, r_p)`.
- **Asimov likelihood** — replaces each `(V_i − V_sys)²` term in the Gaussian log-likelihood with its expectation under truth, `σ_tot,truth²(R_i) = σ_los,truth²(R_i) + σ_ε,i²`, so no synthetic V-dataset is generated. The expected log-likelihood at truth is therefore the value of the Asimov log-likelihood at truth, and maximising over θ places the MLE at truth (verified numerically: `scipy.optimize.minimize` recovers truth from every starting point to machine precision).

A consequence of the Asimov-likelihood construction is that V_sys does not appear and is **unconstrained by the data**; the V posterior equals the V prior. The Asimov summary flags V as `prior_only` and does not report a z-score on it. All other halo parameters and derived quantities (`M(r_½, 2D/3D)`, J/D at the four reporting angles) are constrained as in the MC case.

What the Asimov test validates: that the likelihood, Jeans projection, dynesty wrapper, and J/D integrators run end-to-end and place the posterior peak at truth in the absence of statistical noise. What it cannot validate (by construction, n=1): the calibration metrics std(z), cov68%, KS-vs-N(0,1), or the population-level J-median offset — those are the 15-realization MC's job. Asimov medians can drift from truth by a fraction of σ even when the MLE is at truth; this is posterior asymmetry under a curved transform, not bias, and the offset should be smaller than the realization-to-realization spread measured by the MC. [`stage3.md`](./stage3.md) § Asimov verification of the J-bias source confirms this on the J posterior directly.

Asimov halo / `M(r_½)` results (run with `python run_ufd_population.py --asimov`, ~25 s wall): MLE at truth, `M(r_½, 3D)` median 0.04 dex above truth (1σ width 0.14 dex, matches MC ⟨σ⟩ = 0.15 dex). The corresponding Asimov J/D-factor results (offsets 0.15–0.23 dex on J and ≤ 0.11 dex on D, with their full chain-MAP-vs-median decomposition) live in [`stage3.md`](./stage3.md). Result artifacts: `mc_results/compact_ufd_asimov.npz`, `mc_results/ufd_asimov_table.json`, `mc_results/ufd_asimov_jd.json`.

What the MC test does *not* validate: nuisance marginalization (`d`, `r_p`, `ε`, proper motions held fixed at truth), the `r_s > r_1/2(median)` constraint (omitted to avoid prejudging the recovery), and the production sampler config (test used `sample='unif'`, `nlive=300`, `dlogz=0.5` for run-time tractability vs. production's `'rwalk'`, `nlive=500`, `dlogz=0.1`). A second-pass MC with full nuisances and production sampler settings is a Stage 2 prerequisite once nuisance ingestion is wired up; until then the MC result establishes the halo-likelihood + projection + sampler chain works on clean data, not that the production pipeline as a whole is calibrated. Scripts and result artifacts: see `mc_test/README_mc_test.md`.

---

## Halo profile interface

NFW only for the initial implementation. **Architectural requirement:** the halo profile lives behind a single `HaloProfile` interface used by both Stage 2 (for `enclosed_mass(r)` in the Jeans projection) and Stage 3 (for `density(r)` in the J/D integrands). The interface also carries:

- the profile's free-parameter list,
- their Jeffreys-prior ranges,
- the expanded-range rule for unresolved-σ_los galaxies (currently NFW-specific: `log ρ_s` lower bound drops from 4 to 0),
- and any profile-specific constraints (currently NFW-specific: `r_s > r_1/2(median)`).

Stage 2 and Stage 3 callers never reference NFW-specific parameter names directly. This seam keeps the cost of adding alternative profiles (Burkert, Einasto, generalized α-β-γ) bounded to "implement a new `HaloProfile` subclass and register it in the config" — the Jeans projection, J/D integrators, prior-transform code, perspective correction, and tidal-radius formula all stay unchanged. Profile-specific cross-checks (e.g., the analytic J-factor formula in P&S Appendix A) are gated on the active profile and disabled gracefully when not applicable.

---

## Perspective-motion correction

For classical satellites with measured proper motions, include the Kaplinghat–Strigari perspective-motion correction (Walker et al. 2008, Appendix; Kaplinghat & Strigari 2008). The correction adjusts each per-star line-of-sight velocity by the projection of the galaxy's bulk proper motion along the star's line of sight, which differs slightly from the line of sight to the galaxy center for stars far from the projected center.

Applied to: galaxies with measured proper motions in the LVDB (`pmra`, `pmdec` both present with finite errors). Not applied to ultra-faints without measured proper motions (matches P&S 2018 §3).

---

## Sampler

`dynesty`. Configuration:

- `nlive=500` (live points)
- `dlogz=0.1` (evidence-tolerance stopping criterion)
- `bound='multi'`, `sample='rwalk'` (default ensemble bound + random-walk proposals; well-suited to the curved degeneracies typical of Jeans posteriors)
- Per-galaxy parallelism via `multiprocessing.Pool` inside each SLURM job (one job per galaxy, `--cpus-per-task=8` typical)
- Explicit RNG seed in config for reproducibility

The same `nlive` / `dlogz` settings are used in Stage 1 since both are dynesty runs of similar dimensionality and difficulty.

### Sampler-comparison note (validation only)

If at any point the dynesty results need cross-checking against P&S 2018 — for paper-fidelity reproduction or to debug a galaxy where dynesty and the published value disagree by more than the validation tolerance — the comparison can be done with a one-off MultiNest run on the affected galaxies. This is a separate scripted exercise rather than a pipeline backend, and uses MultiNest with matched settings (`n_live_points=500`, `evidence_tolerance=0.1`, `sampling_efficiency=0.3`). No swappable-sampler abstraction is maintained in the production pipeline; if needed, the comparison is set up by hand for the specific galaxies in question.

---

## Outputs

- **Full posterior chain** on `(V, log r_s, log ρ_s, β̃, d, r_p, ε, μ_α cosδ, μ_δ)` per galaxy, stored as a NumPy `npz` archive keyed by parameter name. Equal-weight samples after dynesty's `resample_equal`. Feeds Stage 3 (J/D-factor integration).
- **Derived `M(r_½)` chains**, computed per posterior draw by evaluating `nfw_M(r_½, r_s, ρ_s)` at the *per-draw* `r_p` (so the photometric nuisance variation propagates through), and stored as `log10_M_half_2d` and `log10_M_half_3d` in the same `npz`. The 2D and 3D Plummer half-light radii are `R_½,2D = r_p` (analytic) and `r_½,3D = 1.30477 r_p` (numerical root of `r³/(r² + r_p²)^(3/2) = 1/2`). The 3D quantity is the Wolf+2010 estimator's mass radius and the headline well-constrained number from Stage 2 (see MC recovery test); the 2D quantity matches the pipeline's operational `r_1/2` convention.
- Sampler diagnostics: `logz`, `logz_err`, posterior weights, number of likelihood calls, wall time.
- Per-galaxy run-dir layout (config, samples, derived J/D chains) for reruns and provenance.

Provenance metadata follows the same `_meta` JSON convention as Stage 1 (see `stage1.md` Provenance section).
