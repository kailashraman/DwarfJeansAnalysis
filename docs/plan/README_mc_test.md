# Jeans MC recovery test — scripts and notes

A Monte Carlo test of the Stage-2 Jeans inference procedure (NFW + Plummer +
constant β + dynesty). Validates that the procedure described in `stage2.md`
recovers input halo parameters from synthetic stellar velocity catalogs
before applying it to real dwarfs.

## Status

**Stage-2 halo recovery under Jeffreys (current production prior).** 15 UFD
realizations at fixed truth. Headline numbers:

| param            | mean(z) | std(z) | med bias | cov68% | KS p   |
|------------------|--------:|-------:|---------:|-------:|-------:|
| `log10_M_half_3d`|  −0.61  |  0.79  |  +0.07   |  60%   | 0.071  |
| `log10_M_half_2d`|  −0.83  |  0.82  |  +0.13   |  53%   | 0.013  |

Under the previous loguniform prior, M(r_½, 3D) was recovered cleanly
(median bias < 0.01 dex, std(z) ≈ 1.1, KS p ≈ 0.7); Jeffreys shifts the
joint posterior toward higher ρ_s / lower r_s in the small-x NFW regime
(`r_½/r_s ≈ 0.2`), producing a small ~0.07 dex high-side bias and an
under-dispersed z-distribution.

The user's heuristic ρ_s · r_s³ is *not* the well-constrained combination
on UFDs — small-x NFW gives M ∝ ρ_s · r_s · r², so log(ρ_s · r_s³) is the
*widest* (1σ ≈ 1.5 dex) of the parameters tested.

Headline numbers and "what this test does *not* validate" are documented
canonically in **`stage2.md` → MC recovery test** subsection. Decision-log
entries in **`pipeline_overview.md`**.

**Stage-3 J/D MC recovery: pending under Jeffreys.** The current
`run_ufd_population.py` runs only the halo recovery; per-realization J/D
summaries (formerly produced by the removed `run_jd_summary.py` /
`summarize_jd` helper) need to be reinstated as an inline J/D push inside
the MC loop, mirroring the `J_D_factors` usage in `scripts/run_production.py`.

Historical loguniform baseline: D-factor recovery was clean at all four
reporting angles (median bias ≤ 0.03 dex, std(z) ≤ 1.15). J-factor recovery
was unbiased to ~0.1 dex with mildly under-dispersed posteriors (std(z) up
to 1.18, +0.08 to +0.12 dex high-side median bias) — chain-median offset
under the curved `log J(r_s, ρ_s)` transform on a wide ridge, not a procedure
issue (verified on the Asimov chain — see below). Realization z-scores at
the Wolf+10 angle track `z(M(r_½, 3D))` tightly. Test choices: d = 30 kpc,
r_t = 1 kpc held fixed at truth (synthetic data have no host distance /
3D position).

J/D self-test against the truth gives log J(0.5°) ≈ 19.65 GeV²/cm⁵, log
D(0.5°) ≈ 18.52 GeV/cm² — consistent with P&S 2018 Table A2's reported
UFD range. J and D verified monotonic in θ_max.

**Asimov dev-loop check: added.** Single deterministic realization
(stratified-Plummer R_i + Asimov log-likelihood that substitutes
`(V_i − V_sys)² → σ_tot,truth²(R_i)` per star). MLE at truth verified
numerically; V_sys unconstrained by construction (flagged `prior_only`).
End-to-end halo + J/D in ~30 s. Used as a fast smoke test during development;
does not replace the 15-realization MC as the calibration gate.

**Asimov J-bias source verified under Jeffreys.** On the Asimov chain
(`compact_ufd_asimov.npz`): M(r_½, 3D) recovered with median ~0.04 dex
above truth. The 1D-KDE MAP of `log₁₀ J(α_c)` sits at +0.04 dex from truth
while the median sits at +0.162 dex — the bias lives largely in the median
of a positively-skewed transformation distribution, not in posterior
centering. The "small-x analytic" attribution (J ∝ ρ_s² r_s³) is wrong:
chain spans 3 dex in `log r_s` and crosses the small-x → large-x regime,
so `log J(r_s, ρ_s)` is curved, not log-linear, over the chain ridge.
Marginal-projection decomposition shows the bias is a small (+0.16 dex)
residual between large opposite-sign 1D offsets (+0.45 from `log r_s`,
−0.32 from `log ρ_s`). Run `python tests/integration/analyze_asimov.py` to reproduce; full write-up
in **`stage3.md` → Asimov verification of the J-bias source**.

**Reporting policy** (per `pipeline_overview.md` Stage 4): the main pipeline
publishes median + q16/q84 + 1D-KDE MAP for `log J(θ)` and `log D(θ)` per
galaxy at every angle. We are reproducing P&S 2018 and do not attempt to
remove the median−MAP gap (no informative concentration-mass prior, no
reparametrization). Validation/diagnostic runs (this MC, the Asimov)
compare *both* median and MAP to truth.

## Files

### Original procedure (from the user, unmodified)

- `jeans.py` — spherical Jeans projection (NFW + Plummer + constant β).
- `test_jeans.py`, `test_jeans_vs_quad.py` — original test harnesses.
- `stage1.md`, `data_sources.md`, `uncertainty_conventions.md` — design docs.

### Modified design docs

- `pipeline_overview.md` — added Decisions Log entries (2026-05-01) for the
  MC validation and the M(r_½) chain additions; added M(r_½, 3D) to the
  Stage 4 diagnostic-plot list. Stage 3 section reduced to a brief summary
  pointing to `stage3.md` (mirrors the `stage2.md` convention).
- `stage2.md` — new "MC recovery test" subsection (sibling of "Verification"),
  scoped to halo + `M(r_½)` recovery; J/D recovery and the Asimov verification
  moved to `stage3.md`. Outputs section extended with `log10_M_half_2d`
  and `log10_M_half_3d` chain spec.
- `stage3.md` — **new** canonical Stage 3 reference. Inputs/outputs, the
  J/D integrals and small-angle approximation, numerical implementation
  (u-grid, R-grid, knob calibration), the four reporting angles, the
  tidal-radius computation, the MC recovery test, and the Asimov
  verification of the J-bias source.

### MC test scripts

- `src/dwarfjeans/mocks/galaxy.py` — generates per-star (R, V, σ_ε, p) catalogs from a
  Plummer surface density + Jeans-evaluated σ_los. Stores both 2D-projected
  and 3D-half-mass radii in the truth dict. Exposes
  `make_asimov_galaxy()` (deterministic stratified-CDF R, truth-evaluated
  σ_los, V placeholder) and `plummer_R_stratified()` for reuse.
- `src/dwarfjeans/jeans/inference.py` — likelihood, prior_transform, dynesty wrapper.
  Standard `make_loglike` plus `make_loglike_asimov` (substitutes
  `(V_i − V_sys)² → σ_tot,truth²(R_i)`). `run_inference(..., asimov=True)`
  routes to the Asimov likelihood. Halo-side summary:
    - `summarize_posterior(samples_eq, truth, asimov=False)` → V, log r_s,
      log ρ_s, β̃, log(ρ_s·r_s³), log M(r_½, 2D), log M(r_½, 3D). When
      `asimov=True`, V is flagged `prior_only` and its z-score is `nan`.
  J/D push is not in this module; live callers compute it inline via
  `dwarfjeans.jd.factors` (see `scripts/run_production.py`).
- `src/dwarfjeans/jd/factors.py` — Stage-3 J/D integrals for an NFW halo with hard
  truncation at r_t. Small-angle approximation (R = d·θ).
  Includes Msun/kpc → GeV/cm unit conversion factors.
- `tests/integration/run_ufd_population.py` — runs 15 UFD seeds, prints running
  table with TRUTH row, computes population diagnostics for halo recovery.
  `--asimov` flag runs the single Asimov realization instead. (The original
  standalone `run_jd_summary.py` was not migrated and has been removed; the
  inline J/D push inside the MC loop is a pending follow-up — see
  **Stage-3 J/D MC recovery** above.)
- `tests/integration/analyze_asimov.py` — Asimov J-bias source diagnostic. Reads
  `compact_ufd_asimov.npz`, pushes the full chain through J/D at the
  four reporting angles, computes the chain-MAP-vs-median-vs-truth
  decomposition for `log J`, the small-x analytic test, the
  marginal-projection decomposition, the J/D bias-ratio scaling, and
  the prior-edge diagnostic. Writes
  `results/tests/ufd_population/asimov_chain_diagnostics.npz` and (if matplotlib is
  available) the triptych plot
  `results/tests/ufd_population/asimov_diagnostic.png`. Run after a fresh Asimov chain
  to re-verify the bias attribution.

### Output artifacts (in `results/tests/ufd_population/`)

- `compact_ufd_seed{0..14}.npz` — per-realization equal-weight chains
  + per-star data + truth dict. Bundled in this zip (~1.3 MB total).
- `compact_ufd_asimov.npz` — Asimov chain + Asimov per-star data + truth
  + `is_asimov: True` flag. Same schema as the MC chains except V is a
  placeholder (the Asimov likelihood does not consume V_i).
- `ufd_pop_table_extended.json`, `ufd_pop_diagnostics_extended.json` —
  Stage-2 halo + M(r_½) results from the validated run.
- `ufd_pop_jd.json`, `ufd_pop_jd_diagnostics.json` — Stage-3 J/D results
  on the 15 MC chains (with `thin_to`, `n_R`, `n_u` recorded).
- `ufd_asimov_table.json`, `ufd_asimov_jd.json` — Stage-2 + Stage-3
  Asimov-mode summaries.
- `asimov_chain_diagnostics.npz`, `asimov_diagnostic.png` — Asimov J-bias
  source diagnostics: per-sample chain values for `log_M3`, `log J` at
  the four angles, `log D` at α_c, the `2 log ρ_s + 3 log r_s` small-x
  proxy, plus a triptych figure (M(r_½) marginal, joint
  `(log r_s, log ρ_s)` posterior with truth + 2D MAP + prior bounds,
  log J(α_c) marginal with median + 1D-KDE MAP + truth).

## How to reproduce / continue

```bash
# Stage 2 halo + M(r_½) recovery + inline Stage 3 J/D push:
python tests/integration/run_ufd_population.py

# Asimov dev-loop check (single deterministic realization):
python tests/integration/run_ufd_population.py --asimov  # produces compact_ufd_asimov.npz

# Asimov J-bias source decomposition:
python tests/integration/analyze_asimov.py
```

Sequential because the test box has 1 CPU. On a multi-core node the MC
sweep wraps trivially in `concurrent.futures.ProcessPoolExecutor` over
the seed list. The Asimov runs are inherently single-threaded (one chain).

## Known limitations

See "What the MC test does *not* validate" in `stage2.md` → "MC recovery
test". Specific to the J/D extension:

- **d, r_t held fixed at truth.** Production Stage 3 marginalizes d via
  the LVDB distance-modulus split-normal prior and computes r_t from
  Springel+08 + Eadie & Harris 16. The synthetic data has no host distance
  or 3D position, so neither is meaningful here.
- **`r_t = 1 kpc` is the unresolved-σ_los default.** Our UFD mocks are
  resolved (σ_los ≈ 6.5 km/s) and a real galaxy at this scale would get a
  Springel-computed r_t. The choice is documented in `tests/integration/run_ufd_population.py` and `src/dwarfjeans/jd/factors.py`.
- **Small-angle approximation.** All reported angles are below 1°; small-
  angle gives <0.1% error at θ = 1°, well below posterior widths.
