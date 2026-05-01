# Stage 3 — J/D-factor computation from posteriors

Pushes the Stage 2 halo posterior chains through line-of-sight + solid-angle integrals to produce the J(θ_max) and D(θ_max) chains that are the headline output of the pipeline. Cheap relative to Stage 2 (a few seconds per galaxy chain), embarrassingly parallel across galaxies and across draws within a galaxy.

This document is the canonical Stage 3 reference. The high-level role in the pipeline is summarized in [`pipeline_overview.md`](./pipeline_overview.md); deviations from P&S 2018 are consolidated in the Differences section there.

---

## Inputs / outputs

**Per-galaxy input.** The Stage 2 posterior chain over `(V, log r_s, log ρ_s, β̃)` plus joint posterior draws of the nuisances `(d, r_p, ε, μ_α cosδ, μ_δ)`. Production runs marginalize over `d` (and over `r_t` via the Springel et al. 2008 host-mass formula); the validation runs in `mc_test/` hold both at truth.

**Per-galaxy output.** Equal-weight posterior chains for `log₁₀ J(θ_max)` and `log₁₀ D(θ_max)` at four θ_max values each:

- `θ_max ∈ {0.1°, 0.2°, 0.5°, α_c}` for J, where `α_c = 2 r_½,3D / d` is the Wolf+2010 critical angle.
- `θ_max ∈ {0.1°, 0.2°, 0.5°, α_c/2}` for D. (D's natural angle is α_c/2; the J convention follows P&S 2018.)

Reported per galaxy as median + q16/q84 + 1D-KDE MAP for each angle (see `pipeline_overview.md` Stage 4 outputs and the Stage 4 reporting decision-log entry).

---

## The integrals

In the small-angle approximation, valid for the θ_max ≤ 0.5° regime of all reported angles (max α_c ≈ 0.5° for typical UFDs at d ≳ 25 kpc):

```
R = d · θ                                 (impact parameter)
dΩ ≈ R dR / d² = dA / d²                  (small-angle solid angle element)
J(θ_max) = (2π / d²) ∫_0^{R_max} R dR · I_2(R; r_t)
D(θ_max) = (2π / d²) ∫_0^{R_max} R dR · I_1(R; r_t)
```

where `R_max = min(d · θ_max, r_t)` and the line-of-sight column at impact parameter `R` for an NFW with hard tidal-radius truncation at `r_t` is

```
I_n(R; r_t) = 2 ∫_R^{r_t} ρ^n(r) · r / √(r² − R²) dr        (n = 1 or 2).
```

The integrable `√(r² − R²)` singularity at `r = R` is removed by `r² = R² + u²`:

```
I_n(R; r_t) = 2 ∫_0^{u_max(R)} ρ^n(√(R² + u²)) du,    u_max(R) = √(r_t² − R²).
```

For NFW `ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]` both integrands are smooth on `u ∈ [0, u_max]`.

**Reporting units.** The pipeline keeps `(M⊙, kpc)` internally and converts to `(GeV²/cm⁵)` for J and `(GeV/cm²)` for D at output (P&S 2018 convention). Conversion factors are module-level constants in `j_d_factors.py`.

---

## Numerical implementation

(See `j_d_factors.py` for the implementation; this is the design rationale.)

**u-grid for the column integrals.** Per impact parameter `R`, log-spaced from `u_min = 1e-6 · r_s` to `u_max(R)`, plus an explicit `u = 0` endpoint where the column integrand is largest. Vectorized as a 2D `(n_R, n_u)` array. Trapezoid rule on log-spaced points.

**R-grid for the outer integrals.** Log-spaced from `R_min = 1e-6 · r_s` to `R_max`. Critically, **`R = 0` is not a grid point** — the integrand `R · I_n(R)` is zero there (area element kills the on-axis NFW cusp), but evaluating `I_n(0)` would hit `ρ(0) = ∞` in float64 and produce a `0 · ∞ → NaN`. The trapezoid is closed by prepending an explicit `(R = 0, integrand = 0)` endpoint.

**Default knobs and their cost.** `n_R = 128, n_u = 512` is the validated production setting, calibrated against `scipy.integrate.quad` to better than ~10⁻³ relative error over the parameter range relevant for MW dwarfs. A single `(J, D)` evaluation across all four reporting angles takes a few ms; a 4000-sample chain pushes through in a few seconds.

The MC-sweep speedup `(thin_to=200, n_R=48, n_u=96)` documented in the [Decisions Log](./pipeline_overview.md#decisions-log) is calibrated against `(500, 64, 128)` and agrees on population diagnostics to within 15-realization MC noise. CLI knobs `--thin-to`, `--n-R`, `--n-u` in `run_jd_summary.py` toggle between speed and stricter resolution.

---

## Tidal-radius computation

Per posterior draw, `r_t` is computed from Springel et al. 2008 eq. 12 using the host's enclosed-mass profile and the satellite's instantaneous distance from the host:

- **MW satellites (default).** Eadie & Harris 2016 host mass profile. The pipeline scope is MW satellites only, so this is the default for every galaxy.
- **LMC- and SMC-hosted satellites.** Use the LMC/SMC enclosed-mass profile at the appropriate level of the host chain (i.e., compute the satellite's `r_t` in the LMC/SMC potential, not the MW potential). Falls back to MW if the LMC/SMC profile is not yet implemented — acceptable approximation given the small contribution of the satellite-host's mass to `r_t`.
- **Unresolved-σ_los systems.** Fixed `r_t = 1 kpc` independent of host. The Stage 1 classification flags the galaxy and the production runner reads the flag and forces this value, since for unresolved systems the kinematic data don't constrain the halo well enough for the host-orbit dependence of `r_t` to matter.

`r_t` is finite for every angle of interest (`R_max ≪ r_t` for all four reporting angles in MW-satellite parameter space), so the truncation only affects the column integrals' upper limits, not the regime where the integrand is large.

---

## MC recovery test

End-to-end statistical validation: 15 mock UFD realizations at fixed truth (`r_s = 0.3 kpc`, `ρ_s = 3 × 10⁸ M⊙ kpc⁻³`, `r_p = 0.05 kpc`, `β = 0`, `N_stars = 30`, `σ_ε = 2 km/s`), with `d = 30 kpc` and `r_t = 1 kpc` held fixed at truth (synthetic data have no host distance or 3D position to drive a Springel+08 r_t computation). Each `(r_s, ρ_s)` draw is pushed through the small-angle line-of-sight integrals at the four J angles and the four D angles. `α_c = 2 r_½,3D / d ≈ 0.249°` at truth. Run as:

```
python run_jd_summary.py        # 15-MC J/D push, ~30 s wall at the sped-up settings
python run_jd_summary.py --asimov   # Asimov J/D push, ~3 s
```

**D-factor recovery is clean** at all four angles: median bias ≤ 0.03 dex, std(z) ≤ 1.15, max `|z|` ≈ 2.1, KS p ∈ [0.37, 0.79], with mean 1σ widths 0.15–0.36 dex.

**J-factor recovery is unbiased to ~0.1 dex** with mildly under-dispersed posteriors: std(z) up to 1.18 at J(α_c), 1–3 realizations crossing `|z| = 2` across the four J angles, max `|z|` ≈ 2.3, with a small population-level high-side median bias of +0.08 to +0.12 dex. The bias is the median of `log₁₀ J` along a wide-and-curved chain ridge being offset from `log₁₀ J` at the ridge peak; the J/D ratio of the bias magnitude (≈ 2× across angles) is consistent with the density-squared kernel in J vs. linear in D. The realization-by-realization z-scores for J(α_c) and D(α_c/2) track `z(M(r_½, 3D))` tightly (sign agrees on every seed, magnitude correlates), as expected since the Wolf+10 angle probes the same well-constrained mass profile that anchors `M(r_½)`. See the [Asimov verification](#asimov-verification-of-the-j-bias-source) below for the chain-MAP-vs-median decomposition that pins this attribution.

The J/D push uses `thin_to=200, n_R=48, n_u=96`, calibrated against `(thin_to=500, n_R=64, n_u=128)` — population diagnostics agree to within 15-realization MC noise — for an ~8× speedup (~30 s wall vs. ~4 min for the full sweep). Result artifacts: `mc_results/ufd_pop_jd.json`, `mc_results/ufd_pop_jd_diagnostics.json`.

The Stage 2 halo-side validation on the same chains (the `M(r_½, 3D)` recovery check, the Asimov dev-loop construction) lives in `stage2.md` § MC recovery test. The two pieces share the same chains and the same scripts; the split here is by what's being validated.

What this test does *not* validate: nuisance marginalization (`d`, `r_t` held at truth), the production sampler config (test uses `sample='unif'`, `nlive=300`, `dlogz=0.5` for run-time tractability vs. production's `'rwalk'`, `nlive=500`, `dlogz=0.1`), or the Springel+08 `r_t` computation in the integrals. A second-pass run with full nuisances and production sampler settings is a Stage 3 prerequisite once the host-mass profile machinery is wired up.

---

## Asimov verification of the J-bias source

The Asimov chain (`mc_results/compact_ufd_asimov.npz`, 1803 equal-weight samples; constructed in `stage2.md` § Asimov dev-loop check) is informative enough by itself to localize the J-factor bias. The diagnostics below resolve two questions: (i) is `M(r_½)` recovered cleanly, and (ii) where does the +0.15–0.23 dex Asimov J-bias actually come from. Reproduce with `python analyze_asimov.py` (full diagnostic), output saved to `mc_results/asimov_chain_diagnostics.npz` and `mc_results/asimov_diagnostic.png`.

**`M(r_½, 3D)` is clean.** Pushing each chain `(r_s, ρ_s)` draw through `M(r_½ = r_½,truth)` gives median offset +0.038 dex on σ = 0.142 dex (matches MC ⟨σ⟩), skewness +0.19, excess kurtosis +0.06, and (q84−q50) − (q50−q16) = −0.005 dex. Near-symmetric, near-Gaussian, σ matches MC. The +0.18 / −0.16 dex marginal offsets in `log r_s` and `log ρ_s` project out cleanly along the well-constrained `M(r_½)` direction.

**The J-bias is post-chain transformation skew on the curved J(r_s, ρ_s) surface.** The single cleanest diagnostic: chain-MAP vs. chain-median of `log₁₀ J(α_c)`.

| quantity | offset from truth (dex) |
|---|---|
| 1D-KDE MAP of `log₁₀ J(α_c)` | **+0.024** |
| chain median of `log₁₀ J(α_c)` | +0.162 |
| chain mean of `log₁₀ J(α_c)` | +0.212 |

The MAP of the J posterior sits essentially at truth; the bias lives entirely in the median of a positively-skewed transformation distribution. Same pattern at all four angles (1D-KDE MAP offsets +0.13 / +0.07 / +0.09 / +0.02 vs median offsets +0.13 / +0.16 / +0.23 / +0.16). This is the Asimov-procedure-correctness check: the procedure puts the J-posterior peak at truth; the median offset is a property of the posterior shape, not a bias in what the procedure is recovering.

**The "small-x analytic" attribution (`log J ≈ 2 log ρ_s + 3 log r_s + const`) is wrong.** The chain median of `2 log₁₀ ρ_s + 3 log₁₀ r_s` is +0.42 dex, vs. the actual +0.13 dex bias on J(0.1°) where the small-x limit applies most. Per-sample regression of `Δ log₁₀ J` on the small-x prediction has slopes 0.29–0.53 (not 1) and r² of 0.08–0.52 across the four angles. The chain spans `log r_s ∈ [−2, +1]` (3 dex), against an integration aperture of order 0.05 kpc, so the chain crosses the small-x → large-x regime and `log J(r_s, ρ_s)` is *not* log-linear over its support.

**Marginal-projection decomposition at α_c, holding one channel at truth and pushing the other:**

| projection | median bias (dex) |
|---|---|
| `log r_s` only (`log ρ_s = truth`) | +0.46 |
| `log ρ_s` only (`log r_s = truth`) | −0.32 |
| sum (would equal full if chain were log-linear in both channels) | +0.14 |
| full joint chain | +0.16 |
| non-additive remainder | +0.02 |

The full bias is a small residual between two large opposite-sign 1D-projection offsets. Non-additivity is small (0.02 of 0.16), so the chain ridge is approximately log-linearly aligned and the residual curvature is what's left over after projecting through the J transform.

**D-bias scales like density-channel-only.** The J-bias / D-bias ratio at the four angles is 2.0–2.6 — consistent with J's density-squared kernel and D's linear kernel both seeing the same posterior-curvature skew, but with different powers in the density channel. (The "small-x analytic" picture predicts 2× to 3× depending on which channel dominates, so this part of the small-x intuition survives; the absolute prediction does not.)

**Joint 2D KDE peak in `(log r_s, log ρ_s)` is at (−0.77, +8.86) vs truth (−0.52, +8.48).** Even with the MLE at truth, the *posterior* peak is offset along the ridge by a uniform-prior volume effect. `log J` evaluated at the 2D KDE peak is +0.10 dex above truth — partway between the 1D-MAP-of-log-J (+0.02) and the median (+0.16). The 1D MAP of log J pulls back further toward truth because the J-surface curvature on the ridge maps the joint-peak position to a J-value that the marginal still places at truth.

**Prior-edge influence is small.** Of 1803 chain samples, 7.5% sit within 0.20 dex of the upper `log r_s` prior bound and 3.4% within 0.20 of the lower. Nonzero but the bias is dominated by the ridge interior, not the edges; the prior box is `U(−2, 1)` to match P&S 2018 and we keep it.

**Conclusion.** The Asimov procedure is correct: MLE at truth, M(r_½) recovered cleanly with σ matching the MC, and the J-posterior peak (1D MAP of log J) at truth to within 0.02–0.13 dex across angles. The +0.15–0.23 dex J *median* offsets are a property of pushing a wide curved ridge through a curved transform — what an earlier handoff called "Jensen on a wide log ρ_s posterior" was right in spirit (a curved transform applied to a wide posterior) but the quantitative attribution to a small-x convex transform is wrong: the chain leaves the small-x regime, J(r_s, ρ_s) is not log-linear over the chain, and the bias is the median offset of a nonlinear projection on a strongly degenerate ridge.

The pipeline does not attempt to remove this offset — we are reproducing P&S 2018 and a wide posterior is what the data give at UFD-scale `N_stars`. The reporting policy is to publish median + q16/q84 + 1D-KDE MAP for every J(θ) and D(θ), per galaxy, so the gap is visible to downstream consumers. Validation runs (this Asimov, the 15-realization MC) compare both median and MAP to truth, separately. See `pipeline_overview.md` Stage 4 — Outputs & diagnostics, and the 2026-05-01 reporting-strategy decision-log entry.

---

## Halo profile interface

The J/D integrators consume `density(r)` from the same `HaloProfile` interface used by Stage 2 (which consumes `enclosed_mass(r)`). See `stage2.md` § Halo profile interface for the architectural rationale. Adding non-NFW profiles (Burkert, Einasto, generalized α-β-γ) to Stage 3 is a no-op once the new `HaloProfile` subclass is registered: the column-integral and outer-integral structure is profile-agnostic, modulo profile-specific cross-checks (e.g. the analytic J formula in P&S Appendix A is gated on the active profile being NFW).
