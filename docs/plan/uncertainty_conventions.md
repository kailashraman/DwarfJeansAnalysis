# Uncertainty Conventions

This document collects the settled conventions for storing, modeling, and propagating uncertainties throughout the pipeline. It is the single source of truth for these decisions; other docs link here rather than restating the rules.

**Status (Segue 1 test).** The split-normal infrastructure described here is the design target for the population pipeline. The Segue 1 test (`Segue1_test/run_segue1.py`) uses **symmetric Gaussians** as an interim step (truncated for ε): `d ~ N(23.0, 2.0)`, `ε ~ N(0.47, 0.11)` truncated to `[0, 1)`, `rhalf_arcmin ~ N(4.31, 1.03)`. The implementation lives in `make_prior_transform_with_nuisances` in `docs/plan/jeans_inference.py`; swapping in split-normal inverse-CDFs (per "Implementation note" below) is a one-function-replacement once the LVDB asymmetric-error fields are pulled per-galaxy.

The constant-σ block (Walker+2006) added to `Segue1_test/run_segue1.py` uses a different prior — the **proper Jeffreys (Fisher-determinant) prior on σ_los**, derived from the (V̄, σ) Fisher matrix of the membership-weighted likelihood. The algebra and the implementation pointer are in `docs/plan/segue1_test.md`. That prior is for a different parameter (σ_los, not d/ε/rhalf) and so is orthogonal to the split-normal convention here.

---

## Storage policy

For every measured input quantity, the registry stores **raw asymmetric error bars** (`_em`, `_ep`) in the LVDB's native units. **No symmetrization, no log-space conversion, no derived single-σ estimate is ever baked into the stored value.** Symmetrization and any prior-shape choice happen at the *point of use* (Stage 2 prior transform), never at ingest.

This applies uniformly to:
- `rhalf`, `ellipticity`, `position_angle` (structural)
- `distance_modulus` (distance)
- `apparent_magnitude_v` (photometric)
- `vlos_systemic`, `vlos_sigma` (kinematic)
- `pmra`, `pmdec` (proper motions, when measured)

Derived quantities (e.g., `r_1/2 = rhalf × √(1−ε)`, `d` from μ) are **not stored with associated errors**. Their full distributions are obtained at the point of use either analytically (for monotonic single-input transforms) or by sampling from the joint priors of the raw inputs.

A diagnostic-only summary (median + 16/84 percentiles in physical units) may be computed from the raw inputs and stored alongside, clearly labeled as derived/diagnostic, for plots and tables. These derived summaries never enter the likelihood.

---

## Prior modeling: split-normal distributions

When a quantity needs to enter a likelihood as a prior, build a **split-normal distribution**: stitch a half-normal on the lower side using `_em` as the scale, joined to another half-normal on the upper side using `_ep` as the scale.

**Sign convention.** Per the LVDB documentation, `_em` and `_ep` are both stored as **positive magnitudes**: `_em` is the size of the lower 1σ error bar (the value at the 16th percentile is `x_0 − _em`), and `_ep` is the size of the upper 1σ error bar (the 84th percentile is `x_0 + _ep`). The split-normal scales `σ_-` and `σ_+` are taken as `_em` and `_ep` directly, with no sign flip and no symmetrization.

### Split-normal density

For a value `x_0` with lower scale `σ_-` and upper scale `σ_+`:

```
f(x) = A · exp[ -(x - x_0)² / (2σ_-²) ]   for x ≤ x_0
f(x) = A · exp[ -(x - x_0)² / (2σ_+²) ]   for x ≥ x_0

A = √(2/π) / (σ_- + σ_+)
```

Properties (verified numerically):
- Both halves take value `A` at `x_0` — density is **continuous** at the seam by construction.
- The derivative is **discontinuous** at `x_0` unless `σ_- = σ_+`. Irrelevant for nested samplers (no gradient information used); would matter for HMC/NUTS.
- Integrates to 1 over (−∞, +∞).
- Mode is at `x_0`. Mean and variance shift off `x_0` when the halves are unequal.
- Reduces exactly to a regular Gaussian when `σ_- = σ_+`.

**Approximation note.** LVDB central values for most quantities are reported as the median of the original posterior, with `_em` / `_ep` as 16/84 percentile half-widths around it. Our split-normal centers `x_0` on the LVDB central value and treats `_em` / `_ep` as half-Gaussian scales. The mode of the resulting split-normal is at `x_0` (matching the LVDB central value as if it were a mode), but the 16/84 percentiles of the split-normal are not exactly at `x_0 ∓ σ_∓` — there's an O(σ_+ − σ_-) correction. For LVDB asymmetries typical of MW dwarfs (`_ep / _em` within ~10–20% of unity for most quantities) this is a sub-percent effect on the prior shape and does not warrant correction. We accept the approximation: the split-normal interprets `_em` / `_ep` as half-Gaussian scales rather than as exact 16/84 percentile offsets.

### Where the split-normal is used

This single primitive is used for every observation-based prior in Stage 2:

- `μ` (distance modulus) — split-normal in mag. The asymmetric distribution in `d [kpc]` emerges automatically from the `μ → d` transform, no special handling.
- `rhalf` — split-normal in arcmin.
- `ellipticity` — split-normal, truncated to `[0, 1)` (see truncation policy below).
- `pmra`, `pmdec` — split-normal each in mas/yr, treated independently (see proper-motion correlation below).

The free parameters of Stage 2 (`V`, `log r_s`, `log ρ_s`, `β̃`) use flat / Jeffreys priors set by us, not informative priors built from observations, so the split-normal machinery doesn't apply to them.

The LVDB-tabulated `vlos_sigma` is **not** consumed as a prior in the standard run — Stage 2 fits σ_los from the per-star Jeans likelihood directly. It's stored in the registry as metadata and used as a sanity check against Stage 1's fitted σ_los (the Stage 1 fit, not the LVDB value, is what determines the resolved/unresolved classification feeding Stage 2 priors).

---

## Truncation policy

Truncate prior distributions at every physical boundary, **uniformly** — without judgment calls about "is the lower tail small enough to skip the correction?"

| Quantity | Physical bounds | Truncation |
|---|---|---|
| `distance_modulus` (μ) | finite | none |
| `d` (distance, derived from μ) | `> 0` | inherits — never reaches 0 in practice |
| `rhalf` | `> 0` | truncate at 0 |
| `r_1/2` (derived) | `> 0` | inherits from `rhalf` |
| `ellipticity` | `[0, 1)` | truncate at 0 and 1 |
| `pmra`, `pmdec` | unbounded (signed) | none |

When the LVDB has no measured ellipticity (or only an upper limit), Stage 0a substitutes a uniform `U(0, 0.5)` prior for that galaxy in place of the split-normal — see `pipeline_overview.md` Stage 0a. This is an alternative prior for the missing-data case and does not conflict with the truncation rule above (which applies to the split-normal informed by LVDB measurements).

Truncate by computing the truncated CDF over the allowed range and renormalizing:

```
f_trunc(x) = f(x) / [F(b) − F(a)]   for x ∈ [a, b]
```

The renormalization integral is computed once at galaxy-config time and cached.

---

## Prior-transform implementation

dynesty's prior transform takes a unit-cube point `u → θ` mapping the unit hypercube to physical parameters. For a split-normal, the inverse-CDF is piecewise but closed-form:

```
p_lower = σ_- / (σ_- + σ_+)        # area in lower half
if u ≤ p_lower:
    u' = 0.5 · u / p_lower         # remap to [0, 0.5]
    return x_0 + σ_- · Φ⁻¹(u')
else:
    u' = 0.5 + 0.5 · (u − p_lower) / (1 − p_lower)
    return x_0 + σ_+ · Φ⁻¹(u')
```

For truncated priors: compute the truncated CDF endpoints (`F(a)`, `F(b)`) once, then rescale `u` to span `[F(a), F(b)]` before inverting.

---

## Propagation

Two cases when a likelihood needs a derived quantity:

1. **Monotonic single-input transforms** — analytic. Sample the input at prior-transform time and apply the transform deterministically. Example: `μ → d`. The prior-transform code returns `d`; the asymmetric `d` distribution emerges automatically from the change of variables.

2. **Multi-input transforms** — sampled. Sample independently from the joint priors of the raw inputs and compute the derived quantity per draw. Example: `r_1/2 = rhalf × √(1 − ε)`. Inside Stage 2, sample `rhalf` and `ε` as separate nuisance parameters and compute `r_1/2` on the fly. Never store a "derived prior on `r_1/2`."

For diagnostic outputs only (registry summary, plots), Monte-Carlo propagation gives empirical 16/84 percentiles for derived quantities. These are clearly labeled as derived and never used in the likelihood.

---

## Proper-motion correlation

The LVDB ships `pmra_pmdec_corr`, encoding the joint error correlation from the original astrometric fit (Gaia EDR3 / HST). We **ignore this correlation** and treat `pmra` and `pmdec` priors as independent split-normals.

Justification (sketch — the bound is qualitative, not a calibrated figure):

- The correlation only affects the **uncertainty** on the perspective-motion correction (Walker et al. 2008 Appendix; Kaplinghat & Strigari 2008).
- The perspective correction itself reaches only ~0.01–0.04 km/s at the half-light radius for classical dwarfs.
- The correlation can change σ(perspective correction) by up to ~25%, but in absolute terms that's a sub-mK shift to the per-star velocity error budget.
- Propagated through the σ_los inference and then through the J-factor (`J ∝ σ_los^4`), the resulting shift in log J is far below the P&S reported per-galaxy errors of 0.05–0.3 dex — small enough to be observationally undetectable.
- Ultra-faints don't get the perspective correction applied at all (no measured proper motions in P&S), so the choice is moot for half the sample.

Operationally: independent priors match P&S 2018 exactly, simplify validation, avoid the technical complications of a true bivariate split-normal, and the deviation from a "more honest" treatment is observationally undetectable.

`pmra_pmdec_corr` is stored in the registry as metadata so a future analysis can retrieve it without re-ingesting.

---

## What this convention replaces

The pre-decision drafts of this pipeline used various ad-hoc symmetrization schemes (averaging `_em` and `_ep`, log-space symmetrization, split priors only when ratio exceeded a threshold). All of those are dropped.

The split-normal approach is **strictly more general**: in the symmetric case (`_em == _ep`, which is most galaxies for most quantities) it reduces exactly to an ordinary Gaussian. There is no asymmetric-vs-symmetric branching anywhere in the code; every observation-based prior gets the same treatment.

---

## Decisions Log entry

Cross-reference for `pipeline_overview.md`:

| Date | Decision | Notes |
|---|---|---|
| 2026-04-30 | Split-normal priors with native asymmetric errors | See uncertainty_conventions.md |
| 2026-04-30 | Truncate priors at all physical boundaries uniformly | Same |
| 2026-04-30 | Independent (uncorrelated) proper-motion priors | Effect on log J negligible vs reported errors; matches P&S 2018 |
