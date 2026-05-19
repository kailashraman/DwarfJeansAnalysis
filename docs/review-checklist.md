# Review checklist

Recurring bug classes for adversarial review. Reviewers consult this before signing off on non-trivial diffs. **Append new entries when a bug is found whose class isn't already listed** — that's how this file earns its keep.

## Flags and membership

- **Graded vs. binary flags.** 0/1 hard flags with non-members tagged 0 are NOT "already cut upstream" — the cut must apply at use.

## Boundaries

- **Strict vs. non-strict inequality at boundary values.** `p > p_min` with `p == p_min`; `R < R_max` with `R == R_max`; `r_s > r_p` prior bounds. Decide deliberately and match the spec.
- **NaN / empty / single-element inputs.** χ² and variance helpers often break silently. Check what happens at length 0, length 1, and all-NaN.

## Units

- **Unit slips at module boundaries.** kpc ↔ pc, arcmin ↔ rad, deg ↔ rad. Re-derive direction at every call site, don't trust the variable name.
- **Project conventions.** `R` is **kpc** across staged catalogs; `rhalf_major_pc` is **pc** in the registry.

## Catalog metadata

- **`_meta` assumptions.** Code reading `_meta["X"]` should fail loudly, not silently default, when X is missing on a catalog where the field is load-bearing.
- **Adapter docstring vs. `_meta` vs. paper.** Verify per-paper claims (instruments, σ_sys, etc.) against the source paper and the adapter's own `_meta` notes.

## Defaults and fallbacks

- **Caller-side defaults to silently-biasing values.** When a function defaults a missing kwarg / registry-row field to a "neutral" value (e.g. `ellipticity → 0`, `sigma_sys → 0`, `pmra → 0`), check *every* production call-site explicitly passes the field. A field that the function "supports" but every caller forgets to plumb is a silent fallback that biases results without surfacing. Defense: distinguish *missing key* (caller bug, raise) from *NaN value* (legitimate "unmeasured", apply the documented neutral default).

## Vectorization

- **`np.interp(arr.ravel(), xp, fp).reshape(arr.shape)` assumes C-contiguity.** The ravel+reshape round-trip preserves element-to-position mapping only when the source is C-contiguous (or a view that ravels in C order). Refactors that introduce transposes, `np.moveaxis`, or strided slicing upstream of the interp can silently scramble the mapping. When using this idiom, keep the source array C-contiguous (e.g. build it directly from broadcast ops, not from a transpose) and prefer `arr.reshape(-1)` over `arr.ravel()` if you want a hard error on non-contiguous inputs.
- **`np.trapezoid(y, x, axis=k)` with 2D `x`.** Integrates row-/column-wise with per-row abscissae when `x.ndim == y.ndim`. Correct, but verify the `axis` matches the abscissa-varying axis; mismatched axis silently integrates over the wrong dimension and produces plausible-looking nonsense.
- **Default grid sizes that are tuned in tests but used at lower resolution in production.** When a tabulation function exposes `n_inner`/`n_outer` defaults, ensure the unit test that gates its accuracy exercises the *production* default (or pins a separate, looser gate at the production default). A test that passes explicit high-resolution args won't catch a default downgrade.

## Documented-but-unplumbed registry overrides

- **Registry override fields mentioned in docs but not wired in code.** Fields like `vlos_prior_halfwidth` are described in plan docs and doc strings as "per-galaxy overrides" but may never be read from the registry row at the call site — the module-level constant is used instead. When a doc change repeats or extends such a claim, verify the field is actually consumed. Defense: `row.get("vlos_prior_halfwidth", V_HALFWIDTH)` at the call site, not a module constant.

## Silent weight domination from near-zero inputs

- **Clipping rather than rejecting near-zero denominators in weighted means.** When computing IVW (or any 1/σ² weighting), a `np.clip(sigma, floor, None)` silently promotes an anomalously small σ to enormous weight, pulling the weighted mean to that star's value. For physical inputs (spectroscopic σ_eps ≥ 1 km/s in practice), the correct defense is an assertion / loud error on `sigma_eps.min() < physical_floor`, not a silent clip to 1e-6.

## CLI arg vs. run metadata mismatch for output paths

- **Script output path uses CLI arg instead of run's recorded metadata.** When a script accepts both `--run-dir` (explicit run) and a parameter flag (e.g. `--prior`), the output path must be derived from the *run's own metadata* (e.g. `audit["prior_name"]`), not from the CLI flag. Using the CLI flag silently places outputs in the wrong directory when the two disagree — e.g. `--run-dir results/production/X/loguniform/ --prior jeffreys` writes plots to `plots/X/jeffreys/`. Defense: after loading audit/metadata, resolve the effective parameter from the run record and use that for all downstream path construction.

## Guard conditions

- **Docstring quantifier wrong on a guard predicate.** When a flag or boolean (`perspective_correction_applicable`, `vlos_sigma_unresolved`, etc.) is set by checking N fields, the docstring must state the correct N. Off-by-one or wrong-count prose ("all four" when the code checks six) misleads future callers about what inputs are required to set the flag. Verify count of variables in the `all(...)` / `any(...)` call matches the prose.

- **Proxy-variable gates re-introduce the bias the correction was meant to remove.** When deciding whether to apply a correction, gate on the actual quantity being corrected — not a structural correlate. A half-light-radius gate on the perspective-motion correction (R_h ≥ 5′) silently skips compact high-μ UFDs (Segue 1, Boötes II) whose peak |Δv| ≳ 1.5 km/s injects ~8% bias on σ_los². Defense: gate on `max|Δv_persp|` (or a per-star noise-floor ratio) directly. Reach for proxies only when the actual quantity is unavailable, and document the failure modes explicitly.

## Calibration claims and statistical power

- **Bias bound vs. N_realisations.** When a mock-calibration harness reports `bias = X%` with dispersion `s` over `N` realisations, the 1σ uncertainty on the bias is `s/√N / truth`. A claim "recovers Y to <Z%" is supported only if `s/√N / truth ≪ Z%`. Concretely: N=8 with per-realisation dispersion ~12% of truth has SE on the bias ≈ 4% of truth — it cannot support a "<5%" claim, let alone a "<1%" one. Defense: write the claim in the form "no bias detected at the ±(SE)% level at N=K realisations", or run enough realisations that SE is comfortably below the tolerance.
- **PM (or any nuisance) truth equal to prior central.** A calibration that draws each realisation with the same truth as the prior central tests prior plumbing, not data-driven recovery. The posterior on that nuisance will read back the prior at ~100% coverage and look "perfect". To exercise the data, offset truth from the prior central by ≥1σ_prior in a follow-up.
- **Nuisance priors pinned at truth.** Tight Gaussian priors centered exactly at the mock's truth values isolate the component under test but inflate the apparent calibration when reported without that caveat. Either widen to production-realistic priors, or label the result "given truth-centered nuisance priors".

## Inverse-CDF / sampler boundary handling

- **Hard-clip of a Gaussian (or other unbounded) conditional to a box produces a delta at the boundary.** When a prior_transform synthesises a Gaussian conditional `mu + sigma * ndtri(u)` and then `np.clip`s the result to `[lo, hi]`, draws whose tail crosses the boundary pile up *exactly at the boundary* — a point mass that (i) is not the truncated Gaussian one would get from `truncnorm.ppf`, and (ii) biases marginals toward the bound without surfacing. Tests that filter out the boundary-pinned draws before checking moments cannot detect drift. Defense: use `truncnorm` so the conditional is properly renormalised on the box, or instrument a counter and abort if the boundary-clip fraction exceeds a small tolerance over a representative draw set.
- **Defensive clipping on a continuous prior_transform.** More generally: don't add `np.clip`/clamping to numerical transforms unless the underlying math requires it. "Schema consistency with the other priors" is not a math requirement — it is a defensive instinct that creates a delta mass where the spec calls for unbounded support. If the chain schema demands a box, use a properly normalised truncated distribution; if not, leave the conditional alone.

## Monotonicity fixes

- **Nudge-loop + forced terminal assignment re-introduces a tie.** When enforcing strict monotonicity by nudging ties upward in a loop (`for i in range(1, n): if a[i] <= a[i-1]: a[i] = a[i-1] + eps`) and then unconditionally setting `a[-1] = constant`, a cascade that advances `a[-2]` to ≥ `constant` produces a non-monotone endpoint. Fix: stop the loop at `n-1` (exclude the last element) before the forced assignment, so the terminal pin is always the last write.

## Fix verification

- **A bug fix without a regression test that fails on pre-fix code.** "Manually reran and the symptom is gone" is not verification — it confirms the symptom didn't reproduce *this time*, not that the underlying defect is gone. The fix must come with a test that (i) fails on the pre-fix code and (ii) passes after. If the test is hard to construct, that itself is information about whether the fix is well-scoped. Particularly important when the bug was discovered by review rather than by failing tests: by definition the existing test suite did not catch it, so a new test is needed to keep it caught.
- **Tests that mask the defect they would catch.** Filtering out boundary-pinned draws, dropping non-finite returns, or skipping degenerate-input cases *inside the test* before checking moments hides the very pattern the test should exercise. If the production code returns values in some regime, the test must check those values, not exclude them.

## Writeup drift on third-party formulae

- **Paraphrased external formula loses a branch.** When a docstring or LaTeX paragraph summarises a sibling repo's piecewise formula (e.g. Santos-Santos+22 scatter `np.where(Vmax < 57, -1.26*log10(Vmax/88.6), 0.24)`), the prose must reproduce *both* branches. Collapsing "0.24 above the break" to "zero above the break" silently misrepresents the upstream — and unlike a code bug, no test will catch it. Defense: when paraphrasing a `np.where(...)` from another repo, write out both branches verbatim, even when one looks "close enough" to zero.

## Adaptive-floor proxy vs. actual per-bin ESS

## Density-ratio weights described as plain likelihoods

- **Importance-weight construction misread as a Gaussian likelihood.** When an upstream / sibling-repo weight is built as a density ratio `f_obs(x_i) / p_catalog(x_i)` (importance reweighting from the catalog distribution onto the observation), prose that calls this "a Gaussian in x at the upstream model's predicted mean and scatter" loses the denominator entirely. The denominator (a KDE or analytic model of the catalog distribution) reshapes the effective weight in ways the plain-Gaussian description cannot capture, and the numerator may itself be a *split-normal of the observation* (asymmetric obs error bars) rather than a symmetric Gaussian at the model's intrinsic scatter. Defense: when documenting any weight produced by another repo, read the full construction path before paraphrasing — specifically check whether the weight is normalised against a catalog distribution, and whether the "Gaussian" is centred on the model's predicted value (with model scatter) or on the observation (with obs error). The two are not interchangeable.

## Adaptive-floor proxy vs. actual per-bin ESS

- **Global ESS/n_bins proxy for per-bin ESS can be loose when weights are heterogeneous within bins.** The adaptive-bin-reduction guard (`if ESS / n_bins < floor: reduce n_bins`) uses the *global* ESS divided by the proposed n_bins as a proxy. But weighted-quantile binning equalises *total weight* per bin, not ESS. If one halo dominates weight within a bin, the actual per-bin ESS can be a fraction of the proxy. Example: danieli23_const/segue_1 has global ESS=4350, n_bins reduced to 145 (proxy=30 exactly), but 28/145 bins have actual per-bin ESS < 30, and 4 bins have ESS < 5. The table is still used by the transform without any runtime warning. Defense: after binning, check `ess_per_bin.min()` against the floor and either raise (or warn loudly) if more than a small fraction of bins fall below it. At minimum, record `ess_per_bin_min` in the build summary and emit a warning at table-load time if it is critically low (e.g., < 5).
