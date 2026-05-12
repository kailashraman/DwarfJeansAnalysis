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
