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
