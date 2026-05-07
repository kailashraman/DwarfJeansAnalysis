# Per-paper multi-epoch combiner conventions

This is the single review table for the per-paper combiner handlers in
`src/dwarfjeans/ingest/combiners/`. A deep-reviewer audits each row
against the cited paper sections; QA-sweep #4 calibrates the σ_sys
floor and (where applicable) zero-point offsets and updates this table
in lock-step with the handler `DEFAULT_POLICY`.

The framework defaults (`CombinePolicy(sigma_sys_kms=0.0,
p_threshold=0.01, zero_point_offsets_kms=None)`) are inherited unless
the cited paper explicitly publishes a different choice. Every entry
that says **"not published, set 0"** is a flag for QA-sweep #4 to
investigate, not a claim that the paper guarantees zero.

## Schema

| field | meaning |
|---|---|
| `bibcode` | NASA ADS bibcode; matches `_meta["source_paper_bibcode"]` for dispatch |
| `galaxy/ies` | LVDB key(s) the handler is dispatched for |
| `instruments` | spectrographs contributing velocities in the published table |
| `σ_sys (km/s)` | systematic-floor added in quadrature *after* IVW |
| `p_threshold` | χ² p-value below which a star is flagged variable |
| `zero-point offsets` | per-instrument shifts applied to V *before* IVW; `None` = single instrument or already-applied in source |
| `paper §refs` | section / table references that justify the choices |
| `URL` | ADS link for verification |
| `status` | `verified` (deep-reviewer signed off) / `TODO` (framework only, calibration pending) |

## Table

| bibcode | galaxy/ies | instruments | σ_sys (km/s) | p_threshold | zero-point offsets | paper §refs | URL | status |
|---|---|---|---|---|---|---|---|---|
| `2017ApJ...838....8L` | `eridanus_2` | M2FS | 0.0 (not published, set 0) | 0.01 (default) | None | TODO §refs | https://ui.adsabs.harvard.edu/abs/2017ApJ...838....8L | TODO |
| `2018ApJ...857..145L` | `carina_2`, `carina_3` | M2FS, IMACS | 0.0 (not published, set 0) | 0.01 (default) | None (TODO confirm — paper may quote per-instrument zero-points) | TODO §refs | https://ui.adsabs.harvard.edu/abs/2018ApJ...857..145L | TODO |
| `2022ApJ...939...41C` | `grus_1` | MIKE, M2FS | 0.0 (not published, set 0) | 0.01 (default) | None (TODO confirm) | TODO §refs | https://ui.adsabs.harvard.edu/abs/2022ApJ...939...41C | TODO |
| `2023AJ....165...55C` | `tucana_2` | M2FS, IMACS, MIKE, MagE | 0.0 (not published, set 0) | 0.01 (default) | None (TODO confirm — multi-instrument compilation) | TODO §refs | https://ui.adsabs.harvard.edu/abs/2023AJ....165...55C | TODO |
| `2020ApJ...892..137S` | `tucana_4` | M2FS | 0.0 (not published, set 0) | 0.01 (default) | None | TODO §refs | https://ui.adsabs.harvard.edu/abs/2020ApJ...892..137S | TODO |
| `2024ApJ...968...21H` | `tucana_5` | M2FS | 0.0 (not published, set 0) | 0.01 (default) | None | TODO §refs | https://ui.adsabs.harvard.edu/abs/2024ApJ...968...21H | TODO |

## Verification protocol (QA-sweep #4)

For each row above, the reviewer:

1. Opens the paper at the cited section(s) and confirms (or corrects)
   the σ_sys floor, p_threshold, and zero-point offset choices.
2. Records the verbatim quote and section number in the handler's
   module docstring (the `paper §refs` column should always be
   resolvable to a specific section number, not just the paper).
3. Updates `DEFAULT_POLICY` in the handler if the paper publishes a
   different value than the framework default.
4. Flips `status` from `TODO` to `verified` in the table.
5. If the paper compiles velocities from multiple instruments with
   per-instrument zero-points, populates
   `policy.zero_point_offsets_kms` and adds the offset table to the
   docstring.

The deep-reviewer pass on the analysis pipeline (per CLAUDE.md
adversarial-review policy) is a hard gate before any of these handlers
are used in a production Stage 1 run.
