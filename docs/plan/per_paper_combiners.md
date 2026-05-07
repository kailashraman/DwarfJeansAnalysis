# Per-paper multi-epoch combiner conventions

This is the single review table for the per-paper combiner handlers in
`src/dwarfjeans/ingest/combiners/`. Each row is verifiable against the
cited paper sections; QA-sweep #5 (calibration) revisits any value
flagged TODO and updates this table in lock-step with the handler
`DEFAULT_POLICY`.

## Framework convention (decided 2026-05-07)

The ingest combiner treats every published per-epoch error as
statistical and feeds it straight into the IVW. This is an
**approximation**: in reality σ_sys is instrument-correlated and should
not be averaged down by 1/√N. The strict treatment (deconvolve σ_sys,
IVW the σ_stat, re-add σ_sys post-combine) is a future swap-in; the
hooks (`CombinePolicy.sigma_sys_kms`, `multi_epoch.ivw_combine`'s
`sigma_sys` argument) are already in place.

Consequence: combined σ_vbar is biased low by ~10–30% at typical N=2–5
multi-epoch counts, χ² p-values biased high (under-flag variables).
Bias is below chain noise for current σ_los~few km/s inferences.

Under this convention, **`CombinePolicy.sigma_sys_kms = 0.0` for every
per-paper handler** because σ_sys is already in the published e_RVel.
Setting it non-zero would double-count.

See `src/dwarfjeans/ingest/multi_epoch.py` module docstring for the
full rationale and the recipe to switch to fully-correlated treatment
later if needed.

## Schema

| field | meaning |
|---|---|
| `bibcode` | NASA ADS bibcode; matches `_meta["source_paper_bibcode"]` for dispatch |
| `galaxy/ies` | LVDB key(s) the handler is dispatched for |
| `instruments` | spectrographs contributing velocities in the published table |
| `σ_sys (km/s)` | published per-epoch systematic floor (already absorbed into e_RVel) |
| `p_threshold` | χ² p-value below which a star is flagged variable (framework default 0.01) |
| `zero-point offsets` | per-instrument shifts; "none / applied / TODO unverified / TODO not applied" |
| `paper §refs` | section / table references that justify the choices |
| `URL` | ADS link |
| `status` | `verified` (deep-reviewer signed off) / `TODO` (open issue listed in Notes) |

## Table

| bibcode | galaxy/ies | instruments | σ_sys (km/s) | p_threshold | zero-point offsets | paper §refs | URL | status |
|---|---|---|---|---|---|---|---|---|
| `2017ApJ...838....8L` | `eridanus_2` | IMACS | 1.2 (Oct'15) / 1.0 (Nov'15) | n/a (single epoch) | none (single-instrument) | §3.1 | https://ui.adsabs.harvard.edu/abs/2017ApJ...838....8L | verified |
| `2018ApJ...857..145L` | `carina_2`, `carina_3` | IMACS + AAOmega/2dF + GIRAFFE+FLAMES | IMACS=1.0, AAT=0.5, VLT=0.9 (instrument-dependent) | 0.01 default; paper hand-flags 2 binaries + 2 RR Lyrae in Car II by Δv~25 km/s inspection | none (paper §3.1: "no significant zero-point shift") | §3.1, Table footnote (c) | https://ui.adsabs.harvard.edu/abs/2018ApJ...857..145L | verified |
| `2022ApJ...939...41C` | `grus_1` | IMACS (single-instrument; Walker+2016 M2FS NOT ingested) | 1.1 | 0.01 default; paper makes ad hoc per-star calls (p=0.01 strong, p=0.04 marginal) | none applied; v_IMACS−v_M2FS = −2.6±0.8 km/s in §3.4.1 is a cross-paper check vs external Walker+2016 M2FS, not within Table 2. Adapter stamps `Inst="IMACS"` for forward-compat. | §3.1, §3.4.1, §3.5, §4.1 | https://ui.adsabs.harvard.edu/abs/2022ApJ...939...41C | verified |
| `2023AJ....165...55C` | `tucana_2` | M2FS + IMACS + MIKE + MagE | 0.9 (new MIKE) / 1.2 (archival MIKE) | 0.01 default; paper uses Δv > 8 km/s rule of thumb | **applied here** (MIKE-ref): M2FS +2.5, IMACS +2.2, MagE +1.0 km/s. Byte-verified 2026-05-07: Table 6 velocities are raw per-instrument; stars.csv (Table 1) matches our MIKE-only IVW to ~0.1 km/s. | §3.1, Table 1 | https://ui.adsabs.harvard.edu/abs/2023AJ....165...55C | verified |
| `2020ApJ...892..137S` | `tucana_4` | IMACS | 1.0 (post-Nov'15) / 1.2 (pre) | 0.01; paper §3.4 χ² test with same threshold | none (single-instrument) | §3.1, §3.4 | https://ui.adsabs.harvard.edu/abs/2020ApJ...892..137S | verified |
| `2024ApJ...968...21H` | `tucana_5` | MIKE + IMACS | **not explicitly published** (Table 1 ±σ values may be template-fit stat-only) | 0.01 default; paper uses orbital fit (TheJoker) for Tuc V-1 binary | none published; §2.3 notes no MIKE−IMACS offset detected via Tuc V-2/3 agreement | §2, §2.3, §5.2.1, Table 1 | https://ui.adsabs.harvard.edu/abs/2024ApJ...968...21H | TODO (σ_sys not published; verify error model) |

## Open issues (QA-sweep #5)

1. **~~Chiti+2022 / Grus I zero-point.~~ RESOLVED 2026-05-07.** The
   IMACS−M2FS = −2.6 km/s shift in §3.4.1 is a **cross-paper**
   calibration check: Chiti+2022 IMACS vs *external* Walker+2016 M2FS.
   Walker+2016 M2FS data are not in Chiti+2022 Table 2 and are not
   ingested by our chiti2022 path-B adapter. Within our pipeline, the
   Grus I per-epoch table is single-instrument (IMACS-only across 3
   campaigns: 2015, 2019, 2021), so no within-table offset applies and
   no σ_los inflation from inter-instrument scatter is incurred.
   Adapter stamps every row `Inst="IMACS"` for provenance and
   forward-compat: should a future commit merge Walker+2016 into the
   Grus I per-epoch table, replace the `DEFAULT_POLICY` constructor in
   `combiners/chiti2022.py` with `CombinePolicy(sigma_sys_kms=0.0,
   p_threshold=0.01, zero_point_offsets_kms={"IMACS": +2.6, "M2FS":
   0.0})` (CombinePolicy is frozen, so reassign — don't mutate). The
   framework hook (issue #4, resolved) takes care of the rest.
2. **~~Chiti+2023 / Tuc II offset-application convention.~~ RESOLVED
   2026-05-07.** Byte-verify against the staged tucana_2.npz showed:
   (a) Empirical inter-instrument means: MIKE−M2FS = +1.5 (n=5),
   MIKE−IMACS = +1.6 (n=2), MIKE−MagE = +1.6 km/s after excluding
   the TucII-309 binary outlier (n=5; +2.8 with all 6) — all in the
   same direction as the paper's §3.1 quoted offsets (+2.5, +2.2,
   +1.0) and within 1σ small-N scatter. The TucII-309 exclusion
   matches the paper's §3.1 text. (b) `stars.csv` (the high-res
   5-star sample) matches MIKE-only IVW from `table6.csv` to within
   ~0.1 km/s, confirming Chiti+2023 do NOT pre-shift Table 6 onto a
   common zero-point. Conclusion: Table 6 carries raw per-instrument
   velocities. The chiti2023 handler now applies the §3.1 offsets
   (MIKE-ref) via `CombinePolicy.zero_point_offsets_kms`. Effect:
   individual v_bar shifts by up to +2.5 km/s, σ_los proxy across 19
   stars drops 4.02 → 3.88 km/s (~3% reduction).
3. **Hansen+2024 / Tuc V σ_sys.** Paper does not publish a per-epoch
   systematic floor. The Table 1 ±σ values look like χ²-template-fit
   stat errors only. Confirm whether the MIKE pipeline applies a
   systematic floor by default; if not, propose a value (likely 0.5–1.0
   km/s typical of MIKE) and re-test.
4. **~~Framework: zero-point hook.~~ RESOLVED 2026-05-07** (commit
   `62415e1`). `CombinePolicy.zero_point_offsets_kms` is now wired into
   `default.combine`: when non-empty, requires an `Inst` column on
   per_epoch and additively shifts V before the IVW; missing column or
   unknown instrument tags raise. Mutable-default
   `dict = None` replaced with a frozen `MappingProxyType` factory.
5. **Framework: σ_sys treatment.** The "treat as statistical"
   approximation is documented in `multi_epoch.py`. If a future
   Stage-1 result is sensitive to it, swap in the deconvolution path.
   Both options are wired; the swap is a 5-line edit in `default.combine`.

## Verification protocol (QA-sweep #5)

1. For each open issue above, perform the action listed.
2. For each handler, byte-verify the σ_sys / p_threshold / offset
   claims against the cited paper section.
3. Update `status` to `verified` only after a deep-reviewer pass.
4. Re-run the round-trip on every per-epoch catalog and diff
   `(n_input_rows, n_stars, n_variable, n_final)` and σ_combined
   distributions against the pre-sweep values.
