# Data Sources

The pipeline consumes two distinct kinds of data, sourced separately:

1. **Global dwarf galaxy properties** — distance, half-light radius, ellipticity, σ_los, luminosity, proper motions, etc. Sourced from the **Local Volume Database (LVDB)**, pinned to v1.0.5. Detailed below.
2. **Per-star spectroscopic catalogs** — the `(R_i, V_i, σ_ε,i, p_i)` tuples that feed the unbinned Jeans likelihood (P&S 2018, eq. 8). Sourced per-galaxy from original spectroscopic studies. Plan TBD — see [Per-Star Spectroscopic Catalogs](#per-star-spectroscopic-catalogs) below.

The Plummer scale radius `r_p` for Stage 2's tracer model is **not** fit by us from photometric data. We use the LVDB-tabulated structural parameters via documented per-profile conversion factors. See [Half-Light Radius Handling](#half-light-radius-handling) for details.

---

## Local Volume Database (LVDB)

**Status:** decided. Use LVDB v1.0.5 as the source of global / population-level dwarf properties.

The LVDB is a community catalog of observed properties of dwarf galaxies and star clusters in the Local Group and Local Volume, maintained by Andrew Pace (same author as Pace & Strigari 2018 — convenient for methodology consistency).

- **Repository:** [github.com/apace7/local_volume_database](https://github.com/apace7/local_volume_database)
- **Overview paper:** [Pace 2025, OJAp 8, 142](https://doi.org/10.33232/001c.144859) ([arXiv:2411.07424](https://arxiv.org/abs/2411.07424))
- **Pinned version:** v1.0.5 (released 2025-05-20)
- **Zenodo DOI:** [10.5281/zenodo.15476348](https://doi.org/10.5281/zenodo.15476348)
- **License:** CC0 (public domain)
- **Completeness at v1.0.5:** dwarf galaxies complete to ~6 Mpc relative to HyperLeda

### Why v1.0.5 specifically

Pinned for reproducibility. The current latest release is **v1.1.0** (2026-03-21), two minor versions ahead of our pin. The relevant changes since v1.0.5 are:

- **v1.0.6** renamed the combined-catalog column `confirmed_dwarf` → `confirmed_galaxy`. The YAML key remained `confirmed_dwarf` upstream; only the combined CSV column was renamed. At our pinned v1.0.5 we still see the older `confirmed_dwarf` name in `comb_all.csv`, which is what the ingest filter targets.
- **v1.1.0** ships an official ECSV export with units, and stops releasing the per-host sub-tables (only `comb_all` is released going forward). Consumers must filter `comb_all` by the `table` column to recover the host subsets.

We accept this two-version lag in exchange for a stable, citable data baseline. Version bumps are planned as deliberate, versioned changes, not passive "always grab latest." When we eventually bump (likely to v1.1.0 or later), the upstream ECSV-with-units export can replace our hand-rolled CSV → ECSV registry plumbing for the raw LVDB columns.

### What the LVDB provides

Per-dwarf global properties that map directly onto P&S 2018's Jeans-modeling inputs and the split-normal nuisance priors of [`uncertainty_conventions.md`](./uncertainty_conventions.md):

- **Position:** RA, Dec (with errors)
- **Distance:** `d` and error
- **Structural:** half-light radius and ellipticity (with errors). See [Half-Light Radius Handling](#half-light-radius-handling) for which columns we consume and how we sphericalize.
- **Photometric:** absolute magnitude `M_V`, V-band luminosity `L_V`
- **Kinematics:** systemic velocity, line-of-sight velocity dispersion `σ_los` and error, proper motions `(μ_α cosδ, μ_δ)` where measured
- **Classification flags:** `confirmed_dwarf` (filter on this), host galaxy association (LVDB lists MW / M31 / Local Field / external; we restrict to MW satellites — see Filtering rules below)
- **Provenance:** reference papers for each measurement, which is useful for tracing kinematic data back to its spectroscopic source

The combined catalog is published as `comb_all.csv` (and FITS) on each release page. Loading is a one-liner via `astropy.table.Table.read` from the GitHub release URL.

### Half-Light Radius Handling

**The LVDB carries 2D (projected) half-light radii only — there is no 3D half-light column.** That is the right starting point: P&S 2018 §3 also works with 2D quantities throughout, and we sphericalize using ellipticity exactly as they do.

The LVDB columns relevant here:

| Column | Meaning | Units |
|---|---|---|
| `rhalf` | 2D **major-axis** half-light radius (raw input) | arcmin |
| `rhalf_physical` | 2D major-axis in physical units (`rhalf × distance`) | parsec |
| `rhalf_sph_physical` | 2D **azimuthally-averaged** half-light radius: `rhalf × distance × √(1−ε)` | parsec |
| `ellipticity` | `1 − b/a` | unitless |

All three columns have associated `_em` / `_ep` Monte-Carlo error columns derived from the joint distance, `rhalf`, and ellipticity errors.

**Decision: ingest the raw `rhalf` and apply transformations ourselves.**

We deliberately do **not** read `rhalf_sph_physical` (or `rhalf_physical`) directly. Instead we ingest the raw inputs and perform every conversion ourselves, keeping all intermediate values in the registry. The reasons:

- **Provenance.** Every number in the registry has a transformation we wrote and can audit. We don't inherit any silent assumptions baked into the LVDB's Monte-Carlo procedure.
- **Error propagation under our control.** The LVDB's `_em`/`_ep` for derived columns come from a Monte-Carlo over distance, `rhalf`, and ellipticity errors. We may want to use Gaussian priors on the input quantities directly (matching P&S 2018's approach where these are nuisance parameters in the sampler), in which case we never want to consume a pre-marginalized derived error.
- **Consistency with sampler design.** Distance, half-light radius, and ellipticity are split-normal nuisance priors in the Stage 2 Jeans likelihood (see [`uncertainty_conventions.md`](./uncertainty_conventions.md)). We need their raw values and errors as separate quantities so the sampler can vary them independently. A pre-combined `rhalf_sph_physical` would have to be re-decomposed.
- **Future flexibility.** If we later want a different sphericalization (e.g., arithmetic mean of axes, or no sphericalization for an axisymmetric Jeans extension), we already have the raw inputs in hand.

### Columns we ingest from the LVDB

The raw inputs we read into the registry (each with its `_em` / `_ep` errors):

| LVDB column | Symbol | Units | Notes |
|---|---|---|---|
| `rhalf` | `R_half,maj^(ang)` | arcmin | 2D major-axis half-light radius, angular |
| `ellipticity` | `ε` | unitless | `1 − b/a` |
| `distance_modulus` | `μ` | mag | converted to `d` ourselves |
| `position_angle` | `PA` | deg | kept for diagnostics; not used in spherical Jeans |
| `spatial_model` | — | string | `plummer`, `exponential`, `sersic`, etc. — flag if not Plummer |

Plus all the other registry fields (kinematics, photometry, etc.) covered above.

### Transformations we apply

Computed at registry-build time and stored alongside the raw inputs:

1. **Distance from distance modulus:**
   `d [kpc] = 10^((μ − 10) / 5)`

   Equivalent to the standard `d [pc] = 10^((μ + 5) / 5)` rewritten in kpc.
2. **Major-axis half-light radius in physical units:**
   `R_half,maj [pc] = rhalf [arcmin] × (π/180/60) × d [kpc] × 1000`
3. **Azimuthally-averaged (sphericalized) half-light radius — this is P&S 2018's `r_1/2`:**
   `r_1/2 [pc] = R_half,maj × √(1 − ε)`

   This is the geometric mean of major and minor axes, matching P&S §3 and the analogous LVDB derived column. It is the projected (2D) radius; we keep it 2D throughout the Jeans likelihood.

4. **3D half-light radius — used only at the analytic-J-factor / Wolf-mass-estimator step (see Stage 3 / P&S Appendix A):**
   `r_{3D,1/2} [pc] ≈ (4/3) × r_1/2`

   This factor is profile-dependent; the value above is the Plummer convention. Computed on the fly at the point of use, not stored as a registry field, so the conversion is explicit at every call site.

5. **Plummer scale radius — for use in P&S eqs. 5–6:**

   The Plummer model (P&S eq. 5) requires a single scale radius `r_p`. We do **not** refit the photometric profile ourselves — we derive `r_p` from the LVDB structural parameters using documented per-profile conversion factors. This matches P&S 2018's approach exactly (their §3 footnote 4) and avoids a separate per-galaxy photometric ingestion.

   The conversion depends on the LVDB's `spatial_model` flag for each galaxy:
   - `spatial_model == "plummer"` (most ultra-faints and several classicals): `r_p = r_1/2` directly. The Plummer scale radius equals the projected half-light radius by definition.
   - `spatial_model == "exponential"` (some bright dSphs): `r_p = 1.68 × r_exponential`, where `r_exponential` is the exponential-fit scale radius. This is the Plummer-equivalent half-light radius for an exponential profile (P&S 2018 §3 footnote 4).
   - `spatial_model == "sersic"`: **non-P&S extension.** P&S 2018 footnote 4 only documents the Plummer ↔ exponential conversion; we extend by setting `r_p = r_1/2`, treating the Sersic-fit half-light radius as the equivalent Plummer scale. The approximation accuracy depends on the Sersic index and the radial range of the data, but for our purposes it's adequate (the Plummer approximation is itself an approximation to the true tracer profile, and these galaxies tend to be the bright classicals where any adequate scale-length proxy gives consistent J-factors). Logged as a non-P&S extension in the Differences section of `pipeline_overview.md`.
   - `spatial_model == "king"`: rare for our sample; same non-P&S extension as Sersic, convert via `r_1/2`.

   For galaxies in the non-Plummer cases, the conversion factor is applied to the LVDB-tabulated quantity in its native units, then propagated to physical units alongside the rest of the structural transformations. The conversion factor is logged in the registry-build output for any non-Plummer system so the assumption is visible.

   `r_p` is stored as a separate registry field with its (asymmetric) errors propagated from the underlying LVDB column.

   **Note:** for the dominant Plummer case, our derived `r_1/2` from §3 above (computed as `R_half,maj × √(1−ε)`) and `r_p` are the same number. Both are kept in the registry under separate names so the role of each (Plummer scale parameter vs. azimuthally-averaged P&S `r_1/2`) is clear at every call site.

   **Tracer-profile assumption.** Stage 2's Jeans projection assumes the *full* Plummer 3D density (P&S 2018 eq. 5) for every galaxy regardless of the LVDB's `spatial_model` flag. The flag only affects how `r_p` is derived from the photometric scale radius — not which tracer profile enters the projection integral. So a galaxy with `spatial_model == "sersic"` is modeled as a Plummer tracer with `r_p` set to the Sersic half-light radius. This matches P&S 2018 §3, which assumes Plummer throughout and notes that for ultra-faint dwarfs Plummer is adequate. For the bright classicals where the photometric profile is genuinely closer to Sersic or exponential, this is an additional approximation; we accept it for consistency with P&S.

### Error propagation

- We read `rhalf_em` / `rhalf_ep`, `ellipticity_em` / `ellipticity_ep`, and `distance_modulus_em` / `distance_modulus_ep` from the LVDB and store them raw — asymmetric, in native units. No symmetrization, no log-space conversion, no derived single-σ estimate.
- These feed into the Stage 2 sampler as **split-normal priors** on the corresponding nuisance parameters (`d` derived from μ, `rhalf`, `ε`). See [`uncertainty_conventions.md`](./uncertainty_conventions.md) for the full convention.
- We do **not** pre-combine them into a derived error on `r_1/2`. The sampler keeps `rhalf` and `ε` factorized; `r_1/2` is computed on the fly per draw.
- For diagnostic plotting and registry-summary output, derived quantities like `r_1/2` get Monte-Carlo-propagated 16/84-percentile bounds, stored as separate clearly-labeled fields. Useful for sanity-checking against `rhalf_sph_physical_em`/`_ep` from the LVDB, but never used in the likelihood.

### Sanity check against the LVDB derived columns

At registry build time we compute `r_1/2` ourselves and verify that for all galaxies:

```
| our r_1/2  −  LVDB rhalf_sph_physical | / LVDB rhalf_sph_physical < 1e-3
```

If any galaxy fails this check, we flag it for review (it would indicate a unit-conversion bug, an `ellipticity` of NaN handled differently, or a unit interpretation mismatch). This gives us the LVDB's value as a free cross-check without ever consuming it as input.

### Summary of registry fields for half-light geometry

For each galaxy the registry stores, with provenance:

- `rhalf_arcmin` (raw, with errors) — direct from LVDB
- `ellipticity` (raw, with errors) — direct from LVDB
- `distance_modulus` (raw, with errors) — direct from LVDB
- `distance_kpc` (with errors) — derived
- `rhalf_major_pc` (with errors) — derived
- `r_half_2d_pc` — our `r_1/2`, derived as `rhalf_major × √(1−ε)`; this is what enters the Jeans likelihood as P&S `r_1/2`
- `r_half_2d_pc_err` — derived diagnostic only, not used in the likelihood
- `plummer_radius_pc` — `r_p`, derived from LVDB structural parameters via per-profile conversion (`r_p = r_1/2` for Plummer fits, `r_p = 1.68 × r_exponential` for exponential fits, etc. — see Half-Light Radius Handling)
- `spatial_model` — string flag, propagated from the LVDB (kept as metadata for the Plummer-fit consistency check)

The 3D half-light radius is **not** stored; it's computed at the call site whenever needed (and the call site always passes through a profile-aware factor explicitly).

### What the LVDB does *not* provide

The LVDB summarizes kinematic results — it does **not** redistribute per-star spectroscopic catalogs. Star-by-star data must be sourced separately (see below).

### How this changes Stage 0 of the pipeline

Stage 0 splits cleanly into two sub-stages:

- **Stage 0a — Global properties from LVDB.** Download `comb_all.csv` at v1.0.5 once, store it as a versioned data artifact in the repo, filter to confirmed dwarfs, generate the internal galaxy registry.
- **Stage 0b — Per-star spectroscopic catalogs.** Assembled separately per galaxy from original spectroscopic studies. The LVDB's reference column makes provenance tracking easier here.

### Loading approach

For reproducibility on the cluster: download the v1.0.5 `comb_all.csv` (and/or the dwarf-galaxy-only subtable) **once** and commit it as a versioned data artifact in the repo, rather than re-fetching from GitHub at runtime. This protects against:

- Network unavailability during cluster runs
- Accidental version drift if the URL pattern ever changes
- Silent upstream changes (defensive, even though pinned releases are immutable)

A small loader script verifies the file's checksum on read and refuses to proceed if it doesn't match the expected v1.0.5 hash.

### Filtering rules at ingest

When building the registry from `comb_all.csv`:

- Keep only `confirmed_dwarf == True` (excludes ambiguous Milky Way systems and globular clusters)
- **Restrict to systems whose host (transitively) is the Milky Way.** The LVDB's `host` column lists the immediate parent (e.g., `mw`, `m31`, `lmc`, `smc`, `cena`, etc.). Some MW dwarf satellites are listed as hosted by an MW subhalo (notably the LMC and SMC), e.g. recent DES/DELVE discoveries that orbit the LMC rather than the MW directly. We resolve this by walking the host chain: keep a system iff the top of its host chain is `mw`, where `mw → mw` is the root. So `host == 'mw'` is kept directly; `host ∈ {'lmc', 'smc'}` is kept because both have `host == 'mw'`; M31 satellites and Local Field dwarfs are dropped. This implementation handles arbitrary depth and any future MW subhalos automatically.
- Drop systems lacking the minimum required fields: `rhalf`, `distance_modulus`, `apparent_magnitude_v` (i.e. `M_V` after derivation). Ellipticity is handled per the missing-data rule (uniform fallback prior). `vlos_systemic` and `vlos_sigma` are **not** required: a brand-new spectroscopic discovery without a tabulated systemic velocity can still be processed if the per-galaxy override `vlos_prior_halfwidth` is widened in config and the prior center is taken from the source paper. `vlos_sigma` is purely metadata for cross-checking against the Stage 1 fit and never enters any likelihood.
- Flag systems with unresolved σ_los (these still get processed in Stage 2 but with the expanded ρ_s prior; see `pipeline_overview.md` Stage 1)

The host-walking rule is more robust than enumerating allowed host strings — if a future LVDB version introduces a new MW satellite-of-satellite (e.g., a system orbiting Sagittarius), it gets included automatically without a code change. We log the resolved host chain for each accepted galaxy in the registry-build output.

### Citation requirements

Add to the methodology references list:

- **Pace 2025** (LVDB overview paper) — cite alongside Pace & Strigari 2018
- **LVDB Zenodo DOI** for v1.0.5 — cite as a software/data product

---

## Per-Star Spectroscopic Catalogs

**Status:** plan TBD — to be fleshed out in this section.

The unbinned Jeans likelihood (P&S eq. 8) needs star-by-star measurements per galaxy:

- `R_i` — projected radius from galaxy center
- `V_i` — heliocentric line-of-sight velocity
- `σ_ε,i` — velocity measurement error
- `p_i` — membership probability (or member/non-member flag)

The LVDB does not provide these; they live in the original spectroscopic discovery / follow-up papers (e.g. Walker et al. 2009 for the classicals, Simon & Geha 2007 for many ultra-faints, etc., as enumerated in P&S 2018 Table A1 and references therein).

**Settled conventions:**

- **Per-galaxy modifications from P&S 2018 happen at Stage 0b ingest, not in Stage 1 or Stage 2.** This includes RR Lyrae removal (CVn I, Coma Ber, Leo IV, UMa I), binary-star removal (Hercules), foreground-dwarf removal (UMa II), and any analogous per-galaxy cuts. Doing this at ingest keeps Stage 1 and Stage 2 operating on identical samples (no "Stage 1 saw 18 stars but Stage 2 saw 17" failure mode), and concentrates membership configuration in one place.
- **Removed-star provenance is preserved.** Rather than physically deleting rows, ingestion stores the full pre-cut star list plus a per-star **inclusion mask** (boolean) and a **reason code** (string: `"member"`, `"rr_lyrae"`, `"binary"`, `"foreground"`, `"low_membership_prob"`, etc.) for each row. Stage 1 and Stage 2 read only the masked-in rows, but the full list with reasons is queryable for diagnostics, sensitivity tests, and reproducing alternative-cut analyses.
- **Velocity errors used as reported by the source paper.** No homogenization, no inflation. Per-star `σ_ε,i` enters Stage 1 and Stage 2 likelihoods directly as published.
- **Projected radius `R_i` computed from LVDB-tabulated galaxy center** (`ra`, `dec` columns), regardless of any alternative center quoted in the source paper.
- **Storage format: per-galaxy NumPy `npz` archives** keyed by quantity name (`R`, `V`, `sigma_eps`, `p`, plus inclusion mask and reason codes). Provenance metadata is stored under a reserved key `_meta` as a JSON-serialized dict (a 0-d `numpy` array of `dtype=object` holding the JSON string), per the convention defined in `pipeline_overview.md` Stage 1. The global registry is an Astropy ECSV (units-aware) file referencing the per-galaxy `npz` paths.
- **Provenance tracking:** the `_meta` dict for each per-star catalog records its source paper(s) at the file level (LVDB-style ADS-bibcode references, since the source papers usually overlap with the LVDB's `ref_vlos` column).

**Open questions to resolve before drafting the full ingestion plan:**

- Sources: VizieR catalog mirrors? Direct paper supplements? Existing private compilations?
- Schema normalization beyond what's already settled: differing column conventions, frame conventions (heliocentric vs. solar-system-barycenter), and any other systematic differences across papers that we haven't encountered yet.
- Membership probability handling: papers vary between binary member/non-member flags, EM-method probabilities, and Bayesian probabilities. P&S 2018 applies different rules per source; we need to enumerate the rule per galaxy in the config.

This section will be expanded in a follow-up working session.
