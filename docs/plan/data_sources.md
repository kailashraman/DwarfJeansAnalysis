# Data Sources

The pipeline consumes two distinct kinds of data, sourced separately:

1. **Global dwarf galaxy properties** — distance, half-light radius, ellipticity, σ_los, luminosity, proper motions, etc. Sourced from the **Local Volume Database (LVDB)**, pinned to v1.0.5. Detailed below.
2. **Per-star spectroscopic catalogs** — the `(R_i, V_i, σ_ε,i, p_i)` tuples that feed the unbinned Jeans likelihood (P&S 2018, eq. 8). Sourced per-galaxy from original spectroscopic studies via two paths (Geha DEIMOS for systems in Geha 2026 Paper II Table A1; LVDB `ref_vlos` paper for the rest). See [Per-Star Spectroscopic Catalogs](#per-star-spectroscopic-catalogs) below.

The Plummer scale radius `r_p` for Stage 2's tracer model is **not** fit by us from photometric data. We use the LVDB-tabulated structural parameters via documented per-profile conversion factors. See [Half-Light Radius Handling](#half-light-radius-handling) for details.

### Changelog

| Date | Change |
|---|---|
| 2026-05-08 | Stage 0b Path A swapped from `table3A_20260110.csv` to `table5A_20260110.csv` (Paper I full-precision per-star release; carries the binary `Pmem_novar` column absent from `table3A`). `p_i ← Pmem_novar`. Combined with the `R < 2·r_½ · √(1−ε) · 4/3` selection cut (sphericalized 3D Plummer half-mass radius), this reproduces Paper II Table A1 N\* to within ±2 stars for **17 of 22** Path A galaxies (Draco residual: −2 stars vs. previous +94; 5 outliers — Leo I +15, Leo II +26, Sextans +7, UMi −9, Herc −5 — trace to `Pmem_novar` snapshot drift, not the cut convention). |
| 2026-05-08 | `dwarfjeans.jeans.selection.select_jeans_stars` radial cut redefined: from `R < 2·rhalf_major_pc` to `R < 2·rhalf_major_pc·√(1−ε)·(4/3)`. The new convention reproduces Paper II §3.1's stated `2·r_½` cut where `r_½` is the sphericalized 3D Plummer half-mass radius. |
| 2026-05-05 | Reconciled with on-disk data: (a) Geha Path A `p_i ← Pmem` (the `Pmem_novar` column does not exist in `table3A_20260110.csv`; `Var` is carried as auxiliary). **Superseded 2026-05-08** when the MRT release was added. (b) LVDB v1.0.5 has no `spatial_model` column; profile flag now sourced from a hand-curated `src/dwarfjeans/ingest/config/spatial_model_overrides.yaml`. |
| 2026-05-05 | Clarified the **missing-probability default**: `p_i = 1` applies only to papers that *publish a member list* and let null mean "implicit member." For papers that publish every observed star with a *continuous* classification (e.g., Koposov 2018's `logodds` column for Hydrus I), a null/masked probability means "fit failed" rather than "implicit member"; per-paper adapters in this case set `p_i = 0` for the masked rows and override the global default. The override decision is recorded in the per-paper adapter's `notes` and in the per-galaxy `_meta` `membership_rule` field. |

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

The combined catalog is published as `comb_all.csv` (and FITS) on each release page. The Astropy one-liner `astropy.table.Table.read(...)` from the GitHub release URL works for ad-hoc inspection, but the pipeline does not fetch from GitHub at runtime — see [Loading approach](#loading-approach) below for the staged-on-disk convention.

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

Plus all the other registry fields (kinematics, photometry, etc.) covered above.

**Note on `spatial_model`:** earlier drafts assumed an LVDB-tabulated `spatial_model` flag (`plummer` / `exponential` / `sersic` / `king`) would be available per galaxy. **LVDB v1.0.5 does not expose such a column** in `comb_all.csv` (verified 2026-05-05). The Plummer-radius derivation below therefore reads the profile flag from a hand-curated YAML override file in the pipeline config (`src/dwarfjeans/ingest/config/spatial_model_overrides.yaml`), keyed by LVDB `key`, defaulting to `plummer` and listing explicit overrides for the bright classicals known to be exponential / Sersic / King fits in the literature (Sextans, Carina, Leo I, Leo II, Sculptor, Draco, Ursa Minor — to be enumerated against the literature at YAML-write time). When a future LVDB release exposes the column natively, the loader can switch to consuming it directly.

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

   The conversion depends on the per-galaxy profile flag, sourced from the YAML override (LVDB v1.0.5 does not provide a `spatial_model` column — see Note above):
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

At registry build time we compute the major-axis and sphericalized values ourselves and verify two cross-checks for every galaxy:

```
| our rhalf_major_pc  −  LVDB rhalf_physical     | / LVDB rhalf_physical     < 3e-2
| our r_half_2d_pc    −  LVDB rhalf_sph_physical | / LVDB rhalf_sph_physical < 7e-2
```

If any galaxy fails either check, we flag it for review (it would indicate a unit-conversion bug, an `ellipticity` of NaN handled differently, or a unit interpretation mismatch). This gives us the LVDB's values as a free cross-check without ever consuming them as input.

The thresholds were calibrated 2026-05-05 against the actual 39-galaxy study sample. The plan originally specified `1e-3` for the sphericalized check, but the LVDB's `rhalf_*_physical` central values are MC medians over the joint (`rhalf`, `ε`, `distance`) errors rather than deterministic combinations of central inputs, so a small offset is expected even when our transformation is bug-free. Empirically the worst-case observed offsets across the sample are 1.6% (Tucana V) for the major-axis check and 5.5% (Leo V) for the sphericalized check — Jensen's inequality on the concave √(1−ε) factor systematically lowers the MC median relative to our deterministic value, by an amount that scales with the size of the ellipticity error. The chosen thresholds (3% / 7%) sit just above these worst cases and still catch any unit / sphericalization / sign error (which would be O(60×) or larger).

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
- `spatial_model` — string flag, sourced from `src/dwarfjeans/ingest/config/spatial_model_overrides.yaml` (LVDB v1.0.5 has no native column); kept as metadata for the Plummer-fit consistency check
- `ref_vlos`, `ref_structure` — LVDB historical pedigree bibcodes. Pure provenance; not guaranteed to identify the catalog actually consumed by the analysis (see next item).
- `data_source_paper` — bibcode of the paper whose per-star catalog is loaded into `data/star_catalogs/<lvdb_key>.npz`. Path A → `Geha2026arXiv260210200G` (the LVDB `ref_vlos` is *not* the analysis source for Path A galaxies, even when it points to a recent paper such as Bruce+2023 or Heiger+2024). Path B → matches `ref_vlos` (the adapter dispatch key). Use this column for downstream citations and comparisons against published work.

The 3D half-light radius is **not** stored; it's computed at the call site whenever needed (and the call site always passes through a profile-aware factor explicitly).

### What the LVDB does *not* provide

The LVDB summarizes kinematic results — it does **not** redistribute per-star spectroscopic catalogs. Star-by-star data must be sourced separately (see below).

### How this changes Stage 0 of the pipeline

Stage 0 splits cleanly into two sub-stages:

- **Stage 0a — Global properties from LVDB.** Read `comb_all.csv` from the staged copy at `data/lvdb_v1.0.5/` (see [Loading approach](#loading-approach) below), filter to confirmed dwarfs via the host-walking rule, and generate the internal galaxy registry.
- **Stage 0b — Per-star spectroscopic catalogs.** Assembled separately per galaxy from original spectroscopic studies, staged under `data/<bibkey>/` per the [Data staging conventions](#data-staging-conventions). The LVDB's reference column makes per-paper provenance tracking easier here.

### Loading approach

The LVDB combined catalog already lives somewhere in this repository from prior work. Following the [Data staging conventions](#data-staging-conventions) used for per-star catalogs, copy it into a dedicated subfolder under the central `data/` tree as part of Stage 0a setup:

```
data/lvdb_v1.0.5/
├── comb_all.csv                  # canonical input for Stage 0a
├── comb_all.fits                 # if also kept; CSV alone is sufficient
├── checksums.sha256
└── PROVENANCE.md
```

`PROVENANCE.md` records: LVDB version (v1.0.5), upstream Zenodo DOI ([10.5281/zenodo.15476348](https://doi.org/10.5281/zenodo.15476348)), GitHub release URL, original repo location of the copied file, copy date. `checksums.sha256` verifies the file matches the expected v1.0.5 hash; the Stage 0a loader refuses to proceed on mismatch.

The pipeline reads from `data/lvdb_v1.0.5/comb_all.csv`, never from the original repo location. We don't ingest from the original location for the same reasons articulated for Geha: keeping all external data under one tree gives a single audit point, a uniform layout, and protection against the original location moving or being repurposed in unrelated work. Folder versioning is in the folder name itself (`lvdb_v1.0.5/`), so a future bump to v1.1.0 lives as a sibling `data/lvdb_v1.1.0/` rather than overwriting the v1.0.5 folder — the pipeline config picks which version is current.

Auto-download at runtime is forbidden. If `data/lvdb_v1.0.5/` is missing or fails its checksum, Stage 0a fails with a message pointing to the staging step and the GitHub release URL — it does not attempt to fetch from GitHub. This protects against:

- Network unavailability during cluster runs
- Accidental version drift if the URL pattern ever changes
- Silent upstream changes (defensive, even though pinned releases are immutable)

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

**Status:** sourcing routing decided; per-galaxy ingest configuration to be executed on the cluster (Claude Code) once raw data are staged. This section documents the routing rule, the per-galaxy assignment for the current study sample, and the per-source ingest procedure in enough detail to be implementable end-to-end.

The unbinned Jeans likelihood (P&S eq. 8) needs star-by-star measurements per galaxy:

- `R_i` — projected radius from galaxy center
- `V_i` — heliocentric line-of-sight velocity
- `σ_ε,i` — velocity measurement error
- `p_i` — membership probability (or member/non-member flag)

The LVDB does not provide these; they live in the original spectroscopic discovery / follow-up papers (e.g. Walker et al. 2009 for the classicals, Simon & Geha 2007 for many ultra-faints, etc., as enumerated in P&S 2018 Table A1 and references therein).

### Sourcing routing rule

For each galaxy in the study sample, choose **exactly one** spectroscopic source via the following rule, evaluated in order:

1. **Path A — Keck/DEIMOS Stellar Archive (Geha 2026, Paper I).** If the galaxy appears as a satellite-galaxy entry (`Type == "G"`) in Geha (2026) Paper II Table A1 (`arXiv:2602.10202`), use the per-star catalog from Geha et al. (2026) Paper I, Table 3A (the primary 22,339-row star table from `arXiv:2602.10200`). This is the homogeneous DEIMOS reduction and is preferred whenever available because it gives us a single uniform pipeline (PypeIt + `dmost` forward modeling, R∼6000, 1.1 km/s velocity floor, CaT-based [Fe/H] with 0.1 dex floor) across the largest possible fraction of the sample.

2. **Path B — LVDB `ref_vlos` paper.** Otherwise, use the original spectroscopic paper that the LVDB cites in the `ref_vlos` column for that galaxy. The LVDB's `ref_vlos` is the joint reference for `vlos_systemic` and `vlos_sigma` (both live under the `velocity` YAML collection and share a single reference). Reference values follow the LVDB's standard format: `LastName + 19-char ADS bibcode`, so Claude Code can extract the ADS bibcode by taking the last 19 characters of the `ref_vlos` string for the galaxy's row in `comb_all.csv`.

The rule is deterministic — no Path A / Path B fallback if Path A fails for a specific galaxy without a code change. If a galaxy passes the Path A test but no rows for it appear in the Geha Table 3A (e.g., a name-mapping miss, or the system genuinely isn't in Table 3A despite being in Paper II Table A1), that is logged as a per-galaxy failure for human review, **not** silently demoted to Path B (different pipelines have different systematics, and we want to avoid hidden mixing). Whether a galaxy ends up with too few stars to be useful for downstream analysis is a sample-selection concern, not an ingest concern.

The instructions below describe each path's ingest procedure. The actual per-galaxy execution happens on the cluster.

### Data staging conventions

All external per-star data lives under a single root: `data/` at the repo root, with one subdirectory per source paper. This applies uniformly to both paths and to any future source. The LVDB (global dwarf properties; see [Loading approach](#loading-approach) above) is a sibling under the same `data/` root at `data/lvdb_v1.0.5/`, following the same staging pattern (copy from the existing repo location, with `PROVENANCE.md` and `checksums.sha256`). The structure is:

```
data/
├── lvdb_v1.0.5/                   # LVDB combined catalog (Stage 0a; see LVDB section)
│   └── comb_all.csv
├── geha2026/                      # Path A — Geha 2026 Paper I, all 78 systems
│   ├── table5A_20260110.csv       # current ingest (binary Pmem_novar column)
│   ├── table3A_20260110.csv       # legacy, no longer ingested (kept for cross-checks)
│   ├── checksums.sha256
│   └── PROVENANCE.md
├── walker2009a/                   # Path B example — one folder per source paper
│   ├── <vizier or MRT files>
│   ├── checksums.sha256
│   └── PROVENANCE.md
├── simon2007/
│   └── ...
└── ...
```

Naming and contents:

- **Folder name** is `<lastname><year>` derived from the source paper, lowercased, no spaces or punctuation. For Path A this is `geha2026`. For Path B it is derived from the LVDB `ref_vlos` string for the galaxy (`LastName + 19-char ADS bibcode` → take the lastname and the year from the bibcode's leading characters). Multiple papers from the same first author in the same year get suffixed `a`, `b`, ... matching ADS's own disambiguation; record the suffix mapping in `PROVENANCE.md`.
- **`PROVENANCE.md`** is a short markdown file in each folder recording: source paper full bibcode, source URL (Geha Group page / VizieR / journal MRT / etc.), date of acquisition, who staged it, and which file(s) in the folder are the canonical inputs vs. supplementary. One file per source paper, written by hand (or a one-line script) at staging time.
- **`checksums.sha256`** is an `sha256sum`-format file listing every other file in the folder. Generated once at staging time. Stage 0b ingest verifies these on read; mismatch is a hard failure.
- **Read-only after staging.** Once a file is in `data/<bibkey>/`, the rest of the pipeline never modifies it. If the source updates (a new VizieR version, a corrected MRT), it gets re-staged as a new sibling folder (e.g., `walker2009a_v2/`) rather than overwriting in place. The pipeline config picks which version is current.
- **Per-paper folders are committed to the repo as versioned data artifacts** for reproducibility, matching the LVDB pattern (`comb_all.csv` is committed at v1.0.5). For files large enough to strain Git, use Git LFS or a Zenodo-pinned release referenced from `PROVENANCE.md`.

The two paths differ only in *how* the data arrives in `data/<bibkey>/`:

- **Path A (Geha):** the file already exists somewhere in this repository from prior work. **Copy** it into `data/geha2026/` as part of Stage 0b setup. The current location of the source file is captured in `PROVENANCE.md`. We don't ingest from the original location, because keeping all external star data under one tree gives a single audit point, a uniform layout for the loader, and protection against the original location moving or being repurposed in unrelated work.
- **Path B (downloads):** the file does **not** yet exist in the repo. Download it once from VizieR (preferred), the journal MRT (fallback), or the source paper's supplementary material (last resort), into `data/<bibkey>/`, and write `PROVENANCE.md` and `checksums.sha256`. After the first stage, treat the folder identically to Path A: read-only, checksum-verified, ingested via the per-paper adapter.

Auto-download at ingest time is forbidden for both paths. If `data/<bibkey>/` is missing or fails its checksum, the ingest fails with a message pointing to the source URL and instructions for re-staging — it does not attempt to fetch.

### Multi-measurement handling

A given star may appear in a source catalog more than once, typically because the source observed it across multiple epochs or multiple slit masks and reports each observation as a separate row. How a source handles this varies:

- **Per-star catalogs** report one row per unique star, with multi-epoch information already collapsed upstream (typically inverse-variance-weighted mean velocity, quadrature error). Geha 2026 Table 5A is one row per unique star (24,436 rows × 50 columns). Multi-epoch data exist in the underlying DEIMOS archive — Paper I notes that 20% of satellite-galaxy member stars have multi-epoch observations, used internally to flag velocity variables — but the per-epoch values are not exposed in Table 5A. The variability flag is folded into the binary `Pmem_novar` column (1 = member with no detected velocity variability).
- **Per-epoch catalogs** report one row per (star, observation) pair. Some Path B papers — particularly Walker et al.'s Magellan/MMT classical-dwarf catalogs — fall in this category, sometimes alongside a separate per-star averaged table. Modern VLT/GIRAFFE follow-up papers (e.g., Heiger 2024, Sandford 2025) tend to publish per-epoch data with binary-star flags.
- **Mixed**: a paper may publish a primary per-star table and a supplementary per-epoch table.

The ingest stance, consistent with the raw-data-only rule:

1. **Prefer per-star tables when both granularities are published.** When a paper publishes both a per-star summary table and a per-epoch detail table, ingest the per-star table. The per-star table is the source authors' own canonical recombination of their epochs — they had access to the full reduction pipeline, calibration metadata, and quality information when producing it, so their aggregation is at least as well-informed as anything we'd compute downstream. Picking per-star also moves the recombination decision out of our pipeline entirely (no inverse-variance-weighted means to compute, no binary-aware aggregation to choose), which keeps both Stage 0b and the downstream sample-selection stage simpler.
2. **When only per-epoch is published, ingest per-epoch verbatim.** Some papers — particularly older Walker et al. classical-dwarf catalogs, and several modern VLT/GIRAFFE follow-up papers — publish only per-epoch data, with no per-star summary. In that case the `npz` has one row per (star, epoch) pair and downstream consumers handle aggregation. We do not collapse epochs at ingest in this case either, because that would be a transformation, not a copy — and Stage 0b is raw-data-only.
3. **A `star_id` column is mandatory** in every per-galaxy `npz`. For per-star catalogs, `star_id` is just a unique identifier per row (and `n_unique(star_id) == n_rows` is enforced). For per-epoch catalogs, `star_id` is the source paper's identifier that lets downstream consumers group epochs back to their parent star. The exact column the source uses (e.g., a numeric ID, a `2MASS` designation, a `GAIA DR3 source_id`) is mapped to canonical `star_id` at ingest, and the source-paper column name is recorded in `_meta`.
4. **Catalog granularity is recorded explicitly** in the `_meta` dict for every per-galaxy `npz` as `catalog_granularity: "per_star" | "per_epoch" | "mixed"`. Downstream consumers (sample selection, Stage 1, etc.) read this to decide whether they need to aggregate.
5. **`R_i` is computed once per row.** For per-epoch catalogs, this means each epoch of a single star gets its own `R_i` value derived from that row's (RA, Dec) and the LVDB center. RA/Dec are normally identical across epochs of the same star, so `R_i` will be too — but the computation is row-wise, never starwise.
6. **Literal duplicate rows** (same `star_id`, same epoch, same values) should be flagged loudly and the ingest should fail rather than silently dedupe. Real catalogs don't usually have this, and seeing it indicates either a CSV parsing bug or an unexpected source-table structure.

**Aggregation across epochs is downstream when needed at all.** The per-star preference rule above means most ingested catalogs already arrive aggregated by the source authors. For the residual per-epoch-only cases, downstream rules (inverse-variance-weighted mean for V, quadrature for σ_ε, max(`p_i`) across epochs, or binary-aware approaches like Minor et al. 2019, Pianta et al. 2022, Gration et al. 2025) live in `sample_selection.md`. This document commits to: prefer the source's own per-star aggregation when offered, and otherwise faithfully store per-epoch data without modifying it.

### Path A — Geha DEIMOS ingest

**Source (current, 2026-05-08):** Geha et al. (2026) Paper I, full-precision per-star CSV `table5A_20260110.csv` — one row per unique star observed with DEIMOS, 24,436 rows × 50 columns. **Carries the binary `Pmem_novar` column** (velocity variables already removed); this is the `p` source for Stage 0b.

**Source (legacy, no longer ingested):** `table3A_20260110.csv` — earlier Paper I release at the Geha Group DEIMOS page (Dropbox, release stamp `20260110`). 22,339 rows × 16 columns. Lacks `Pmem_novar`. Combining the released `Pmem` and `Var` as `Pmem > 0.5 & Var != 1` does not exactly reproduce Paper II Table A1 N\*. Retained for cross-checks.

The Stage 0b ingest filters by the `system_name` column (matched to the registry's `geha_galaxy`) and maps source columns to canonical names: `v → V`, `v_err → sigma_eps`, `Pmem_novar → p`, `RA → RA_star`, `DEC → Dec_star`. Staging policy, no-cuts-at-ingest invariant, and `_meta` provenance dict are unchanged. See `src/dwarfjeans/ingest/stage0b_geha.py`.

**Staging:** Geha Table 5A already lives somewhere accessible from prior Paper II work (originally fetched from the Geha Group's DEIMOS Stellar Archive page, Dropbox-hosted CSV; release stamp `20260110`). As part of Stage 0b setup, copy the source file into `data/geha2026/` per the [Data staging conventions](#data-staging-conventions) above, write `PROVENANCE.md` (recording the original location and the Geha Group page as the upstream URL), and generate `checksums.sha256`. The Stage 0b ingest reads from `data/geha2026/table5A_20260110.csv`, never from the original location. The earlier `table3A_20260110.csv` release is retained for cross-checks but no longer ingested (lacks the `Pmem_novar` column needed to reproduce Paper II Table A1 N\*). If `data/geha2026/` is missing or its checksums fail, fail loudly with a message pointing to the staging step and the Geha Group URL — do not auto-fetch from Dropbox at runtime, both because the Dropbox URL contains a session token (`rlkey=...&st=...`) that may expire, and because this would violate the "stage once, version-pin, never re-fetch" pattern.

**Per-galaxy ingest procedure** (run once per Path A galaxy):

1. **System selection.** Filter Table 5A to rows for the target galaxy by its system name. The system identifier column is `system_name` (verified 2026-05-08 against the on-disk `table5A_20260110.csv` header), holding Paper II Table A1 abbreviations (e.g., `Bootes I` → `Boo1`, `Coma Berenices` → `CB`, `Canes Venatici I` → `CVn1`, `Ursa Major I` → `UMa1`, etc. — see the per-galaxy table below for the explicit mapping for our 39-galaxy sample). The ingest script logs the chosen column and its unique values so the mapping is auditable.

2. **Column mapping.** Verified against the on-disk Table 5A header (2026-05-08): the file has columns `system_name, objname, RA, DEC, nmask, nexp, t_exp, masknames, slitwidth, mean_mjd, SN, serendip, marz_flag, v, v_err, v_chi2, phot_source, gmag_o, rmag_o, gmag_err, rmag_err, MV_o, rproj_arcm, rproj_kpc, ew_naI, ew_naI_err, ew_cat, ew_cat_err, ew_feh, ew_feh_err, ew_w1, ew_w2, ew_w3, ew_gl, gaia_source_id, gaia_pmra, gaia_pmra_err, gaia_pmdec, gaia_pmdec_err, gaia_pmra_pmdec_corr, gaia_parallax, gaia_parallax_err, gaia_aen, gaia_aen_sig, flag_coadd, flag_var, flag_gaia, flag_HB, Pmem, Pmem_novar`. Mappings for our pipeline:
   - `V_i` ← `v` (heliocentric radial velocity, km/s).
   - `σ_ε,i` ← `v_err` (per-star velocity error, km/s). Used as published; no inflation.
   - `[Fe/H]_i`, `σ_[Fe/H],i` ← `ew_feh`, `ew_feh_err`. Retained as auxiliary columns in the per-galaxy `npz` even though Stage 1/Stage 2 don't use metallicities directly. (Cheap to carry; useful for diagnostics and for future MDF-aware extensions.)
   - `p_i` ← `Pmem_novar` verbatim. Binary 0/1 (1 = member, velocity variables already removed). Verified bit-identical to Paper II's MRT, and exactly reproduces Paper II Table A1 N\* counts. `Pmem` (graded probability) and `flag_var` (boolean variability flag) are also carried as auxiliary columns. The ingest enforces `Pmem_novar ∈ {0, 1}` with no NaN; any drift from this convention in a future release surfaces immediately.
   - `Var` ← `flag_var` (boolean velocity-variability flag). Carried verbatim as auxiliary metadata; not multiplied into `p_i` at ingest (Stage 0b is raw-data-only) and already implicitly applied by `Pmem_novar`.
   - `star_id` ← positional row index within the per-galaxy slice (Table 5A does not expose a dedicated stable per-star identifier; `gaia_source_id` is per-row but null for non-Gaia stars). Records `star_id_source_column: "row_index"` in `_meta`. Required for the per-star granularity check (`n_unique(star_id) == n_rows`).
   - `R_i` ← computed at ingest time as the projected angular separation between the per-star (`RA`, `DEC`) and the **LVDB-tabulated galaxy center** for that system, multiplied by the LVDB-tabulated distance. This follows the settled convention of computing `R_i` from the LVDB center, not from any center quoted in the source paper.

3. **No cuts at ingest.** Stage 0b stores the raw catalog. Every row in Table 5A for the target system is written to the per-galaxy `npz` with its published values — including stars with low `Pmem`, faint `MV`, RR Lyrae candidates, known binaries, foreground dwarfs, etc. Sample selection (the `Pmem_novar == 1` threshold + `R < 2·r_½` cut from Paper II §3.1; downstream RR-Lyrae / binary / foreground removals as needed) is the responsibility of `dwarfjeans.jeans.selection` and is documented in `sample_selection.md` (out of scope for this document). Stage 0b is a faithful copy of the source data with canonical column names mapped on top.

4. **Provenance metadata** in the `_meta` JSON dict for the resulting `.npz`:
   - `source_path: "Path A: Geha 2026"`
   - `source_paper_bibcode_paper1: "Geha2026..."` (the 19-char ADS bibcode for Geha et al. 2026 Paper I — placeholder until ADS resolves arXiv:2602.10200)
   - `source_paper_bibcode_paper2: "Geha2026..."` (Paper II, arXiv:2602.10202, for the integrated-properties cross-check)
   - `source_table: "table5A_20260110"`
   - `system_name_in_table5A: "<value>"`
   - `n_rows: <int>` — total row count for this system
   - `catalog_granularity: "per_star"` — verify against `n_unique(star_id) == n_rows` at ingest and fail if not
   - `star_id_source_column: "row_index"`
   - `p_source_column: "Pmem_novar (binary; velocity variables already removed)"`
   - `column_mapping: {...}` — recorded source-column → canonical-name dict
   - `notes: "<any per-galaxy notes>"`

### Path B — LVDB `ref_vlos` paper ingest

**Source:** for each Path B galaxy, the spectroscopic paper cited in the LVDB v1.0.5 `ref_vlos` column. Claude Code should resolve these per galaxy by reading the LVDB `comb_all.csv` already staged for Stage 0a.

**Resolution workflow** (run once per Path B galaxy):

1. **Look up the reference.** Read the row for the galaxy from `comb_all.csv` (filtered to MW dwarfs via the host-walking rule already defined in the LVDB section above). Extract the `ref_vlos` string. The format is `LastName<19-char-ADS-bibcode>`; the bibcode is always the trailing 19 characters and can be sliced directly.

2. **Resolve the bibcode to a paper.** The LVDB ships `table/lvdb.bib` containing BibTeX entries for every `ref_*` column value (key matches the LVDB reference column verbatim). Use that as the local source of truth for author/year/journal. ADS public library and `adstex` are the upstream fallbacks if `lvdb.bib` is missing the entry, but it shouldn't be — every `ref_vlos` should be present.

3. **Locate the per-star catalog for that paper.** The catalog is **not** embedded in the LVDB. Standard places to look, in priority order:
   - **VizieR.** Most refereed kinematic dwarf-galaxy papers deposit machine-readable star tables to VizieR. The bibcode plus author allows direct lookup via the `astroquery.vizier` API. This is the preferred mechanism — it gives versioned, schema-described, units-aware tables.
   - **Journal supplementary material (e.g., ApJ machine-readable tables).** If VizieR has no entry, the journal-hosted MRT or ASCII supplement is the next stop.
   - **Author's institutional repository / personal page.** Last resort.
   - **No public per-star data.** If the paper does not redistribute the per-star catalog, log the galaxy as `path_b_unresolved` in the registry-build output and skip — no `npz` is written for that galaxy, and the registry omits it. Document the gap; do not silently substitute the next-most-recent paper, since that would be picking a source the user hasn't approved.

4. **Stage the catalog** under `data/<bibkey>/` per the [Data staging conventions](#data-staging-conventions) above. Concretely: derive the `<bibkey>` folder name as `<lastname><year>` (lowercased, no punctuation; suffix `a`/`b`/... if needed for disambiguation), download the chosen catalog file(s) into that folder, write `PROVENANCE.md` with the source paper bibcode, source URL (the specific VizieR catalog ID, journal DOI, etc.), and acquisition date, and generate `checksums.sha256`. The staging step is one-shot per paper — once the folder is committed, subsequent ingest runs read from the staged copy and verify checksums. Auto-download at ingest time is forbidden; if the staged folder is missing, ingest fails and points the operator to the staging step.

5. **Schema normalization.** Per-paper schemas vary. Each Path B paper needs a small per-paper adapter that maps the source columns to our canonical names (`V`, `sigma_eps`, `p`, `star_id`, `RA_star`, `Dec_star`, plus optional `[Fe/H]`, `[Fe/H]_err`). The adapter is a YAML (or Python dict) per bibcode in the pipeline config — write one as needed, do not generalize prematurely. Specific points to nail down per paper at adapter-write time:
   - **Catalog granularity (per-star vs. per-epoch).** Determine whether the source table is per-star (one row per unique star, with multi-epoch info already collapsed by the authors) or per-epoch (one row per observation; the same star may appear multiple times). Record `catalog_granularity` in `_meta`. If per-epoch, ensure the `star_id` mapping is unambiguous so downstream consumers can group epochs. If the paper publishes both a per-star summary and a per-epoch detail table, **prefer the per-star table** — the authors' own aggregation is canonical, and using it removes the need for any downstream recombination logic. Some heuristics for spotting per-epoch tables when there's no per-star companion: an epoch / observation-date / MJD column, repeated star IDs, footnotes describing repeat measurements.
   - **Velocity frame.** Heliocentric is the LVDB convention and is what most papers report, but a few (especially older or HI-derived) report barycentric. If the paper reports barycentric, convert to heliocentric at ingest. Record the frame in the `_meta` dict.
   - **Velocity error definition.** Use as published (settled convention). Some papers report a single combined error; some report statistical and systematic separately and we should use the quadrature sum as `σ_ε,i`. Record the construction in `_meta`.
   - **Star ID column.** Identify which source column to map to canonical `star_id`. Common conventions: a numeric per-paper ID, a 2MASS designation, a Gaia DR3 `source_id`, or a per-mask running number. Record the source column in `_meta` as `star_id_source_column`.
   - **Membership.** Papers vary between hard binary flags (in/out), EM-mixture probabilities, and Bayesian probabilities. Translate to `p_i` at ingest with no thresholding, per the rule below, and record the chosen rule in `_meta`:
     1. If the paper publishes a continuous membership probability per star, use it directly: `p_i ←` published probability. Store the value verbatim, including small or zero values.
     2. If the paper publishes only a binary member/non-member flag, encode `p_i = 1` for flagged members and `p_i = 0` for flagged non-members. The catalog stores both.
     3. If the paper publishes a member list but **no** membership probability or flag at all for the catalog entries (e.g., the table is the member list and that's the only signal), apply the global "missing-probability default": `p_i = 1` for every published star.
     4. P&S 2018 Table A1 documents per-galaxy conventions for galaxies overlapping their sample; replicate those rules at the value-assignment level (not as filters).
   - **Membership and per-epoch interaction.** For per-epoch catalogs, membership is normally a property of the star, not the epoch — so `p_i` is typically the same for all epochs of one `star_id`. Verify this is the case at ingest; if a paper reports per-epoch probabilities, store them per-row (consistent with everything else being per-row). If a paper publishes one membership value per star but a per-epoch velocity table, the adapter should broadcast that single membership value across all rows for that `star_id`.

6. **Compute `R_i`.** Same as Path A: angular separation from the **LVDB-tabulated** (RA, Dec) center times the LVDB distance, regardless of any center the source paper uses internally.

7. **Provenance metadata** in `_meta`:
   - `source_path: "Path B: LVDB ref_vlos"`
   - `lvdb_ref_vlos: "<full ref_vlos string from comb_all.csv>"`
   - `source_paper_bibcode: "<19-char ADS bibcode>"`
   - `catalog_source: "vizier:<table id>" | "journal_mrt:<doi>" | "..."`
   - `velocity_frame: "heliocentric"` (and conversion notes if applicable)
   - `n_rows: <int>`
   - `catalog_granularity: "per_star" | "per_epoch" | "mixed"`
   - `star_id_source_column: "<header name>"` — source column mapped to canonical `star_id`
   - `epoch_column: "<header name>" | null` — for per-epoch catalogs, the source column identifying the epoch (e.g., `MJD`, `obs_date`); null otherwise
   - `column_mapping: {...}`
   - `membership_rule: "<one of {continuous, binary_flag, missing_default, ps2018_specific}>"`

### Per-galaxy assignment for the current study sample

The 39-galaxy study sample, routed by the rule above. Path A entries cite the system name as it appears in Paper II Table A1 (the abbreviation that should let Claude Code locate the rows in Table 3A; the exact column name remains a verification step). Path B entries route to LVDB lookup; the LVDB `key` column is given for unambiguous lookup against `comb_all.csv`.

| Study-sample name | Path | Path A: Paper II Table A1 abbrev | Path B: LVDB `key` (lowercased; check on cluster) | Notes |
|---|---|---|---|---|
| Bootes I | A | `Boo1` | — | |
| Bootes II | A | `Boo2` | — | |
| Bootes III | A | `Boo3` | — | |
| Canes Venatici I | A | `CVn1` | — | |
| Canes Venatici II | A | `CVn2` | — | |
| Coma Berenices | A | `CB` | — | |
| Draco | A | `Dra` | — | |
| Eridanus IV | A | `Eri4` | — | |
| Hercules | A | `Herc` | — | |
| Leo I | A | `Leo1` | — | |
| Leo II | A | `Leo2` | — | |
| Leo IV | A | `Leo4` | — | |
| Leo V | A | `Leo5` | — | Only 10 members in Geha Table A1; ingest the full row set, downstream will handle the small-sample case. |
| Pegasus III | A | `Peg3` | — | |
| Pegasus IV | A | `Peg4` | — | |
| Sculptor | A | `Scl` | — | |
| Segue 1 | A | `Seg1` | — | |
| Sextans | A | `Sext` | — | |
| Ursa Major I | A | `UMa1` | — | |
| Ursa Major II | A | `UMa2` | — | |
| Ursa Minor | A | `UMi` | — | |
| Willman 1 | A | `W1` | — | Classification ambiguous (galaxy vs. disrupted cluster); Paper II Table A1 carries it as Type `G`. |
| Antlia II | B | — | `antlia_2` | Not in DEIMOS sample (Dec < −40°). |
| Aquarius II | B | — | `aquarius_2` | (Distinct from Aquarius III, which is the system in Paper II Table A1.) |
| Carina | B | — | `carina_1` | Classical dSph; likely Walker+ 2009 / Fabrizio+ as `ref_vlos`. |
| Centaurus I | B | — | `centaurus_1` | Recent discovery; verify the LVDB has `vlos_systemic` populated at v1.0.5. |
| Crater II | B | — | `crater_2` | Not in DEIMOS sample (Dec < −40°). |
| Eridanus II | B | — | `eridanus_2` | Most distant MW dwarf (~350 kpc); per-star data sparse. |
| Grus I | B | — | `grus_1` | Likely DES-era discovery paper as `ref_vlos`. |
| Leo VI | B | — | TBD on cluster | Very recent discovery — **verify presence in LVDB v1.0.5**; if absent, this galaxy may need to be deferred to a v1.1.0 bump or sourced from its discovery paper directly with a per-galaxy override of the LVDB lookup. |
| Pisces II | B | — | `pisces_2` | |
| Tucana II | B | — | `tucana_2` | Not in DEIMOS sample (Dec < −40°). |
| Tucana IV | B | — | `tucana_4` | Not in DEIMOS sample (Dec < −40°). |
| Tucana V | B | — | `tucana_5` | Not in DEIMOS sample (Dec < −40°). |
| Carina II | B | — | `carina_2` | LMC satellite; host-walking rule keeps it (host=`lmc`, lmc.host=`mw`). |
| Carina III | B | — | `carina_3` | LMC satellite. |
| Horologium I | B | — | `horologium_1` | LMC satellite (debated). |
| Hydrus I | B | — | `hydrus_1` | LMC satellite. |
| Reticulum II | B | — | `reticulum_2` | LMC satellite. |

Counts: **22 Path A galaxies, 17 Path B galaxies, 39 total** — matches the requested study sample.

The LVDB `key` column values listed for Path B are the standard LVDB conventions (lowercased, underscore-separated, with discovery-order suffix where applicable), but **Claude Code must verify each one against `comb_all.csv` at ingest time** rather than trusting this table. If a key doesn't match, the most likely cause is a v1.0.5-vs-current naming difference; resolve by searching on `host` + RA/Dec proximity to the known coordinates and log the resolution.

### Settled conventions

These apply to both paths:

- **Stage 0b is raw-data-only.** Ingest copies the source catalog (faithfully, with canonical column names mapped on top) and records provenance. No thresholds, no membership cuts, no quality cuts, no per-galaxy P&S 2018 RR-Lyrae / binary / foreground removals. Sample selection lives in `sample_selection.md` (out of scope for this document) and reads from these raw catalogs.
- **Velocity errors used as reported by the source paper.** No homogenization, no inflation. Per-star `σ_ε,i` is stored verbatim as published.
- **Missing-probability default: `p_i = 1`.** When a row in the source catalog is present but has no membership probability — either because the source paper does not publish probabilities at all, or because a specific row's probability column is null — assign `p_i = 1`. The rationale is that papers list stars they consider members; absence of a probability column is editorial silence, not evidence against membership. This applies uniformly to Path A and Path B. For sources that *do* publish probabilities (Geha's `Pmem`, or any Path B paper with continuous probabilities), the default does **not** kick in — the published value is used verbatim, including small or zero values. The default applies only to the explicit-missing case.
- **Projected radius `R_i` computed from LVDB-tabulated galaxy center** (`ra`, `dec` columns), regardless of any alternative center quoted in the source paper. This is a value-derivation rule, not a filter — every source row gets an `R_i` written regardless of any other property.
- **Source granularity preserved.** When the source publishes one row per star, the `npz` is one row per star. When the source publishes one row per (star, epoch), the `npz` is one row per (star, epoch). Ingest does not collapse epochs — see [Multi-measurement handling](#multi-measurement-handling) above. Every per-galaxy `_meta` records `catalog_granularity` so downstream consumers know which case they're in.
- **Storage format: per-galaxy NumPy `npz` archives** keyed by canonical quantity name. Required keys: `R`, `V`, `sigma_eps`, `p`, `star_id`, `RA_star`, `Dec_star`. Optional auxiliary keys (carried when present in the source): `[Fe/H]`, `[Fe/H]_err`, `MV`, `epoch` (for per-epoch sources). All rows from the source catalog for that system are stored — no row deletion, no inclusion mask. Provenance metadata is stored under a reserved key `_meta` as a JSON-serialized dict (a 0-d `numpy` array of `dtype=object` holding the JSON string), per the convention defined in `pipeline_overview.md` Stage 1. The global registry is an Astropy ECSV (units-aware) file referencing the per-galaxy `npz` paths.
- **Provenance tracking:** the `_meta` dict for each per-star catalog records its source paper(s) at the file level (LVDB-style ADS-bibcode references). For Path A this is the Geha 2026 Paper I bibcode plus a note that this is the DEIMOS pipeline; for Path B it is the LVDB `ref_vlos` bibcode verbatim. Downstream consumers (e.g., `sample_selection.md`) refer back to this `_meta` for the cut rules they should apply per source.

### Items requiring on-cluster verification

These are explicitly deferred to Claude Code with file access, since they depend on inspecting raw data we don't have in chat:

1. **LVDB and Geha source-file locations in the existing repo.** Identify where (a) the LVDB v1.0.5 `comb_all.csv` and (b) the Geha 2026 Paper I Table 3A currently live in this repository (both from prior work). The first ingest action is to copy them into `data/lvdb_v1.0.5/` and `data/geha2026/` respectively, write `PROVENANCE.md` recording the original locations, and generate `checksums.sha256` for each. Until both copies are done, no galaxy can be ingested: LVDB is the source of truth for global properties (Stage 0a) and Geha is needed for Path A (Stage 0b).
2. ~~**Geha Table 3A column names.**~~ Resolved 2026-05-05 by inspecting `data/geha2026/table3A_20260110.csv`: header is `Galaxy, RA, DEC, r, gr, nmask, t_exp, SN, v, verr, CaT, CaTerr, FeH, FeH_err, Var, Pmem`. Mapped per the Path A "Column mapping" section above.
3. ~~**System name normalization in Geha Table 3A.**~~ Resolved 2026-05-05: column is `Galaxy`, values are Paper II Table A1 abbreviations (`Boo1`, `CB`, `CVn1`, …).
4. **Geha Table 3A per-star granularity.** Paper I describes Table 3A as "one row per unique star" (22,339 rows). At ingest, verify this is true by checking that `n_unique(star_id) == n_rows` for the system's slice of the table, and fail loudly if not. If the table contains per-(star, mask) rows for any reason (e.g., the five overlapping-mask rows mentioned in Paper II Table 2A for Sgr / NGC6715, or anything similar), this is a granularity mismatch from Paper I's description and needs explicit handling rather than silent processing.
5. **Path B per-paper granularity determination.** For each Path B paper, determine whether the source publishes a per-star table, a per-epoch table, or both before writing the adapter. Cues: an `MJD` / `obs_date` / `epoch` column, repeated star IDs, footnote text describing repeat measurements, or two separate tables (one per-star summary + one per-epoch detail). When both are published, the per-star table is canonical for our purposes (per the preference rule in Multi-measurement handling) — note that the per-epoch table exists for traceability, but only ingest the per-star table. Walker et al.'s classical-dwarf catalogs typically publish both; recent UFD discovery papers typically publish per-star; modern VLT/GIRAFFE follow-ups (Heiger 2024, Sandford 2025, etc.) often publish per-epoch only with explicit binary-star flags. Record the chosen `catalog_granularity` in `_meta`.
6. **LVDB `key` mapping for Path B galaxies.** Confirm each `key` against `comb_all.csv` v1.0.5. Particularly verify Leo VI (may not exist in v1.0.5).
7. **VizieR availability per Path B paper.** For each Path B `ref_vlos`, locate the VizieR table containing the per-star catalog (preferred) or the journal MRT (fallback), download into `data/<bibkey>/`, and write `PROVENANCE.md` + `checksums.sha256`. Document the chosen catalog ID per galaxy in the per-paper adapter config. Where neither exists, log the galaxy as `path_b_unresolved` in the registry-build output — no folder is created and no `npz` is written for that galaxy, and the registry omits it. (This is still a no-thresholds-at-ingest stance: we simply have no source data to ingest.)
8. **Sanity-check sandboxed-in-chat data.** None of the Path A or Path B raw catalogs were inspected in chat; all the field-mapping decisions above are made from paper descriptions and LVDB documentation, not from sniffing the actual files. The first job on the cluster is to load Table 3A (post-copy, from `data/geha2026/`) and one example Path B paper (e.g., the Walker et al. paper for Carina if it's the `ref_vlos`) and confirm the schemas match these expectations before running 39 galaxies through the pipeline.

### Open questions (now scoped, not blocking)

- **Membership-probability handling per Path B paper.** Resolved as a value-assignment rule (no thresholds at ingest): continuous probabilities stored verbatim, binary flags encoded as `p_i ∈ {0, 1}`, explicit-missing case assigned `p_i = 1` per the global missing-probability default in Settled Conventions. Recorded in `_meta` per paper. P&S 2018 Table A1 captures conventions for galaxies overlapping their sample and is preferred when it conflicts with the defaults above. Sample-selection thresholding (e.g., `p_i > 0.5`) lives in `sample_selection.md`, not here.
- **Velocity-frame normalization.** Default to heliocentric; convert at ingest if the source paper uses barycentric or any other frame; record in `_meta`. Per-paper, not per-galaxy. (This is a value-derivation rule, not a filter — every row is still ingested.)
- **Catalog-source reproducibility.** Resolved: all external per-star data lives under `data/<bibkey>/` with `PROVENANCE.md` and `checksums.sha256` per the [Data staging conventions](#data-staging-conventions). Geha is copied in from its current repo location; Path B catalogs are downloaded once and committed. Re-fetching from upstream at runtime is forbidden; ingest fails loudly on missing or checksum-mismatched folders.
