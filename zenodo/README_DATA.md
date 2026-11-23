# Data for "Semi-analytic Inference of Satellite Densities in the Cold Dark Matter Model": DwarfJeansAnalysis production posteriors

Raman, Folsom, Kaplinghat, Lisanti & Safdi.
Part I: arXiv:2607.27316. Part II: arXiv:2607.27326.
Code: https://github.com/kailashraman/DwarfJeansAnalysis

This record holds the spherical-Jeans production posteriors that the
DwarfJeansAnalysis repository produced for the two papers: 39 Milky Way dwarf
spheroidal galaxies, each fitted under 8 prior configurations, 312 runs in
total. The SatelliteDensityInference repository
(https://github.com/kailashraman/SatelliteDensityInference) reads them to build
the Jeans-analysis comparisons in Part II.

The record has three files: `dja_production.tar.gz`, `MANIFEST.json` and this
README. The archive stores paths relative to the repository root and unpacks at
the DwarfJeansAnalysis repository root, under `results/production/`.
`MANIFEST.json` gives the archive's size, md5 and sha256, the sha256 of every
member, and `dja_git_commit`, the commit of the repository at packing time (the
packing commit, not the code that produced the runs; see "Provenance gaps").
Check it out after cloning for an exact match with the tools and documents.

## What is in the archive

All counts below were derived from the packed tree.

- 39 dwarfs (`<lvdb_key>`, the Local Volume Database key, e.g. `draco_1`,
  `ursa_major_2`) times 8 priors = 312 run directories
  `results/production/<lvdb_key>/<prior>/`, with 5 files each: 1560 files,
  965,784,994 bytes (0.97 GB) before compression.
- The 8 priors, as directory names: `jeffreys`, `loguniform`, `satgen`,
  `satgen_box`, `satgen_shmr_danieli23_const`, `satgen_shmr_fattahi18`,
  `satgen_shmr_kim24`, `satgen_shmr_moster18`. The code also accepts a
  `uniform` prior; no `uniform` run exists in this record.
- Not in the archive: `results/production/README.md` (tracked in git), and
  `data/` (the input catalogs, in git; see "Third-party data").

Every run was made with `rseed=0`, `nlive=1500`, `dlogz=0.05`, a 9-parameter
model (four halo and velocity parameters, three nuisance parameters and two
proper-motion parameters), and the pool size `npool` recorded in its `run.log`:
`npool=1` for 232 runs, `npool=8` for 80. The 80 are the launcher's `classical`
cohort (`scripts/submit_batch.sh --cohort classical`, galaxies with more than
100 stars after the selection cuts) under all eight priors: antlia_2,
canes_venatici_1, carina_1, crater_2, draco_1, leo_1, leo_2, sculptor_1,
sextans_1, ursa_minor_1. The deposited `audit.json` files do not record
`npool`; only `run.log` does (runs made with the current code record it in
`audit.json` as well).

## File formats

Checked across all 312 runs: the `.npz` files all have the same keys, dtypes
and trailing dimensions (below), and differ only in the number of equal-weight
samples N (`n_eq`; minimum 7936, median 13101, maximum 55202). `audit.json`
has the same top-level keys in every run, plus `shmr` in the 156 `satgen_shmr_*`
runs only.

### `posterior_samples.npz`

One key, `samples_eq`: float64, shape (N, 9), the equal-weight posterior
sample. The deposited file holds nothing else: no weights, log-likelihoods,
parameter names or run metadata (the code's `save_chain` writes more; the
deposited files were reduced to `samples_eq`, see "Provenance gaps"). Column
order, from `postprocess.PARAM_NAMES_9`:

| column | name | meaning |
|---|---|---|
| 0 | `V` | systemic velocity (km/s) |
| 1 | `log10_rs` | log10 NFW scale radius (kpc) |
| 2 | `log10_rhos` | log10 NFW scale density (Msun/kpc^3) |
| 3 | `beta_tilde` | transformed constant velocity anisotropy |
| 4 | `d_kpc` | distance (kpc) |
| 5 | `eps` | ellipticity |
| 6 | `rhalf_arcmin` | half-light radius (arcmin) |
| 7 | `pmra` | proper motion in RA (mas/yr) |
| 8 | `pmdec` | proper motion in Dec (mas/yr) |

The names are read from the code, not stored in the file; units follow the
column names in `summary.csv`.

### `derived.npz`

Per-draw derived quantities, regenerated from `samples_eq` by
`scripts/reprocess.py` (`postprocess.save_derived`). 38 float64/int64 1-D
arrays, in five length classes:

- Length N (the full chain): `V`, `log10_rs`, `log10_rhos`, `beta_tilde`,
  `beta`, `d_kpc_chain`, `eps_chain`, `rhalf_arcmin_chain`, `r_p_chain`,
  `R_half_2d_chain`, `r_half_3d_chain`, `alpha_c_chain`, `log10_M_half_2d`,
  `log10_M_half_3d`, `pmra`, `pmdec`.
- Length 500 (the J/D thinning; `idx_jd` maps back to chain rows): `r_t_kpc`,
  `R_GC_kpc`, `theta95_J_deg`, `theta95_D_deg`, `log10_J_pi2`, `log10_D_pi2`,
  `log10_J_0p1deg`, `log10_J_0p2deg`, `log10_J_0p5deg`, `log10_J_alphac`,
  `log10_D_0p1deg`, `log10_D_0p2deg`, `log10_D_0p5deg`,
  `log10_D_alphacover2`, and the index array `idx_jd` (int64).
- Length 2000: `sigma_los_at_Rhalf2d` and its index array `idx_sig` (int64).
- Length 30 (the sigma_los profile): `R_grid_sigma_profile`,
  `sigma_profile_q16`, `sigma_profile_q50`, `sigma_profile_q84`.
- Length 300: `idx_prof` (int64).

`r_t_kpc` is a per-draw tidal radius (Tormen/Springel convention). Which host
potential it used is not recorded in any deposited file; the code default,
`jd.tidal.DEFAULT_HOST` (the Gala MilkyWayPotential2022), is what
`reprocess.py` uses for a chain that records no host. The units of the J and D arrays are
those of the `summary.csv` rows with the same names (GeV^2 cm^-5 for J, GeV
cm^-2 for D, log10); `theta95_*_deg` are in degrees. The thinning sizes (2000,
500, 300) are in each run's `audit.json` under `thinning`.

### `summary.csv`

32 lines per run: the header `quantity,q16,q50,q84` and 31 rows, each with the
16th, 50th and 84th percentiles. The rows, in order: `V_sys_kms`,
`log10_rs_kpc`, `log10_rhos_Msun_kpc3`, `beta`, `d_kpc`, `eps`,
`rhalf_arcmin`, `r_p_kpc`, `R_half_2d_kpc`, `r_half_3d_kpc`,
`log10_M_half_2d_Msun`, `log10_M_half_3d_Msun`, `sigma_los_at_Rhalf2d_kms`,
`sigma_los_walker_kms`, `alpha_c_rad`, `R_GC_kpc`, `r_t_kpc`, `theta95_J_deg`,
`theta95_D_deg`, `pmra_mas_yr`, `pmdec_mas_yr`, `log10_J_{0p1,0p2,0p5}deg`,
`log10_J_alphac`, `log10_J_pi2` (each `_GeV2_cm5`) and
`log10_D_{0p1,0p2,0p5}deg`, `log10_D_alphacover2`, `log10_D_pi2` (each
`_GeV_cm2`).

All rows are computed from `samples_eq` except `sigma_los_walker_kms`, a
separate radius-independent constant-sigma fit to the star velocities (with a
Jeffreys prior for the `satgen`-family runs and the run's own prior
otherwise). `alpha_c` is `2 r_1/2,3D / d`; `log10_J_alphac` is J within
`alpha_c`, `log10_D_alphacover2` is D within `alpha_c / 2`, and the `pi2` rows
are the totals within the tidal radius (exact geometry, theta = pi/2).
`theta95_*` are separate containment angles.

### `audit.json`

Top-level keys: `lvdb_key`, `prior_name`, `timestamp_utc`, `registry_row`,
`selection_policy`, `nuisance_priors`, `prepare_jeans_input_audit`, `dynesty`,
`thinning`, `r_t_kpc`; the 156 `satgen_shmr_*` runs also have `shmr`. The
`dynesty` block holds `nlive`, `dlogz`, `rseed`, `logZ`, `logZ_err`, `n_eq` and
`wallclock_s`. It records no git commit, no dirty flag, no package versions,
no input hashes and no `npool`. Its `r_t_kpc` (1.0 in all 312 runs) is
superseded: it is the fixed tidal radius of the May runs and is not what
`derived.npz` and `summary.csv` use (see "Provenance gaps"). `registry_row` and
`nuisance_priors` repeat catalog values from the Local Volume Database and the
literature (see "Third-party data").

### `run.log`

The stdout of the run: the registry row, selection policy, the constant-sigma
fit, the proper-motion marginalisation, the dynesty header line (`dynesty (9D,
prior=..., nlive=..., dlogz=..., npool=...)`), `logZ`, `n_eq` and timings.
Three things to know: its first lines record an absolute output path on the
authors' cluster (an earlier checkout location); for the `satgen_shmr_*` runs
the dynesty header says `prior=satgen_shmr` without the SHMR name (the SHMR is
in `audit.json`'s `shmr`); and its final "J/D integrals" block, written with a
fixed `r_t=1.0 kpc`, is superseded in the same way as `audit.json`'s `r_t_kpc`.

## How to fetch

```bash
git clone https://github.com/kailashraman/DwarfJeansAnalysis
cd DwarfJeansAnalysis
git checkout <dja_git_commit from MANIFEST.json>             # optional, exact match
conda env create -f environment.yml && conda activate DwarfJeans
python scripts/zenodo_fetch.py --record <RECORD_ID>
```

`environment.yml` is the tested route; `pip install -e .` also works but was
not what the record was checked with. Options (`python scripts/zenodo_fetch.py
--help`): `--dest REPO_ROOT` (where to unpack; default the repository containing
the script), `--tmp DIR` (where the archive is staged; default under `--dest`,
so leave room for the archive there), `--sandbox` (sandbox.zenodo.org),
`--files-from DIR` (read `MANIFEST.json` and the archive from a local
directory, with no network and only the manifest checks), `--force`.

What the fetch checks: the Zenodo md5 of `MANIFEST.json` and the archive, and
the archive's sha256 and md5 against the manifest. Before extracting, it
checks that the archive's member names equal the manifest's member set, that
each is a regular file under `results/production/`, and that none has an
absolute or `..` path. After extracting each file it verifies that file's
sha256 against the manifest. It refuses to overwrite a differing existing file
unless given `--force`, and an existing directory or symlink at a target path
aborts the fetch even with `--force`. Extraction is not transactional: if it
fails part-way (a corrupt archive, a checksum mismatch, a full disk), the files
extracted before the failure stay in place; fix the cause and rerun with
`--force`.

It needs `requests` (in `environment.yml`) and a Python with
`tarfile.data_filter` (3.12 or later, or a 3.8-3.11 patch release from
mid-2023 onward; `environment.yml`'s 3.11.15 has it). The fetch recreates
`results/production/<lvdb_key>/<prior>/`, the layout the code and
SatelliteDensityInference expect.

## What reproduces, and how far

Checked, with the scope of each check:

- **`derived.npz` and `summary.csv` from the deposited chains.**
  `scripts/reprocess.py` regenerated both bit for bit from the deposited
  `samples_eq` for 4 runs: draco_1/satgen_shmr_moster18, pisces_2/satgen,
  segue_1/jeffreys and ursa_major_2/loguniform. The check ran at commit
  6acd829 plus changes later committed as 29c705c and c449a5b, which do not
  touch the derived-product path. The other 308 runs were not compared.
  `reprocess.py` rewrites in place; use `--base` on a copy (see the
  repository `README.md`). It needs `data/` (star catalogs and registry, in
  git) and each run's `audit.json` (seed, thinning, selection policy).
- **Rerunning the sampler (`npool=1`).** Two runs were rerun from scratch with
  `scripts/run_production.py` at `rseed=0`, `npool=1`: pisces_2/satgen and
  pisces_2/loguniform. `logZ`, `logZ_err` and `n_eq` matched the deposited
  values exactly, and for pisces_2/satgen the deposited chain was shown to be
  a valid systematic resample of the rerun's weighted points. These reruns
  predate the seeding change (commit 29c705c), so they show that the sampler
  reproduces, and that the deposited `samples_eq` is not bit-identical because
  its resample was unseeded.
- **Equal-weight resample, going forward.** Since 29c705c the resample is
  seeded from `rseed`, so a new run is expected to reproduce its own
  `samples_eq` given the same `rseed` and `npool`. This was tested on a toy
  likelihood for `npool=1` and `npool=2`
  (`tests/unit/test_inference_determinism.py`); no seeded rerun of a real
  galaxy is recorded in this README yet. The deposited chains were made
  before this change, so they reproduce only statistically.
- **Pooled runs (`npool` > 1).** The 80 `npool=8` runs were not rerun. A run is
  expected to be deterministic given `rseed` and `npool` (dynesty 3.0.0
  derives its per-queue seeds from the sampler's generator), but this was not
  verified on real data. A different `npool` gives a different chain (the
  shift in posterior medians was not quantified here), so reproduce a
  deposited run with the `npool` in its `run.log`.

Not checked: any run other than those named above; any run on a different
dynesty or numpy version than the one in `environment.yml`; any sampling rerun
for the `jeffreys`, `loguniform` (other than pisces_2), `satgen_box` or
`satgen_shmr_*` priors, or for any `npool=8` run.

## Provenance gaps

- **Dates and code.** `audit.json`'s `timestamp_utc` values span 2026-05-14 to
  2026-05-19. The deposited `audit.json` and `run.log` record no git commit,
  no dirty flag, no package versions and no input hashes. The code and inputs
  of the sampling runs are not pinned; the repository history and
  `environment.yml` are the closest record, and `MANIFEST.json`'s
  `dja_git_commit` pins the code at packing, not at the runs.
- **Superseded tidal radius.** The May runs used a fixed `r_t = 1.0 kpc`, which
  is what the deposited `audit.json` `r_t_kpc` and `run.log`'s "J/D integrals"
  block record. `derived.npz` and `summary.csv` were regenerated on 2026-07-09
  by `scripts/reprocess.py`, from an uncommitted working tree shortly before
  commit 6301f87, and use a per-draw tidal radius in the
  MilkyWayPotential2022 host. No deposited file records the host. The J and D
  values in `derived.npz` and `summary.csv` therefore differ from the ones the
  `run.log` J/D block prints, and the `audit.json` / `run.log` values should
  not be used.
- **Reduced chains.** On the same date `posterior_samples.npz` was reduced to
  `samples_eq` only, by a step that is not in git history. The sampler's own
  weighted output (weights, log-likelihoods, nested-sampling points) is not in
  this record. The 2026-07-09 dates rest on file modification times, which the
  archive does not preserve (members carry mtime 0, and the fetch stamps
  extraction time). The consistency of `derived.npz` with the chain does hold
  in all 312 runs: `derived.npz`'s `V` equals `samples_eq[:, 0]` and
  `audit.json`'s `n_eq` equals N.
- **Prior tables.** The satgen-family prior tables the runs used
  (`data/satgen_prior/`, 157 files: 1 global table and 4 SHMRs times 39
  dwarfs, tracked in git) were rebuilt from SatelliteDensityInference's
  deposited `inputs` tier with scratch copies of
  `scripts/build_satgen_prior_table.py` and
  `scripts/build_satgen_shmr_prior_tables.py` (paths from environment
  variables; the tracked versions with `--h5` and `--weights-dir` differ only
  in argument plumbing). Every numeric array of all 157 files was identical.
  The recorded metadata differs: the tables record the original SatGen_Dwarf
  paths and an `h5` sha256 beginning `4b9026`, while the deposited SatelliteDensityInference file
  has sha256 beginning `a9880a` (all 67 datasets in it are identical). The
  record does not prove these are the exact tables the SHMR runs read: the
  `fattahi18` runs began 2026-05-16, and the `moster18`, `danieli23_const` and
  `kim24` runs began before the commit that wired them (30675a7) and before
  their tables were committed. For pisces_2/satgen the identical `logZ` from
  the rerun confirms the global `satgen` table. The builders write into
  `data/satgen_prior/` in place (no output flag), and the SHMR builder must be
  run once per `--shmr` (default `fattahi18`).
- **Segue 1 data.** `data/segue1/` (used only by the legacy single-galaxy
  test `tests/integration/run_segue1.py`, not by the production runs) has no
  `PROVENANCE.md`.

## Relation to the SatelliteDensityInference record

- **Inputs.** The satgen-family priors (`satgen`, `satgen_box`,
  `satgen_shmr_*`) are built from SatelliteDensityInference's SatGen catalog
  `data/satgen_catalogs/m12res8_10k_Diemer+scatter_sim.h5` and, for the SHMR
  priors, its frozen weights `data/halo_weights_published/Diemer/`, both in
  that record's `inputs` tier. Both builders take `--h5` (the SHMR builder
  also `--weights-dir`), defaulting to a sibling SatelliteDensityInference
  checkout.
- **Outputs.** SatelliteDensityInference reads `derived.npz` through
  `Dwarf.get_Jeans_results`: the J-factor panels for 7 of the 8 priors (all
  but `jeffreys`), and the Fermi recasting, tables, contours and
  number-function figures for the priors each names. It finds this tree
  through `SDI_DJA_RESULTS_DIR` (default
  `../DwarfJeansAnalysis/results/production`), so after fetching this record
  into a sibling checkout no variable is needed; otherwise
  `export SDI_DJA_RESULTS_DIR=<this checkout>/results/production`.
- **`jeans_corner.pdf`.** SatelliteDensityInference's `fetch_jeans_corner.py`
  runs this repository's `scripts/plot_posteriors.py` in this checkout. That
  script reads `ursa_major_2/loguniform`'s `posterior_samples.npz` and
  `audit.json` and the repository's `data/`, and recomputes the derived
  quantities itself; it does not read `derived.npz`. It needs LaTeX and
  `corner`.
- **Its snapshot.** SatelliteDensityInference's own record carries
  `dja_results_snapshot.tar.gz`: 274 files sha256-identical to this tree
  (`derived.npz` for the 7 non-`jeffreys` priors times 39 dwarfs, plus
  `ursa_major_2/loguniform/posterior_samples.npz`), unpacked under its
  `data/dja_results/`. It has no `audit.json`, `run.log` or `summary.csv`.

## Third-party data

The deposited files are our outputs, but they are not free of third-party
values: `audit.json`'s `registry_row` and `nuisance_priors`, and `run.log`,
repeat Local Volume Database (CC0) and literature values (distances,
half-light radii, ellipticities, proper motions), and the posteriors derive
from the per-star catalogs. Those inputs are in the git repository under
`data/`, not in this record. Each per-paper directory has a `PROVENANCE.md`
naming its source; `data/star_catalogs/`, `data/registry/` and
`data/satgen_prior/` (derived products the runs read) and the test-only
`data/segue1/` have none. The Walker et al. 2009 table is the star catalog
for `carina_1` only (the `sculptor_1` and `sextans_1` catalogs are from Geha
et al. 2026; Fornax is not in the sample). The repository `README.md` lists
the citations.

## How this record was checked

Not yet recorded: the clean-room check (a fresh clone holding only this
record, unpacked with `scripts/zenodo_fetch.py`, in an environment built from
`environment.yml`) is run on this packing, and its results replace this
paragraph before publication.

## Licence

CC BY 4.0 for the deposited products. The posteriors derive from third-party
catalogs that are not in this record and keep their own terms (the Local
Volume Database is CC0); cite the original works listed in the repository
`README.md`.

Cite the two papers above; the Zenodo DOI is added on publication.
