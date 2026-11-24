# Production Posteriors — Zenodo Deposit

The posterior chains from the 39-galaxy production sweep are not stored in
this git repository. They are archived on Zenodo; `zenodo/README_DATA.md` (also
the record's README) describes the deposit, its file formats, what reproduces
and how far, and its provenance gaps. Read it before relying on anything here.

## Deposit

- **DOI:** [10.5281/zenodo.24508809](https://doi.org/10.5281/zenodo.24508809)
- **Record:** https://zenodo.org/records/24508809
- **Version:** `MANIFEST.json`'s `dja_git_commit` is the packing commit (the
  tools and documents), not the code that produced the runs.
- **Contents:** 312 runs: 39 galaxies times 8 prior directories (`jeffreys`,
  `loguniform`, `satgen`, `satgen_box`, and `satgen_shmr_<shmr>` for the four
  SHMRs `danieli23_const`, `fattahi18`, `kim24`, `moster18`), 5 files each
  (1560 files, 0.97 GB before compression). The code also accepts a
  `uniform` prior, but no `uniform` run is in the deposit.

## Layout

Each run is `<lvdb_key>/<prior>/` with:

| file | contents |
|---|---|
| `posterior_samples.npz` | `samples_eq`: equal-weight posterior samples, shape (N, 9), columns `V`, `log10_rs`, `log10_rhos`, `beta_tilde`, `d_kpc`, `eps`, `rhalf_arcmin`, `pmra`, `pmdec`. Nothing else is stored |
| `derived.npz` | per-draw derived arrays (J/D-factors, M(r_1/2), sigma_los, tidal radius, theta95, ...) |
| `summary.csv` | q16/q50/q84 of 31 quantities, all from `samples_eq` except `sigma_los_walker_kms` (a separate constant-sigma fit) |
| `audit.json` | registry row, selection policy, sampler configuration and thinning. Its `r_t_kpc` (1.0) is superseded |
| `run.log` | run stdout, including the pool size (`npool`). Its "J/D integrals" block (fixed `r_t=1.0 kpc`) is superseded |

`derived.npz` and `summary.csv` use a per-draw tidal radius and were
regenerated on 2026-07-09 from `samples_eq`; the `r_t=1.0 kpc` values in
`audit.json` and `run.log` are from the original May runs and should not be
used. `zenodo/README_DATA.md` lists the keys and rows and the other provenance
gaps. See `ARCHITECTURE.md` for the output-path convention.

## Fetching

From the repository root, in the environment built from `environment.yml`:

```bash
python scripts/zenodo_fetch.py --record 24508809
```

This unpacks into `results/production/`, checks the archive's checksums and
refuses to overwrite a differing local file unless given `--force`
(`python scripts/zenodo_fetch.py --help`).

## Reproduction

`--prior` is one of `uniform`, `loguniform`, `jeffreys`, `satgen`,
`satgen_box`, `satgen_shmr`; `--shmr` is required iff `--prior satgen_shmr`.
The chains can be regenerated from the tracked code and input catalogs
(`data/star_catalogs/`) with the settings of the deposit (`--nlive 1500
--dlogz 0.05 --rseed 0`, the default seed, recorded as `dynesty.rseed` in
`audit.json`):

```bash
# Single galaxy (--output-base keeps it from overwriting this tree)
python scripts/run_production.py --lvdb-key <key> --prior <prior> \
    [--shmr <shmr>] --nlive 1500 --dlogz 0.05 --rseed 0 --npool <npool> \
    --output-base results/scratch
```

Use the `npool` printed in the deposited run's `run.log` (1 for 232 runs, 8
for the 80 runs of the ten classical dwarfs); a different `npool` gives a
different chain (the size of the shift was not quantified). `scripts/submit_batch.sh` runs the sweep as a SLURM array and
is specific to the authors' cluster (see `README.md`).

How far this reproduces, as checked: the sampling stage reproduced `logZ`,
`logZ_err` and `n_eq` exactly for two `npool=1` runs (pisces_2 `satgen` and
`loguniform`), made before the equal-weight resample was seeded; the deposited
`samples_eq` reproduces only statistically (it was resampled with an unseeded
generator). Since the resample was seeded, pisces_2/satgen rerun twice gave
identical chains. `derived.npz` and `summary.csv` reproduced bit for bit from the
deposited chains for five runs (`scripts/reprocess.py`, on a copy; see
`README.md`). No `jeffreys`, SHMR-prior or `npool=8` run was rerun through
the sampler. The details
and the unchecked remainder are in `zenodo/README_DATA.md` ("What reproduces,
and how far"). The deposit records no git commit, package versions or input
hashes for the sampling runs.

## Citation

These posteriors support a two-part study, *Semi-analytic Inference of
Satellite Densities in the Cold Dark Matter Model* (Raman, Folsom,
Kaplinghat, Lisanti & Safdi, 2026):

- Part I — Comparison to Ultra-faint Dwarf Kinematics, arXiv:2607.27316
- Part II — Implications for Dark Matter Indirect Detection Constraints,
  arXiv:2607.27326

Cite the Zenodo record as doi:10.5281/zenodo.24508809.
