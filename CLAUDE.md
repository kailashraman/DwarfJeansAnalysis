# CLAUDE.md

Guidance for Claude Code when helping someone use this repository: run the
Jeans analysis, regenerate derived products from the deposited posteriors, or
check the deposit. `README.md` is the overview; `zenodo/README_DATA.md`
describes the data record and what reproduces. Read those before giving
instructions. The authors' development rules are in
`docs/claude.development.md`.

## What the repository is

`dwarfjeans`: spherical-Jeans modelling of Milky Way dwarf-galaxy stellar
kinematics (an NFW halo, a Plummer tracer, constant velocity anisotropy),
sampled with dynesty and propagated to J- and D-factors. It produced the Jeans
posteriors for a two-part paper series (Raman, Folsom, Kaplinghat, Lisanti,
Safdi) whose figures are made in the SatelliteDensityInference repository,
which consumes this one's `results/production/`.

```
src/dwarfjeans/   the package (ingest, jeans, jd, postprocess)
scripts/          production driver, reprocess, plots, cluster launchers,
                  Zenodo pack/fetch/upload tools
data/             inputs in git: registry, per-star catalogs, per-paper raw
                  tables, LVDB snapshot, SatGen prior tables
results/          production/ (posteriors, from Zenodo), audits/, tests/
zenodo/           README_DATA.md and metadata.json of the data record
docs/             pipeline writeup, plans, development guide
tests/            pytest suite
```

## Setting up

- Environment: `conda env create -f environment.yml && conda activate
  DwarfJeans`, from the repository root (it installs the package editable).
  `environment.yml` is the tested route; `pip install -e .[dev]` (Python 3.10
  or later) resolves unpinned versions and has not been checked.
- LaTeX is not in the environment. The plotting scripts set `text.usetex`, so
  they need a `latex` binary on the `PATH`.
- Inputs are in git (`data/`); the production posteriors are not (next section).

## The Zenodo record

The 312 production runs (39 dwarfs by 8 priors, 5 files each, about 0.97 GB
before compression) are one archive, `dja_production.tar.gz`, in a Zenodo
record (`zenodo/README_DATA.md` is its README; the DOI and record ID are added
to `README.md` on publication). Fetch from the repository root:

```bash
python scripts/zenodo_fetch.py --record <RECORD_ID>
```

It unpacks into `results/production/`. It checks the archive against Zenodo's
md5 and `MANIFEST.json`'s checksums, requires the archive's members to equal
the manifest's set (regular files under `results/production/`), and verifies
each extracted file's sha256. It refuses to overwrite a differing local file
unless given `--force`, and an existing directory or symlink at a target path
aborts even with `--force`. Extraction is not transactional: after a failure
the files already extracted remain, so fix the cause and rerun with
`--force`. `--files-from DIR` unpacks from a local directory with no network.
It needs `requests` and a Python with `tarfile.data_filter` (both satisfied by
`environment.yml`). `MANIFEST.json`'s `dja_git_commit` is the commit at packing
time, not the code that produced the runs. `scripts/zenodo_pack.py` and
`scripts/zenodo_upload.py` are the authors' tools; a user does not need them.

What each file holds, and what has and has not been checked to reproduce, is in
`zenodo/README_DATA.md`; quote its scope, not a shorter version. In particular:
the deposited `posterior_samples.npz` holds only `samples_eq`; `audit.json`
and `run.log` record no commit, package versions or input hashes; their
`r_t = 1.0 kpc` values are superseded by the per-draw tidal radius in
`derived.npz` and `summary.csv`; and the deposited chains' `samples_eq`
reproduce only statistically.

## Running things

- **One galaxy and prior:** `python scripts/run_production.py --lvdb-key
  pisces_2 --prior jeffreys --nlive 1500 --dlogz 0.05 --npool 1 --output-base
  results/scratch` (pisces_2 was deposited at `npool=1`; draco_1 was `npool=8`
  and takes hours). Always pass `--output-base` unless you mean to replace the
  canonical `results/production/<key>/<prior>/`, which downstream work reads.
  The defaults (`--nlive 500 --dlogz 0.1`) are looser than the production
  sweep's. A run replaces the directory's `posterior_samples.npz`,
  `summary.csv`, `audit.json` and `run.log` but not `derived.npz`, so a stale
  `derived.npz` survives next to a new chain until `reprocess.py` runs there.
- **Regenerate derived products from saved chains:** `python
  scripts/reprocess.py --base <copy of results/production> --glob
  '<key>/<prior>' --no-plots`. A bare run defaults to `--base
  results/production --glob '*/*'` and rewrites all 312 runs in place; plots
  go to `<repo>/plots/` regardless of `--base` and need LaTeX and `corner`. It
  needs `data/` and each run's `audit.json`; the tidal host is the code
  default (MilkyWayPotential2022).
- **The `scripts/submit_*.sh` launchers are specific to the authors' SLURM
  cluster** (account, partition, an excluded-node list on their scratch, a
  `DwarfJeans` conda environment activated through `~/.bashrc`). One call
  runs one prior on a cohort; the deposit's configuration is 16 submissions
  (classical at `npool=8`, ufd at `npool=1`, times 8 priors; see `README.md`). Don't run
  them as they stand elsewhere, and don't submit cluster jobs without the
  user's say-so. Run `scripts/run_production.py` and `scripts/reprocess.py`
  directly instead.
- **The satgen-family priors** read `data/satgen_prior/` (in git). To rebuild
  them, `scripts/build_satgen_prior_table.py` and
  `scripts/build_satgen_shmr_prior_tables.py` take `--h5` (the SHMR builder
  also `--weights-dir`), defaulting to a sibling
  SatelliteDensityInference checkout (the inputs are in that record's `inputs`
  tier).

## Things that will bite

- **`--npool` changes the chain.** A different pool size gives a different
  chain (the shift was not quantified). 232 deposited runs used `npool=1`, 80
  used `npool=8`; for the deposited runs the value is in `run.log`, not in
  `audit.json` (new runs record it in both).
- **Rerunning does not reproduce the deposited `samples_eq` exactly.** The
  resample that produced it was unseeded; it is seeded now, so new runs
  are expected to reproduce their own `samples_eq` given the same seed and
  `npool`; this was tested on a toy likelihood and on pisces_2/satgen at
  `npool=1` (rerun twice, identical chains). Pooled reruns were not tested. Say "reproduces statistically" for the deposited
  chains.
- **Dwarf names are LVDB keys** (`draco_1`, `ursa_major_2`), not display names.
- **`uniform` is a valid `--prior` but no deposited run uses it.** The SHMR
  priors are selected with `--prior satgen_shmr` plus a required `--shmr
  <name>`, and write to a `satgen_shmr_<name>/` directory.
- **SatelliteDensityInference reads this tree** (its `SDI_DJA_RESULTS_DIR`,
  default `../DwarfJeansAnalysis/results/production`). Replacing a run here
  changes figures there.

## Tests

`pytest` (bare) runs `tests/` per `pyproject.toml`'s `testpaths`, which
collects the `test_*.py` modules under `tests/unit/`. There are no markers or
opt-in tiers: no `--rundata` or `--runslow`. The files under
`tests/integration/` (`run_*.py`, `analyze_asimov.py`) are not named `test_*`,
so bare `pytest` does not collect them. They are scripts run by hand and some
are slow: `run_perspective_calibration.py` is a mock-data calibration with its
command in its docstring, `run_ufd_population.py` a 15-realization mock study,
`analyze_asimov.py` an Asimov bias check, and `run_segue1.py` a real-data
Segue 1 smoke test (its non-default `simon` and `pace` sources read
`data/segue1/`, which is not in the repository). Their outputs go to `results/tests/`, which is gitignored.
`tests/unit/test_jeans_vs_quad.py` is collected and is slow (its docstring
says about 80 seconds). `tests/unit/test_scale_sep95.py` skips its per-galaxy
tests for a galaxy whose `jeffreys` run is absent from `results/production/`,
and its summary test needs all 39 (fetch the record first). Say which tests
ran and which skipped.
