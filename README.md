# dwarfjeans

Spherical-Jeans modelling of Milky Way dwarf-galaxy stellar kinematics. Given a
per-star spectroscopic catalog, it fits an NFW halo with a Plummer stellar tracer
and constant velocity anisotropy, then propagates the posterior through to the
J- and D-factor integrals used in indirect-detection work.

Pre-release (`version = 0.0.0`): APIs and output formats may still change.

## Install

Python ≥ 3.10, from a checkout:

```bash
pip install -e .        # runtime
pip install -e .[dev]   # adds pytest
```

Dependencies are declared in `pyproject.toml`. There is no PyPI or conda
release; install from source.

`environment.yml` is the tested route: it pins the package versions of the
authors' environment (linux-64), and the record was checked in an environment
built from it. The `pip` route resolves unpinned versions and has not been
checked. Create the environment from the repository root, which also installs
the package editable:

```bash
conda env create -f environment.yml && conda activate DwarfJeans
```

LaTeX is not included; the plotting scripts set `text.usetex`, so they need a
`latex` binary on the `PATH`. The SLURM launchers activate a conda environment
named `DwarfJeans` (see "Cluster launchers" below).

## Run

The registry (`data/registry/galaxies.ecsv`) and all 39 staged catalogs
(`data/star_catalogs/<lvdb_key>.npz`) are committed, so there is no build step.
For a single galaxy:

```bash
python scripts/run_production.py --lvdb-key draco_1 --prior jeffreys \
    --nlive 1500 --dlogz 0.05 --npool 8 --output-base results/scratch
```

**Runs overwrite in place.** Without `--output-base`, results are written to the
canonical `results/production/<key>/<prior>/`, replacing its `posterior_samples.npz`,
`summary.csv`, `audit.json` and `run.log` — and that tree is read directly by
downstream work. A run does not write `derived.npz`: an old one stays until
`scripts/reprocess.py` is run on that directory, and SatelliteDensityInference
reads `derived.npz`. The sampler defaults
(`--nlive 500 --dlogz 0.1`) are also looser than the production sweep's
`1500`/`0.05`, so an unqualified re-run silently downgrades an archived chain.
Pass `--output-base` unless you intend to replace the canonical result.

A run directory contains:

| file | contents |
|---|---|
| `summary.csv` | q16/q50/q84 for V_sys, halo parameters, σ_los at R_½, M_half, J/D (31 rows; `zenodo/README_DATA.md` lists them) |
| `posterior_samples.npz` | equal-weight posterior samples (`samples_eq`) plus reproducibility metadata; the deposited copies hold `samples_eq` only |
| `audit.json` | registry inputs, selection policy, sampler configuration |
| `run.log` | run stdout |

`scripts/reprocess.py` regenerates the derived views from a saved chain without
re-running the sampler, and additionally writes `derived.npz` — the per-draw
derived arrays that external consumers read. A bare `scripts/reprocess.py`
defaults to `--base results/production --glob '*/*'` and rewrites all runs
there in place, and it writes plots under `<repo>/plots/<key>/<prior>/`
regardless of `--base` (they need LaTeX and `corner`). To regenerate one run
on a copy, without plots:

```bash
python scripts/reprocess.py --base <copy of results/production> \
    --glob '<lvdb_key>/<prior>' --no-plots
```

It needs `data/` and the run's `audit.json` (seed, thinning, selection
policy). The tidal host is the code default (MilkyWayPotential2022) unless
`--host` is given. Plots for a run (they read `results/production/<key>/<prior>`
unless `--run-dir` points elsewhere, and need LaTeX and `corner`):

```bash
python scripts/plot_posteriors.py --lvdb-key draco_1
python scripts/plot_posteriors.py --lvdb-key draco_1 --run-dir results/scratch/draco_1/jeffreys
```

`--prior` accepts `uniform`, `loguniform`, `jeffreys`, `satgen`, `satgen_box`,
and `satgen_shmr` (the last requires `--shmr`, and writes to a
`satgen_shmr_<shmr>/` leaf rather than `satgen_shmr/`). The production sweep
used all of these except `uniform`; no `uniform` run is in the deposit. See
`python scripts/run_production.py --help`.

### Cluster launchers

The full sweep runs as a SLURM array. **`scripts/submit_batch.sh` and
`scripts/submit_reprocess.sh` are written for the authors' cluster**: their
`#SBATCH` headers name our account, partition and excluded nodes (both scripts read a node list from the authors' scratch), and they
`source ~/.bashrc` and activate the `DwarfJeans` conda environment. They will
not run unmodified elsewhere. Elsewhere, run `scripts/run_production.py` (one
galaxy and prior per call, as above) and `scripts/reprocess.py` directly, or
adapt the headers to your scheduler. On the authors' cluster, invoke the
sweep through `bash`, not `sbatch` — the wrapper resolves cohort membership and
node exclusions before resubmitting:

```bash
bash scripts/submit_batch.sh --cohort classical  # heavier galaxies, pool of 8
bash scripts/submit_batch.sh --cohort ufd        # lighter galaxies, serial
```

Each call runs one prior (default `jeffreys`) on its cohort; `bash
scripts/submit_batch.sh` with no flags runs all 39 galaxies at a pool of 8
under `jeffreys` only. The deposit's configuration is each cohort under each
of the 8 prior configurations, 16 submissions: `--prior jeffreys`, `--prior
loguniform`, `--prior satgen`, `--prior satgen_box`, and `--prior satgen_shmr
--shmr <name>` for `danieli23_const`, `fattahi18`, `kim24` and `moster18`.

## Production posteriors (Zenodo)

The 312 production runs (39 dwarfs by 8 priors; about 0.97 GB) are not in git.
They are in a Zenodo record (DOI added on publication) described by
`zenodo/README_DATA.md`, which gives the file formats, what reproduces and how
far, and the provenance gaps. Fetch them from the repository root:

```bash
python scripts/zenodo_fetch.py --record <RECORD_ID>
```

This unpacks into `results/production/` and checks the archive's checksums
(`--help` for the options). `results/production/README.md` describes the
layout. `scripts/zenodo_pack.py` and `scripts/zenodo_upload.py` are the
authors' packing and upload tools; a user does not need them.

### Reproducibility caveats

Stated precisely in `zenodo/README_DATA.md`; in short:

- Re-running the sampler at the deposit's settings (`--nlive 1500 --dlogz
  0.05`, seed 0) reproduced `logZ`, `logZ_err` and `n_eq` exactly for two
  `npool=1` runs (pisces_2), made before the resample was seeded. The
  deposited `samples_eq` was resampled with an unseeded generator, so it
  reproduces only statistically. The resample is now seeded, so new runs are
  expected to reproduce their own `samples_eq` given the same seed and
  `npool`; this was tested on a toy likelihood, not yet on a real galaxy.
- A different `--npool` gives a different chain (the shift was not quantified
  here). 232 deposited runs used `npool=1` and 80 used `npool=8`; use the
  `npool` in each run's `run.log`. Pooled runs were not rerun.
- `scripts/reprocess.py` reproduced `derived.npz` and `summary.csv` bit for bit
  from the deposited chains for four runs, not all 312.
- The deposited `audit.json` and `run.log` record no git commit, package
  versions or input hashes; their `r_t = 1.0 kpc` values are superseded by the
  per-draw tidal radius in `derived.npz` and `summary.csv`; and the deposited
  `posterior_samples.npz` was reduced to `samples_eq` by a step not in git
  history.
- `data/satgen_prior/` was rebuilt from SatelliteDensityInference's deposited
  inputs: every numeric array of the 157 files was identical, but the recorded
  metadata differs, and the record does not prove these are the exact tables
  the SHMR runs read. See `zenodo/README_DATA.md`.

## Documentation

- `docs/writeup/pipeline.tex` — the pipeline writeup; the compiled PDF is
  committed alongside it.
- `docs/plan/` — per-stage specifications, ingest sources, and derivations.
- `ARCHITECTURE.md` — layout, module boundaries, output-path conventions.
- `zenodo/README_DATA.md` — the Zenodo record: contents, formats, reproducibility.
- `CLAUDE.md` — a guide for using the repository with Claude Code. The
  authors' development rules are in `docs/claude.development.md`.

## Data

Global dwarf properties come from the Local Volume Database (v1.0.5). Per-star
velocities come from the original spectroscopic studies, ingested per paper under
`data/`. Each per-paper ingest directory carries a `PROVENANCE.md` recording
where the source came from and a `checksums.sha256` pinning it. The legacy
single-galaxy test `tests/integration/run_segue1.py` reads a `data/segue1/`
directory that is not part of the repository (the production Segue 1 run uses
the Geha et al. 2026 catalog), so that test cannot run from a clone. These data are in git, not in
the Zenodo record, and keep their original terms. If you use them, cite the
original works:

- **Local Volume Database**, `data/lvdb_v1.0.5/` (CC0): Pace 2025, OJAp 8, 142
  ([arXiv:2411.07424](https://arxiv.org/abs/2411.07424)); v1.0.5 release,
  [doi:10.5281/zenodo.15476348](https://doi.org/10.5281/zenodo.15476348).
- `data/bruce2023/`: Bruce et al. 2023, ApJ 950, 167 (Aquarius II).
- `data/chiti2022/`: Chiti et al. 2022, ApJ 939, 41 (Grus I).
- `data/chiti2023/`: Chiti et al. 2023, AJ 165, 55 (Tucana II; VizieR J/AJ/165/55).
- `data/geha2026/`: Geha et al. 2026, DEIMOS Stellar Archive Paper I
  ([arXiv:2602.10200](https://arxiv.org/abs/2602.10200)); the companion Paper II
  is [arXiv:2602.10202](https://arxiv.org/abs/2602.10202). Release stamp
  `20260110`, from https://geha-group.github.io/deimos/.
- `data/hansen2024/`: Hansen et al. 2024, ApJ 968, 21 (Tucana V).
- `data/heiger2024/`: Heiger et al. 2024, ApJ 961, 234 (Centaurus I).
- `data/ji2021/`: Ji et al. 2021, ApJ 921, 32 (Antlia II, Crater II; VizieR J/ApJ/921/32).
- `data/kirby2015/`: Kirby, Simon & Cohen 2015, ApJ 810, 56 (Pisces II).
- `data/koposov2015/`: Koposov et al. 2015, ApJ 811, 62 (Horologium I).
- `data/koposov2018/`: Koposov et al. 2018, MNRAS 479, 5343 (Hydrus I; VizieR J/MNRAS/479/5343).
- `data/li2017/`: Li et al. 2017, ApJ 838, 8 (Eridanus II).
- `data/li2018/`: Li et al. 2018, ApJ 857, 145 (Carina II and III; VizieR J/ApJ/857/145).
- `data/simon2020/`: Simon et al. 2020, ApJ 892, 137 (Tucana IV; VizieR J/ApJ/892/137).
- `data/tan2025/`: Tan et al. 2025, ApJ 979, 176 (Leo VI).
- `data/walker2009/`: Walker et al. 2009, AJ 137, 3100 (the star catalog for Carina only; VizieR J/AJ/137/3100).
- `data/walker2015/`: Walker et al. 2015, ApJ 808, 108 (Reticulum II; VizieR J/ApJ/808/108).

`data/registry/`, `data/star_catalogs/` and `data/satgen_prior/` are derived
here from the sources above and from SatelliteDensityInference's catalogs, and
have no `PROVENANCE.md`.

Posterior sampling uses `dynesty`: Speagle 2020, MNRAS 493, 3132. The citation
snippet used for the writeup, which also cites the `dynesty` software release,
is `docs/writeup/dynesty_citations.tex`.

## Citation

This code supports a two-part study, *Semi-analytic Inference of Satellite
Densities in the Cold Dark Matter Model*:

- **Part I — Comparison to Ultra-faint Dwarf Kinematics.**
  Raman, Folsom, Kaplinghat, Lisanti & Safdi (2026),
  [arXiv:2607.27316](https://arxiv.org/abs/2607.27316).
- **Part II — Implications for Dark Matter Indirect Detection Constraints.**
  Same authors (2026),
  [arXiv:2607.27326](https://arxiv.org/abs/2607.27326).

The posteriors produced here are consumed by
[SatelliteDensityInference](https://github.com/kailashraman/SatelliteDensityInference),
which generates the figures for both papers.

## License

MIT, declared in `pyproject.toml`; see `LICENSE`.

Maintainer: Kailash Raman (`kailash.raman@berkeley.edu`).
