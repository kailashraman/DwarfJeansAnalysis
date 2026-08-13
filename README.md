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

## Run

The registry (`data/registry/galaxies.ecsv`) and all 39 staged catalogs
(`data/star_catalogs/<lvdb_key>.npz`) are committed, so there is no build step.
For a single galaxy:

```bash
python scripts/run_production.py --lvdb-key draco_1 --prior jeffreys \
    --nlive 1500 --dlogz 0.05 --npool 8 --output-base results/scratch
```

**Runs overwrite in place.** Without `--output-base`, results are written to the
canonical `results/production/<key>/<prior>/`, replacing whatever is there — and
that tree is read directly by downstream work. The sampler defaults
(`--nlive 500 --dlogz 0.1`) are also looser than the production sweep's
`1500`/`0.05`, so an unqualified re-run silently downgrades an archived chain.
Pass `--output-base` unless you intend to replace the canonical result.

A run directory contains:

| file | contents |
|---|---|
| `summary.csv` | q16/q50/q84 for V_sys, halo parameters, σ_los at R_½, M_half, J/D |
| `posterior_samples.npz` | equal-weight posterior samples plus thinned chains |
| `audit.json` | registry inputs, selection policy, sampler configuration |
| `run.log` | run stdout |

`scripts/reprocess.py` regenerates the derived views from a saved chain without
re-running the sampler, and additionally writes `derived.npz` — the per-draw
derived arrays that external consumers read. Plots:

```bash
python scripts/plot_posteriors.py --lvdb-key draco_1
```

The full sweep runs as a SLURM array. Invoke it through `bash`, not `sbatch` —
the wrapper resolves cohort membership and node exclusions before resubmitting:

```bash
bash scripts/submit_batch.sh                     # all staged catalogs
bash scripts/submit_batch.sh --cohort classical  # heavier galaxies, larger pool
bash scripts/submit_batch.sh --cohort ufd        # lighter galaxies, serial
```

`--prior` accepts `uniform`, `loguniform`, `jeffreys`, `satgen`, `satgen_box`,
and `satgen_shmr` (the last requires `--shmr`, and writes to a
`satgen_shmr_<shmr>/` leaf rather than `satgen_shmr/`). See
`python scripts/run_production.py --help`.

## Documentation

- `docs/writeup/pipeline.tex` — the pipeline writeup; the compiled PDF is
  committed alongside it.
- `docs/plan/` — per-stage specifications, ingest sources, and derivations.
- `ARCHITECTURE.md` — layout, module boundaries, output-path conventions.
- `CLAUDE.md` — conventions for agent-assisted work in this repo.

## Data

Global dwarf properties come from the Local Volume Database (v1.0.5). Per-star
velocities come from the original spectroscopic studies, ingested per paper under
`data/`; each per-paper ingest directory carries a `PROVENANCE.md` recording
where the source came from and a `checksums.sha256` pinning it.

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

MIT, declared in `pyproject.toml`; a `LICENSE` file has not yet been added.

Maintainer: Kailash Raman (`kailash.raman@berkeley.edu`).
