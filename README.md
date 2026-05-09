# dwarfjeans

Spherical-Jeans modeling pipeline for Milky Way dwarf-galaxy stellar kinematics — extracts line-of-sight velocity dispersion, integrated halo mass, and indirect-detection J/D factors from per-star spectroscopic catalogs.

> **Status:** pre-release (`version = 0.0.0`). APIs and output formats may still shift; no PyPI release yet.

## Overview

`dwarfjeans` solves the spherical Jeans equation for an NFW dark-matter halo embedding a Plummer stellar tracer with constant velocity anisotropy `β`, fits the resulting `σ_los(R)` profile to a per-star kinematic catalog under a Bayesian framework (`dynesty` nested sampling), and propagates the posterior through to the dark-matter J- and D-factor integrals used in indirect-detection studies. The pipeline is built around the ~39 confirmed Milky Way dwarf spheroidals catalogued in [LVDB v1.0.5](https://github.com/apace7/local_volume_database) and the per-paper spectroscopic catalogs they reference (Geha+2026, Chiti+2022/2023, Hansen+2024, Li+2017/2018, Simon+2020, ...).

The headline design choices, in roughly the order they enter the pipeline:

- Per-paper ingest adapters preserve the upstream catalog's granularity (per-epoch where the paper publishes one row per spectrum, per-star where they publish IVW-combined velocities). Combination to per-star happens at *analysis time*, not at ingest, so σ_sys, zero-point offsets, and variability thresholds remain tunable without re-ingesting.
- Three-prior registry (`uniform`, `loguniform`, `jeffreys`) with a Fisher-determinant `½ ln D` correction for the Jeffreys variant; the V_sys prior is centered on the post-selection IVW mean of the catalog (data-driven, not registry-driven).
- A Walker+2006 constant-σ baseline runs alongside the full Jeans posterior as a model-free cross-check.
- Per-galaxy SLURM array submission with cohort-aware pool sizing (heavy classicals at `pool=8`, UFDs at `pool=1`) for the full 39-galaxy production sweep in ~15 min wall.

The mathematical derivations and pipeline-stage specifications live in `docs/plan/` (markdown specs) and `docs/writeup/pipeline.tex` (single-document writeup).

## Installation

Python ≥ 3.10. From a checkout of this repository:

```bash
pip install -e .          # runtime
pip install -e .[dev]     # adds pytest
```

Runtime dependencies (declared in `pyproject.toml`): `numpy`, `scipy`, `astropy`, `dynesty`, `multiprocess`, `matplotlib`, `pandas`, `pyyaml`. The `multiprocess` dependency (dill-backed pickling) is required because the dynesty likelihood closure cannot be serialized by stdlib `multiprocessing`.

No PyPI / conda release; install from source only at this stage.

## Quickstart

Once the registry and at least one staged catalog exist (`data/registry/galaxies.ecsv` and `data/star_catalogs/<lvdb_key>.npz`), run the production driver for a single galaxy:

```bash
python scripts/run_production.py --lvdb-key draco_1 --prior jeffreys --npool 8
```

Output lands at the canonical path `results/production/draco_1/jeffreys/` and contains:

- `summary.csv` — q16/q50/q84 percentiles for V_sys, halo parameters, σ_los at R_½, M_half(2D/3D), and J/D factors at the four reporting angles
- `posterior_samples.npz` — equal-weight samples + thinned chains
- `audit.json` — registry inputs, selection policy, dynesty configuration, timestamp
- `run.log` — full stdout

Generate the four standard plots:

```bash
python scripts/plot_posteriors.py --lvdb-key draco_1
```

For the full 39-galaxy sweep on SLURM:

```bash
sbatch scripts/submit_batch.sh                 # full --array=0-38
# or split by cohort:
bash scripts/submit_batch.sh --cohort classical    # 10 heavy, pool=8
bash scripts/submit_batch.sh --cohort ufd          # 29 light, pool=1
```

## Key features

- **Pipeline stages.** Stage 0a (registry build from LVDB) → 0b (per-paper kinematic ingest) → 1 (Walker+2006 constant-σ inference) → 2 (full NFW Jeans posterior, 4D fixed-nuisance or 7D nuisance-marginalized) → 3 (J/D integrals from the halo posterior).
- **NFW + Plummer + constant β** spherical Jeans solver with vectorized inner-integration on a 2048×512 log-spaced (R, u) grid; ≤ 5×10⁻³ error vs. adaptive quadrature at the production default.
- **Three prior families** (`uniform`, `loguniform`, `jeffreys`) selectable per run; the Jeffreys variant adds the proper Fisher-determinant `½ ln D` correction (derivation in `docs/plan/jeffreys_jeans_derivation.md`).
- **Per-paper multi-epoch combiner** (`ingest/combiners/`) for per-paper zero-point offsets, σ_sys floors, and χ² variability thresholds; `chiti2022`, `chiti2023`, `hansen2024`, `li2017`, `li2018`, `simon2020` handlers + a default IVW+χ² fallback.
- **dynesty pool wiring** via `multiprocess` (dill-backed) for parallelized likelihood evaluation; cohort-aware SLURM array driver.
- **Adversarial-review gate** before each non-trivial commit; recurring bug classes accumulated in `docs/review-checklist.md`.

## Documentation

The repo is the manual at this stage:

- `ARCHITECTURE.md` — top-level layout, module boundaries, output-path conventions.
- `docs/writeup/pipeline.tex` — the canonical pipeline writeup (single document, all stages). PDF built by CI on `docs/writeup/**` changes; not committed.
- `docs/plan/` — living markdown specs for each stage, ingest sources, uncertainty conventions, per-paper combiner notes, Jeffreys derivation.
- `docs/review-checklist.md` — recurring bug classes for adversarial review.
- `CLAUDE.md` — agent collaboration guidelines for this repo.

A hosted documentation site (Read the Docs / GitHub Pages) is not yet set up.

## Citation

*To be filled in.* No paper or DOI yet; a `CITATION.cff` will land before the first tagged release. For now, please contact the author before citing in published work.

## License

MIT (declared in `pyproject.toml`). A `LICENSE` file at the repo root is *to be added*.

## Contributing and contact

No `CONTRIBUTING.md` or formal issue tracker workflow yet. Author and maintainer: Kailash Raman (`kailash.raman@berkeley.edu`).

## Acknowledgments

*To be filled in* — funding sources, institutional support, upstream data products (LVDB, individual spectroscopic surveys) will be acknowledged here before the first release.
