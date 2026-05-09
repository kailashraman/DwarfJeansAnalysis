# ARCHITECTURE

This document describes the **current** layout of DwarfJeansAnalysis
and the conventions that govern where new work goes. Per-module APIs,
equations, and calibration claims live in `docs/plan/` and
`docs/writeup/pipeline.tex`. This file describes only **where things
go and why**.

The `src/` migration that this document originally planned is
complete; see "Migration status" below.

## Top-level layout

```
DwarfJeansAnalysis/
├── ARCHITECTURE.md                  this file
├── CLAUDE.md                        agent behavioral guidelines
├── README.md                        one-screen orientation (future)
├── pyproject.toml                   package metadata + deps
├── src/
│   └── dwarfjeans/
│       ├── __init__.py
│       ├── ingest/
│       │   ├── stage0a_registry.py
│       │   ├── stage0b_geha.py
│       │   ├── stage0b_pathb.py
│       │   ├── stage0b_pathb_worklist.py
│       │   ├── staging.py
│       │   ├── path_b_adapters/     per-paper raw → per-star OR per-epoch
│       │   ├── multi_epoch.py       IVW + χ² primitives (analysis-time)
│       │   ├── combiners/           per-paper per-epoch → per-star handlers
│       │   └── config/              YAML overrides (spatial models, samples)
│       ├── jeans/
│       │   ├── inference.py         dynesty driver (4D and 7D)
│       │   ├── solver.py            σ_los(R) Jeans integrator
│       │   ├── priors.py            three-prior registry (uniform,
│       │   │                         loguniform, jeffreys); Fisher-
│       │   │                         determinant log term
│       │   ├── constant_sigma.py    Walker+2006 constant-σ baseline
│       │   ├── selection.py         per-star cuts (p, R, var)
│       │   └── preprocess.py        per-epoch → per-star → selection
│       │                             orchestrator (combine_policy +
│       │                             selection_policy)
│       ├── jd/
│       │   ├── factors.py
│       │   └── summary.py
│       └── mocks/
│           └── galaxy.py
├── scripts/
│   ├── run_production.py            one-galaxy production driver
│   ├── submit_batch.sh              SLURM array driver (one task / catalog)
│   ├── run_batch.py                 local-node batch (multiprocessing)
│   ├── plot_posteriors.py           per-galaxy plot regen
│   └── bench_*.py                   solver / sampler benchmarks
├── tests/
│   ├── unit/                        per-function checks
│   │   ├── test_jeans.py
│   │   ├── test_jeans_vs_quad.py
│   │   ├── test_priors.py
│   │   ├── test_selection.py
│   │   ├── test_preprocess.py
│   │   ├── test_multi_epoch.py
│   │   ├── test_default_combiner.py
│   │   └── test_combiner_dispatch.py
│   └── integration/                 whole-pipeline tests (gold-standard)
│       ├── run_segue1.py
│       ├── run_ufd_population.py
│       └── analyze_asimov.py
├── docs/
│   ├── writeup/
│   │   └── pipeline.tex             single doc, all stages
│   │                                 (PDF built by CI — not committed)
│   ├── plan/                        living markdown specs
│   ├── original-plan/               frozen reference (read-only)
│   └── review-checklist.md          recurring bug classes for reviewers
├── data/
│   ├── registry/                    galaxies.ecsv + build_log
│   ├── star_catalogs/               per-galaxy .npz inputs to inference
│   ├── lvdb_v1.0.5/                 LVDB snapshot
│   └── <per-paper raw dirs>         raw ingest inputs (kirby2015/, walker2015/, …)
├── plots/                           per-galaxy regen plots (gitignored)
└── results/                         run outputs (gitignored)
    ├── production/<lvdb_key>/<prior>/   canonical, overwritten on each run
    └── tests/<test_name>/               outputs from integration-test runs
```

## Layout rationale

- **`src/` layout.** Forces installed-package imports
  (`from dwarfjeans.jeans import inference`) and prevents
  cwd-relative shadowing. Standard for installable Python packages.

- **`scripts/` for production drivers.** Production drivers
  (`run_production.py`), the SLURM submission script
  (`submit_batch.sh`), local-node batch (`run_batch.py`), plot regen
  (`plot_posteriors.py`), and benchmarks (`bench_*.py`) live here.
  These are entry points, not library code; keeping them outside
  `src/dwarfjeans/` keeps the package surface clean.

- **`tests/unit/` vs. `tests/integration/`.** Enforces the CLAUDE.md
  gold-standard rule. `integration/` holds whole-pipeline tests:
  `run_segue1.py` (single-galaxy mock-pipeline check),
  `run_ufd_population.py` (15-realization MC recovery), and
  `analyze_asimov.py` (Asimov bias check).

- **`results/` gitignored, split by purpose.**
  - `results/production/<lvdb_key>/<prior>/` — canonical per-galaxy
    production output. **Overwritten on each run**; wrong/old results
    are not retained for provenance (provenance is captured inside
    `audit.json` via `timestamp_utc`, dynesty config, registry row).
  - `results/tests/<test_name>/` — outputs from integration-test runs
    (Segue 1 testbed, etc.).

  Splitting prevents test artifacts from colliding with science
  outputs. The flat per-(galaxy, prior) layout under `production/`
  keeps the central tree small across re-runs — earlier per-SLURM-job
  / per-timestamp subdirectories ballooned the tree and were collapsed
  on 2026-05-08.

- **`docs/writeup/`.** Single `pipeline.tex` covering all stages,
  per CLAUDE.md. The `.tex` source lives in the repo; the compiled
  PDF is built by `.github/workflows/writeup.yml` on changes to
  `docs/writeup/**` and exposed as a workflow artifact (not
  committed).

- **`jeans/priors.py` is its own module** because we run with multiple
  prior choices (loguniform vs. Jeffreys) and the Stage 2 MC and
  Segue 1 testbed both compare across them. The module also owns
  the `V_HALFWIDTH` constant; the V prior is centered on the
  post-selection IVW mean of the catalog (computed in
  `scripts/run_production.py`), with the halfwidth overridable per
  galaxy via the optional `vlos_prior_halfwidth_kms` registry column.

- **`jeans/constant_sigma.py`** holds the Walker+2006
  radius-independent constant-σ inference. It is the model-free
  counterpart to the Jeans posterior σ_los(R) and is the canonical
  cross-paper comparison quantity (P&S 2018, Walker 2006).

- **Ingest preserves provenance; combination is downstream.** Adapters
  in `path_b_adapters/` write the catalog at whatever granularity the
  upstream paper provides — per-star when the paper publishes
  IVW-combined per-star velocities (Walker 2009 for Carina I), and
  per-epoch when the paper publishes one row per spectrum (Li 2018,
  Chiti 2023, etc.). The `_meta["catalog_granularity"]` field
  declares which it is.

  Per-epoch → per-star is **not** an ingest step. The σ_sys floor,
  zero-point offsets, and χ² p-threshold are *analysis choices*
  (which we will retune across runs), not provenance facts. Folding
  them into the on-disk catalog would force a 39-galaxy re-ingest
  every time we revisit them, and would erase the per-epoch
  information needed for binary-aware σ-deconvolution and period
  searches.

- **`ingest/multi_epoch.py` + `ingest/combiners/`** hold the
  combination machinery, *invoked at analysis time* by
  `jeans/preprocess.py` rather than by the ingest stage drivers.

  `multi_epoch.py` holds the dataset-agnostic primitives:

  - **IVW combine** with per-instrument zero-point offsets applied
    *before* the weighted mean (caller responsibility), and the
    instrument systematic error floor added in quadrature *after*:
    `v̄ = Σ vᵢ/σᵢ² / Σ 1/σᵢ²`,
    `σ_v̄ = (Σ 1/σᵢ²)^(−1/2) ⊕ σ_sys`.
  - **Variability flag** via `χ² = Σ (vᵢ − v̄)² / σᵢ²` against a
    p-value threshold (e.g. `p < 0.01`); single-epoch stars are
    left unflagged.

  `combiners/<paper>.py` holds per-paper handlers (chiti2022,
  chiti2023, hansen2024, li2017, li2018, simon2020) that know that
  paper's zero-point offsets, error floors, and variability
  thresholds. Handlers are organized **per paper, not per dwarf**,
  since the combine procedure is a property of the survey/instrument.
  The registry in `combiners/__init__.py` dispatches by
  `_meta["source_paper_bibcode"]`; an unregistered bibcode falls
  back to `combiners/default.py` (IVW + χ² with no offsets and zero
  σ_sys floor — adequate when published velocities are already on a
  single zero-point and errors include the survey's systematic
  budget).

- **`jeans/preprocess.py`** is the orchestrator that turns a raw
  catalog into Stage-1-ready arrays. It reads the catalog's
  granularity flag, runs the appropriate combiner if needed, then
  applies `select_jeans_stars`. It records both the `CombinePolicy`
  and the `SelectionPolicy` in the run's audit dict so any inference
  output is reproducible from the raw `.npz` plus the policy
  records. Per-star catalogs (e.g. Walker 2009 Carina I) skip the
  combiner step.

- **`docs/plan/` keeps its current role** as the living markdown
  spec set (`pipeline_overview.md`, `stage1.md`, `stage2.md`,
  `stage3.md`, `data_sources.md`, `jeffreys_jeans_derivation.md`,
  `per_paper_combiners.md`, `uncertainty_conventions.md`,
  `segue1_test.md`, `README_mc_test.md`). New edits go there, **not**
  into `docs/original-plan/`, which remains a frozen reference
  snapshot. `docs/review-checklist.md` accumulates recurring bug
  classes used by adversarial review.

## Module conventions

- **Public API** is exposed via `src/dwarfjeans/__init__.py` and the
  per-subpackage `__init__.py` files. Internal helpers stay
  underscore-prefixed.
- **Physical-units and uncertainty conventions** are defined once in
  `docs/plan/uncertainty_conventions.md`. Module docstrings reference
  it; they do not duplicate it.
- **Random-seed handling.** Every stochastic entry point takes an
  explicit `rng` or `seed` argument. No module-level `np.random`
  state, no implicit globals.

## Migration status

The `src/` migration originally laid out in this document is
**complete** (concluded 2026-05-07). `pyproject.toml` exists; the
`src/dwarfjeans/` package is editable-installed; ingest, jeans, jd,
and mocks all moved out of `docs/plan/*.py`; `priors.py` is extracted;
tests live under `tests/unit/` and `tests/integration/`; `results/`
is gitignored; `docs/writeup/pipeline.tex` exists and is built by CI.

The multi-epoch combiner machinery (`ingest/multi_epoch.py`,
`ingest/combiners/`) was originally listed as out-of-band feature
work; it is now mainline (seven per-paper handlers + default fallback,
unit tests under `tests/unit/test_*combiner*.py` and
`test_multi_epoch.py`).

## Maintenance contract

`ARCHITECTURE.md` is updated when:

- A new top-level directory is added.
- Module boundaries shift (a new `dwarfjeans/` subpackage, a moved
  responsibility).
- A repo-wide convention changes (units, RNG handling, output paths).

Routine code edits inside an existing module do not require updates
here. If you find yourself updating ARCHITECTURE.md to describe
specific code, that content probably belongs in a docstring or in
`docs/plan/` instead.
