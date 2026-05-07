# ARCHITECTURE

This document declares the **target** layout for DwarfJeansAnalysis and
the migration path from the current state. It is the contract that
subsequent work — code promotion, LaTeX writeup, test scaffolding, CI —
will conform to.

It is **not** a reference manual. Per-module APIs, equations, and
calibration claims live in `docs/plan/` and (eventually)
`docs/writeup/pipeline.tex`. This file describes only **where things
go and why**.

## Target top-level layout

```
DwarfJeansAnalysis/
├── ARCHITECTURE.md                  this file
├── CLAUDE.md                        agent behavioral guidelines
├── README.md                        one-screen orientation (future)
├── pyproject.toml                   package metadata + deps (future)
├── src/
│   └── dwarfjeans/
│       ├── __init__.py
│       ├── ingest/                  ← from data_ingest/
│       │   ├── stage0a_registry.py
│       │   ├── stage0b_geha.py
│       │   ├── stage0b_pathb.py
│       │   ├── stage0b_pathb_worklist.py
│       │   ├── staging.py
│       │   ├── path_b_adapters/     per-paper raw → per-epoch
│       │   ├── multi_epoch.py       (future) IVW + χ² primitives
│       │   └── combiners/           (future) per-dataset per-epoch → per-star
│       ├── jeans/
│       │   ├── inference.py         ← from docs/plan/jeans_inference.py
│       │   ├── solver.py            ← from docs/plan/jeans.py
│       │   └── priors.py            (extracted from jeans_inference.py:
│       │                             prior-transform builders, Jeffreys
│       │                             log-term, prior bounds)
│       ├── jd/                      ← from docs/plan/j_d_factors.py
│       │   ├── factors.py
│       │   └── summary.py           ← from run_jd_summary.py
│       └── mocks/                   ← from docs/plan/mock_galaxy.py
│           └── galaxy.py
├── tests/
│   ├── unit/                        per-function checks
│   │   ├── test_jeans.py            ← from docs/plan/
│   │   └── test_jeans_vs_quad.py    ← from docs/plan/
│   └── integration/                 whole-pipeline tests (gold-standard)
│       ├── run_segue1.py            ← from Segue1_test/run_segue1.py
│       ├── run_ufd_population.py    ← from docs/plan/
│       └── analyze_asimov.py        ← from docs/plan/
├── docs/
│   ├── writeup/
│   │   ├── pipeline.tex             single doc, all stages
│   │   └── pipeline.pdf
│   ├── plan/                        living markdown specs
│   └── original-plan/               frozen reference (read-only)
├── data/                            unchanged
│   ├── registry/                    galaxies.ecsv + build_log
│   ├── star_catalogs/               per-galaxy .npz inputs to inference
│   ├── lvdb_v1.0.5/                 LVDB snapshot
│   └── <per-paper raw dirs>         raw ingest inputs (kirby2015/, walker2015/, …)
└── results/                         run outputs (gitignored)
    ├── tests/                       outputs from integration-test runs
    │   └── segue1/                  ← from Segue1_test/baseline_*, *.npz, *.png
    └── <galaxy_key>/                future per-galaxy production runs
                                     (e.g. results/reticulum_2/, results/carina_1/)
```

## Layout rationale

- **`src/` layout.** Forces installed-package imports
  (`from dwarfjeans.jeans import inference`) and prevents
  cwd-relative shadowing. Standard for installable Python packages.

- **`tests/unit/` vs. `tests/integration/`.** Enforces the CLAUDE.md
  gold-standard rule. `integration/` holds whole-pipeline tests:
  `run_segue1.py` (single-galaxy mock-pipeline check),
  `run_ufd_population.py` (15-realization MC recovery), and
  `analyze_asimov.py` (Asimov bias check). These are tests, not
  drivers — there is **no separate `scripts/` directory** in the
  target layout.

- **`results/` gitignored, split by purpose.**
  - `results/tests/<test_name>/` — outputs from integration-test runs
    (Segue 1 testbed today; future test runs go alongside).
  - `results/<galaxy_key>/` — per-galaxy production runs once Stage 1
    inference is run on the Path B galaxies.

  Splitting prevents test artifacts from colliding with real science
  outputs.

- **`docs/writeup/`.** Single `pipeline.tex` covering all stages,
  per CLAUDE.md. The `.tex` source and the latest compiled PDF both
  live in the repo and are updated whenever the pipeline changes.

- **`jeans/priors.py` is its own module** because we already run with
  multiple prior choices (logflat vs. Jeffreys for Segue 1) and Stage 2
  has an open TODO to rerun the MC under Jeffreys. Today the
  prior-transform builders, Jeffreys log-term, and prior bounds all
  live inside `jeans_inference.py`; pulling them out lets new prior
  variants be added without touching the inference driver.

- **`ingest/multi_epoch.py` + `ingest/combiners/`** (future, not yet
  implemented) handle the per-epoch → per-star step that feeds Jeans
  analysis. Adapters in `path_b_adapters/` produce per-epoch arrays
  (one row per (star, epoch) measurement); the combiner stage
  produces the per-star Jeans-ready schema:

  ```
  star_id, v_bar_kms, sigma_vbar_kms, n_epoch, var_flag
  ```

  `multi_epoch.py` holds the dataset-agnostic primitives:

  - **IVW combine** with per-instrument zero-point offsets applied
    *before* the weighted mean, and per-instrument systematic error
    floor added in quadrature *after*:
    `v̄ = Σ vᵢ/σᵢ² / Σ 1/σᵢ²`,
    `σ_v̄ = (Σ 1/σᵢ²)^(−1/2) ⊕ σ_sys`.
  - **Variability flag** via `χ² = Σ (vᵢ − v̄)² / σᵢ²` against a
    p-value threshold (e.g. `p < 0.01`); single-epoch stars are left
    unflagged.

  `combiners/<dataset>.py` holds the per-dataset handler that knows
  the survey's zero-point offsets, error floor (e.g. 1.1 km/s for
  DEIMOS), variability threshold, and any dataset-specific quality
  cuts (S/N, membership probability, sky-line proximity). Handlers
  are organized **per dataset, not per dwarf**, since the combine
  procedure is a property of the survey/instrument. Each handler
  produces the same per-star schema, so downstream Jeans code is
  dataset-agnostic.

- **`docs/plan/` keeps its current role** as the living markdown
  spec set (`pipeline_overview.md`, `stage1.md`, `stage2.md`,
  `stage3.md`, `data_sources.md`, `jeffreys_jeans_derivation.md`,
  `uncertainty_conventions.md`). New edits go there, **not** into
  `docs/original-plan/`, which remains a frozen reference snapshot.

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

## Migration ordering

Each step is a separate PR. Each step leaves the repo importable and
the existing Segue 1 testbed runnable. No step bundles a behavioral
change with a layout change.

1. **Land this `ARCHITECTURE.md`.**
2. Add `pyproject.toml`. Create empty `src/dwarfjeans/` package with
   only `__init__.py`. Install in editable mode.
3. Promote `data_ingest/` → `src/dwarfjeans/ingest/`. `git mv` plus
   import-path fixups; adapter logic unchanged.
4a. **Pure `git mv`** of pipeline modules from `docs/plan/*.py` →
    `src/dwarfjeans/{jeans,jd,mocks}/` with import-path fixups only.
    Before the `git mv`, clean any `docs/plan/.ipynb_checkpoints/`
    copies so checkpoint shadows don't dangle.
    `docs/plan/segue1_test.md` and `docs/plan/README_mc_test.md`
    stay put as living specs. **No factoring**: `priors.py` is not
    extracted in this step; `jeans_inference.py` retains its current
    prior-transform builders and Jeffreys log-term inline. This step
    must produce identical posteriors to pre-rename.
4b. **Extract `jeans/priors.py`** from `jeans_inference.py`: pull
    `make_prior_transform`, `make_prior_transform_with_nuisances`,
    `_jeffreys_log_term`, and the prior-bound constants into the new
    module; have `inference.py` import them. Verify by rerunning the
    Segue 1 testbed and confirming posterior parity (chains agree to
    sampling-noise floor) before declaring done.
5. Move tests **and update their output paths in the same commit**
   so the testbed remains runnable. Existing unit tests →
   `tests/unit/`. Pipeline tests `run_segue1.py`,
   `run_ufd_population.py`, and `analyze_asimov.py` →
   `tests/integration/`. Update each test's output directory to
   `results/tests/<test_name>/` so writes land in the new location.
6. Move existing `Segue1_test/` run outputs (`*.npz`, `*.png`,
   `*.csv`, `baseline_logflat/`) → `results/tests/segue1/`. Add
   `results/` to `.gitignore`. Retire the `Segue1_test/` directory
   (including the one-shot diagnostic
   `compare_pace_vs_bpr08.py`, which has served its purpose).
7. Create `docs/writeup/pipeline.tex` skeleton (single document,
   sections per stage) and check in the compiled PDF.
8. Add CI under `.github/workflows/` running `pytest tests/unit` and
   the LaTeX build on each PR.

Out-of-band feature work (not numbered above because it is new code,
not migration):

- **Multi-epoch combiner.** Implement `ingest/multi_epoch.py` (IVW
  + χ² primitives) and per-dataset handlers under
  `ingest/combiners/`. Required before Stage 1 inference can be run
  on multi-epoch Path B galaxies; until then those galaxies have
  only per-epoch arrays in `data/star_catalogs/`.

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
