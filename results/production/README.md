# Production Posteriors — Zenodo Deposit

The full posterior chains from the 39-galaxy production sweep are not stored
in this git repository. They are archived on Zenodo with a permanent DOI.

## Deposit

- **DOI:** _TBD — fill in after Zenodo upload_
- **URL:** _TBD_
- **Version:** _TBD (matches git tag / commit)_
- **Size:** ~930 MB (312 chains: 39 galaxies × 8 configurations — the
  `jeffreys`, `loguniform`, `satgen` and `satgen_box` priors, plus
  `satgen_shmr` run once per SHMR (`fattahi18`, `moster18`,
  `danieli23_const`, `kim24`), which land in `satgen_shmr_<shmr>/` leaves)

## Contents

Each chain is stored as a `posterior_samples.npz` under
`<lvdb_key>/<prior>/`, alongside its `summary.csv`, `audit.json`, and
`run.log`. See `ARCHITECTURE.md` for the output-path convention.

`posterior_samples.npz` fields: _TBD — document keys (samples, logl,
weights, …) and shapes._

## Reproduction

The chains can be regenerated from the tracked code and input catalogs
(`data/star_catalogs/`) using:

`--prior` is one of `uniform`, `loguniform`, `jeffreys`, `satgen`,
`satgen_box`, `satgen_shmr`; `--shmr` is required iff `--prior satgen_shmr`.

```bash
# Single galaxy
python scripts/run_production.py --lvdb-key <key> --prior <prior> \
    [--shmr <shmr>] --nlive 1500 --dlogz 0.05

# Full sweep (SLURM array; invoke through bash, not sbatch)
bash scripts/submit_batch.sh --cohort {classical|ufd} --prior <prior> [--shmr <shmr>]
```

_TBD — pin commit SHA, runtime envelope, and seed handling for exact
reproducibility._

## Fetching from Zenodo

_TBD — fetch command or script._

## Citation

These posteriors support a two-part study, *Semi-analytic Inference of
Satellite Densities in the Cold Dark Matter Model* (Raman, Folsom,
Kaplinghat, Lisanti & Safdi, 2026):

- Part I — Comparison to Ultra-faint Dwarf Kinematics, arXiv:2607.27316
- Part II — Implications for Dark Matter Indirect Detection Constraints,
  arXiv:2607.27326

_TBD — Zenodo citation block, once the deposit is made._
