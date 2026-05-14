# Production Posteriors — Zenodo Deposit

The full posterior chains from the 39-galaxy production sweep are not stored
in this git repository. They are archived on Zenodo with a permanent DOI.

## Deposit

- **DOI:** _TBD — fill in after Zenodo upload_
- **URL:** _TBD_
- **Version:** _TBD (matches git tag / commit)_
- **Size:** ~240 MB (78 chains: 39 galaxies × {jeffreys, loguniform} priors)

## Contents

Each chain is stored as a `posterior_samples.npz` under
`<lvdb_key>/<prior>/`, alongside its `summary.csv`, `audit.json`, and
`run.log`. See `ARCHITECTURE.md` for the output-path convention.

`posterior_samples.npz` fields: _TBD — document keys (samples, logl,
weights, …) and shapes._

## Reproduction

The chains can be regenerated from the tracked code and input catalogs
(`data/star_catalogs/`) using:

```bash
# Single galaxy
python scripts/run_production.py --lvdb-key <key> --prior {jeffreys,loguniform}

# Full sweep
python scripts/submit_batch.py --cohort {classical,ufd} --prior <prior>
```

_TBD — pin commit SHA, runtime envelope, and seed handling for exact
reproducibility._

## Fetching from Zenodo

_TBD — fetch command or script._

## Citation

_TBD — Zenodo citation block + reference to the accompanying paper._
