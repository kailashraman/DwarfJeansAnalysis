# LVDB v1.0.5 — Local Volume Database

## Source

- **Project:** Local Volume Database (LVDB), maintained by Andrew Pace
- **Version:** v1.0.5 (released 2025-05-20)
- **Upstream repo:** https://github.com/apace7/local_volume_database
- **Zenodo DOI:** https://doi.org/10.5281/zenodo.15476348
- **GitHub release:** https://github.com/apace7/local_volume_database/releases/tag/v1.0.5
- **License:** CC0 (public domain)
- **Overview paper:** Pace 2025, OJAp 8, 142 ([arXiv:2411.07424](https://arxiv.org/abs/2411.07424))

## Files

- `comb_all.csv` — combined catalog (canonical Stage 0a input)
- `checksums.sha256` — SHA-256 of `comb_all.csv`

## Acquisition

- **Date staged:** 2026-05-05
- **Staged by:** Kailash Raman
- **Original repo location:** `Segue1_test/data/lvdb_v1.0.5_comb_all.csv`
  - That copy was originally fetched at runtime by `Segue1_test/run_segue1.py:fetch_lvdb()` from
    `https://github.com/teaghan/LVDB/releases/download/v1.0.5/comb_all.csv` (a mirror of the upstream LVDB v1.0.5 release).
- **Renamed to:** `comb_all.csv` (canonical name; matches upstream filename in the v1.0.5 release).

## Read-only after staging

The pipeline reads from this folder and never writes back. Future LVDB version bumps live as sibling folders (e.g., `data/lvdb_v1.1.0/`); v1.0.5 stays pinned. Auto-fetch from GitHub at runtime is forbidden — Stage 0a fails loudly on missing or checksum-mismatched files.
