# Geha 2026 Paper I — DEIMOS satellite-galaxy stellar archive

## Source

- **Source paper (Paper I):** Geha et al. 2026, "DEIMOS Stellar Archive: Paper I" (arXiv:2602.10200)
- **Companion paper (Paper II):** Geha et al. 2026, "Paper II — integrated properties" (arXiv:2602.10202)
- **Upstream URL:** https://geha-group.github.io/deimos/ (Dropbox-hosted CSV/FITS)
- **Release stamp:** `20260110` (2026-01-10)

## Files

- `table3A_20260110.csv` — Paper I Table 3A: per-star catalog, 22,340 rows × 16 columns
- `checksums.sha256` — SHA-256 of `table3A_20260110.csv`

## Header (verified 2026-05-05)

```
Galaxy, RA, DEC, r, gr, nmask, t_exp, SN, v, verr, CaT, CaTerr, FeH, FeH_err, Var, Pmem
```

`Galaxy` is the system identifier and uses the Paper II Table A1 abbreviations (`Boo1`, `CB`, `CVn1`, `Seg1`, `UMa2`, …). `Pmem` is the published membership probability; `Var` is a boolean velocity-variability flag.

Note: earlier drafts of `docs/plan/data_sources.md` referenced a `Pmem_novar` column that does not exist in this release — the on-disk file has a single `Pmem` column plus the `Var` flag. Resolved in `data_sources.md` 2026-05-05 changelog.

## Acquisition

- **Date staged:** 2026-05-05
- **Staged by:** Kailash Raman
- **Original repo location:** `Segue1_test/data/table3A_20260110.csv`

## Read-only after staging

Stage 0b reads from this folder and never writes back. Auto-fetch from the Dropbox URL at runtime is forbidden — the Dropbox link contains a session token that may expire, and re-fetch would violate the "stage once, version-pin, never re-fetch" pattern.
