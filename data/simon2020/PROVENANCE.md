# Simon et al. 2020 — Spectroscopy of Tucana IV (and Grus II, Tucana V)

## Source

- Paper: Simon et al. 2020, ApJ 892, 137
- ADS bibcode: `2020ApJ...892..137S`
- VizieR: [J/ApJ/892/137](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/ApJ/892/137)

## Files

- `table3.csv` — 515 rows, per-epoch star catalog covering 3 galaxies (`Gal ∈ {GruII, TucIV, TucV}`). 223 rows for Tucana IV (132 unique DES IDs, multi-epoch). Columns: `Gal`, `DES` (per-star ID), `Nrv` (number of epochs combined), `RAJ2000`, `DEJ2000`, `gmag`, `rmag`, `MJD` (epoch), `S/N`, `HRV` (heliocentric, km/s), `e_HRV`, `EW`, `e_EW`, `[Fe/H]`, `e_[Fe/H]`, `Mm` (binary 0/1).
- `table2.csv` — 210 rows, separate per-epoch table without `Gal`/`Mm` columns (raw observations); kept for traceability but not ingested.
- `checksums.sha256`.

For our 39-galaxy study, only Tucana IV is served by Simon 2020 (Grus II and Tucana V have different `ref_vlos` in LVDB).

Granularity: per-epoch. Membership: binary `Mm`.
