# Walker et al. 2015 — Magellan/M2FS spectroscopy of Reticulum 2

## Source

- **Source paper:** Walker et al. 2015, ApJ 808, 108
- **ADS bibcode:** `2015ApJ...808..108W`
- **VizieR catalog:** [J/ApJ/808/108](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/ApJ/808/108)

## Files

- `table1.csv` — `J/ApJ/808/108/table1`, 38 rows. M2FS per-star spectroscopy of Reticulum 2 targets. Columns include `Ret2` (per-star ID), `RAJ2000`, `DEJ2000` (sexagesimal), `R` (arcmin from center), `Vlos`, `e_Vlos` (heliocentric, km/s), `[Fe/H]`, `e_[Fe/H]`, `Mm?` (Y/N membership flag), plus skewness/kurtosis posteriors for V/Teff/logg/[Fe/H].
- `checksums.sha256`.

Per-star granularity (one row per unique star). Velocity frame: heliocentric / "solar rest frame" (paper §2). Membership: binary `Mm?` flag from Walker+15's Bayesian mixture analysis (Y for 18 stars; paper text reports 17 confirmed members — one ambiguous).

## Acquisition

- **Date staged:** 2026-05-05
- **Method:** `astroquery.vizier.Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJ/808/108")` — `columns=["**"]` is critical because the default fetch drops `e_Vlos` and other error columns.
- **Staged by:** Kailash Raman

## Read-only after staging
