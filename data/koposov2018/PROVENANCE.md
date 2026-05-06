# Koposov et al. 2018 — kinematics of Hydrus I (and other UFDs)

## Source

- **Source paper:** Koposov et al. 2018, MNRAS 479, 5343 ("Snake in the Clouds: A new nearby dwarf galaxy in the Magellanic bridge")
- **ADS bibcode:** `2018MNRAS.479.5343K`
- **VizieR catalog:** [J/MNRAS/479/5343](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/MNRAS/479/5343)

## Files

- `table2.csv` — `J/MNRAS/479/5343/table2`, 139 rows. Per-star spectroscopic catalog of stars in the Hydrus I field. Columns: `ID`, `RAJ2000`, `DEJ2000` (decimal degrees), `S/N`, `HRV`, `e_HRV` (heliocentric km/s), `Teff`, `e_Teff`, `loggmean`, `e_loggmean`, `[Fe/H]`, `e_[Fe/H]`, `gmag`, `rmag`, `logodds`, `e_logodds`. Membership encoded as the `logodds` column (log-odds of being a Hydrus I member from Koposov+18's Bayesian mixture); convert to p via sigmoid `p = 1/(1+exp(-logodds))`. 3 rows have masked logodds.
- `checksums.sha256`.

Per-star granularity. Velocity frame: heliocentric (`HRV`).

## Acquisition

- **Date staged:** 2026-05-05
- **Method:** `astroquery.vizier.Vizier(columns=["**"]).get_catalogs("J/MNRAS/479/5343")`.
