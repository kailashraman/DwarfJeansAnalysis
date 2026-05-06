# Li et al. 2018 — Carina II and Carina III spectroscopy

## Source

- **Source paper:** Li et al. 2018, ApJ 857, 145
- **ADS bibcode:** `2018ApJ...857..145L`
- **VizieR catalog:** [J/ApJ/857/145](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/ApJ/857/145)

## Files

- `table2.csv` — galaxy summary (2 rows: Carina II, Carina III). Reference for paper-published values per galaxy. Not directly ingested.
- `table4.csv` — per-epoch star catalog, **407 rows / 283 unique MagLiteS IDs**. Per-epoch granularity (the same star may appear at multiple `MJD` / `Inst` rows).
  - Columns: `MagLiteS` (per-star ID, J2000 sexagesimal-encoded), `RAJ2000`, `DEJ2000`, `g0mag`, `r0mag`, `Inst` (instrument/mask), `MJD` (epoch), `S/N`, `HRV`, `e_HRV` (heliocentric velocity, km/s), `EW`, `e_EW`, `[Fe/H]`, `e_[Fe/H]`, `Mm` (multi-state membership: 0 = non-member, 2 = Carina II member, 3 = Carina III member).
- `checksums.sha256`.

Velocity frame: heliocentric (column `HRV`).
Membership: tri-state `Mm ∈ {0, 2, 3}`. The paper uses the same per-epoch table for both galaxies; ambiguous candidates are non-members of both. We ingest **all 407 rows into each per-galaxy npz** (raw-data-only) with `p` encoded per the target galaxy: for `carina_2.npz` `p=1` iff `Mm==2`, for `carina_3.npz` `p=1` iff `Mm==3`. R_i is computed from the LVDB center of the target galaxy.

## Acquisition

- **Date staged:** 2026-05-05
- **Method:** `astroquery.vizier.Vizier(columns=["**"]).get_catalogs("J/ApJ/857/145")`.
