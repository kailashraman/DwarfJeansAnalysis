# Li et al. 2017 — Eridanus II spectroscopy

## Source

- Paper: Li, Simon, Drlica-Wagner et al. 2017, ApJ 838, 8 — "Farthest Neighbor: The Distant Milky Way Satellite Eridanus II"
- ADS bibcode: `2017ApJ...838....8L`
- Article: https://iopscience.iop.org/article/10.3847/1538-4357/aa6113
- Not on VizieR; user downloaded the ASCII Table 2 from iopscience.

## Files

- `apjaa6113t2_ascii.txt` — **Table 2** ("Velocity and Metallicity Measurements for Eridanus II"). Tab-separated ASCII. 54 unique target stars, 93 per-epoch rows total (most stars observed in the November 2015 + October 2015 IMACS runs, MJD = 57345.7 / 57312.8). Columns: `ID`, `MJD`, `RA`, `Decl.`, `g`, `r`, `S/N`, `v` (km/s, `<v> +or- <err>`), `EW`, `[Fe/H]`, `MEM` (`0` = non-member, `1` = member).
- `checksums.sha256`.

The first row of each star carries the `ID`, coords, mags, and `MEM`. Subsequent epochs of the same star are **continuation rows** with an empty `ID` cell — only `MJD`, `S/N`, `v`, `EW`, `[Fe/H]` are repopulated. The adapter inherits ID / RA / Dec / MEM from the most recent header row.

Granularity: per-epoch (each MJD row is one entry). Velocity frame: heliocentric.

## Paper-reported Eridanus II numbers (cross-check targets)

- N_unique_stars = 54.
- N_member_stars = 28 (`MEM == 1`).
- σ_los = 6.9 km/s (paper text); LVDB v_sys = 75.6 km/s, σ = 6.9 km/s.
