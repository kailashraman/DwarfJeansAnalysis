# Chiti et al. 2022 — Grus I spectroscopy

## Source

- Paper: Chiti, Frebel, Mardini, Ji, Andales, Hattori, Karakas, Norris, Roederer 2022, ApJ 939, 41 — "Detailed Properties of the Ultrafaint Dwarf Galaxy Grus I"
- ADS bibcode: `2022ApJ...939...41C`
- Article: https://iopscience.iop.org/article/10.3847/1538-4357/ac96ed
- Not on VizieR; user downloaded the ASCII Table 2 from iopscience.

## Files

- `apjac96edt2_ascii.txt` — **Table 2** ("Velocity Measurements for all Stars"). Tab-separated ASCII. 70 unique target stars, 80 per-epoch rows total (some stars observed at up to 3 MJDs). Columns: `ID`, `MJD`, `R.A.`, `Decl.`, `g`, `r`, `S/N`, `v` (km/s, `<v> +or- <err>`), `MEM`. Continuation rows for additional epochs of the same star have an empty `ID` cell.
- `checksums.sha256`.

`MEM` flag values per Note `^c` of the table:
- `M` — confirmed member (8 stars).
- `NM` — non-member (58 stars).
- `CM` — candidate member (consistent radial velocity, no derived metallicity; 4 stars).

The adapter encodes `M -> p=1`, both `NM` and `CM -> p=0`, and preserves the original flag in `member_flag` so downstream selection can opt back in to candidates.

Granularity: per-epoch (each MJD row is one entry). Velocity frame: heliocentric.

## Paper-reported Grus I numbers (cross-check targets)

- N_unique_stars = 70.
- N_member_stars (`MEM == M`) = 8.
- LVDB v_sys = -143.5 km/s, σ = 2.5 km/s.
