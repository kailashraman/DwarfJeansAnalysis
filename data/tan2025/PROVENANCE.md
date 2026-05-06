# Tan et al. 2025 — Leo VI discovery + Keck/DEIMOS spectroscopy

## Source

- Paper: Tan et al. 2025, ApJ 979, 176 — "A Pride of Satellites in the Constellation Leo? Discovery of the Leo VI Milky Way Satellite Ultra-faint Dwarf Galaxy with DELVE Early Data Release 3"
- ADS bibcode: `2025ApJ...979..176T`
- Article: https://iopscience.iop.org/article/10.3847/1538-4357/ad9b0c
- Not on VizieR; user downloaded the ASCII Table 2 from iopscience.

## Files

- `apjad9b0ct2_ascii.txt` — **Table 2** ("Properties of Spectroscopically Confirmed Member Stars and Candidate Member Stars of Leo VI Ordered by Decreasing Keck/DEIMOS Spectrum Signal-to-noise Ratio"). Tab-separated ASCII, 13 stars. Columns: `Star Name` (Gaia DR3 source_id or Tan+25 internal name), decimal-deg `R.A.`, `Decl.`, `g_0`, `r_0`, `S/N`, `v_hel` (km/s, formatted `<v> +or- <err>`), `Sigma EW CaT`, `[Fe/H]`, `Type` (RGB/BHB).
- `checksums.sha256`.

The paper-provided footnote distinguishes the two sub-sections by velocity precision: **confirmed members** have ε_vhel < 10 km/s (top 9 rows), **candidate members** have ε_vhel > 10 km/s (bottom 4 rows). The file does **not** include a section header or blank line between the two; the adapter splits on the published ε_vhel < 10 km/s rule.

## Adapter encoding choice

- p = 1 for the 9 confirmed members (`ε_vhel < 10 km/s`).
- p = 0 for the 4 candidate members (`ε_vhel > 10 km/s`).

`member_flag` column carries `M` / `C` so downstream sample selection can opt back in to the candidates if desired.

Granularity: per-star. Velocity frame: heliocentric.

## Paper-reported Leo VI numbers (cross-check targets)

- 9 confirmed members; `<v_hel> ≈ 171 km/s`, `σ_naive ≈ 4.2 km/s` over confirmed.
- LVDB v_sys = 170.03 km/s, σ = 2.85 km/s (deconvolved; the paper's own ML estimate accounts for per-star ε_vhel).
