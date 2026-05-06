# Bruce et al. 2023 — Aquarius II spectroscopy

## Source

- Paper: Bruce, Li, Pace, Heiger, Song, Simon 2023, ApJ 950, 167 — "Spectroscopic analysis of Milky Way outer halo satellites: Aquarius II and Bootes II"
- ADS bibcode: `2023ApJ...950..167B`
- Article: https://iopscience.iop.org/article/10.3847/1538-4357/acc943
- Not on VizieR; user downloaded the MRT for Table 3 (Aquarius II Observed Spectra) from iopscience.

## Files

- `apjacc943t3_mrt.txt` — **Table 3** in CDS MRT byte-by-byte format. 12 rows, one per Aquarius II target. Columns include Gaia DR3 source_id, decimal degree coords, dereddened LS DR9 g/r magnitudes, two-epoch SNR, two-epoch CaT EWs, combined-fit `Vhelio` / `e_Vhelio` (km/s, heliocentric), `[Fe/H]`, and `Mem` (`M` = confirmed member, `NM` = confirmed non-member).
- `checksums.sha256`.

Granularity: per-star (combined-epoch fit). Velocity frame: heliocentric.

## Paper-reported Aquarius II numbers (cross-check targets)

- Member rows: 8 (`Mem == 'M'`).
- Member `<v_hel> ≈ -65 km/s`, `σ_naive ≈ 4.6 km/s`. LVDB v_sys = -65.3, σ = 4.7 km/s.
