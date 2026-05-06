# Heiger et al. 2024 — Centaurus I spectroscopy

## Source

- Paper: Heiger, Li, Pace, Simon, Ji, Chiti et al. 2024, ApJ 961, 234 — "Reading Between the (Spectral) Lines: Magellan/IMACS spectroscopy of the Ultra-faint Dwarf Galaxies Eridanus IV and Centaurus I"
- ADS bibcode: `2024ApJ...961..234H`
- Article: https://iopscience.iop.org/article/10.3847/1538-4357/ad0cf7
- Not on VizieR; user downloaded the MRT for Table 4 from iopscience.

## Files

- `apjad0cf7t4_mrt.txt` — **Table 4** in CDS MRT byte-by-byte format. 62 rows, full Cen I observed sample. Per-mask velocities (`vhel-1-58881.3`, `vhel-1-59608.6`, `vhel-2-59410.0`, `vhel-3-59609.3`) plus a combined `vhel`/`e_vhel`. Per-row `Member` flag (`1` = confirmed member, `0` = non-member) and `Comments` (RGB / HB / 999999 sentinel).
- `checksums.sha256`.

Granularity: per-star (combined-fit `vhel` is the canonical column). Velocity frame: heliocentric.

## Paper-reported Centaurus I numbers (cross-check targets)

- Member rows (`Member == 1`): 34. `<v_hel>_mem ≈ 44.8 km/s`, `σ_naive ≈ 4.08 km/s`.
- LVDB v_sys = 44.8 km/s, σ = 4.2 km/s.
