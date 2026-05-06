# Koposov et al. 2015 — Horologium I spectroscopy

## Source

- Paper: Koposov, Casey, Belokurov et al. 2015, ApJ 811, 62 — "Kinematics and chemistry of recently discovered Reticulum 2 and Horologium 1 dwarf galaxies"
- ADS bibcode: `2015ApJ...811...62K`
- Article: https://iopscience.iop.org/article/10.1088/0004-637X/811/1/62
- Not on VizieR; user downloaded the ASCII Table 2 from iopscience.

## Files

- `apj519209t2_ascii.txt` — **Table 2** ("Positions, Velocities, Stellar Parameters, and Membership for Reticulum 2 and Horologium 1 Candidates"). Tab-separated ASCII, 43 stars total, stacked in two sub-tables: Reticulum 2 (lines 9–33, 25 stars) and Horologium 1 (lines 36–53, 18 stars). Columns: `Object`, `alpha(J2000)` (decimal deg), `delta(J2000)` (decimal deg, wrapped in `$…$` LaTeX math mode), `g`, `V_hel` (km/s), `T_eff`, `log g`, `[Fe/H]`, `[alpha/Fe]`, χ²_red, `Member?`.
- `checksums.sha256`.

The adapter slices the **Horologium 1** sub-block (between the section headers `Horologium 1` and end-of-file).

## Format quirks handled by the adapter

- Decimal-deg `delta(J2000)` cells are wrapped in `$…$` LaTeX math mode (e.g. `$-54.1260$`); the adapter strips `$` before float conversion.
- `V_hel` cells appear in two forms:
  - the bare ASCII `value +or- err` (most rows); and
  - LaTeX `$value\pm err$` (used for two RGB-MW interlopers in the Reti 2 block). The adapter accepts both.
- `Member?` values: `Yes`, `Yes?`, `cdots`, blank → encoded as p=1 for `Yes` / `Yes?`, else 0.

Granularity: per-star. Velocity frame: heliocentric.

## Paper-reported Horologium I numbers (cross-check targets)

- N_members = 5 (Horo 9, 10, 11, 15, 17 — `Member? == Yes`).
- σ_los = 4.9 km/s (paper text, naive over the 5 members).
- LVDB v_sys = 112.8 km/s, σ = 4.9 km/s.
