# Chiti et al. 2023 — Tucana II spectroscopic compilation

## Source

- Paper: Chiti et al. 2023, AJ 165, 55
- ADS bibcode: `2023AJ....165...55C`
- VizieR: [J/AJ/165/55](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/AJ/165/55)

## Files

- `table6.csv` — **kinematic compilation**, 60 rows, **per-epoch**, **19 unique** Tuc II member stars. Columns: `Name`, `RAJ2000`/`DEJ2000` (decimal deg), `MJD`, `Inst` (M2FS/IMACS/MIKE/MagE), `RVel` (heliocentric km/s), `f_RVel` (`'b'` = binary candidate, else blank), `e_RVel`, `Ref` (1=this work, 2=Walker+16, 3=Chiti+18/21, 4=Ji+16, 5=Chiti+22). This is the **canonical Stage 0b input**.
- `stars.csv` — 5-star high-res sample (this work's own MIKE observations); per-star summary; kept for traceability but only `table6` is ingested.
- `checksums.sha256`.

Granularity: per-epoch. Every row in `table6` is a confirmed Tuc II member (the paper compiles literature kinematics for known members), so we apply `p_i = 1` per the data_sources.md "missing-probability default" rule (the paper publishes a member list, not a continuous probability column). Velocity frame: heliocentric (column documented as `Heliocentric radial velocity`).

The `f_RVel` flag (`'b'` for binary candidates) is carried as auxiliary so downstream sample selection can decide whether to demote suspected binaries.
