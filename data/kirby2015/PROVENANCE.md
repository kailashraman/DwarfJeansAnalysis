# Kirby et al. 2015 — Pisces II spectroscopic confirmation

## Source

- Paper: Kirby, Simon, & Cohen 2015, ApJ 810, 56 — "Spectroscopic Confirmation of the Dwarf Galaxies Hydra II and Pisces II and the Globular Cluster Laevens 1"
- ADS bibcode: `2015ApJ...810...56K`
- Article: https://iopscience.iop.org/article/10.1088/0004-637X/810/1/56
- Not on VizieR (verified 2026-05-05). MRT URL pattern on iopscience is bot-walled; user downloaded the ASCII Table 2 in a browser session.

## Files

- `apj518514t2_ascii.txt` — **Table 2** ("Target List"), tab-separated ASCII, 60 stars total spanning three systems in stacked sub-tables: Hydra II (lines 9–39, 31 stars), Pisces II (lines 42–54, 13 stars), Laevens 1 (lines 57–70, 14 stars). Columns: `ID`, sexagesimal `R.A. (J2000)` / `decl. (J2000)`, `g_0`, `(g-r)_0`, `(g-I)_0`, `S/N`, `v_helio` (heliocentric km/s, formatted `<v> +or- <err>`), `Member?` (`Y`/`N`), `T_eff`, `log g`, `[Fe/H]`. Missing values are ` ... `.
- `checksums.sha256`.

The adapter slices to the **Pisces II** block (between the `Pisces II` and `Laevens 1` section headers), parses sexagesimal coords to decimal degrees, splits `v_helio` on `+or-`, and maps `Member?` `Y`→`p=1`, `N`→`p=0`.

Granularity: per-star (Table 2 reports a single co-added velocity per star). Velocity frame: heliocentric.

## Paper-reported Pisces II numbers (cross-check targets)

- N_members = 7 (stars marked `Y` in Pisces II block: 9004, 9833, 10694, 12924, 13387, 13560, 14179).
- σ_los = 5.4 km/s (naive sample std over the 7 Y stars; paper text reports σ = 5.4 km/s).
- `<v_helio>` ≈ -226 km/s.
