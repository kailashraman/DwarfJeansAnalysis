# Hansen et al. 2024 — Tucana V kinematics + abundance analysis

## Source

- Paper: Hansen, Marshall, Frebel, Ji, et al. 2024, ApJ 968, 21 — "Chemical Diversity on Small Scales: Abundance Analysis of the Tucana V Ultrafaint Dwarf Galaxy"
- ADS bibcode: `2024ApJ...968...21H`
- Article: https://iopscience.iop.org/article/10.3847/1538-4357/ad3a52
- Not on VizieR; user downloaded the ASCII Table 1 from iopscience.

## Files

- `apjad3a52t1_ascii.txt` — **Table 1** ("Observing Log"). Tab-separated ASCII. 3 unique Tuc V member stars (Tuc V-1, -2, -3), 17 per-epoch rows total across MIKE and IMACS. Columns: Gaia DR3 ID, sexagesimal R.A. / Decl., DECam g/r/i/z₀, `MJD`, `t_exp`, two S/N entries, `V_hel +or- σ` (km/s), `Instrument`. The first row of each star carries the Gaia ID and coords; continuation rows have `(Tuc V-N)` or empty first cell and `cdots` for static columns.
- `checksums.sha256`.

The earlier ingest attempt used Table 2 ("Stellar Parameters") which has only T_eff / log g / ξ / [Fe/H]; ξ there is microturbulence, not heliocentric velocity. Table 1 is the canonical Stage 0b input.

## Format quirks handled by the adapter

- RA cells use a non-standard `hh:mm:ss:ff` separator on Tuc V-1 and Tuc V-2 (e.g. `23:37:07:09`), where the last colon should be a decimal point (`23:37:07.09`). Tuc V-3 already uses `hh:mm:ss.ff`. The adapter normalizes the four-colon form to three-colon-plus-decimal before SkyCoord parsing.
- Continuation rows for additional epochs of the same star carry `cdots` placeholders for coords, mags, and S/N, and either `(Tuc V-N)` or an empty cell in the first column. The adapter inherits the Gaia ID and decimal-deg coords from the most recent header row.
- Membership: paper publishes a member list (3 stars), so all 17 epochs receive `p = 1` per the data_sources.md missing-probability default.

Granularity: per-epoch. Velocity frame: heliocentric.

## Paper-reported Tucana V numbers (cross-check targets)

- 3 unique member stars (Tuc V-1 large multi-epoch v_hel range — likely binary; Tuc V-2 and Tuc V-3 stable to within ~1 km/s).
- LVDB v_sys = -34.7 km/s, σ = 1.2 km/s (the paper's binary-aware estimate; expect our naive per-star std to differ when Tuc V-1 is included).
