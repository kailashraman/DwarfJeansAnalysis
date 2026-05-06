# Ji et al. 2021 — Antlia II + Crater II spectroscopy

## Source

- Paper: Ji et al. 2021, ApJ 921, 32 ("The Crater II–Antlia 2 Pair: kinematics and chemistry")
- ADS bibcode: `2021ApJ...921...32J`
- VizieR: [J/ApJ/921/32](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/ApJ/921/32)

## Files

- `table4.csv` — **Antlia II**, 508 per-star rows. Columns: Gaia (DR2 source_id), RA_ICRS, DE_ICRS (decimal deg), Gmag, pmRA, pmDE, HRV (heliocentric km/s), e_HRV, [Fe/H]1, e_[Fe/H]1, [Fe/H]2, e_[Fe/H]2, Bin (binary flag), Mm (continuous membership probability), S/N.
- `table5.csv` — **Crater II**, 207 per-star rows; same schema.
- `checksums.sha256`.

Granularity: per-star (n_unique Gaia == n_rows for both tables). Membership: continuous `Mm ∈ [0, 1]` from Ji+21's Bayesian mixture analysis. Velocity frame: heliocentric (`HRV`).
