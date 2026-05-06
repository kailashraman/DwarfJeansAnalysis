# Walker et al. 2009 — radial velocities of 4 dSph galaxies

## Source

- **Source paper:** Walker et al. 2009, AJ 137, 3100 ("Stellar velocities in the Carina, Fornax, Sculptor, and Sextans dSph galaxies: data from the Magellan/MMFS survey")
- **ADS bibcode:** `2009AJ....137.3100W`
- **VizieR catalog:** [J/AJ/137/3100](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/AJ/137/3100)

## Files

- `stars.csv` — per-star summary table (`J/AJ/137/3100/stars`), 7103 rows. **Canonical Stage 0b input** per data_sources.md (per-star preferred over per-epoch when both are published). Columns: `Target` (per-star ID, e.g. `Car-0001`), `o_Target` (occurrence count), `RAJ2000`, `DEJ2000`, `Vmag`, `V-I`, `Mmb` (binary membership flag), `<HV>` (epoch-averaged heliocentric velocity, km/s), `e_<HV>`, `<SigMg>`, `e_<SigMg>`, `Simbad`.
- `tables.csv` — per-epoch table (`J/AJ/137/3100/tables`), 8855 rows. Kept for traceability per data_sources.md "When both are published, the per-star table is canonical for our purposes — note that the per-epoch table exists for traceability, but only ingest the per-star table."
- `checksums.sha256` — SHA-256 of the two CSVs.

Galaxy identification is by the `Car-`, `For-`, `Scl-`, `Sex-` prefix on the `Target` column. Carina has 1982 per-star rows total; 441 marked as members (`Mmb == 1`).

Velocity frame: heliocentric (`<HV>`).

## Acquisition

- **Date staged:** 2026-05-05
- **Staged by:** Kailash Raman (via Claude Code)
- **Method:** `astroquery.vizier.Vizier.get_catalogs("J/AJ/137/3100")`, written to ECSV with `astropy.table.Table.write(format="ascii.ecsv")`. The CSV format is ECSV (units + metadata in the leading commented header) for round-trippable astropy reads.

## Read-only after staging

Stage 0b reads from this folder and never writes back. Auto-fetch from VizieR at runtime is forbidden — the staged copy is the source of truth, version-pinned by checksum.
