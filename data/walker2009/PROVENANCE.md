# Walker et al. 2009 — radial velocities of 4 dSph galaxies

## Source

- **Source paper:** Walker et al. 2009, AJ 137, 3100 ("Stellar velocities in the Carina, Fornax, Sculptor, and Sextans dSph galaxies: data from the Magellan/MMFS survey")
- **ADS bibcode:** `2009AJ....137.3100W`
- **VizieR catalog:** [J/AJ/137/3100](https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=J/AJ/137/3100)

## Files

- `stars.csv` — per-star summary table (`J/AJ/137/3100/stars`), 7103 rows. **Canonical Stage 0b input** per data_sources.md (per-star preferred over per-epoch when both are published). Columns: `Target` (per-star ID, e.g. `Car-0001`), `o_Target` (occurrence count), `RAJ2000`, `DEJ2000`, `Vmag`, `V-I`, `Mmb` (binary membership flag), `<HV>` (epoch-averaged heliocentric velocity, km/s), `e_<HV>`, `<SigMg>`, `e_<SigMg>`, `Simbad`.
- `tables.csv` — per-epoch table (`J/AJ/137/3100/tables`), 8855 rows. **Required at ingest** to recover σ_eps for Carina single-epoch stars (see "Hybrid σ_eps rule" below). Multi-epoch stars use the IVW-combined `e_<HV>` from `stars.csv`; single-epoch stars (`o_Target == 1`) have masked `e_<HV>` and pull `e_HV` from the matching per-spectrum row in this file.
- `checksums.sha256` — SHA-256 of the two CSVs.

Galaxy identification is by the `Car-`, `For-`, `Scl-`, `Sex-` prefix on the `Target` column. Carina has 1982 per-star rows total; 441 marked as members (`Mmb == 1`).

Velocity frame: heliocentric (`<HV>`).

## Acquisition

- **Date staged:** 2026-05-05
- **Staged by:** Kailash Raman
- **Method:** `astroquery.vizier.Vizier.get_catalogs("J/AJ/137/3100")`, written to ECSV with `astropy.table.Table.write(format="ascii.ecsv")`. The CSV format is ECSV (units + metadata in the leading commented header) for round-trippable astropy reads.

## Hybrid σ_eps rule (QA-sweep #2, 2026-05-06)

Walker+09's per-star averaged catalog (`stars.csv`) masks `e_<HV>` for
single-epoch stars (the IVW-combined error formula needs ≥2
measurements). For Carina that's 1619 of 1982 rows, including 269 of
441 strict members.

The walker2009 adapter resolves this without imputation:

- **Multi-epoch stars (`o_Target ≥ 2`):** use `stars.csv` `e_<HV>`
  verbatim — Walker's IVW-combined per-star error is canonical.
- **Single-epoch stars (`o_Target == 1`):** join on `Target` against
  `tables.csv` (per-spectrum) and pull the single matching row's
  `e_HV`. For a single-epoch star the IVW combination reduces to that
  one observation's error, so this is the per-star error Walker would
  have published had the IVW formula been applicable.

The npz carries an auxiliary `sigma_eps_source` column
(`"stars.<HV>_combined"` or `"tables.HV_per_spectrum"` per row) and an
`o_Target` column for audit. The adapter raises if any of the
consistency checks C1–C6 (see the QA-sweep #2 commit / plan file) fail.

Walker+09's abstract reports a median per-spectrum precision of
±2.1 km/s; the adapter does not assume this value, but uses the actual
per-spectrum errors. For the Carina single-epoch subset the median
`e_HV` is 2.9 km/s — broadly consistent with Walker's stated
precision (the single-epoch subset is biased toward fainter / lower-S/N
targets that were not re-observed).

## Read-only after staging

Stage 0b reads from this folder and never writes back. Auto-fetch from VizieR at runtime is forbidden — the staged copy is the source of truth, version-pinned by checksum.
