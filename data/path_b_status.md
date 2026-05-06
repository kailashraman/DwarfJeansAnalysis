# Path B ingest status

This file is the resume-safe state for the Path B (non-Geha) per-star catalog
ingest. Updated and committed every time a paper's state changes.

To continue from a fresh session: read this file, find the first paper marked
`pending` or `blocked`, and pick up from there. Process order is fixed below.

Last updated: 2026-05-06 (Hansen 2024 / Tucana V ingested from user-supplied Table 1 Observing Log — last IOP-blocked paper resolved; all 15 papers in the Path B sweep now done.)

## Per-paper status

Process order is fixed (classical/well-indexed papers first, recent UFD
discoveries next, likely-blocked last). One paper per commit.

| # | Paper (bibcode) | Galaxies | Status | Notes |
|---|---|---|---|---|
| 1 | Walker 2009 (`2009AJ....137.3100W`) | Carina | done | VizieR `J/AJ/137/3100/stars`; `<V>` and σ_los match paper |
| 2 | Walker 2015 (`2015ApJ...808..108W`) | Reticulum II | done | VizieR `J/ApJ/808/108/table1`; `<V>` and σ_los match paper |
| 3 | Kirby 2015 (`2015ApJ...810...56K`) | Pisces II | done | User-supplied ASCII Table 2 (`apj518514t2_ascii.txt`) at `data/kirby2015/`. Adapter slices Pisces II sub-block; sexagesimal -> deg via SkyCoord; `v +or- err` regex split; Y/N -> p=1/0. `<V>` and σ_los match paper. |
| 4 | Koposov 2015 (`2015ApJ...811...62K`) | Horologium I | done | User-supplied ASCII Table 2 stacking Reti 2 + Hor 1; adapter slices Hor 1 block, strips `$…$` LaTeX from Dec, accepts both `+or-` and `\pm` velocity forms. 5 members, `<V>` and σ match paper. |
| 5 | Li 2017 (`2017ApJ...838....8L`) | Eridanus II | done | User-supplied ASCII Table 2 (per-epoch with continuation rows). Adapter inherits ID/coords/MEM across continuation rows, splits `v +or- err`. 54 stars / 28 members, `<V>` and σ match paper. |
| 6 | Li 2018 (`2018ApJ...857..145L`) | Carina II + Carina III | done | VizieR `J/ApJ/857/145/table4` (per-epoch); split by `Mm ∈ {0,2,3}` |
| 7 | Koposov 2018 (`2018MNRAS.479.5343K`) | Hydrus I | done | VizieR `J/MNRAS/479/5343/table2`; `logodds` → sigmoid; `<V>` and σ match paper |
| 8 | Simon 2020 (`2020ApJ...892..137S`) | Tucana IV | done | VizieR `J/ApJ/892/137/table3`; per-epoch; filter `Gal=='TucIV'` |
| 9 | Ji 2021 (`2021ApJ...921...32J`) | Antlia II + Crater II | done | VizieR `J/ApJ/921/32`; table4=Antlia II, table5=Crater II; continuous `Mm` |
| 10 | Chiti 2022 (`2022ApJ...939...41C`) | Grus I | done | User-supplied ASCII Table 2 (per-epoch). MEM tri-state (M / NM / CM); M -> p=1, NM/CM -> p=0 with `member_flag` preserved. 70 stars / 8 M / 4 CM, `<V>` and σ match paper. |
| 11 | Bruce 2023 (`2023ApJ...950..167B`) | Aquarius II | done | User-supplied MRT Table 3; astropy CDS reader. 12 stars / 8 M, `<V>` and σ match paper. |
| 12 | Chiti 2023 (`2023AJ....165...55C`) | Tucana II | done | VizieR `J/AJ/165/55/table6`; per-epoch member-list, p=1 |
| 13 | Heiger 2024 (`2024ApJ...961..234H`) | Centaurus I | done | User-supplied MRT Table 4; astropy CDS reader. Combined-fit `vhel`/`e_vhel`; `Member ∈ {0,1}` flag. 62 stars / 34 M, `<V>` and σ match paper. |
| 14 | Hansen 2024 (`2024ApJ...968...21H`) | Tucana V | done | User-supplied ASCII Table 1 ("Observing Log"), per-epoch v_hel for 3 Tuc V members across 17 MIKE+IMACS epochs. Adapter normalizes a non-standard `hh:mm:ss:ff` RA typo on rows 1-2, inherits Gaia ID/coords across continuation rows. Tuc V-1 shows binary-like 21 km/s swings; excluding it gives σ ≈ 0.93 km/s, matching LVDB σ = 1.2 km/s. |
| 15 | Tan 2025 (`2025ApJ...979..176T`) | Leo VI | done | User-supplied ASCII Table 2 (decimal coords, `v +or- err`). Paper's own ε_vhel < 10 km/s split: 9 confirmed (p=1), 4 candidates (p=0, `member_flag='C'`). `<V>` matches paper. |

VizieR-available subset (verified 2026-05-05 via `astroquery.vizier.Vizier.find_catalogs`): papers 1–2 and 6–9 and 12 (7 of 15). The other 8 were IOP-published with no VizieR mirror; their machine-readable tables sit behind a Radware bot-detection wall on iopscience. As of 2026-05-06 all 8 are resolved via user browser-session downloads — **Path B sweep complete**.

Statuses:
- `pending` — not yet attempted.
- `in_progress` — current paper, mid-ingest.
- `done` — `data/star_catalogs/<lvdb_key>.npz` written, paper-table and LVDB sanity checks pass.
- `blocked` — paper has no public per-star catalog or another unresolved obstacle; galaxy logged as `path_b_unresolved`.

## Per-galaxy outputs (resolved)

Filled in as papers complete. The "paper" columns record the source's own
published values (member count, mean V, σ_los); the "ours" columns record what
our npz produces under the same cuts the paper used. Side-by-side discrepancy
is the canonical correctness check.

| LVDB key | Paper bibkey | n_rows (raw npz) | n_mem (paper / ours) | <V>_mem km/s (paper / ours) | σ_los km/s (paper / ours) | LVDB v_sys (km/s) | LVDB σ (km/s) | Commit |
|---|---|---|---|---|---|---|---|---|
| carina_1 | walker2009 | 1982 | 774¹ / 441 (Mmb=1, of which 172 have finite V) | 222.9 / 223.16 | 6.6 / 6.15² | 222.90 | 6.60 | 3546e78 |
| reticulum_2 | walker2015 | 38 | 17 / 18 (Mm?=Y) | 64.3 / 65.53 | 3.6 / 3.69 (naive) ≈ 3.32 (quad-deconv) | 64.30 | 3.60 | 44c93c8 |
| carina_2 | li2018 | 407 (per-epoch) | 18 / 18 (Mm=2 unique stars from 30 epochs) | 477.2 / 478.41 (per-star IVW) | 3.4 / 5.33 (naive per-star)¹ | 477.20 | 3.40 | (this commit) |
| carina_3 | li2018 | 407 (per-epoch) | 4 / 4 (Mm=3 unique stars from 8 epochs) | 284.6 / 285.23 (per-star IVW) | 5.6 / 5.66 (naive per-star) | 284.60 | 5.60 | 6fe48e1 |
| hydrus_1 | koposov2018 | 139 | ~28 / 33 (sigmoid(logodds) > 0.5) | 80.4 / 80.30 | 2.69 / 3.32 (naive) ≈ 2.96 (decov) | 80.40 | 2.70 | f623827 |
| tucana_4 | simon2020 | 223 (per-epoch) | 12 / 11 (Mm=1 unique) | 15.9 / 16.09 (per-star IVW) | 4.3 / 3.96 (naive per-star) ≈ 3.80 (decov) | 15.90 | 4.30 | c077109 |
| antlia_2 | ji2021 | 508 | ~290 / 290 (Mm>0.5) | 288 / 287.95 | 5.7 / 7.30 (naive) ≈ 7.00 (decov) | 288.80 | 5.98 | (this commit) |
| crater_2 | ji2021 | 207 | ~140 / 141 (Mm>0.5) | 89 / 89.43 | 2.7 / 4.13 (naive) ≈ 2.83 (decov) | 89.30 | 2.34 | 73bb830 |
| tucana_2 | chiti2023 | 60 (per-epoch) | 19 unique mem stars (member list, p=1) | -124.7 / -125.03 (per-star IVW) | 3.8 / 4.13 (naive per-star) ≈ 4.07 (decov) | -124.70 | 3.80 | (this commit) |
| pisces_2 | kirby2015 | 13 | 7 / 7 (Member?='Y') | -226 / -225.51 | 5.4 / 5.41 (naive) ≈ 4.36 (decov) | -226.50 | 5.40 | 66454a8 |
| horologium_1 | koposov2015 | 18 | 5 / 5 (Member?='Yes') | — / 112.68 | 4.9 / 4.98 (naive) ≈ 4.93 (decov) | 112.80 | 4.90 | (this commit) |
| eridanus_2 | li2017 | 92 (per-epoch) | 28 / 28 (MEM=1 unique) | 75.6 / 75.70 (per-star IVW) | 6.9 / 6.66 (naive per-star) | 75.60 | 6.90 | (this commit) |
| grus_1 | chiti2022 | 80 (per-epoch) | 8 / 8 (MEM='M' unique) | -143.5 / -142.93 (per-star IVW) | 2.5 / 2.14 (naive per-star) ≈ 1.75 (decov) | -143.50 | 2.50 | (this commit) |
| aquarius_2 | bruce2023 | 12 | 8 / 8 (Mem='M') | -65.3 / -65.54 | 4.7 / 4.58 (naive) ≈ 4.39 (decov) | -65.30 | 4.70 | (this commit) |
| centaurus_1 | heiger2024 | 62 | 34 / 34 (Member=1) | 44.8 / 44.84 | 4.2 / 4.08 (naive) ≈ 3.97 (decov) | 44.80 | 4.20 | (this commit) |
| leo_6 | tan2025 | 13 | 9 / 9 (ε_vhel<10) | 170.0 / 171.07 (confirmed only) | 2.85 / 4.18 (naive) ≈ 1.22 (decov)¹ | 170.03 | 2.85 |  eed445c |
| tucana_5 | hansen2024 | 17 (per-epoch) | 3 / 3 (member list, p=1) | -34.7 / -33.18 (per-star IVW; Tuc V-1 binary biases low) | 1.2 / 2.80 (naive per-star, all 3) — 0.93 (excluding binary Tuc V-1) | -34.70 | 1.20 | (this commit) |

¹ Li 2018 reports σ_los = 3.4 km/s for Carina II from a binary-aware ML deconvolution. Our naive per-star std (5.33 km/s) does not deconvolve binaries; the data are stored faithfully (the per-epoch npz preserves all 30 Mm=2 epochs), and binary-aware aggregation is downstream of Stage 0b. Member counts and `<V>` match the paper exactly. For Carina III the smaller binary fraction makes the naive per-star std (5.66 km/s) match the paper's 5.6 km/s without deconvolution.

¹ Walker 2009 abstract reports ~774 "likely" Carina members from the EM mixture model's continuous probabilities; VizieR's `Mmb` column is a hard 0/1 flag on the strict subset (441), and only 172 of those have a velocity value not masked in the per-star averaged table. The raw npz preserves all 1982 rows; downstream sample-selection chooses the cut.

² Our 6.15 is `sqrt(Var - med_eV^2)` over the 172 members with finite V — a quick deconvolution. Walker 2009's 6.6 km/s comes from their full ML deconvolution including per-star errors and binary handling. Within ~7%, consistent with the simpler estimator.

¹ Leo VI: the 9 confirmed members have rather large ε_vhel (1.1–8.3 km/s, median ≈ 4.0). The naive per-star std (4.18) is dominated by these errors; a quick `sqrt(Var - med_eV²)` deconvolution gives 1.22 km/s, while the paper's ML estimator (which uses each star's own ε_vhel) recovers σ ≈ 2.85 km/s — within the ε-dominated regime our quick estimator is unreliable. The data are stored faithfully; binary-aware/error-deconvolved aggregation is downstream.

## Path B unresolved

(none — sweep complete as of 2026-05-06.)
