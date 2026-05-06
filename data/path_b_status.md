# Path B ingest status

This file is the resume-safe state for the Path B (non-Geha) per-star catalog
ingest. Updated and committed every time a paper's state changes.

To continue from a fresh session: read this file, find the first paper marked
`pending` or `blocked`, and pick up from there. Process order is fixed below.

Last updated: 2026-05-05 (Walker 2009 ingested)

## Per-paper status

Process order is fixed (classical/well-indexed papers first, recent UFD
discoveries next, likely-blocked last). One paper per commit.

| # | Paper (bibcode) | Galaxies | Status | Notes |
|---|---|---|---|---|
| 1 | Walker 2009 (`2009AJ....137.3100W`) | Carina | done | VizieR `J/AJ/137/3100/stars`; `<V>` and σ_los match paper |
| 2 | Walker 2015 (`2015ApJ...808..108W`) | Reticulum II | pending | — |
| 3 | Kirby 2015 (`2015ApJ...810...56K`) | Pisces II | pending | — |
| 4 | Koposov 2015 (`2015ApJ...811...62K`) | Horologium I | pending | — |
| 5 | Li 2017 (`2017ApJ...838....8L`) | Eridanus II | pending | — |
| 6 | Li 2018 (`2018ApJ...857..145L`) | Carina II + Carina III | pending | first multi-galaxy paper |
| 7 | Koposov 2018 (`2018MNRAS.479.5343K`) | Hydrus I | pending | — |
| 8 | Simon 2020 (`2020ApJ...892..137S`) | Tucana IV | pending | — |
| 9 | Ji 2021 (`2021ApJ...921...32J`) | Antlia II + Crater II | pending | — |
| 10 | Chiti 2022 (`2022ApJ...939...41C`) | Grus I | pending | — |
| 11 | Bruce 2023 (`2023ApJ...950..167B`) | Aquarius II | pending | — |
| 12 | Chiti 2023 (`2023AJ....165...55C`) | Tucana II | pending | — |
| 13 | Heiger 2024 (`2024ApJ...961..234H`) | Centaurus I | pending | — |
| 14 | Hansen 2024 (`2024ApJ...968...21H`) | Tucana V | pending | — |
| 15 | Tan 2025 (`2025ApJ...979..176T`) | Leo VI | pending | very recent — may not be on VizieR; expect MRT or arXiv supp fallback |

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
| carina_1 | walker2009 | 1982 | 774¹ / 441 (Mmb=1, of which 172 have finite V) | 222.9 / 223.16 | 6.6 / 6.15² | 222.90 | 6.60 | (this commit) |

¹ Walker 2009 abstract reports ~774 "likely" Carina members from the EM mixture model's continuous probabilities; VizieR's `Mmb` column is a hard 0/1 flag on the strict subset (441), and only 172 of those have a velocity value not masked in the per-star averaged table. The raw npz preserves all 1982 rows; downstream sample-selection chooses the cut.

² Our 6.15 is `sqrt(Var - med_eV^2)` over the 172 members with finite V — a quick deconvolution. Walker 2009's 6.6 km/s comes from their full ML deconvolution including per-star errors and binary handling. Within ~7%, consistent with the simpler estimator.

## Path B unresolved

No galaxies marked unresolved yet.
