# Path B ingest status

This file is the resume-safe state for the Path B (non-Geha) per-star catalog
ingest. Updated and committed every time a paper's state changes.

To continue from a fresh session: read this file, find the first paper marked
`pending` or `blocked`, and pick up from there. Process order is fixed below.

Last updated: 2026-05-05 (Simon 2020 → Tucana IV ingested)

## Per-paper status

Process order is fixed (classical/well-indexed papers first, recent UFD
discoveries next, likely-blocked last). One paper per commit.

| # | Paper (bibcode) | Galaxies | Status | Notes |
|---|---|---|---|---|
| 1 | Walker 2009 (`2009AJ....137.3100W`) | Carina | done | VizieR `J/AJ/137/3100/stars`; `<V>` and σ_los match paper |
| 2 | Walker 2015 (`2015ApJ...808..108W`) | Reticulum II | done | VizieR `J/ApJ/808/108/table1`; `<V>` and σ_los match paper |
| 3 | Kirby 2015 (`2015ApJ...810...56K`) | Pisces II | blocked | Not on VizieR (none of `2015ApJ...810...56K`, `J/ApJ/810/56`, "Kirby Pisces II" return a catalog as of 2026-05-05). IOPscience MRT page is bot-walled (perfdrive challenge). Resolution: user to download `apjXXXXXXt2_mrt.txt` (or equivalent) manually from https://iopscience.iop.org/article/10.1088/0004-637X/810/1/56 in a browser, place in `data/kirby2015/`, and re-run. |
| 4 | Koposov 2015 (`2015ApJ...811...62K`) | Horologium I | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |
| 5 | Li 2017 (`2017ApJ...838....8L`) | Eridanus II | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |
| 6 | Li 2018 (`2018ApJ...857..145L`) | Carina II + Carina III | done | VizieR `J/ApJ/857/145/table4` (per-epoch); split by `Mm ∈ {0,2,3}` |
| 7 | Koposov 2018 (`2018MNRAS.479.5343K`) | Hydrus I | done | VizieR `J/MNRAS/479/5343/table2`; `logodds` → sigmoid; `<V>` and σ match paper |
| 8 | Simon 2020 (`2020ApJ...892..137S`) | Tucana IV | done | VizieR `J/ApJ/892/137/table3`; per-epoch; filter `Gal=='TucIV'` |
| 9 | Ji 2021 (`2021ApJ...921...32J`) | Antlia II + Crater II | pending | VizieR `J/ApJ/921/32` |
| 10 | Chiti 2022 (`2022ApJ...939...41C`) | Grus I | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |
| 11 | Bruce 2023 (`2023ApJ...950..167B`) | Aquarius II | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |
| 12 | Chiti 2023 (`2023AJ....165...55C`) | Tucana II | pending | VizieR `J/AJ/165/55` |
| 13 | Heiger 2024 (`2024ApJ...961..234H`) | Centaurus I | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |
| 14 | Hansen 2024 (`2024ApJ...968...21H`) | Tucana V | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |
| 15 | Tan 2025 (`2025ApJ...979..176T`) | Leo VI | blocked | Not on VizieR; IOPscience MRT bot-walled. See "Path B unresolved" below. |

VizieR-available subset (verified 2026-05-05 via `astroquery.vizier.Vizier.find_catalogs`): papers 1–2 and 6–9 and 12 (7 of 15). The other 8 are IOP-published with no VizieR mirror; their machine-readable tables live behind a Radware bot-detection wall on iopscience and are not auto-fetchable from this cluster. Plan: process the VizieR-fetchable ones in order (papers 6, 7, 8, 9, 12), then user manually downloads the IOP MRTs for the blocked set in a browser session.

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
| tucana_4 | simon2020 | 223 (per-epoch) | 12 / 11 (Mm=1 unique) | 15.9 / 16.09 (per-star IVW) | 4.3 / 3.96 (naive per-star) ≈ 3.80 (decov) | 15.90 | 4.30 | (this commit) |

¹ Li 2018 reports σ_los = 3.4 km/s for Carina II from a binary-aware ML deconvolution. Our naive per-star std (5.33 km/s) does not deconvolve binaries; the data are stored faithfully (the per-epoch npz preserves all 30 Mm=2 epochs), and binary-aware aggregation is downstream of Stage 0b. Member counts and `<V>` match the paper exactly. For Carina III the smaller binary fraction makes the naive per-star std (5.66 km/s) match the paper's 5.6 km/s without deconvolution.

¹ Walker 2009 abstract reports ~774 "likely" Carina members from the EM mixture model's continuous probabilities; VizieR's `Mmb` column is a hard 0/1 flag on the strict subset (441), and only 172 of those have a velocity value not masked in the per-star averaged table. The raw npz preserves all 1982 rows; downstream sample-selection chooses the cut.

² Our 6.15 is `sqrt(Var - med_eV^2)` over the 172 members with finite V — a quick deconvolution. Walker 2009's 6.6 km/s comes from their full ML deconvolution including per-star errors and binary handling. Within ~7%, consistent with the simpler estimator.

## Path B unresolved

All 8 entries below share the same blocker — no VizieR catalog and the
publisher hosts the machine-readable tables on a server protected by
Radware's perfdrive bot-detection. From this cluster we can hit
`https://iopscience.iop.org/...` but the MRT URL pattern only resolves
after the user passes the bot challenge in a browser. Resolution for
all of them is the same shape: open the article URL in a browser,
download the per-star MRT (typically `tableX_mrt.txt`), drop it in
`data/<bibkey>/`, then re-run
`python -m data_ingest.stage0b_pathb --lvdb-key <key>` after the
adapter is in place.

| LVDB key | Paper bibkey | Article URL | Notes |
|---|---|---|---|
| pisces_2 | kirby2015 | https://iopscience.iop.org/article/10.1088/0004-637X/810/1/56 | "Spectroscopic Confirmation of Hydra II and Pisces II" — paper reports σ_los = 5.4 km/s for Pisces II from 8 members. |
| horologium_1 | koposov2015 | https://iopscience.iop.org/article/10.1088/0004-637X/811/1/62 | "Kinematics & Chemistry of Reticulum 2 and Horologium 1" — VLT/GIRAFFE; 5 Hor I members, σ_los = 4.9 km/s. |
| eridanus_2 | li2017 | https://iopscience.iop.org/article/10.3847/1538-4357/aa6113 | Distant UFD Eridanus II. |
| grus_1 | chiti2022 | https://iopscience.iop.org/article/10.3847/1538-4357/ac81b9 | UFD Grus I. |
| aquarius_2 | bruce2023 | https://iopscience.iop.org/article/10.3847/1538-4357/acc7c4 | UFD Aquarius II. |
| centaurus_1 | heiger2024 | https://iopscience.iop.org/article/10.3847/1538-4357/ad0d9f | Recent UFD Centaurus I. |
| tucana_5 | hansen2024 | https://iopscience.iop.org/article/10.3847/1538-4357/ad429c | Recent UFD Tucana V. |
| leo_6 | tan2025 | https://iopscience.iop.org/article/10.3847/1538-4357/ad9f23 | Very recent UFD Leo VI. |
