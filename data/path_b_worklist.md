# Path B — per-paper staging worklist

Generated from `data/registry/galaxies.ecsv` — 17 Path B galaxies across 15 distinct `ref_vlos` papers.

For each entry below, follow `docs/plan/data_sources.md` §"Path B — LVDB `ref_vlos` paper ingest" steps 3–5: locate the per-star catalog (VizieR preferred, journal MRT next, author page last resort), download into the suggested `data/<bibkey>/` folder, write `PROVENANCE.md`, and run `sha256sum *.* > checksums.sha256`.

After staging, write a per-paper adapter (a small dict / YAML mapping the paper's column names to canonical `V`, `sigma_eps`, `p`, `star_id`, `RA_star`, `Dec_star`) and run a per-galaxy ingest analogous to `data_ingest/stage0b_geha.py:ingest_one`.

---

## Walker 2009 — `2009AJ....137.3100W`

- Suggested folder: `data/walker2009/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2009AJ....137.3100W
- Galaxies served by this paper:
  - **Carina** (`carina_1`) — center (RA, Dec) = (100.4065°, -50.9593°)

## Walker 2015 — `2015ApJ...808..108W`

- Suggested folder: `data/walker2015/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2015ApJ...808..108W
- Galaxies served by this paper:
  - **Reticulum II** (`reticulum_2`) — center (RA, Dec) = (53.9203°, -54.0513°)

## Kirby 2015 — `2015ApJ...810...56K`

- Suggested folder: `data/kirby2015/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2015ApJ...810...56K
- Galaxies served by this paper:
  - **Pisces II** (`pisces_2`) — center (RA, Dec) = (344.6365°, 5.9555°)

## Koposov 2015 — `2015ApJ...811...62K`

- Suggested folder: `data/koposov2015/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2015ApJ...811...62K
- Galaxies served by this paper:
  - **Horologium I** (`horologium_1`) — center (RA, Dec) = (43.8755°, -54.1174°)

## Li 2017 — `2017ApJ...838....8L`

- Suggested folder: `data/li2017/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2017ApJ...838....8L
- Galaxies served by this paper:
  - **Eridanus II** (`eridanus_2`) — center (RA, Dec) = (56.0925°, -43.5329°)

## Li 2018 — `2018ApJ...857..145L`

- Suggested folder: `data/li2018/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2018ApJ...857..145L
- Galaxies served by this paper:
  - **Carina II** (`carina_2`) — center (RA, Dec) = (114.1066°, -57.9991°)
  - **Carina III** (`carina_3`) — center (RA, Dec) = (114.6298°, -57.8997°)

## Koposov 2018 — `2018MNRAS.479.5343K`

- Suggested folder: `data/koposov2018/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5343K
- Galaxies served by this paper:
  - **Hydrus I** (`hydrus_1`) — center (RA, Dec) = (37.3890°, -79.3089°)

## Simon 2020 — `2020ApJ...892..137S`

- Suggested folder: `data/simon2020/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2020ApJ...892..137S
- Galaxies served by this paper:
  - **Tucana IV** (`tucana_4`) — center (RA, Dec) = (0.7170°, -60.8300°)

## Ji 2021 — `2021ApJ...921...32J`

- Suggested folder: `data/ji2021/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2021ApJ...921...32J
- Galaxies served by this paper:
  - **Antlia II** (`antlia_2`) — center (RA, Dec) = (143.8079°, -36.6991°)
  - **Crater II** (`crater_2`) — center (RA, Dec) = (177.3100°, -18.4130°)

## Chiti 2022 — `2022ApJ...939...41C`

- Suggested folder: `data/chiti2022/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2022ApJ...939...41C
- Galaxies served by this paper:
  - **Grus I** (`grus_1`) — center (RA, Dec) = (344.1660°, -50.1680°)

## Chiti 2023 — `2023AJ....165...55C`

- Suggested folder: `data/chiti2023/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2023AJ....165...55C
- Galaxies served by this paper:
  - **Tucana II** (`tucana_2`) — center (RA, Dec) = (342.9796°, -58.5689°)

## Bruce 2023 — `2023ApJ...950..167B`

- Suggested folder: `data/bruce2023/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2023ApJ...950..167B
- Galaxies served by this paper:
  - **Aquarius II** (`aquarius_2`) — center (RA, Dec) = (338.4813°, -9.3274°)

## Heiger 2024 — `2024ApJ...961..234H`

- Suggested folder: `data/heiger2024/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2024ApJ...961..234H
- Galaxies served by this paper:
  - **Centaurus I** (`centaurus_1`) — center (RA, Dec) = (189.5908°, -40.9043°)

## Hansen 2024 — `2024ApJ...968...21H`

- Suggested folder: `data/hansen2024/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2024ApJ...968...21H
- Galaxies served by this paper:
  - **Tucana V** (`tucana_5`) — center (RA, Dec) = (354.3470°, -63.2660°)

## Tan 2025 — `2025ApJ...979..176T`

- Suggested folder: `data/tan2025/`
- ADS link: https://ui.adsabs.harvard.edu/abs/2025ApJ...979..176T
- Galaxies served by this paper:
  - **Leo VI** (`leo_6`) — center (RA, Dec) = (171.0770°, 24.8740°)
