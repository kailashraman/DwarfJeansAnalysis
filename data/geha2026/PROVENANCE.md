# Geha 2026 Paper I — DEIMOS satellite-galaxy stellar archive

## Source

- **Source paper (Paper I):** Geha et al. 2026, "DEIMOS Stellar Archive: Paper I" (arXiv:2602.10200)
- **Companion paper (Paper II):** Geha et al. 2026, "Paper II — integrated properties" (arXiv:2602.10202)
- **Upstream URL:** https://geha-group.github.io/deimos/ (Dropbox-hosted CSV/FITS)
- **Release stamp:** `20260110` (2026-01-10)

## Files

- `table5A_20260110.csv` — Paper I per-star catalog, 24,436 rows × 50 columns, full precision. **Authoritative source for Stage 0b Path A as of 2026-05-08.** Carries the binary `Pmem_novar` column that reproduces Paper II Table A1 N* counts.
- `table3A_20260110.csv` — Earlier Paper I release, 22,340 rows × 16 columns. Lacks the `Pmem_novar` column. Retained for cross-checks; **no longer ingested**.
- `checksums.sha256` — SHA-256 of both data files.

## table5A header (verified 2026-05-08)

```
system_name, objname, RA, DEC, nmask, nexp, t_exp, masknames, slitwidth,
mean_mjd, SN, serendip, marz_flag, v, v_err, v_chi2, phot_source,
gmag_o, rmag_o, gmag_err, rmag_err, MV_o, rproj_arcm, rproj_kpc,
ew_naI, ew_naI_err, ew_cat, ew_cat_err, ew_feh, ew_feh_err,
ew_w1, ew_w2, ew_w3, ew_gl, gaia_source_id, gaia_pmra, gaia_pmra_err,
gaia_pmdec, gaia_pmdec_err, gaia_pmra_pmdec_corr, gaia_parallax,
gaia_parallax_err, gaia_aen, gaia_aen_sig, flag_coadd, flag_var,
flag_gaia, flag_HB, Pmem, Pmem_novar
```

`v` is heliocentric line-of-sight velocity (km/s), `v_err` its 1σ error, `Pmem_novar` is the binary 0/1 membership flag (1 = member, velocity variables already removed). Verified bit-identical to the MRT for `Pmem_novar` across the full sample; numerical columns (`v`, `RA`, `DEC`, etc.) carry higher precision than the MRT.

## CSV header (legacy, no longer ingested)

```
Galaxy, RA, DEC, r, gr, nmask, t_exp, SN, v, verr, CaT, CaTerr, FeH, FeH_err, Var, Pmem
```

`Pmem` is the graded probability; `Var` is the boolean velocity-variability flag. Combining them as `Pmem > 0.5 & Var != 1` does NOT exactly reproduce Paper II Table A1 N*: residuals scatter ±10% across the sample. The `Pmem-novar` column from the MRT closes this gap.

## Acquisition

- **CSV staged:** 2026-05-05 by Kailash Raman from `Segue1_test/data/`.
- **MRT staged:** 2026-05-08 by Kailash Raman from the ApJ supplement (apjae290dt5_mrt.txt).

## Read-only after staging

Stage 0b reads from this folder and never writes back. Auto-fetch from the Dropbox URL at runtime is forbidden — the Dropbox link contains a session token that may expire, and re-fetch would violate the "stage once, version-pin, never re-fetch" pattern.
