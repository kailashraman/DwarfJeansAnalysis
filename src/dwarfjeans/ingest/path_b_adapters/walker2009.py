"""Walker et al. 2009 (AJ 137, 3100) — adapter for the Path B Carina ingest.

Sources (both VizieR sub-tables of `J/AJ/137/3100` are required):
- `stars.csv`  (`J/AJ/137/3100/stars`)  — per-star summary, 1982 Carina rows.
- `tables.csv` (`J/AJ/137/3100/tables/table2`) — per-spectrum data,
  used only to recover `sigma_eps` for single-epoch stars whose
  `e_<HV>` is masked in `stars.csv`.

Per-galaxy filter is by the `Target` column's prefix: `Car-`, `For-`,
`Scl-`, `Sex-`. Walker 2009 supplies four dSphs but in our 39-galaxy
study Carina is the only one routed via Path B (the other three are
in Geha Path A).

Velocity-error rule (QA-sweep #2):
- Multi-epoch stars (`e_<HV>` finite in stars.csv): use the published
  IVW-combined per-star value verbatim.
- Single-epoch stars (`e_<HV>` masked in stars.csv): pull the
  per-spectrum `e_HV` from the matching `tables.csv` row.

This avoids imputation while keeping per-star granularity.

Schema (stars.csv):
    Target        per-star ID (e.g. "Car-0001")
    o_Target      number of observations (1 -> single-epoch)
    RAJ2000       sexagesimal, J2000
    DEJ2000       sexagesimal, J2000
    Vmag          V-band apparent magnitude
    V-I           V-I color
    Mmb           membership probability (Walker EM mixture)
    <HV>          per-star heliocentric velocity, km/s
    e_<HV>        IVW-combined error (km/s) — masked for o_Target==1
    <SigMg>       Mg index
    e_<SigMg>     Mg index error
    Simbad        SIMBAD link

Velocity frame: heliocentric. No frame conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# Map LVDB key -> Target prefix in stars.csv
_LVDB_KEY_TO_PREFIX = {
    "carina_1": "Car-",
}

COLUMN_MAPPING = {
    "<HV>": "V",
    "e_<HV>": "sigma_eps",
    "Mmb": "p",
    "RAJ2000": "RA_star",
    "DEJ2000": "Dec_star",
    "Target": "star_id_source",  # also used to build star_id
    "Vmag": "Vmag",
    "V-I": "V_minus_I",
    "<SigMg>": "SigMg",
    "e_<SigMg>": "SigMg_err",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    """Return (arrays, meta_extra) for the Path B Carina ingest.

    arrays carries the canonical per-star schema; meta_extra is paper-specific
    provenance the driver merges into the per-galaxy `_meta` dict.
    """
    lvdb_key = registry_row["lvdb_key"]
    if lvdb_key not in _LVDB_KEY_TO_PREFIX:
        raise ValueError(f"walker2009 adapter does not serve lvdb_key={lvdb_key!r}")
    prefix = _LVDB_KEY_TO_PREFIX[lvdb_key]

    stars = Table.read(staged_dir / "stars.csv", format="ascii.ecsv")
    is_galaxy = np.array([str(t).startswith(prefix) for t in stars["Target"]])
    sub = stars[is_galaxy]
    if len(sub) == 0:
        raise RuntimeError(f"{lvdb_key}: 0 rows for prefix {prefix!r} in walker2009 stars.csv")

    # Per-spectrum table — used only to fill σ_eps for single-epoch stars.
    spectra = Table.read(staged_dir / "tables.csv", format="ascii.ecsv")
    is_galaxy_spec = np.array([str(t).startswith(prefix) for t in spectra["Target"]])
    spec = spectra[is_galaxy_spec]

    star_id = np.arange(len(sub), dtype=np.int64)

    # Helper: realize a (possibly masked) column as a float ndarray with NaN where masked.
    def _col(name: str) -> np.ndarray:
        c = sub[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    def _spec_col(name: str) -> np.ndarray:
        c = spec[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    # Walker 2009's VizieR delivery has RAJ2000 / DEJ2000 in sexagesimal
    # (e.g. "06 42 17.94", "-50 58 15.7"); convert to decimal degrees.
    sky = SkyCoord(
        ra=np.asarray(sub["RAJ2000"], dtype=str),
        dec=np.asarray(sub["DEJ2000"], dtype=str),
        unit=(u.hourangle, u.deg),
    )

    targets = np.asarray(sub["Target"], dtype=str)
    o_target = np.asarray(sub["o_Target"], dtype=int)
    e_hv_combined = _col("e_<HV>")
    hv_combined = _col("<HV>")
    spec_targets = np.asarray(spec["Target"], dtype=str)
    spec_e_hv = _spec_col("e_HV")
    spec_hv = _spec_col("HV")

    # ---- Consistency checks (C1-C5) -----------------------------------
    # C1: Target uniqueness in stars (Carina subset).
    if len(np.unique(targets)) != len(targets):
        raise RuntimeError("walker2009 C1: duplicate Target rows in stars.csv (Carina subset)")

    # Build per-Target row indices in spec for C2/C4/C5.
    spec_rows_by_target: dict[str, list[int]] = {}
    for i, t in enumerate(spec_targets):
        spec_rows_by_target.setdefault(t, []).append(i)

    # C2: o_Target ↔ tables row count.
    bad_c2 = []
    for t, n_obs in zip(targets, o_target):
        n_spec = len(spec_rows_by_target.get(t, []))
        if n_obs != n_spec:
            bad_c2.append((t, int(n_obs), n_spec))
    if bad_c2:
        sample = bad_c2[:5]
        raise RuntimeError(
            f"walker2009 C2: o_Target ≠ count(tables.Target) for {len(bad_c2)} stars; "
            f"sample (Target, o_Target, n_spec_rows): {sample}"
        )

    # C3: single-epoch ⇔ masked e_<HV>.
    is_single = (o_target == 1)
    is_masked = ~np.isfinite(e_hv_combined)
    if not np.array_equal(is_single, is_masked):
        n_violations = int(np.sum(is_single != is_masked))
        raise RuntimeError(
            f"walker2009 C3: (o_Target==1) ⇔ isnan(e_<HV>) violated on "
            f"{n_violations} rows."
        )

    # C4: each single-epoch Target resolves to exactly one finite-error spectrum row.
    sigma_eps = np.where(is_single, np.nan, e_hv_combined)
    sigma_source = np.where(
        is_single,
        "tables.HV_per_spectrum",
        "stars.<HV>_combined",
    ).astype(object)
    for i, (t, single) in enumerate(zip(targets, is_single)):
        if not single:
            continue
        rows = spec_rows_by_target.get(t, [])
        if len(rows) != 1:
            raise RuntimeError(
                f"walker2009 C4: single-epoch Target {t!r} has {len(rows)} "
                f"spec rows, expected 1."
            )
        e = spec_e_hv[rows[0]]
        if not np.isfinite(e):
            raise RuntimeError(
                f"walker2009 C4: single-epoch Target {t!r} has non-finite "
                f"e_HV in tables.csv."
            )
        sigma_eps[i] = e

    # C5: round-trip multi-epoch HV. Median |HV_ivw_from_spectra - <HV>_stars| ≤ 0.5 km/s.
    multi_devs = []
    for t, n_obs, hv_pub in zip(targets, o_target, hv_combined):
        if n_obs < 2 or not np.isfinite(hv_pub):
            continue
        rows = spec_rows_by_target[t]
        v = spec_hv[rows]
        e = spec_e_hv[rows]
        good = np.isfinite(v) & np.isfinite(e) & (e > 0)
        if good.sum() < 2:
            continue
        w = 1.0 / e[good] ** 2
        ivw = float(np.sum(w * v[good]) / np.sum(w))
        multi_devs.append(ivw - float(hv_pub))
    if multi_devs:
        med_dev = float(np.median(np.abs(multi_devs)))
        if med_dev > 0.5:
            raise RuntimeError(
                f"walker2009 C5: median |HV_ivw_per_spec - <HV>_stars| = "
                f"{med_dev:.3f} km/s exceeds 0.5 — Target join may be wrong."
            )

    # C6: σ_eps post-build sanity.
    if np.any(~np.isfinite(sigma_eps)):
        n_bad = int(np.sum(~np.isfinite(sigma_eps)))
        raise RuntimeError(f"walker2009 C6: {n_bad} rows still have non-finite sigma_eps after join.")
    out_of_range = (sigma_eps < 0.1) | (sigma_eps > 30.0)
    if np.any(out_of_range):
        # Warning, not raise — some real measurements may be at the edges.
        n_oor = int(np.sum(out_of_range))
        print(
            f"  walker2009 [warn]: {n_oor} sigma_eps values outside [0.1, 30] km/s "
            f"(min={sigma_eps.min():.3f}, max={sigma_eps.max():.3f})"
        )

    arrays = {
        "V": hv_combined,
        "sigma_eps": sigma_eps.astype(float),
        "p": _col("Mmb"),
        "star_id": star_id,
        "RA_star": sky.ra.deg,
        "Dec_star": sky.dec.deg,
        # Auxiliary
        "Target_source_id": targets,
        "o_Target": o_target.astype(float),
        "sigma_eps_source": np.asarray(sigma_source, dtype=str),
        "Vmag": _col("Vmag"),
        "V_minus_I": _col("V-I"),
        "SigMg": _col("<SigMg>"),
        "SigMg_err": _col("e_<SigMg>"),
    }

    n_imputed = int(np.sum(is_single))
    meta_extra = {
        "vizier_catalog": "J/AJ/137/3100",
        "vizier_table": "stars",
        "vizier_aux_table": "tables (J/AJ/137/3100/tables/table2)",
        "system_target_prefix": prefix,
        "membership_rule": "Walker+09 EM-mixture probability (continuous)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "row_index_within_galaxy",
        "column_mapping": COLUMN_MAPPING,
        "sigma_eps_rule": (
            "multi-epoch (o_Target>=2): stars.csv e_<HV> verbatim "
            "(IVW-combined). single-epoch (o_Target==1): tables.csv e_HV "
            "of the matching per-spectrum row. Per-row provenance in "
            "auxiliary `sigma_eps_source` column."
        ),
        "n_single_epoch_filled_from_tables": n_imputed,
        "consistency_checks_passed": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "notes": (
            "Per-star table is canonical (per data_sources.md). For "
            "single-epoch stars Walker+09's stars.csv masks e_<HV>; we "
            "fill it from the matching row in tables.csv (per-spectrum) "
            "so every star has a real σ_eps with no imputation."
        ),
    }
    return arrays, meta_extra
