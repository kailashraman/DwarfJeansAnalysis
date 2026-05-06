"""Walker et al. 2009 (AJ 137, 3100) — adapter for the Path B Carina ingest.

Source: VizieR catalog `J/AJ/137/3100`, table `stars` (per-star summary;
8855-row per-epoch `tables` is staged for traceability but not ingested,
per data_sources.md "When both are published, the per-star table is
canonical for our purposes").

Per-galaxy filter is by the `Target` column's prefix: `Car-`, `For-`,
`Scl-`, `Sex-`. Walker 2009 supplies four dSphs but in our 39-galaxy
study Carina is the only one routed via Path B (the other three are
in Geha Path A).

Schema:
    Target        per-star ID (e.g. "Car-0001")
    o_Target      occurrence count (per-star — not used)
    RAJ2000       deg, J2000
    DEJ2000       deg, J2000
    Vmag          V-band apparent magnitude
    V-I           V-I color
    Mmb           binary membership flag (0/1) — assigned by Walker+09
    <HV>          epoch-averaged heliocentric velocity, km/s
    e_<HV>        velocity error, km/s
    <SigMg>       Mg index
    e_<SigMg>     Mg index error
    Simbad        SIMBAD link

Membership rule (per data_sources.md "Path B" step 5): binary flag, encode
verbatim as p_i ∈ {0, 1}. Members and non-members both stored
(Stage 0b is raw-data-only; sample selection is downstream).

Velocity frame: heliocentric (column header reads `<HV>` = heliocentric
velocity); no frame conversion needed.
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

    star_id = np.arange(len(sub), dtype=np.int64)

    # Helper: realize a (possibly masked) column as a float ndarray with NaN where masked.
    def _col(name: str) -> np.ndarray:
        c = sub[name]
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

    arrays = {
        "V": _col("<HV>"),
        "sigma_eps": _col("e_<HV>"),
        "p": _col("Mmb"),                  # 0/1, encoded verbatim
        "star_id": star_id,
        "RA_star": sky.ra.deg,
        "Dec_star": sky.dec.deg,
        # Auxiliary
        "Target_source_id": np.asarray(sub["Target"], dtype=str),
        "Vmag": _col("Vmag"),
        "V_minus_I": _col("V-I"),
        "SigMg": _col("<SigMg>"),
        "SigMg_err": _col("e_<SigMg>"),
    }

    meta_extra = {
        "vizier_catalog": "J/AJ/137/3100",
        "vizier_table": "stars",
        "system_target_prefix": prefix,
        "membership_rule": "binary_flag",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "row_index_within_galaxy",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "Per-star table preferred over per-epoch (per data_sources.md). "
            "Per-epoch `tables.csv` staged in the same folder for traceability "
            "but not ingested. Mmb is a binary 0/1 flag from Walker+09's "
            "EM-mixture analysis; encoded verbatim as p_i."
        ),
    }
    return arrays, meta_extra
