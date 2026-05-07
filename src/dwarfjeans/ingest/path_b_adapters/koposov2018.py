"""Koposov et al. 2018 (MNRAS 479, 5343) — adapter for Hydrus I.

Source: VizieR `J/MNRAS/479/5343/table2`, 139 rows, per-star.

Membership is published as `logodds` (log-odds of Hydrus I membership);
we convert to p via the logistic transform p = 1/(1+exp(-logodds)).
The published log-odds dynamic range is huge (-1000 to +10) so the
sigmoid output is effectively binary in practice; we still store the
continuous p and carry `logodds` as an auxiliary so downstream can
re-thresholding however it likes.

Velocity frame: heliocentric (`HRV`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

COLUMN_MAPPING = {
    "HRV": "V",
    "e_HRV": "sigma_eps",
    "logodds (sigmoid)": "p",
    "RAJ2000": "RA_star",
    "DEJ2000": "Dec_star",
    "ID": "star_id_source",
}


def load(staged_dir: Path, registry_row) -> tuple[dict, dict]:
    if registry_row["lvdb_key"] != "hydrus_1":
        raise ValueError(
            f"koposov2018 adapter only serves hydrus_1; got {registry_row['lvdb_key']!r}"
        )
    t = Table.read(staged_dir / "table2.csv", format="ascii.ecsv")
    n = len(t)

    def _col(name: str) -> np.ndarray:
        c = t[name]
        if hasattr(c, "filled"):
            return np.asarray(c.filled(np.nan), dtype=float)
        return np.asarray(c, dtype=float)

    lodds = _col("logodds")
    # Sigmoid with clip to avoid overflow at extreme negative log-odds.
    p = np.zeros(n, dtype=float)
    fin = np.isfinite(lodds)
    p[fin] = 1.0 / (1.0 + np.exp(-np.clip(lodds[fin], -700, 700)))
    # Masked logodds (3 of 139 here) -> p=0 (non-member). The data_sources.md
    # missing-prob default (p=1) is intended for sources that publish member
    # lists and let null mean "implicitly a member"; Koposov 2018 publishes
    # *every* observed star with a continuous classification, so a masked
    # logodds is "fit failed", not "implicit member" — defaulting to p=1
    # would promote 3 stars with outlier velocities into the member sample
    # and inflate sigma from 3.3 to 23 km/s. Documented in data_sources.md
    # changelog 2026-05-05.

    arrays = {
        "V": _col("HRV"),
        "sigma_eps": _col("e_HRV"),
        "p": p,
        "star_id": np.arange(n, dtype=np.int64),
        "RA_star": _col("RAJ2000"),
        "Dec_star": _col("DEJ2000"),
        # Auxiliary
        "ID_source": np.asarray(t["ID"], dtype=str),
        "logodds": lodds,
        "logodds_err": _col("e_logodds"),
        "FeH": _col("[Fe/H]"),
        "FeH_err": _col("e_[Fe/H]"),
        "Teff": _col("Teff"),
        "Teff_err": _col("e_Teff"),
        "loggmean": _col("loggmean"),
        "loggmean_err": _col("e_loggmean"),
        "gmag": _col("gmag"),
        "rmag": _col("rmag"),
        "SN": _col("S/N"),
    }

    meta_extra = {
        "vizier_catalog": "J/MNRAS/479/5343",
        "vizier_table": "table2",
        "membership_rule": "continuous: p = sigmoid(logodds)",
        "velocity_frame": "heliocentric",
        "catalog_granularity": "per_star",
        "star_id_source_column": "row_index",
        "column_mapping": COLUMN_MAPPING,
        "notes": (
            "logodds is a log-odds of membership from Koposov+18's Bayesian "
            "mixture; effective dynamic range is huge so sigmoid is nearly "
            "binary in practice. 3 of 139 rows have masked logodds; the "
            "adapter assigns p=0 to those (NOT the global p=1 default), "
            "because Koposov 2018 publishes every observed star with a "
            "continuous classification — masked logodds means 'fit failed', "
            "not 'implicit member'. See data_sources.md 2026-05-05 changelog."
        ),
    }
    return arrays, meta_extra
