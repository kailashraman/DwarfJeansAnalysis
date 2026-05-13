"""End-to-end plumbing check: prepare_jeans_input → audit['perspective']
on real ingested catalogs. Catches center/distance/PM lookup regressions
that the in-memory unit tests can't.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from dwarfjeans.jeans.perspective import perspective_correction, sanity_check
from dwarfjeans.jeans.preprocess import prepare_jeans_input

REPO = Path(__file__).resolve().parents[2]


def _row(reg, key):
    sub = reg[reg["lvdb_key"] == key]
    if len(sub) == 0:
        return None
    return {c: sub[c][0] for c in reg.colnames}


@pytest.mark.parametrize("key", ["sculptor_1", "segue_1"])
def test_audit_matches_standalone_sanity_check(key):
    """audit['perspective'] reproduces what the standalone helper reports
    on the same post-selection sample."""
    npz_path = REPO / "data" / "star_catalogs" / f"{key}.npz"
    reg_path = REPO / "data" / "registry" / "galaxies.ecsv"
    if not (npz_path.exists() and reg_path.exists()):
        pytest.skip("real registry/catalog not staged")

    reg_tab = Table.read(reg_path, format="ascii.ecsv")
    row = _row(reg_tab, key)
    cat = np.load(npz_path, allow_pickle=True)

    arrays, audit = prepare_jeans_input(cat, row)
    assert audit["perspective"]["applied"] is True
    assert audit["perspective"]["max_abs_kms"] > 0.0

    # Recompute on the same filtered RA/Dec — should match audit exactly.
    rep = sanity_check(
        ra_deg=arrays["RA_star"], dec_deg=arrays["Dec_star"],
        ra_center_deg=float(row["ra_deg"]),
        dec_center_deg=float(row["dec_deg"]),
        distance_kpc=float(row["distance_kpc"]),
        pm_alpha_star_masyr=audit["perspective"]["pm_alpha_star_masyr"],
        pm_delta_masyr=audit["perspective"]["pm_delta_masyr"],
        v_sys_kms=float(row["vlos_systemic_kms"]),
    )
    assert audit["perspective"]["max_abs_kms"] == pytest.approx(rep.max_abs_kms, rel=1e-12)
    assert audit["perspective"]["rms_kms"] == pytest.approx(rep.rms_kms, rel=1e-12)


def test_v_observed_preserved_for_sculptor():
    """V_obs round-trips: V == V_observed − Δv_persp at the audited PM."""
    npz_path = REPO / "data" / "star_catalogs" / "sculptor_1.npz"
    reg_path = REPO / "data" / "registry" / "galaxies.ecsv"
    if not (npz_path.exists() and reg_path.exists()):
        pytest.skip("real registry/catalog not staged")

    reg_tab = Table.read(reg_path, format="ascii.ecsv")
    row = _row(reg_tab, "sculptor_1")
    cat = np.load(npz_path, allow_pickle=True)
    arrays, audit = prepare_jeans_input(cat, row)

    expected_dv = perspective_correction(
        ra_deg=arrays["RA_star"], dec_deg=arrays["Dec_star"],
        ra_center_deg=float(row["ra_deg"]),
        dec_center_deg=float(row["dec_deg"]),
        distance_kpc=float(row["distance_kpc"]),
        pm_alpha_star_masyr=audit["perspective"]["pm_alpha_star_masyr"],
        pm_delta_masyr=audit["perspective"]["pm_delta_masyr"],
    )
    np.testing.assert_allclose(arrays["V"], arrays["V_observed"] - expected_dv, rtol=1e-12)
