"""Unit tests for dwarfjeans.jeans.preprocess.prepare_jeans_input."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from dwarfjeans.ingest.combiners import CombinePolicy
from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.jeans.selection import SelectionPolicy

REPO = Path(__file__).resolve().parents[2]
PER_EPOCH_KEYS = (
    "carina_2", "carina_3", "eridanus_2", "grus_1",
    "tucana_2", "tucana_4", "tucana_5",
)


def _per_star_npz(R, V, sigma_eps, p, *, pm_meta: dict | None = None, **extras) -> np.lib.npyio.NpzFile:
    """Build an in-memory per-star NpzFile with valid _meta.

    ``pm_meta`` (optional) is merged into ``_meta`` so tests can enable the
    perspective-motion path (keys: ``lvdb_pmra_mas_yr``, ``lvdb_pmdec_mas_yr``,
    ``perspective_correction_applicable``, …). Extras may include
    ``RA_star``/``Dec_star`` arrays needed for the perspective code path.
    """
    meta = {
        "catalog_granularity": "per_star",
        "R_unit": "kpc",
        "source_paper_bibcode": "fake2026Test",
    }
    if pm_meta:
        meta.update(pm_meta)
    buf = io.BytesIO()
    np.savez(buf,
             _meta=np.array(json.dumps(meta), dtype=object),
             R=np.asarray(R, dtype=float),
             V=np.asarray(V, dtype=float),
             sigma_eps=np.asarray(sigma_eps, dtype=float),
             p=np.asarray(p, dtype=float),
             **extras)
    buf.seek(0)
    return np.load(buf, allow_pickle=True)


def _per_epoch_npz(star_id, V, sigma_eps, p, **extras) -> np.lib.npyio.NpzFile:
    meta = {
        "catalog_granularity": "per_epoch",
        "R_unit": "kpc",
        "source_paper_bibcode": "fake2026Test",
    }
    buf = io.BytesIO()
    np.savez(buf,
             _meta=np.array(json.dumps(meta), dtype=object),
             star_id=np.asarray(star_id),
             V=np.asarray(V, dtype=float),
             sigma_eps=np.asarray(sigma_eps, dtype=float),
             p=np.asarray(p, dtype=float),
             **extras)
    buf.seek(0)
    return np.load(buf, allow_pickle=True)


def test_per_star_skips_combiner():
    npz = _per_star_npz(
        R=[0.001, 0.002, 0.5],   # last star at 0.5 kpc fails radial cut
        V=[100.0, 101.0, 99.0],
        sigma_eps=[1.0, 1.0, 1.0],
        p=[1.0, 1.0, 1.0],
    )
    reg = {"rhalf_major_pc": 50.0, "ellipticity": 0.0}  # cut = 0.1 kpc
    arrays, audit = prepare_jeans_input(npz, reg)
    assert audit["granularity"] == "per_star"
    assert audit["combine"] is None
    assert audit["combine_policy"] is None
    assert audit["selection"]["n_input"] == 3
    assert audit["selection"]["n_final"] == 2


def test_per_epoch_runs_combiner_then_selection():
    # Two stars (id 0, 1), 2 epochs each. Star 0 is non-variable; star 1
    # has a 5σ outlier so the variability cut should drop it.
    npz = _per_epoch_npz(
        star_id=[0, 0, 1, 1],
        V=[100.0, 100.5, 100.0, 110.0],
        sigma_eps=[1.0, 1.0, 1.0, 1.0],
        p=[1.0, 1.0, 1.0, 1.0],
        R=[0.001, 0.001, 0.001, 0.001],
    )
    reg = {"rhalf_major_pc": 100.0, "ellipticity": 0.0}  # cut at 0.2 kpc — both pass
    arrays, audit = prepare_jeans_input(npz, reg)
    assert audit["granularity"] == "per_epoch"
    assert audit["combine"] is not None
    assert audit["combine"]["n_input_rows"] == 4
    assert audit["combine"]["n_stars"] == 2
    # Star 1 should be flagged variable; selection drops it.
    assert audit["selection"]["n_input"] == 2
    assert audit["selection"]["n_final"] == 1
    np.testing.assert_array_equal(arrays["star_id"], np.array([0]))


def test_combine_policy_threaded_through():
    """Threading test for CombinePolicy. With sigma_sys > 0, default.combine
    routes through the strict-deconvolution path (combine_star_strict):
    σ_stat² = σ_total² − σ_sys², IVW on σ_stat, re-add σ_sys post-combine."""
    npz = _per_epoch_npz(
        star_id=[0, 0, 0],
        V=[100.0, 100.0, 100.0],
        sigma_eps=[2.0, 2.0, 2.0],   # σ_total per epoch
        p=[1.0, 1.0, 1.0],
        R=[0.001, 0.001, 0.001],
    )
    reg = {"rhalf_major_pc": 100.0, "ellipticity": 0.0}
    pol = CombinePolicy(sigma_sys_kms=1.5, p_threshold=0.05)
    arrays, audit = prepare_jeans_input(npz, reg, combine_policy=pol)
    # σ_stat² = 4 − 2.25 = 1.75; IVW gives σ_stat/√3 → σ_post = √(1.75/3 + 2.25)
    expected = np.sqrt(1.75 / 3 + 2.25)
    np.testing.assert_allclose(arrays["sigma_eps"], expected, rtol=1e-12)
    assert audit["combine_policy"]["sigma_sys_kms"] == 1.5
    assert audit["combine_policy"]["p_threshold"] == 0.05


def test_selection_policy_threaded_through():
    npz = _per_star_npz(
        R=[0.001, 0.001],
        V=[100.0, 100.0],
        sigma_eps=[1.0, 1.0],
        p=[0.7, 0.3],
    )
    reg = {"rhalf_major_pc": 100.0, "ellipticity": 0.0}
    # Tighter membership cut than default
    pol = SelectionPolicy(p_min=0.6)
    arrays, audit = prepare_jeans_input(npz, reg, selection_policy=pol)
    assert audit["selection"]["n_final"] == 1
    assert audit["selection_policy"]["p_min"] == 0.6


def test_per_epoch_without_bibcode_raises():
    meta = {"catalog_granularity": "per_epoch"}  # no source_paper_bibcode
    buf = io.BytesIO()
    np.savez(buf,
             _meta=np.array(json.dumps(meta), dtype=object),
             star_id=np.array([0]),
             V=np.array([100.0]),
             sigma_eps=np.array([1.0]),
             p=np.array([1.0]),
             R=np.array([0.001]))
    buf.seek(0)
    npz = np.load(buf, allow_pickle=True)
    with pytest.raises(ValueError, match="source_paper_bibcode"):
        prepare_jeans_input(npz, {"rhalf_major_pc": 100.0, "ellipticity": 0.0})


def test_unknown_granularity_raises():
    meta = {"catalog_granularity": "weird_format"}
    buf = io.BytesIO()
    np.savez(buf,
             _meta=np.array(json.dumps(meta), dtype=object),
             R=np.array([0.001]),
             V=np.array([100.0]),
             sigma_eps=np.array([1.0]),
             p=np.array([1.0]))
    buf.seek(0)
    npz = np.load(buf, allow_pickle=True)
    with pytest.raises(ValueError, match="catalog_granularity"):
        prepare_jeans_input(npz, {"rhalf_major_pc": 100.0, "ellipticity": 0.0})


def test_per_star_default_granularity_when_meta_absent():
    """A plain dict with no _meta defaults to per_star."""
    cat = {
        "R": np.array([0.001, 0.5]),
        "V": np.array([100.0, 99.0]),
        "sigma_eps": np.array([1.0, 1.0]),
        "p": np.array([1.0, 1.0]),
    }
    reg = {"rhalf_major_pc": 50.0, "ellipticity": 0.0}
    arrays, audit = prepare_jeans_input(cat, reg)
    assert audit["granularity"] == "per_star"
    assert audit["selection"]["n_final"] == 1


def test_selection_blocks_per_epoch_directly():
    """selection.select_jeans_stars must refuse a per-epoch catalog."""
    from dwarfjeans.jeans.selection import select_jeans_stars
    npz = _per_epoch_npz(
        star_id=[0, 0, 1],
        V=[100.0, 101.0, 99.0],
        sigma_eps=[1.0, 1.0, 1.0],
        p=[1.0, 1.0, 1.0],
        R=[0.001, 0.001, 0.001],
    )
    with pytest.raises(ValueError, match="per-epoch"):
        select_jeans_stars(npz, {"rhalf_major_pc": 100.0, "ellipticity": 0.0})


def _registry_row_for(lvdb_key: str) -> dict:
    """Read rhalf_major_pc out of data/registry/galaxies.ecsv for the given
    LVDB key. Uses shlex to honor quoted tokens (e.g. ``"Segue 1"``)."""
    import shlex
    ecsv = REPO / "data" / "registry" / "galaxies.ecsv"
    header = None
    for line in ecsv.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        toks = shlex.split(line)
        if header is None:
            header = toks
            continue
        if toks[0] == lvdb_key:
            row = dict(zip(header, toks))
            eps_str = row.get("ellipticity", "nan")
            try:
                eps = float(eps_str)
            except ValueError:
                eps = float("nan")
            return {
                "rhalf_major_pc": float(row["rhalf_major_pc"]),
                "ellipticity": eps,
            }
    raise KeyError(lvdb_key)


@pytest.mark.parametrize("key", PER_EPOCH_KEYS)
def test_per_epoch_real_catalog_round_trip(key):
    """Smoke test: every per-epoch catalog in data/star_catalogs must
    survive prepare_jeans_input under the registered handler."""
    path = REPO / "data" / "star_catalogs" / f"{key}.npz"
    if not path.exists():
        pytest.skip(f"{path} not present")
    cat = np.load(path, allow_pickle=True)
    reg = _registry_row_for(key)
    arrays, audit = prepare_jeans_input(cat, reg)
    assert audit["granularity"] == "per_epoch"
    assert audit["combine"] is not None
    assert audit["combine"]["n_input_rows"] > 0
    assert audit["combine"]["n_stars"] > 0
    assert audit["selection"]["n_final"] >= 0
    assert "V" in arrays and "sigma_eps" in arrays and "R" in arrays
    assert len(arrays["V"]) == audit["selection"]["n_final"]


# ---- perspective-motion correction integration into prepare_jeans_input ----

_SCULPTOR_PM_META = {
    "lvdb_pmra_mas_yr": 0.10,
    "lvdb_pmdec_mas_yr": -0.158,
    "lvdb_ref_proper_motion": "Pace2022ApJ...940..136P",
    "perspective_correction_applicable": True,
}

_SCULPTOR_REG = {
    "rhalf_major_pc": 273.0, "ellipticity": 0.0,
    "ra_deg": 15.0183, "dec_deg": -33.7186,
    "distance_kpc": 86.0, "vlos_systemic_kms": 111.4,
}


def test_perspective_pm_present_shifts_V():
    """PM in _meta + RA/Dec → V is replaced by V_obs − Δv_persp; audit records it."""
    import math
    from dwarfjeans.jeans.perspective import A_KMS_PER_MASYR_KPC
    # Two stars within R_h; one east of center, one north of center.
    ra0, dec0 = _SCULPTOR_REG["ra_deg"], _SCULPTOR_REG["dec_deg"]
    cos_d0 = math.cos(math.radians(dec0))
    ra = np.array([ra0 + 0.1 / cos_d0, ra0])  # 0.1° east, then center
    dec = np.array([dec0, dec0 + 0.1])         # center-RA, 0.1° north
    npz = _per_star_npz(
        R=[0.001, 0.001],
        V=[100.0, 100.0],
        sigma_eps=[1.0, 1.0],
        p=[1.0, 1.0],
        RA_star=ra, Dec_star=dec,
        pm_meta=_SCULPTOR_PM_META,
    )
    arrays, audit = prepare_jeans_input(npz, _SCULPTOR_REG)
    assert audit["perspective"]["applied"] is True
    assert audit["perspective"]["pm_alpha_star_masyr"] == 0.10
    assert audit["perspective"]["pm_delta_masyr"] == -0.158
    assert audit["perspective"]["max_abs_kms"] > 0.0
    # V_observed preserved; V is shifted.
    assert "V_observed" in arrays
    np.testing.assert_array_equal(arrays["V_observed"], [100.0, 100.0])
    expected_dv = A_KMS_PER_MASYR_KPC * 86.0 * np.array([
        0.10 * math.radians(0.1),     # east star: μ_α* · Δα*
        -0.158 * math.radians(0.1),   # north star: μ_δ · Δδ
    ])
    np.testing.assert_allclose(arrays["V"], arrays["V_observed"] - expected_dv, rtol=1e-12)


def test_perspective_skipped_when_no_pm_meta():
    """Default fixture (no PM in _meta) → no shift, audit records skip reason."""
    npz = _per_star_npz(
        R=[0.001, 0.001],
        V=[100.0, 101.0],
        sigma_eps=[1.0, 1.0],
        p=[1.0, 1.0],
    )
    reg = {"rhalf_major_pc": 100.0, "ellipticity": 0.0}
    arrays, audit = prepare_jeans_input(npz, reg)
    assert audit["perspective"]["applied"] is False
    assert "applicable" in audit["perspective"]["reason"]
    assert "V_observed" not in arrays
    np.testing.assert_array_equal(arrays["V"], [100.0, 101.0])


def test_perspective_skipped_when_RA_missing():
    """PM-applicable in _meta but per-star RA/Dec missing → skip with reason, no shift."""
    npz = _per_star_npz(
        R=[0.001, 0.001],
        V=[100.0, 101.0],
        sigma_eps=[1.0, 1.0],
        p=[1.0, 1.0],
        pm_meta=_SCULPTOR_PM_META,
    )
    arrays, audit = prepare_jeans_input(npz, _SCULPTOR_REG)
    assert audit["perspective"]["applied"] is False
    assert "RA_star" in audit["perspective"]["reason"]
    assert "V_observed" not in arrays
    np.testing.assert_array_equal(arrays["V"], [100.0, 101.0])
