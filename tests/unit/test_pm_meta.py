"""Tests for `staging.pm_meta_from_registry_row` — the helper that maps
registry PM columns into the per-galaxy `_meta` block written by Stage 0b."""

import math

import numpy as np

from dwarfjeans.ingest.staging import pm_meta_from_registry_row


def _row(**overrides):
    base = {
        "pmra_mas_yr": 0.10,
        "pmra_em_mas_yr": 0.002,
        "pmra_ep_mas_yr": 0.002,
        "pmdec_mas_yr": -0.158,
        "pmdec_em_mas_yr": 0.002,
        "pmdec_ep_mas_yr": 0.002,
        "ref_proper_motion": "Pace2022ApJ...940..136P",
    }
    base.update(overrides)
    return base


def test_classical_carries_pm_and_flag_true():
    m = pm_meta_from_registry_row(_row())
    assert m["lvdb_pmra_mas_yr"] == 0.10
    assert m["lvdb_pmdec_mas_yr"] == -0.158
    assert m["lvdb_pmra_em_mas_yr"] == 0.002
    assert m["lvdb_pmdec_ep_mas_yr"] == 0.002
    assert m["lvdb_ref_proper_motion"] == "Pace2022ApJ...940..136P"
    assert m["perspective_correction_applicable"] is True


def test_nan_pmra_yields_none_and_flag_false():
    m = pm_meta_from_registry_row(_row(pmra_mas_yr=math.nan))
    assert m["lvdb_pmra_mas_yr"] is None
    assert m["perspective_correction_applicable"] is False


def test_missing_error_disables_flag():
    m = pm_meta_from_registry_row(_row(pmdec_em_mas_yr=math.nan))
    assert m["lvdb_pmdec_em_mas_yr"] is None
    assert m["perspective_correction_applicable"] is False


def test_empty_reference_string_normalizes_to_none():
    m = pm_meta_from_registry_row(_row(ref_proper_motion=""))
    assert m["lvdb_ref_proper_motion"] is None


def test_json_round_trip_with_none_fields():
    import json
    m = pm_meta_from_registry_row(_row(pmra_mas_yr=math.nan, ref_proper_motion=""))
    # Must be JSON-serializable — Stage 0b writes _meta as a JSON string.
    s = json.dumps(m)
    back = json.loads(s)
    assert back["lvdb_pmra_mas_yr"] is None
    assert back["lvdb_ref_proper_motion"] is None
    assert back["perspective_correction_applicable"] is False


def test_numpy_scalar_inputs_handled():
    # astropy Table rows return numpy scalars, not Python floats. Make sure
    # the helper digests them correctly.
    m = pm_meta_from_registry_row(_row(
        pmra_mas_yr=np.float64(0.532),
        pmdec_mas_yr=np.float64(0.127),
    ))
    assert m["lvdb_pmra_mas_yr"] == 0.532
    assert m["lvdb_pmdec_mas_yr"] == 0.127
    assert isinstance(m["lvdb_pmra_mas_yr"], float)
