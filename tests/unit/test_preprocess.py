"""Unit tests for dwarfjeans.jeans.preprocess.prepare_jeans_input."""

import io
import json

import numpy as np
import pytest

from dwarfjeans.ingest.combiners import CombinePolicy
from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.jeans.selection import SelectionPolicy


def _per_star_npz(R, V, sigma_eps, p, **extras) -> np.lib.npyio.NpzFile:
    """Build an in-memory per-star NpzFile with valid _meta."""
    meta = {
        "catalog_granularity": "per_star",
        "R_unit": "kpc",
        "source_paper_bibcode": "fake2026Test",
    }
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
    reg = {"rhalf_major_pc": 50.0}  # cut = 0.1 kpc
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
    reg = {"rhalf_major_pc": 100.0}  # cut at 0.2 kpc — both pass
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
    npz = _per_epoch_npz(
        star_id=[0, 0, 0],
        V=[100.0, 100.0, 100.0],
        sigma_eps=[1.0, 1.0, 1.0],
        p=[1.0, 1.0, 1.0],
        R=[0.001, 0.001, 0.001],
    )
    reg = {"rhalf_major_pc": 100.0}
    pol = CombinePolicy(sigma_sys_kms=2.0, p_threshold=0.05)
    arrays, audit = prepare_jeans_input(npz, reg, combine_policy=pol)
    # σ_combined = √(1/3 + 4) = √4.333 ≈ 2.082
    np.testing.assert_allclose(arrays["sigma_eps"], np.sqrt(1/3 + 4.0), rtol=1e-12)
    assert audit["combine_policy"]["sigma_sys_kms"] == 2.0
    assert audit["combine_policy"]["p_threshold"] == 0.05


def test_selection_policy_threaded_through():
    npz = _per_star_npz(
        R=[0.001, 0.001],
        V=[100.0, 100.0],
        sigma_eps=[1.0, 1.0],
        p=[0.7, 0.3],
    )
    reg = {"rhalf_major_pc": 100.0}
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
        prepare_jeans_input(npz, {"rhalf_major_pc": 100.0})


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
        prepare_jeans_input(npz, {"rhalf_major_pc": 100.0})


def test_per_star_default_granularity_when_meta_absent():
    """A plain dict with no _meta defaults to per_star."""
    cat = {
        "R": np.array([0.001, 0.5]),
        "V": np.array([100.0, 99.0]),
        "sigma_eps": np.array([1.0, 1.0]),
        "p": np.array([1.0, 1.0]),
    }
    reg = {"rhalf_major_pc": 50.0}
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
        select_jeans_stars(npz, {"rhalf_major_pc": 100.0})
