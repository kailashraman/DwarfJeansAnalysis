"""Unit tests for dwarfjeans.jeans.selection."""

import numpy as np
import pytest

from dwarfjeans.jeans.selection import SelectionPolicy, select_jeans_stars


def _fake_registry(rhalf_major_pc: float) -> dict:
    return {"rhalf_major_pc": rhalf_major_pc}


def test_all_pass_no_rows_removed():
    cat = {
        "R": np.array([0.001, 0.002, 0.003]),  # kpc
        "p": np.array([1.0, 1.0, 1.0]),
        "V": np.array([100.0, 101.0, 99.0]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)  # 2*20pc/1000 = 0.04 kpc → all pass
    out, rep = select_jeans_stars(cat, reg)
    assert rep["n_input"] == 3
    assert rep["n_final"] == 3
    assert rep["membership_noop"] is True
    np.testing.assert_array_equal(out["R"], cat["R"])


def test_membership_cut_with_graded_p():
    cat = {
        "R": np.array([0.001, 0.001, 0.001, 0.001]),
        "p": np.array([0.9, 0.6, 0.3, 0.1]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["membership_noop"] is False
    assert rep["n_after_p"] == 2  # only 0.9 and 0.6 survive p>0.5
    assert rep["n_final"] == 2
    np.testing.assert_array_equal(out["p"], np.array([0.9, 0.6]))


def test_p_uniformly_one_is_noop():
    cat = {
        "R": np.array([0.001, 0.001, 0.001]),
        "p": np.array([1.0, 1.0, 1.0]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["membership_noop"] is True
    assert rep["n_after_p"] == 3


def test_p_absent_is_noop():
    cat = {"R": np.array([0.001, 0.002])}
    reg = _fake_registry(rhalf_major_pc=20.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["membership_noop"] is True
    assert rep["n_final"] == 2


def test_radial_cut_matches_hand_calc():
    # rhalf_major_pc = 100 → 2 × 100/1000 = 0.2 kpc cut
    cat = {
        "R": np.array([0.05, 0.15, 0.25, 0.35]),  # kpc
        "p": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    reg = _fake_registry(rhalf_major_pc=100.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["R_max_kpc"] == pytest.approx(0.2)
    assert rep["n_after_R"] == 2  # 0.05 and 0.15
    np.testing.assert_array_equal(out["R"], np.array([0.05, 0.15]))


def test_var_flag_dropped():
    cat = {
        "R": np.array([0.001, 0.001, 0.001]),
        "p": np.array([1.0, 1.0, 1.0]),
        "var_flag": np.array([False, True, False]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["n_after_var"] == 2
    np.testing.assert_array_equal(out["var_flag"], np.array([False, False]))


def test_geha_var_column_dropped():
    """Geha Path A catalogs use ``Var`` (float, 1.0=variable)."""
    cat = {
        "R": np.array([0.001, 0.001, 0.001]),
        "p": np.array([1.0, 1.0, 1.0]),
        "Var": np.array([0.0, 1.0, 0.0]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["n_after_var"] == 2


def test_variability_off_keeps_all():
    cat = {
        "R": np.array([0.001, 0.001]),
        "p": np.array([1.0, 1.0]),
        "var_flag": np.array([True, True]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)
    policy = SelectionPolicy(drop_variable=False)
    out, rep = select_jeans_stars(cat, reg, policy)
    assert rep["n_final"] == 2


def test_missing_R_column_raises():
    cat = {"p": np.array([1.0])}
    reg = _fake_registry(rhalf_major_pc=20.0)
    with pytest.raises(KeyError, match="R"):
        select_jeans_stars(cat, reg)


def test_invalid_rhalf_raises():
    cat = {"R": np.array([0.001])}
    reg = _fake_registry(rhalf_major_pc=0.0)
    with pytest.raises(ValueError):
        select_jeans_stars(cat, reg)


def test_combined_cuts_compose():
    cat = {
        "R": np.array([0.05, 0.15, 0.05, 0.05, 0.05]),
        "p": np.array([0.9, 0.9, 0.3, 0.9, 0.9]),
        "var_flag": np.array([False, False, False, True, False]),
    }
    reg = _fake_registry(rhalf_major_pc=100.0)  # cut at 0.2 kpc
    out, rep = select_jeans_stars(cat, reg)
    # row 0: pass all
    # row 1: R=0.15 < 0.2, p=0.9, no var → pass
    # row 2: p=0.3 fails membership
    # row 3: var_flag=True fails variability
    # row 4: pass all
    assert rep["n_final"] == 3
    assert rep["n_input"] == 5
    np.testing.assert_array_equal(out["R"], np.array([0.05, 0.15, 0.05]))


def test_npz_compatibility():
    """select_jeans_stars accepts an actual NpzFile from np.load."""
    import io
    buf = io.BytesIO()
    np.savez(buf,
             R=np.array([0.05, 0.15, 0.30]),
             p=np.array([1.0, 1.0, 1.0]),
             V=np.array([100.0, 101.0, 99.0]))
    buf.seek(0)
    npz = np.load(buf)
    reg = _fake_registry(rhalf_major_pc=100.0)
    out, rep = select_jeans_stars(npz, reg)
    assert rep["n_final"] == 2
