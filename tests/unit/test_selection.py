"""Unit tests for dwarfjeans.jeans.selection."""

import numpy as np
import pytest

from dwarfjeans.jeans.selection import SelectionPolicy, select_jeans_stars


def _fake_registry(rhalf_major_pc: float, ellipticity: float = 0.0) -> dict:
    return {"rhalf_major_pc": rhalf_major_pc, "ellipticity": ellipticity}


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


def test_membership_cut_with_binary_p_drops_zeros():
    """Catalogs whose `p` is a 0/1 hard flag with non-members retained
    must apply the cut (drop p == 0). Regression for the bug where
    binary p was misread as 'already hard-cut upstream' and the cut
    silently no-op'd."""
    cat = {
        "R": np.array([0.001, 0.001, 0.001, 0.001]),
        "p": np.array([1.0, 0.0, 1.0, 0.0]),
    }
    reg = _fake_registry(rhalf_major_pc=20.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["membership_noop"] is False
    assert rep["n_after_p"] == 2
    np.testing.assert_array_equal(out["p"], np.array([1.0, 1.0]))


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
    # rhalf_major=100 pc, eps=0 → r_½_sph_3d = 100 × 1 × 4/3 = 133.33 pc
    # → R_max = 2 × 133.33 / 1000 = 0.2667 kpc
    cat = {
        "R": np.array([0.05, 0.15, 0.30, 0.40]),  # kpc
        "p": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    reg = _fake_registry(rhalf_major_pc=100.0, ellipticity=0.0)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["R_max_kpc"] == pytest.approx(0.26667, rel=1e-4)
    assert rep["rhalf_sph_3d_pc"] == pytest.approx(133.33, rel=1e-3)
    assert rep["n_after_R"] == 2  # 0.05 and 0.15 below 0.2667; 0.30 and 0.40 above
    np.testing.assert_array_equal(out["R"], np.array([0.05, 0.15]))


def test_radial_cut_applies_ellipticity():
    # rhalf_major=100, eps=0.36 → sqrt(1-0.36)=0.8 → r_½_sph_3d = 100 × 0.8 × 4/3 ≈ 106.67 pc
    # → R_max = 2 × 106.67 / 1000 = 0.2133 kpc
    cat = {
        "R": np.array([0.05, 0.15, 0.21, 0.25]),
        "p": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    reg = _fake_registry(rhalf_major_pc=100.0, ellipticity=0.36)
    out, rep = select_jeans_stars(cat, reg)
    assert rep["R_max_kpc"] == pytest.approx(0.21333, rel=1e-4)
    assert rep["ellipticity"] == 0.36
    assert rep["n_after_R"] == 3  # 0.05, 0.15, 0.21


def test_missing_ellipticity_key_raises():
    # Absent ellipticity KEY is a caller bug (silent fallback to eps=0
    # would bias results). Must raise.
    cat = {"R": np.array([0.001]), "p": np.array([1.0])}
    reg = {"rhalf_major_pc": 100.0}  # no ellipticity key at all
    with pytest.raises(KeyError, match="ellipticity"):
        select_jeans_stars(cat, reg)


def test_nan_ellipticity_defaults_to_zero_with_audit_flag():
    # NaN value (= "no measurement", per Stage 0a convention) is OK and
    # defaults to eps=0; the audit records ellipticity_missing=True so
    # downstream readers can tell that sphericalization was a no-op.
    cat = {"R": np.array([0.001]), "p": np.array([1.0])}
    reg = {"rhalf_major_pc": 100.0, "ellipticity": float("nan")}
    out, rep = select_jeans_stars(cat, reg)
    assert rep["ellipticity"] == 0.0
    assert rep["ellipticity_missing"] is True
    assert rep["rhalf_sph_3d_pc"] == pytest.approx(133.33, rel=1e-3)


def test_invalid_ellipticity_raises():
    cat = {"R": np.array([0.001]), "p": np.array([1.0])}
    with pytest.raises(ValueError):
        select_jeans_stars(cat, _fake_registry(100.0, ellipticity=1.5))
    with pytest.raises(ValueError):
        select_jeans_stars(cat, _fake_registry(100.0, ellipticity=-0.1))


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
    reg = _fake_registry(rhalf_major_pc=100.0)  # cut at 0.2667 kpc
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
    reg = _fake_registry(rhalf_major_pc=100.0)  # cut at 0.2667 kpc
    out, rep = select_jeans_stars(npz, reg)
    assert rep["n_final"] == 2  # 0.05 and 0.15; 0.30 > 0.2667 fails
