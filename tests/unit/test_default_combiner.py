"""Unit tests for the default per-epoch → per-star combiner.

Covers in particular the per-instrument zero-point offset hook: when
``CombinePolicy.zero_point_offsets_kms`` is non-empty, ``default.combine``
must (a) require an ``Inst`` column on the per_epoch dict, (b) raise on
unknown instrument tags, and (c) shift V additively before the IVW.
"""

import numpy as np
import pytest

from dwarfjeans.ingest.combiners import CombinePolicy
from dwarfjeans.ingest.combiners.default import combine


def _per_epoch_two_inst():
    """One star with 2 IMACS + 2 M2FS epochs at identical underlying v.

    IMACS quoted velocities are shifted by -2.6 km/s relative to M2FS,
    matching the Chiti+2022 §3.4.1 convention. Equal sigma so the IVW
    weights are equal and post-shift v_bar should land exactly at the
    common true value.
    """
    v_true = 100.0
    v_imacs = v_true - 2.6
    v_m2fs = v_true
    return {
        "star_id": np.array(["s1"] * 4),
        "V": np.array([v_imacs, v_imacs, v_m2fs, v_m2fs]),
        "sigma_eps": np.array([1.0, 1.0, 1.0, 1.0]),
        "p": np.array([1.0, 1.0, 1.0, 1.0]),
        "Inst": np.array(["IMACS", "IMACS", "M2FS", "M2FS"]),
    }


def test_default_combine_no_offsets_preserves_input_zero_point():
    """With empty offsets, IVW averages the raw (shifted) velocities —
    the inter-instrument scatter biases v_bar down by 1.3 km/s."""
    per_epoch = _per_epoch_two_inst()
    per_star, diag = combine(per_epoch, registry_row=None, policy=CombinePolicy())
    assert per_star["V"].shape == (1,)
    # Equal weights → v_bar = mean of (97.4, 97.4, 100, 100) = 98.7
    assert per_star["V"][0] == pytest.approx(98.7, abs=1e-6)
    assert diag["zero_point_offsets_kms"] == {}


def test_default_combine_offsets_shift_to_reference():
    """With M2FS as reference, +2.6 added to IMACS rows recovers the
    true common-zero-point v_bar."""
    per_epoch = _per_epoch_two_inst()
    policy = CombinePolicy(zero_point_offsets_kms={"IMACS": +2.6, "M2FS": 0.0})
    per_star, diag = combine(per_epoch, registry_row=None, policy=policy)
    assert per_star["V"][0] == pytest.approx(100.0, abs=1e-6)
    assert diag["zero_point_offsets_kms"] == {"IMACS": 2.6, "M2FS": 0.0}


def test_default_combine_missing_inst_raises():
    """Non-empty offsets without an Inst column is a configuration bug,
    not a silent no-op."""
    per_epoch = _per_epoch_two_inst()
    del per_epoch["Inst"]
    policy = CombinePolicy(zero_point_offsets_kms={"IMACS": +2.6, "M2FS": 0.0})
    with pytest.raises(KeyError, match="Inst"):
        combine(per_epoch, registry_row=None, policy=policy)


def test_default_combine_unknown_instrument_raises():
    """Unknown instrument tag is a paper-vs-policy mismatch — refuse to
    silently apply zero offset to it."""
    per_epoch = _per_epoch_two_inst()
    per_epoch["Inst"] = np.array(["IMACS", "IMACS", "M2FS", "MIKE"])
    policy = CombinePolicy(zero_point_offsets_kms={"IMACS": +2.6, "M2FS": 0.0})
    with pytest.raises(KeyError, match="MIKE"):
        combine(per_epoch, registry_row=None, policy=policy)


def test_default_combine_does_not_mutate_input_V():
    """The shift must happen on a copy — the caller's per_epoch dict is
    not allowed to be modified."""
    per_epoch = _per_epoch_two_inst()
    V_before = per_epoch["V"].copy()
    policy = CombinePolicy(zero_point_offsets_kms={"IMACS": +2.6, "M2FS": 0.0})
    combine(per_epoch, registry_row=None, policy=policy)
    np.testing.assert_array_equal(per_epoch["V"], V_before)
