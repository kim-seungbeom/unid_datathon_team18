"""Train-only scaling and binary missing indicators from independent synthetic inputs."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from preprocessing import SensorPreprocessor
from schema import RAW_SENSOR_COLUMNS


def invented_frame(values):
    return pd.DataFrame(
        {
            **{name: list(values) for name in RAW_SENSOR_COLUMNS},
            "experiment_id": ["synthetic_train"] * len(values),
            "relative_time_s": np.arange(len(values), dtype=float) * 7,
        }
    )


def test_robust_uses_linear_train_quartiles_after_median_fill():
    train = invented_frame([1, 2, 3, 4, 100, np.nan])
    prep = SensorPreprocessor.fit(train, ["meas_v"], scaling="robust")
    # Filled values [1, 2, 3, 4, 100, 3] have Q25=2.25, Q50=3, Q75=3.75.
    np.testing.assert_array_equal(prep.medians, [3])
    np.testing.assert_array_equal(prep.centers, [3])
    np.testing.assert_array_equal(prep.scales, [1.5])
    assert prep.centers is prep.means and prep.scaling == "robust"
    expected = ((np.array([1, 2, 3, 4, 100, 3]) - 3) / 1.5).astype(np.float32)
    np.testing.assert_array_equal(prep.transform(train)[:, 0], expected)
    assert prep.transform(train)[4, 0] > 60  # The extreme observation is retained.


def test_robust_centers_use_causal_filled_train_but_fallback_uses_observed_train():
    train = invented_frame([np.nan, 1, np.nan, np.nan, 9, 100])
    prep = SensorPreprocessor.fit(train, ["meas_v"], imputation="forward_fill", scaling="robust")
    assert prep.medians[0] == 9  # Median of observed [1, 9, 100].
    np.testing.assert_array_equal(prep.impute(train)[:, 0], [9, 1, 1, 1, 9, 100])
    assert prep.centers[0] == 5 and prep.scales[0] == 8


def test_robust_validation_is_frozen_and_v4_roundtrips():
    train = invented_frame([1, 2, 3, 4, 100, np.nan])
    prep = SensorPreprocessor.fit(train, ["meas_v"], scaling="robust")
    state = copy.deepcopy(prep.to_dict())
    assert state["format_version"] == 4 and state["scaling"] == "robust"
    assert state["means"] == [3]  # Legacy serialized key stores the selected center.
    validation = invented_frame([10000, np.nan, -10000])
    validation["experiment_id"] = "synthetic_validation"
    actual = prep.transform(validation)
    np.testing.assert_array_equal(actual[:, 0], np.array([9997 / 1.5, 0, -10003 / 1.5], np.float32))
    assert prep.to_dict() == state
    restored = SensorPreprocessor.from_dict(json.loads(json.dumps(state)))
    assert restored.scaling == "robust"
    assert restored.to_dict() == state
    np.testing.assert_array_equal(actual, restored.transform(validation))


@pytest.mark.parametrize("scaling", ["standard", "robust"])
def test_constant_all_missing_and_near_zero_spread_remain_finite(scaling):
    train = invented_frame([np.nan, np.nan, np.nan, np.nan])
    train["meas_v"] = 7.0
    train["meas_t1"] = [0, 1e-10, 2e-10, 3e-10]
    prep = SensorPreprocessor.fit(train, scaling=scaling)
    np.testing.assert_array_equal(prep.scales, np.ones(len(RAW_SENSOR_COLUMNS)))
    output = prep.transform(train)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[:, :2], 0)
    assert prep.medians[1] == 0 and prep.centers[1] == 0


def test_standard_scaling_keeps_existing_numerical_behavior():
    train = invented_frame([1, 2, 3, 4, 100, np.nan])
    train["meas_p"] = 11
    train["meas_a"] = np.nan
    raw = train.loc[:, list(RAW_SENSOR_COLUMNS)].to_numpy(dtype=float)
    expected_medians = np.array([3, 0, 3, 3, 11], dtype=float)
    filled = np.where(np.isnan(raw), expected_medians, raw)
    expected_scales = filled.std(axis=0)
    expected_scales[expected_scales < 1e-8] = 1
    expected_centers = filled.mean(axis=0)
    expected = ((filled - expected_centers) / expected_scales).astype(np.float32)
    for prep in (SensorPreprocessor.fit(train), SensorPreprocessor.fit(train, scaling="standard")):
        assert prep.scaling == "standard"
        np.testing.assert_array_equal(prep.medians, expected_medians)
        np.testing.assert_array_equal(prep.means, expected_centers)
        np.testing.assert_array_equal(prep.scales, expected_scales)
        np.testing.assert_array_equal(prep.transform(train), expected)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_formats_default_to_standard_scaling(version):
    train = invented_frame([1, 2, np.nan, 10])
    prep = SensorPreprocessor.fit(train)
    state = prep.to_dict()
    state["format_version"] = version
    state.pop("scaling")
    if version < 3:
        state.pop("imputation")
        state.pop("temporal_version")
    if version == 1:
        state["sensor_columns"] = state.pop("raw_sensor_columns")
        state.pop("feature_names")
        state.pop("feature_version")
    restored = SensorPreprocessor.from_dict(state)
    assert restored.scaling == "standard" and restored.imputation == "median"
    np.testing.assert_array_equal(restored.transform(train), prep.transform(train))


def test_v3_history_retains_temporal_compatibility():
    train = invented_frame([1, np.nan, 3, 10])
    prep = SensorPreprocessor.fit(train, imputation="forward_fill")
    state = prep.to_dict()
    state["format_version"] = 3
    state.pop("scaling")
    restored = SensorPreprocessor.from_dict(state)
    assert restored.scaling == "standard" and restored.imputation == "forward_fill"
    np.testing.assert_array_equal(restored.transform(train), prep.transform(train))
    state["temporal_version"] += 1
    with pytest.raises(ValueError, match="temporal version"):
        SensorPreprocessor.from_dict(state)


@pytest.mark.parametrize("scaling", ["unknown", "minmax", "", None])
def test_reject_unknown_scaling_in_fit_and_saved_state(scaling):
    train = invented_frame([1, 2, 3])
    with pytest.raises(ValueError, match="sensor_scaling"):
        SensorPreprocessor.fit(train, scaling=scaling)
    state = SensorPreprocessor.fit(train).to_dict()
    state["scaling"] = scaling
    with pytest.raises(ValueError, match="sensor_scaling"):
        SensorPreprocessor.from_dict(state)


def test_legacy_format_cannot_claim_robust_scaling():
    state = SensorPreprocessor.fit(invented_frame([1, 2, 100]), scaling="robust").to_dict()
    state["format_version"] = 3
    with pytest.raises(ValueError, match="format_version 4"):
        SensorPreprocessor.from_dict(state)


@pytest.mark.parametrize("scaling", ["standard", "robust"])
@pytest.mark.parametrize("imputation", ["median", "forward_fill"])
def test_missing_flags_stay_binary_in_train_validation_and_roundtrip(scaling, imputation):
    train = invented_frame([1, np.nan, np.nan, np.nan])
    names = ("meas_v", "meas_v_missing")
    prep = SensorPreprocessor.fit(train, names, scaling=scaling, imputation=imputation)
    assert prep.centers[1] == 0 and prep.scales[1] == 1
    np.testing.assert_array_equal(prep.transform(train)[:, 1], [0, 1, 1, 1])
    validation = invented_frame([5, np.nan, np.inf, "invalid"])
    validation["experiment_id"] = "synthetic_validation"
    np.testing.assert_array_equal(prep.transform(validation)[:, 1], [0, 1, 1, 1])
    restored = SensorPreprocessor.from_dict(json.loads(json.dumps(prep.to_dict())))
    np.testing.assert_array_equal(prep.transform(validation), restored.transform(validation))


@pytest.mark.parametrize("scaling", ["standard", "robust"])
def test_first_missing_value_only_in_validation_still_has_flag_one(scaling):
    train = invented_frame([1, 2, 3, 4])
    prep = SensorPreprocessor.fit(train, ["meas_v", "meas_v_missing"], scaling=scaling)
    validation = invented_frame([np.nan, 10])
    np.testing.assert_array_equal(prep.transform(train)[:, 1], [0, 0, 0, 0])
    np.testing.assert_array_equal(prep.transform(validation)[:, 1], [1, 0])


def test_saved_missing_flags_reject_nonbinary_scaling():
    prep = SensorPreprocessor.fit(invented_frame([1, np.nan]), ["meas_v_missing"])
    state = prep.to_dict()
    state["means"] = [0.5]
    with pytest.raises(ValueError, match="missing indicators"):
        SensorPreprocessor.from_dict(state)
