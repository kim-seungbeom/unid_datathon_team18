"""Sensor contracts using invented values only; no competition fixtures."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from dataset import BatteryDataset
from model import FusionClassifier
from preprocessing import SensorPreprocessor
from runtime import ROOT, load_config
from schema import RAW_SENSOR_COLUMNS, read_frame
from sensor_features import (
    TEMPERATURE_SENSOR_FEATURES,
    build_sensor_features,
    validate_feature_names,
)


def invented_frame():
    return pd.DataFrame(
        {
            "meas_v": [2, 3, 4],
            "meas_a": [7, 7, 7],
            "meas_t1": [10, 2, np.nan],
            "meas_t2": [4, 11, 30],
            "meas_p": ["", np.inf, "invalid"],
            "sample_id": ["a", "b", "c"],
            "label": [1, 2, 3],
            "experiment_id": ["invented"] * 3,
            "relative_time_s": [0, 5, 10],
        }
    )


def test_feature_formulas_order_and_missing_semantics():
    frame = invented_frame()
    names = ("temp_diff", "meas_v", "temp_mean", "temp_abs_diff")
    actual = build_sensor_features(frame, names)
    assert actual.columns.tolist() == list(names)
    np.testing.assert_allclose(actual["temp_diff"], [-6, 9, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual["temp_mean"], [7, 6.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual["temp_abs_diff"], [6, 9, np.nan], equal_nan=True)
    # Derived values never become required raw CSV columns.
    assert set(names) - set(RAW_SENSOR_COLUMNS)
    assert "temp_diff" not in frame


def test_features_ignore_metadata_other_rows_and_column_order():
    frame = invented_frame()
    reference = build_sensor_features(frame, TEMPERATURE_SENSOR_FEATURES)
    altered = frame.iloc[[2, 0]].copy()
    altered["label"] = 6
    altered["experiment_id"] = "changed"
    altered["relative_time_s"] = -123
    altered["sample_id"] = "ignored"
    altered["image_path"] = "ignored"
    altered = altered.loc[:, list(reversed(altered.columns))]
    pd.testing.assert_frame_equal(
        reference.iloc[[2, 0]], build_sensor_features(altered, TEMPERATURE_SENSOR_FEATURES)
    )


@pytest.mark.parametrize(
    "names",
    [
        [],
        "meas_v",
        ["label"],
        ["experiment_id"],
        ["relative_time_s"],
        ["unknown"],
        ["meas_v", "meas_v"],
        [None],
    ],
)
def test_reject_unsafe_or_ambiguous_feature_selection(names):
    with pytest.raises(ValueError):
        validate_feature_names(names)


def test_derived_preprocessor_is_frozen_and_roundtrips():
    frame = invented_frame()
    names = ("temp_diff", "meas_a", "meas_p", "temp_mean")
    preprocessor = SensorPreprocessor.fit(frame, names)
    assert preprocessor.input_dim == 4
    before = preprocessor.to_dict()
    # A constant feature and an entirely unavailable feature remain finite.
    np.testing.assert_array_equal(preprocessor.transform(frame)[:, 1:3], 0)
    validation = frame.copy()
    validation["meas_t2"] = [1000, 2000, 3000]
    actual = preprocessor.transform(validation)
    assert np.isfinite(actual).all() and actual[0, 0] > 100
    assert before == preprocessor.to_dict()
    restored = SensorPreprocessor.from_dict(json.loads(json.dumps(before)))
    assert restored.feature_names == names
    np.testing.assert_array_equal(actual, restored.transform(validation))
    np.testing.assert_array_equal(actual[[2, 0]], restored.transform(validation.iloc[[2, 0]]))


def test_preprocessor_rejects_incompatible_saved_features():
    state = SensorPreprocessor.fit(invented_frame(), TEMPERATURE_SENSOR_FEATURES).to_dict()
    bad = copy.deepcopy(state)
    bad["feature_version"] += 1
    with pytest.raises(ValueError, match="feature version"):
        SensorPreprocessor.from_dict(bad)
    bad = copy.deepcopy(state)
    bad["feature_names"] = list(RAW_SENSOR_COLUMNS)
    with pytest.raises(ValueError, match="Invalid saved"):
        SensorPreprocessor.from_dict(bad)


def test_legacy_config_and_preprocessor_still_work(tmp_path):
    old_config = json.loads((ROOT / "configs/baseline.json").read_text())
    old_config.pop("mode")
    old_config.pop("sensor_features")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(old_config))
    config = load_config(path)
    assert config["mode"] == "multimodal"
    assert config["sensor_features"] == list(RAW_SENSOR_COLUMNS)
    preprocessor = SensorPreprocessor.fit(invented_frame())
    saved = preprocessor.to_dict()
    legacy = {
        "sensor_columns": list(RAW_SENSOR_COLUMNS),
        **{key: saved[key] for key in ("medians", "means", "scales")},
    }
    np.testing.assert_array_equal(
        preprocessor.transform(invented_frame()),
        SensorPreprocessor.from_dict(legacy).transform(invented_frame()),
    )


def test_sensor_only_dataset_never_opens_images(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Image opened during sensor-only ablation.")

    monkeypatch.setattr("dataset.Image.open", forbidden)
    frame = invented_frame()
    path = tmp_path / "invented.csv"
    frame.to_csv(path, index=False)
    loaded = read_frame(path, labeled=True, require_images=False)
    with pytest.raises(ValueError, match="image_path"):
        read_frame(path, labeled=True)
    preprocessor = SensorPreprocessor.fit(loaded, TEMPERATURE_SENSOR_FEATURES)
    dataset = BatteryDataset(
        loaded, tmp_path / "no_images", preprocessor, 16, labeled=True, use_images=False
    )
    assert set(dataset[0]) == {"sensors", "targets"}
    assert dataset[0]["sensors"].shape == (8,)
    model = FusionClassifier(sensor_input_dim=8, mode="sensor_only", dropout=0).eval()
    sensors = torch.randn(2, 8, requires_grad=True)
    output = model(None, sensors)
    assert output.shape == (2, 6)
    torch.testing.assert_close(output, model(torch.randn(2, 3, 16, 16), sensors))
    assert not any(key.startswith("image_encoder.") for key in model.state_dict())
    output.square().sum().backward()
    assert sensors.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="dimension"):
        model(None, torch.ones(2, 5))
