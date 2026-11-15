"""Invented irregular-time experiments; no competition data or artifacts."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from check_submission import check_arbitrary_subset
from preprocessing import SensorPreprocessor
from runtime import ROOT
from schema import RAW_SENSOR_COLUMNS
from sensor_features import NONTHERMAL_HISTORY_FEATURES, build_sensor_features


def invented_frame():
    # Deliberately unordered, with uneven gaps and distinct experiment values.
    return pd.DataFrame(
        {
            "experiment_id": ["a", "b", "a", "a", "b", "a"],
            "relative_time_s": [40, 0, 0, 7, 15, 18],
            **{name: [np.nan, 99, np.nan, 2, np.nan, 8] for name in RAW_SENSOR_COLUMNS},
        },
        index=[15, 14, 13, 12, 11, 10],
    )


def test_forward_fill_only_previous_values_and_same_experiment():
    frame = invented_frame()
    features = build_sensor_features(frame, imputation="forward_fill")
    np.testing.assert_allclose(features["meas_v"], [8, 99, np.nan, 2, 99, 8], equal_nan=True)
    assert features.index.tolist() == frame.index.tolist()
    # Current temperature features are derived after filling the original sensors.
    frame.loc[15, "meas_t2"] = 12
    actual = build_sensor_features(frame, ["temp_diff"], imputation="forward_fill")
    assert actual.loc[15, "temp_diff"] == 4


def test_window_uses_seconds_observed_values_and_last_valid_lag():
    frame = invented_frame()
    features = build_sensor_features(frame, NONTHERMAL_HISTORY_FEATURES, imputation="forward_fill")
    # At t=40, the window is (10,40]; only the actual observation at t=18 is finite.
    assert features.loc[15, "meas_v_mean30s"] == 8
    # Most recent valid observation at/before t=10 is t=7, value 2.
    assert features.loc[15, "meas_v_delta30s"] == 6
    assert features.loc[10, "meas_v_mean30s"] == 5
    assert np.isnan(features.loc[10, "meas_v_delta30s"])
    boundary = frame.copy()
    boundary.loc[15, "relative_time_s"] = 37
    actual = build_sensor_features(boundary, NONTHERMAL_HISTORY_FEATURES, imputation="forward_fill")
    assert actual.loc[15, "meas_v_mean30s"] == 8  # t=7 excluded from (7,37].
    assert actual.loc[15, "meas_v_delta30s"] == 6  # t=7 included in lag <= 7.


def test_fallback_is_train_only_and_transform_has_no_cross_call_history():
    train = invented_frame()
    prep = SensorPreprocessor.fit(train, imputation="forward_fill")
    # Fallback is median of observed raw Train values [99, 2, 8], not forward-filled copies.
    assert prep.medians[0] == 8
    val = pd.DataFrame(
        {
            "experiment_id": ["val"] * 3,
            "relative_time_s": [0, 4, 20],
            **{name: [np.nan, 500, np.nan] for name in RAW_SENSOR_COLUMNS},
        }
    )
    before = prep.to_dict()
    np.testing.assert_array_equal(prep.impute(val)[:, 0], [8, 500, 500])
    prep.transform(train)
    np.testing.assert_array_equal(prep.impute(val)[:, 0], [8, 500, 500])
    np.testing.assert_array_equal(prep.impute(val.iloc[[2]])[:, 0], [8])
    assert prep.to_dict() == before
    restored = SensorPreprocessor.from_dict(json.loads(json.dumps(before)))
    np.testing.assert_array_equal(prep.transform(val), restored.transform(val))
    bad = copy.deepcopy(before)
    bad["temporal_version"] += 1
    with pytest.raises(ValueError, match="temporal version"):
        SensorPreprocessor.from_dict(bad)


def test_history_invariant_to_shuffle_other_experiments_and_future_changes():
    frame = invented_frame()
    names = NONTHERMAL_HISTORY_FEATURES
    expected = build_sensor_features(frame, names, imputation="forward_fill")
    shuffled = frame.sample(frac=1, random_state=13)
    pd.testing.assert_frame_equal(
        expected.loc[shuffled.index],
        build_sensor_features(shuffled, names, imputation="forward_fill"),
    )
    earlier = frame[frame.relative_time_s <= 18]
    pd.testing.assert_frame_equal(
        expected.loc[earlier.index],
        build_sensor_features(earlier, names, imputation="forward_fill"),
    )
    changed = frame.copy()
    changed.loc[15, list(RAW_SENSOR_COLUMNS)] = 999999  # Future relative to t<=18.
    modified = build_sensor_features(changed, names, imputation="forward_fill")
    pd.testing.assert_frame_equal(expected.loc[earlier.index], modified.loc[earlier.index])
    for _, group in frame.groupby("experiment_id"):
        pd.testing.assert_frame_equal(
            expected.loc[group.index],
            build_sensor_features(group, names, imputation="forward_fill"),
        )


@pytest.mark.parametrize(
    "issue", ["missing_time", "nan_time", "negative_time", "duplicate", "empty_id"]
)
def test_reject_ambiguous_history(issue):
    frame = invented_frame()
    if issue == "missing_time":
        frame = frame.drop(columns=["relative_time_s"])
    elif issue == "nan_time":
        frame.loc[15, "relative_time_s"] = np.nan
    elif issue == "negative_time":
        frame.loc[15, "relative_time_s"] = -1
    elif issue == "duplicate":
        frame.loc[15, "relative_time_s"] = 7
    else:
        frame.loc[15, "experiment_id"] = ""
    with pytest.raises(ValueError):
        build_sensor_features(frame, imputation="forward_fill")


def test_subset_diagnostic_does_not_mask_current_row_or_strict_failures(capsys):
    reference = pd.DataFrame({"sample_id": ["a", "b"], "label": [1, 2]})
    candidate = pd.DataFrame({"sample_id": ["b"], "label": [3]})
    assert (
        check_arbitrary_subset(reference, candidate, uses_history=True, diagnose_history=True)
        is False
    )
    assert "DIAGNOSTIC" in capsys.readouterr().out
    for kwargs in ({}, {"uses_history": True}, {"diagnose_history": True}):
        with pytest.raises(ValueError):
            check_arbitrary_subset(reference, candidate, **kwargs)


def test_v2_checkpoint_preprocessing_still_uses_median():
    saved = SensorPreprocessor.fit(invented_frame()).to_dict()
    saved["format_version"] = 2
    saved.pop("imputation")
    saved.pop("temporal_version")
    restored = SensorPreprocessor.from_dict(saved)
    assert restored.imputation == "median" and not restored.uses_history


def test_forward_fill_does_not_change_complete_raw_inputs():
    frame = invented_frame().fillna(3)
    median = SensorPreprocessor.fit(frame)
    forward = SensorPreprocessor.fit(frame, imputation="forward_fill")
    np.testing.assert_array_equal(median.transform(frame), forward.transform(frame))


def test_temporal_training_and_reload_inference(tmp_path):
    data = tmp_path / "invented_data"
    for split in ("train", "validation"):
        folder = data / split
        folder.mkdir(parents=True)
        frame = invented_frame().reset_index(drop=True)
        frame["sample_id"] = [f"{split}_{i}" for i in range(len(frame))]
        frame["experiment_id"] = split + "_" + frame["experiment_id"]
        frame["label"] = [1, 2, 3, 4, 5, 6]
        frame.to_csv(folder / f"{split}.csv", index=False)
    config = json.loads((ROOT / "configs/sensor_ffill_history.json").read_text())
    config.update(epochs=1, batch_size=3, sensor_hidden=8, fusion_hidden=8)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    run = tmp_path / "run"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(script, *arguments):
        process = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, arguments)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert process.returncode == 0, process.stdout + process.stderr

    command(
        "train.py", "--data_dir", data, "--config", config_path, "--run_dir", run, "--device", "cpu"
    )
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    prep = SensorPreprocessor.from_dict(saved["preprocessor"])
    assert prep.imputation == "forward_fill" and prep.uses_history and prep.input_dim == 11
    full = tmp_path / "full.csv"
    command(
        "test.py",
        "--input",
        data / "validation/validation.csv",
        "--image_root",
        tmp_path,
        "--checkpoint",
        run / "best.pt",
        "--output",
        full,
        "--device",
        "cpu",
    )
    expected = pd.read_csv(full).set_index("sample_id")["label"]
    for i, subset in enumerate(
        (frame.sample(frac=1, random_state=13), frame[frame.relative_time_s <= 18])
    ):
        input_path, output_path = tmp_path / f"input{i}.csv", tmp_path / f"output{i}.csv"
        subset.to_csv(input_path, index=False)
        command(
            "test.py",
            "--input",
            input_path,
            "--image_root",
            tmp_path,
            "--checkpoint",
            run / "best.pt",
            "--output",
            output_path,
            "--device",
            "cpu",
        )
        actual = pd.read_csv(output_path)
        np.testing.assert_array_equal(expected.loc[actual.sample_id], actual.label)
