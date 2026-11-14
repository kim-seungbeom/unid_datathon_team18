"""Invented sensor series only: causality, train-only fitting, saved-model replay."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom.neural_config import normalize_config
from experiments.seungbeom.neural_sensors import (
    SLOPE_SOURCES,
    TEMPORAL_VARIANTS,
    NeuralSensorPreprocessor,
    neural_sensor_values,
)
from preprocessing import SensorPreprocessor
from schema import RAW_SENSOR_COLUMNS


def invented_frame():
    # a: values rise 2 units/s; b: values fall 3 units/s. Unequal observation gaps.
    times = np.array([40, 0, 0, 7, 15, 18], dtype=float)
    values = np.array([90, 400, 10, 24, 355, 46], dtype=float)
    return pd.DataFrame(
        {
            "sample_id": [f"invented_{i}" for i in range(6)],
            "experiment_id": ["a", "b", "a", "a", "b", "a"],
            "relative_time_s": times,
            **{name: values + j for j, name in enumerate(RAW_SENSOR_COLUMNS)},
            "label": [1, 2, 3, 4, 5, 6],
        },
        index=[15, 14, 13, 12, 11, 10],
    )


def feature_frame(frame, names=TEMPORAL_VARIANTS["combined30"]):
    return pd.DataFrame(neural_sensor_values(frame, names), index=frame.index, columns=names)


def test_slope_uses_elapsed_seconds_in_current_30s_window():
    frame = invented_frame()
    actual = feature_frame(frame)
    np.testing.assert_allclose(
        actual["meas_t1_slope30s"], [2, np.nan, np.nan, 2, -3, 2], equal_nan=True
    )
    # At 40s, the voltage lag is the most recent observation <=10s (7s).
    assert actual.loc[15, "meas_v_delta30s"] == 90 - 24
    assert actual.loc[15, "meas_v_mean30s"] == (90 + 46) / 2
    # At exactly 37s, 7s is excluded from the slope window (7,37].
    frame.loc[15, "relative_time_s"] = 37
    frame.loc[15, "meas_t1"] = 86
    frame.loc[12, "meas_t1"] = 1e6
    assert feature_frame(frame).loc[15, "meas_t1_slope30s"] == pytest.approx(2)


def test_missing_observations_are_not_imputed_before_fitting_a_slope():
    frame = invented_frame()
    frame.loc[10, "meas_t1"] = np.nan
    frame.loc[12, "meas_t1"] = np.inf
    actual = feature_frame(frame)
    # At 40s, only one finite measurement exists inside (10,40].
    assert np.isnan(actual.loc[15, "meas_t1_slope30s"])
    assert np.isnan(actual.loc[10, "meas_t1_slope30s"])
    assert actual.loc[11, "meas_t1_slope30s"] == pytest.approx(-3)


def test_history_ignores_labels_ids_other_experiments_and_future_values():
    frame = invented_frame()
    expected = feature_frame(frame)
    shuffled = frame.sample(frac=1, random_state=19)
    pd.testing.assert_frame_equal(feature_frame(shuffled), expected.loc[shuffled.index])
    for _, experiment in frame.groupby("experiment_id"):
        pd.testing.assert_frame_equal(feature_frame(experiment), expected.loc[experiment.index])
    changed = frame.copy()
    changed["label"] = 99
    changed["sample_id"] = "not_a_feature"
    changed["experiment_id"] = changed["experiment_id"].map({"a": "x", "b": "y"})
    pd.testing.assert_frame_equal(feature_frame(changed), expected)
    earlier = frame[frame["relative_time_s"] <= 18]
    pd.testing.assert_frame_equal(feature_frame(earlier), expected.loc[earlier.index])
    changed = frame.copy()
    changed.loc[15, list(RAW_SENSOR_COLUMNS)] = 987654
    pd.testing.assert_frame_equal(
        feature_frame(changed).loc[earlier.index], expected.loc[earlier.index]
    )


def test_removing_required_past_rows_is_not_claimed_invariant():
    frame = invented_frame()
    complete = feature_frame(frame)
    isolated = feature_frame(frame.loc[[15]])
    assert np.isfinite(complete.loc[15, "meas_t1_slope30s"])
    assert np.isnan(isolated.loc[15, "meas_t1_slope30s"])


@pytest.mark.parametrize("issue", ["missing_time", "negative", "duplicate", "missing_group"])
def test_temporal_inputs_reject_ambiguous_timing(issue):
    frame = invented_frame()
    if issue == "missing_time":
        frame = frame.drop(columns="relative_time_s")
    elif issue == "negative":
        frame.loc[15, "relative_time_s"] = -1
    elif issue == "duplicate":
        frame.loc[15, "relative_time_s"] = 7
    else:
        frame.loc[15, "experiment_id"] = ""
    with pytest.raises(ValueError):
        feature_frame(frame)


def test_train_only_statistics_and_roundtrip_do_not_accumulate_validation_history():
    frame = invented_frame()
    names = TEMPORAL_VARIANTS["combined30"]
    prep = NeuralSensorPreprocessor.fit(frame, names)
    expected = feature_frame(frame).to_numpy()
    observed_medians = np.nanmedian(expected, axis=0)
    np.testing.assert_allclose(prep.medians, observed_medians)
    filled = np.where(np.isnan(expected), observed_medians, expected)
    np.testing.assert_allclose(prep.means, filled.mean(axis=0))
    frozen = prep.to_dict()
    validation = invented_frame()
    validation.loc[:, list(RAW_SENSOR_COLUMNS)] += 500
    before = prep.transform(validation)
    prep.transform(frame)
    np.testing.assert_array_equal(before, prep.transform(validation))
    assert prep.to_dict() == frozen
    restored = NeuralSensorPreprocessor.from_dict(json.loads(json.dumps(frozen)))
    np.testing.assert_array_equal(before, restored.transform(validation))
    assert prep.uses_history


def test_all_missing_slopes_fall_back_to_zero_and_unit_scale():
    frame = invented_frame().iloc[[0, 1]]  # One point in each experiment.
    prep = NeuralSensorPreprocessor.fit(frame, tuple(SLOPE_SOURCES))
    np.testing.assert_array_equal(prep.medians, [0, 0])
    np.testing.assert_array_equal(prep.scales, [1, 1])
    np.testing.assert_array_equal(prep.transform(frame), np.zeros((2, 2)))


@pytest.mark.parametrize("variant", ["raw", "voltage30", "pressure30"])
def test_existing_preprocessing_codecs_and_results_stay_unchanged(variant):
    frame = invented_frame()
    names = TEMPORAL_VARIANTS[variant]
    legacy = SensorPreprocessor.fit(frame, names)
    new = NeuralSensorPreprocessor.fit(frame, names)
    assert legacy.to_dict() == new.to_dict()
    restored = NeuralSensorPreprocessor.from_dict(legacy.to_dict())
    np.testing.assert_array_equal(legacy.transform(frame), restored.transform(frame))
    assert normalize_config({"sensor_features": list(names)})["sensor_features"] == list(names)


@pytest.mark.parametrize("bad", ["version", "feature", "scale", "imputation", "nan"])
def test_corrupt_private_preprocessors_rejected(bad):
    prep = NeuralSensorPreprocessor.fit(invented_frame(), TEMPORAL_VARIANTS["temperature30"])
    state = copy.deepcopy(prep.to_dict())
    if bad == "version":
        state["formula_version"] += 1
    elif bad == "feature":
        state["feature_names"][-1] = "label"
    elif bad == "scale":
        state["scales"][-1] = 0
    elif bad == "imputation":
        state["imputation"] = "forward_fill"
    else:
        state["medians"][-1] = float("nan")
    with pytest.raises(ValueError):
        NeuralSensorPreprocessor.from_dict(state)


def test_dataset_precomputes_history_before_shuffle_and_batches(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.seungbeom.neural_data import VisualDataset

    frame = invented_frame()
    names = TEMPORAL_VARIANTS["combined30"]
    prep = NeuralSensorPreprocessor.fit(frame, names)
    config = normalize_config({"sensor_features": list(names)})
    vectors = np.zeros((len(frame), 512), dtype=np.float32)
    dataset = VisualDataset(frame, tmp_path, prep, config, vectors=vectors)
    for index in [4, 0, 2, 3, 1, 5]:
        torch.testing.assert_close(
            dataset[index]["sensors"], torch.from_numpy(prep.transform(frame)[index])
        )


def test_temporal_neural_train_save_predict_on_independent_cpu_fixtures(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from experiments.seungbeom import neural_experiment as trainer

    data = tmp_path / "invented"
    for split in ("train", "validation"):
        folder = data / split
        folder.mkdir(parents=True)
        frame = invented_frame().reset_index(drop=True)
        frame["experiment_id"] = split + "_" + frame["experiment_id"]
        frame["sample_id"] = [f"{split}_{i}" for i in range(len(frame))]
        frame["image_path"] = "invented.png"
        if split == "validation":
            frame.loc[:, list(RAW_SENSOR_COLUMNS)] += 1000
        frame.to_csv(folder / f"{split}.csv", index=False)
        Image.new("RGB", (32, 32), (20, 30, 40)).save(folder / "invented.png")
    config = normalize_config(
        {
            "encoder": "small_cnn",
            "image_size": 32,
            "epochs": 1,
            "batch_size": 3,
            "sensor_hidden": 4,
            "fusion_hidden": 8,
            "dropout": 0.0,
            "sensor_features": list(TEMPORAL_VARIANTS["combined30"]),
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    run = tmp_path / "run"
    fits = []
    original = NeuralSensorPreprocessor.fit.__func__

    def only_train(cls, frame, *args, **kwargs):
        assert frame["experiment_id"].str.startswith("train_").all()
        fits.append(len(frame))
        return original(cls, frame, *args, **kwargs)

    monkeypatch.setattr(NeuralSensorPreprocessor, "fit", classmethod(only_train))
    trainer.main(
        [
            "train",
            "--config",
            str(config_path),
            "--data_dir",
            str(data),
            "--run_dir",
            str(run),
            "--device",
            "cpu",
        ]
    )
    assert fits == [6]
    assert not (data / "test").exists()
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert saved["preprocessor"]["kind"] == "seungbeom_neural_temporal_sensors"
    assert saved["model_config"]["sensor_dim"] == 11
    summary = json.loads((run / "summary.json").read_text())
    assert summary["sensor_history"]
    assert summary["arbitrary_subset_invariance"].startswith("not_guaranteed")

    validation = pd.read_csv(data / "validation/validation.csv").drop(columns="label")

    def predict(frame, name):
        input_path, output_path = tmp_path / f"{name}_input.csv", tmp_path / f"{name}_output.csv"
        frame.to_csv(input_path, index=False)
        trainer.main(
            [
                "predict",
                "--input",
                str(input_path),
                "--image_root",
                str(data / "validation"),
                "--checkpoint",
                str(run / "best.pt"),
                "--output",
                str(output_path),
                "--device",
                "cpu",
            ]
        )
        result = pd.read_csv(output_path)
        assert result["sample_id"].tolist() == frame["sample_id"].tolist()
        assert result["label"].isin(range(1, 7)).all()
        return result.set_index("sample_id")["label"]

    expected = predict(validation, "full")
    shuffled = validation.sample(frac=1, random_state=23)
    pd.testing.assert_series_equal(
        predict(shuffled, "shuffle"), expected.loc[shuffled["sample_id"]]
    )
    earlier = validation[validation["relative_time_s"] <= 18]
    pd.testing.assert_series_equal(predict(earlier, "prefix"), expected.loc[earlier["sample_id"]])
    for i, (_, frame) in enumerate(validation.groupby("experiment_id")):
        pd.testing.assert_series_equal(
            predict(frame, f"experiment_{i}"), expected.loc[frame["sample_id"]]
        )
    assert fits == [6], "Prediction must never refit preprocessing."
