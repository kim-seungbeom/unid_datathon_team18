"""Selected optional columns and ablations, using only independently invented inputs."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from preprocessing import SensorPreprocessor
from runtime import ROOT, load_config
from schema import RAW_SENSOR_COLUMNS
from sensor_features import build_sensor_features, required_sensor_columns, validate_feature_names


def invented_frame():
    return pd.DataFrame(
        {
            **{name: [2, 4, np.nan, 8] for name in RAW_SENSOR_COLUMNS},
            "meas_a": [0, 0, 0, 0],
            "meas_t1": [20, 40, np.nan, 80],
            "meas_t2": [25, 45, 65, 85],
            "atmp_t": [10, 15, 20, np.nan],
            "thermal_t": [30, 50, np.inf, 90],
            "meas_co": [1, "", "bad", 100],
            "meas_co2": [2, 3, 4, 5],
            "meas_ch4": [0, 6, np.nan, 8],
            "experiment_id": ["synthetic_a"] * 4,
            "relative_time_s": [0, 12, 38, 70],
        },
        index=[9, 4, 7, 1],
    )


def test_optional_columns_required_only_when_selected():
    frame = invented_frame()
    base = frame.loc[:, list(RAW_SENSOR_COLUMNS)]
    pd.testing.assert_frame_equal(build_sensor_features(base), build_sensor_features(frame))
    assert required_sensor_columns(RAW_SENSOR_COLUMNS) == RAW_SENSOR_COLUMNS
    assert required_sensor_columns(["temp1_minus_ambient"]) == (*RAW_SENSOR_COLUMNS, "atmp_t")
    with pytest.raises(ValueError, match="meas_co"):
        build_sensor_features(base, ["meas_co_missing"])
    with pytest.raises(ValueError, match="atmp_t"):
        build_sensor_features(base, ["temp2_minus_ambient"])
    # A present-but-empty sensor is supported; a missing column is a contract error.
    frame["meas_co"] = np.nan
    prep = SensorPreprocessor.fit(frame, ["meas_co", "meas_co_missing"])
    np.testing.assert_array_equal(prep.transform(frame), [[0, 1]] * len(frame))


def test_gas_ambient_and_thermal_formulas_and_row_locality():
    frame = invented_frame()
    before = frame.copy(deep=True)
    names = ["thermal_minus_ambient", "temp2_minus_ambient", "temp1_minus_ambient", "meas_co"]
    result = build_sensor_features(frame, names)
    np.testing.assert_allclose(
        result,
        [[20, 15, 10, 1], [35, 30, 25, np.nan], [np.nan, 45, np.nan, np.nan], [np.nan] * 3 + [100]],
        equal_nan=True,
    )
    assert result.columns.tolist() == names and result.index.tolist() == frame.index.tolist()
    subset = frame.iloc[[2, 0]].assign(label=6, sample_id="ignored", relative_time_s=-99)
    pd.testing.assert_frame_equal(result.iloc[[2, 0]], build_sensor_features(subset, names))
    pd.testing.assert_frame_equal(frame, before)


def test_missing_flags_precede_fill_and_optional_ffill_is_causal():
    frame = invented_frame()
    names = ["meas_co", "meas_co_missing", "thermal_t_missing", "temp1_minus_ambient"]
    result = build_sensor_features(frame, names, imputation="forward_fill")
    np.testing.assert_array_equal(result["meas_co"], [1, 1, 1, 100])
    np.testing.assert_array_equal(result["meas_co_missing"], [0, 1, 1, 0])
    np.testing.assert_array_equal(result["thermal_t_missing"], [0, 0, 1, 0])
    np.testing.assert_array_equal(result["temp1_minus_ambient"], [10, 25, 20, 60])
    earlier = frame.iloc[:3]
    pd.testing.assert_frame_equal(
        result.iloc[:3], build_sensor_features(earlier, names, imputation="forward_fill")
    )


@pytest.mark.parametrize("version", [2, 3])
def test_old_formats_cannot_silently_accept_extended_features(version):
    state = SensorPreprocessor.fit(invented_frame(), ["meas_co"]).to_dict()
    state["format_version"] = version
    with pytest.raises(ValueError, match="Extended sensor features"):
        SensorPreprocessor.from_dict(state)


@pytest.mark.parametrize("name", ["label_missing", "bat_class", "indu_t", "sample_id_missing"])
def test_unregistered_columns_are_not_model_features(name):
    with pytest.raises(ValueError, match="Unknown sensor feature"):
        validate_feature_names([name])


def test_ablation_configs_hold_other_training_settings_fixed():
    base = load_config(ROOT / "configs/sensor_raw.json")
    expected_counts = {
        "sensor_history_voltage": 7,
        "sensor_history_current": 7,
        "sensor_history_pressure": 7,
        "sensor_gas": 8,
        "sensor_ambient": 7,
        "sensor_thermal": 6,
        "sensor_thermal_ambient": 7,
        "sensor_missing": 10,
        "sensor_raw_robust": 5,
        "sensor_gas_robust": 8,
        "sensor_gas_missing": 11,
        "sensor_no_current": 4,
    }
    for name, count in expected_counts.items():
        config = load_config(ROOT / f"configs/{name}.json")
        assert len(config["sensor_features"]) == count
        assert config["sensor_scaling"] == ("robust" if name.endswith("_robust") else "standard")
        assert {
            k: v for k, v in config.items() if k not in {"sensor_features", "sensor_scaling"}
        } == {k: v for k, v in base.items() if k not in {"sensor_features", "sensor_scaling"}}
    for name, sensor in (("voltage", "meas_v"), ("current", "meas_a"), ("pressure", "meas_p")):
        config = load_config(ROOT / f"configs/sensor_history_{name}.json")
        assert config["sensor_features"] == [
            *RAW_SENSOR_COLUMNS,
            f"{sensor}_mean30s",
            f"{sensor}_delta30s",
        ]


def test_extended_scaling_training_and_saved_inference(tmp_path):
    data_root = tmp_path / "invented_data"
    for split in ("train", "validation"):
        folder = data_root / split
        folder.mkdir(parents=True)
        frame = invented_frame().reset_index(drop=True)
        frame["sample_id"] = [f"{split}_{i}" for i in range(len(frame))]
        frame["experiment_id"] = f"synthetic_{split}"
        frame["label"] = [1, 2, 5, 6]
        frame.to_csv(folder / f"{split}.csv", index=False)
    config = load_config(ROOT / "configs/sensor_gas_robust.json")
    config["sensor_features"] += ["meas_co_missing", "temp1_minus_ambient", "thermal_minus_ambient"]
    config.update(epochs=1, batch_size=2, sensor_hidden=8, fusion_hidden=8)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(dict(config, sensor_scaling="unrecognized")))
    with pytest.raises(ValueError, match="sensor_scaling"):
        load_config(config_path)
    config_path.write_text(json.dumps(config))
    run = tmp_path / "invented_run"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(script, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    command(
        "train.py",
        "--data_dir",
        data_root,
        "--config",
        config_path,
        "--run_dir",
        run,
        "--device",
        "cpu",
    )
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    assert checkpoint["format_version"] == 4
    prep = SensorPreprocessor.from_dict(checkpoint["preprocessor"])
    assert prep.scaling == "robust" and prep.input_dim == 11
    expected_features = prep.transform(frame)
    assert np.isfinite(expected_features).all()
    assert set(expected_features[:, 8]) <= {0, 1}
    original = tmp_path / "original.csv"
    command(
        "test.py",
        "--input",
        data_root / "validation/validation.csv",
        "--image_root",
        tmp_path,
        "--checkpoint",
        run / "best.pt",
        "--output",
        original,
        "--device",
        "cpu",
    )
    expected = pd.read_csv(original).set_index("sample_id")["label"]
    subset = frame.iloc[[3, 1]].copy()
    subset_path, output_path = tmp_path / "subset.csv", tmp_path / "subset_predictions.csv"
    subset.to_csv(subset_path, index=False)
    command(
        "test.py",
        "--input",
        subset_path,
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
    np.testing.assert_array_equal(actual["label"], expected.loc[actual["sample_id"]])
