"""All fixtures are generated from scratch. Never read competition data or outputs."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from check_submission import compare_predictions, read_submission
from dataset import BatteryDataset
from evaluate import classification_metrics
from model import FusionClassifier
from preprocessing import SensorPreprocessor
from runtime import ROOT, SOURCE_FILES
from schema import SENSOR_COLUMNS, read_frame, validate_labels, validate_split
from scripts.check_staged import blocked_path


def synthetic_frame(count: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{i:05}" for i in range(count)],
            "image_path": [f"images/synthetic_{i}.png" for i in range(count)],
            "experiment_id": ["synthetic_experiment"] * count,
            "relative_time_s": np.arange(count) * 5,
            **{column: np.arange(count, dtype=float) for column in SENSOR_COLUMNS},
            "label": np.arange(count) % 6 + 1,
        }
    )


def test_train_only_preprocessing_and_roundtrip():
    train = synthetic_frame()
    train["meas_a"] = ""
    preprocessor = SensorPreprocessor.fit(train)
    before = preprocessor.to_dict()
    validation = synthetic_frame(2)
    validation["meas_v"] = [10000, ""]
    validation["meas_p"] = [np.inf, -np.inf]
    actual = preprocessor.transform(validation)
    assert np.isfinite(actual).all()
    assert actual[0, 0] > 1000  # No validation refit or clipping to its own distribution.
    assert preprocessor.to_dict() == before
    np.testing.assert_array_equal(
        actual, SensorPreprocessor.from_dict(before).transform(validation)
    )


def test_csv_identity_and_label_validation(tmp_path):
    frame = synthetic_frame(2)
    frame["sample_id"] = ["00001", "NA"]
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    loaded = read_frame(path, labeled=True)
    assert loaded["sample_id"].tolist() == ["00001", "NA"]
    frame["sample_id"] = ["same", "same"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_frame(path)


@pytest.mark.parametrize("values", [[0], [7], [1.5], [None], [float("inf")]])
def test_reject_invalid_labels(values):
    with pytest.raises(ValueError, match="1..6"):
        validate_labels(values)


def test_group_split_and_six_class_macro():
    train = synthetic_frame()
    validation = synthetic_frame()
    validation["sample_id"] = "val_" + validation["sample_id"]
    with pytest.raises(ValueError, match="experiment IDs overlap"):
        validate_split(train, validation)
    validation["experiment_id"] = "validation_experiment"
    validate_split(train, validation)
    assert classification_metrics([1, 1], [1, 1])["macro_f1"] == pytest.approx(1 / 6)
    assert classification_metrics([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])["macro_f1"] == 1.0


def test_image_root_is_explicit(tmp_path):
    frame = synthetic_frame(1)
    frame["image_path"] = "../outside.png"
    dataset = BatteryDataset(frame, tmp_path, SensorPreprocessor.fit(frame), 16)
    with pytest.raises(ValueError, match="outside"):
        dataset[0]


def test_both_modalities_reach_the_model():
    torch.manual_seed(42)
    model = FusionClassifier(dropout=0).eval()
    images = torch.rand(2, 3, 16, 16, requires_grad=True)
    sensors = torch.rand(2, 5, requires_grad=True)
    model(images, sensors).square().sum().backward()
    assert images.grad.abs().sum() > 0
    assert sensors.grad.abs().sum() > 0


def test_sensitive_paths_are_not_stageable():
    for name in (
        "open/file.py",
        "weights/model.pt",
        "runs/config.json",
        "raw.csv",
        ".env",
        "a.pt.tmp",
    ):
        assert blocked_path(name)
    for name in ("model.py", "configs/baseline.json", "README.md", ".env.example"):
        assert not blocked_path(name)


def test_submission_checker_rejects_changed_predictions_and_float_labels(tmp_path):
    reference = pd.DataFrame({"sample_id": ["00001", "NA"], "label": [1, 2]})
    altered = reference.copy()
    altered.loc[0, "label"] = 6
    with pytest.raises(ValueError, match="Predictions changed"):
        compare_predictions(reference, altered)
    path = tmp_path / "synthetic_float_labels.csv"
    reference.assign(label=[1.0, 2.0]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="stored as integers"):
        read_submission(path, ["00001", "NA"])


def test_synthetic_training_and_submission_contract(tmp_path):
    # Reproduce the documented split-relative image paths without using any actual data.
    generator = np.random.default_rng(1234)
    data_root = tmp_path / "synthetic_data"
    for split, count in (("train", 12), ("validation", 6), ("test", 6)):
        folder = data_root / split
        (folder / "images").mkdir(parents=True)
        frame = synthetic_frame(count)
        frame["sample_id"] = split + "_" + frame["sample_id"]
        if split == "test":
            frame["sample_id"] = ["00001", "NA", "00003", "00004", "00005", "00006"]
        frame["experiment_id"] = [f"{split}_group_{i // 3}" for i in range(count)]
        for relative_path in frame["image_path"]:
            pixels = generator.integers(0, 256, (20, 24, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / relative_path)
        if split == "test":
            frame = frame.drop(columns=["label"])
        frame.to_csv(folder / f"{split}.csv", index=False)
    template = frame[["sample_id"]].assign(label=1)
    template.to_csv(data_root / "sample_submission.csv", index=False)

    config = json.loads((ROOT / "configs/baseline.json").read_text())
    config.update(image_size=16, epochs=1, batch_size=6, sensor_hidden=8, fusion_hidden=8)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # A code-only temporary submission package tests the exact three-argument CLI.
    package = tmp_path / "submission_package"
    package.mkdir()
    for name in (*SOURCE_FILES, "check_submission.py"):
        shutil.copy2(ROOT / name, package / name)
    run_dir = tmp_path / "synthetic_run"
    weights = package / "weights/model.pt"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(*arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)],
            cwd=package,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    command(
        "train.py",
        "--data_dir",
        data_root,
        "--config",
        config_path,
        "--run_dir",
        run_dir,
        "--model_out",
        weights,
        "--device",
        "cpu",
    )
    for name in ("best.pt", "config.json", "preprocessing.json", "environment.json", "split.json"):
        assert (run_dir / name).is_file()
    original = tmp_path / "original.csv"
    command(
        "test.py",
        "--input",
        data_root / "test/test.csv",
        "--image_root",
        data_root / "test",
        "--output",
        original,
    )
    expected = read_submission(original, frame["sample_id"].tolist())
    # The checkpoint's preprocessing must remain fixed; extra label/metadata are never features.
    modified = frame.copy()
    modified["label"] = [6, 5, 4, 3, 2, 1]
    modified["relative_time_s"] = 999999
    modified["experiment_id"] = "irrelevant"
    modified_path = tmp_path / "modified.csv"
    modified.to_csv(modified_path, index=False)
    modified_output = tmp_path / "modified_output.csv"
    command(
        "test.py",
        "--input",
        modified_path,
        "--image_root",
        data_root / "test",
        "--output",
        modified_output,
        "--batch_size",
        "1",
    )
    compare_predictions(expected, read_submission(modified_output, frame["sample_id"].tolist()))
    report = command("check_submission.py", "--data_dir", data_root, "--final", original)
    assert report.count("PASS:") == 6
