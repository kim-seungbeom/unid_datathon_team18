"""Tiny independent Train/Validation fixtures for controlled loss/sampling experiments."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from runtime import ROOT, load_config
from schema import RAW_SENSOR_COLUMNS


def test_repeat_seed_configs_change_only_seed():
    for name in ("sensor_raw", "sensor_history_voltage"):
        base = json.loads((ROOT / f"configs/{name}.json").read_text())
        for seed in (43, 44):
            actual = json.loads((ROOT / f"configs/{name}_seed{seed}.json").read_text())
            assert actual == dict(base, seed=seed)


def test_imbalance_configs_change_one_training_intervention():
    variants = {
        "weighted_ce": {"loss": "weighted_cross_entropy"},
        "focal": {"loss": "focal"},
        "balanced_sampling": {"sampling": "balanced"},
    }
    for name in ("sensor_raw", "sensor_history_voltage"):
        base = load_config(ROOT / f"configs/{name}.json")
        for variant, change in variants.items():
            actual = load_config(ROOT / f"configs/{name}_{variant}.json")
            assert actual == dict(base, **change)


@pytest.mark.parametrize("method", ["weighted_ce", "focal", "balanced_sampling"])
def test_train_only_objective_metadata_and_saved_inference(tmp_path, method):
    data = tmp_path / "invented_data"
    train_labels = [1] * 6 + [2] * 2 + [3, 4, 5, 6]
    validation_labels = [1, 2, 3, 4, 5, 6]
    for split, labels in (("train", train_labels), ("validation", validation_labels)):
        folder = data / split
        folder.mkdir(parents=True)
        n = len(labels)
        frame = pd.DataFrame(
            {
                "sample_id": [f"{split}_{i}" for i in range(n)],
                "experiment_id": f"synthetic_{split}",
                "relative_time_s": np.arange(n) * 11,
                **{
                    name: np.arange(n, dtype=float) + j for j, name in enumerate(RAW_SENSOR_COLUMNS)
                },
                "label": labels,
            }
        )
        frame.to_csv(folder / f"{split}.csv", index=False)
    base = "sensor_history_voltage" if method == "balanced_sampling" else "sensor_raw"
    config = load_config(ROOT / f"configs/{base}_{method}.json")
    config.update(epochs=1, batch_size=4, sensor_hidden=8, fusion_hidden=8)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(dict(config, loss="invalid")))
    with pytest.raises(ValueError, match="loss"):
        load_config(config_path)
    config_path.write_text(json.dumps(config))
    run = tmp_path / "invented_run"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(script, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    command(
        "train.py", "--data_dir", data, "--config", config_path, "--run_dir", run, "--device", "cpu"
    )
    metadata = json.loads((run / "training_objective.json").read_text())
    assert metadata["train_class_counts"] == {"1": 6, "2": 2, "3": 1, "4": 1, "5": 1, "6": 1}
    assert metadata["sampler"]["num_samples"] == len(train_labels)
    assert metadata["loss"] == config["loss"] and metadata["sampling"] == config["sampling"]
    assert metadata["sampler"]["replacement"] == (method == "balanced_sampling")
    metrics = json.loads((run / "best_metrics.json").read_text())
    assert metrics["support"] == {str(label): 1 for label in range(1, 7)}
    assert (
        set(metrics["class_precision"]) == set(metrics["class_recall"]) == set(metrics["support"])
    )
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert saved["training_objective"] == metadata
    output = tmp_path / "invented_predictions.csv"
    command(
        "test.py",
        "--input",
        data / "validation/validation.csv",
        "--image_root",
        tmp_path,
        "--checkpoint",
        run / "best.pt",
        "--output",
        output,
        "--device",
        "cpu",
    )
    predictions = pd.read_csv(output)
    assert predictions.sample_id.tolist() == frame.sample_id.tolist()
    assert predictions.label.between(1, 6).all()
