"""Personal neural training checks using independent CPU-only fixtures."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from experiments.seungbeom import neural_experiment as N  # noqa: E402
from experiments.seungbeom.neural_config import normalize_config  # noqa: E402
from preprocessing import SensorPreprocessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ["meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p"]


def make_data(root):
    rng = np.random.default_rng(914)
    for split, count in (("train", 12), ("validation", 6)):
        folder = root / split
        (folder / "images").mkdir(parents=True)
        rows = []
        for index in range(count):
            sid = f"{split}_{index:03d}"
            if split == "validation" and index < 2:
                sid = ("00001", "NA")[index]
            rel = f"images/{index}.png"
            Image.fromarray(rng.integers(0, 256, (12, 17, 3), dtype=np.uint8)).save(folder / rel)
            rows.append(
                {
                    "sample_id": sid,
                    "experiment_id": f"{split}_{index // 6}",
                    "relative_time_s": float(index % 6),
                    "image_path": rel,
                    **{
                        name: float(index + j + (100 if split == "validation" else 0))
                        for j, name in enumerate(SENSORS)
                    },
                    "label": index % 6 + 1,
                }
            )
        frame = pd.DataFrame(rows)
        if split == "train":
            frame.loc[0, "meas_p"] = np.nan
        frame.to_csv(folder / f"{split}.csv", index=False)


def config_file(root, **overrides):
    config = normalize_config(
        {
            "encoder": "small_cnn",
            "fusion": "gated",
            "image_size": 32,
            "sensor_hidden": 4,
            "fusion_hidden": 8,
            "projection_dim": 4,
            "dropout": 0.0,
            "epochs": 2,
            "batch_size": 6,
            "early_stopping_patience": 0,
            **overrides,
        }
    )
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def cli(*arguments):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/seungbeom/neural_experiment.py"),
            *map(str, arguments),
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-5000:]
    return result.stdout


def train_args(data, config, run):
    return N.parser().parse_args(
        [
            "train",
            "--data_dir",
            str(data),
            "--config",
            str(config),
            "--run_dir",
            str(run),
            "--device",
            "cpu",
        ]
    )


def test_early_stopping_ties_and_patience_boundary():
    state = N.EarlyStopping(patience=2)
    assert state.update(0.3, 1) == (True, False)
    assert state.update(0.3, 2) == (False, False)
    assert state.update(0.2, 3) == (False, True)
    assert state.best_score == 0.3
    assert state.best_epoch == 1
    assert state.bad_epochs == 2


def test_early_stopping_small_improvements_save_best_without_resetting_patience():
    state = N.EarlyStopping(patience=3, min_delta=0.05)
    assert state.update(0.4, 1) == (True, False)
    assert state.update(0.42, 2) == (True, False)
    assert state.bad_epochs == 1 and state.reference_score == 0.4
    assert state.update(0.46, 3) == (True, False)
    assert state.bad_epochs == 0 and state.reference_score == 0.46
    assert state.update(0.47, 4) == (True, False)
    assert state.update(0.48, 5) == (True, False)
    assert state.update(0.49, 6) == (True, True)
    assert state.best_epoch == 6 and state.best_score == 0.49


def test_early_stopping_disabled_still_tracks_best():
    state = N.EarlyStopping(patience=0)
    assert state.update(0.8, 1) == (True, False)
    for epoch in range(2, 12):
        assert state.update(0.1, epoch) == (False, False)
    assert state.best_epoch == 1


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_early_stopping_rejects_nonfinite_score(score):
    with pytest.raises(ValueError):
        N.EarlyStopping().update(score, 1)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("personal_neural_synthetic")
    data, run = root / "synthetic", root / "run"
    make_data(data)
    config = config_file(root)
    cli("train", "--data_dir", data, "--config", config, "--run_dir", run, "--device", "cpu")
    return data, run


def test_cli_train_without_test_data_and_best_checkpoint_roundtrip(trained):
    data, run = trained
    assert not (data / "test").exists()
    assert (run / "best.pt").is_file()
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    best = json.loads((run / "best_metrics.json").read_text())
    assert np.isfinite(best["macro_f1"])
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["label_values"] == [1, 2, 3, 4, 5, 6]
    assert (run / "identity.json").is_file()
    assert cli("report", "--run_dir", run)


def test_predictions_preserve_id_order_repeat_and_subset(trained, tmp_path):
    data, run = trained
    validation = pd.read_csv(
        data / "validation/validation.csv", dtype={"sample_id": str}, keep_default_na=False
    )

    def infer(frame, name):
        input_path, output = tmp_path / f"{name}_input.csv", tmp_path / f"{name}.csv"
        frame.drop(columns="label").to_csv(input_path, index=False)
        cli(
            "predict",
            "--input",
            input_path,
            "--image_root",
            data / "validation",
            "--checkpoint",
            run / "best.pt",
            "--output",
            output,
            "--device",
            "cpu",
        )
        result = pd.read_csv(output, dtype={"sample_id": str}, keep_default_na=False)
        assert result["sample_id"].tolist() == frame["sample_id"].tolist()
        assert list(result.columns) == ["sample_id", "label"]
        assert result["label"].isin([1, 2, 3, 4, 5, 6]).all()
        return result.set_index("sample_id")["label"]

    full = infer(validation, "full")
    pd.testing.assert_series_equal(infer(validation, "repeat"), full)
    shuffled = validation.iloc[[4, 1, 5, 0, 2, 3]]
    pd.testing.assert_series_equal(infer(shuffled, "shuffle"), full.loc[shuffled["sample_id"]])
    subset = validation.iloc[[1, 3, 5]]
    pd.testing.assert_series_equal(infer(subset, "subset"), full.loc[subset["sample_id"]])


def test_sensor_preprocessor_is_fitted_only_on_train(tmp_path, monkeypatch):
    data = tmp_path / "synthetic"
    make_data(data)
    config = config_file(tmp_path, epochs=1)
    fits = []
    original = SensorPreprocessor.fit.__func__

    def tracked(cls, frame, *args, **kwargs):
        fits.append(frame["experiment_id"].tolist())
        assert frame["experiment_id"].str.startswith("train_").all()
        return original(cls, frame, *args, **kwargs)

    monkeypatch.setattr(SensorPreprocessor, "fit", classmethod(tracked))
    N.train(train_args(data, config, tmp_path / "run"))
    assert len(fits) == 1


def test_final_evaluation_reloads_strict_best_checkpoint(tmp_path, monkeypatch):
    data, run = tmp_path / "synthetic", tmp_path / "run"
    make_data(data)
    config = config_file(tmp_path, epochs=3, early_stopping_patience=2)
    original_evaluate = N.evaluate
    snapshots = []
    expected_scores = (0.8, 0.4, 0.5)
    final_checked = []

    def controlled(model, loader, device, **kwargs):
        metrics = original_evaluate(model, loader, device, **kwargs)
        if len(snapshots) < len(expected_scores):
            snapshots.append(
                {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            )
            metrics["macro_f1"] = expected_scores[len(snapshots) - 1]
        else:
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor.cpu(), snapshots[0][name], rtol=0, atol=0)
            final_checked.append(True)
            metrics["macro_f1"] = expected_scores[0]
        return metrics

    monkeypatch.setattr(N, "evaluate", controlled)
    N.train(train_args(data, config, run))
    assert len(snapshots) == 3
    assert final_checked, "The final evaluation must use the reloaded best checkpoint."


def test_nonfinite_validation_is_failed_not_complete(tmp_path, monkeypatch):
    data, run = tmp_path / "synthetic", tmp_path / "run"
    make_data(data)
    config = config_file(tmp_path, epochs=1)
    original_evaluate = N.evaluate

    def invalid(model, loader, device, **kwargs):
        result = original_evaluate(model, loader, device, **kwargs)
        result["macro_f1"] = float("nan")
        return result

    monkeypatch.setattr(N, "evaluate", invalid)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        N.train(train_args(data, config, run))
    assert json.loads((run / "status.json").read_text())["status"] == "failed"


def test_frozen_vector_training_without_images_and_cache_identity(tmp_path):
    data, embeddings, run = tmp_path / "synthetic", tmp_path / "vectors", tmp_path / "run"
    make_data(data)
    rng = np.random.default_rng(918)
    for split in ("train", "validation"):
        frame = pd.read_csv(
            data / split / f"{split}.csv", dtype={"sample_id": str}, keep_default_na=False
        )
        cache = embeddings / split
        cache.mkdir(parents=True)
        # Only these test-created images are removed: cached mode must not open them.
        shutil.rmtree(data / split / "images")
        frame.iloc[::-1][["sample_id"]].to_csv(cache / "sample_ids.csv", index=False)
        np.save(cache / "features.npy", rng.normal(size=(len(frame), 512)).astype(np.float32))
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model": "resnet18",
                    "dimension": 512,
                    "weights": "SYNTHETIC_TEST_RANDOM_WEIGHTS",
                    "trained": False,
                    "preprocessing": {"test_fixture": "independent_synthetic"},
                    "l2_normalize": False,
                }
            )
        )
    config = config_file(tmp_path, encoder="embedding", epochs=1)
    arguments = [
        "train",
        "--data_dir",
        data,
        "--config",
        config,
        "--run_dir",
        run,
        "--train_embeddings",
        embeddings / "train",
        "--val_embeddings",
        embeddings / "validation",
        "--device",
        "cpu",
    ]
    cli(*arguments)
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["visual_contract"]["trained"] is False
    assert checkpoint["model_config"]["encoder"] == "embedding"
    identity_arguments = dict(
        train_embeddings=embeddings / "train", val_embeddings=embeddings / "validation"
    )
    normalized = json.loads(config.read_text())
    original_identity = N.run_identity(
        normalized,
        data / "train/train.csv",
        data / "validation/validation.csv",
        **identity_arguments,
    )
    values = np.load(embeddings / "train/features.npy")
    values[0, 0] += 0.25
    np.save(embeddings / "train/features.npy", values)
    changed_identity = N.run_identity(
        normalized,
        data / "train/train.csv",
        data / "validation/validation.csv",
        **identity_arguments,
    )
    assert original_identity["input_provenance"] != changed_identity["input_provenance"]
