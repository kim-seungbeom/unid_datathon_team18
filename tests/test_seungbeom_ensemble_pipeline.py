"""Independent synthetic CLI/OOF checks; no competition files or trained artifacts."""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")

from sklearn.pipeline import Pipeline  # noqa: E402

from experiments.seungbeom import classwise_ensemble as ce  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402

SENSORS = ("meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p")


def create_split(data, embeddings, split, groups, rows):
    folder = data / split
    folder.mkdir(parents=True)
    rng = np.random.default_rng(14 + len(split))
    records = []
    for group in groups:
        for row in range(rows):
            label = row % 6 + 1
            sid = f"{split}_{group}_{row:03d}"
            records.append(
                {
                    "sample_id": sid,
                    "experiment_id": f"experiment_{group}",
                    "relative_time_s": row * 3.0,
                    "image_path": f"images/{sid}.png",
                    **{
                        name: float(label * (i + 1) + rng.normal(0, 0.02))
                        for i, name in enumerate(SENSORS)
                    },
                    "label": label,
                }
            )
    frame = pd.DataFrame(records)
    if split == "validation":
        frame.loc[0, "sample_id"] = "NA"
        frame.loc[1, "sample_id"] = "00001"
    else:
        frame.loc[::7, "meas_p"] = np.nan
    frame.to_csv(folder / f"{split}.csv", index=False)
    vectors = rng.normal(0, 0.2, size=(len(frame), 512)).astype(np.float32)
    vectors[:, :6] += np.eye(6, dtype=np.float32)[frame["label"].to_numpy() - 1] * 2
    cached = embeddings / split
    cached.mkdir(parents=True)
    # Deliberately reverse cache order: the CLI must join by exact sample ID.
    np.save(cached / "features.npy", vectors[::-1])
    frame.iloc[::-1][["sample_id"]].to_csv(cached / "sample_ids.csv", index=False)
    (cached / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "trained": False,
                "model": "resnet18",
                "dimension": 512,
                "rows": len(frame),
                "weights": "SYNTHETIC_TEST_RANDOM_WEIGHTS",
                "l2_normalize": False,
                "preprocessing": {"test_fixture": "independent_synthetic"},
            }
        )
    )
    return frame


def cli(*arguments):
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(ce.ROOT / "experiments/seungbeom/classwise_ensemble.py"),
            *map(str, arguments),
        ],
        cwd=ce.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def train_arguments(data, embeddings, run):
    return [
        "train",
        "--data_dir",
        str(data),
        "--run_dir",
        str(run),
        "--train_embeddings",
        str(embeddings / "train"),
        "--val_embeddings",
        str(embeddings / "validation"),
        "--models",
        "logistic",
        "extra_trees",
        "--folds",
        "3",
        "--pca_dim",
        "4",
        "--max_rounds",
        "1",
        "--device",
        "cpu",
    ]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("seungbeom_synthetic_cli")
    data, embeddings, run = root / "synthetic", root / "embeddings", root / "first_fit"
    train = create_split(data, embeddings, "train", range(3), rows=12)
    validation = create_split(data, embeddings, "validation", range(3, 5), rows=6)
    output = cli(*train_arguments(data, embeddings, run))
    return {
        "root": root,
        "data": data,
        "embeddings": embeddings,
        "run": run,
        "train": train,
        "validation": validation,
        "output": output,
    }


def test_cli_train_and_report_without_any_test_csv(trained):
    assert not (trained["data"] / "test").exists()
    assert "Validation comparison" in trained["output"]
    report = cli("report", "--run_dir", trained["run"], "--class_details")
    assert "Fixed class weights" in report
    assert "Validation class details" in report
    assert "precision" in report and "recall" in report
    status = json.loads((trained["run"] / "status.json").read_text())
    assert status == {"status": "complete"}
    config = json.loads((trained["run"] / "config.json").read_text())
    assert config["weight_fit_split"] == "train_group_oof_only"
    assert config["validation_used_for_fit"] is False
    assert "handler" not in config
    bundle = ce.load_bundle(trained["run"] / "ensemble.joblib")
    assert bundle["models"] == ["logistic", "extra_trees"]
    assert bundle["weights"].shape == (2, 6)
    table = pd.read_csv(trained["run"] / "comparison.tsv", sep="\t")
    assert set(table["method"]) == {
        "logistic",
        "extra_trees",
        "equal_average",
        "classwise_ensemble",
    }


def test_each_train_row_has_one_group_disjoint_oof_prediction(trained):
    frame = trained["train"]
    coverage = np.zeros(len(frame), dtype=int)
    saved = np.load(trained["run"] / "train_oof.npz", allow_pickle=False)
    assert saved["probabilities"].shape == (len(frame), 2, 6)
    assert np.isfinite(saved["probabilities"]).all()
    for fold, (fit_indices, held_indices) in enumerate(ce.group_splits(frame, 3)):
        coverage[held_indices] += 1
        assert set(frame.iloc[fit_indices]["experiment_id"]).isdisjoint(
            frame.iloc[held_indices]["experiment_id"]
        )
        np.testing.assert_array_equal(saved["fold"][held_indices], fold)
    np.testing.assert_array_equal(coverage, np.ones(len(frame), dtype=int))


def test_validation_labels_do_not_change_weights_or_models_and_preprocessing_is_fold_local(
    trained, tmp_path, monkeypatch
):
    data = tmp_path / "changed_validation_labels"
    shutil.copytree(trained["data"], data)
    changed = trained["validation"].copy()
    changed["label"] = changed["label"] % 6 + 1
    changed.to_csv(data / "validation/validation.csv", index=False)
    captured = []
    original_fit = Pipeline.fit

    def record_fit(self, features, labels=None, **kwargs):
        result = original_fit(self, features, labels, **kwargs)
        if "features" in self.named_steps and "classifier" in self.named_steps:
            preprocessor = self.named_steps["features"]
            image_columns = [name for name in features if name.startswith("img_")]
            sensor_columns = [name for name in features if name.startswith("sensor_")]
            scale = preprocessor.named_transformers_["image"].named_steps["scale"]
            imputer = preprocessor.named_transformers_["sensor"]
            np.testing.assert_allclose(
                scale.mean_, features[image_columns].to_numpy(dtype=np.float64).mean(axis=0)
            )
            np.testing.assert_allclose(
                imputer.statistics_,
                features[sensor_columns].median(axis=0),
                equal_nan=True,
            )
            captured.append(tuple(features.index))
        return result

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    run = tmp_path / "second_fit"
    args = ce.parser().parse_args(train_arguments(data, trained["embeddings"], run))
    args.handler(args)
    first = ce.load_bundle(trained["run"] / "ensemble.joblib")
    second = ce.load_bundle(run / "ensemble.joblib")
    np.testing.assert_array_equal(first["weights"], second["weights"])
    old_pred = np.load(trained["run"] / "validation_predictions.npz", allow_pickle=False)
    new_pred = np.load(run / "validation_predictions.npz", allow_pickle=False)
    np.testing.assert_allclose(old_pred["probabilities"], new_pred["probabilities"], atol=1e-14)
    old_metrics = json.loads((trained["run"] / "metrics.json").read_text())
    new_metrics = json.loads((run / "metrics.json").read_text())
    assert old_metrics["validation"] != new_metrics["validation"]
    assert old_metrics["train_oof"] == new_metrics["train_oof"]
    expected = [tuple(indices) for indices, _ in ce.group_splits(trained["train"], 3)]
    assert captured[:3] == expected
    assert captured[3:6] == expected
    assert captured[6:] == [tuple(range(len(trained["train"])))] * 2


def test_prediction_roundtrip_preserves_ids_and_is_order_subset_repeat_invariant(
    trained, tmp_path, monkeypatch
):
    original = mm.make_features
    calls = []

    def fixture_features(frame, csv_path, image_root, spec, **kwargs):
        assert "embeddings_dir" not in kwargs  # production predict always requests fresh images
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        calls.append(frame["sample_id"].tolist())
        return original(
            frame,
            csv_path,
            image_root,
            spec,
            embeddings_dir=trained["embeddings"] / "validation",
            **kwargs,
        )

    monkeypatch.setattr(mm, "make_features", fixture_features)
    frame = trained["validation"]

    def predict_frame(name, selected, template=None):
        csv = tmp_path / f"{name}_input.csv"
        output = tmp_path / f"{name}_prediction.csv"
        selected.to_csv(csv, index=False)
        args = argparse.Namespace(
            input=csv,
            image_root=tmp_path,
            output=output,
            device="cpu",
            checkpoint=trained["run"] / "ensemble.joblib",
            sample_submission=template,
        )
        ce.predict(args)
        result = pd.read_csv(output, dtype={"sample_id": str}, keep_default_na=False)
        assert list(result.columns) == ["sample_id", "label"]
        assert result["label"].isin(range(1, 7)).all()
        return result

    full = predict_frame("full", frame)
    assert full["sample_id"].tolist() == frame["sample_id"].tolist()
    assert full["sample_id"].tolist()[:2] == ["NA", "00001"]
    repeated = predict_frame("repeated", frame)
    pd.testing.assert_frame_equal(full, repeated)
    reordered = predict_frame("shuffled", frame.sample(frac=1.0, random_state=42))
    subset = predict_frame("subset", frame.iloc[[0, 1, 5, 9]])
    for result in (reordered, subset):
        expected = full.set_index("sample_id").loc[result["sample_id"], "label"].to_numpy()
        np.testing.assert_array_equal(result["label"], expected)
    template = tmp_path / "sample_submission.csv"
    frame.iloc[::-1][["sample_id"]].assign(label=1).to_csv(template, index=False)
    result = predict_frame("template", frame, template)
    assert result["sample_id"].tolist() == frame["sample_id"].tolist()[::-1]
    np.testing.assert_array_equal(result["label"], full["label"].to_numpy()[::-1])
    assert calls[3] == frame.iloc[[0, 1, 5, 9]]["sample_id"].tolist()


def test_reject_corrupted_weights(trained, tmp_path):
    bundle = joblib.load(trained["run"] / "ensemble.joblib")
    bundle["weights"] = np.zeros((2, 6))
    path = tmp_path / "corrupt.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="sum to one"):
        ce.load_bundle(path)


def test_training_rejects_overlapping_experiments_before_any_fit(trained, tmp_path):
    data = tmp_path / "overlap"
    shutil.copytree(trained["data"], data)
    validation = trained["validation"].copy()
    validation["experiment_id"] = trained["train"]["experiment_id"].iloc[0]
    validation.to_csv(data / "validation/validation.csv", index=False)
    args = ce.parser().parse_args(
        train_arguments(data, trained["embeddings"], tmp_path / "bad_fit")
    )
    with pytest.raises(ValueError, match="experiment IDs overlap"):
        args.handler(args)
    assert not args.run_dir.exists()
