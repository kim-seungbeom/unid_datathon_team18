"""CV3 export tests use real tiny CPU trees/Torch and invented image vectors."""

# ruff: noqa: E402
import argparse
import copy
import shutil

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
joblib = pytest.importorskip("joblib")
pytest.importorskip("sklearn")

from test_seungbeom_group_cv import invented as invented

from experiments.seungbeom import group_cv as cv
from experiments.seungbeom import group_cv_export as export
from experiments.seungbeom import group_cv_training as training
from experiments.seungbeom import midpoint_blend as midpoint
from experiments.seungbeom import multimodal as mm
from experiments.seungbeom.export_submissions import read_json, sha256, write_json
from experiments.seungbeom.neural_config import model_config
from experiments.seungbeom.neural_model import NeuralFusionClassifier
from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor
from runtime import seed_everything
from sensor_features import build_sensor_features


@pytest.fixture
def case(invented, tmp_path, monkeypatch):
    args, train, vectors, _, tree, config, contract, old_model, old_csv = invented
    frame = train.iloc[:12].copy().reset_index(drop=True)
    frame["sample_id"] = [f"val_{i:04d}" for i in range(len(frame))]
    frame["experiment_id"] = "independent_validation_experiment"
    validation = tmp_path / "invented_validation.csv"
    frame.to_csv(validation, index=False)
    by_id = dict(zip(frame["sample_id"], vectors[: len(frame)], strict=True))
    calls = []

    def features(current):
        sensor = build_sensor_features(current).reset_index(drop=True).add_prefix("sensor_")
        image = pd.DataFrame(
            np.stack([by_id[sid] for sid in current["sample_id"]]),
            columns=[f"img_{i:04d}" for i in range(512)],
        )
        return pd.concat([sensor, image], axis=1)

    def extract(current, csv_path, image_root, spec, **kwargs):
        assert "label" not in current
        assert csv_path is None and kwargs.get("embeddings_dir") is None
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        calls.append(current["sample_id"].tolist())
        return features(current), copy.deepcopy(contract)

    monkeypatch.setattr(mm, "make_features", extract)
    seed_everything(42)
    model = NeuralFusionClassifier(**model_config(config)).eval()
    prep = NeuralSensorPreprocessor.fit(train)
    probs = {
        "neural": training.neural_probabilities(
            model, prep, frame, vectors[:12], torch.device("cpu")
        ),
        "extra_trees": mm.predict_proba_six_stable(tree, features(frame)),
    }
    probs["blend_50_50"] = midpoint.blend_probabilities(probs["neural"], probs["extra_trees"])
    base = read_json(args.recipe)
    base["metrics"] = {
        name: training.classification_metrics(frame.label, prob.argmax(axis=1) + 1)
        for name, prob in probs.items()
    }
    base["provenance"]["validation_csv_sha256"] = sha256(validation)
    base["runtime_signature"] = midpoint.runtime_signature("cpu")
    base["image_precision"] = "fp32"
    write_json(args.recipe, base)
    original_loader = export.blend_ratios.load_base

    def load(base, device):
        loaded = original_loader(base, device)
        loaded.update(model=model, preprocessor=prep)
        loaded["bundle"].update(sensor_spec=mm.sensor_spec(), embedding=contract)
        return loaded

    monkeypatch.setattr(export.blend_ratios, "load_base", load)
    cv.run(args)  # real synthetic nested CV and serialized fold pipelines
    compare_args = argparse.Namespace(
        recipe=args.recipe,
        cv_dir=args.output_dir,
        input=validation,
        image_root=tmp_path,
        output_dir=tmp_path / "comparison",
        device="cpu",
    )
    return {
        "compare": compare_args,
        "cv_args": args,
        "frame": frame,
        "calls": calls,
        "features": features,
        "old_model": old_model,
        "old_csv": old_csv,
    }


def prediction_args(case, tmp_path, *, suffix="full", frame=None, template=True):
    current = case["frame"] if frame is None else frame
    input_csv = tmp_path / f"test_{suffix}.csv"
    current.to_csv(input_csv, index=False)
    sample = tmp_path / f"template_{suffix}.csv"
    pd.DataFrame({"sample_id": current["sample_id"].tolist()[::-1], "label": 0}).to_csv(
        sample, index=False
    )
    return argparse.Namespace(
        recipe=case["compare"].output_dir / "recipe.json",
        input=input_csv,
        image_root=tmp_path,
        sample_submission=sample if template else None,
        output_dir=tmp_path / f"submission_{suffix}",
        device="cpu",
    )


def test_compare_then_export_exact_probability_average_without_any_fit(case, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline

    old = {path: path.read_bytes() for path in (case["old_model"], case["old_csv"])}

    def cannot_fit(*args, **kwargs):
        raise AssertionError("Export must never fit a model or preprocessor")

    monkeypatch.setattr(Pipeline, "fit", cannot_fit)
    recipe = export.compare(case["compare"])
    assert recipe["replay_matches"] == dict.fromkeys(midpoint.METHODS, True)
    assert len(case["calls"]) == 1
    args = prediction_args(case, tmp_path)
    export.predict(args)
    assert len(case["calls"]) == 2  # one live extraction per command, shared by all members
    expected = (
        np.stack(
            [
                mm.predict_proba_six_stable(joblib.load(path), case["features"](case["frame"]))
                for path in recipe["models"]
            ]
        )
        .mean(axis=0)
        .argmax(axis=1)
        + 1
    )
    output = pd.read_csv(
        args.output_dir / "submission_extratrees_cv3.csv", dtype={"sample_id": str}
    )
    assert list(output.columns) == ["sample_id", "label"]
    assert output.sample_id.tolist() == case["frame"].sample_id.tolist()[::-1]
    np.testing.assert_array_equal(output.label, expected[::-1])
    assert output.label.between(1, 6).all()
    for path, contents in old.items():
        assert path.read_bytes() == contents


def test_predict_needs_no_train_validation_caches_or_original_models(case, tmp_path, monkeypatch):
    export.compare(case["compare"])
    case["cv_args"].train_csv.unlink()
    shutil.rmtree(case["cv_args"].train_embeddings)
    case["compare"].input.unlink()
    case["compare"].recipe.unlink()
    case["old_model"].unlink()
    args = prediction_args(case, tmp_path)

    def never_load_baseline(*args, **kwargs):
        raise AssertionError("Predict must not reload original trained members")

    monkeypatch.setattr(export.blend_ratios, "load_base", never_load_baseline)
    export.predict(args)
    assert read_json(args.output_dir / "status.json")["status"] == "complete"


def test_subset_shuffle_and_supplied_labels_do_not_change_predictions(case, tmp_path):
    export.compare(case["compare"])
    full = prediction_args(case, tmp_path)
    export.predict(full)
    expected = (
        pd.read_csv(full.output_dir / "submission_extratrees_cv3.csv").set_index("sample_id").label
    )
    frame = case["frame"].iloc[[7, 2, 0, 9]].copy()
    frame["label"] = "ignored_input_label"
    partial = prediction_args(case, tmp_path, suffix="partial", frame=frame, template=False)
    export.predict(partial)
    actual = pd.read_csv(partial.output_dir / "submission_extratrees_cv3.csv")
    assert actual.sample_id.tolist() == frame.sample_id.tolist()
    np.testing.assert_array_equal(actual.label, expected.loc[actual.sample_id])


@pytest.mark.parametrize(
    "problem", ["incomplete", "source", "reference", "fold_hash", "fold_result", "summary_hash"]
)
def test_compare_rejects_incompatible_cv(case, problem):
    directory = case["compare"].cv_dir
    if problem == "incomplete":
        write_json(directory / "status.json", {"status": "failed"})
    elif problem in ("source", "reference"):
        manifest = read_json(directory / "experiment.json")
        key = "source_sha256" if problem == "source" else "input_sha256"
        manifest[key] = {}
        write_json(directory / "experiment.json", manifest)
    elif problem == "summary_hash":
        write_json(directory / "summary.json", {})
    else:
        marker = read_json(directory / "fold_1.json")
        filename = "extra_trees.joblib" if problem == "fold_hash" else "result.json"
        (directory / marker["attempt"] / filename).write_bytes(b"changed")
    with pytest.raises(ValueError):
        export.compare(case["compare"])
    assert not case["compare"].output_dir.exists()


def test_wrong_compare_csv_is_rejected_before_reading_rows(case, monkeypatch):
    case["compare"].input = case["cv_args"].train_csv
    monkeypatch.setattr(export, "read_frame", lambda *a, **k: pytest.fail("unexpected row read"))
    with pytest.raises(ValueError, match="Validation CSV"):
        export.compare(case["compare"])


def test_baseline_replay_failure_prevents_recipe(case, monkeypatch):
    predict = training.neural_probabilities

    def wrong_predictions(*args, **kwargs):
        predict(*args, **kwargs)
        labels = case["frame"].label.to_numpy() % 6
        return np.eye(6)[labels]

    monkeypatch.setattr(training, "neural_probabilities", wrong_predictions)
    with pytest.raises(ValueError, match="do not replay"):
        export.compare(case["compare"])
    assert not (case["compare"].output_dir / "recipe.json").exists()


def test_predict_rejects_modified_weights_before_deserialization(case, tmp_path, monkeypatch):
    recipe = export.compare(case["compare"])
    from pathlib import Path

    Path(recipe["models"][0]).write_bytes(b"changed model")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("must check hashes first"))
    args = prediction_args(case, tmp_path)
    with pytest.raises(ValueError, match="files changed"):
        export.predict(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("problem", ["weights", "runtime", "source", "template", "existing"])
def test_predict_rejects_invalid_export(case, tmp_path, problem):
    recipe = export.compare(case["compare"])
    args = prediction_args(case, tmp_path)
    if problem == "weights":
        recipe["rule"] = {**recipe["rule"], "weights": [1, 0, 0]}
    elif problem == "runtime":
        recipe["runtime"]["device"] = "unexpected"
    elif problem == "source":
        recipe["source_sha256"] = {}
    elif problem == "template":
        args.sample_submission.write_text("sample_id,label\nmissing,0\n")
    else:
        args.output_dir.mkdir()
    write_json(args.recipe, recipe)
    with pytest.raises(ValueError):
        export.predict(args)
    assert not (args.output_dir / "submission_extratrees_cv3.csv").exists()


@pytest.mark.parametrize("problem", ["missing", "negative", "nan", "sum", "shape"])
def test_probability_average_rejects_invalid_members(problem):
    values = [np.full((4, 6), 1 / 6) for _ in range(3)]
    if problem == "missing":
        values.pop()
    elif problem == "negative":
        values[0][0, 0] = -1
    elif problem == "nan":
        values[1][0, 0] = np.nan
    elif problem == "sum":
        values[2] *= 0.5
    else:
        values[0] = values[0][:2]
    with pytest.raises(ValueError):
        export.probability_mean(values)
