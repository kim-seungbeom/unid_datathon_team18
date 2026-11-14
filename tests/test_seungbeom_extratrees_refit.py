"""Synthetic CPU-only checks for the authorized Train+Validation refit."""

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

from experiments.seungbeom import extratrees_refit as refit
from experiments.seungbeom import multimodal as mm
from experiments.seungbeom.export_submissions import read_json, sha256, write_json
from runtime import seed_everything
from schema import RAW_SENSOR_COLUMNS
from sensor_features import build_sensor_features


@pytest.fixture
def case(invented, tmp_path, monkeypatch):
    original, train, train_vectors, _, tree, _, contract, old_model, old_csv = invented
    validation = train.iloc[:12].copy().reset_index(drop=True)
    validation["sample_id"] = [f"val_{i}" for i in range(len(validation))]
    validation["experiment_id"] = "independent_validation"
    validation["image_path"] = [f"independent_val_{i}.png" for i in range(len(validation))]
    validation[list(RAW_SENSOR_COLUMNS)] += 20.0
    val_csv = tmp_path / "independent_validation.csv"
    validation.to_csv(val_csv, index=False)
    val_vectors = np.random.default_rng(663).normal(5, 1, (len(validation), 512)).astype(np.float32)
    cache = tmp_path / "validation_cache"
    cache.mkdir()
    np.save(cache / "features.npy", val_vectors[::-1])
    validation.iloc[::-1][["sample_id"]].to_csv(cache / "sample_ids.csv", index=False)
    write_json(cache / "metadata.json", {**contract, "status": "complete", "rows": len(validation)})
    base = read_json(original.recipe)
    base["provenance"]["validation_csv_sha256"] = sha256(val_csv)
    seed_everything(42)
    base["runtime_signature"] = refit.midpoint.runtime_signature("cpu")
    base["image_precision"] = "fp32"
    write_json(original.recipe, base)
    loader = refit.blend_ratios.load_base

    def reference(base, device):
        loaded = loader(base, device)
        loaded["bundle"].update(sensor_spec=mm.sensor_spec(), embedding=contract)
        loaded["identity"]["input_provenance"]["validation"] = {
            name: sha256(cache / name) for name in refit.CACHE_FILES
        }
        # Freeze expected hashes at fixture construction, not on each call.
        return loaded

    loaded = reference(base, "cpu")

    def frozen_reference(base, device):
        seed_everything(42)
        return loaded

    monkeypatch.setattr(refit.blend_ratios, "load_base", frozen_reference)
    args = argparse.Namespace(
        recipe=original.recipe,
        train_csv=original.train_csv,
        validation_csv=val_csv,
        train_embeddings=original.train_embeddings,
        validation_embeddings=cache,
        output_dir=tmp_path / "refit_model",
        device="cpu",
    )
    images = dict(zip(validation.image_path, val_vectors, strict=True))
    calls = []
    original_make_features = mm.make_features

    def make_features(current, csv_path, image_root, spec, **kwargs):
        assert "label" not in current
        if kwargs.get("embeddings_dir") is not None:
            calls.append("cached")
            assert image_root is None
            return original_make_features(current, csv_path, image_root, spec, **kwargs)
        calls.append(current.sample_id.tolist())
        assert csv_path is None and kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        sensor = build_sensor_features(current).reset_index(drop=True).add_prefix("sensor_")
        vectors = np.stack([images[path] for path in current.image_path])
        features = pd.concat(
            [sensor, pd.DataFrame(vectors, columns=[f"img_{i:04d}" for i in range(512)])], axis=1
        )
        return features, copy.deepcopy(contract)

    monkeypatch.setattr(mm, "make_features", make_features)
    return {
        "args": args,
        "train": train,
        "validation": validation,
        "train_vectors": train_vectors,
        "val_vectors": val_vectors,
        "original_tree": tree,
        "reference": loaded,
        "calls": calls,
        "old_model": old_model,
        "old_csv": old_csv,
    }


def prediction_args(case, tmp_path, suffix="full", frame=None, template=True):
    current = case["validation"].copy() if frame is None else frame.copy()
    current.loc[current.index[:2], "sample_id"] = ["00001", "NA"][: min(2, len(current))]
    path = tmp_path / f"test_{suffix}.csv"
    current.to_csv(path, index=False)
    sample = tmp_path / f"template_{suffix}.csv"
    pd.DataFrame({"sample_id": current.sample_id.tolist()[::-1], "label": 0}).to_csv(
        sample, index=False
    )
    return argparse.Namespace(
        model_dir=case["args"].output_dir,
        input=path,
        image_root=tmp_path,
        sample_submission=sample if template else None,
        output_dir=tmp_path / f"submission_{suffix}",
        device="cpu",
    )


def test_combined_inputs_align_cache_ids_and_use_both_label_sets(case):
    args = case["args"]
    features, labels, *_ = refit.load_training_inputs(args)
    assert len(features) == len(case["train"]) + len(case["validation"])
    np.testing.assert_array_equal(
        features.iloc[:, 5:], np.concatenate([case["train_vectors"], case["val_vectors"]])
    )
    np.testing.assert_array_equal(
        labels, np.concatenate([case["train"].label, case["validation"].label])
    )
    assert not set(features.columns) & {"label", "sample_id", "experiment_id", "image_path"}
    assert case["calls"] == ["cached", "cached"]


def test_refit_relearns_all_preprocessing_and_trees_on_combined_data(case, monkeypatch):
    from sklearn.base import clone

    args = case["args"]
    originals = {
        path: path.read_bytes() for path in (args.recipe, case["old_model"], case["old_csv"])
    }
    original_tree = case["original_tree"]
    parameter_hash = joblib.hash(clone(original_tree))
    old_mean = (
        original_tree.named_steps["features"]
        .named_transformers_["image"]
        .named_steps["scale"]
        .mean_.copy()
    )
    read_paths = []
    reader = refit.read_frame

    def permitted(path, **kwargs):
        assert path in (args.train_csv, args.validation_csv)
        read_paths.append(path)
        return reader(path, **kwargs)

    monkeypatch.setattr(refit, "read_frame", permitted)
    metadata = refit.train(args)
    assert read_paths == [args.train_csv, args.validation_csv]
    model = joblib.load(args.output_dir / "model.joblib")
    assert joblib.hash(clone(model)) == parameter_hash
    assert metadata["fit_rows"] == 84
    assert metadata["evaluation"]["validation_macro_f1"] is None
    assert metadata["evaluation"]["held_out"] is False
    assert metadata["fit_splits"] == ["train", "validation"]
    sensor_values = pd.concat([case["train"], case["validation"]])[list(RAW_SENSOR_COLUMNS)]
    prep = model.named_steps["features"].named_transformers_
    np.testing.assert_allclose(prep["sensor"].statistics_, np.nanmedian(sensor_values, axis=0))
    expected_vectors = np.concatenate([case["train_vectors"], case["val_vectors"]])
    np.testing.assert_allclose(
        prep["image"].named_steps["scale"].mean_, expected_vectors.mean(axis=0, dtype=np.float64)
    )
    assert not np.allclose(old_mean, prep["image"].named_steps["scale"].mean_)
    forest = model.named_steps["classifier"].estimator_
    assert all(tree.tree_.n_node_samples[0] == 84 for tree in forest.estimators_)
    assert all(
        tree.tree_.n_node_samples[0] == 72
        for tree in original_tree.named_steps["classifier"].estimator_.estimators_
    )
    assert forest.feature_importances_[-2:].sum() > 0  # both sensor and image-PCA inputs exist
    np.testing.assert_array_equal(
        old_mean,
        original_tree.named_steps["features"]
        .named_transformers_["image"]
        .named_steps["scale"]
        .mean_,
    )
    for path, contents in originals.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize(
    "problem",
    [
        "train_csv",
        "validation_csv",
        "train_cache",
        "validation_cache",
        "feature_order",
        "sensor_spec",
    ],
)
def test_mismatched_inputs_rejected_before_fit(case, problem):
    args = case["args"]
    if problem.endswith("csv"):
        getattr(args, problem).write_bytes(b"different split, not authorized input")
    elif problem.endswith("cache"):
        path = args.train_embeddings if problem == "train_cache" else args.validation_embeddings
        (path / "metadata.json").write_bytes(b"changed")
    elif problem == "feature_order":
        case["reference"]["bundle"]["columns"].reverse()
    else:
        case["reference"]["bundle"]["sensor_spec"]["sensor_features"] = ["meas_v"]
    with pytest.raises(ValueError):
        refit.train(args)
    assert not args.output_dir.exists()


def test_overlapping_experiments_rejected(case):
    args = case["args"]
    frame = case["validation"].copy()
    frame["experiment_id"] = case["train"].experiment_id.iloc[0]
    frame.to_csv(args.validation_csv, index=False)
    base = read_json(args.recipe)
    base["provenance"]["validation_csv_sha256"] = sha256(args.validation_csv)
    write_json(args.recipe, base)
    with pytest.raises(ValueError, match="experiment IDs overlap"):
        refit.train(args)
    assert not args.output_dir.exists()


def test_failed_fit_is_not_loadable_and_does_not_change_reference(case, monkeypatch):
    from sklearn.pipeline import Pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("independent synthetic failure")

    monkeypatch.setattr(Pipeline, "fit", fail)
    old = case["old_model"].read_bytes()
    with pytest.raises(RuntimeError):
        refit.train(case["args"])
    assert read_json(case["args"].output_dir / "status.json")["status"] == "failed"
    assert case["old_model"].read_bytes() == old
    with pytest.raises(ValueError, match="complete"):
        refit.load_model(case["args"].output_dir, "cpu")


def test_export_is_self_contained_without_training_data_or_old_models(case, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline

    args = case["args"]
    refit.train(args)
    args.train_csv.unlink()
    args.validation_csv.unlink()
    args.recipe.unlink()
    case["old_model"].unlink()
    shutil.rmtree(args.train_embeddings)
    shutil.rmtree(args.validation_embeddings)

    def forbidden(*args, **kwargs):
        raise AssertionError("Predict cannot fit or load the original reference")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(refit.blend_ratios, "load_base", forbidden)
    predict = prediction_args(case, tmp_path)
    original = case["old_csv"].read_bytes()
    refit.predict(predict)
    output = pd.read_csv(
        predict.output_dir / "submission_extratrees_trainval.csv",
        dtype={"sample_id": str},
        keep_default_na=False,
    )
    template = pd.read_csv(predict.sample_submission, dtype=str, keep_default_na=False)
    assert output.sample_id.tolist() == template.sample_id.tolist()
    assert list(output.columns) == ["sample_id", "label"]
    assert pd.api.types.is_integer_dtype(output.label) and output.label.between(1, 6).all()
    assert "00001" in output.sample_id.tolist() and "NA" in output.sample_id.tolist()
    assert case["old_csv"].read_bytes() == original


def test_subset_shuffle_and_supplied_labels_do_not_change_predictions(case, tmp_path):
    refit.train(case["args"])
    full = prediction_args(case, tmp_path)
    refit.predict(full)
    rows = (
        pd.read_csv(full.input, dtype={"sample_id": str}, keep_default_na=False)
        .iloc[[7, 4, 3, 8]]
        .copy()
    )
    rows["label"] = "ignored"
    partial = prediction_args(case, tmp_path, "partial", rows, template=False)
    # Keep IDs after the helper's leading-zero-ID fixture customization.
    rows.to_csv(partial.input, index=False)
    refit.predict(partial)
    expected = (
        pd.read_csv(full.output_dir / "submission_extratrees_trainval.csv", keep_default_na=False)
        .set_index("sample_id")
        .label
    )
    actual = pd.read_csv(
        partial.output_dir / "submission_extratrees_trainval.csv", keep_default_na=False
    )
    np.testing.assert_array_equal(actual.label, expected.loc[actual.sample_id])


def test_changed_weights_rejected_before_deserializing(case, monkeypatch):
    refit.train(case["args"])
    (case["args"].output_dir / "model.joblib").write_bytes(b"changed")
    monkeypatch.setattr(
        joblib, "load", lambda *a, **k: pytest.fail("must verify bytes before load")
    )
    with pytest.raises(ValueError, match="files changed"):
        refit.load_model(case["args"].output_dir, "cpu")


@pytest.mark.parametrize(
    "problem", ["source", "runtime", "evaluation", "fit_splits", "model_path", "metadata_hash"]
)
def test_bad_metadata_is_not_accepted(case, problem):
    args = case["args"]
    metadata = refit.train(args)
    if problem == "source":
        metadata["source_sha256"] = {}
    elif problem == "runtime":
        metadata["runtime"]["device"] = "invalid"
    elif problem == "evaluation":
        metadata["evaluation"]["validation_macro_f1"] = 0.99
    elif problem == "fit_splits":
        metadata["fit_splits"] = ["train", "validation", "test"]
    elif problem == "model_path":
        metadata["model_file"] = "../elsewhere.joblib"
    else:
        metadata["fit_rows"] = 999
    write_json(args.output_dir / "metadata.json", metadata)
    if problem != "metadata_hash":
        write_json(
            args.output_dir / "status.json",
            {"status": "complete", "metadata_sha256": sha256(args.output_dir / "metadata.json")},
        )
    with pytest.raises(ValueError):
        refit.load_model(args.output_dir, "cpu")


def test_invalid_template_and_existing_directories_are_not_overwritten(case, tmp_path):
    refit.train(case["args"])
    with pytest.raises(ValueError, match="exists"):
        refit.train(case["args"])
    predict = prediction_args(case, tmp_path)
    predict.sample_submission.write_text("sample_id,label\nwrong,0\n")
    with pytest.raises(ValueError, match="IDs must match"):
        refit.predict(predict)
    assert not predict.output_dir.exists()
    predict.output_dir.mkdir()
    protected = predict.output_dir / "submission_extratrees_trainval.csv"
    protected.write_bytes(b"keep this file")
    with pytest.raises(ValueError, match="exists"):
        refit.predict(predict)
    assert protected.read_bytes() == b"keep this file"


def test_image_contract_mismatch_prevents_csv(case, tmp_path, monkeypatch):
    refit.train(case["args"])
    original = mm.make_features

    def different(*args, **kwargs):
        values, meta = original(*args, **kwargs)
        meta["weights"] = "wrong_image_weights"
        return values, meta

    monkeypatch.setattr(mm, "make_features", different)
    predict = prediction_args(case, tmp_path)
    with pytest.raises(ValueError, match="image/sensor contract"):
        refit.predict(predict)
    assert not (predict.output_dir / "submission_extratrees_trainval.csv").exists()
