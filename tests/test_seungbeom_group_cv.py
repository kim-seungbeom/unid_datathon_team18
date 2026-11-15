"""Independent synthetic CPU checks; never use competition files or GPU access."""

# Optional runtime imports follow importorskip so minimal code-only environments skip.
# ruff: noqa: E402

import argparse
import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from experiments.seungbeom import group_cv as cv
from experiments.seungbeom import group_cv_training as training
from experiments.seungbeom import multimodal as mm
from experiments.seungbeom.export_submissions import read_json, sha256, write_json
from experiments.seungbeom.neural_config import normalize_config
from runtime import seed_everything
from schema import RAW_SENSOR_COLUMNS


@pytest.fixture
def invented(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    rng = np.random.default_rng(481)
    n = 72
    frame = pd.DataFrame(
        {
            "sample_id": [f"synthetic_{i:04d}" for i in range(n)],
            "experiment_id": np.repeat([f"invented_group_{i}" for i in range(12)], 6),
            "image_path": [f"nonexistent_{i}.png" for i in range(n)],
            "label": np.tile(np.arange(1, 7), 12),
            **{name: rng.normal(size=n) for name in RAW_SENSOR_COLUMNS},
        }
    )
    frame.loc[::7, "meas_v"] = np.nan
    csv = tmp_path / "invented_train.csv"
    frame.to_csv(csv, index=False)
    vectors = rng.normal(size=(n, 512)).astype(np.float32)
    cache = tmp_path / "invented_embeddings"
    cache.mkdir()
    # Reverse the cache order to exercise ID alignment rather than row-order assumptions.
    np.save(cache / "features.npy", vectors[::-1])
    frame.iloc[::-1][["sample_id"]].to_csv(cache / "sample_ids.csv", index=False)
    contract = {
        "model": "resnet18",
        "weights": "INDEPENDENT_SYNTHETIC_VECTORS",
        "dimension": 512,
        "preprocessing": {"synthetic": True},
        "l2_normalize": False,
        "trained": False,
    }
    write_json(cache / "metadata.json", {**contract, "status": "complete", "rows": n})
    config = normalize_config(
        {
            "sensor_hidden": 2,
            "fusion_hidden": 4,
            "epochs": 2,
            "early_stopping_patience": 1,
            "batch_size": 24,
        }
    )
    sensor = frame[list(RAW_SENSOR_COLUMNS)].copy()
    sensor.columns = [f"sensor_{name}" for name in sensor.columns]
    features = pd.concat(
        [sensor, pd.DataFrame(vectors, columns=[f"img_{i:04d}" for i in range(512)])], axis=1
    )
    tree = mm.build_pipeline("extra_trees", features.columns, pca_dim=2)
    tree.set_params(classifier__estimator__n_estimators=3, classifier__estimator__n_jobs=1)
    tree.fit(features, frame["label"])
    old_model = tmp_path / "preserved_reference.bin"
    old_model.write_bytes(b"invented fitted model artifact, never overwritten")
    old_csv = tmp_path / "preserved_submission.csv"
    old_csv.write_text("sample_id,label\ninvented,1\n")
    perfect = training.classification_metrics(frame["label"], frame["label"])
    base = {
        "provenance": {
            "train_csv_sha256": sha256(csv),
            "files": {str(old_model): sha256(old_model)},
        },
        "metrics": {name: perfect for name in training.METHODS},
    }
    recipe = tmp_path / "invented_recipe.json"
    write_json(recipe, base)
    loaded = {
        "config": config,
        "identity": {
            "input_provenance": {
                "train": {
                    name: sha256(cache / name)
                    for name in ("features.npy", "sample_ids.csv", "metadata.json")
                }
            }
        },
        "checkpoint": {"visual_contract": contract},
        "bundle": {
            "models": ["extra_trees"],
            "pipelines": [tree],
            "columns": list(features.columns),
        },
    }

    def fake_load_base(base, device):
        # Only the trusted-reference deserializer is replaced; actual group CV,
        # CSV/cache IO, Torch, preprocessing, PCA and tree fitting run normally.
        seed_everything(42)
        return loaded

    monkeypatch.setattr(cv.blend_ratios, "load_base", fake_load_base)
    args = argparse.Namespace(
        recipe=recipe,
        train_csv=csv,
        train_embeddings=cache,
        output_dir=tmp_path / "new_cv",
        device="cpu",
        resume=False,
    )
    return args, frame, vectors, features, tree, config, contract, old_model, old_csv


def test_group_partitions_are_disjoint_complete_and_repeatable(invented):
    _, frame, *_ = invented
    plan = training.split_plan(frame)
    assert plan == training.split_plan(frame)
    assert len(plan) == 3
    seen = []
    for part in plan:
        f, h, i, s = (set(part[key]) for key in ("fit", "holdout", "inner_fit", "inner_stop"))
        assert not f & h and f | h == set(range(len(frame)))
        assert not i & s and i | s == f
        assert not set(frame.iloc[list(f)].experiment_id) & set(frame.iloc[list(h)].experiment_id)
        assert not set(frame.iloc[list(i)].experiment_id) & set(frame.iloc[list(s)].experiment_id)
        seen.extend(h)
    assert sorted(seen) == list(range(len(frame)))


@pytest.mark.parametrize(
    "problem", ["missing_group", "blank_group", "duplicate_id", "few_groups", "missing_class"]
)
def test_invalid_group_inputs_fail(invented, problem):
    frame = invented[1].copy()
    if problem == "missing_group":
        frame.loc[0, "experiment_id"] = None
    elif problem == "blank_group":
        frame.loc[0, "experiment_id"] = " "
    elif problem == "duplicate_id":
        frame.loc[0, "sample_id"] = frame.loc[1, "sample_id"]
    elif problem == "few_groups":
        frame["experiment_id"] = "one_group"
    else:
        frame.loc[frame.label == 6, "label"] = 5
    with pytest.raises(ValueError):
        training.split_plan(frame)


def test_cache_ids_are_joined_and_original_tree_is_not_reused(invented):
    args, _, vectors, _, tree, *_ = invented
    manifest, _, actual, _, template, _ = cv.load_inputs(args)
    np.testing.assert_array_equal(actual, vectors)
    assert template is tree  # fit_fold must clone this fitted template.
    assert manifest["config"] == invented[5]
    assert len(manifest["plan"]) == 3


def test_rare_class_in_one_experiment_keeps_groups_intact(invented, tmp_path):
    _, frame, vectors, features, tree, config, contract, *_ = invented
    frame = frame.copy()
    frame.loc[(frame.label == 6) & (frame.experiment_id != "invented_group_0"), "label"] = 5
    plan = training.split_plan(frame)
    part = next(part for part in plan if part["counts"]["fit"]["support"]["6"] == 0)
    destination = tmp_path / "rare_class"
    destination.mkdir()
    result = training.fit_fold(
        frame,
        vectors,
        features,
        tree,
        {**config, "epochs": 1},
        part,
        destination,
        torch.device("cpu"),
        contract,
    )
    assert result["metrics"]["extra_trees"]["class_f1"]["6"] == 0
    with np.load(destination / "oof.npz") as predictions:
        assert predictions["extra_trees"].shape[1] == 6
        assert (predictions["extra_trees"][:, 5] == 0).all()


@pytest.mark.parametrize(
    "problem", ["csv", "cache", "wrong_seed", "wrong_loss", "contract", "column_order"]
)
def test_reference_mismatch_fails_before_output(invented, monkeypatch, problem):
    args = invented[0]
    if problem == "csv":
        args.train_csv.write_text("not the original Train CSV")
    elif problem == "cache":
        (args.train_embeddings / "metadata.json").write_text("changed")
    else:
        original = cv.blend_ratios.load_base

        def mismatch(base, device):
            loaded = copy.deepcopy(original(base, device))
            if problem == "wrong_seed":
                loaded["config"]["seed"] = 44
            elif problem == "wrong_loss":
                loaded["config"]["loss"] = "focal"
            elif problem == "contract":
                loaded["checkpoint"]["visual_contract"]["weights"] = "different"
            else:
                loaded["bundle"]["columns"] = loaded["bundle"]["columns"][::-1]
            return loaded

        monkeypatch.setattr(cv.blend_ratios, "load_base", mismatch)
    with pytest.raises(ValueError):
        cv.run(args)
    assert not args.output_dir.exists()


def test_outer_holdout_cannot_change_training_or_preprocessing(invented, tmp_path):
    import joblib

    args, frame, vectors, features, tree, config, contract, *_ = invented
    part = training.split_plan(frame)[0]
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    config = {**config, "epochs": 1}
    original_tree_mean = (
        tree.named_steps["features"].named_transformers_["image"].named_steps["scale"].mean_.copy()
    )
    training.fit_fold(
        frame, vectors, features, tree, config, part, first, torch.device("cpu"), contract
    )
    changed, changed_vectors, changed_features = frame.copy(), vectors.copy(), features.copy()
    h = part["holdout"]
    changed.loc[h, list(RAW_SENSOR_COLUMNS)] = 100000.0
    changed.loc[h, "label"] = 7 - changed.loc[h, "label"]
    changed_vectors[h] = -100000.0
    changed_features.loc[h, :] = -100000.0
    training.fit_fold(
        changed,
        changed_vectors,
        changed_features,
        tree,
        config,
        part,
        second,
        torch.device("cpu"),
        contract,
    )
    left = torch.load(first / "neural.pt", weights_only=True)
    right = torch.load(second / "neural.pt", weights_only=True)
    assert left["epoch"] == right["epoch"]
    assert left["preprocessor"] == right["preprocessor"]
    for key in left["state_dict"]:
        assert torch.equal(left["state_dict"][key], right["state_dict"][key])
    fitted = joblib.load(first / "extra_trees.joblib")
    other = joblib.load(second / "extra_trees.joblib")
    np.testing.assert_array_equal(fitted.predict_proba(features), other.predict_proba(features))
    transformer = fitted.named_steps["features"]
    np.testing.assert_allclose(
        transformer.named_transformers_["sensor"].statistics_,
        np.nanmedian(features.iloc[part["fit"], :5], axis=0),
    )
    np.testing.assert_allclose(
        transformer.named_transformers_["image"].named_steps["scale"].mean_,
        vectors[part["fit"]].mean(axis=0, dtype=np.float64),
    )
    np.testing.assert_allclose(
        left["preprocessor"]["medians"],
        np.nanmedian(frame.iloc[part["fit"]][list(RAW_SENSOR_COLUMNS)], axis=0),
    )
    np.testing.assert_array_equal(
        original_tree_mean,
        tree.named_steps["features"].named_transformers_["image"].named_steps["scale"].mean_,
    )


def test_inner_preprocessing_uses_only_inner_fit(invented):
    _, frame, vectors, _, _, config, *_ = invented
    part = training.split_plan(frame)[0]
    i, s = part["inner_fit"], part["inner_stop"]
    stop = frame.iloc[s].copy()
    stop[list(RAW_SENSOR_COLUMNS)] = 1e6
    _, prep, selected, history = training.train_neural(
        frame.iloc[i],
        vectors[i],
        config,
        torch.device("cpu"),
        epochs=2,
        stopping=(stop, vectors[s]),
    )
    np.testing.assert_allclose(
        prep.medians, np.nanmedian(frame.iloc[i][list(RAW_SENSOR_COLUMNS)], axis=0)
    )
    assert selected == max(history, key=lambda row: row["inner_macro_f1"])["epoch"]


def test_complete_run_report_resume_and_reference_preservation(invented, monkeypatch, capsys):
    args, _, _, _, _, _, _, model, submission = invented
    originals = {path: path.read_bytes() for path in (model, submission, args.recipe)}
    read_paths = []
    reader = cv.read_frame

    def train_only(path, **kwargs):
        read_paths.append(path)
        assert path == args.train_csv
        return reader(path, **kwargs)

    monkeypatch.setattr(cv, "read_frame", train_only)
    cv.run(args)
    summary = read_json(args.output_dir / "summary.json")
    assert len(summary["folds"]) == 3
    assert all(
        sum(item["pooled_oof"]["support"].values()) == 72 for item in summary["methods"].values()
    )
    assert read_paths == [args.train_csv]
    old_summary = (args.output_dir / "summary.json").read_bytes()
    args.resume = True

    def never_fit(*args, **kwargs):
        raise AssertionError("A complete fold must not be trained twice")

    monkeypatch.setattr(training, "fit_fold", never_fit)
    cv.run(args)
    assert (args.output_dir / "summary.json").read_bytes() == old_summary
    cv.report(args)
    assert "pooled_oof_f1" in capsys.readouterr().out
    for path, value in originals.items():
        assert path.read_bytes() == value
    source = cv.source_hashes
    monkeypatch.setattr(cv, "source_hashes", lambda: {**source(), "changed": "code"})
    with pytest.raises(ValueError, match="cannot resume"):
        cv.run(args)


def test_failed_run_resumes_completed_folds_and_detects_tampering(invented, monkeypatch):
    args = invented[0]
    original = training.fit_fold
    calls = []

    def interrupt_once(*values):
        part = values[5]
        calls.append(part["fold"])
        if part["fold"] == 2:
            raise RuntimeError("synthetic interruption")
        return original(*values)

    monkeypatch.setattr(training, "fit_fold", interrupt_once)
    with pytest.raises(RuntimeError):
        cv.run(args)
    assert calls == [1, 2]
    first = (args.output_dir / "fold_1.json").read_bytes()
    with pytest.raises(ValueError, match="1/3"):
        cv.report(args)
    args.resume = True
    monkeypatch.setattr(training, "fit_fold", original)
    cv.run(args)
    assert (args.output_dir / "fold_1.json").read_bytes() == first
    assert read_json(args.output_dir / "fold_2.json")["attempt"].endswith("002")
    saved = read_json(args.output_dir / "fold_1.json")
    (args.output_dir / saved["attempt"] / "oof.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifacts changed"):
        cv.run(args)


def test_existing_output_is_never_overwritten(invented):
    args = invented[0]
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="exists"):
        cv.run(args)
    assert list(args.output_dir.iterdir()) == []


def test_fold_average_and_pooled_oof_are_distinct():
    labels = np.array([1, 2, 3, 4, 5, 6] * 3)
    positions = [np.arange(0, 6), np.arange(6, 9), np.arange(9, 18)]
    pred = labels.copy()
    pred[6:9] = 6
    prob = np.eye(6)[pred - 1]
    results = [
        {
            "metrics": {
                name: training.classification_metrics(labels[i], pred[i])
                for name in training.METHODS
            }
        }
        for i in positions
    ]
    result = training.summarize(labels, results, dict.fromkeys(training.METHODS, prob))
    item = result["methods"]["blend_50_50"]
    expected = [row["metrics"]["blend_50_50"]["macro_f1"] for row in results]
    assert item["fold_macro_f1_mean"] == pytest.approx(np.mean(expected))
    assert item["fold_macro_f1_std"] == pytest.approx(np.std(expected, ddof=1))
    assert item["pooled_oof"] == training.classification_metrics(labels, pred)
    assert item["fold_macro_f1_mean"] != item["pooled_oof"]["macro_f1"]
