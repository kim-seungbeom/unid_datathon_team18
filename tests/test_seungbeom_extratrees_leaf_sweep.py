"""Independent synthetic CPU checks for leaf-only Train fitting and fixed inference."""

import argparse
import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom import extratrees_leaf_sweep as sweep
from experiments.seungbeom import frozen_tree_blend as blend
from experiments.seungbeom import multimodal as mm
from experiments.seungbeom.export_submissions import read_json, sha256, write_json
from runtime import seed_everything
from schema import RAW_SENSOR_COLUMNS
from sensor_features import build_sensor_features

MAKE_FEATURES = mm.make_features


@pytest.fixture
def case(tmp_path, monkeypatch):
    import joblib

    seed_everything(42)
    rng = np.random.default_rng(81)
    frames, paths, matrices, vectors = {}, {}, {}, {}
    for split, size in (("train", 72), ("validation", 18)):
        frame = pd.DataFrame(
            {
                "sample_id": [f"{split}_{i:04d}" for i in range(size)],
                "image_path": [f"{split}_image_{i}.png" for i in range(size)],
                "experiment_id": [f"{split}_group_{i // 6}" for i in range(size)],
                "label": np.tile(np.arange(1, 7), size // 6),
                **{
                    name: rng.normal(size=size) + (20 if split == "validation" else 0)
                    for name in RAW_SENSOR_COLUMNS
                },
            }
        )
        frame.loc[0, "meas_v"] = np.nan
        frames[split] = frame
        paths[split] = tmp_path / f"synthetic_{split}.csv"
        frame.to_csv(paths[split], index=False)
        visual = rng.normal(size=(size, 512)).astype(np.float32)
        vectors.update(dict(zip(frame.image_path, visual, strict=True)))
        matrices[split] = pd.concat(
            [
                frame[list(RAW_SENSOR_COLUMNS)].add_prefix("sensor_"),
                pd.DataFrame(visual, columns=[f"img_{i:04d}" for i in range(512)]),
            ],
            axis=1,
        )
    metadata = {
        "model": "resnet18",
        "weights": "SYNTHETIC_ONLY",
        "dimension": 512,
        "preprocessing": {"synthetic": True},
        "l2_normalize": False,
        "trained": False,
    }
    cache = tmp_path / "synthetic_train_cache"
    cache.mkdir()
    np.save(cache / "features.npy", matrices["train"].iloc[::-1, 5:].to_numpy())
    frames["train"].iloc[::-1][["sample_id"]].to_csv(cache / "sample_ids.csv", index=False)
    write_json(cache / "metadata.json", {**metadata, "status": "complete", "rows": 72})
    template = mm.build_pipeline("extra_trees", matrices["train"].columns, pca_dim=3, seed=42)
    template.set_params(classifier__estimator__n_estimators=9, classifier__estimator__n_jobs=1)
    template.fit(matrices["train"], frames["train"].label)
    old_file = tmp_path / "original_tree.joblib"
    joblib.dump(template, old_file)
    frozen_file = tmp_path / "original_frozen.pt"
    frozen_file.write_bytes(b"independent synthetic neural stub; not deserialized")
    neural = rng.uniform(0.01, 1, (18, 6))
    neural /= neural.sum(axis=1, keepdims=True)
    by_image = dict(zip(frames["validation"].image_path, neural, strict=True))
    tree_prob = mm.predict_proba_six_stable(template, matrices["validation"])

    def metrics(prob):
        return sweep.classification_metrics(frames["validation"].label, prob.argmax(1) + 1)

    base = {
        "provenance": {
            "train_csv_sha256": sha256(paths["train"]),
            "validation_csv_sha256": sha256(paths["validation"]),
            "files": {str(old_file): sha256(old_file)},
        },
        "runtime_signature": sweep.runtime_signature("cpu"),
        "image_precision": "fp32",
        "metrics": {
            "neural": metrics(neural),
            "extra_trees": metrics(tree_prob),
            "blend_50_50": metrics(sweep.midpoint.blend_probabilities(neural, tree_prob)),
        },
    }
    record = {
        "run_dir": str(tmp_path),
        "files": {str(frozen_file): sha256(frozen_file)},
        "metrics": metrics(neural),
    }
    reference = {
        "kind": blend.KIND,
        "format_version": 1,
        "status": "complete",
        "rule": blend.RULE,
        "label_values": list(range(1, 7)),
        "replay_matches": dict.fromkeys(blend.REPLAY_NAMES, True),
        "source_sha256": blend.source_hashes(),
        "frozen": record,
        "base_recipe": base,
        "image_execution": {"frozen": "fp32", "extra_trees": "fp32"},
        "metrics": {
            "frozen": metrics(neural),
            "extra_trees": metrics(tree_prob),
            "frozen_tree_50_50": base["metrics"]["blend_50_50"],
            "historical_neural": metrics(neural),
            "baseline_50_50": base["metrics"]["blend_50_50"],
        },
    }
    source = tmp_path / "reference"
    source.mkdir()
    write_json(source / "recipe.json", reference)
    write_json(
        source / "status.json",
        {"status": "complete", "recipe_sha256": sha256(source / "recipe.json")},
    )
    legacy = {
        "config": {"seed": 42, "sensor_features": list(RAW_SENSOR_COLUMNS)},
        "identity": {
            "input_provenance": {
                "train": {name: sha256(cache / name) for name in sweep.CACHE_FILES}
            }
        },
        "bundle": {
            "models": ["extra_trees"],
            "pipelines": [template],
            "refit_train_val": False,
            "sensor_spec": mm.sensor_spec(),
            "encoder": "resnet18",
            "embedding": metadata,
            "columns": list(matrices["train"].columns),
        },
    }
    # Heavy legacy checkpoint loading is already exercised by frozen_tree_blend tests.
    # This fixture keeps actual sklearn training, PCA, serialization and probabilities.
    monkeypatch.setattr(blend, "load_members", lambda *a: (legacy, {}, copy.deepcopy(record)))
    calls, neural_calls, policies = [], [], []

    def extract(current, csv_path, image_root, spec, **kwargs):
        import torch

        assert "label" not in current and csv_path is None
        if kwargs.get("embeddings_dir") is not None:
            calls.append("train_cache")
            return MAKE_FEATURES(current, csv_path, image_root, spec, **kwargs)
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        calls.append(current.sample_id.tolist())
        policies.append((torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))
        return pd.concat(
            [
                build_sensor_features(current).reset_index(drop=True).add_prefix("sensor_"),
                pd.DataFrame(
                    np.stack([vectors[p] for p in current.image_path]),
                    columns=[f"img_{i:04d}" for i in range(512)],
                ),
            ],
            axis=1,
        ), copy.deepcopy(metadata)

    def frozen_prob(current, image_root, saved, device):
        assert "label" not in current
        neural_calls.append(current.sample_id.tolist())
        return np.stack([by_image[p] for p in current.image_path])

    monkeypatch.setattr(mm, "make_features", extract)
    monkeypatch.setattr(blend, "frozen_probabilities", frozen_prob)
    args = argparse.Namespace(
        recipe=source / "recipe.json",
        train_csv=paths["train"],
        validation_csv=paths["validation"],
        train_embeddings=cache,
        image_root=tmp_path,
        output_dir=tmp_path / "sweep",
        device="cpu",
        group_cv=False,
    )
    return {
        "args": args,
        "frames": frames,
        "matrices": matrices,
        "legacy": legacy,
        "template": template,
        "calls": calls,
        "neural_calls": neural_calls,
        "policies": policies,
        "old_files": [old_file, frozen_file, source / "recipe.json"],
        "neural": neural,
    }


def prediction_args(case, tmp_path, rows=None, suffix="one"):
    frame = case["frames"]["validation"] if rows is None else rows
    path, form = tmp_path / f"input_{suffix}.csv", tmp_path / f"template_{suffix}.csv"
    frame.assign(label="never_used").to_csv(path, index=False)
    pd.DataFrame({"sample_id": frame.sample_id.iloc[::-1], "label": 0}).to_csv(form, index=False)
    return argparse.Namespace(
        recipe=case["args"].output_dir / "recipe.json",
        input=path,
        image_root=tmp_path,
        sample_submission=form,
        leaves=[2, 5, 10],
        output_dir=tmp_path / f"export_{suffix}",
        device="cpu",
    )


def test_only_leaf_changes_train_preprocessing_serialization_and_original_preserved(
    case, monkeypatch
):
    import joblib
    from sklearn.pipeline import Pipeline

    before = {str(p): sha256(p) for p in case["old_files"]}
    fits = []
    fit = Pipeline.fit

    def spy(self, X, y=None, **kwargs):
        if "classifier" in self.named_steps:
            fits.append((X.copy(), np.asarray(y).copy()))
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", spy)
    result = sweep.train(case["args"])
    assert result["replay_matches"] == dict.fromkeys(sweep.REPLAY_NAMES, True)
    assert len(fits) == 3
    for X, y in fits:
        pd.testing.assert_frame_equal(X, case["matrices"]["train"])
        np.testing.assert_array_equal(y, case["frames"]["train"].label)
    for leaf in sweep.LEAVES:
        model = joblib.load(case["args"].output_dir / f"leaf{leaf}.joblib")
        assert model.get_params()[sweep.PARAMETER] == leaf
        assert sweep.parameter_fingerprint(
            model, normalize_leaf=True
        ) == sweep.parameter_fingerprint(case["template"])
        np.testing.assert_allclose(
            model.named_steps["features"].named_transformers_["sensor"].statistics_,
            np.nanmedian(case["frames"]["train"][list(RAW_SENSOR_COLUMNS)], axis=0),
        )
        assert all(
            t.tree_.n_node_samples[0] == 72
            for t in model.named_steps["classifier"].estimator_.estimators_
        )
        prob = mm.predict_proba_six_stable(model, case["matrices"]["validation"])
        expected = sweep.midpoint.blend_probabilities(case["neural"], prob)
        assert result["metrics"][f"blend_leaf{leaf}_50_50"] == sweep.classification_metrics(
            case["frames"]["validation"].label, expected.argmax(1) + 1
        )
    assert {str(p): sha256(p) for p in case["old_files"]} == before
    assert case["calls"] == ["train_cache", case["frames"]["validation"].sample_id.tolist()]
    assert len(case["neural_calls"]) == 1
    sweep.report(
        argparse.Namespace(recipe=case["args"].output_dir / "recipe.json", class_details=True)
    )


def test_group_cv_fits_fold_preprocessing_without_neural_or_validation(case, monkeypatch):
    from sklearn.model_selection import GroupKFold

    calls = []
    original = sweep.fit_candidate

    def spy(template, X, y, leaf):
        calls.append((X.index.tolist(), leaf))
        return original(template, X, y, leaf)

    monkeypatch.setattr(sweep, "fit_candidate", spy)
    frame = case["frames"]["train"]
    result = sweep.tree_group_cv(frame, case["matrices"]["train"], case["template"])
    expected = list(GroupKFold(3).split(frame, groups=frame.experiment_id))
    assert len(calls) == 9 and not case["neural_calls"]
    for part, (fit, held) in zip(result["folds"], expected, strict=True):
        assert part["fit"] == fit.tolist() and part["holdout"] == held.tolist()
        assert not set(frame.iloc[fit].experiment_id) & set(frame.iloc[held].experiment_id)
    assert calls == [(fit.tolist(), leaf) for fit, held in expected for leaf in sweep.LEAVES]
    assert set(result["summary"]) == {"2", "5", "10"}
    assert result["scope"] == "train_only_trees_only"


def test_predict_current_rows_no_fit_no_training_files_subset_order_and_precision(
    case, tmp_path, monkeypatch
):
    from sklearn.pipeline import Pipeline

    case["args"].group_cv = True
    sweep.train(case["args"])
    case["args"].train_csv.unlink()
    case["args"].validation_csv.unlink()
    case["args"].recipe.unlink()
    shutil.rmtree(case["args"].train_embeddings)

    def fail(*args, **kwargs):
        pytest.fail("predict must not fit")

    monkeypatch.setattr(Pipeline, "fit", fail)
    case["calls"].clear()
    predictions = []
    for i, rows in enumerate(
        (case["frames"]["validation"], case["frames"]["validation"].iloc[[5, 0, 2]])
    ):
        args = prediction_args(case, tmp_path, rows, str(i))
        sweep.predict(args)
        current = {}
        for leaf in sweep.LEAVES:
            output = pd.read_csv(
                args.output_dir / f"submission_frozen_extratrees_leaf{leaf}_50_50.csv",
                dtype={"sample_id": str},
            )
            assert list(output) == ["sample_id", "label"]
            assert output.sample_id.tolist() == rows.sample_id.iloc[::-1].tolist()
            assert output.label.between(1, 6).all()
            current[leaf] = output.set_index("sample_id").label
            if i:
                pd.testing.assert_series_equal(
                    current[leaf], predictions[0][leaf].loc[current[leaf].index]
                )
        predictions.append(current)
    assert case["calls"] == [
        case["frames"]["validation"].sample_id.tolist(),
        case["frames"]["validation"].iloc[[5, 0, 2]].sample_id.tolist(),
    ]


@pytest.mark.parametrize(
    "problem", ["train", "validation", "cache", "leaf", "refit_train_val", "source", "output"]
)
def test_bad_training_inputs_rejected_before_fit_or_images(case, monkeypatch, problem):
    args = case["args"]
    if problem in ("train", "validation"):
        getattr(args, f"{problem}_csv").write_text("different input must not be parsed")
    elif problem == "cache":
        (args.train_embeddings / "features.npy").write_bytes(b"different cache")
    elif problem == "leaf":
        case["template"].set_params(**{sweep.PARAMETER: 5})
    elif problem == "refit_train_val":
        case["legacy"]["bundle"]["refit_train_val"] = True
    elif problem == "source":
        monkeypatch.setattr(blend, "source_hashes", lambda: {"changed": "source"})
    else:
        args.output_dir.mkdir()
    with pytest.raises(ValueError):
        sweep.train(args)
    assert not case["calls"] and not case["neural_calls"]


def test_confusion_replay_drift_blocks_export(case):
    args = case["args"]
    reference = read_json(args.recipe)
    reference["metrics"]["frozen_tree_50_50"]["confusion_matrix"][0][0] += 1
    write_json(args.recipe, reference)
    write_json(
        args.recipe.parent / "status.json",
        {"status": "complete", "recipe_sha256": sha256(args.recipe)},
    )
    with pytest.raises(ValueError, match="replay differs"):
        sweep.train(args)
    assert read_json(args.output_dir / "status.json")["status"] == "failed"
    assert not (args.output_dir / "recipe.json").exists()


def test_refitted_leaf2_probability_drift_is_rejected_even_if_argmax_is_unchanged(
    case, monkeypatch
):
    original = mm.predict_proba_six_stable

    def drift(model, X):
        prob = original(model, X)
        if model is not case["template"] and model.get_params()[sweep.PARAMETER] == 2:
            # Keep decisions but change confidence: aggregate scores alone would miss this.
            prob = prob * 0.999 + 0.001 / 6
        return prob

    monkeypatch.setattr(mm, "predict_proba_six_stable", drift)
    with pytest.raises(ValueError, match="replay differs"):
        sweep.train(case["args"])
    result = read_json(case["args"].output_dir / "validation_metrics.json")
    assert result["replay_matches"]["refitted_leaf2_probabilities"] is False
    assert not (case["args"].output_dir / "recipe.json").exists()


def test_live_tree_features_keep_original_precision_policy_for_all_candidates(case):
    args = case["args"]
    reference = read_json(args.recipe)
    reference["base_recipe"]["image_precision"] = "cudnn_tf32"
    reference["image_execution"]["extra_trees"] = "cudnn_tf32"
    write_json(args.recipe, reference)
    write_json(
        args.recipe.parent / "status.json",
        {"status": "complete", "recipe_sha256": sha256(args.recipe)},
    )
    sweep.train(args)
    assert case["policies"] == [(True, False)]


@pytest.mark.parametrize(
    "problem",
    [
        "leaf_model",
        "frozen_model",
        "source",
        "runtime",
        "template",
        "status",
        "output",
        "duplicate",
    ],
)
def test_prediction_guards_before_images(case, tmp_path, monkeypatch, problem):
    sweep.train(case["args"])
    args = prediction_args(case, tmp_path)
    case["calls"].clear()
    case["neural_calls"].clear()
    if problem == "leaf_model":
        (args.recipe.parent / "leaf5.joblib").write_bytes(b"not deserialized")
    elif problem == "frozen_model":
        case["old_files"][1].write_bytes(b"changed checkpoint")
    elif problem == "source":
        monkeypatch.setattr(sweep, "source_hashes", lambda: {"changed": "source"})
    elif problem == "runtime":
        monkeypatch.setattr(sweep, "runtime_signature", lambda device: {"changed": "runtime"})
    elif problem == "template":
        pd.DataFrame({"sample_id": ["missing"], "label": [1]}).to_csv(
            args.sample_submission, index=False
        )
    elif problem == "status":
        write_json(args.recipe.parent / "status.json", {"status": "running"})
    elif problem == "duplicate":
        args.leaves = [5, 5]
    else:
        args.output_dir.mkdir()
    with pytest.raises(ValueError):
        sweep.predict(args)
    assert not case["calls"] and not case["neural_calls"]
