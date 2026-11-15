"""Recover independent synthetic artifacts; no competition data or GPU access."""

import argparse
import shutil

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_extratrees_leaf_audit import diagnostic as diagnostic  # noqa: F401
from test_seungbeom_extratrees_leaf_sweep import case as case  # noqa: F401

from experiments.seungbeom import extratrees_leaf_audit as audit
from experiments.seungbeom import extratrees_leaf_recovery as recovery
from experiments.seungbeom import extratrees_leaf_sweep as sweep
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


@pytest.fixture
def ready(diagnostic, tmp_path):
    import joblib

    case, diagnostic_args = diagnostic
    # Simulate a real difference in the refit tree while keeping its configuration
    # and training-derived preprocessing. It must never be called the original.
    path = diagnostic_args.failed_run / "leaf2.joblib"
    model = joblib.load(path)
    values = model.named_steps["classifier"].estimator_.estimators_[0].tree_.value
    values[:] = 0.999 * values + 0.001 * values.sum(axis=-1, keepdims=True) / values.shape[-1]
    joblib.dump(model, path)
    prior = read_json(diagnostic_args.failed_run / "validation_metrics.json")
    labels = case["frames"]["validation"].label
    probability = sweep.mm.predict_proba_six_stable(model, case["matrices"]["validation"])
    prior["metrics"]["extra_trees_leaf2"] = sweep.classification_metrics(
        labels, probability.argmax(1) + 1
    )
    combined = sweep.midpoint.blend_probabilities(case["neural"], probability)
    prior["metrics"]["blend_leaf2_50_50"] = sweep.classification_metrics(
        labels, combined.argmax(1) + 1
    )
    write_json(diagnostic_args.failed_run / "validation_metrics.json", prior)
    audit.audit(diagnostic_args)
    assert read_json(diagnostic_args.output_dir / "audit.json")["probabilities"]["exact"] is False
    args = argparse.Namespace(
        recipe=case["args"].recipe,
        failed_run=diagnostic_args.failed_run,
        audit=diagnostic_args.output_dir / "audit.json",
        train_csv=case["args"].train_csv,
        validation_csv=case["args"].validation_csv,
        train_embeddings=case["args"].train_embeddings,
        image_root=case["args"].image_root,
        output_dir=tmp_path / "recovered",
        group_cv=False,
        device="cpu",
    )
    return case, args


def forbid_fit(monkeypatch):
    from sklearn.pipeline import Pipeline

    def fail(*args, **kwargs):
        pytest.fail("Recovery/inference without --group_cv must not train")

    monkeypatch.setattr(Pipeline, "fit", fail)


def test_recovery_preserves_original_and_imports_distinct_refit_without_training(
    ready, monkeypatch
):
    case, args = ready
    forbid_fit(monkeypatch)
    files = [*case["old_files"], args.failed_run / "status.json", args.audit]
    files.extend(args.failed_run / f"leaf{leaf}.joblib" for leaf in sweep.LEAVES)
    before = {p: sha256(p) for p in files}
    result = recovery.recover(args)
    assert result["policy"] == recovery.POLICY
    assert result["original_replay"] == dict.fromkeys(recovery.ORIGINAL_NAMES, True)
    assert result["candidate_replay"] == dict.fromkeys(sweep.METHODS, True)
    assert result["live_leaf2_difference"]["exact"] is False
    assert result["live_leaf2_difference"]["max_abs"] > 0
    assert result["audit"]["prior_replay_matches"]["refitted_leaf2_probabilities"] is False
    assert (
        result["metrics"]["original_frozen_tree_50_50"]
        == read_json(args.recipe)["metrics"]["frozen_tree_50_50"]
    )
    assert before == {p: sha256(p) for p in files}
    for leaf in sweep.LEAVES:
        assert (
            sha256(args.output_dir / f"leaf{leaf}.joblib")
            == before[args.failed_run / f"leaf{leaf}.joblib"]
        )
    recovery.report(argparse.Namespace(recipe=args.output_dir / "recipe.json", class_details=True))


def test_optional_cv_trains_fold_local_trees_only(ready, monkeypatch):
    case, args = ready
    args.group_cv = True
    original = sweep.fit_candidate
    calls = []

    def fit(template, features, labels, leaf):
        assert len(features) < len(case["frames"]["train"])
        calls.append(leaf)
        return original(template, features, labels, leaf)

    monkeypatch.setattr(sweep, "fit_candidate", fit)
    result = recovery.recover(args)
    assert calls == [2, 5, 10] * 3
    assert result["group_cv"]["scope"] == "train_only_trees_only"


def prediction_args(case, args, tmp_path, frame, suffix):
    path, form = tmp_path / f"input_{suffix}.csv", tmp_path / f"form_{suffix}.csv"
    frame.assign(label="ignored").to_csv(path, index=False)
    pd.DataFrame({"sample_id": frame.sample_id.iloc[::-1], "label": 0}).to_csv(form, index=False)
    return argparse.Namespace(
        recipe=args.output_dir / "recipe.json",
        input=path,
        image_root=args.image_root,
        sample_submission=form,
        leaves=[2, 5, 10],
        output_dir=tmp_path / f"submission_{suffix}",
        device="cpu",
    )


def test_prediction_uses_copied_models_current_rows_and_no_training_or_audit_paths(
    ready, tmp_path, monkeypatch
):
    import joblib

    case, args = ready
    forbid_fit(monkeypatch)
    recovery.recover(args)
    full = case["frames"]["validation"]
    expected = {}
    for leaf in sweep.LEAVES:
        model = joblib.load(args.output_dir / f"leaf{leaf}.joblib")
        prob = sweep.mm.predict_proba_six_stable(model, case["matrices"]["validation"])
        labels = sweep.midpoint.blend_probabilities(case["neural"], prob).argmax(1) + 1
        expected[leaf] = pd.Series(labels, index=full.sample_id)
    args.train_csv.unlink()
    args.validation_csv.unlink()
    args.recipe.unlink()
    shutil.rmtree(args.failed_run)
    shutil.rmtree(args.audit.parent)
    shutil.rmtree(args.train_embeddings)
    # Inference does not follow audit.input_sha256 to the old Validation cache.
    for folder in tmp_path.glob("synthetic_validation_cache"):
        shutil.rmtree(folder)
    for index, frame in enumerate((full, full.iloc[[5, 0, 2]])):
        prediction = prediction_args(case, args, tmp_path, frame, str(index))
        recovery.predict(prediction)
        for leaf in sweep.LEAVES:
            output = pd.read_csv(
                prediction.output_dir / f"submission_frozen_refit_extratrees_leaf{leaf}_50_50.csv",
                dtype={"sample_id": str},
            )
            assert list(output) == ["sample_id", "label"]
            assert output.sample_id.tolist() == frame.sample_id.iloc[::-1].tolist()
            assert output.label.tolist() == expected[leaf].loc[output.sample_id].tolist()
        assert read_json(prediction.output_dir / "status.json")["status"] == "complete"


@pytest.mark.parametrize(
    "problem",
    [
        "audit_source",
        "audited_file",
        "validation",
        "different_preprocessor",
        "leaf_param",
        "train_rows",
        "output",
    ],
)
def test_bad_imports_rejected_before_live_image_inference(ready, monkeypatch, problem):
    import joblib

    case, args = ready
    forbid_fit(monkeypatch)
    if problem == "audit_source":
        value = read_json(args.audit)
        value["source_sha256"] = {}
        write_json(args.audit, value)
    elif problem == "audited_file":
        (args.failed_run / "leaf2.joblib").write_bytes(b"changed")
    elif problem == "validation":
        args.validation_csv.write_text("different")
    elif problem in ("different_preprocessor", "leaf_param", "train_rows"):
        path = args.failed_run / "leaf5.joblib"
        model = joblib.load(path)
        if problem == "different_preprocessor":
            model.named_steps["features"].named_transformers_["image"].named_steps[
                "pca"
            ].components_[0, 0] += 0.1
        elif problem == "leaf_param":
            model.set_params(**{sweep.PARAMETER: 10})
        else:
            model.named_steps["classifier"].estimator_.estimators_[0].tree_.n_node_samples[0] -= 1
        joblib.dump(model, path)
    else:
        args.output_dir.mkdir()
    case["neural_calls"].clear()
    with pytest.raises(ValueError):
        recovery.recover(args)
    assert not case["neural_calls"]


@pytest.mark.parametrize("member", ["original", "refit5"])
def test_live_replay_mismatch_still_stops_recovery(ready, monkeypatch, member):
    case, args = ready
    original = recovery.infer

    def drift(*a, **kwargs):
        result = original(*a, **kwargs)
        name = "original_extra_trees" if member == "original" else "refit_extra_trees_leaf5"
        result[name] = np.zeros_like(result[name])
        result[name][:, 0] = 1
        return result

    monkeypatch.setattr(recovery, "infer", drift)
    with pytest.raises(ValueError, match="did not replay"):
        recovery.recover(args)
    assert read_json(args.output_dir / "status.json")["status"] == "failed"
    assert not (args.output_dir / "recipe.json").exists()


@pytest.mark.parametrize("problem", ["model", "source", "status", "output", "template"])
def test_predict_guards_preserve_existing_outputs(ready, tmp_path, monkeypatch, problem):
    case, args = ready
    recovery.recover(args)
    prediction = prediction_args(case, args, tmp_path, case["frames"]["validation"], "guard")
    if problem == "model":
        (args.output_dir / "leaf5.joblib").write_bytes(b"changed")
    elif problem == "source":
        monkeypatch.setattr(recovery, "source_hashes", lambda: {})
    elif problem == "status":
        write_json(args.output_dir / "status.json", {"status": "running"})
    elif problem == "template":
        pd.DataFrame({"sample_id": ["unknown"], "label": 0}).to_csv(
            prediction.sample_submission, index=False
        )
    else:
        prediction.output_dir.mkdir()
    case["neural_calls"].clear()
    with pytest.raises(ValueError):
        recovery.predict(prediction)
    assert not case["neural_calls"]
