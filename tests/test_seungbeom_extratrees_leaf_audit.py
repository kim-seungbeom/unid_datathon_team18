"""Replay diagnosis on synthetic data only; audit does not train or export."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pytest
from test_seungbeom_extratrees_leaf_sweep import case as case  # noqa: F401

from experiments.seungbeom import extratrees_leaf_audit as audit
from experiments.seungbeom import extratrees_leaf_sweep as sweep
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


@pytest.fixture
def diagnostic(case, tmp_path):
    sweep.train(case["args"])
    run = case["args"].output_dir
    (run / "recipe.json").unlink()
    write_json(run / "status.json", {"status": "failed", "error_type": "ValueError"})
    saved = read_json(run / "validation_metrics.json")
    saved["replay_matches"]["refitted_leaf2_probabilities"] = False
    write_json(run / "validation_metrics.json", saved)
    cache = tmp_path / "synthetic_validation_cache"
    cache.mkdir()
    np.save(cache / "features.npy", case["matrices"]["validation"].iloc[::-1, 5:].to_numpy())
    frame = case["frames"]["validation"]
    frame.iloc[::-1][["sample_id"]].to_csv(cache / "sample_ids.csv", index=False)
    write_json(
        cache / "metadata.json",
        {**case["legacy"]["bundle"]["embedding"], "status": "complete", "rows": len(frame)},
    )
    case["legacy"]["identity"]["input_provenance"]["validation"] = {
        name: sha256(cache / name) for name in sweep.CACHE_FILES
    }
    args = argparse.Namespace(
        recipe=case["args"].recipe,
        failed_run=run,
        validation_csv=case["args"].validation_csv,
        validation_embeddings=cache,
        output_dir=tmp_path / "audit",
        device="cpu",
    )
    return case, args


def test_audit_uses_saved_trees_and_original_cache_without_fitting_or_images(
    diagnostic, monkeypatch
):
    from sklearn.pipeline import Pipeline

    case, args = diagnostic
    before = {str(path): sha256(path) for path in case["old_files"]}
    before.update(
        {
            str(args.failed_run / name): sha256(args.failed_run / name)
            for name in ("leaf2.joblib", "status.json", "validation_metrics.json")
        }
    )

    def fail(*args, **kwargs):
        pytest.fail("Audit must not fit a model or run the image model")

    monkeypatch.setattr(Pipeline, "fit", fail)
    monkeypatch.setattr(sweep.frozen_blend, "frozen_probabilities", fail)
    case["calls"].clear()
    result = audit.audit(args)
    assert result["probabilities"]["exact"]
    assert result["probabilities"]["max_abs"] == 0
    assert result["probabilities"]["changed_tree_predictions"] == 0
    assert result["fitted_state"]["trees"]["different_tree_count"] == 0
    assert result["cache_matches_saved_live_metrics"] == {"original": True, "refitted": True}
    assert result["prior_replay_matches"]["refitted_leaf2_probabilities"] is False
    assert len(case["calls"]) == 1  # original recorded cache only, no live extraction
    assert {path: sha256(Path(path)) for path in before} == before
    assert not (args.output_dir / "recipe.json").exists()
    assert not list(args.output_dir.glob("*.csv"))


def test_probability_diagnostic_distinguishes_roundoff_from_changed_decisions():
    p = np.array([[0.7, 0.3, 0, 0, 0, 0]], dtype=np.float64)
    tiny = p.copy()
    tiny[0, 0] = np.nextafter(tiny[0, 0], 1)
    result = audit.probability_difference(p, tiny)
    assert not result["exact"] and 0 < result["max_abs"] < 1e-15
    assert result["changed_tree_predictions"] == 0
    changed = np.roll(p, 1, axis=1)
    assert audit.probability_difference(p, changed)["changed_tree_predictions"] == 1


def test_model_state_diagnostic_distinguishes_pca_and_tree_changes(case):
    tree = copy.deepcopy(case["template"])
    pca = tree.named_steps["features"].named_transformers_["image"].named_steps["pca"]
    pca.components_[0, 0] += 1e-8
    result = audit.fitted_state_difference(case["template"], tree)
    assert not result["preprocessing"]["image.pca.components_"]["exact"]
    assert result["trees"]["different_tree_count"] == 0
    tree.named_steps["classifier"].estimator_.estimators_[0].tree_.threshold[0] += 1
    result = audit.fitted_state_difference(case["template"], tree)
    assert result["trees"]["different_tree_count"] == 1
    assert result["trees"]["fields"]["threshold"]["max_abs"] == 1


@pytest.mark.parametrize("problem", ["csv", "cache", "params", "running", "output"])
def test_audit_rejects_unverified_sources_before_feature_reads(diagnostic, monkeypatch, problem):
    import joblib

    case, args = diagnostic
    if problem == "csv":
        args.validation_csv.write_text("not the original validation")
    elif problem == "cache":
        (args.validation_embeddings / "features.npy").write_bytes(b"wrong cache")
    elif problem == "params":
        model = joblib.load(args.failed_run / "leaf2.joblib")
        model.set_params(**{sweep.PARAMETER: 5})
        joblib.dump(model, args.failed_run / "leaf2.joblib")
    elif problem == "running":
        write_json(args.failed_run / "status.json", {"status": "running"})
    else:
        args.output_dir.mkdir()
    case["calls"].clear()
    with pytest.raises(ValueError):
        audit.audit(args)
    assert not case["calls"]
