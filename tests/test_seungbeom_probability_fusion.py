"""Synthetic CPU-only calibration/stacking and cached artifact workflow tests."""

# ruff: noqa: F811

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_inner_best_cached_ratios import cached_case  # noqa: F401
from test_seungbeom_inner_best_prior_neighbors import prior_case  # noqa: F401
from test_seungbeom_inner_best_ratios import completed, make_export  # noqa: F401
from test_seungbeom_inner_best_refine import prepared  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import probability_fusion as run
from experiments.seungbeom import probability_fusion_core as core
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def synthetic(n=360):
    rng = np.random.default_rng(15)
    y = np.tile(np.arange(1, 7), n // 6)
    left, right = rng.normal(size=(n, 6)), rng.normal(size=(n, 6))
    left[np.arange(n), y - 1] += 2
    right[np.arange(n), y - 1] += 1.5
    return softmax(left * 2, axis=1), softmax(right, axis=1), y


def test_temperature_fits_nll_identity_and_changes_blend_not_member_argmax():
    for module in (run, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    rng = np.random.default_rng(4)
    calibrated = softmax(rng.normal(size=(2000, 6)), axis=1)
    y = np.array([rng.choice(6, p=p) + 1 for p in calibrated])
    overconfident = core.temperature(calibrated, 0.5)
    state = core.fit_temperature(overconfident, y)
    assert state["T"] > 1 and state["fit_nll"] < state["before_nll"]
    np.testing.assert_array_equal(core.temperature(overconfident, 1), overconfident)
    np.testing.assert_array_equal(
        core.temperature(overconfident, state["T"]).argmax(1), overconfident.argmax(1)
    )
    neural = np.array([[0.99, 0.01, 0, 0, 0, 0]])
    tree = np.array([[0.35, 0.65, 0, 0, 0, 0]])
    assert (0.3 * neural + 0.7 * tree).argmax(1).item() == 0
    assert (0.3 * core.temperature(neural, 4) + 0.7 * tree).argmax(1).item() == 1
    with pytest.raises(ValueError):
        core.temperature(neural, -1)
    assert np.isfinite(core.log_loss([2], np.array([[1, 0, 0, 0, 0, 0]])))


def test_saved_coefficients_classes_determinism_and_partial_input():
    p, q, y = synthetic()
    state = core.fit(p, q, y)
    loaded = json.loads(json.dumps(state))
    assert loaded == core.fit(p, q, y)
    full = core.apply(p, q, loaded)
    assert loaded["stack"]["classes"] == [1, 2, 3, 4, 5, 6]
    for values in full.values():
        np.testing.assert_allclose(values.sum(1), 1, atol=1e-12)
    for rows in ([359, 2, 8], [1], list(range(len(p)))[::-1]):
        partial = core.apply(p[rows], q[rows], loaded)
        for name in full:
            np.testing.assert_array_equal(full[name][rows], partial[name])
    loaded["stack"]["classes"] = [6, 5, 4, 3, 2, 1]
    with pytest.raises(ValueError, match="stacking"):
        core.apply(p, q, loaded)
    with pytest.raises(ValueError, match="six classes"):
        core.fit(p[y != 6], q[y != 6], y[y != 6])


def test_crossfit_never_fits_held_fold_rows(monkeypatch):
    p, q, y = synthetic(180)
    folds = np.arange(180) // 36
    calls = []
    original = core.fit

    def tracked(left, right, labels):
        calls.append((left.copy(), labels.copy()))
        return original(left, right, labels)

    monkeypatch.setattr(core, "fit", tracked)
    state, diagnostic = core.crossfit(p, q, y, folds)
    assert len(calls) == 6
    for f in range(5):
        np.testing.assert_array_equal(calls[f][0], p[folds != f])
        np.testing.assert_array_equal(calls[f][1], y[folds != f])
    np.testing.assert_array_equal(calls[-1][0], p)
    assert set(diagnostic["fold_excluded_diagnostic"]) == {core.BASELINE, *core.METHODS}
    assert state["contract"]["prior_tau"] is None


def test_duplicate_candidates_are_not_exported_twice():
    labels = {
        core.BASELINE: np.array([1, 1]),
        run.prior30.METHOD: np.array([1, 2]),
        core.METHODS[0]: np.array([2, 2]),
        core.METHODS[1]: np.array([2, 2]),
    }
    decisions = run.duplicate_decisions(labels)
    assert decisions[core.METHODS[0]] == "export"
    assert "skip" in decisions[core.METHODS[1]]
    labels[core.METHODS[0]] = labels[core.BASELINE]
    assert "skip" in run.duplicate_decisions(labels)[core.METHODS[0]]


def test_compare_and_two_exports_preserve_models_no_fit_inputs_at_prediction(
    prior_case, monkeypatch, tmp_path, capsys
):
    training, marker, compare, export = prior_case
    original_sources = run.prior30.sources()
    old_csv = export.output_dir / "submission_inner_best_30_70.csv"
    protected = run.original.file_hashes([old_csv, marker, compare.baseline_recipe])
    recipe = run.compare(compare)
    assert (
        recipe["validation_metrics"][core.BASELINE] == recipe["baseline"]["metrics"][core.BASELINE]
    )
    assert (
        recipe["validation_metrics"][run.prior30.METHOD]
        == recipe["baseline"]["metrics"][run.prior30.METHOD]
    )
    path = compare.output_dir / "recipe.json"
    run.display(run.read_recipe(path), folds=True)
    assert "fold_excluded_NLL" in capsys.readouterr().out
    with pytest.raises(ValueError, match="exists"):
        run.compare(compare)
    # Deliberate synthetic probabilities exercise both distinct export paths.
    with np.load(export.cache_dir / "probabilities.npz", allow_pickle=False) as z:
        ids = z["sample_ids"]
    neural, tree = np.zeros((len(ids), 6)), np.zeros((len(ids), 6))
    neural[:, 0] = np.resize([0.99, 0.1], len(ids))
    neural[:, 1] = 1 - neural[:, 0]
    tree[:, 0] = np.resize([0.35, 0.7], len(ids))
    tree[:, 1] = 1 - tree[:, 0]
    np.savez(
        export.cache_dir / "probabilities.npz", sample_ids=ids, neural=neural, extra_trees=tree
    )
    status = read_json(export.cache_dir / "status.json")
    status["files"]["probabilities.npz"] = sha256(export.cache_dir / "probabilities.npz")
    write_json(export.cache_dir / "status.json", status)
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.oof_dir)
    shutil.rmtree(compare.cache_dir)

    def forbidden(*a, **kw):
        raise AssertionError("No fitting in predict")

    monkeypatch.setattr(core, "fit", forbidden)
    monkeypatch.setattr(core, "fit_temperature", forbidden)
    # inherited cached_case already forbids base-model restoration/inference/fitting
    export.recipe = path
    export.output_dir = tmp_path / "fusion_exports"
    outputs = run.predict(export)
    assert len(outputs) == 2
    _, expected = run.prediction_labels(
        neural, tree, recipe["fusion"], recipe["baseline"]["correction"]
    )
    for name in outputs:
        actual = pd.read_csv(export.output_dir / f"submission_{name}.csv")
        np.testing.assert_array_equal(
            actual.label, pd.Series(expected[name], index=ids).loc[actual.sample_id]
        )
    assert protected == run.original.file_hashes([Path(p) for p in protected])
    assert run.prior30.sources() == original_sources
    with pytest.raises(ValueError, match="exists"):
        run.predict(export)
    bad = read_json(path)
    bad["contract"]["neural_weight"] = 0.9
    write_json(path, bad)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    with pytest.raises(ValueError, match="contract"):
        run.read_recipe(path)


def test_tampered_oof_is_rejected_before_fitting(prior_case, monkeypatch):
    _, _, compare, _ = prior_case
    (compare.oof_dir / "oof.npz").write_bytes(b"changed")

    def forbidden(*a, **kw):
        raise AssertionError("Unverified OOF must not be fitted")

    monkeypatch.setattr(core, "fit", forbidden)
    with pytest.raises(ValueError, match="provenance"):
        run.compare(compare)
