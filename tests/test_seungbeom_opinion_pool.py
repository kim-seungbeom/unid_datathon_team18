"""Synthetic CPU-only calibration/stacking and cached artifact workflow tests."""

# ruff: noqa: F811

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_inner_best_cached_ratios import cached_case  # noqa: F401
from test_seungbeom_inner_best_prior_neighbors import prior_case  # noqa: F401
from test_seungbeom_inner_best_ratios import completed, make_export  # noqa: F401
from test_seungbeom_inner_best_refine import prepared  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import opinion_pool as run
from experiments.seungbeom import opinion_pool_core as core
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def invented(n=1200):
    rng = np.random.default_rng(31)
    return rng.dirichlet(np.ones(6), n), rng.dirichlet(np.ones(6), n)


def test_formula_zeros_contract_and_row_invariance():
    for module in (run, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    p, q = invented()
    p[0] = [1, 0, 0, 0, 0, 0]
    q[0] = [0, 1, 0, 0, 0, 0]
    state = json.loads(json.dumps({"contract": core.CONTRACT}))
    result = core.apply(p, q, state)
    expected = np.clip(p, 1e-12, 1) ** 0.3 * np.clip(q, 1e-12, 1) ** 0.7
    expected /= expected.sum(1, keepdims=True)
    np.testing.assert_allclose(result[core.METHODS[0]], expected, atol=1e-14)
    np.testing.assert_array_equal(result[core.BASELINE], 0.3 * p + 0.7 * q)
    for rows in ([1], [1199, 3, 0], list(range(len(p)))[::-1]):
        partial = core.apply(p[rows], q[rows], state)
        for name, value in result.items():
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value.sum(1), 1)
            np.testing.assert_array_equal(value[rows], partial[name])
    with pytest.raises(ValueError, match="contract"):
        core.apply(p, q, {"contract": {}})
    p[0, 0] = -1
    with pytest.raises(ValueError):
        core.apply(p, q, state)


def test_oof_fixed_rules_no_fit_and_fold_scores():
    p, q = invented(180)
    y = np.tile(np.arange(1, 7), 30)
    folds = np.repeat(np.arange(5), 36)
    state, report = core.evaluate_oof(p, q, y, folds)
    # Label changes can change metrics but never rules/parameters.
    second, _ = core.evaluate_oof(p, q, 7 - y, folds)
    assert state == second == {"contract": core.CONTRACT}
    values = core.apply(p, q, state)
    for f in report["per_fold"]:
        rows = folds == f["fold"]
        for name, probabilities in values.items():
            assert f["metrics"][name] == core.score(y[rows], probabilities[rows])
    for name, probabilities in values.items():
        assert report["pooled_oof"][name] == core.score(y, probabilities)
    with pytest.raises(ValueError, match="folds"):
        core.evaluate_oof(p, q, y, np.zeros(len(y), dtype=int))


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


def test_compare_export_no_training_no_fit_data_preserves_existing(
    prior_case, tmp_path, monkeypatch
):
    training, marker, compare, export = prior_case
    old_sources = run.prior30.sources()
    protected = run.original.file_hashes(
        [
            marker,
            compare.baseline_recipe,
            export.output_dir / "submission_inner_best_30_70.csv",
        ]
    )

    def forbidden(*a, **kw):
        raise AssertionError("Fitting is forbidden")

    monkeypatch.setattr(run.verified.core, "fit", forbidden)
    monkeypatch.setattr(run.verified.core, "fit_temperature", forbidden)
    recipe = run.compare(compare)
    for name in (core.BASELINE, run.prior30.METHOD):
        assert recipe["validation_metrics"][name] == recipe["baseline"]["metrics"][name]
    path = compare.output_dir / "recipe.json"
    run.display(run.read_recipe(path), folds=True)
    with pytest.raises(ValueError, match="exists"):
        run.compare(compare)
    # Synthetic rows exercise distinct geometric and hybrid exports.
    with np.load(export.cache_dir / "probabilities.npz", allow_pickle=False) as z:
        ids = z["sample_ids"]
    p, q = invented()
    values = core.apply(p, q, recipe["fusion"])
    a, g, h = [values[n].argmax(1) for n in (core.BASELINE, *core.METHODS)]
    first = np.flatnonzero((a != g) & (h == a))[0]
    second = np.flatnonzero((a != g) & (h == g))[0]
    rows = np.resize([first, second], len(ids))
    p, q = p[rows], q[rows]
    np.savez(export.cache_dir / "probabilities.npz", sample_ids=ids, neural=p, extra_trees=q)
    status = read_json(export.cache_dir / "status.json")
    status["files"]["probabilities.npz"] = sha256(export.cache_dir / "probabilities.npz")
    write_json(export.cache_dir / "status.json", status)
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.oof_dir)
    shutil.rmtree(compare.cache_dir)
    export.recipe = path
    export.output_dir = tmp_path / "pool_exports"
    outputs = run.predict(export)
    assert len(outputs) == 2
    _, labels = run.prediction_labels(p, q, recipe["fusion"], recipe["baseline"]["correction"])
    for name in outputs:
        csv = pd.read_csv(export.output_dir / f"submission_{name}.csv")
        np.testing.assert_array_equal(
            csv.label, pd.Series(labels[name], index=ids).loc[csv.sample_id]
        )
    assert protected == run.original.file_hashes([Path(p) for p in protected])
    assert old_sources == run.prior30.sources()
    with pytest.raises(ValueError, match="exists"):
        run.predict(export)
    changed = read_json(path)
    changed["contract"]["neural_weight"] = 0.8
    write_json(path, changed)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    with pytest.raises(ValueError, match="contract"):
        run.read_recipe(path)


def test_tampered_oof_rejected_before_comparison(prior_case, monkeypatch):
    _, _, compare, _ = prior_case
    (compare.oof_dir / "oof.npz").write_bytes(b"changed")

    def forbidden(*a, **kw):
        raise AssertionError("Unverified OOF must not be evaluated")

    monkeypatch.setattr(core, "evaluate_oof", forbidden)
    with pytest.raises(ValueError, match="provenance"):
        run.compare(compare)
