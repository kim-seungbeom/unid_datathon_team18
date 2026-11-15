"""Independent synthetic CPU checks; never open competition inputs."""

# ruff: noqa: F811

import ast
import copy
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

from experiments.seungbeom import confidence_blend as run
from experiments.seungbeom import confidence_blend_core as core
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def invented():
    y = np.tile(np.arange(1, 7), 30)
    p = np.eye(6)[y - 1]
    q = 0.28 * p + 0.72 * np.roll(p, 1, axis=1)
    return p, q, y, np.arange(len(y)) // 36


def test_entropy_bounds_sign_and_exact_zero_baseline():
    for module in (run, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    np.testing.assert_allclose(core.normalized_entropy(np.full((3, 6), 1 / 6)), 1)
    np.testing.assert_array_equal(core.normalized_entropy(np.eye(6)), 0)
    p, q, _, _ = invented()
    positive, a = core.mixture(p, q, 1)
    negative, b = core.mixture(p, q, -1)
    assert (a > 0.3).all() and (b < 0.3).all()
    for strength in core.STRENGTHS:
        mixed, weights = core.mixture(p, q, strength)
        assert (weights >= 0.2).all() and (weights <= 0.4).all()
        np.testing.assert_allclose(mixed.sum(1), 1)
        assert np.isfinite(mixed).all()
    baseline, a = core.mixture(p, q, 0)
    np.testing.assert_array_equal(baseline, run.prior30.mixture(p, q))
    np.testing.assert_array_equal(a, np.full(len(p), 0.3))
    assert not np.array_equal(positive, negative)
    with pytest.raises(ValueError, match="grid"):
        core.mixture(p, q, 2)


def test_selection_identity_subset_and_serialization():
    p, q, y, _ = invented()
    state = core.fit(p, q, y)
    assert state == core.fit(p, q, y)
    assert state["strength"] > 0
    assert state["tau"] == 0
    full = core.apply(p, q, json.loads(json.dumps(state)))
    assert (full[core.METHOD] == y).all()
    assert not (full[core.BASELINE] == y).all()
    for rows in ([179, 2, 1], [0], list(range(len(y)))[::-1]):
        partial = core.apply(p[rows], q[rows], state)
        for n in full:
            np.testing.assert_array_equal(partial[n], full[n][rows])
    assert core.fit(p, p, y)["strength"] == 0
    with pytest.raises(ValueError, match="contract"):
        core.apply(p, q, {**state, "strength": 2})


def test_crossfit_excludes_held_rows(monkeypatch):
    p, q, y, folds = invented()
    calls, original = [], core.fit

    def track(left, right, labels):
        calls.append((left.copy(), right.copy(), labels.copy()))
        return original(left, right, labels)

    monkeypatch.setattr(core, "fit", track)
    _, diagnostic = core.crossfit(p, q, y, folds)
    assert len(calls) == 6
    for f in range(5):
        for actual, expected in zip(
            calls[f], (p[folds != f], q[folds != f], y[folds != f]), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    assert diagnostic["recommended"] and diagnostic["wins"] == 5
    _, unchanged = core.crossfit(p, p, y, folds)
    assert not unchanged["recommended"]


def test_replay_and_exploratory_export_without_fit_inputs(prior_case, monkeypatch, tmp_path):
    training, marker, compare, export = prior_case
    old_csv = export.output_dir / "submission_inner_best_30_70.csv"
    protected = run.original.file_hashes([marker, old_csv, compare.baseline_recipe])
    old_sources = run.prior30.sources()
    record = run.compare(compare)
    assert not record["diagnostics"]["recommended"]
    assert record["correction"]["strength"] == 0
    assert (
        record["validation_metrics"][core.BASELINE] == record["baseline"]["metrics"][core.BASELINE]
    )
    path = compare.output_dir / "recipe.json"
    assert run.read_recipe(path) == record
    export.recipe, export.output_dir = path, tmp_path / "confidence_export"
    with monkeypatch.context() as scoped:
        scoped.setattr(
            run.previous, "load_reference", lambda *a: pytest.fail("No Test access for strength0")
        )
        assert run.predict(export) == {}
    # A fixed synthetic nonzero state exercises exploratory export even when the
    # recommendation is False. Selection itself is independently tested above.
    record["correction"]["strength"] = 1.0
    write_json(path, record)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    with np.load(export.cache_dir / "probabilities.npz", allow_pickle=False) as z:
        ids = z["sample_ids"]
    p = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (len(ids), 1))
    q = np.tile([0.28, 0.72, 0.0, 0.0, 0.0, 0.0], (len(ids), 1))
    np.savez(export.cache_dir / "probabilities.npz", sample_ids=ids, neural=p, extra_trees=q)
    status = read_json(export.cache_dir / "status.json")
    status["files"]["probabilities.npz"] = sha256(export.cache_dir / "probabilities.npz")
    write_json(export.cache_dir / "status.json", status)
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.oof_dir)
    shutil.rmtree(compare.cache_dir)
    monkeypatch.setattr(core, "fit", lambda *a: pytest.fail("No fitting at export"))
    outputs = run.predict(export)
    assert set(outputs) == {core.METHOD}
    actual = pd.read_csv(export.output_dir / f"submission_{core.METHOD}.csv")
    expected = core.apply(p, q, record["correction"])[core.METHOD]
    np.testing.assert_array_equal(
        actual.label, pd.Series(expected, index=ids).loc[actual.sample_id]
    )
    assert run.prior30.sources() == old_sources
    assert run.original.file_hashes([Path(p) for p in protected]) == protected
    bad = copy.deepcopy(record)
    bad["contract"]["base_neural_weight"] = 0.8
    write_json(path, bad)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    with pytest.raises(ValueError, match="changed"):
        run.read_recipe(path)


def test_tampered_oof_rejected_before_fitting(prior_case, monkeypatch):
    _, _, compare, _ = prior_case
    (compare.oof_dir / "oof.npz").write_bytes(b"changed")
    monkeypatch.setattr(core, "fit", lambda *a: pytest.fail("Unverified OOF"))
    with pytest.raises(ValueError, match="provenance"):
        run.compare(compare)
