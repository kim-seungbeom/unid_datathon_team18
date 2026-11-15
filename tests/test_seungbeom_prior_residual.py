"""Independent synthetic CPU checks; no competition data access."""

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

from experiments.seungbeom import prior_residual as run
from experiments.seungbeom import prior_residual_core as core
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def invented():
    y = np.tile(np.arange(1, 7), 30)
    p = np.eye(6)[y - 1] * 0.9 + 0.1 / 6
    p[y == 2] = [0.51, 0.49, 0, 0, 0, 0]
    folds = np.arange(len(y)) // 36
    return p, y, folds


def test_bounded_deterministic_offsets_and_subset_invariance():
    for module in (run, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    p, y, _ = invented()
    state = core.fit(p, y)
    assert state == core.fit(p, y)
    assert max(abs(x) for x in state["offsets"]) <= 0.15
    predictions = core.apply(p, json.loads(json.dumps(state)))
    assert core.fast_f1(y, predictions[core.METHOD]) > core.fast_f1(y, predictions[core.BASELINE])
    for values in predictions.values():
        assert core.fast_f1(y, values) == pytest.approx(
            run.prior30.classification_metrics(y, values)["macro_f1"]
        )
    for rows in ([179, 2, 1], [0], list(range(len(p)))[::-1]):
        actual = core.apply(p[rows], state)
        for n in predictions:
            np.testing.assert_array_equal(actual[n], predictions[n][rows])
    zero = {**state, "offsets": [0.0] * 6}
    actual = core.apply(p, zero)
    np.testing.assert_array_equal(actual[core.METHOD], actual[core.BASELINE])
    with pytest.raises(ValueError, match="bounded"):
        core.apply(p, {**state, "offsets": [0.2] * 6})


def test_crossfit_excludes_rows_and_refits_prior_tau(monkeypatch):
    p, y, folds = invented()
    original, calls = core.fit, []

    def checked(values, labels):
        calls.append((values.copy(), labels.copy()))
        return original(values, labels)

    monkeypatch.setattr(core, "fit", checked)
    state, diagnosis = core.crossfit(p, p, y, folds)
    assert len(calls) == 6
    mixed = run.prior30.mixture(p, p)
    for i in range(5):
        np.testing.assert_array_equal(calls[i][0], mixed[folds != i])
        np.testing.assert_array_equal(calls[i][1], y[folds != i])
    assert diagnosis["adopt"] and diagnosis["wins"] == 5
    assert state["tau"] == run.prior30.choose(mixed, y, run.prior30.prior_for(y))[0]
    perfect = np.eye(6)[y - 1]
    _, no_gain = core.crossfit(perfect, perfect, y, folds)
    assert not no_gain["adopt"]


def test_compare_replay_gate_and_export_without_fit_inputs(prior_case, monkeypatch, tmp_path):
    training, marker, compare, export = prior_case
    old_csv = export.output_dir / "submission_inner_best_30_70.csv"
    protected = run.original.file_hashes([marker, old_csv, compare.baseline_recipe])
    old_sources = run.prior30.sources()
    record = run.compare(compare)
    assert not record["diagnostics"]["adopt"]  # fixture's base OOF is already perfect
    assert (
        record["validation_metrics"][core.BASELINE] == record["baseline"]["metrics"][core.BASELINE]
    )
    path = compare.output_dir / "recipe.json"
    assert run.read_recipe(path) == record
    export.recipe = path
    export.output_dir = tmp_path / "residual_export"
    # Failed gate must not read Test, caches or even reload base models.
    with monkeypatch.context() as scoped:
        scoped.setattr(run.previous, "load_reference", lambda *a: pytest.fail("Test path accessed"))
        assert run.predict(export) == {}
        assert not export.output_dir.exists()
    # Exercise supported export with a synthetic bounded state. Actual selection
    # and gating are tested above with invented scores rather than forced here.
    record["diagnostics"]["adopt"] = True
    record["correction"]["offsets"] = [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    write_json(path, record)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    with np.load(export.cache_dir / "probabilities.npz", allow_pickle=False) as z:
        ids = z["sample_ids"]
    p = np.tile([0.51, 0.49, 0.0, 0.0, 0.0, 0.0], (len(ids), 1))
    np.savez(export.cache_dir / "probabilities.npz", sample_ids=ids, neural=p, extra_trees=p)
    status = read_json(export.cache_dir / "status.json")
    status["files"]["probabilities.npz"] = sha256(export.cache_dir / "probabilities.npz")
    write_json(export.cache_dir / "status.json", status)
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.oof_dir)
    shutil.rmtree(compare.cache_dir)
    monkeypatch.setattr(core, "fit", lambda *a: pytest.fail("No fitting allowed in predict"))
    outputs = run.predict(export)
    assert set(outputs) == {core.METHOD}
    actual = pd.read_csv(export.output_dir / f"submission_{core.METHOD}.csv")
    expected = core.apply(run.prior30.mixture(p, p), record["correction"])[core.METHOD]
    np.testing.assert_array_equal(
        actual.label, pd.Series(expected, index=ids).loc[actual.sample_id]
    )
    assert run.prior30.sources() == old_sources
    assert run.original.file_hashes([Path(p) for p in protected]) == protected
    bad = copy.deepcopy(record)
    bad["contract"]["penalty"] = 0
    write_json(path, bad)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    with pytest.raises(ValueError, match="changed"):
        run.read_recipe(path)


def test_modified_oof_rejected_before_any_fit(prior_case, monkeypatch):
    _, _, compare, _ = prior_case
    (compare.oof_dir / "oof.npz").write_bytes(b"changed")
    monkeypatch.setattr(core, "fit", lambda *a: pytest.fail("Unverified OOF must never fit"))
    with pytest.raises(ValueError, match="provenance"):
        run.compare(compare)
