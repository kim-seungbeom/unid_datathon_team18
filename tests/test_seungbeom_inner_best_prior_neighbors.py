"""Independent synthetic CPU tests; actual organizer data is never accessed."""

# ruff: noqa: F811

import argparse
import ast
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_inner_best_cached_ratios import cached_case  # noqa: F401
from test_seungbeom_inner_best_prior30 import invented
from test_seungbeom_inner_best_ratios import completed, make_export  # noqa: F401
from test_seungbeom_inner_best_refine import prepared  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import inner_best_prior_neighbors as run
from experiments.seungbeom.export_submissions import read_json, sha256, write_json

pc = run.prior30


def test_ratio_specific_selection_and_old_selection_match():
    ast.parse(Path(run.__file__).read_text(), feature_version=(3, 10))
    p, y, f = invented()
    tree = 0.7 * p + 0.3 / 6
    old = pc.select_arrays(p, tree, y, f)
    replay = run.select_ratio(p, tree, y, f, 0.3)
    for key in replay.keys() - {"weight"}:
        assert replay[key] == old[key]
    for weight in run.RATIOS.values():
        c = run.select_ratio(p, tree, y, f, weight)
        mixed = weight * p + (1 - weight) * tree
        tau, _ = pc.choose(mixed, y, pc.prior_for(y))
        assert c["weight"] == weight and c["tau"] == tau
        assert c["adopt"] and c["tau"] > 0
        assert c["baseline"] == run.metrics_for(y, mixed.argmax(1) + 1)


def test_fixed_corrections_are_rowwise_and_dedup_gate_is_explicit():
    rng = np.random.default_rng(9)
    p = rng.dirichlet(np.ones(6), 20)
    q = rng.dirichlet(np.ones(6), 20)
    correction = {"tau": 0.15, "prior": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]}
    selections = {name: {**correction, "adopt": True} for name in run.RATIOS}
    full = run.predictions(p, q, correction, selections)
    for rows in ([3], [19, 0, 7], list(range(20))[::-1]):
        partial = run.predictions(p[rows], q[rows], correction, selections)
        for name in full:
            np.testing.assert_array_equal(partial[name], full[name][rows])
    first, second = [name + "_prior_tau" for name in run.RATIOS]
    pred = {
        pc.BASELINE: np.array([1, 2]),
        pc.METHOD: np.array([2, 2]),
        first: np.array([3, 2]),
        second: np.array([3, 2]),
    }
    decision = run.export_decisions(pred, selections)
    assert decision[first] == "export" and "identical" in decision[second]
    pred[first] = pred[pc.METHOD].copy()
    decision = run.export_decisions(pred, selections)
    assert "identical" in decision[first] and decision[second] == "export"
    selections["inner_best_25_75"]["adopt"] = False
    assert "OOF" in run.export_decisions(pred, selections)[second]


@pytest.fixture
def prior_case(cached_case, tmp_path):
    training, marker, old_compare, export = cached_case
    reference, _, _, _ = pc.cache.load_reference(old_compare.baseline_recipe, "cpu")
    train = training.data_dir / "train/train.csv"
    frame = pc.read_frame(train, labeled=True)
    labels = frame.label.to_numpy()
    p = np.eye(6)[labels - 1] * 0.9 + 0.1 / 6
    oof = tmp_path / "invented_oof"
    oof.mkdir()
    np.savez(
        oof / "oof.npz",
        sample_ids=frame.sample_id.to_numpy(dtype=str),
        labels=labels,
        fold=pd.factorize(frame.experiment_id)[0],
        neural=p,
        extra_trees=p,
    )
    write_json(
        oof / "oof.json",
        {
            "kind": pc.KIND + "_oof",
            "source_sha256": pc.sources(),
            "reference": reference,
            "train_csv_sha256": sha256(train),
            "oof_sha256": sha256(oof / "oof.npz"),
        },
    )
    write_json(
        oof / "status.json", {"status": "complete", "oof_json_sha256": sha256(oof / "oof.json")}
    )
    sel = argparse.Namespace(
        baseline_recipe=old_compare.baseline_recipe,
        device="cpu",
        train_csv=train,
        oof_dir=oof,
        output_dir=tmp_path / "selection",
    )
    pc.select(sel)
    old_compare.correction = sel.output_dir / "correction.json"
    pc.compare(old_compare)
    compare = copy.copy(old_compare)
    compare.baseline_recipe = old_compare.output_dir / "recipe.json"
    compare.oof_dir = oof
    compare.train_csv = train
    compare.output_dir = tmp_path / "neighbors"
    return training, marker, compare, export


def test_compare_replays_and_rejects_changed_oof_without_inference(prior_case):
    _, marker, compare, _ = prior_case
    before = sha256(marker)
    old_sources = pc.sources()
    recipe = run.compare(compare)
    assert recipe["metrics"][pc.METHOD] == recipe["baseline"]["metrics"][pc.METHOD]
    assert all(not c["adopt"] for c in recipe["selections"].values())
    assert run.read_recipe(compare.output_dir / "recipe.json") == recipe
    assert sha256(marker) == before and pc.sources() == old_sources
    with pytest.raises(ValueError, match="exists"):
        run.compare(compare)
    compare.output_dir = compare.output_dir.with_name("corrupt_rejected")
    (compare.oof_dir / "oof.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        run.compare(compare)


def test_predict_two_csvs_and_skips_without_reading_fit_inputs(prior_case, tmp_path):
    training, marker, compare, export = prior_case
    recipe = run.compare(compare)
    # Construct fixed synthetic supported corrections to exercise both export paths;
    # actual selection/adoption is covered separately with imbalanced synthetic labels.
    for c in recipe["selections"].values():
        c["adopt"] = True
        c["tau"] = 0.5
    path = compare.output_dir / "recipe.json"
    write_json(path, recipe)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    # Synthetic cache with one row changing only at 35%, another only at 25%.
    with np.load(export.cache_dir / "probabilities.npz", allow_pickle=False) as z:
        ids = z["sample_ids"]
    neural = np.zeros((len(ids), 6))
    neural[:, 0] = 1
    tree = np.zeros_like(neural)
    tree[:, 0] = np.resize([0.24, 0.32], len(ids))
    tree[:, 1] = 1 - tree[:, 0]
    np.savez(
        export.cache_dir / "probabilities.npz", sample_ids=ids, neural=neural, extra_trees=tree
    )
    cache_status = read_json(export.cache_dir / "status.json")
    cache_status["files"]["probabilities.npz"] = sha256(export.cache_dir / "probabilities.npz")
    write_json(export.cache_dir / "status.json", cache_status)
    old_csv = export.output_dir / "submission_inner_best_30_70.csv"
    protected = pc.original.file_hashes([old_csv, marker, compare.baseline_recipe])
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.cache_dir)
    shutil.rmtree(compare.oof_dir)
    export.recipe = path
    export.output_dir = tmp_path / "neighbor_exports"
    outputs = run.predict(export)
    assert len(outputs) == 2
    expected = run.predictions(neural, tree, recipe["baseline"]["correction"], recipe["selections"])
    for method in outputs:
        actual = pd.read_csv(export.output_dir / f"submission_{method}.csv")
        np.testing.assert_array_equal(
            actual.label, pd.Series(expected[method], index=ids).loc[actual.sample_id]
        )
    assert protected == pc.original.file_hashes([Path(p) for p in protected])
    with pytest.raises(ValueError, match="exists"):
        run.predict(export)
    for c in recipe["selections"].values():
        c["adopt"] = False
    write_json(path, recipe)
    write_json(path.parent / "status.json", {"status": "complete", "sha256": sha256(path)})
    export.output_dir = tmp_path / "skipped_exports"
    assert run.predict(export) == {}
    assert not list(export.output_dir.glob("*.csv"))
