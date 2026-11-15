"""Synthetic CPU OOF selection and cache-only export for the new 40:60 experiment."""

# ruff: noqa: F811

import argparse
import shutil

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_inner_best_cached_ratios import cached_case  # noqa: F401
from test_seungbeom_inner_best_prior30 import invented
from test_seungbeom_inner_best_ratios import completed, make_export  # noqa: F401
from test_seungbeom_inner_best_refine import prepared  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import inner_best_prior40 as pc


def test_ratio_is_40_60_and_fold_tau_is_selected_without_held_labels():
    p, y, folds = invented()
    tree = np.roll(p, 1, axis=1) * 0.2 + p * 0.8
    mixture = pc.mixture(p, tree)
    np.testing.assert_array_equal(mixture, 0.4 * p + 0.6 * tree)
    before = pc.origin.sources()
    result = pc.select_arrays(p, tree, y, folds)
    assert result["tau"] == pc.choose(mixture, y, pc.prior_for(y))[0]
    for row in result["per_fold"]:
        fit = folds != row["fold"]
        assert row["tau"] == pc.choose(mixture[fit], y[fit], pc.prior_for(y[fit]))[0]
    labels = pc.corrected(mixture, result["prior"], result["tau"])
    subset = [17, 2, 600]
    np.testing.assert_array_equal(
        pc.corrected(mixture[subset], result["prior"], result["tau"]),
        labels[subset],
    )
    assert pc.origin.sources() == before


def test_select_existing_prior30_oof_then_compare_export_without_training(
    cached_case,
    tmp_path,
):
    training, marker, compare, old_export = cached_case
    reference, base, manifest, _ = pc.cache.load_reference(compare.baseline_recipe, "cpu")
    train = training.data_dir / "train/train.csv"
    frame = pc.read_frame(train, labeled=True)
    y = frame.label.to_numpy()
    probabilities = np.eye(6)[y - 1] * 0.9 + 0.1 / 6
    oof = tmp_path / "prior30_oof"
    oof.mkdir()
    np.savez(
        oof / "oof.npz",
        sample_ids=frame.sample_id.to_numpy(dtype=str),
        labels=y,
        fold=pd.factorize(frame.experiment_id)[0],
        neural=probabilities,
        extra_trees=probabilities,
    )
    pc.write_json(
        oof / "oof.json",
        {
            "kind": pc.origin.KIND + "_oof",
            "source_sha256": pc.origin.sources(),
            "reference": reference,
            "train_csv_sha256": pc.sha256(train),
            "oof_sha256": pc.sha256(oof / "oof.npz"),
        },
    )
    pc.write_json(
        oof / "status.json",
        {
            "status": "complete",
            "oof_json_sha256": pc.sha256(oof / "oof.json"),
        },
    )
    args = argparse.Namespace(
        baseline_recipe=compare.baseline_recipe,
        device="cpu",
        train_csv=train,
        oof_dir=oof,
        output_dir=tmp_path / "selection",
    )
    old_hash = pc.sha256(oof / "oof.npz")
    c = pc.select(args)
    assert c["weight"] == 0.4 and c["tau"] == 0
    assert not c["adopt"]
    assert pc.sha256(oof / "oof.npz") == old_hash
    assert pc.read_correction(args.output_dir / "correction.json") == c
    compare.correction = args.output_dir / "correction.json"
    result = pc.compare(compare)
    assert result["metrics"][pc.BASELINE] == reference["metrics"][pc.BASELINE]
    old_csv = old_export.output_dir / "submission_inner_best_30_70.csv"
    old_csv_hash = pc.sha256(old_csv)
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(oof)
    shutil.rmtree(args.output_dir)
    shutil.rmtree(compare.cache_dir)
    old_export.recipe = compare.output_dir / "recipe.json"
    old_export.output_dir = tmp_path / "new40"
    output = pc.predict(old_export)
    assert pc.sha256(old_csv) == old_csv_hash
    record = pc.read_json(output.with_suffix(".manifest.json"))
    assert record["new_40_60_tau_experiment"]
    assert not record["original_submission_match_verified"]
    assert record["public_score"] is None
    with pytest.raises(ValueError, match="exists"):
        pc.predict(old_export)
