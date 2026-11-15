"""Synthetic inputs only; no organizer files, GPU, network, or real predictions."""

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
from test_seungbeom_inner_best_ratios import completed, make_export  # noqa: F401
from test_seungbeom_inner_best_refine import prepared  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import inner_best_prior30 as pc
from experiments.seungbeom.export_submissions import read_json, sha256, write_json
from schema import RAW_SENSOR_COLUMNS


def invented():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(600, int), np.repeat(np.arange(2, 7), 80)])
    logits = rng.normal(size=(len(y), 6))
    logits[np.arange(len(y)), y - 1] += 2.5
    logits[:, 0] += 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return p, y, np.arange(len(y)) % 5


def test_selection_and_fixed_rowwise_correction():
    ast.parse(Path(pc.__file__).read_text(), feature_version=(3, 10))
    p, y, folds = invented()
    result = pc.select_arrays(p, p, y, folds)
    assert result["adopt"] and result["tau"] > 0
    assert result["fold_excluded_diagnostic"]["macro_f1"] > result["baseline"]["macro_f1"]
    np.testing.assert_array_equal(pc.corrected(p, result["prior"], 0), p.argmax(1) + 1)
    full = pc.corrected(p, result["prior"], result["tau"])
    for rows in ([9, 0, 7], [700], list(range(len(p) - 1, -1, -1))):
        np.testing.assert_array_equal(
            pc.corrected(p[rows], result["prior"], result["tau"]), full[rows]
        )
    assert pc.choose(p, y, np.full(6, 1 / 6))[0] == 0
    for prior in ([0, 1, 0, 0, 0, 0], [1] * 6, [np.nan] * 6):
        with pytest.raises(ValueError):
            pc.corrected(p, prior, 0.5)


def test_oof_identity_and_group_boundaries():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(18)],
            "label": np.tile(np.arange(1, 7), 3),
            "experiment_id": np.repeat(["a", "b", "c"], 6),
        }
    )
    z = {
        "sample_ids": frame.sample_id.to_numpy(),
        "labels": frame.label.to_numpy(),
        "fold": np.repeat([0, 1, 2], 6),
    }
    pc.check_oof_rows(z, frame)
    bad = copy.deepcopy(z)
    bad["fold"][0] = 2
    with pytest.raises(ValueError, match="split an experiment"):
        pc.check_oof_rows(bad, frame)
    bad = copy.deepcopy(z)
    bad["sample_ids"][0] = bad["sample_ids"][1]
    with pytest.raises(ValueError, match="exactly once"):
        pc.check_oof_rows(bad, frame)
    with pytest.raises(ValueError):
        pc.prior_for([1, 2, 3, 4, 5])


def test_oof_head_generation_is_deterministic_and_train_only(tiny_backbone, tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    n = 30
    frame = pd.DataFrame({name: rng.normal(size=n) for name in RAW_SENSOR_COLUMNS})
    frame["sample_id"] = [f"invented_{i}" for i in range(n)]
    frame["label"] = np.tile(np.arange(1, 7), 5)
    frame["experiment_id"] = np.repeat(np.arange(5).astype(str), 6)
    frame["image_path"] = "not_read.png"
    train = tmp_path / "data/train"
    train.mkdir(parents=True)
    frame.to_csv(train / "train.csv", index=False)
    tree = tmp_path / "tree"
    embeddings = tree / "embeddings/train"
    embeddings.mkdir(parents=True)
    features = rng.normal(size=(n, 512)).astype(np.float32)
    np.save(embeddings / "features.npy", features)
    frame[["sample_id"]].to_csv(embeddings / "sample_ids.csv", index=False)
    write_json(
        embeddings / "metadata.json",
        {
            "status": "complete",
            "model": "resnet18",
            "weights": pc.oof_code.ft.WEIGHTS_SOURCE,
            "l2_normalize": False,
            "dimension": 512,
        },
    )
    p = np.eye(6)[frame.label.to_numpy() - 1] * 0.9 + 0.1 / 6
    np.savez(
        tree / "train_oof.npz",
        sample_ids=frame.sample_id.to_numpy(dtype=str),
        labels=frame.label.to_numpy(),
        fold=np.repeat(np.arange(5), 6),
        model_names=np.array(["extra_trees"]),
        probabilities=p[:, None, :],
    )
    config = pc.oof_code.ft.normalize_config(
        {"sensor_hidden": 4, "fusion_hidden": 8, "batch_size": 16, "epochs": 1}
    )
    manifest = {"config": config, "inputs": {"train": {"csv_sha256": sha256(train / "train.csv")}}}
    monkeypatch.setattr(
        pc.oof_code.fc, "read_result", lambda path: (manifest, {"selected_epoch": 1})
    )
    args = argparse.Namespace(
        output_dir=tmp_path / "oof1",
        device="cpu",
        data_dir=train.parent,
        tree_run=tree,
        checkpoint_run=tmp_path / "unused",
    )
    pc.oof_code.oof(args)
    with np.load(args.output_dir / "oof.npz", allow_pickle=False) as z:
        first = z["neural"]
        pc.check_oof_rows(z, frame)
        np.testing.assert_array_equal(z["extra_trees"], p)
    args.output_dir = tmp_path / "oof2"
    pc.oof_code.oof(args)
    with np.load(args.output_dir / "oof.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(first, z["neural"])
    assert read_json(args.output_dir / "status.json")["status"] == "complete"


def test_select_pipeline_binds_train_oof_sources_and_preserves_input(cached_case, tmp_path):
    training, marker, compare, _ = cached_case
    reference, base, manifest, _ = pc.cache.load_reference(compare.baseline_recipe, "cpu")
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
    args = argparse.Namespace(
        baseline_recipe=compare.baseline_recipe,
        device="cpu",
        train_csv=train,
        oof_dir=oof,
        output_dir=tmp_path / "selection",
    )
    before = sha256(oof / "oof.npz")
    c = pc.select(args)
    assert c["tau"] == 0 and not c["adopt"]
    assert sha256(oof / "oof.npz") == before
    assert pc.read_correction(args.output_dir / "correction.json") == c
    (oof / "oof.npz").write_bytes(b"corrupt")
    args.output_dir = tmp_path / "reject"
    with pytest.raises(ValueError, match="provenance"):
        pc.select(args)


def test_cached_validation_export_no_fit_no_train_val_reads(cached_case, tmp_path):
    training, marker, compare, old_export = cached_case
    # cached_case makes model loading/inference raise; only verified caches can work.
    reference, _, _, _ = pc.cache.load_reference(compare.baseline_recipe, "cpu")
    p, y, folds = invented()
    c = pc.select_arrays(p, p, y, folds)
    c.update(
        {
            "kind": pc.KIND,
            "weight": 0.3,
            "labels": list(pc.LABELS),
            "source_sha256": pc.sources(),
            "reference": reference,
            "reference_path": str(compare.baseline_recipe.resolve()),
        }
    )
    selection = tmp_path / "synthetic_correction"
    selection.mkdir()
    write_json(selection / "correction.json", c)
    write_json(
        selection / "status.json",
        {"status": "complete", "sha256": sha256(selection / "correction.json")},
    )
    compare.correction = selection / "correction.json"
    recipe = pc.compare(compare)
    assert recipe["metrics"][pc.BASELINE] == reference["metrics"][pc.BASELINE]
    # Export must not read Train/Validation inputs or fitting artifacts.
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.cache_dir)
    shutil.rmtree(selection)
    old_csv = old_export.output_dir / "submission_inner_best_30_70.csv"
    before = sha256(old_csv)
    old_export.recipe = compare.output_dir / "recipe.json"
    old_export.output_dir = tmp_path / "new_prior_export"
    output = pc.predict(old_export)
    assert sha256(old_csv) == before
    exported = pd.read_csv(output)
    with np.load(old_export.cache_dir / "probabilities.npz", allow_pickle=False) as z:
        pred = pc.corrected(pc.mixture(z["neural"], z["extra_trees"]), c["prior"], c["tau"])
        expected = pd.Series(pred, index=z["sample_ids"]).loc[exported.sample_id]
    np.testing.assert_array_equal(exported.label, expected)
    with pytest.raises(ValueError, match="exists"):
        pc.predict(old_export)
    # A rejected correction cannot be exported by adding an override flag.
    saved = read_json(old_export.recipe)
    saved["correction"]["adopt"] = False
    write_json(old_export.recipe, saved)
    write_json(
        old_export.recipe.parent / "status.json",
        {"status": "complete", "sha256": sha256(old_export.recipe)},
    )
    old_export.output_dir = tmp_path / "not_adopted"
    with pytest.raises(ValueError, match="did not support"):
        pc.predict(old_export)
