"""Synthetic CPU tests only: exact caches, no training, no private data/network."""

import argparse
import ast
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_inner_best_ratios import completed, make_export  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import inner_best_refine as experiment
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def test_fixed_ratios_keep_original_40_60_arithmetic():
    ast.parse(Path(experiment.__file__).read_text(), feature_version=(3, 10))
    neural = np.array([[0.95, 0.05, 0, 0, 0, 0]], dtype=np.float32)
    tree = np.array([[0.28, 0.72, 0, 0, 0, 0]], dtype=np.float64)
    values = experiment.candidates(neural, tree)
    assert values["inner_best_35_65"].argmax() == 0
    assert values["inner_best_30_70"].argmax() == 1
    for name, weight in experiment.RATIOS.items():
        np.testing.assert_array_equal(
            values[name], weight * neural.astype(float) + (1 - weight) * tree
        )
    for name in experiment.original.RATIOS:
        np.testing.assert_array_equal(
            values[name], experiment.original.weighted_probabilities(neural, tree, name)
        )


@pytest.fixture
def prepared(completed, tmp_path):  # noqa: F811
    args, frames, marker, old_compare = completed
    # The completed fixture already makes training/preprocessing fit raise if called again.
    experiment.original.compare(old_compare)
    refine = argparse.Namespace(
        baseline_recipe=old_compare.output_dir / "recipe.json",
        input=old_compare.input,
        image_root=old_compare.image_root,
        output_dir=tmp_path / "refinement",
        cache_dir=tmp_path / "caches" / "validation",
        device="cpu",
    )
    return args, frames, marker, refine


def test_cache_reuse_exports_and_subset_invariance_without_training_or_validation_access(
    prepared, monkeypatch, tmp_path, capsys
):
    args, frames, marker, refine = prepared
    before = experiment.original.run_files(args.output_dir)
    original_files = experiment.original.file_hashes(
        [marker, refine.baseline_recipe, refine.baseline_recipe.parent / "status.json"]
    )
    calls = []
    infer = experiment.original.infer_members

    def tracking(frame, *a):
        calls.append(frame.sample_id.tolist())
        return infer(frame, *a)

    def forbidden(*a, **kw):
        raise AssertionError("A cache hit must not load or run either model")

    monkeypatch.setattr(experiment.original, "infer_members", tracking)
    recipe = experiment.compare(refine)
    assert recipe["replay_matches"] == dict.fromkeys(experiment.original.METHODS, True)
    assert len(calls) == 1
    assert set(recipe["metrics"]) == set((*experiment.original.METHODS, *experiment.RATIOS))
    experiment.report(
        argparse.Namespace(recipe=refine.output_dir / "recipe.json", class_details=True)
    )
    assert "delta_vs_40_60" in capsys.readouterr().out
    rerun = copy.copy(refine)
    rerun.output_dir = tmp_path / "cached_comparison"
    with monkeypatch.context() as isolated:
        isolated.setattr(experiment.original, "load_members", forbidden)
        isolated.setattr(experiment.original, "infer_members", forbidden)
        repeated = experiment.compare(rerun)
    assert repeated["metrics"] == recipe["metrics"]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Output directory exists"):
        experiment.compare(refine)

    # Test inference must not open Train/Validation CSVs or the Validation probability cache.
    for split in frames:
        (args.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(refine.cache_dir)
    frame = frames["validation"].assign(label="never a feature", experiment_id="never a feature")
    export = make_export(tmp_path, refine, frame, "test_all")
    export.cache_dir = tmp_path / "caches" / "test"
    outputs = experiment.predict(export)
    assert len(calls) == 2
    assert set(outputs) == set(experiment.RATIOS)
    assert len(list(export.output_dir.glob("*.csv"))) == 2
    status = read_json(export.output_dir / "status.json")
    assert status["status"] == "complete"
    cache_context = read_json(export.cache_dir / "metadata.json")
    (neural, tree), _ = experiment.read_cache(
        export.cache_dir, cache_context, frame.sample_id.tolist()
    )
    expected = experiment.candidates(neural, tree)
    predictions = {}
    for name in experiment.RATIOS:
        path = export.output_dir / f"submission_{name}.csv"
        result = pd.read_csv(path).set_index("sample_id").label
        predictions[name] = result
        assert result.index.tolist() == frame.sample_id.iloc[::-1].tolist()
        np.testing.assert_array_equal(result.to_numpy(), (expected[name].argmax(1) + 1)[::-1])
        assert result.between(1, 6).all() and result.dtype.kind in "iu"
        assert status["changed_rows_vs_40_60"][name] == experiment.changes(expected)[name]
    cached_export = copy.copy(export)
    cached_export.output_dir = tmp_path / "cached_exports"
    with monkeypatch.context() as isolated:
        isolated.setattr(experiment.original, "load_members", forbidden)
        isolated.setattr(experiment.original, "infer_members", forbidden)
        experiment.predict(cached_export)
    for name in experiment.RATIOS:
        assert sha256(export.output_dir / f"submission_{name}.csv") == sha256(
            cached_export.output_dir / f"submission_{name}.csv"
        )
    with pytest.raises(ValueError, match="Output directory exists"):
        experiment.predict(export)

    subset = make_export(tmp_path, refine, frame.iloc[[9, 1, 7]], "test_subset")
    subset.cache_dir = export.cache_dir
    with pytest.raises(ValueError, match="Cache inputs/models/runtime differ"):
        experiment.predict(subset)
    assert len(calls) == 2  # Full-input cached predictions are never looked up for a subset.
    subset.output_dir = tmp_path / "uncached_subset"
    subset.cache_dir = None
    experiment.predict(subset)
    for name in experiment.RATIOS:
        result = (
            pd.read_csv(subset.output_dir / f"submission_{name}.csv").set_index("sample_id").label
        )
        pd.testing.assert_series_equal(result, predictions[name].loc[result.index])
    assert experiment.original.run_files(args.output_dir) == before
    experiment.check_files(original_files)


def test_cache_preserves_float_bits_rejects_mismatch_and_corruption(tmp_path):
    ids = ["id_02", "id_01"]
    neural = np.full((2, 6), 1 / 6, dtype=np.float32)
    tree = np.full((2, 6), 1 / 6, dtype=np.float64)
    identity = {"input": "exact invented input", "runtime": "synthetic cpu"}
    directory = tmp_path / "cache"
    experiment.save_cache(directory, identity, ids, neural, tree)
    (left, right), _ = experiment.read_cache(directory, identity, ids)
    assert left.dtype == neural.dtype and right.dtype == tree.dtype
    assert left.tobytes() == neural.tobytes() and right.tobytes() == tree.tobytes()
    with pytest.raises(FileExistsError):
        experiment.save_cache(directory, identity, ids, neural, tree)
    with pytest.raises(ValueError, match="Cache inputs/models/runtime differ"):
        experiment.read_cache(directory, {**identity, "runtime": "different"}, ids)
    with pytest.raises(ValueError, match="sample order"):
        experiment.read_cache(directory, identity, ids[::-1])
    with (directory / "probabilities.npz").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="Source run files changed"):
        experiment.read_cache(directory, identity, ids)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="Cache incomplete"):
        experiment.read_cache(incomplete, identity, ids)
    with pytest.raises(ValueError):
        experiment.save_cache(tmp_path / "invalid", identity, ids, neural * np.nan, tree)
    assert not (tmp_path / "invalid").exists()


def test_cached_invalid_values_are_rejected_even_with_updated_file_checksum(tmp_path):
    directory = tmp_path / "invalid_cache"
    ids = ["invented"]
    values = np.full((1, 6), 1 / 6, dtype=np.float32)
    experiment.save_cache(directory, {}, ids, values, values)
    np.savez(
        directory / "probabilities.npz",
        sample_ids=np.array(ids),
        neural=values * 0,
        extra_trees=values,
    )
    status = read_json(directory / "status.json")
    status["files"]["probabilities.npz"] = sha256(directory / "probabilities.npz")
    write_json(directory / "status.json", status)
    with pytest.raises(ValueError, match="sum to one"):
        experiment.read_cache(directory, {}, ids)


def test_replay_mismatch_never_creates_exportable_recipe(prepared, monkeypatch):
    _, _, _, refine = prepared
    infer = experiment.original.infer_members

    def changed(*a):
        neural, tree = infer(*a)
        return np.roll(neural, 1, axis=1), tree

    monkeypatch.setattr(experiment.original, "infer_members", changed)
    with pytest.raises(ValueError, match="Original Validation replay differs"):
        experiment.compare(refine)
    assert read_json(refine.output_dir / "status.json")["status"] == "failed"
    assert not (refine.output_dir / "recipe.json").exists()


def test_rejects_wrong_split_runtime_model_recipe_and_export_ids(prepared, monkeypatch, tmp_path):
    args, frames, _, refine = prepared
    wrong = copy.copy(refine)
    wrong.input = args.data_dir / "train" / "train.csv"
    with pytest.raises(ValueError, match="Validation CSV"):
        experiment.compare(wrong)
    with monkeypatch.context() as isolated:
        isolated.setattr(
            experiment.checkpoint, "runtime_signature", lambda device: {"changed": True}
        )
        with pytest.raises(ValueError, match="models/runtime/replay"):
            experiment.compare(refine)
    experiment.compare(refine)
    export = make_export(tmp_path, refine, frames["validation"], "reject")
    export.cache_dir = tmp_path / "test_cache"
    pd.DataFrame({"sample_id": ["wrong"], "label": [1]}).to_csv(
        export.sample_submission, index=False
    )
    with pytest.raises(ValueError, match="IDs"):
        experiment.predict(export)
    assert not export.output_dir.exists()
    recipe_path = refine.output_dir / "recipe.json"
    saved = recipe_path.read_bytes()
    recipe_path.write_bytes(saved + b" ")
    with pytest.raises(ValueError, match="recipe changed"):
        experiment.predict(export)
    recipe_path.write_bytes(saved)
    with (args.output_dir / "inner_best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source run files changed"):
        experiment.predict(export)
