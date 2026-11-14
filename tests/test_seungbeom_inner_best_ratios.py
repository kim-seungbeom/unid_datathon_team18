"""Independent synthetic CPU-only inference checks; no competition files or network."""

import argparse
import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import frozen_checkpoint as checkpoint
from experiments.seungbeom import inner_best_ratios as experiment
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def test_python310_and_probability_weights_not_label_averages():
    ast.parse(Path(experiment.__file__).read_text(), feature_version=(3, 10))
    neural = np.array([[0.8, 0.2, 0, 0, 0, 0]], dtype=np.float32)
    tree = np.array([[0.1, 0.9, 0, 0, 0, 0]], dtype=np.float32)
    assert experiment.weighted_probabilities(neural, tree, "inner_best_40_60").argmax() == 1
    assert experiment.weighted_probabilities(neural, tree, "inner_best_60_40").argmax() == 0
    for name, weight in experiment.RATIOS.items():
        actual = experiment.weighted_probabilities(neural, tree, name)
        np.testing.assert_allclose(actual, weight * neural.astype(float) + (1 - weight) * tree)
        np.testing.assert_allclose(actual.sum(1), 1)
    np.testing.assert_array_equal(
        experiment.weighted_probabilities(neural, tree, "inner_best_50_50"),
        experiment.blend.midpoint.blend_probabilities(neural, tree),
    )
    with pytest.raises(ValueError):
        experiment.weighted_probabilities(neural, tree[:, :5], "inner_best_40_60")
    with pytest.raises(ValueError):
        experiment.weighted_probabilities(neural, tree * np.nan, "inner_best_60_40")
    with pytest.raises(ValueError):
        experiment.weighted_probabilities(neural, tree, "choose_by_test")


def test_source_compatibility_only_allows_reviewed_driver_change():
    current = checkpoint.source_hashes()
    recorded = dict(current)
    recorded[experiment.DRIVER] = next(
        value for value in experiment.REVIEWED_DRIVERS if value != current[experiment.DRIVER]
    )
    assert experiment.check_training_sources(current, current) == "exact"
    assert (
        experiment.check_training_sources(recorded, current)
        == "reviewed_training_budget_only_change"
    )
    dependency = next(name for name in current if name != experiment.DRIVER)
    for altered in (
        {**recorded, dependency: "changed"},
        {**recorded, experiment.DRIVER: "unknown"},
        {**recorded, "extra.py": "unknown"},
        {k: v for k, v in recorded.items() if k != dependency},
    ):
        with pytest.raises(ValueError):
            experiment.check_training_sources(altered, current)
    with pytest.raises(ValueError):
        experiment.check_training_sources(recorded, {**current, experiment.DRIVER: "unknown"})


@pytest.fixture
def completed(case, monkeypatch, tmp_path):  # noqa: F811
    args, frames, _, original = case
    checkpoint.run(args)

    def forbidden(*a, **kw):
        raise AssertionError("Inference must never fit or train")

    monkeypatch.setattr(checkpoint.core, "train_phase", forbidden)
    monkeypatch.setattr(checkpoint.ft, "train_phase", forbidden)
    monkeypatch.setattr(checkpoint.ft.NeuralSensorPreprocessor, "fit", forbidden)
    compare = argparse.Namespace(
        run_dir=args.output_dir,
        input=args.data_dir / "validation" / "validation.csv",
        image_root=args.data_dir / "validation",
        output_dir=tmp_path / "comparison",
        device="cpu",
    )
    return args, frames, original, compare


def make_export(tmp_path, compare, frame, suffix):
    path = tmp_path / f"invented_{suffix}.csv"
    frame.to_csv(path, index=False)
    template = tmp_path / f"template_{suffix}.csv"
    pd.DataFrame({"sample_id": frame.sample_id.iloc[::-1], "label": 1}).to_csv(
        template, index=False
    )
    return argparse.Namespace(
        recipe=compare.output_dir / "recipe.json",
        input=path,
        image_root=compare.image_root,
        output_dir=tmp_path / f"exports_{suffix}",
        device="cpu",
        sample_submission=template,
    )


def test_replay_and_two_exports_once_per_member_preserve_subset_and_originals(
    completed, monkeypatch, tmp_path, capsys
):
    args, frames, original, compare = completed
    before = experiment.run_files(args.output_dir)
    marker = sha256(original)
    calls = {"neural": 0, "tree": 0}
    neural_method = checkpoint.probability
    tree_method = experiment.blend.midpoint.method_probabilities

    def neural(*a, **kw):
        calls["neural"] += 1
        assert "label" not in a[2]
        return neural_method(*a, **kw)

    def tree(*a, **kw):
        calls["tree"] += 1
        assert "label" not in a[0]
        return tree_method(*a, **kw)

    monkeypatch.setattr(checkpoint, "probability", neural)
    monkeypatch.setattr(experiment.blend.midpoint, "method_probabilities", tree)
    recipe = experiment.compare(compare)
    assert recipe["replay_matches"] == dict.fromkeys(experiment.REPLAY, True)
    assert calls == {"neural": 1, "tree": 1}
    with pytest.raises(ValueError, match="Output directory exists"):
        experiment.compare(compare)
    experiment.report(
        argparse.Namespace(recipe=compare.output_dir / "recipe.json", class_details=True)
    )
    assert "delta_vs_50_50" in capsys.readouterr().out
    assert calls == {"neural": 1, "tree": 1}  # report never infers

    # Test inference must not depend on Train/Validation CSVs or original recipe files.
    args.recipe.unlink()
    for split in frames:
        (args.data_dir / split / f"{split}.csv").unlink()
    predictions = {}
    for suffix, indexes in (("all", list(range(12))), ("subset", [9, 1, 7])):
        frame = frames["validation"].iloc[indexes].copy()
        frame["label"] = "not an input"
        frame["experiment_id"] = "also not an input"
        export = make_export(tmp_path, compare, frame, suffix)
        old_calls = calls.copy()
        outputs = experiment.predict(export)
        assert calls == {key: value + 1 for key, value in old_calls.items()}
        assert set(outputs) == set(experiment.EXPORTS)
        assert len(list(export.output_dir.glob("*.csv"))) == 2
        for name in experiment.EXPORTS:
            path = export.output_dir / f"submission_{name}.csv"
            values = pd.read_csv(path).set_index("sample_id").label
            assert values.index.tolist() == frame.sample_id.iloc[::-1].tolist()
            assert values.between(1, 6).all() and values.dtype.kind in "iu"
            record = read_json(path.with_suffix(".manifest.json"))
            assert record["weights"]["inner_best_neural"] == experiment.RATIOS[name]
            if suffix == "all":
                predictions[name] = values
            else:
                pd.testing.assert_series_equal(values, predictions[name].loc[values.index])
        with pytest.raises(ValueError, match="Output directory exists"):
            experiment.predict(export)
    assert experiment.run_files(args.output_dir) == before
    assert sha256(original) == marker


def test_legacy_training_budget_revision_works_without_editing_old_artifacts(completed, tmp_path):
    args, frames, _, compare = completed
    # Emulate f639869's completed artifact format, using independent tiny weights.
    manifest, result = checkpoint.read_result(args.output_dir)
    manifest["source_sha256"][experiment.DRIVER] = (
        "60c51f4e9768a93e14f8df7c7cf7b6d4ee3fdc0ae69e8b7771b9723ed4bbc3ef"
    )
    for key in ("original_config", "budget_overrides"):
        manifest.pop(key)
    for stage in ("inner_best", "refit"):
        result["identities"][stage]["source_sha256"] = manifest["source_sha256"]
        path = args.output_dir / f"{stage}.pt"
        state = torch.load(path, weights_only=True)
        state["source_sha256"] = manifest["source_sha256"]
        torch.save(state, path)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "result.json", result)
    refresh_status(args.output_dir)
    before = experiment.run_files(args.output_dir)
    recipe = experiment.compare(compare)
    assert recipe["source_compatibility"] == "reviewed_training_budget_only_change"
    export = make_export(tmp_path, compare, frames["validation"], "legacy")
    experiment.predict(export)
    assert experiment.run_files(args.output_dir) == before


def refresh_status(directory):
    write_json(
        directory / "status.json",
        {
            "status": "complete",
            "files": {name: sha256(directory / name) for name in checkpoint.FILES},
        },
    )


def test_validation_replay_mismatch_cannot_export(completed, monkeypatch):
    _, _, _, compare = completed
    original = checkpoint.probability

    def changed(*a, **kw):
        return np.roll(original(*a, **kw), 1, axis=1)

    monkeypatch.setattr(checkpoint, "probability", changed)
    with pytest.raises(ValueError, match="Validation replay differs"):
        experiment.compare(compare)
    assert read_json(compare.output_dir / "status.json")["status"] == "failed"
    assert not (compare.output_dir / "recipe.json").exists()


def test_wrong_split_images_or_runtime_rejected_before_inference(completed, monkeypatch, tmp_path):
    args, _, _, compare = completed
    wrong = copy.copy(compare)
    wrong.input = args.data_dir / "train" / "train.csv"
    with pytest.raises(ValueError, match="Validation CSV"):
        experiment.compare(wrong)
    wrong = copy.copy(compare)
    wrong.image_root = args.data_dir / "train"
    # Alter the content behind one matching filename in the wrong split.
    (wrong.image_root / "invented_0.png").write_bytes(b"independent invalid image")
    with pytest.raises(ValueError, match="Validation images differ"):
        experiment.compare(wrong)
    monkeypatch.setattr(checkpoint, "runtime_signature", lambda device: {"wrong": True})
    with pytest.raises(ValueError, match="runtime environment"):
        experiment.compare(compare)
    assert not compare.output_dir.exists()


def test_modified_models_recipe_and_template_are_rejected(completed, tmp_path):
    args, frames, _, compare = completed
    experiment.compare(compare)
    export = make_export(tmp_path, compare, frames["validation"], "tampering")
    pd.DataFrame({"sample_id": ["wrong"], "label": [1]}).to_csv(
        export.sample_submission, index=False
    )
    with pytest.raises(ValueError, match="IDs"):
        experiment.predict(export)
    assert not export.output_dir.exists()
    recipe_path = compare.output_dir / "recipe.json"
    original = recipe_path.read_bytes()
    recipe_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="recipe changed"):
        experiment.predict(export)
    recipe_path.write_bytes(original)
    with (args.output_dir / "inner_best.pt").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="Source run files changed"):
        experiment.predict(export)
    assert not export.output_dir.exists()
