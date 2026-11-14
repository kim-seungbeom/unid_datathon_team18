"""Independent CPU caches only; new commands must never load models or infer."""

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
from test_seungbeom_inner_best_refine import prepared  # noqa: F401
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401

from experiments.seungbeom import inner_best_cached_ratios as experiment
from experiments.seungbeom.export_submissions import read_json, sha256


def test_ratios_preserve_30_70_and_depend_only_on_the_current_row():
    ast.parse(Path(experiment.__file__).read_text(), feature_version=(3, 10))
    neural = np.array([[0.95, 0.05, 0, 0, 0, 0], [0.1, 0.9, 0, 0, 0, 0]], dtype=np.float32)
    tree = np.array([[0.36, 0.64, 0, 0, 0, 0], [0.7, 0.3, 0, 0, 0, 0]], dtype=np.float64)
    actual = experiment.candidates(neural, tree)
    assert actual["inner_best_25_75"][0].argmax() == 0
    assert actual["inner_best_20_80"][0].argmax() == 1
    previous = experiment.previous.candidates(neural, tree)
    for name in experiment.REPLAY:
        np.testing.assert_array_equal(actual[name], previous[name])
    for name, weight in experiment.RATIOS.items():
        np.testing.assert_array_equal(
            actual[name], weight * neural.astype(float) + (1 - weight) * tree
        )
        for positions in ([1, 0], [1]):
            np.testing.assert_array_equal(
                experiment.candidates(neural[positions], tree[positions])[name],
                actual[name][positions],
            )


@pytest.fixture
def cached_case(prepared, monkeypatch, tmp_path):  # noqa: F811
    training, frames, marker, earlier = prepared
    experiment.previous.compare(earlier)
    test_frame = frames["validation"].drop(columns="label").copy()
    test_frame["sample_id"] = [f"invented_test_{i}" for i in range(len(test_frame))]
    export = make_export(tmp_path, earlier, test_frame, "earlier_test")
    export.cache_dir = tmp_path / "test_probability_cache"
    experiment.previous.predict(export)

    def forbidden(*a, **kw):
        raise AssertionError("Cache-only commands must not load models, fit, or infer")

    original = experiment.previous.original
    for name in ("load_members", "infer_members"):
        monkeypatch.setattr(original, name, forbidden)
    for name in ("restore", "probability"):
        monkeypatch.setattr(original.checkpoint, name, forbidden)
    monkeypatch.setattr(experiment.previous, "member_probabilities", forbidden)
    compare = argparse.Namespace(
        baseline_recipe=earlier.output_dir / "recipe.json",
        input=earlier.input,
        image_root=earlier.image_root,
        cache_dir=earlier.cache_dir,
        output_dir=tmp_path / "lower_ratios",
        device="cpu",
    )
    return training, marker, compare, export


def test_cached_compare_and_two_exports_preserve_prior_artifacts_and_do_not_read_validation(
    cached_case, tmp_path, capsys
):
    training, marker, compare, earlier_export = cached_case
    old_csv = earlier_export.output_dir / "submission_inner_best_30_70.csv"
    original_files = experiment.previous.original.file_hashes(
        [old_csv, marker, compare.baseline_recipe]
    )
    original_files.update(experiment.previous.original.run_files(training.output_dir))
    for cache in (compare.cache_dir, earlier_export.cache_dir):
        original_files.update(
            experiment.previous.original.file_hashes(
                [cache / "status.json", cache / "metadata.json", cache / "probabilities.npz"]
            )
        )
    recipe = experiment.compare(compare)
    assert recipe["replay_matches"] == dict.fromkeys(experiment.REPLAY, True)
    assert recipe["changed_rows"][experiment.BASELINE] == 0
    experiment.report(
        argparse.Namespace(recipe=compare.output_dir / "recipe.json", class_details=True)
    )
    assert "delta_vs_30_70" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Output directory exists"):
        experiment.compare(compare)
    experiment.previous.check_files(original_files)

    # Keep model artifacts and comparison recipes, but remove Train/Validation inputs/cache.
    for split in ("train", "validation"):
        (training.data_dir / split / f"{split}.csv").unlink()
    shutil.rmtree(compare.cache_dir)
    export = copy.copy(earlier_export)
    export.recipe = compare.output_dir / "recipe.json"
    export.output_dir = tmp_path / "lower_exports"
    outputs = experiment.predict(export)
    assert set(outputs) == set(experiment.RATIOS)
    status = read_json(export.output_dir / "status.json")
    assert status["status"] == "complete"
    assert len(list(export.output_dir.glob("*.csv"))) == 2
    frame = pd.read_csv(export.input)
    context = read_json(export.cache_dir / "metadata.json")
    members, _ = experiment.previous.read_cache(export.cache_dir, context, frame.sample_id.tolist())
    probabilities = experiment.candidates(*members)
    for name in experiment.RATIOS:
        output = export.output_dir / f"submission_{name}.csv"
        result = pd.read_csv(output)
        assert list(result.columns) == ["sample_id", "label"]
        assert result.sample_id.tolist() == frame.sample_id.iloc[::-1].tolist()
        np.testing.assert_array_equal(result.label, (probabilities[name].argmax(1) + 1)[::-1])
        assert result.label.between(1, 6).all() and result.label.dtype.kind in "iu"
        assert status["changed_rows_vs_30_70"][name] == experiment.changes(probabilities)[name]
    repeat = copy.copy(export)
    repeat.output_dir = tmp_path / "repeated_exports"
    experiment.predict(repeat)
    for name in experiment.RATIOS:
        assert sha256(export.output_dir / f"submission_{name}.csv") == sha256(
            repeat.output_dir / f"submission_{name}.csv"
        )
    with pytest.raises(ValueError, match="Output directory exists"):
        experiment.predict(export)
    # All remaining original artifacts are unchanged (Validation cache was deliberately removed).
    experiment.previous.check_files(
        {
            path: digest
            for path, digest in original_files.items()
            if not Path(path).is_relative_to(compare.cache_dir)
        }
    )


def test_missing_wrong_or_corrupt_cache_fails_without_inference(cached_case, tmp_path):
    training, _, compare, test_export = cached_case
    wrong = copy.copy(compare)
    wrong.cache_dir = tmp_path / "missing"
    with pytest.raises(ValueError, match="Completed probability cache required"):
        experiment.compare(wrong)
    wrong.cache_dir = test_export.cache_dir
    with pytest.raises(ValueError, match="Cache inputs/models/runtime differ"):
        experiment.compare(wrong)
    wrong.input = training.data_dir / "train" / "train.csv"
    with pytest.raises(ValueError, match="Validation CSV"):
        experiment.compare(wrong)
    with (compare.cache_dir / "probabilities.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Source run files changed"):
        experiment.compare(compare)
    assert not compare.output_dir.exists()


def test_export_rejects_other_input_template_and_changed_recipe(cached_case, tmp_path):
    _, _, compare, earlier = cached_case
    experiment.compare(compare)
    export = copy.copy(earlier)
    export.recipe = compare.output_dir / "recipe.json"
    export.output_dir = tmp_path / "rejected_exports"
    subset = pd.read_csv(export.input).iloc[[1, 5]]
    subset.to_csv(export.input, index=False)
    with pytest.raises(ValueError, match="IDs"):
        experiment.predict(export)
    pd.DataFrame({"sample_id": subset.sample_id, "label": 1}).to_csv(
        export.sample_submission, index=False
    )
    with pytest.raises(ValueError, match="Cache inputs/models/runtime differ"):
        experiment.predict(export)
    with export.recipe.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="recipe changed"):
        experiment.predict(export)
    assert not export.output_dir.exists()


def test_replay_failure_blocks_new_recipe(cached_case, monkeypatch):
    _, _, compare, _ = cached_case
    original = experiment.candidates

    def wrong(*members):
        values = original(*members)
        values[experiment.BASELINE] = np.roll(values[experiment.BASELINE], 1, axis=1)
        return values

    monkeypatch.setattr(experiment, "candidates", wrong)
    with pytest.raises(ValueError, match="do not replay"):
        experiment.compare(compare)
    assert read_json(compare.output_dir / "status.json")["status"] == "failed"
    assert not (compare.output_dir / "recipe.json").exists()
