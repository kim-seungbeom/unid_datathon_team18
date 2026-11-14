"""Fixed ratio comparisons use independent synthetic fixtures, never competition data."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_midpoint_blend import (
    digest,
    dump,
    independent_probabilities,
    invented_sources,  # noqa: F401 -- pytest fixture shared by these synthetic tests
)

from experiments.seungbeom import blend_ratios as ratios
from experiments.seungbeom import midpoint_blend as midpoint


@pytest.fixture
def comparison(request, tmp_path):
    state = request.getfixturevalue("invented_sources")
    midpoint.validate(state["args"])
    parent = state["args"].output_dir / "recipe.json"
    state["calls"].clear()
    state["policies"].clear()
    args = argparse.Namespace(
        recipe=parent,
        input=state["args"].input,
        image_root=state["args"].image_root,
        output_dir=tmp_path / "ratio_comparison",
        device="cpu",
    )
    return state, args


def prediction_args(state, comparison_args, tmp_path, frame=None, suffix="one"):
    frame = state["frame"] if frame is None else frame
    input_csv = tmp_path / f"synthetic_input_{suffix}.csv"
    template = tmp_path / f"synthetic_template_{suffix}.csv"
    frame.drop(columns="label").to_csv(input_csv, index=False)
    pd.DataFrame({"sample_id": frame["sample_id"].iloc[::-1], "label": 6}).to_csv(
        template, index=False
    )
    return argparse.Namespace(
        recipe=comparison_args.output_dir / "recipe.json",
        input=input_csv,
        image_root=comparison_args.image_root,
        sample_submission=template,
        output_dir=tmp_path / f"ratio_submissions_{suffix}",
        candidates=list(ratios.CANDIDATES),
        device="cpu",
    )


def test_neural_weight_direction_and_exact_midpoint_arithmetic():
    neural = np.array([[0.7, 0.3, 0, 0, 0, 0]], dtype=np.float32)
    tree = np.array([[0.3, 0.7, 0, 0, 0, 0]], dtype=np.float64)
    assert ratios.weighted_probabilities(neural, tree, "blend_75_25").argmax() == 0
    assert ratios.weighted_probabilities(neural, tree, "blend_25_75").argmax() == 1
    np.testing.assert_array_equal(
        ratios.weighted_probabilities(neural, tree, "blend_50_50"),
        midpoint.blend_probabilities(neural, tree),
    )
    for candidate in ratios.CANDIDATES:
        output = ratios.weighted_probabilities(neural, tree, candidate)
        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-7)


@pytest.mark.parametrize("problem", ["shape", "classes", "empty", "nan", "negative", "sum"])
def test_invalid_probabilities_are_rejected(problem):
    neural, tree = independent_probabilities()
    if problem == "shape":
        tree = tree[:-1]
    elif problem == "classes":
        neural, tree = neural[:, :-1], tree[:, :-1]
    elif problem == "empty":
        neural, tree = neural[:0], tree[:0]
    elif problem == "nan":
        neural[0, 0] = np.nan
    elif problem == "negative":
        tree[0, :2] = [-0.1, 0.94]
    else:
        tree *= 0.9
    with pytest.raises(ValueError):
        ratios.weighted_probabilities(neural, tree, "blend_75_25")


def test_unknown_weight_recipe_is_rejected():
    neural, tree = independent_probabilities()
    with pytest.raises(ValueError):
        ratios.weighted_probabilities(neural, tree, "blend_80_20")


def test_compare_reuses_members_once_and_keeps_parent_unchanged(comparison):
    state, args = comparison
    original_hash = digest(args.recipe)
    parent = json.loads(args.recipe.read_text())
    result = ratios.compare(args)
    saved = json.loads((args.output_dir / "recipe.json").read_text())
    assert saved == result
    assert saved["kind"] == "seungbeom_blend_ratios"
    assert saved["status"] == "complete"
    assert saved["format_version"] == 1
    assert saved["candidates"] == ratios.CANDIDATES
    assert saved["base_recipe"] == parent
    for name in midpoint.METHODS:
        assert saved["metrics"][name] == parent["metrics"][name]
    assert set(ratios.CANDIDATES).issubset(saved["metrics"])
    assert state["calls"] == [state["frame"]["sample_id"].tolist()]
    assert state["policies"] == [parent["image_precision"]]
    assert digest(args.recipe) == original_hash
    assert (args.output_dir / "validation_metrics.json").is_file()


@pytest.mark.parametrize("member", ["neural", "extra_trees", "blend_50_50"])
def test_compare_requires_all_parent_scores_to_replay(comparison, member):
    state, args = comparison
    parent = json.loads(args.recipe.read_text())
    parent["metrics"][member]["macro_f1"] = 0.0
    dump(args.recipe, parent)
    with pytest.raises(ValueError):
        ratios.compare(args)
    assert not (args.output_dir / "recipe.json").exists()


def test_compare_rejects_confusion_drift_even_with_unchanged_macro_f1(comparison):
    _, args = comparison
    parent = json.loads(args.recipe.read_text())
    parent["metrics"]["blend_50_50"]["confusion_matrix"][0][0] += 1
    dump(args.recipe, parent)
    with pytest.raises(ValueError):
        ratios.compare(args)
    assert not (args.output_dir / "recipe.json").exists()


@pytest.mark.parametrize(
    "problem",
    ["old_recipe", "runtime", "source", "model", "train", "different_validation", "output"],
)
def test_compare_guards_fail_before_inference(comparison, problem):
    state, args = comparison
    parent = json.loads(args.recipe.read_text())
    if problem == "old_recipe":
        parent["format_version"] = 1
    elif problem == "runtime":
        parent["runtime_signature"]["packages"]["torch"] = "invented_incompatible_version"
    elif problem == "source":
        name = next(iter(parent["source_sha256"]))
        parent["source_sha256"][name] = "different"
    elif problem == "model":
        state["checkpoints"][0].write_bytes(b"changed independent synthetic checkpoint")
    elif problem == "train":
        args.input = state["train_csv"]
    elif problem == "different_validation":
        args.input.write_text(args.input.read_text() + "\n")
    else:
        args.output_dir.mkdir()
        (args.output_dir / "keep.txt").write_text("preserve")
    dump(args.recipe, parent)
    with pytest.raises(ValueError):
        ratios.compare(args)
    assert not state["calls"]
    assert not (args.output_dir / "recipe.json").exists()
    if problem == "output":
        assert (args.output_dir / "keep.txt").read_text() == "preserve"


def test_prediction_preserves_labels_template_ids_and_subset_values(comparison, tmp_path):
    state, args = comparison
    ratios.compare(args)
    state["calls"].clear()
    neural, tree = independent_probabilities()
    for index, frame in enumerate((state["frame"], state["frame"].iloc[[5, 1, 0]])):
        prediction = prediction_args(state, args, tmp_path, frame, suffix=str(index))
        ratios.predict(prediction)
        for candidate, neural_weight in ratios.CANDIDATES.items():
            path = prediction.output_dir / f"submission_{candidate}.csv"
            output = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
            expected = (neural_weight * neural + (1 - neural_weight) * tree).argmax(axis=1) + 1
            expected = pd.Series(expected, index=state["frame"]["sample_id"])
            ordered = frame["sample_id"].iloc[::-1].tolist()
            assert list(output.columns) == ["sample_id", "label"]
            assert output["sample_id"].tolist() == ordered
            assert output["label"].tolist() == expected.loc[ordered].tolist()
            assert pd.api.types.is_integer_dtype(output["label"])
            assert output["label"].between(1, 6).all()
        status = json.loads((prediction.output_dir / "status.json").read_text())
        assert status["status"] == "complete"
    assert len(state["calls"]) == 2  # Three candidates share each full image/member inference.


def test_midpoint_csv_is_byte_identical_to_original_export(comparison, tmp_path):
    state, args = comparison
    ratios.compare(args)
    prediction = prediction_args(state, args, tmp_path)
    ratios.predict(prediction)
    original = argparse.Namespace(
        recipe=args.recipe,
        input=prediction.input,
        image_root=prediction.image_root,
        sample_submission=prediction.sample_submission,
        output=tmp_path / "original_midpoint.csv",
        device="cpu",
    )
    midpoint.predict(original)
    assert original.output.read_bytes() == (
        prediction.output_dir / "submission_blend_50_50.csv"
    ).read_bytes()


@pytest.mark.parametrize(
    "problem",
    ["status", "source", "candidates", "model", "runtime", "template", "output", "selection"],
)
def test_prediction_guards_fail_before_inference(comparison, tmp_path, problem):
    state, args = comparison
    ratios.compare(args)
    state["calls"].clear()
    prediction = prediction_args(state, args, tmp_path)
    recipe = json.loads(prediction.recipe.read_text())
    if problem == "status":
        recipe["status"] = "running"
    elif problem == "source":
        name = next(iter(recipe["source_sha256"]))
        recipe["source_sha256"][name] = "different"
    elif problem == "candidates":
        recipe["candidates"]["blend_75_25"] = 0.8
    elif problem == "model":
        state["checkpoints"][0].write_bytes(b"changed independent synthetic checkpoint")
    elif problem == "runtime":
        recipe["base_recipe"]["runtime_signature"]["packages"]["torch"] = "incompatible"
    elif problem == "template":
        pd.DataFrame({"sample_id": ["NA", "NA"], "label": [1, 1]}).to_csv(
            prediction.sample_submission, index=False
        )
    elif problem == "selection":
        prediction.candidates = ["blend_80_20"]
    else:
        prediction.output_dir.mkdir()
        (prediction.output_dir / "keep.csv").write_text("preserve existing submission")
    dump(prediction.recipe, recipe)
    with pytest.raises(ValueError):
        ratios.predict(prediction)
    assert not state["calls"]
    assert not (prediction.output_dir / "submission_blend_75_25.csv").exists()
    if problem == "output":
        assert (prediction.output_dir / "keep.csv").read_text() == "preserve existing submission"


def test_prediction_does_not_open_validation_csv_or_cache(comparison, tmp_path, monkeypatch):
    state, args = comparison
    parent = json.loads(args.recipe.read_text())
    cache = tmp_path / "synthetic_reference_cache"
    cache.write_bytes(b"independent synthetic cache")
    parent["image_precision"] = "cudnn_tf32"
    parent["precision_audit"] = {
        "image_precision": "cudnn_tf32",
        "reference_files": {str(cache): digest(cache)},
    }
    dump(args.recipe, parent)
    ratios.compare(args)
    cache.unlink()
    args.input.unlink()
    state["policies"].clear()
    monkeypatch.setattr(
        midpoint, "audit_cache", lambda *a, **k: pytest.fail("Read Validation cache")
    )
    prediction = prediction_args(state, args, tmp_path)
    ratios.predict(prediction)
    assert state["policies"] == ["cudnn_tf32"]
    assert (prediction.output_dir / "submission_blend_75_25.csv").is_file()


def test_report_only_reads_saved_metrics(comparison, monkeypatch, capsys):
    state, args = comparison
    ratios.compare(args)
    capsys.readouterr()
    monkeypatch.setattr(midpoint, "load_sources", lambda *a, **k: pytest.fail("Loaded models"))
    monkeypatch.setattr(
        midpoint, "method_probabilities", lambda *a, **k: pytest.fail("Ran model inference")
    )
    ratios.report(argparse.Namespace(recipe=args.output_dir / "recipe.json"))
    output = capsys.readouterr().out
    for candidate in ratios.CANDIDATES:
        assert candidate in output
