"""Grouped reporting on independent synthetic CPU inputs, never competition material."""

import argparse

import numpy as np
import pandas as pd
import pytest
from test_seungbeom_frozen_tree_blend import case as frozen_case  # noqa: F401
from test_seungbeom_frozen_tree_blend import forbid_fit
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401
from test_seungbeom_seed_blend import synthetic_case  # noqa: F401

from experiments.seungbeom import experiment_diagnostics as diag
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def invented_frame():
    return pd.DataFrame(
        {
            "sample_id": [f"fictional_{i}" for i in range(8)],
            "experiment_id": ["0001"] * 6 + ["NA"] * 2,
            "label": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    )


def test_population_quantifies_unequal_sizes_and_class_concentration():
    summary, groups, classes = diag.population(invented_frame(), "train")
    assert summary["rows"] == 8 and summary["experiments"] == 2
    assert summary["min_rows"] == 2 and summary["max_rows"] == 6
    assert summary["max_to_min_rows"] == 3
    assert summary["largest_10pct_groups_row_share"] == 0.75
    assert summary["effective_groups_by_row_share"] == 1.6
    assert groups["experiment_id"].tolist() == ["0001", "NA"]
    assert groups["classes_present"].tolist() == [2, 1]
    assert classes["experiments_present"].tolist() == [1, 2, 0, 0, 0, 0]
    assert classes.iloc[0]["largest_group_class_share"] == 1
    assert pd.isna(classes.iloc[2]["largest_group_class_share"])


def test_errors_preserve_totals_distinguish_absent_classes_and_error_rankings():
    frame = invented_frame()
    predicted = np.array([1, 2, 1, 1, 1, 1, 1, 1])
    summary, groups, classes, confusion = diag.grouped_errors(frame, predicted, "invented")
    assert summary["errors"] == 5
    assert groups["errors"].tolist() == [3, 2]
    assert groups["error_rate"].tolist() == [0.5, 1.0]
    assert groups["error_share"].tolist() == [0.6, 0.4]
    assert groups["macro_f1_present"].iloc[0] > groups["macro_f1_6"].iloc[0]
    assert groups["errors"].sum() == classes["errors"].sum() == 5
    assert confusion["count"].sum() == len(frame)
    assert confusion.loc[confusion.true_class != confusion.predicted_class, "count"].sum() == 5
    absent = classes[(classes.experiment_id == "NA") & (classes["class"] == 1)].iloc[0]
    assert absent["support"] == 0 and absent["predicted"] == 2
    assert pd.isna(absent["recall"]) and absent["f1"] == 0
    missing = classes[classes["class"] == 6]
    assert missing["f1"].isna().all()
    assert not {"sample_id", "image_path", "label"} & set(confusion.columns)
    order = [7, 1, 3, 6, 0, 2, 4, 5]
    reordered = diag.grouped_errors(frame.iloc[order], predicted[order], "invented")
    assert reordered[0] == summary
    for left, right in zip((groups, classes, confusion), reordered[1:], strict=True):
        pd.testing.assert_frame_equal(left, right)


def test_perfect_predictions_produce_zero_error_share_without_nan():
    frame = invented_frame()
    summary, groups, _, _ = diag.grouped_errors(frame, frame["label"], "invented")
    assert summary["errors"] == 0 and summary["top_error_groups_error_share"] == 0
    assert groups["error_share"].eq(0).all()
    assert groups["macro_f1_present"].eq(1).all()


@pytest.mark.parametrize("problem", ["empty", "missing", "blank", "null", "duplicate", "label"])
def test_invalid_metadata_is_rejected(problem):
    frame = invented_frame()
    if problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "missing":
        frame = frame.drop(columns="experiment_id")
    elif problem == "blank":
        frame.loc[0, "experiment_id"] = " "
    elif problem == "null":
        frame.loc[0, "experiment_id"] = None
    elif problem == "duplicate":
        frame.loc[0, "sample_id"] = frame.loc[1, "sample_id"]
    else:
        frame.loc[0, "label"] = 0
    with pytest.raises(ValueError):
        diag.population(frame, "train")


@pytest.fixture
def diagnosis(frozen_case, tmp_path):  # noqa: F811
    case = frozen_case
    diag.blend.compare(case["compare_args"])
    args = argparse.Namespace(
        recipe=case["compare_args"].output_dir / "recipe.json",
        train_csv=case["compare_args"].input.parent.parent / "train/train.csv",
        validation_csv=case["compare_args"].input,
        image_root=case["compare_args"].image_root,
        output_dir=tmp_path / "diagnosis",
        device="cpu",
    )
    case["calls"].clear()
    return case, args


def test_analyze_replays_existing_models_preserves_inputs_and_report_needs_no_inference(
    diagnosis, monkeypatch, capsys
):
    case, args = diagnosis
    forbid_fit(monkeypatch)
    reference = diag.blend.read_comparison(args.recipe)
    protected = {
        **reference["frozen"]["files"],
        **reference["base_recipe"]["provenance"]["files"],
        str(args.recipe): sha256(args.recipe),
        str(args.train_csv): sha256(args.train_csv),
        str(args.validation_csv): sha256(args.validation_csv),
    }
    summary = diag.analyze(args)
    assert summary["replay_matches"] == dict.fromkeys(diag.blend.METHODS, True)
    for name in diag.METHODS:
        assert summary["metrics"][name] == reference["metrics"][name]
    assert summary["population"]["train"]["rows"] == len(case["frame"])
    assert case["calls"] == [case["frame"]["sample_id"].tolist()]
    assert protected == {name: sha256(name) for name in protected}
    assert read_json(args.output_dir / "status.json")["status"] == "complete"
    for name in diag.TABLES:
        table = pd.read_csv(args.output_dir / name)
        assert not {"sample_id", "image_path", "label"} & set(table)
    assert not (args.output_dir / "recipe.json").exists()
    args.train_csv.unlink()
    args.validation_csv.unlink()
    args.recipe.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Saved report must not load models or predict")

    monkeypatch.setattr(diag.blend, "load_members", forbidden)
    monkeypatch.setattr(diag.blend, "infer_members", forbidden)
    for method in diag.METHODS:
        assert (
            diag.report(argparse.Namespace(output_dir=args.output_dir, method=method, top=2))
            == summary
        )
    text = capsys.readouterr().out
    assert "top3_error_share" in text and "overall Validation class metrics" in text
    assert "No training, Test access or submission export" in text


@pytest.mark.parametrize("problem", ["train_csv", "validation_csv", "output", "source"])
def test_preflight_rejects_changed_inputs_before_loading_models(diagnosis, monkeypatch, problem):
    _, args = diagnosis
    if problem.endswith("csv"):
        path = getattr(args, problem)
        path.write_text(path.read_text() + "\n")
    elif problem == "output":
        args.output_dir.mkdir()
    else:
        original = diag.blend.source_hashes
        monkeypatch.setattr(diag.blend, "source_hashes", lambda: {**original(), "changed": "yes"})

    def forbidden(*args, **kwargs):
        pytest.fail("Must reject before model loading")

    monkeypatch.setattr(diag.blend, "load_members", forbidden)
    with pytest.raises(ValueError):
        diag.analyze(args)


def test_replay_failure_blocks_error_report_even_when_macro_f1_matches(diagnosis, monkeypatch):
    _, args = diagnosis
    reference = read_json(args.recipe)
    # A different saved class confusion cannot be excused by an unchanged scalar F1.
    reference["metrics"]["frozen_tree_50_50"]["confusion_matrix"][0][0] += 1
    write_json(args.recipe, reference)
    write_json(
        args.recipe.parent / "status.json",
        {
            "status": "complete",
            "recipe_sha256": sha256(args.recipe),
        },
    )
    forbid_fit(monkeypatch)
    with pytest.raises(ValueError, match="Saved Validation metrics differ"):
        diag.analyze(args)
    assert (
        read_json(args.output_dir / "replay.json")["replay_matches"]["frozen_tree_50_50"] is False
    )
    assert read_json(args.output_dir / "status.json")["status"] == "failed"
    assert not (args.output_dir / "summary.json").exists()
    assert not list(args.output_dir.glob("*.csv"))


def test_report_detects_modified_output(diagnosis):
    _, args = diagnosis
    diag.analyze(args)
    path = args.output_dir / "experiment_counts.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="output changed"):
        diag.report(argparse.Namespace(output_dir=args.output_dir, method=diag.METHODS[0], top=2))


def test_wrong_prediction_count_and_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="row count"):
        diag.grouped_errors(invented_frame(), [1], "invented")
    with pytest.raises(ValueError, match="Labels"):
        diag.grouped_errors(invented_frame(), [0] * 8, "invented")
