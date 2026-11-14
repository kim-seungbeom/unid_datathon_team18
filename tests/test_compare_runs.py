"""Invented metric files only; never read existing experiment directories."""

import json

import pytest

from evaluate import classification_metrics
from scripts.compare_runs import class_detail_rows, comparison_rows


def test_comparison_checks_identical_splits(tmp_path):
    directories = [tmp_path / "invented_raw", tmp_path / "invented_derived"]
    for directory in directories:
        directory.mkdir()
        files = {
            "config.json": {"mode": "sensor_only", "sensor_features": ["meas_v"], "seed": 42},
            "split.json": {
                "train_csv_sha256": "invented_train",
                "validation_csv_sha256": "invented_val",
            },
            "best_metrics.json": {
                "epoch": 1,
                "macro_f1": 0.25,
                "class_f1": {str(k): 0.25 for k in range(1, 7)},
            },
        }
        for name, content in files.items():
            (directory / name).write_text(json.dumps(content))
    rows = comparison_rows(directories)
    assert len(rows) == 2 and rows[0]["macro_f1"] == "0.250000"
    assert rows[0]["loss"] == "cross_entropy" and rows[0]["sampling"] == "shuffle"
    (directories[1] / "split.json").write_text(
        json.dumps(
            {
                "train_csv_sha256": "invented_train",
                "validation_csv_sha256": "different_validation",
            }
        )
    )
    with pytest.raises(ValueError, match="splits differ"):
        comparison_rows(directories)


def test_precision_recall_and_legacy_confusion_details(tmp_path):
    metrics = classification_metrics([1, 1, 1, 2], [1, 2, 2, 2])
    assert metrics["class_precision"]["1"] == 1
    assert metrics["class_recall"]["1"] == pytest.approx(1 / 3)
    assert metrics["class_precision"]["2"] == pytest.approx(1 / 3)
    assert metrics["class_recall"]["2"] == 1
    assert metrics["class_precision"]["3"] == metrics["class_recall"]["3"] == 0
    # Legacy files contain the matrix but no saved precision/recall dictionaries.
    (tmp_path / "best_metrics.json").write_text(
        json.dumps({"confusion_matrix": metrics["confusion_matrix"]})
    )
    rows = class_detail_rows([tmp_path])
    assert rows[0]["precision"] == "1.000000" and rows[0]["recall"] == "0.333333"
    assert rows[0]["f1"] == "0.500000" and rows[0]["support"] == 3
    assert rows[1]["precision"] == "0.333333" and rows[1]["recall"] == "1.000000"
    assert rows[2]["precision"] == rows[2]["recall"] == rows[2]["f1"] == "0.000000"


@pytest.mark.parametrize("matrix", [[[1]], [[-1] * 6] * 6, [[0.5] * 6] * 6])
def test_invalid_class_details_are_rejected(tmp_path, matrix):
    (tmp_path / "best_metrics.json").write_text(json.dumps({"confusion_matrix": matrix}))
    with pytest.raises(ValueError, match="confusion matrix"):
        class_detail_rows([tmp_path])
