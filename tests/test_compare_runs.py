"""Invented metric files only; never read existing experiment directories."""

import json

import pytest

from scripts.compare_runs import comparison_rows


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
