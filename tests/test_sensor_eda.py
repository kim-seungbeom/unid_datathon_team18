"""Independent synthetic distributions only; no existing data or report files."""

import json

import numpy as np
import pandas as pd
import pytest

from schema import RAW_SENSOR_COLUMNS
from scripts.sensor_eda import generate_report, summarize
from sensor_features import TEMPERATURE_SENSOR_FEATURES


def invented_frame():
    return pd.DataFrame(
        {
            "meas_v": ["", "bad", np.inf, 3],
            "meas_a": [2] * 4,
            "meas_t1": [4, 8, 12, 16],
            "meas_t2": [8, 12, 16, 20],
            "meas_p": [np.nan] * 4,
        }
    )


def test_missing_and_constant_distributions():
    summary, histograms, features = summarize(invented_frame(), TEMPERATURE_SENSOR_FEATURES, 10)
    assert summary.loc["meas_v", "source_missing_count"] == 1
    assert summary.loc["meas_v", "parse_error_count"] == 1
    assert summary.loc["meas_v", "infinite_count"] == 1
    assert summary.loc["meas_v", "unavailable_count"] == 3
    assert summary.loc["meas_a", "unique_valid_count"] == 1
    assert histograms["meas_p"]["counts"] == []
    assert sum(histograms["meas_a"]["counts"]) == 4
    assert features["temp_diff"].tolist() == [4] * 4


@pytest.mark.parametrize("plots", [False, True])
def test_reports_are_local_and_do_not_read_labels(tmp_path, plots):
    if plots:
        pytest.importorskip("matplotlib")  # Optional requirements-eda.txt dependency.
    root = tmp_path / "invented"
    (root / "train").mkdir(parents=True)
    frame = invented_frame().assign(label="DO_NOT_EXPORT", sample_id="DO_NOT_EXPORT")
    frame.to_csv(root / "train/train.csv", index=False)
    output = tmp_path / "report"
    generate_report(root, "train", "temperature", output, bins=10, plots=plots)
    assert set(pd.read_csv(output / "summary.csv")["feature"]) == set(TEMPERATURE_SENSOR_FEATURES)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["imputation_or_scaling_applied"] is False
    report = (output / "report.html").read_text()
    assert "DO_NOT_EXPORT" not in report
    assert "https://" not in report and "<script" not in report
    if plots:
        from PIL import Image

        for name in TEMPERATURE_SENSOR_FEATURES:
            with Image.open(output / f"{name}.png") as image:
                assert image.width > 500 and image.height > 200
    with pytest.raises(FileExistsError):
        generate_report(root, "train", "raw", output)


def test_test_split_and_bad_bins_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Test"):
        generate_report(tmp_path, "test", "raw", tmp_path / "report")
    with pytest.raises(ValueError, match="bins"):
        summarize(invented_frame(), RAW_SENSOR_COLUMNS, 0)
