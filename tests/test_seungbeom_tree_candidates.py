"""Candidate export checks with independent synthetic inputs and saved sklearn models."""

import hashlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")

from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom import tree_candidates as candidates  # noqa: E402


@pytest.fixture
def saved(tmp_path, monkeypatch):
    ids = ["00001", "NA", "last"]
    frame = pd.DataFrame(
        {
            "sample_id": ids,
            "image_path": [f"synthetic/{name}.png" for name in ids],
            **{name: [1.0, 2.0, 3.0] for name in mm.RAW_SENSOR_COLUMNS},
        }
    )
    input_csv = tmp_path / "synthetic.csv"
    frame.to_csv(input_csv, index=False)
    template = tmp_path / "template.csv"
    pd.DataFrame({"sample_id": ids[::-1], "label": [6, 6, 6]}).to_csv(template, index=False)
    labels = np.tile(np.arange(1, 7), 2)
    features = pd.DataFrame({"sensor_meas_v": labels.astype(float), "img_0000": labels * 0.5})
    pipelines = []
    for constant in range(3):
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    mm.EncodedClassifier(
                        DummyClassifier(strategy="constant", constant=constant)
                        if constant < 2
                        else DecisionTreeClassifier(max_depth=4, random_state=42)
                    ),
                ),
            ]
        )
        pipeline.fit(features, labels)
        pipelines.append(pipeline)
    bundle = {
        "kind": "seungbeom_classwise_multimodal",
        "format_version": 1,
        "label_values": list(range(1, 7)),
        "models": ["xgboost", "lightgbm", "extra_trees"],
        "pipelines": pipelines,
        "weights": np.asarray([[0.0] * 6, [1.0] * 6, [0.0] * 6]),
        "sensor_spec": mm.sensor_spec(),
        "columns": list(features.columns),
        "encoder": "resnet18",
        "embedding": {
            "model": "resnet18",
            "weights": "SYNTHETIC_TEST_RANDOM_WEIGHTS",
            "dimension": 512,
            "l2_normalize": False,
            "preprocessing": {"fixture": "independent synthetic"},
            "trained": False,
        },
        "refit_train_val": False,
        "weight_fit_split": "train_group_oof_only",
    }
    checkpoint = tmp_path / "synthetic_bundle.joblib"
    joblib.dump(bundle, checkpoint)
    calls = []

    def make_features(current, csv_path, image_root, spec, **kwargs):
        del csv_path, image_root, spec
        assert "embeddings_dir" not in kwargs
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        assert kwargs["device"] == "cpu"
        calls.append(current["sample_id"].tolist())
        values = current["meas_v"].to_numpy(dtype=float)
        return pd.DataFrame({"sensor_meas_v": values, "img_0000": values * 0.5}), bundle[
            "embedding"
        ]

    monkeypatch.setattr(mm, "make_features", make_features)
    # Fixture fitting is complete; prediction must not trigger any subsequent fit.
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **k: pytest.fail("Inference attempted fitting"))
    return {
        "frame": frame,
        "checkpoint": checkpoint,
        "input_csv": input_csv,
        "image_root": tmp_path / "independent_synthetic_images",
        "template_path": template,
        "output_dir": tmp_path / "candidate_csvs",
        "device": "cpu",
        "calls": calls,
    }


def args(saved, **changes):
    return {
        key: value for key, value in {**saved, **changes}.items() if key not in ("calls", "frame")
    }


def read(path):
    return pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)


def test_default_four_candidates_reuse_one_live_extraction_and_fixed_methods(saved):
    outputs = candidates.predict_candidates(**args(saved))
    assert saved["calls"] == [["00001", "NA", "last"]]
    assert [row["method"] for row in outputs] == list(candidates.DEFAULT_METHODS)
    expected_labels = {
        "classwise_ensemble": [2, 2, 2],
        "extra_trees": [3, 2, 1],
        "equal_average": [1, 2, 1],
        "xgboost": [1, 1, 1],
    }
    for metadata in outputs:
        path = saved["output_dir"] / metadata["filename"]
        table = read(path)
        assert list(table.columns) == ["sample_id", "label"]
        assert table["sample_id"].tolist() == ["last", "NA", "00001"]
        assert table["label"].tolist() == expected_labels[metadata["method"]]
        assert pd.api.types.is_integer_dtype(table["label"])
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert metadata["rows"] == 3


def test_repeated_reordered_and_subset_input_preserve_predictions(saved, tmp_path):
    reference = candidates.predict_candidates(**args(saved))
    for index, frame in enumerate(
        (saved["frame"], saved["frame"].iloc[::-1], saved["frame"].iloc[[1]])
    ):
        input_csv, template = tmp_path / f"input_{index}.csv", tmp_path / f"template_{index}.csv"
        frame.to_csv(input_csv, index=False)
        pd.DataFrame({"sample_id": frame["sample_id"], "label": 4}).to_csv(template, index=False)
        output_dir = tmp_path / f"iteration_{index}"
        result = candidates.predict_candidates(
            **args(saved, input_csv=input_csv, template_path=template, output_dir=output_dir)
        )
        for first, later in zip(reference, result, strict=True):
            expected = read(saved["output_dir"] / first["filename"]).set_index("sample_id")
            actual = read(output_dir / later["filename"]).set_index("sample_id")
            pd.testing.assert_frame_equal(expected.loc[actual.index], actual)


@pytest.mark.parametrize(
    "methods", [[], ["../escape"], ["xgboost", "xgboost"], "xgboost", ["logistic"]]
)
def test_invalid_or_absent_methods_fail_before_image_extraction(saved, methods):
    with pytest.raises(ValueError):
        candidates.predict_candidates(**args(saved, methods=methods))
    assert saved["calls"] == []
    assert not saved["output_dir"].exists()


@pytest.mark.parametrize(
    "template",
    [
        {"sample_id": ["NA", "NA", "last"], "label": [1, 1, 1]},
        {"sample_id": ["00001", "NA", "other"], "label": [1, 1, 1]},
        {"sample_id": ["NA"], "label": [1]},
        {"label": [1], "sample_id": ["NA"]},
    ],
)
def test_invalid_template_fails_before_image_extraction(saved, template):
    pd.DataFrame(template).to_csv(saved["template_path"], index=False)
    with pytest.raises(ValueError, match="template"):
        candidates.predict_candidates(**args(saved))
    assert saved["calls"] == []


def test_existing_output_is_preserved_before_any_inference(saved):
    saved["output_dir"].mkdir()
    existing = saved["output_dir"] / "submission_equal_average.csv"
    existing.write_text("existing result must remain untouched\n")
    with pytest.raises(ValueError, match="already exists"):
        candidates.predict_candidates(**args(saved))
    assert existing.read_text() == "existing result must remain untouched\n"
    assert saved["calls"] == []


def test_cli_accepts_one_member_with_explicit_template(saved, capsys):
    candidates.main(
        [
            "--checkpoint",
            str(saved["checkpoint"]),
            "--input",
            str(saved["input_csv"]),
            "--image_root",
            str(saved["image_root"]),
            "--sample_submission",
            str(saved["template_path"]),
            "--output_dir",
            str(saved["output_dir"]),
            "--methods",
            "lightgbm",
            "--device",
            "cpu",
        ]
    )
    table = read(saved["output_dir"] / "submission_lightgbm.csv")
    assert table["label"].tolist() == [2, 2, 2]
    assert '"method": "lightgbm"' in capsys.readouterr().out
