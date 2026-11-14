"""Export orchestration checks with synthetic manifests and fake inference writers."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.seungbeom import export_submissions as E
from experiments.seungbeom.neural_config import normalize_config
from schema import RAW_SENSOR_COLUMNS


def dump(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source_runs(tmp_path):
    neural, tree = tmp_path / "neural", tmp_path / "tree"
    neural.mkdir()
    tree.mkdir()
    config = normalize_config({"fusion": "concat"})
    split = {}
    for name in ("train", "validation"):
        synthetic_csv = tmp_path / f"synthetic_{name}.csv"
        synthetic_csv.write_text(f"sample_id,label\n{name}_fixture,1\n")
        split[f"{name}_csv_sha256"] = E.sha256(synthetic_csv)
    for run in (neural, tree):
        dump(run / "status.json", {"status": "complete"})
    dump(neural / "config.json", config)
    dump(neural / "identity.json", {"config": config, **split})
    dump(neural / "best_metrics.json", {"macro_f1": 0.6, "epoch": 2})
    (neural / "best.pt").write_bytes(b"independent fake neural checkpoint, never deserialized")
    dump(tree / "split.json", split)
    dump(
        tree / "metrics.json",
        {"validation": {method: {"macro_f1": 0.5} for method in E.TREE_METHODS}},
    )
    (tree / "ensemble.joblib").write_bytes(b"independent fake tree checkpoint")
    inputs = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "sample_id": ["NA", "00001", "third"],
            "image_path": ["unused_a.png", "unused_b.png", "unused_c.png"],
            **{name: [1.0, 2.0, 3.0] for name in RAW_SENSOR_COLUMNS},
        }
    ).to_csv(inputs, index=False)
    template = tmp_path / "template.csv"
    pd.DataFrame({"sample_id": ["third", "NA", "00001"], "label": [1, 1, 1]}).to_csv(
        template, index=False
    )
    return argparse.Namespace(
        neural_run=neural,
        ensemble_run=tree,
        input=inputs,
        image_root=tmp_path,
        sample_submission=template,
        output_dir=tmp_path / "export",
        device="cpu",
    )


def fake_neural(command, cwd, check):
    assert cwd == E.ROOT and check is True
    assert command[2] == "predict" and "--config" not in command
    args = {command[i]: command[i + 1] for i in range(3, len(command), 2)}
    template = pd.read_csv(args["--sample_submission"], dtype=str, keep_default_na=False)
    template["label"] = [3, 1, 6]
    template.to_csv(args["--output"], index=False)
    return SimpleNamespace(returncode=0)


def fake_trees(**kwargs):
    assert kwargs["methods"] == E.TREE_METHODS
    assert kwargs["device"] == "cpu"
    template = pd.read_csv(kwargs["template_path"], dtype=str, keep_default_na=False)
    values = ([3, 1, 6], [6, 5, 4], [3, 2, 6], [1, 2, 3])
    for method, labels in zip(kwargs["methods"], values, strict=True):
        frame = template.copy()
        frame["label"] = labels
        frame.to_csv(kwargs["output_dir"] / f"submission_{method}.csv", index=False)
    return []


def test_export_five_valid_csvs_and_provenance_with_exact_ids(source_runs, capsys):
    args = source_runs
    candidates = E.export(args, neural_runner=fake_neural, tree_runner=fake_trees)
    assert len(candidates) == 5
    assert [item["method"] for item in candidates] == ["neural_concat", *E.TREE_METHODS]
    assert candidates[1]["identical_to"] == "submission_neural_concat.csv"
    assert all(item["identical_to"] is None for item in candidates[2:])
    for item in candidates:
        path = args.output_dir / item["filename"]
        frame = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
        assert frame["sample_id"].tolist() == ["third", "NA", "00001"]
        assert frame["label"].dtype.kind in "iu"
        assert item["rows"] == 3 and item["sha256"] == E.sha256(path)
    manifest = E.read_json(args.output_dir / "manifest.json")
    assert manifest["candidates"] == candidates
    assert manifest["input_sha256"] == E.sha256(args.input)
    assert manifest["template_sha256"] == E.sha256(args.sample_submission)
    assert manifest["selection_basis"] == "saved_validation_metrics_before_test_inference"
    assert manifest["source_sha256"]["experiments/seungbeom/export_submissions.py"]
    assert E.read_json(args.output_dir / "export_status.json") == {"status": "complete", "files": 5}
    assert "format\tidentical_to" in capsys.readouterr().out


@pytest.mark.parametrize(
    "problem",
    [
        "incomplete",
        "split",
        "config",
        "template_ids",
        "template_columns",
        "missing_checkpoint",
        "existing_output",
        "nonfinite_score",
    ],
)
def test_bad_preflight_never_calls_inference(source_runs, problem):
    args = source_runs
    if problem == "incomplete":
        dump(args.neural_run / "status.json", {"status": "running"})
    elif problem == "split":
        split = E.read_json(args.ensemble_run / "split.json")
        split["validation_csv_sha256"] = "another-synthetic-split"
        dump(args.ensemble_run / "split.json", split)
    elif problem == "config":
        config = E.read_json(args.neural_run / "config.json")
        config["seed"] = 44
        dump(args.neural_run / "config.json", config)
    elif problem == "template_ids":
        pd.DataFrame({"sample_id": ["NA", "NA", "third"], "label": [1, 1, 1]}).to_csv(
            args.sample_submission, index=False
        )
    elif problem == "template_columns":
        pd.DataFrame({"label": [1], "sample_id": ["NA"]}).to_csv(
            args.sample_submission, index=False
        )
    elif problem == "missing_checkpoint":
        (args.neural_run / "best.pt").unlink()
    elif problem == "existing_output":
        args.output_dir.mkdir()
        (args.output_dir / "keep.txt").write_text("keep")
    else:
        dump(args.neural_run / "best_metrics.json", {"macro_f1": float("nan"), "epoch": 2})

    def fail(*args, **kwargs):
        pytest.fail("Inference must not start")

    with pytest.raises((ValueError, FileNotFoundError)):
        E.export(args, neural_runner=fail, tree_runner=fail)
    if problem == "existing_output":
        assert (args.output_dir / "keep.txt").read_text() == "keep"
    else:
        assert not args.output_dir.exists()


@pytest.mark.parametrize("problem", ["reordered", "float_labels", "invalid_label", "missing_file"])
def test_bad_csv_marks_failed_and_never_completes_manifest(source_runs, problem):
    args = source_runs

    def malformed_tree(**kwargs):
        fake_trees(**kwargs)
        path = kwargs["output_dir"] / "submission_xgboost.csv"
        if problem == "missing_file":
            path.unlink()
            return
        frame = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
        if problem == "reordered":
            frame = frame.iloc[::-1]
        elif problem == "float_labels":
            frame["label"] = frame["label"].astype(float)
        else:
            frame.loc[0, "label"] = 0
        frame.to_csv(path, index=False)

    with pytest.raises((ValueError, FileNotFoundError)):
        E.export(args, neural_runner=fake_neural, tree_runner=malformed_tree)
    assert E.read_json(args.output_dir / "export_status.json")["status"] == "failed"
    assert not (args.output_dir / "manifest.json").exists()


def test_source_checkpoint_change_during_inference_marks_failed(source_runs):
    args = source_runs

    def changing_tree(**kwargs):
        fake_trees(**kwargs)
        kwargs["checkpoint"].write_bytes(b"changed fake model")

    with pytest.raises(ValueError, match="checkpoint changed"):
        E.export(args, neural_runner=fake_neural, tree_runner=changing_tree)
    assert E.read_json(args.output_dir / "export_status.json")["status"] == "failed"
    assert not (args.output_dir / "manifest.json").exists()


def test_inference_exception_keeps_failed_status(source_runs):
    def failed_neural(*args, **kwargs):
        raise RuntimeError("Synthetic inference failure")

    with pytest.raises(RuntimeError, match="Synthetic"):
        E.export(source_runs, neural_runner=failed_neural, tree_runner=fake_trees)
    assert E.read_json(source_runs.output_dir / "export_status.json") == {
        "status": "failed",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize("source", ["input", "sample_submission"])
def test_input_csv_change_during_export_marks_failed(source_runs, source):
    def changing_tree(**kwargs):
        fake_trees(**kwargs)
        path = getattr(source_runs, source)
        path.write_text(path.read_text() + "\n")

    with pytest.raises(ValueError, match="CSV changed"):
        E.export(source_runs, neural_runner=fake_neural, tree_runner=changing_tree)
    assert E.read_json(source_runs.output_dir / "export_status.json")["status"] == "failed"
    assert not (source_runs.output_dir / "manifest.json").exists()
