"""Export five saved multimodal candidates on the organizer server, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom.neural_config import load_config  # noqa: E402
from schema import read_frame, validate_labels  # noqa: E402

TREE_METHODS = ("classwise_ensemble", "extra_trees", "equal_average", "xgboost")
SOURCE_FILES = (
    "schema.py",
    "preprocessing.py",
    "sensor_features.py",
    "temporal.py",
    "dataset.py",
    "runtime.py",
    "image_features/encoders.py",
    "image_features/extract_features.py",
    "experiments/seungbeom/neural_config.py",
    "experiments/seungbeom/neural_data.py",
    "experiments/seungbeom/neural_sensors.py",
    "experiments/seungbeom/neural_model.py",
    "experiments/seungbeom/neural_experiment.py",
    "experiments/seungbeom/classwise_ensemble.py",
    "experiments/seungbeom/ensemble_core.py",
    "experiments/seungbeom/multimodal.py",
    "experiments/seungbeom/tree_candidates.py",
    "experiments/seungbeom/export_submissions.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_score(metrics):
    score = metrics["macro_f1"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid saved Validation Macro F1.")
    return float(score)


def submission_ids(input_csv, template_path):
    frame = read_frame(input_csv)
    template = pd.read_csv(template_path, dtype=str, keep_default_na=False)
    if list(template.columns) != ["sample_id", "label"]:
        raise ValueError("Template columns must be sample_id,label in that order.")
    ids = template["sample_id"]
    if (
        len(ids) != len(frame)
        or not ids.is_unique
        or ids.str.strip().eq("").any()
        or set(ids) != set(frame["sample_id"])
    ):
        raise ValueError("Template IDs must match the input exactly, with no duplicates.")
    return ids.tolist()


def check_csv(path, expected_ids):
    frame = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
    if list(frame.columns) != ["sample_id", "label"]:
        raise ValueError("Output columns must be sample_id,label in that order.")
    if frame["sample_id"].tolist() != expected_ids:
        raise ValueError("Output IDs/order differ from the submission template.")
    validate_labels(frame["label"])
    if not pd.api.types.is_integer_dtype(frame["label"]):
        raise ValueError("Output label tokens must be integers, not floats.")
    return {"filename": Path(path).name, "sha256": sha256(path), "rows": len(frame)}


def preflight(args):
    if args.output_dir.exists():
        raise ValueError("Output directory already exists. Choose a new output_dir.")
    for run in (args.neural_run, args.ensemble_run):
        if read_json(run / "status.json").get("status") != "complete":
            raise ValueError("Both source runs must be complete before exporting.")
    config = load_config(args.neural_run / "config.json")
    identity = read_json(args.neural_run / "identity.json")
    if identity.get("config") != config:
        raise ValueError("Neural config and saved run identity disagree.")
    split = read_json(args.ensemble_run / "split.json")
    for name in ("train_csv_sha256", "validation_csv_sha256"):
        if not identity.get(name) or identity[name] != split.get(name):
            raise ValueError("Candidate runs must use identical Train and Validation CSVs.")
    neural_metrics = read_json(args.neural_run / "best_metrics.json")
    tree_metrics = read_json(args.ensemble_run / "metrics.json")["validation"]
    candidates = [
        {
            "method": f"neural_{config['fusion']}",
            "run_dir": str(args.neural_run.resolve()),
            "checkpoint": str((args.neural_run / "best.pt").resolve()),
            "validation_macro_f1": checked_score(neural_metrics),
            "best_epoch": neural_metrics["epoch"],
            "config": config,
            "image_source": "live_input_images",
        }
    ]
    candidates.extend(
        {
            "method": method,
            "run_dir": str(args.ensemble_run.resolve()),
            "checkpoint": str((args.ensemble_run / "ensemble.joblib").resolve()),
            "validation_macro_f1": checked_score(tree_metrics[method]),
            "image_source": "live_input_images",
        }
        for method in TREE_METHODS
    )
    for candidate in candidates:
        candidate["checkpoint_sha256"] = sha256(candidate["checkpoint"])
        candidate["filename"] = f"submission_{candidate['method']}.csv"
    expected_ids = submission_ids(args.input, args.sample_submission)
    return (
        candidates,
        expected_ids,
        {name: split[name] for name in ("train_csv_sha256", "validation_csv_sha256")},
    )


def export(args, *, neural_runner=None, tree_runner=None):
    candidates, ids, split = preflight(args)
    input_hashes = {
        "input_sha256": sha256(args.input),
        "template_sha256": sha256(args.sample_submission),
    }
    if neural_runner is None:
        neural_runner = subprocess.run
    if tree_runner is None:
        from experiments.seungbeom.tree_candidates import predict_candidates

        tree_runner = predict_candidates
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "export_status.json"
    write_json(status_path, {"status": "running"})
    try:
        command = [
            sys.executable,
            str(ROOT / "experiments/seungbeom/neural_experiment.py"),
            "predict",
            "--input",
            str(args.input.resolve()),
            "--image_root",
            str(args.image_root.resolve()),
            "--sample_submission",
            str(args.sample_submission.resolve()),
            "--checkpoint",
            candidates[0]["checkpoint"],
            "--output",
            str((args.output_dir / candidates[0]["filename"]).resolve()),
            "--device",
            args.device,
        ]
        print("Exporting saved neural best checkpoint...", flush=True)
        neural_runner(command, cwd=ROOT, check=True)
        check_csv(args.output_dir / candidates[0]["filename"], ids)
        print("Exporting four tree candidates with one live image extraction...", flush=True)
        tree_runner(
            checkpoint=args.ensemble_run / "ensemble.joblib",
            input_csv=args.input,
            image_root=args.image_root,
            template_path=args.sample_submission,
            output_dir=args.output_dir,
            methods=TREE_METHODS,
            device=args.device,
        )
        seen = {}
        for candidate in candidates:
            candidate.update(check_csv(args.output_dir / candidate["filename"], ids))
            candidate["identical_to"] = seen.get(candidate["sha256"])
            seen.setdefault(candidate["sha256"], candidate["filename"])
            if sha256(candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
                raise ValueError("A source checkpoint changed during export.")
        if (
            sha256(args.input) != input_hashes["input_sha256"]
            or sha256(args.sample_submission) != input_hashes["template_sha256"]
        ):
            raise ValueError("Input or template CSV changed during export.")
        write_json(
            args.output_dir / "manifest.json",
            {
                "format_version": 1,
                "selection_basis": "saved_validation_metrics_before_test_inference",
                **input_hashes,
                "split": split,
                "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
                "device": args.device,
                "candidates": candidates,
                "note": "CSV format checks do not measure Test accuracy or certify all rules.",
            },
        )
        write_json(status_path, {"status": "complete", "files": len(candidates)})
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error_type": type(error).__name__})
        raise
    print("filename\tvalidation_macro_f1\tformat\tidentical_to")
    for candidate in candidates:
        print(
            f"{candidate['filename']}\t{candidate['validation_macro_f1']:.6f}\tPASS\t"
            f"{candidate['identical_to'] or '-'}"
        )
    print("Export complete. CSV files are ready for manual upload from:", args.output_dir)
    return candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "neural_run",
        "ensemble_run",
        "input",
        "image_root",
        "sample_submission",
        "output_dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    export(parser.parse_args(argv))


if __name__ == "__main__":
    main()
