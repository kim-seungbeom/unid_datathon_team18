"""Refit the original ExtraTrees on authorized Train+Validation; export separately.

The frozen ResNet18/raw5 architecture and hyperparameters stay fixed. All learned
imputation/scaling/PCA/forest state is refitted. Test is inference-only. There is
no held-out Validation score after this refit, and no neural member is retrained.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import blend_ratios  # noqa: E402
from experiments.seungbeom import midpoint_blend as midpoint  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame, validate_split  # noqa: E402

KIND = "seungbeom_extratrees_train_validation_refit"
SELF = "experiments/seungbeom/extratrees_refit.py"
FIT_SPLITS = ["train", "validation"]
CACHE_FILES = ("features.npy", "sample_ids.csv", "metadata.json")


def source_hashes():
    return {**blend_ratios.source_hashes(), SELF: sha256(ROOT / SELF)}


def check_unchanged(files, sources, runtime, device):
    midpoint.check_files(files)
    if source_hashes() != sources or midpoint.runtime_signature(device) != runtime:
        raise ValueError("Source code or runtime changed; use the recorded environment and code.")


def load_training_inputs(args):
    base = read_json(args.recipe)
    inputs = {"train": args.train_csv, "validation": args.validation_csv}
    caches = {"train": args.train_embeddings, "validation": args.validation_embeddings}
    files = {str(args.recipe.resolve()): sha256(args.recipe), **base["provenance"]["files"]}
    # Reject swapped CSVs, Test, or modified rows before parsing or fitting.
    for split, path in inputs.items():
        digest = sha256(path)
        if digest != base["provenance"][f"{split}_csv_sha256"]:
            raise ValueError(f"Use the exact original {split} CSV recorded by the reference.")
        files[str(path.resolve())] = digest
    reference = blend_ratios.load_base(base, args.device)
    if reference["config"]["seed"] != 42:
        raise ValueError("Use the original seed42 midpoint recipe.")
    for split, path in caches.items():
        digests = {name: sha256(path / name) for name in CACHE_FILES}
        if digests != reference["identity"]["input_provenance"][split]:
            raise ValueError(f"Use the original frozen {split} embedding cache.")
        files.update({str((path / name).resolve()): value for name, value in digests.items()})
    frames = {split: read_frame(path, labeled=True) for split, path in inputs.items()}
    validate_split(frames["train"], frames["validation"])
    bundle = reference["bundle"]
    if bundle["sensor_spec"] != mm.sensor_spec() or reference["config"]["sensor_features"] != list(
        RAW_SENSOR_COLUMNS
    ):
        raise ValueError("This refit keeps the original raw-five sensor contract.")
    features = []
    for split in FIT_SPLITS:
        values, meta = mm.make_features(
            frames[split].drop(columns="label"),
            None,
            None,
            bundle["sensor_spec"],
            embeddings_dir=caches[split],
            encoder="resnet18",
            embedding_spec=bundle["embedding"],
        )
        if list(values.columns) != bundle["columns"] or embedding_contract(
            meta
        ) != embedding_contract(bundle["embedding"]):
            raise ValueError("Image/sensor feature contract differs from the original ExtraTrees.")
        features.append(values)
    combined = pd.concat(features, ignore_index=True)
    labels = np.concatenate([frames[split]["label"].to_numpy() for split in FIT_SPLITS])
    spec = {
        "columns": bundle["columns"],
        "sensor_spec": bundle["sensor_spec"],
        "embedding": bundle["embedding"],
        "image_precision": base["image_precision"],
    }
    template = bundle["pipelines"][bundle["models"].index("extra_trees")]
    counts = {
        split: {"rows": len(frame), "experiments": int(frame["experiment_id"].nunique())}
        for split, frame in frames.items()
    }
    return combined, labels, template, spec, files, base["runtime_signature"], counts


def train(args):
    import joblib
    from sklearn.base import clone

    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    features, labels, template, spec, files, runtime, counts = load_training_inputs(args)
    sources = source_hashes()
    check_unchanged(files, sources, runtime, args.device)
    # clone retains only constructor parameters, not fitted preprocessing/trees.
    model = clone(template)
    parameters = joblib.hash(clone(template))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        print(
            "Fitting fresh ExtraTrees + imputation/scaling/PCA on Train + Validation...", flush=True
        )
        started = time.perf_counter()
        model.fit(features, labels)
        seconds = time.perf_counter() - started
        if joblib.hash(clone(model)) != parameters:
            raise ValueError("Model hyperparameters changed during refit.")
        model_path = args.output_dir / "model.joblib"
        temporary = args.output_dir / "model.joblib.tmp"
        joblib.dump(model, temporary)
        temporary.replace(model_path)
        check_unchanged(files, sources, runtime, args.device)
        metadata = {
            "kind": KIND,
            "format_version": 1,
            "refit_train_val": True,
            "fit_splits": FIT_SPLITS,
            "test_used_for_fit": False,
            "label_values": list(LABELS),
            "model_file": "model.joblib",
            "model_sha256": sha256(model_path),
            "parameter_fingerprint": parameters,
            "features": spec,
            "source_sha256": sources,
            "runtime": runtime,
            "training_input_sha256": files,
            "fit_counts": counts,
            "fit_rows": len(features),
            "training_seconds": seconds,
            "evaluation": {
                "held_out": False,
                "validation_macro_f1": None,
                "reason": "Official Validation was included in training.",
            },
        }
        write_json(args.output_dir / "metadata.json", metadata)
        write_json(
            status,
            {"status": "complete", "metadata_sha256": sha256(args.output_dir / "metadata.json")},
        )
        print(f"Refit complete in {seconds:.1f}s. Model saved: {model_path}")
        print("No held-out Validation score: Validation was used for fitting. Test was not read.")
        return metadata
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def load_model(directory, device):
    """Self-contained inference: never follow training_input_sha256 paths."""
    import joblib
    from sklearn.base import clone

    from runtime import seed_everything

    directory = directory.resolve()
    status = read_json(directory / "status.json")
    if status.get("status") != "complete" or sha256(directory / "metadata.json") != status.get(
        "metadata_sha256"
    ):
        raise ValueError("Use a complete, unchanged Train+Validation refit.")
    metadata = read_json(directory / "metadata.json")
    if (
        metadata.get("kind") != KIND
        or metadata.get("format_version") != 1
        or metadata.get("refit_train_val") is not True
        or metadata.get("fit_splits") != FIT_SPLITS
        or metadata.get("test_used_for_fit") is not False
        or metadata.get("label_values") != list(LABELS)
        or metadata.get("model_file") != "model.joblib"
        or metadata.get("evaluation", {}).get("held_out") is not False
        or metadata["evaluation"].get("validation_macro_f1") is not None
    ):
        raise ValueError("Unsupported refit metadata.")
    files = {
        str(directory / name): sha256(directory / name) for name in ("status.json", "metadata.json")
    }
    model_path = directory / "model.joblib"
    files[str(model_path)] = metadata["model_sha256"]
    seed_everything(42)
    check_unchanged(files, metadata["source_sha256"], metadata["runtime"], device)
    # Verify model bytes before trusted local pickle deserialization.
    model = joblib.load(model_path)
    if (
        list(model.feature_names_in_) != metadata["features"]["columns"]
        or joblib.hash(clone(model)) != metadata["parameter_fingerprint"]
    ):
        raise ValueError("Refit model features or parameters differ from its metadata.")
    return model, metadata, files


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    model, metadata, files = load_model(args.model_dir, args.device)
    files[str(args.input.resolve())] = sha256(args.input)
    if args.sample_submission:
        files[str(args.sample_submission.resolve())] = sha256(args.sample_submission)
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids) if args.sample_submission else ids
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        spec = metadata["features"]
        with midpoint.precision_context(spec["image_precision"]):
            features, meta = mm.make_features(
                frame.drop(columns="label", errors="ignore"),
                None,
                args.image_root,
                spec["sensor_spec"],
                encoder="resnet18",
                embedding_spec=spec["embedding"],
                device=args.device,
                batch_size=1,
                workers=0,
            )
        if list(features.columns) != spec["columns"] or embedding_contract(
            meta
        ) != embedding_contract(spec["embedding"]):
            raise ValueError("Live features differ from the refit image/sensor contract.")
        print("Image features ready; predicting refitted ExtraTrees...", flush=True)
        probabilities = mm.predict_proba_six_stable(model, features)
        labels = (
            pd.Series(probabilities.argmax(axis=1) + 1, index=ids)
            .loc[ordered]
            .to_numpy(dtype=np.int64)
        )
        check_unchanged(files, metadata["source_sha256"], metadata["runtime"], args.device)
        output = args.output_dir / "submission_extratrees_trainval.csv"
        with output.open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame({"sample_id": ordered, "label": labels}).to_csv(handle, index=False)
        result = check_csv(output, ordered)
        write_json(
            output.with_suffix(".manifest.json"),
            {
                "kind": KIND,
                "model_sha256": metadata["model_sha256"],
                "source_sha256": metadata["source_sha256"],
                "runtime": metadata["runtime"],
                "input_sha256": files,
                "fit_splits": FIT_SPLITS,
                "output": result,
            },
        )
        write_json(status, {"status": "complete", "output": result})
        print("Submission CSV saved:", output)
        return result
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in (
        "recipe",
        "train_csv",
        "validation_csv",
        "train_embeddings",
        "validation_embeddings",
    ):
        fit.add_argument("--" + name, type=Path, required=True)
    fit.set_defaults(handler=train)
    export = commands.add_parser("predict")
    for name in ("model_dir", "input", "image_root"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--sample_submission", type=Path)
    export.set_defaults(handler=predict)
    for command in (fit, export):
        command.add_argument("--output_dir", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
