"""Fixed ResNet18/raw5 neural + ExtraTrees soft vote; user-run server inference only."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import classwise_ensemble as trees  # noqa: E402
from experiments.seungbeom import neural_experiment as neural  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.neural_config import model_config, normalize_config  # noqa: E402
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor  # noqa: E402
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame  # noqa: E402

RULE = {"method": "probability_mean", "neural": 0.5, "extra_trees": 0.5}
KIND = "seungbeom_midpoint_blend"
METHODS = ("neural", "extra_trees", "blend_50_50")
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *neural.SOURCE_FILES,
            *trees.SOURCE_FILES,
            "experiments/seungbeom/ensemble_core.py",
            "experiments/seungbeom/tree_candidates.py",
            "experiments/seungbeom/export_submissions.py",
            "experiments/seungbeom/midpoint_blend.py",
        )
    )
)


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def check_files(files):
    if not files or any(sha256(Path(path)) != digest for path, digest in files.items()):
        raise ValueError("Source run files changed; repeat Validation in a new directory.")


def blend_probabilities(neural_prob, tree_prob):
    left, right = np.asarray(neural_prob), np.asarray(tree_prob)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Members must have matching [rows, 6] probability shapes.")
    values = validate_probabilities(np.stack([left, right], axis=1))
    return values.mean(axis=1)


def load_sources(neural_run, ensemble_run):
    """Load trusted team checkpoints; require matching Train/Validation and raw5 contracts."""
    import torch

    from experiments.seungbeom import multimodal as mm
    from experiments.seungbeom.neural_model import NeuralFusionClassifier

    neural_run, ensemble_run = Path(neural_run), Path(ensemble_run)
    files = {}
    for run, filenames in (
        (
            neural_run,
            ("status.json", "config.json", "identity.json", "best_metrics.json", "best.pt"),
        ),
        (ensemble_run, ("status.json", "split.json", "metrics.json", "ensemble.joblib")),
    ):
        if read_json(run / "status.json").get("status") != "complete":
            raise ValueError("Both source runs must be complete.")
        files.update({str((run / name).resolve()): sha256(run / name) for name in filenames})
    config = normalize_config(read_json(neural_run / "config.json"))
    identity = read_json(neural_run / "identity.json")
    split = read_json(ensemble_run / "split.json")
    if identity.get("config") != config:
        raise ValueError("Neural config and identity disagree.")
    for key in ("train_csv_sha256", "validation_csv_sha256"):
        if not identity.get(key) or identity[key] != split.get(key):
            raise ValueError("Members must use identical Train and Validation CSVs.")
    checkpoint = torch.load(neural_run / "best.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint.get("kind") != "seungbeom_neural_fusion"
        or checkpoint.get("format_version") != 1
        or checkpoint.get("label_values") != list(LABELS)
        or checkpoint.get("config") != config
        or checkpoint.get("model_config") != model_config(config)
    ):
        raise ValueError("Neural checkpoint/config/label contract differs.")
    if (
        config["encoder"] != "embedding"
        or config["embedding_encoder"] != "resnet18"
        or config["fusion"] != "concat"
        or config["sensor_features"] != list(RAW_SENSOR_COLUMNS)
    ):
        raise ValueError("This fixed comparison requires ResNet18/raw5/concat.")
    preprocessor = NeuralSensorPreprocessor.from_dict(checkpoint["preprocessor"])
    if (
        preprocessor.feature_names != RAW_SENSOR_COLUMNS
        or preprocessor.imputation != "median"
        or preprocessor.scaling != "standard"
        or preprocessor.uses_history
    ):
        raise ValueError("Neural preprocessing must be saved raw5/median/standard.")
    bundle = trees.load_bundle(ensemble_run / "ensemble.joblib")
    if (
        "extra_trees" not in bundle["models"]
        or bundle["encoder"] != "resnet18"
        or bundle["sensor_spec"] != mm.sensor_spec()
    ):
        raise ValueError("Tree bundle must include ResNet18/raw5/median ExtraTrees.")
    if embedding_contract(checkpoint["visual_contract"]) != mm.embedding_info(bundle["embedding"]):
        raise ValueError("Members have different image weights or preprocessing contracts.")
    model = NeuralFusionClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    provenance = {
        "files": files,
        "train_csv_sha256": split["train_csv_sha256"],
        "validation_csv_sha256": split["validation_csv_sha256"],
        "saved_macro_f1": {
            "neural": checked_score(read_json(neural_run / "best_metrics.json")),
            "extra_trees": checked_score(
                read_json(ensemble_run / "metrics.json")["validation"]["extra_trees"]
            ),
        },
    }
    check_files(files)
    return {
        "config": config,
        "checkpoint": checkpoint,
        "preprocessor": preprocessor,
        "model": model,
        "bundle": bundle,
    }, provenance


def method_probabilities(frame, image_root, loaded, device):
    """Extract current images once; apply each member's saved preprocessing separately."""
    import torch

    from experiments.seungbeom import multimodal as mm
    from runtime import seed_everything, select_device

    seed_everything(loaded["config"]["seed"])
    target = select_device(device)
    bundle = loaded["bundle"]
    # Labels/IDs are never classifier inputs. The extractor sees only current rows.
    features, meta = mm.make_features(
        frame,
        None,
        image_root,
        bundle["sensor_spec"],
        encoder=bundle["encoder"],
        embedding_spec=bundle["embedding"],
        device=device,
        batch_size=1,
        workers=0,
    )
    if list(features.columns) != bundle["columns"]:
        raise ValueError("Tree feature order differs from training.")
    if embedding_contract(meta) != embedding_contract(loaded["checkpoint"]["visual_contract"]):
        raise ValueError("Live image contract differs from the neural checkpoint.")
    image_columns = [f"{mm.EMBED_PREFIX}{i:04d}" for i in range(loaded["config"]["embedding_dim"])]
    vectors = features.loc[:, image_columns].to_numpy(dtype=np.float32, copy=True)
    sensors = loaded["preprocessor"].transform(frame)
    model = loaded["model"].to(target).eval()
    neural_prob = []
    print("Image features ready; predicting neural and ExtraTrees members...", flush=True)
    with torch.inference_mode():
        for row in range(len(frame)):
            logits = model(
                torch.from_numpy(vectors[row : row + 1]).to(target),
                torch.from_numpy(sensors[row : row + 1]).to(target),
            )
            neural_prob.append(torch.softmax(logits, dim=1).cpu().numpy())
    neural_prob = np.concatenate(neural_prob, axis=0)
    pipeline = bundle["pipelines"][bundle["models"].index("extra_trees")]
    tree_prob = mm.predict_proba_six_stable(pipeline, features)
    return {
        "neural": neural_prob,
        "extra_trees": tree_prob,
        "blend_50_50": blend_probabilities(neural_prob, tree_prob),
    }


def show_metrics(metrics):
    print("method\tvalidation_macro_f1\t" + "\t".join(f"f1_{i}" for i in LABELS))
    for name in METHODS:
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t"
            + "\t".join(f"{item['class_f1'][str(i)]:.6f}" for i in LABELS)
        )


def validate(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    loaded, provenance = load_sources(args.neural_run, args.ensemble_run)
    digest = sha256(args.input)
    if digest != provenance["validation_csv_sha256"] or digest == provenance["train_csv_sha256"]:
        raise ValueError("Validate requires the saved held-out Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    sources = source_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        probabilities = method_probabilities(frame, args.image_root, loaded, args.device)
        metrics = {
            name: classification_metrics(frame["label"], probabilities[name].argmax(axis=1) + 1)
            for name in METHODS
        }
        show_metrics(metrics)
        replay = {
            name: math.isclose(
                metrics[name]["macro_f1"],
                provenance["saved_macro_f1"][name],
                rel_tol=0,
                abs_tol=1e-6,
            )
            for name in ("neural", "extra_trees")
        }
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        if not all(replay.values()):
            raise ValueError(
                "Member Validation replay differs from saved scores; inspect before Test inference."
            )
        check_files(provenance["files"])
        if sha256(args.input) != digest or source_hashes() != sources:
            raise ValueError("Validation input or source code changed during evaluation.")
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "rule": RULE,
            "label_values": list(LABELS),
            "neural_run": str(args.neural_run.resolve()),
            "ensemble_run": str(args.ensemble_run.resolve()),
            "provenance": provenance,
            "source_sha256": sources,
            "metrics": metrics,
            "weight_selection": "fixed_equal_before_validation_no_search",
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(status, {"status": "complete"})
        delta = metrics["blend_50_50"]["macro_f1"] - metrics["neural"]["macro_f1"]
        print(f"Fixed 50:50 blend minus neural Validation Macro F1: {delta:+.6f}")
        print("Recipe saved. No fitting or Test access; keep the original neural candidate.")
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def predict(args):
    sidecar = args.output.with_suffix(".manifest.json")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Output or manifest exists; choose a new output filename.")
    recipe = read_json(args.recipe)
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or recipe.get("rule") != RULE
        or recipe.get("label_values") != list(LABELS)
    ):
        raise ValueError("Use a completed fixed 50:50 Validation recipe.")
    if recipe["source_sha256"] != source_hashes():
        raise ValueError("Code changed after Validation; validate again before predicting.")
    check_files(recipe["provenance"]["files"])
    loaded, provenance = load_sources(Path(recipe["neural_run"]), Path(recipe["ensemble_run"]))
    if provenance != recipe["provenance"]:
        raise ValueError("Source runs differ from the validated recipe.")
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids)
    input_hashes = {
        "input": sha256(args.input),
        "template": sha256(args.sample_submission),
        "recipe": sha256(args.recipe),
    }
    probabilities = method_probabilities(frame, args.image_root, loaded, args.device)
    labels = (
        blend_probabilities(probabilities["neural"], probabilities["extra_trees"]).argmax(axis=1)
        + 1
    )
    values = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
    check_files(provenance["files"])
    if (
        input_hashes
        != {
            "input": sha256(args.input),
            "template": sha256(args.sample_submission),
            "recipe": sha256(args.recipe),
        }
        or recipe["source_sha256"] != source_hashes()
    ):
        raise ValueError("Input, recipe, or source code changed during inference.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
    metadata = check_csv(args.output, ordered)
    write_json(
        sidecar,
        {
            "kind": KIND,
            "rule": RULE,
            "input_sha256": input_hashes,
            "recipe": recipe,
            "output": metadata,
        },
    )
    print("Fixed 50:50 submission CSV saved:", args.output)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Replay saved members and compare one fixed blend")
    for name in ("neural_run", "ensemble_run", "input", "image_root", "output_dir"):
        check.add_argument("--" + name, type=Path, required=True)
    check.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    check.set_defaults(handler=validate)
    export = commands.add_parser("predict", help="Apply the previously validated fixed recipe")
    for name in ("recipe", "input", "image_root", "sample_submission", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    export.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
