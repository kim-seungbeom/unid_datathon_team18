"""Paired frozen/layer4 ResNet18 experiment; real runs are human-executed on the server."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import neural_experiment as neural  # noqa: E402
from experiments.seungbeom.neural_config import load_config  # noqa: E402
from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor  # noqa: E402
from experiments.seungbeom.replay_precision import runtime_signature  # noqa: E402
from experiments.seungbeom.resnet_finetune_core import (  # noqa: E402
    MODES,
    WEIGHTS_SOURCE,
    FineTuneDataset,
    FineTuneModel,
    make_optimizer,
    probabilities,
)
from runtime import seed_everything, select_device  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame, validate_labels, validate_split  # noqa: E402

KIND = "seungbeom_resnet18_layer4_comparison"
CHECKPOINT_KIND = "seungbeom_resnet18_layer4_model"
DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 20,
    "patience": 5,
    "batch_size": 32,
    "num_workers": 0,
    "head_lr": 1e-4,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "sensor_hidden": 32,
    "fusion_hidden": 64,
    "dropout": 0.1,
}
IMAGE_CONTRACT = {
    "weights": WEIGHTS_SOURCE,
    "preprocessing": "image_features.encoders.preprocess:full_frame_letterbox224_imagenet",
    "augmentation": None,
    "l2_normalize": False,
    "batchnorm_running_statistics": "fixed_in_both_modes",
    "precision": "float32_tf32_disabled",
    "evaluation_batch_size": 1,
}
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *neural.SOURCE_FILES,
            "experiments/seungbeom/export_submissions.py",
            "experiments/seungbeom/replay_precision.py",
            "experiments/seungbeom/resnet_finetune_core.py",
            "experiments/seungbeom/resnet_finetune.py",
        )
    )
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    return {name: neural.file_sha(ROOT / name) for name in SOURCE_FILES}


def normalize_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CONFIG):
        raise ValueError("Unsupported fine-tuning config keys.")
    config = {**DEFAULT_CONFIG, **value}
    for key in ("epochs", "patience", "batch_size", "sensor_hidden", "fusion_hidden"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    if config["seed"] >= 2**32:
        raise ValueError("seed must be below 2**32.")
    for key in ("head_lr", "backbone_lr", "weight_decay", "dropout"):
        number = config[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError(f"Invalid {key}.")
        config[key] = float(number)
    if config["head_lr"] <= 0 or config["backbone_lr"] <= 0 or config["dropout"] >= 1:
        raise ValueError("Learning rates must be positive and dropout below one.")
    return config


def inner_split(frame, seed):
    """One prespecified Train-only group holdout, shared by the paired modes."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = validate_labels(frame["label"])
    if (
        "experiment_id" not in frame
        or frame["experiment_id"].isna().any()
        or frame["experiment_id"].astype(str).str.strip().eq("").any()
        or frame["sample_id"].isna().any()
        or not frame["sample_id"].is_unique
    ):
        raise ValueError("Train requires unique sample IDs and nonempty experiment IDs.")
    groups = frame["experiment_id"].to_numpy()
    if len(set(groups)) < 5 or set(labels) != set(LABELS):
        raise ValueError("Need at least five Train experiments and all six Train classes.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit, stop = next(splitter.split(frame, labels, groups))
    validate_split(frame.iloc[fit], frame.iloc[stop])
    if set(labels[fit]) != set(LABELS):
        raise ValueError("The prespecified inner training split is missing a class.")
    return {
        "fit": fit.tolist(),
        "stop": stop.tolist(),
        "support": {
            name: {str(label): int((labels[index] == label).sum()) for label in LABELS}
            for name, index in (("fit", fit), ("stop", stop))
        },
    }


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_kwargs(config):
    return {name: config[name] for name in ("sensor_hidden", "fusion_hidden", "dropout")}


def evaluation_loader(frame, image_root, prep, config, *, labeled=True):
    return DataLoader(
        FineTuneDataset(frame, image_root, prep, labeled=labeled),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )


def train_phase(frame, image_root, config, mode, device, *, epochs, stopping=None, progress=None):
    """Fit only frame. Optional stopping is exclusively an inner Train group holdout."""
    config = normalize_config(config)
    if type(epochs) is not int or not 1 <= epochs <= config["epochs"]:
        raise ValueError("Invalid phase epoch count.")
    seed_everything(config["seed"])
    prep = NeuralSensorPreprocessor.fit(
        frame, RAW_SENSOR_COLUMNS, imputation="median", scaling="standard"
    )
    dataset = FineTuneDataset(frame, image_root, prep)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model = FineTuneModel(mode, **model_kwargs(config), pretrained=True).to(device)
    initial = state_digest(model)
    optimizer = make_optimizer(
        model,
        head_lr=config["head_lr"],
        backbone_lr=config["backbone_lr"],
        weight_decay=config["weight_decay"],
    )
    stop_loader = None
    stopper = neural.EarlyStopping(config["patience"])
    if stopping is not None:
        stop_frame, stop_root = stopping
        validate_split(frame, stop_frame)
        stop_loader = evaluation_loader(stop_frame, stop_root, prep, config)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, seen = time.monotonic(), 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["visual"].to(device), batch["sensors"].to(device))
            loss = torch.nn.functional.cross_entropy(logits, batch["targets"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite training gradients.")
            optimizer.step()
            total += float(loss.detach()) * len(batch["targets"])
            seen += len(batch["targets"])
            if seen // 4096 != (seen - len(batch["targets"])) // 4096:
                print(f"{mode} epoch={epoch} training rows={seen}/{len(dataset)}", flush=True)
        row = {"epoch": epoch, "train_loss": total / len(dataset)}
        should_stop = False
        if stop_loader is not None:
            print(f"{mode} epoch={epoch}: evaluating inner Train holdout...", flush=True)
            prob = probabilities(model, stop_loader, device, progress_every=2048)
            metrics = classification_metrics(stop_frame["label"], prob.argmax(1) + 1)
            row["inner_macro_f1"] = metrics["macro_f1"]
            _, should_stop = stopper.update(metrics["macro_f1"], epoch)
        row["elapsed_seconds"] = time.monotonic() - started
        history.append(row)
        if progress is not None:
            progress(row)
        if should_stop:
            break
    return model, prep, stopper.best_epoch if stopping is not None else epochs, history, initial


def reference_snapshot(reference, train_csv, val_csv, config):
    reference = Path(reference).resolve()
    names = ("status.json", "config.json", "identity.json", "best_metrics.json", "best.pt")
    files = {str(reference / name): neural.file_sha(reference / name) for name in names}
    if read_json(reference / "status.json").get("status") != "complete":
        raise ValueError("Historical frozen neural run must be complete.")
    original = load_config(reference / "config.json")
    identity = read_json(reference / "identity.json")
    if (
        identity.get("config") != original
        or original["encoder"] != "embedding"
        or original["embedding_encoder"] != "resnet18"
        or original["fusion"] != "concat"
        or original["sensor_features"] != list(RAW_SENSOR_COLUMNS)
        or any(original[name] != config[name] for name in model_kwargs(config))
    ):
        raise ValueError("Reference must be frozen ResNet18/raw5/concat with the same head shape.")
    for name, path in (("train", train_csv), ("validation", val_csv)):
        if identity.get(name + "_csv_sha256") != neural.file_sha(path):
            raise ValueError("Use the identical Train/Validation CSVs as the historical run.")
    metrics = read_json(reference / "best_metrics.json")
    if not math.isfinite(metrics["macro_f1"]) or not 0 <= metrics["macro_f1"] <= 1:
        raise ValueError("Invalid historical Validation score.")
    return {"files": files, "metrics": metrics, "config": original}


def check_reference(snapshot):
    if any(neural.file_sha(path) != digest for path, digest in snapshot["files"].items()):
        raise ValueError("Historical artifacts changed; keep the original run intact.")


def child_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Artifact path escapes the experiment directory.")
    return path


def check_trial(directory):
    status = read_json(Path(directory) / "status.json")
    names = {"model.pt", "result.json", "selection_history.json", "refit_history.json"}
    if status.get("status") != "complete" or set(status.get("files", {})) != names:
        raise ValueError("Trial is incomplete.")
    if any(
        neural.file_sha(Path(directory) / name) != digest
        for name, digest in status["files"].items()
    ):
        raise ValueError("Trial artifacts changed.")
    return read_json(Path(directory) / "result.json")


def check_result_identity(result, manifest, mode):
    if (
        result.get("mode") != mode
        or result.get("config") != manifest["config"]
        or result.get("source_sha256") != manifest["source_sha256"]
        or result.get("runtime") != manifest["runtime"]
        or any(
            result.get(name + "_csv_sha256") != manifest["inputs"][name]["csv_sha256"]
            for name in ("train", "validation")
        )
    ):
        raise ValueError("Trial identity differs from the paired experiment.")


def fit_trial(directory, frames, roots, config, mode, device, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    neural.write_json(directory / "status.json", {"status": "running"})
    try:
        part = manifest["inner_split"]
        inner_fit, inner_stop = (frames[0].iloc[part[key]].copy() for key in ("fit", "stop"))

        def progress_for(phase):
            rows = []

            def progress(row):
                rows.append(row)
                neural.write_json(directory / f"{phase}_history.json", {"epochs": rows})
                score = (
                    f" inner_macro_f1={row['inner_macro_f1']:.6f}"
                    if "inner_macro_f1" in row
                    else ""
                )
                print(
                    f"{mode} {phase} epoch={row['epoch']} loss={row['train_loss']:.6f}{score}",
                    flush=True,
                )

            return progress

        print(f"{mode}: choosing epoch count using Train-only experiment groups...", flush=True)
        inner_model, _, selected, _, initial = train_phase(
            inner_fit,
            roots[0],
            config,
            mode,
            device,
            epochs=config["epochs"],
            stopping=(inner_stop, roots[0]),
            progress=progress_for("selection"),
        )
        del inner_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{mode}: fresh all-Train refit for {selected} epochs...", flush=True)
        model, prep, _, _, refit_initial = train_phase(
            frames[0],
            roots[0],
            config,
            mode,
            device,
            epochs=selected,
            progress=progress_for("refit"),
        )
        if initial != refit_initial:
            raise ValueError("Fresh refit did not start from the same initialization.")
        print(f"{mode}: scoring official Validation once with the refitted model...", flush=True)
        loader = evaluation_loader(frames[1], roots[1], prep, config)
        prob = probabilities(model, loader, device, progress_every=2048)
        metrics = classification_metrics(frames[1]["label"], prob.argmax(1) + 1)
        checkpoint = {
            "kind": CHECKPOINT_KIND,
            "format_version": 1,
            "config": config,
            "model_config": model.get_model_config(),
            "mode": mode,
            "selected_epoch": selected,
            "label_values": list(LABELS),
            "preprocessor": prep.to_dict(),
            "image_contract": IMAGE_CONTRACT,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "source_sha256": manifest["source_sha256"],
            "runtime": manifest["runtime"],
            "train_csv_sha256": manifest["inputs"]["train"]["csv_sha256"],
            "validation_csv_sha256": manifest["inputs"]["validation"]["csv_sha256"],
            "selection": "train_inner_experiment_holdout_then_fresh_all_train_refit",
        }
        torch.save(checkpoint, directory / "model.pt.tmp")
        (directory / "model.pt.tmp").replace(directory / "model.pt")
        # Verify a complete offline reconstruction against the saved scoring path.
        restored = FineTuneModel(mode, **model_kwargs(config), pretrained=False).to(device)
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        restored.load_state_dict(saved["state_dict"], strict=True)
        restored_prep = NeuralSensorPreprocessor.from_dict(saved["preprocessor"])
        replay = probabilities(
            restored,
            evaluation_loader(frames[1], roots[1], restored_prep, config),
            device,
            progress_every=2048,
        )
        if not np.array_equal(prob, replay):
            raise ValueError("Saved checkpoint did not exactly replay Validation probabilities.")
        result = {
            "mode": mode,
            "selected_epoch": selected,
            "metrics": metrics,
            "initial_state_sha256": initial,
            "config": config,
            "train_csv_sha256": checkpoint["train_csv_sha256"],
            "validation_csv_sha256": checkpoint["validation_csv_sha256"],
            "source_sha256": manifest["source_sha256"],
            "runtime": manifest["runtime"],
            "checkpoint_replay_matches": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
        neural.write_json(directory / "result.json", result)
        files = {
            name: neural.file_sha(directory / name)
            for name in ("model.pt", "result.json", "selection_history.json", "refit_history.json")
        }
        neural.write_json(directory / "status.json", {"status": "complete", "files": files})
        print(f"{mode} complete: validation_macro_f1={metrics['macro_f1']:.6f}", flush=True)
        return result
    except BaseException as error:
        neural.write_json(
            directory / "status.json", {"status": "failed", "error_type": type(error).__name__}
        )
        raise


def run(args):
    config = normalize_config(read_json(args.config) if args.config else {})
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise ValueError("Output exists. Use --resume or choose a new output_dir.")
    if output.is_relative_to(args.data_dir.resolve()):
        raise ValueError("Write experiment outputs outside the provided data directory.")
    device = select_device(args.device)
    seed_everything(config["seed"])
    paths = (args.data_dir / "train/train.csv", args.data_dir / "validation/validation.csv")
    frames = [read_frame(path, labeled=True) for path in paths]
    validate_split(*frames)
    reference = reference_snapshot(args.reference_run, *paths, config)
    print("Fingerprinting Train/Validation images; Test is not accessed...", flush=True)
    manifest = {
        "kind": KIND,
        "format_version": 1,
        "config": config,
        "source_sha256": source_hashes(),
        "runtime": runtime_signature(device),
        "image_contract": IMAGE_CONTRACT,
        "reference": reference,
        "inner_split": inner_split(frames[0], config["seed"]),
        "inputs": {
            name: {"csv_sha256": neural.file_sha(path), "images_sha256": neural.image_digest(path)}
            for name, path in zip(("train", "validation"), paths, strict=True)
        },
    }
    if output.exists():
        if read_json(output / "experiment.json") != manifest:
            raise ValueError("Config, source, environment, reference or inputs changed on resume.")
        state = read_json(output / "status.json")
        if state.get("manifest_sha256") != neural.file_sha(output / "experiment.json"):
            raise ValueError("Experiment manifest changed.")
    else:
        output.mkdir(parents=True)
        neural.write_json(output / "experiment.json", manifest)
        state = {
            "status": "running",
            "trials": {},
            "manifest_sha256": neural.file_sha(output / "experiment.json"),
        }
    state["status"] = "running"
    neural.write_json(output / "status.json", state)
    results = []
    try:
        for mode in MODES:
            relative = state["trials"].get(mode)
            directory = child_path(output, relative) if relative else None
            if (
                directory is not None
                and (directory / "status.json").is_file()
                and read_json(directory / "status.json").get("status") == "complete"
            ):
                result = check_trial(directory)
                print(f"{mode}: reusing completed, verified trial.", flush=True)
            else:
                relative = f"{mode}/attempt_{uuid.uuid4().hex[:12]}"
                directory = child_path(output, relative)
                directory.parent.mkdir(exist_ok=True)
                state["trials"][mode] = relative
                neural.write_json(output / "status.json", state)
                result = fit_trial(
                    directory,
                    frames,
                    [path.parent for path in paths],
                    config,
                    mode,
                    device,
                    manifest,
                )
            check_result_identity(result, manifest, mode)
            results.append(result)
        if results[0]["initial_state_sha256"] != results[1]["initial_state_sha256"]:
            raise ValueError("Paired models did not start with identical weights.")
        check_reference(reference)
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Source changed during training; do not pull in a running clone.")
        for name, path in zip(("train", "validation"), paths, strict=True):
            if (
                neural.file_sha(path) != manifest["inputs"][name]["csv_sha256"]
                or neural.image_digest(path) != manifest["inputs"][name]["images_sha256"]
            ):
                raise ValueError("Training/evaluation inputs changed during the run.")
        state["status"] = "complete"
        neural.write_json(output / "status.json", state)
        print("Paired experiment complete. Old models are unchanged; no Test was read.")
        return results
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        neural.write_json(output / "status.json", state)
        raise


def experiment_results(directory):
    directory = Path(directory)
    manifest = read_json(directory / "experiment.json")
    state = read_json(directory / "status.json")
    if (
        manifest.get("kind") != KIND
        or manifest.get("format_version") != 1
        or state.get("status") != "complete"
        or state.get("manifest_sha256") != neural.file_sha(directory / "experiment.json")
        or set(state.get("trials", {})) != set(MODES)
    ):
        raise ValueError("The paired experiment is incomplete or its manifest changed.")
    results = {}
    for mode in MODES:
        trial = child_path(directory, state["trials"][mode])
        result = check_trial(trial)
        check_result_identity(result, manifest, mode)
        results[mode] = result
    if results["frozen"]["initial_state_sha256"] != results["layer4"]["initial_state_sha256"]:
        raise ValueError("Paired initial weights differ.")
    return manifest, results


def report(args):
    manifest, results = experiment_results(args.output_dir)
    historical = manifest["reference"]["metrics"]
    rows = {
        "historical_neural": {
            "metrics": historical,
            "selected_epoch": historical.get("epoch", "NA"),
        },
        **results,
    }
    frozen_score = results["frozen"]["metrics"]["macro_f1"]
    print(
        "method\tselected_epoch\tvalidation_macro_f1\tdelta_vs_matched_frozen\t"
        + "\t".join(f"f1_{label}" for label in LABELS)
    )
    for name, row in rows.items():
        metrics = row["metrics"]
        delta = (
            "NA" if name == "historical_neural" else f"{metrics['macro_f1'] - frozen_score:+.6f}"
        )
        print(
            f"{name}\t{row['selected_epoch']}\t{metrics['macro_f1']:.6f}\t{delta}\t"
            + "\t".join(f"{metrics['class_f1'][str(label)]:.6f}" for label in LABELS)
        )
    if args.class_details:
        print("\nmethod\tclass\tprecision\trecall\tf1\tsupport")
        for name, row in rows.items():
            metrics = row["metrics"]
            for label in LABELS:
                key = str(label)
                print(
                    f"{name}\t{label}\t{metrics['class_precision'][key]:.6f}\t"
                    f"{metrics['class_recall'][key]:.6f}\t{metrics['class_f1'][key]:.6f}\t"
                    f"{metrics['support'][key]}"
                )
    print("Historical neural uses a different optimizer/selection protocol; it is context only.")
    print("Compare layer4 vs matched frozen. Epochs selected inside Train; official Validation")
    print("was used only for comparison/replay, but has informed earlier architecture choices.")
    print("This is one group holdout/seed, not 3-fold CV. Existing 50:50 is unchanged.")
    state = read_json(args.output_dir / "status.json")
    for mode in MODES:
        print(f"{mode} model_dir: {args.output_dir / state['trials'][mode]}")
    return rows


def load_checkpoint(run_dir, device):
    run_dir = Path(run_dir)
    result = check_trial(run_dir)
    if result["source_sha256"] != source_hashes():
        raise ValueError("Training source changed; use the recorded source revision.")
    seed_everything(result["config"]["seed"])
    target = select_device(str(device))
    if result["runtime"] != runtime_signature(target):
        raise ValueError("Use the same inference software/device/precision as this experiment.")
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    config = normalize_config(checkpoint["config"])
    expected_model = {"mode": result["mode"], **model_kwargs(config)}
    if (
        checkpoint.get("kind") != CHECKPOINT_KIND
        or checkpoint.get("format_version") != 1
        or checkpoint.get("model_config") != expected_model
        or checkpoint.get("label_values") != list(LABELS)
        or checkpoint.get("image_contract") != IMAGE_CONTRACT
        or checkpoint.get("source_sha256") != result["source_sha256"]
        or checkpoint.get("runtime") != result["runtime"]
        or config != result["config"]
        or checkpoint.get("mode") != result["mode"]
        or checkpoint.get("selected_epoch") != result["selected_epoch"]
    ):
        raise ValueError("Checkpoint contract differs from its recorded trial.")
    prep = NeuralSensorPreprocessor.from_dict(checkpoint["preprocessor"])
    if (
        prep.feature_names != RAW_SENSOR_COLUMNS
        or prep.imputation != "median"
        or prep.scaling != "standard"
        or prep.uses_history
    ):
        raise ValueError("Prediction requires saved raw5/median/standard preprocessing.")
    model = FineTuneModel(**expected_model, pretrained=False).to(target)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, prep, checkpoint


def predict(args):
    import pandas as pd

    if args.output.exists():
        raise ValueError("Output exists; choose a new filename.")
    model, prep, checkpoint = load_checkpoint(args.run_dir, args.device)
    frame = read_frame(args.input)
    config = {**checkpoint["config"], "batch_size": args.batch_size}
    normalize_config(config)
    loader = evaluation_loader(frame, args.image_root, prep, config, labeled=False)
    prob = probabilities(model, loader, select_device(args.device), progress_every=2048)
    labels = prob.argmax(1) + 1
    ids = frame["sample_id"].tolist()
    if args.sample_submission is not None:
        template = pd.read_csv(args.sample_submission, dtype=str, keep_default_na=False)
        if (
            list(template.columns) != ["sample_id", "label"]
            or len(template) != len(ids)
            or not template["sample_id"].is_unique
            or set(template["sample_id"]) != set(ids)
        ):
            raise ValueError("Submission template must exactly match input IDs.")
        labels = pd.Series(labels, index=ids).loc[template["sample_id"]].to_numpy()
        ids = template["sample_id"].tolist()
    validate_labels(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        pd.DataFrame({"sample_id": ids, "label": labels}).to_csv(handle, index=False)
    print("Prediction CSV saved from current rows and saved weights; no fitting occurred.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("run", help="Sequential matched frozen/layer4 experiment")
    for name in ("reference_run", "data_dir", "output_dir"):
        training.add_argument("--" + name, type=Path, required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    training.add_argument(
        "--resume",
        action="store_true",
        help="Verify/reuse complete trials; restart failed trials in new folders",
    )
    training.set_defaults(handler=run)
    reporting = commands.add_parser("report")
    reporting.add_argument("--output_dir", type=Path, required=True)
    reporting.add_argument("--class_details", action="store_true")
    reporting.set_defaults(handler=report)
    prediction = commands.add_parser("predict")
    for name in ("run_dir", "input", "image_root", "output"):
        prediction.add_argument("--" + name, type=Path, required=True)
    prediction.add_argument("--sample_submission", type=Path)
    prediction.add_argument("--batch_size", type=int, default=32)
    prediction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prediction.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
