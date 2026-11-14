"""Personal neural fusion experiments. Competition runs are user-executed on the server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep run_identity and sweep planning usable without importing Torch.
from experiments.seungbeom.neural_config import load_config, normalize_config  # noqa: E402

SOURCE_FILES = (
    "schema.py",
    "sensor_features.py",
    "temporal.py",
    "preprocessing.py",
    "dataset.py",
    "runtime.py",
    "evaluate.py",
    "training_objectives.py",
    "image_features/encoders.py",
    "image_features/extract_features.py",
    "experiments/seungbeom/neural_config.py",
    "experiments/seungbeom/neural_model.py",
    "experiments/seungbeom/neural_data.py",
    "experiments/seungbeom/neural_experiment.py",
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_digest(csv_path):
    import pandas as pd

    root = Path(csv_path).parent.resolve()
    frame = pd.read_csv(csv_path, usecols=["image_path"], dtype=str, keep_default_na=False)
    digest = hashlib.sha256()
    for relative in frame["image_path"]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image escapes its explicit split directory.")
        digest.update(relative.encode() + b"\0" + file_sha(path).encode())
    return digest.hexdigest()


def run_identity(config, train_csv, val_csv, *, train_embeddings=None, val_embeddings=None):
    config = normalize_config(config)
    if (train_embeddings is None) != (val_embeddings is None):
        raise ValueError("Provide both Train and Validation embedding folders.")
    if config["encoder"] != "embedding" and train_embeddings is not None:
        raise ValueError("Raw-image encoder experiments do not accept cached vectors.")
    provenance = {}
    for name, csv_path, cache in (
        ("train", train_csv, train_embeddings),
        ("validation", val_csv, val_embeddings),
    ):
        if cache is None:
            provenance[name] = {"image_content_sha256": image_digest(csv_path)}
        else:
            provenance[name] = {
                filename: file_sha(Path(cache) / filename)
                for filename in ("features.npy", "sample_ids.csv", "metadata.json")
            }
    return {
        "config": config,
        "train_csv_sha256": file_sha(train_csv),
        "validation_csv_sha256": file_sha(val_csv),
        "source_sha256": {name: file_sha(ROOT / name) for name in SOURCE_FILES},
        "input_provenance": provenance,
    }


class EarlyStopping:
    """Strict best saves independently of a significant-improvement patience reference."""

    def __init__(self, patience=0, min_delta=0.0):
        if (
            type(patience) is not int
            or patience < 0
            or not math.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError("Invalid early stopping settings.")
        self.patience, self.min_delta = patience, min_delta
        self.best_score = self.reference_score = -math.inf
        self.best_epoch = self.bad_epochs = 0

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError("Validation selection metric must be finite.")
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = float(score), int(epoch)
        if score > self.reference_score + self.min_delta:
            self.reference_score, self.bad_epochs = float(score), 0
        else:
            self.bad_epochs += 1
        return improved, bool(self.patience and self.bad_epochs >= self.patience)


def synchronize(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def infer(model, loader, device, *, stable=True, measure_loss=False):
    import numpy as np
    import torch
    from torch.nn import functional as F

    model.eval()
    predictions, total_ce, count = [], 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            size = len(batch["sensors"])
            slices = (slice(i, i + 1) for i in range(size)) if stable else (slice(None),)
            for selection in slices:
                logits = model(
                    batch["visual"][selection].to(device), batch["sensors"][selection].to(device)
                )
                if not torch.isfinite(logits).all():
                    raise ValueError("Model logits are nonfinite.")
                predictions.extend((logits.argmax(dim=1) + 1).cpu().tolist())
                if measure_loss:
                    total_ce += float(
                        F.cross_entropy(
                            logits, batch["targets"][selection].to(device), reduction="sum"
                        ).item()
                    )
                count += len(logits)
    return np.asarray(predictions, dtype=np.int64), total_ce / count if measure_loss else None


def evaluate(model, loader, device, *, stable=False):
    from evaluate import classification_metrics

    predictions, ce = infer(model, loader, device, stable=stable, measure_loss=True)
    metrics = classification_metrics(loader.dataset.frame["label"], predictions)
    if not math.isfinite(ce) or not math.isfinite(metrics["macro_f1"]):
        raise ValueError("Validation metrics are nonfinite.")
    return {"ce_loss": ce, **metrics}


def train(args):
    import torch
    from torch.utils.data import DataLoader

    from experiments.seungbeom.neural_config import model_config
    from experiments.seungbeom.neural_data import VisualDataset, visual_vectors
    from experiments.seungbeom.neural_model import NeuralFusionClassifier
    from preprocessing import SensorPreprocessor
    from runtime import environment_metadata, seed_everything, select_device
    from schema import LABELS, read_frame, validate_split
    from training_objectives import build_training_components

    config = load_config(args.config) if args.config else normalize_config({})
    train_csv = args.data_dir / "train/train.csv"
    val_csv = args.data_dir / "validation/validation.csv"
    frames = [read_frame(path, labeled=True) for path in (train_csv, val_csv)]
    validate_split(*frames)
    identity = run_identity(
        config,
        train_csv,
        val_csv,
        train_embeddings=args.train_embeddings,
        val_embeddings=args.val_embeddings,
    )
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "status.json", {"status": "running"})
    write_json(run / "config.json", config)
    write_json(run / "identity.json", identity)
    try:
        device = select_device(args.device)
        seed_everything(config["seed"])
        write_json(run / "environment.json", environment_metadata(device))
        preprocessor = SensorPreprocessor.fit(
            frames[0],
            config["sensor_features"],
            imputation=config["sensor_imputation"],
            scaling=config["sensor_scaling"],
        )
        vectors, visual_contract = (
            [None, None],
            {
                "mode": "rgb_bilinear_resize_0_1",
                "image_size": config["image_size"],
                "pretrained": False,
            },
        )
        if config["encoder"] == "embedding":
            vectors[0], visual_contract = visual_vectors(
                frames[0],
                train_csv.parent,
                config,
                cache=args.train_embeddings,
                device=args.device,
                work_dir=run / "embeddings/train",
            )
            vectors[1], _ = visual_vectors(
                frames[1],
                val_csv.parent,
                config,
                cache=args.val_embeddings,
                expected=visual_contract,
                device=args.device,
                work_dir=run / "embeddings/validation",
            )
        # Public feature extraction may reset Torch RNG; initialize each training trial here.
        seed_everything(config["seed"])
        datasets = [
            VisualDataset(frame, path.parent, preprocessor, config, vectors=vec)
            for frame, path, vec in zip(frames, (train_csv, val_csv), vectors, strict=True)
        ]
        options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"]}
        train_loader = DataLoader(
            datasets[0],
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
            **options,
        )
        val_loader = DataLoader(datasets[1], shuffle=False, **options)
        model = NeuralFusionClassifier(**model_config(config)).to(device)
        criterion, sampler, objective = build_training_components(
            frames[0]["label"], config, device
        )
        if sampler is not None:
            raise ValueError("This controlled neural comparison requires shuffled sampling.")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        stopper = EarlyStopping(config["early_stopping_patience"], config["min_delta"])
        write_json(run / "preprocessing.json", preprocessor.to_dict())
        write_json(run / "training_objective.json", objective)
        history, stopped = [], False
        synchronize(device)
        started = time.perf_counter()
        for epoch in range(1, config["epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train()
            total = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["visual"].to(device), batch["sensors"].to(device))
                loss = criterion(logits, batch["targets"].to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is nonfinite.")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("Training gradients are nonfinite.")
                optimizer.step()
                total += float(loss.item()) * len(batch["targets"])
            # Epoch selection uses the same row-independent path as exported inference.
            metrics = evaluate(model, val_loader, device, stable=True)
            improved, stopped = stopper.update(metrics["macro_f1"], epoch)
            synchronize(device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(datasets[0]),
                    "validation_ce_loss": metrics["ce_loss"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    **metrics,
                }
            )
            write_json(run / "history.json", {"epochs": history})
            if improved:
                checkpoint = {
                    "kind": "seungbeom_neural_fusion",
                    "format_version": 1,
                    "config": config,
                    "model_config": model.get_model_config(),
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "preprocessor": preprocessor.to_dict(),
                    "visual_contract": visual_contract,
                    "label_values": list(LABELS),
                    "epoch": epoch,
                }
                torch.save(checkpoint, run / "best.pt.tmp")
                (run / "best.pt.tmp").replace(run / "best.pt")
                write_json(run / "best_metrics.json", {"epoch": epoch, **metrics})
            print(
                f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
                f"val_ce={metrics['ce_loss']:.6f} val_macro_f1={metrics['macro_f1']:.6f} "
                f"best_epoch={stopper.best_epoch} stale={stopper.bad_epochs}",
                flush=True,
            )
            if stopped:
                break
        synchronize(device)
        training_seconds = time.perf_counter() - started
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        synchronize(device)
        inference_started = time.perf_counter()
        final_metrics = evaluate(model, val_loader, device, stable=True)
        synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        write_json(run / "best_metrics.json", {"epoch": checkpoint["epoch"], **final_metrics})
        write_json(
            run / "summary.json",
            {
                "elapsed_epochs": len(history),
                "best_epoch": checkpoint["epoch"],
                "stopped_early": stopped,
                "stop_reason": "patience" if stopped else "epoch_limit",
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "inference_rows": len(datasets[1]),
                "inference_batch_size": 1,
                "inference_includes_feature_extraction": config["encoder"] != "embedding",
                "inference_scope": (
                    "cached_visual_and_sensor_head"
                    if config["encoder"] == "embedding"
                    else "image_encoder_and_sensor_head"
                ),
                "optimizer": "AdamW",
                "scheduler": None,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "selection_split": "validation",
                "resume_supported": False,
            },
        )
        write_json(run / "status.json", {"status": "complete"})
        print(
            f"Complete: best_epoch={checkpoint['epoch']} "
            f"validation_macro_f1={final_metrics['macro_f1']:.6f}",
            flush=True,
        )
    except BaseException as error:
        write_json(run / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise


def predict(args):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from experiments.seungbeom.neural_data import VisualDataset, visual_vectors
    from experiments.seungbeom.neural_model import NeuralFusionClassifier
    from preprocessing import SensorPreprocessor
    from runtime import seed_everything, select_device
    from schema import LABELS, read_frame

    if args.output.exists():
        raise ValueError("Output exists. Use a new filename.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("kind") != "seungbeom_neural_fusion"
        or checkpoint.get("format_version") != 1
        or checkpoint.get("label_values") != list(LABELS)
    ):
        raise ValueError(
            "Unsupported neural checkpoint. Existing ensemble checkpoints use their own CLI."
        )
    config = normalize_config(checkpoint["config"])
    from experiments.seungbeom.neural_config import model_config

    if checkpoint["model_config"] != model_config(config):
        raise ValueError("Saved model structure and config disagree.")
    seed_everything(config["seed"])
    device = select_device(args.device)
    frame = read_frame(args.input)
    vectors = None
    if config["encoder"] == "embedding":
        vectors, _ = visual_vectors(
            frame,
            args.image_root,
            config,
            expected=checkpoint["visual_contract"],
            device=args.device,
        )
    preprocessor = SensorPreprocessor.from_dict(checkpoint["preprocessor"])
    if tuple(config["sensor_features"]) != preprocessor.feature_names:
        raise ValueError("Saved sensor feature order differs from model config.")
    dataset = VisualDataset(
        frame, args.image_root, preprocessor, config, vectors=vectors, labeled=False
    )
    model = NeuralFusionClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    labels, _ = infer(model, DataLoader(dataset, batch_size=1), device)
    ids = frame["sample_id"].tolist()
    if args.sample_submission is not None:
        template = pd.read_csv(args.sample_submission, dtype=str, keep_default_na=False)
        requested = template["sample_id"].tolist()
        if (
            len(requested) != len(ids)
            or not template["sample_id"].is_unique
            or set(requested) != set(ids)
        ):
            raise ValueError("Submission template IDs differ from the supplied input.")
        labels = pd.Series(labels, index=ids).loc[requested].to_numpy()
        ids = requested
    if not np.isin(labels, LABELS).all():
        raise ValueError("Predictions must be labels 1..6.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        pd.DataFrame({"sample_id": ids, "label": labels}).to_csv(stream, index=False)
    print("Neural prediction CSV saved.")


def report(args):
    import pandas as pd

    if json.loads((args.run_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("Run is not complete.")
    metrics = json.loads((args.run_dir / "best_metrics.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    print(json.dumps(summary, indent=2))
    print(f"Validation Macro F1: {metrics['macro_f1']:.6f}")
    print(
        pd.DataFrame(
            {
                key: metrics[key]
                for key in ("class_precision", "class_recall", "class_f1", "support")
            }
        ).to_string()
    )
    history = json.loads((args.run_dir / "history.json").read_text())["epochs"]
    print(
        pd.DataFrame(history)[["epoch", "train_loss", "validation_ce_loss", "macro_f1"]].to_string(
            index=False
        )
    )


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--config", type=Path)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--train_embeddings", type=Path)
    p.add_argument("--val_embeddings", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.set_defaults(handler=train)
    p = commands.add_parser("predict")
    for name in ("input", "image_root", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sample_submission", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.set_defaults(handler=predict)
    p = commands.add_parser("report")
    p.add_argument("--run_dir", type=Path, required=True)
    p.set_defaults(handler=report)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
