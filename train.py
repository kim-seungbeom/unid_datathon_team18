"""Train on explicitly supplied CSVs. Run competition data only on the authorized server."""

import argparse
import math
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import BatteryDataset
from evaluate import classification_metrics
from model import FusionClassifier
from preprocessing import SensorPreprocessor
from runtime import (
    ROOT,
    environment_metadata,
    file_sha256,
    load_config,
    predict_labels,
    seed_everything,
    select_device,
    write_json,
)
from schema import LABELS, read_frame, validate_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=Path, help="Organizer root containing train/ and validation/"
    )
    parser.add_argument("--train_csv", type=Path)
    parser.add_argument("--val_csv", type=Path)
    parser.add_argument(
        "--train_image_root", type=Path, help="Defaults to the training CSV directory"
    )
    parser.add_argument(
        "--val_image_root", type=Path, help="Defaults to the validation CSV directory"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument(
        "--run_dir", type=Path, required=True, help="New, unique experiment directory"
    )
    parser.add_argument("--model_out", type=Path, help="Optional export, e.g. weights/model.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.data_dir is not None:
        if args.train_csv is not None or args.val_csv is not None:
            parser.error("Use --data_dir OR --train_csv and --val_csv.")
        args.train_csv = args.data_dir / "train/train.csv"
        args.val_csv = args.data_dir / "validation/validation.csv"
    elif args.train_csv is None or args.val_csv is None:
        parser.error("Supply --data_dir, or both --train_csv and --val_csv.")
    train_image_root = args.train_image_root or args.train_csv.parent
    val_image_root = args.val_image_root or args.val_csv.parent
    if args.model_out is not None and args.model_out.exists():
        parser.error("model_out already exists. Choose a new path to preserve previous weights.")

    config = load_config(args.config)
    use_images = config["mode"] == "multimodal"
    if not use_images:
        print(
            "Sensor-only ablation: validation research only; final model must use both modalities."
        )
    seed_everything(config["seed"])
    device = select_device(args.device)
    train_frame = read_frame(args.train_csv, labeled=True, require_images=use_images)
    val_frame = read_frame(args.val_csv, labeled=True, require_images=use_images)
    validate_split(train_frame, val_frame)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    preprocessor = SensorPreprocessor.fit(
        train_frame, config["sensor_features"], imputation=config["sensor_imputation"]
    )
    model_config = {key: config[key] for key in ("sensor_hidden", "fusion_hidden", "dropout")}
    model_config.update(sensor_input_dim=preprocessor.input_dim, mode=config["mode"])
    model = FusionClassifier(**model_config).to(device)
    train_data = BatteryDataset(
        train_frame,
        train_image_root,
        preprocessor,
        config["image_size"],
        labeled=True,
        use_images=use_images,
    )
    val_data = BatteryDataset(
        val_frame, val_image_root, preprocessor, config["image_size"], use_images=use_images
    )
    loader_options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"]}
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        **loader_options,
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    loss_function = nn.CrossEntropyLoss()
    write_json(args.run_dir / "config.json", config)
    write_json(args.run_dir / "preprocessing.json", preprocessor.to_dict())
    write_json(args.run_dir / "environment.json", environment_metadata(device))
    write_json(
        args.run_dir / "split.json",
        {
            "method": "explicit_train_and_validation_csvs",
            "train_csv_sha256": file_sha256(args.train_csv),
            "validation_csv_sha256": file_sha256(args.val_csv),
            "train_rows": len(train_frame),
            "validation_rows": len(val_frame),
            "shared_experiments_present": bool(
                set(train_frame["experiment_id"]) & set(val_frame["experiment_id"])
            ),
            "mode": config["mode"],
            "sensor_features": list(preprocessor.feature_names),
            "uses_images": use_images,
            "sensor_imputation": preprocessor.imputation,
            "uses_history": preprocessor.uses_history,
            "features": "same-experiment current/past only; no targets or future observations",
        },
    )
    history = []
    best_score = -1.0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["images"].to(device) if use_images else None
            logits = model(images, batch["sensors"].to(device))
            loss = loss_function(logits, batch["targets"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["targets"])
        mean_loss = total_loss / len(train_data)
        if not math.isfinite(mean_loss):
            raise ValueError("Mean training loss is nonfinite.")
        predictions = predict_labels(model, val_loader, device)
        metrics = classification_metrics(val_frame["label"], predictions)
        history.append({"epoch": epoch, "train_loss": mean_loss, **metrics})
        write_json(args.run_dir / "history.json", {"epochs": history})
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            checkpoint = {
                "format_version": 3,
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "model_config": model_config,
                "preprocessor": preprocessor.to_dict(),
                "image_size": config["image_size"],
                "seed": config["seed"],
                "label_values": list(LABELS),
                "epoch": epoch,
            }
            temporary = args.run_dir / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.run_dir / "best.pt")
            write_json(args.run_dir / "best_metrics.json", {"epoch": epoch, **metrics})
        print(f"epoch={epoch} loss={mean_loss:.6f} val_macro_f1={metrics['macro_f1']:.6f}")
    if args.model_out is not None:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / "best.pt").open("rb") as source, args.model_out.open("xb") as target:
            shutil.copyfileobj(source, target)
    print("Training complete. The run directory contains best.pt and reproducibility metadata.")


if __name__ == "__main__":
    main()
