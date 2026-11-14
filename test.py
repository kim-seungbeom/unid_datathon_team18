"""Submission inference: uses only supplied rows, their images, and the saved checkpoint."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import BatteryDataset
from model import FusionClassifier
from preprocessing import SensorPreprocessor
from runtime import ROOT, predict_labels, seed_everything, select_device
from schema import LABELS, read_frame, validate_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", "--model", type=Path, default=ROOT / "weights/model.pt")
    parser.add_argument("--batch_size", type=int, default=32, help="Image-loading batch size")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers must be nonnegative.")
    if args.output.exists():
        parser.error("Output already exists. Choose a new path to avoid overwriting a file.")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint is missing. Supply --checkpoint or prepare weights/model.pt.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in (1, 2, 3) or checkpoint.get("label_values") != list(
        LABELS
    ):
        raise ValueError("Unsupported checkpoint format or class mapping.")
    seed_everything(checkpoint["seed"])
    device = select_device(args.device)
    preprocessor = SensorPreprocessor.from_dict(checkpoint["preprocessor"])
    model = FusionClassifier(**checkpoint["model_config"])
    if model.sensor_input_dim != preprocessor.input_dim:
        raise ValueError("Saved model dimension does not match saved sensor features.")
    if not model.uses_images:
        print("Sensor-only ablation predictions; not a final multimodal competition submission.")
    frame = read_frame(args.input, require_images=model.uses_images)
    dataset = BatteryDataset(
        frame, args.image_root, preprocessor, checkpoint["image_size"], use_images=model.uses_images
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    labels = validate_labels(predict_labels(model, loader, device))
    if len(labels) != len(frame):
        raise ValueError("Prediction count does not match input row count.")
    result = pd.DataFrame({"sample_id": frame["sample_id"], "label": labels})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards against an output file appearing during inference.
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    print("Inference complete: sample_id,label CSV saved in input row order.")


if __name__ == "__main__":
    main()
