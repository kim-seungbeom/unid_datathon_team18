"""Shared configuration, reproducibility, inference, and artifact helpers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from preprocessing import validate_scaling
from sensor_features import DEFAULT_SENSOR_FEATURES, validate_feature_names, validate_imputation

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "schema.py",
    "sensor_features.py",
    "temporal.py",
    "dataset.py",
    "preprocessing.py",
    "model.py",
    "evaluate.py",
    "runtime.py",
    "train.py",
    "test.py",
)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    # Preserve existing raw-sensor multimodal configs.
    config.setdefault("mode", "multimodal")
    config.setdefault("sensor_features", list(DEFAULT_SENSOR_FEATURES))
    config.setdefault("sensor_imputation", "median")
    config.setdefault("sensor_scaling", "standard")
    positive_ints = ("image_size", "batch_size", "epochs", "sensor_hidden", "fusion_hidden")
    expected = set(positive_ints) | {
        "seed",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "dropout",
        "mode",
        "sensor_features",
        "sensor_imputation",
        "sensor_scaling",
    }
    if set(config) != expected:
        raise ValueError("Config contains missing or unsupported keys.")
    for key in positive_ints:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    if config["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32.")
    for key in ("learning_rate", "weight_decay", "dropout"):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]):
            raise ValueError(f"{key} must be a finite number.")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid learning rate or weight decay.")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1).")
    if config["mode"] not in ("multimodal", "sensor_only"):
        raise ValueError("mode must be multimodal or sensor_only.")
    config["sensor_features"] = list(validate_feature_names(config["sensor_features"]))
    validate_imputation(config["sensor_imputation"])
    validate_scaling(config["sensor_scaling"])
    return config


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the organizer's server environment.")
    return torch.device(name)


@torch.inference_mode()
def predict_labels(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    for batch in loader:
        # A fixed single-row forward pass avoids batch-size dependent floating-point paths.
        # batch_size controls image loading only. Optimize only after invariance tests pass.
        for index in range(len(batch["sensors"])):
            images = batch["images"][index : index + 1].to(device) if "images" in batch else None
            sensors = batch["sensors"][index : index + 1].to(device)
            logits = model(images, sensors)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced nonfinite logits.")
            predictions.append(int(logits.argmax(dim=1).item()) + 1)
    return np.asarray(predictions, dtype=np.int64)


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_metadata(device: torch.device) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
        "python": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "Pillow")
        },
        "torch_path": torch.__file__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
