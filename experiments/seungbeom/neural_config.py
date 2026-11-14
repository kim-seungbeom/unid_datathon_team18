"""Canonical settings for personal neural experiments, independent of legacy configs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from experiments.seungbeom.neural_sensors import validate_neural_features
from schema import RAW_SENSOR_COLUMNS

EMBEDDING_DIMENSIONS = {"resnet18": 512, "dinov2": 384, "convnext_tiny": 768}
DEFAULT_CONFIG = {
    "seed": 42,
    "encoder": "embedding",
    "fusion": "concat",
    "sensor_features": list(RAW_SENSOR_COLUMNS),
    "sensor_imputation": "median",
    "sensor_scaling": "standard",
    "sensor_hidden": 32,
    "fusion_hidden": 64,
    "projection_dim": 64,
    "dropout": 0.1,
    "embedding_encoder": "resnet18",
    "embedding_dim": 512,
    "image_size": 128,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "early_stopping_patience": 8,
    "min_delta": 0.0,
    "loss": "cross_entropy",
    "sampling": "shuffle",
    "class_weight_power": 0.5,
    "focal_gamma": 2.0,
    "num_workers": 0,
}


def normalize_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Neural config must be a JSON object.")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown neural config keys: {', '.join(sorted(unknown))}")
    result = copy.deepcopy(DEFAULT_CONFIG)
    result.update(copy.deepcopy(config))
    if result["encoder"] not in ("embedding", "small_cnn", "resnet18"):
        raise ValueError("Unsupported neural encoder.")
    if result["fusion"] not in ("concat", "projected_concat", "gated"):
        raise ValueError("Unsupported neural fusion.")
    if result["embedding_encoder"] not in EMBEDDING_DIMENSIONS:
        raise ValueError("Unsupported frozen embedding encoder.")
    positive = (
        "sensor_hidden",
        "fusion_hidden",
        "projection_dim",
        "embedding_dim",
        "image_size",
        "epochs",
        "batch_size",
    )
    for name in positive:
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("seed", "num_workers", "early_stopping_patience"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if result["seed"] >= 2**32:
        raise ValueError("seed must be below 2**32.")
    if result["image_size"] < 32:
        raise ValueError("image_size must be at least 32 for the encoder comparison.")
    expected = EMBEDDING_DIMENSIONS[result["embedding_encoder"]]
    if result["embedding_dim"] != expected:
        raise ValueError("embedding_dim must match the frozen embedding encoder.")
    bounds = {
        "learning_rate": (0.0, None, False),
        "weight_decay": (0.0, None, True),
        "min_delta": (0.0, None, True),
        "focal_gamma": (0.0, None, True),
        "class_weight_power": (0.0, 1.0, True),
        "dropout": (0.0, 1.0, True),
    }
    for name, (lower, upper, inclusive) in bounds.items():
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < lower or (not inclusive and value == lower):
            raise ValueError(f"Invalid {name}.")
        if upper is not None and (value > upper or (name == "dropout" and value == upper)):
            raise ValueError(f"Invalid {name}.")
        result[name] = float(value)
    names = list(validate_neural_features(result["sensor_features"]))
    if result["sensor_imputation"] != "median" or result["sensor_scaling"] != "standard":
        raise ValueError("Neural structure comparisons use median imputation and standard scaling.")
    result["sensor_features"] = names
    if result["loss"] not in ("cross_entropy", "weighted_cross_entropy", "focal"):
        raise ValueError("Unsupported neural loss.")
    if result["sampling"] != "shuffle":
        raise ValueError("Controlled neural comparisons require sampling='shuffle'.")
    return result


def load_config(path: str | Path) -> dict:
    return normalize_config(json.loads(Path(path).read_text(encoding="utf-8")))


def model_config(config: dict) -> dict:
    normalized = normalize_config(config)
    return {
        "encoder": normalized["encoder"],
        "fusion": normalized["fusion"],
        "sensor_dim": len(normalized["sensor_features"]),
        **{
            key: normalized[key]
            for key in (
                "sensor_hidden",
                "fusion_hidden",
                "projection_dim",
                "dropout",
                "embedding_dim",
            )
        },
    }
