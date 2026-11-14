"""Isolated pretrained ResNet18/raw5 helpers; no data discovery or artifact writes.

Both experimental arms use the same current-image transform and concat head.
The only backbone training difference is whether layer4 is trainable. BatchNorm
running statistics stay frozen in both arms, including inside layer4.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import Dataset

from image_features.encoders import preprocess
from schema import RAW_SENSOR_COLUMNS, validate_labels

WEIGHTS_SOURCE = "torchvision:ResNet18_Weights.IMAGENET1K_V1"
MODES = ("frozen", "layer4")


class FineTuneModel(nn.Module):
    """Pretrained backbone plus the existing raw5 concat classification topology.

    ``pretrained=False`` constructs the same topology without a download and is
    reserved for loading an entire saved state_dict or synthetic CPU tests.
    """

    def __init__(
        self,
        mode: str = "frozen",
        *,
        sensor_hidden: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be frozen or layer4.")
        if any(type(value) is not int or value < 1 for value in (sensor_hidden, fusion_hidden)):
            raise ValueError("Hidden dimensions must be positive integers.")
        if (
            type(dropout) not in (int, float)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1).")
        if type(pretrained) is not bool:
            raise ValueError("pretrained must be a boolean.")
        from torchvision.models import ResNet18_Weights, resnet18

        self.mode = mode
        self._model_config = {
            "mode": mode,
            "sensor_hidden": sensor_hidden,
            "fusion_hidden": fusion_hidden,
            "dropout": float(dropout),
        }
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=False,
        )
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        if mode == "layer4":
            self.backbone.layer4.requires_grad_(True)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(len(RAW_SENSOR_COLUMNS), sensor_hidden), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + sensor_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 6),
        )
        self.train(True)

    def get_model_config(self) -> dict:
        return dict(self._model_config)

    def train(self, mode: bool = True):
        super().train(mode)
        # Earlier blocks always retain their pretrained evaluation behavior.
        self.backbone.eval()
        if self.mode == "layer4":
            self.backbone.layer4.train(mode)
        # Affine parameters inside layer4 can learn, but moving means/variances
        # are never estimated on this competition's minibatches.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Images must have shape [batch, 3, height, width].")
        if sensors.ndim != 2 or sensors.shape[1] != len(RAW_SENSOR_COLUMNS):
            raise ValueError("Sensors must contain exactly the saved raw5 feature order.")
        if len(images) != len(sensors):
            raise ValueError("Image and sensor batch sizes differ.")
        visual = self.backbone(images)
        if visual.ndim != 2 or visual.shape != (len(images), 512):
            raise ValueError("ResNet18 must return one 512-dimensional vector per image.")
        encoded = self.sensor_encoder(sensors)
        return self.classifier(torch.cat((visual, encoded), dim=1))


class FineTuneDataset(Dataset):
    """Current image and current raw5 sensors only; never fits preprocessing."""

    def __init__(self, frame, image_root, preprocessor, labeled=True):
        if (
            tuple(preprocessor.feature_names) != RAW_SENSOR_COLUMNS
            or preprocessor.imputation != "median"
            or preprocessor.scaling != "standard"
            or preprocessor.uses_history
        ):
            raise ValueError("Fine-tuning requires raw5/median/standard current-row preprocessing.")
        if frame.empty or "image_path" not in frame:
            raise ValueError("A nonempty input with image_path is required.")
        self.frame = frame.reset_index(drop=True).copy()
        if self.frame["image_path"].map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).any():
            raise ValueError("Image paths must be nonempty strings.")
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise ValueError("The explicit image root must be an existing directory.")
        self.sensors = preprocessor.transform(self.frame)
        if self.sensors.shape != (len(self.frame), len(RAW_SENSOR_COLUMNS)) or not np.isfinite(
            self.sensors
        ).all():
            raise ValueError("Preprocessing must produce finite raw5 sensor features.")
        self.sensors = np.asarray(self.sensors, dtype=np.float32)
        self.targets = None
        if labeled:
            if "label" not in self.frame:
                raise ValueError("Labeled training/evaluation requires labels.")
            self.targets = validate_labels(self.frame["label"]) - 1

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = (self.image_root / self.frame.iloc[index]["image_path"]).resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError("An image path resolves outside the explicit image root.")
        try:
            with Image.open(path) as source:
                if source.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                    raise ValueError("Only documented 8-bit image modes are supported.")
                visual = preprocess(source)
        except (OSError, UnidentifiedImageError):
            # Do not emit file names or individual competition observations.
            raise ValueError("A referenced image is missing or unreadable.") from None
        item = {"visual": visual, "sensors": torch.from_numpy(self.sensors[index])}
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[index], dtype=torch.long)
        return item


def make_optimizer(model, head_lr, backbone_lr, weight_decay):
    for name, value in (("head_lr", head_lr), ("backbone_lr", backbone_lr)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if (
        type(weight_decay) not in (int, float)
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be finite and nonnegative.")
    head = list(model.sensor_encoder.parameters()) + list(model.classifier.parameters())
    groups = [{"params": head, "lr": float(head_lr), "group_name": "head"}]
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": float(backbone_lr), "group_name": "layer4"})
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def probabilities(model, loader, device, *, progress_every=0):
    """Single-row FP32 inference: the caller's batching/order cannot alter a sample.

    A caller may use different batch sizes for IO throughput; every model forward
    still receives one contiguous image/sensor pair. No labels enter this path.
    """
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer.")
    model.eval()
    result = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch["visual"]) != len(batch["sensors"]):
                raise ValueError("Image and sensor batch sizes differ.")
            for index in range(len(batch["sensors"])):
                image = batch["visual"][index : index + 1].to(
                    device=device, dtype=torch.float32
                ).contiguous()
                sensor = batch["sensors"][index : index + 1].to(
                    device=device, dtype=torch.float32
                ).contiguous()
                logits = model(image, sensor)
                if logits.shape != (1, 6) or not torch.isfinite(logits).all():
                    raise ValueError("Model must return finite logits for six classes.")
                values = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
                result.append(values.copy())
                if progress_every and len(result) % progress_every == 0:
                    print(f"Predicted {len(result)} images", flush=True)
    if not result:
        raise ValueError("Prediction requires at least one input row.")
    return np.asarray(result, dtype=np.float32)
