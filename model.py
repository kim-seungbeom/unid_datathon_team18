"""Multimodal model and sensor-only ablation; no external data or pretrained weights."""

import torch
from torch import nn

from schema import LABELS, RAW_SENSOR_COLUMNS


class FusionClassifier(nn.Module):
    def __init__(
        self,
        sensor_hidden: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.1,
        sensor_input_dim: int = len(RAW_SENSOR_COLUMNS),
        mode: str = "multimodal",
    ):
        super().__init__()
        if mode not in {"multimodal", "sensor_only"}:
            raise ValueError("mode must be multimodal or sensor_only.")
        if type(sensor_input_dim) is not int or sensor_input_dim <= 0:
            raise ValueError("sensor_input_dim must be a positive integer.")
        self.sensor_input_dim = sensor_input_dim
        self.uses_images = mode == "multimodal"
        # No batch-dependent normalization: each observation has independent features.
        self.image_encoder = (
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            if self.uses_images
            else None
        )
        self.sensor_encoder = nn.Sequential(nn.Linear(sensor_input_dim, sensor_hidden), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear((64 if self.uses_images else 0) + sensor_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, len(LABELS)),
        )

    def forward(self, images: torch.Tensor | None, sensors: torch.Tensor) -> torch.Tensor:
        if sensors.ndim != 2 or sensors.shape[1] != self.sensor_input_dim:
            raise ValueError("Sensor feature dimension does not match the model.")
        features = self.sensor_encoder(sensors)
        if self.uses_images:
            if images is None:
                raise ValueError("Multimodal inference requires images.")
            image_features = self.image_encoder(images).mean(dim=(-2, -1))
            features = torch.cat((image_features, features), dim=1)
        return self.classifier(features)
