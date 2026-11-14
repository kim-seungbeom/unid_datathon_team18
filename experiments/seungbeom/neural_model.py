"""Controlled neural fusion candidates; labels never enter the forward path.

Embedding mode consumes frozen vectors produced outside this model. Image modes
train from random initialization. Projected concatenation and gating use the
same Linear + ReLU projections without normalization so the gate has a direct
projection-matched control.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class NeuralFusionClassifier(nn.Module):
    def __init__(
        self,
        *,
        encoder: str = "embedding",
        fusion: str = "concat",
        sensor_dim: int = 5,
        sensor_hidden: int = 32,
        fusion_hidden: int = 64,
        projection_dim: int = 64,
        dropout: float = 0.1,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        if encoder not in ("embedding", "small_cnn", "resnet18"):
            raise ValueError("encoder must be embedding, small_cnn, or resnet18.")
        if fusion not in ("concat", "projected_concat", "gated"):
            raise ValueError("fusion must be concat, projected_concat, or gated.")
        for name, value in (
            ("sensor_dim", sensor_dim),
            ("sensor_hidden", sensor_hidden),
            ("fusion_hidden", fusion_hidden),
            ("projection_dim", projection_dim),
            ("embedding_dim", embedding_dim),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1).")
        self.encoder = encoder
        self.fusion = fusion
        self.sensor_dim = sensor_dim
        self.projection_dim = projection_dim
        self.embedding_dim = embedding_dim
        self._model_config = {
            "encoder": encoder,
            "fusion": fusion,
            "sensor_dim": sensor_dim,
            "sensor_hidden": sensor_hidden,
            "fusion_hidden": fusion_hidden,
            "projection_dim": projection_dim,
            "dropout": float(dropout),
            "embedding_dim": embedding_dim,
        }
        if encoder == "embedding":
            self.image_encoder = nn.Identity()
            image_dim = embedding_dim
        elif encoder == "small_cnn":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            image_dim = 64
        else:
            from torchvision.models import resnet18

            # Explicitly scratch-only; no network access or implicit pretrained weights.
            self.image_encoder = resnet18(weights=None)
            self.image_encoder.fc = nn.Identity()
            image_dim = 512
        self.sensor_encoder = nn.Sequential(nn.Linear(sensor_dim, sensor_hidden), nn.ReLU())
        self.image_projection = None
        self.sensor_projection = None
        self.gate = None
        if fusion == "concat":
            fused_dim = image_dim + sensor_hidden
        else:
            self.image_projection = nn.Sequential(nn.Linear(image_dim, projection_dim), nn.ReLU())
            self.sensor_projection = nn.Sequential(
                nn.Linear(sensor_hidden, projection_dim), nn.ReLU()
            )
            if fusion == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(2 * projection_dim, projection_dim), nn.Sigmoid()
                )
                fused_dim = projection_dim
            else:
                fused_dim = 2 * projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 6),
        )

    def get_model_config(self) -> dict:
        return dict(self._model_config)

    def _representations(self, visual: torch.Tensor, sensors: torch.Tensor):
        if sensors.ndim != 2 or sensors.shape[1] != self.sensor_dim:
            raise ValueError("Sensor dimension differs from the saved model.")
        if not isinstance(visual, torch.Tensor):
            raise ValueError("Visual input is required.")
        if self.encoder == "embedding":
            if visual.ndim != 2 or visual.shape[1] != self.embedding_dim:
                raise ValueError("Visual embedding dimension differs from the saved model.")
        elif visual.ndim != 4 or visual.shape[1] != 3:
            raise ValueError("Image input must have shape [batch, 3, height, width].")
        if len(visual) != len(sensors):
            raise ValueError("Visual and sensor batch sizes differ.")
        image = self.image_encoder(visual)
        sensor = self.sensor_encoder(sensors)
        if self.fusion != "concat":
            image = self.image_projection(image)
            sensor = self.sensor_projection(sensor)
        return image, sensor

    def gate_values(self, visual: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        """Return per-observation gates for diagnostics, not physical importance."""
        if self.gate is None:
            raise ValueError("gate_values is only supported by gated fusion.")
        image, sensor = self._representations(visual, sensors)
        return self.gate(torch.cat((image, sensor), dim=1))

    def forward(self, visual: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        image, sensor = self._representations(visual, sensors)
        if self.gate is None:
            fused = torch.cat((image, sensor), dim=1)
        else:
            gate = self.gate(torch.cat((image, sensor), dim=1))
            fused = gate * image + (1.0 - gate) * sensor
        return self.classifier(fused)
