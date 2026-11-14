"""Small multimodal starter; no external data or pretrained weights."""

import torch
from torch import nn

from schema import LABELS, SENSOR_COLUMNS


class FusionClassifier(nn.Module):
    def __init__(self, sensor_hidden: int = 32, fusion_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # No batch-dependent normalization: each observation has independent features.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(len(SENSOR_COLUMNS), sensor_hidden), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + sensor_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, len(LABELS)),
        )

    def forward(self, images: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images).mean(dim=(-2, -1))
        features = torch.cat((image_features, self.sensor_encoder(sensors)), dim=1)
        return self.classifier(features)
