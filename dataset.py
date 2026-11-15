"""One observation per item: current image plus current sensor readings."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing import SensorPreprocessor


class BatteryDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        preprocessor: SensorPreprocessor,
        image_size: int,
        *,
        labeled: bool = False,
        use_images: bool = True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root).expanduser().resolve()
        self.sensors = preprocessor.transform(self.frame)
        self.image_size = image_size
        self.labeled = labeled
        self.use_images = use_images

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        item = {"sensors": torch.from_numpy(self.sensors[index])}
        if self.use_images:
            item["images"] = self._load_image(row)
        if self.labeled:
            item["targets"] = torch.tensor(int(row["label"]) - 1, dtype=torch.long)
        return item

    def _load_image(self, row) -> torch.Tensor:
        image_path = (self.image_root / str(row["image_path"])).resolve()
        if not image_path.is_relative_to(self.image_root):
            raise ValueError("An image_path resolves outside --image_root.")
        with Image.open(image_path) as source:
            if source.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                raise ValueError(
                    "The starter loader supports 8-bit images only. "
                    "Define a documented, fixed conversion for other thermal image formats."
                )
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values.transpose(2, 0, 1).copy())
