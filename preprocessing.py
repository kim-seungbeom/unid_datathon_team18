"""Training-only median imputation and standardization for the five sensors."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from schema import SENSOR_COLUMNS


def sensor_values(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.loc[:, list(SENSOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass
class SensorPreprocessor:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, train_frame: pd.DataFrame) -> "SensorPreprocessor":
        values = sensor_values(train_frame)
        if not len(values):
            raise ValueError("Cannot fit preprocessing on an empty training split.")
        medians = np.array(
            [
                np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                for col in values.T
            ]
        )
        filled = np.where(np.isnan(values), medians, values)
        scales = filled.std(axis=0)
        scales[scales < 1e-8] = 1.0
        return cls(medians, filled.mean(axis=0), scales)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = sensor_values(frame)
        filled = np.where(np.isnan(values), self.medians, values)
        transformed = ((filled - self.means) / self.scales).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Sensor preprocessing produced nonfinite values.")
        return transformed

    def to_dict(self) -> dict:
        return {
            "sensor_columns": list(SENSOR_COLUMNS),
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "SensorPreprocessor":
        if state["sensor_columns"] != list(SENSOR_COLUMNS):
            raise ValueError("Saved sensor column order does not match this code.")
        arrays = [
            np.asarray(state[key], dtype=np.float64) for key in ("medians", "means", "scales")
        ]
        if any(a.shape != (len(SENSOR_COLUMNS),) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid saved preprocessing state.")
        if (arrays[2] <= 0).any():
            raise ValueError("Saved scales must be positive.")
        return cls(*arrays)
