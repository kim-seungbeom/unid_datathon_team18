"""Training-only imputation/scaling of explicit, versioned sensor features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from schema import RAW_SENSOR_COLUMNS
from sensor_features import (
    DEFAULT_SENSOR_FEATURES,
    EXTENDED_SENSOR_FEATURES,
    SENSOR_FEATURE_VERSION,
    build_sensor_features,
    uses_history,
    validate_feature_names,
    validate_imputation,
)
from temporal import TEMPORAL_VERSION


def validate_scaling(scaling: str) -> str:
    if scaling not in ("standard", "robust"):
        raise ValueError("sensor_scaling must be standard or robust.")
    return scaling


def sensor_values(
    frame: pd.DataFrame, feature_names=DEFAULT_SENSOR_FEATURES, *, imputation="median"
) -> np.ndarray:
    numeric = build_sensor_features(frame, feature_names, imputation=imputation)
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass
class SensorPreprocessor:
    medians: np.ndarray
    # Legacy storage name: standard uses means; robust stores median centers here.
    means: np.ndarray
    scales: np.ndarray
    feature_names: tuple[str, ...] = DEFAULT_SENSOR_FEATURES
    imputation: str = "median"
    scaling: str = "standard"

    @property
    def centers(self) -> np.ndarray:
        return self.means

    @property
    def uses_history(self) -> bool:
        return uses_history(self.feature_names, self.imputation)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        feature_names=DEFAULT_SENSOR_FEATURES,
        *,
        imputation="median",
        scaling="standard",
    ) -> "SensorPreprocessor":
        names = validate_feature_names(feature_names)
        validate_imputation(imputation)
        validate_scaling(scaling)
        # Fallback medians use observed Train features, not replicated forward-filled values.
        values = sensor_values(train_frame, names)
        if not len(values):
            raise ValueError("Cannot fit preprocessing on an empty training split.")
        medians = np.array(
            [
                np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                for col in values.T
            ]
        )
        causal_values = sensor_values(train_frame, names, imputation=imputation)
        filled = np.where(np.isnan(causal_values), medians, causal_values)
        if scaling == "robust":
            lower, means, upper = np.quantile(filled, [0.25, 0.5, 0.75], axis=0, method="linear")
            scales = upper - lower
        else:
            scales = filled.std(axis=0)
            means = filled.mean(axis=0)
        # Project convention for either scale: tiny spread becomes 1, with no clipping.
        scales[scales < 1e-8] = 1.0
        # Missingness is an explicit binary signal, even if Train never had missing values.
        flags = np.array([name.endswith("_missing") for name in names])
        means[flags] = 0.0
        scales[flags] = 1.0
        if not all(np.isfinite(array).all() for array in (medians, means, scales)):
            raise ValueError("Training preprocessing statistics must be finite.")
        return cls(medians, means, scales, names, imputation, scaling)

    def impute(self, frame: pd.DataFrame) -> np.ndarray:
        values = sensor_values(frame, self.feature_names, imputation=self.imputation)
        return np.where(np.isnan(values), self.medians, values)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        filled = self.impute(frame)
        transformed = ((filled - self.means) / self.scales).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Sensor preprocessing produced nonfinite values.")
        return transformed

    def to_dict(self) -> dict:
        return {
            "format_version": 4,
            "feature_version": SENSOR_FEATURE_VERSION,
            "raw_sensor_columns": list(RAW_SENSOR_COLUMNS),
            "feature_names": list(self.feature_names),
            "imputation": self.imputation,
            "scaling": self.scaling,
            "temporal_version": TEMPORAL_VERSION if self.uses_history else None,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "SensorPreprocessor":
        version = state.get("format_version", 1)
        if version == 1:
            # Legacy checkpoints used exactly the five raw sensors, in this order.
            if "feature_names" in state or "feature_version" in state:
                raise ValueError("Feature metadata requires preprocessing format_version 2.")
            raw_columns = state["sensor_columns"]
            names = DEFAULT_SENSOR_FEATURES
        elif version in (2, 3, 4):
            if state["feature_version"] != SENSOR_FEATURE_VERSION:
                raise ValueError("Saved sensor feature version does not match this code.")
            raw_columns = state["raw_sensor_columns"]
            names = validate_feature_names(state["feature_names"])
        else:
            raise ValueError("Unsupported preprocessing format version.")
        imputation = validate_imputation(state["imputation"]) if version >= 3 else "median"
        scaling = validate_scaling(state["scaling"]) if version >= 4 else "standard"
        if version < 4 and state.get("scaling", "standard") != "standard":
            raise ValueError("Nonstandard scaling requires preprocessing format_version 4.")
        if version < 4 and set(names) & set(EXTENDED_SENSOR_FEATURES):
            raise ValueError("Extended sensor features require preprocessing format_version 4.")
        if version < 3 and uses_history(names, imputation):
            raise ValueError("Temporal features require preprocessing format_version 3.")
        if version >= 3:
            expected_temporal = TEMPORAL_VERSION if uses_history(names, imputation) else None
            if state.get("temporal_version") != expected_temporal:
                raise ValueError("Saved temporal version does not match this code.")
        if raw_columns != list(RAW_SENSOR_COLUMNS):
            raise ValueError("Saved sensor column order does not match this code.")
        arrays = [
            np.asarray(state[key], dtype=np.float64) for key in ("medians", "means", "scales")
        ]
        if any(a.shape != (len(names),) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid saved preprocessing state.")
        if (arrays[2] <= 0).any():
            raise ValueError("Saved scales must be positive.")
        flags = np.array([name.endswith("_missing") for name in names])
        if (arrays[1][flags] != 0).any() or (arrays[2][flags] != 1).any():
            raise ValueError("Saved missing indicators require center 0 and scale 1.")
        return cls(*arrays, feature_names=names, imputation=imputation, scaling=scaling)
