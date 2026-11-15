"""Personal neural sensor extensions; shared sensor formulas/checkpoints stay unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import SensorPreprocessor
from schema import RAW_SENSOR_COLUMNS
from sensor_features import (
    SENSOR_FEATURE_VERSION,
    build_sensor_features,
    uses_history,
    validate_feature_names,
)
from temporal import TEMPORAL_VERSION, ordered_experiments

FORMULA_VERSION = 1
WINDOW_SECONDS = 30.0
SLOPE_SOURCES = {f"{name}_slope30s": name for name in ("meas_t1", "meas_t2")}
VOLTAGE_FEATURES = ("meas_v_mean30s", "meas_v_delta30s")
PRESSURE_FEATURES = ("meas_p_mean30s", "meas_p_delta30s")
TEMPORAL_VARIANTS = {
    "raw": tuple(RAW_SENSOR_COLUMNS),
    "voltage30": (*RAW_SENSOR_COLUMNS, *VOLTAGE_FEATURES),
    "pressure30": (*RAW_SENSOR_COLUMNS, *PRESSURE_FEATURES),
    "temperature30": (*RAW_SENSOR_COLUMNS, *SLOPE_SOURCES),
    "combined30": (*RAW_SENSOR_COLUMNS, *VOLTAGE_FEATURES, *PRESSURE_FEATURES, *SLOPE_SOURCES),
}
AMBIENT_VARIANTS = {
    "raw": tuple(RAW_SENSOR_COLUMNS),
    "ambient": (*RAW_SENSOR_COLUMNS, "atmp_t"),
    "ambient_delta": (
        *RAW_SENSOR_COLUMNS,
        "atmp_t",
        "temp1_minus_ambient",
        "temp2_minus_ambient",
    ),
}


def validate_neural_features(names):
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("sensor_features must be a nonempty list.")
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("Sensor feature names must be unique strings.")
    # Validate the ordinary features with the shared allowlist. Private extensions
    # never register names in the shared module or replace an existing formula.
    ordinary = [name for name in names if name not in SLOPE_SOURCES]
    if ordinary:
        validate_feature_names(ordinary)
    return tuple(names)


def uses_neural_history(names, imputation="median"):
    return uses_history(names, imputation) or bool(set(names) & SLOPE_SOURCES.keys())


def sensor_variant(names):
    return next(
        (
            name
            for name, features in {**TEMPORAL_VARIANTS, **AMBIENT_VARIANTS}.items()
            if tuple(names) == features
        ),
        "custom",
    )


def neural_sensor_values(frame, feature_names):
    """Current/raw features plus an observed-only OLS temperature slope in (t-30,t]."""
    names = validate_neural_features(feature_names)
    ordinary = [name for name in names if name not in SLOPE_SOURCES]
    base = build_sensor_features(frame, ordinary or RAW_SENSOR_COLUMNS)
    features = {name: base[name].to_numpy(dtype=np.float64) for name in ordinary}
    slopes = [name for name in names if name in SLOPE_SOURCES]
    if slopes:
        groups = ordered_experiments(frame)
        for name in slopes:
            raw = pd.to_numeric(frame[SLOPE_SOURCES[name]], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            result = np.full(len(frame), np.nan)
            for positions, times in groups:
                observations = raw[positions]
                starts = np.searchsorted(times, times - WINDOW_SECONDS, side="right")
                for end, start in enumerate(starts):
                    values = observations[start : end + 1]
                    valid = np.isfinite(values)
                    if valid.sum() < 2:
                        continue
                    # Center on this window, never the experiment's full time range.
                    x = times[start : end + 1][valid]
                    x = x - x[-1]
                    x = x - x.mean()
                    y = values[valid]
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        slope = np.dot(x, y - y.mean()) / np.dot(x, x)
                    if np.isfinite(slope):
                        result[positions[end]] = slope
            features[name] = result
    result = np.column_stack([features[name] for name in names])
    result[~np.isfinite(result)] = np.nan
    return result


class NeuralSensorPreprocessor(SensorPreprocessor):
    """Use the old codec unchanged unless private slope features are selected."""

    @property
    def uses_history(self):
        return uses_neural_history(self.feature_names, self.imputation)

    @classmethod
    def fit(
        cls,
        train_frame,
        feature_names=RAW_SENSOR_COLUMNS,
        *,
        imputation="median",
        scaling="standard",
    ):
        names = validate_neural_features(feature_names)
        if not set(names) & SLOPE_SOURCES.keys():
            return SensorPreprocessor.fit(
                train_frame, names, imputation=imputation, scaling=scaling
            )
        if imputation != "median" or scaling != "standard":
            raise ValueError(
                "Private slope experiments use median imputation and standard scaling."
            )
        values = neural_sensor_values(train_frame, names)
        if not len(values):
            raise ValueError("Cannot fit preprocessing on an empty training split.")
        medians = np.asarray(
            [
                np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                for col in values.T
            ]
        )
        filled = np.where(np.isnan(values), medians, values)
        means, scales = filled.mean(axis=0), filled.std(axis=0)
        scales[scales < 1e-8] = 1.0
        flags = np.array([name.endswith("_missing") for name in names])
        means[flags], scales[flags] = 0.0, 1.0
        if not all(np.isfinite(a).all() for a in (medians, means, scales)):
            raise ValueError("Training preprocessing statistics must be finite.")
        return cls(medians, means, scales, names, imputation, scaling)

    def impute(self, frame):
        values = neural_sensor_values(frame, self.feature_names)
        return np.where(np.isnan(values), self.medians, values)

    def to_dict(self):
        return {
            "kind": "seungbeom_neural_temporal_sensors",
            "format_version": 1,
            "formula_version": FORMULA_VERSION,
            "sensor_feature_version": SENSOR_FEATURE_VERSION,
            "temporal_version": TEMPORAL_VERSION,
            "feature_names": list(self.feature_names),
            "imputation": self.imputation,
            "scaling": self.scaling,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, state):
        if state.get("kind") != "seungbeom_neural_temporal_sensors":
            return SensorPreprocessor.from_dict(state)
        if (
            state.get("format_version") != 1
            or state.get("formula_version") != FORMULA_VERSION
            or state.get("sensor_feature_version") != SENSOR_FEATURE_VERSION
            or state.get("temporal_version") != TEMPORAL_VERSION
        ):
            raise ValueError("Unsupported private temporal preprocessing version.")
        names = validate_neural_features(state["feature_names"])
        if not set(names) & SLOPE_SOURCES.keys():
            raise ValueError("Private preprocessing state requires a private slope feature.")
        if state.get("imputation") != "median" or state.get("scaling") != "standard":
            raise ValueError("Unsupported private temporal preprocessing settings.")
        arrays = [
            np.asarray(state[key], dtype=np.float64) for key in ("medians", "means", "scales")
        ]
        if (
            any(a.shape != (len(names),) or not np.isfinite(a).all() for a in arrays)
            or (arrays[2] <= 0).any()
        ):
            raise ValueError("Invalid saved private preprocessing statistics.")
        flags = np.array([name.endswith("_missing") for name in names])
        if (arrays[1][flags] != 0).any() or (arrays[2][flags] != 1).any():
            raise ValueError("Saved missing indicators require center 0 and scale 1.")
        return cls(*arrays, feature_names=names)
