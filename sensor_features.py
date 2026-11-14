"""Explicit sensor features, with optional causal history; no fitting or target inputs."""

import numpy as np
import pandas as pd

from schema import OPTIONAL_SENSOR_COLUMNS, RAW_SENSOR_COLUMNS
from temporal import forward_fill, history_features, ordered_experiments

# Bump this when an existing formula or its missing-value semantics changes.
SENSOR_FEATURE_VERSION = 1
DEFAULT_SENSOR_FEATURES = RAW_SENSOR_COLUMNS
DERIVED_SENSOR_FEATURES = {
    "temp_mean": "(meas_t1 + meas_t2) / 2",
    "temp_diff": "meas_t2 - meas_t1",
    "temp_abs_diff": "abs(meas_t2 - meas_t1)",
}
TEMPERATURE_SENSOR_FEATURES = (*RAW_SENSOR_COLUMNS, *DERIVED_SENSOR_FEATURES)
HISTORY_SIGNALS = ("meas_v", "meas_a", "meas_p")
HISTORY_FEATURE_NAMES = tuple(
    f"{signal}_{stat}30s" for signal in HISTORY_SIGNALS for stat in ("mean", "delta")
)
NONTHERMAL_HISTORY_FEATURES = (*RAW_SENSOR_COLUMNS, *HISTORY_FEATURE_NAMES)
AMBIENT_FEATURE_DEPENDENCIES = {
    "temp1_minus_ambient": ("meas_t1", "atmp_t"),
    "temp2_minus_ambient": ("meas_t2", "atmp_t"),
    "thermal_minus_ambient": ("thermal_t", "atmp_t"),
}
ALL_SENSOR_COLUMNS = (*RAW_SENSOR_COLUMNS, *OPTIONAL_SENSOR_COLUMNS)
MISSING_FEATURE_SOURCES = {f"{name}_missing": name for name in ALL_SENSOR_COLUMNS}
EXTENDED_SENSOR_FEATURES = (
    *OPTIONAL_SENSOR_COLUMNS,
    *AMBIENT_FEATURE_DEPENDENCIES,
    *MISSING_FEATURE_SOURCES,
)


def validate_imputation(imputation: str) -> str:
    if imputation not in ("median", "forward_fill"):
        raise ValueError("sensor_imputation must be median or forward_fill.")
    return imputation


def uses_history(feature_names, imputation="median") -> bool:
    return imputation == "forward_fill" or bool(set(feature_names) & set(HISTORY_FEATURE_NAMES))


def validate_feature_names(names) -> tuple[str, ...]:
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("sensor_features must be a nonempty list of feature names.")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("Each sensor feature name must be a string.")
    if len(set(names)) != len(names):
        raise ValueError("sensor_features must not contain duplicates.")
    allowed = (
        set(RAW_SENSOR_COLUMNS)
        | set(DERIVED_SENSOR_FEATURES)
        | set(HISTORY_FEATURE_NAMES)
        | set(EXTENDED_SENSOR_FEATURES)
    )
    if set(names) - allowed:
        raise ValueError("Unknown sensor feature. Register an explicit causal formula first.")
    return tuple(names)


def required_sensor_columns(feature_names) -> tuple[str, ...]:
    """Preserve the five-column baseline contract; add only explicitly needed columns."""
    names = validate_feature_names(feature_names)
    required = set(RAW_SENSOR_COLUMNS)
    for name in names:
        if name in ALL_SENSOR_COLUMNS:
            required.add(name)
        elif name in AMBIENT_FEATURE_DEPENDENCIES:
            required.update(AMBIENT_FEATURE_DEPENDENCIES[name])
        elif name in MISSING_FEATURE_SOURCES:
            required.add(MISSING_FEATURE_SOURCES[name])
    return tuple(name for name in ALL_SENSOR_COLUMNS if name in required)


def build_sensor_features(
    frame: pd.DataFrame, feature_names=DEFAULT_SENSOR_FEATURES, *, imputation="median"
) -> pd.DataFrame:
    """Build in original row order; optional raw forward fill precedes temperature formulas."""
    names = validate_feature_names(feature_names)
    validate_imputation(imputation)
    columns = required_sensor_columns(names)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing raw sensor columns for selected features: {', '.join(sorted(missing))}"
        )
    raw = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    raw = raw.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    groups = ordered_experiments(frame) if uses_history(names, imputation) else []
    current = forward_fill(raw, groups) if imputation == "forward_fill" else raw
    features = {name: current[name] for name in columns}
    # Missingness describes the observation BEFORE any forward fill or median replacement.
    for name in names:
        if name in MISSING_FEATURE_SOURCES:
            features[name] = raw[MISSING_FEATURE_SOURCES[name]].isna().astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        # Halving before addition avoids avoidable overflow for very large finite values.
        features["temp_mean"] = current["meas_t1"] / 2 + current["meas_t2"] / 2
        features["temp_diff"] = current["meas_t2"] - current["meas_t1"]
        features["temp_abs_diff"] = features["temp_diff"].abs()
        for name, (sensor, ambient) in AMBIENT_FEATURE_DEPENDENCIES.items():
            if name in names:
                features[name] = current[sensor] - current[ambient]
        signals = [
            s
            for s in HISTORY_SIGNALS
            if any(f"{s}_{stat}30s" in names for stat in ("mean", "delta"))
        ]
        if signals:
            features.update(history_features(raw, current, signals, groups))
    result = pd.DataFrame({name: features[name] for name in names}, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)
