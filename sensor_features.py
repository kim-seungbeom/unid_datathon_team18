"""Explicit, row-local sensor features. No fitting, labels, IDs, images, or history."""

import numpy as np
import pandas as pd

from schema import RAW_SENSOR_COLUMNS

# Bump this when an existing formula or its missing-value semantics changes.
SENSOR_FEATURE_VERSION = 1
DEFAULT_SENSOR_FEATURES = RAW_SENSOR_COLUMNS
DERIVED_SENSOR_FEATURES = {
    "temp_mean": "(meas_t1 + meas_t2) / 2",
    "temp_diff": "meas_t2 - meas_t1",
    "temp_abs_diff": "abs(meas_t2 - meas_t1)",
}
TEMPERATURE_SENSOR_FEATURES = (*RAW_SENSOR_COLUMNS, *DERIVED_SENSOR_FEATURES)


def validate_feature_names(names) -> tuple[str, ...]:
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("sensor_features must be a nonempty list of feature names.")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("Each sensor feature name must be a string.")
    if len(set(names)) != len(names):
        raise ValueError("sensor_features must not contain duplicates.")
    allowed = set(RAW_SENSOR_COLUMNS) | set(DERIVED_SENSOR_FEATURES)
    if set(names) - allowed:
        raise ValueError("Unknown sensor feature. Register a current-row formula first.")
    return tuple(names)


def build_sensor_features(
    frame: pd.DataFrame, feature_names=DEFAULT_SENSOR_FEATURES
) -> pd.DataFrame:
    """Compute before imputation: either missing temperature makes its derived feature missing."""
    names = validate_feature_names(feature_names)
    missing = set(RAW_SENSOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing raw sensor columns required by the feature builder.")
    raw = frame.loc[:, list(RAW_SENSOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    raw = raw.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    features = {name: raw[name] for name in RAW_SENSOR_COLUMNS}
    with np.errstate(over="ignore", invalid="ignore"):
        # Halving before addition avoids avoidable overflow for very large finite values.
        features["temp_mean"] = raw["meas_t1"] / 2 + raw["meas_t2"] / 2
        features["temp_diff"] = raw["meas_t2"] - raw["meas_t1"]
        features["temp_abs_diff"] = features["temp_diff"].abs()
    result = pd.DataFrame({name: features[name] for name in names}, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)
