"""Explicit sensor features, with optional causal history; no fitting or target inputs."""

import numpy as np
import pandas as pd

from schema import RAW_SENSOR_COLUMNS
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
    allowed = set(RAW_SENSOR_COLUMNS) | set(DERIVED_SENSOR_FEATURES) | set(HISTORY_FEATURE_NAMES)
    if set(names) - allowed:
        raise ValueError("Unknown sensor feature. Register an explicit causal formula first.")
    return tuple(names)


def build_sensor_features(
    frame: pd.DataFrame, feature_names=DEFAULT_SENSOR_FEATURES, *, imputation="median"
) -> pd.DataFrame:
    """Build in original row order; optional raw forward fill precedes temperature formulas."""
    names = validate_feature_names(feature_names)
    validate_imputation(imputation)
    missing = set(RAW_SENSOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing raw sensor columns required by the feature builder.")
    raw = frame.loc[:, list(RAW_SENSOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    raw = raw.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    groups = ordered_experiments(frame) if uses_history(names, imputation) else []
    current = forward_fill(raw, groups) if imputation == "forward_fill" else raw
    features = {name: current[name] for name in RAW_SENSOR_COLUMNS}
    with np.errstate(over="ignore", invalid="ignore"):
        # Halving before addition avoids avoidable overflow for very large finite values.
        features["temp_mean"] = current["meas_t1"] / 2 + current["meas_t2"] / 2
        features["temp_diff"] = current["meas_t2"] - current["meas_t1"]
        features["temp_abs_diff"] = features["temp_diff"].abs()
        signals = [s for s in HISTORY_SIGNALS if any(n.startswith(s + "_") for n in names)]
        if signals:
            features.update(history_features(raw, current, signals, groups))
    result = pd.DataFrame({name: features[name] for name in names}, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)
