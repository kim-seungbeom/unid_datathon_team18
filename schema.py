"""Shared column contracts. No data paths are discovered automatically."""

from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLUMNS = ("meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p")
LABELS = (1, 2, 3, 4, 5, 6)


def read_frame(path: str | Path, *, labeled: bool = False) -> pd.DataFrame:
    # Preserve IDs such as 00001 and NA exactly as strings.
    frame = pd.read_csv(
        path,
        dtype={"sample_id": str, "image_path": str, "experiment_id": str},
        keep_default_na=False,
    )
    required = {"sample_id", "image_path", *SENSOR_COLUMNS}
    if labeled:
        required.add("label")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input CSV must contain at least one row.")
    for column in ("sample_id", "image_path"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Column {column} contains empty values.")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique within the input CSV.")
    if labeled:
        frame["label"] = validate_labels(frame["label"])
    return frame.reset_index(drop=True)


def validate_labels(values) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    if not np.isin(numeric, LABELS).all():
        raise ValueError("Labels must be integers in 1..6, without missing values.")
    return numeric.astype(np.int64)


def validate_split(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("Training and validation sample IDs overlap.")
    for frame in (train, validation):
        if "experiment_id" not in frame or frame["experiment_id"].str.strip().eq("").any():
            raise ValueError("Training and validation require nonempty experiment_id values.")
    shared = set(train["experiment_id"]) & set(validation["experiment_id"])
    if shared:
        raise ValueError("Training and validation experiment IDs overlap. Split by experiment.")
