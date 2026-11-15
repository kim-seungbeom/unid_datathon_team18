"""Causal helpers scoped to the rows supplied by the caller, within each experiment."""

import numpy as np
import pandas as pd

TEMPORAL_VERSION = 1
WINDOW_SECONDS = 30.0


def ordered_experiments(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    required = {"experiment_id", "relative_time_s"}
    if not required.issubset(frame.columns):
        raise ValueError("History requires experiment_id and relative_time_s.")
    groups = frame["experiment_id"].astype("string").reset_index(drop=True)
    if groups.isna().any() or groups.str.strip().eq("").any():
        raise ValueError("History requires nonempty experiment_id values.")
    times = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("History requires finite, nonnegative relative_time_s.")
    result = []
    for positions in groups.groupby(groups, sort=False).indices.values():
        positions = positions[np.argsort(times[positions], kind="stable")]
        ordered_times = times[positions]
        if (np.diff(ordered_times) == 0).any():
            raise ValueError(
                "Duplicate times within an experiment: resolve the observation contract first."
            )
        result.append((positions, ordered_times))
    return result


def forward_fill(raw: pd.DataFrame, groups) -> pd.DataFrame:
    """Carry each sensor's last observed finite value forward, never backward or across groups."""
    values = raw.to_numpy(dtype=float, copy=True)
    for positions, _ in groups:
        values[positions] = raw.iloc[positions].ffill().to_numpy(dtype=float)
    return pd.DataFrame(values, index=raw.index, columns=raw.columns)


def history_features(raw: pd.DataFrame, current: pd.DataFrame, signals, groups) -> dict:
    """Mean of observed values in (t-30,t]; delta to last finite observation at/before t-30."""
    result = {}
    for signal in signals:
        mean = np.full(len(raw), np.nan)
        delta = np.full(len(raw), np.nan)
        for positions, times in groups:
            observed = raw[signal].to_numpy(dtype=float)[positions]
            valid = np.isfinite(observed)
            start = np.searchsorted(times, times - WINDOW_SECONDS, side="right")
            sums = np.r_[0.0, np.cumsum(np.where(valid, observed, 0.0))]
            counts = np.r_[0, np.cumsum(valid)]
            n = counts[1:] - counts[start]
            mean[positions] = np.divide(
                sums[1:] - sums[start], n, out=np.full(len(times), np.nan), where=n > 0
            )
            lag = np.searchsorted(times[valid], times - WINDOW_SECONDS, side="right") - 1
            available = lag >= 0
            delta[positions[available]] = (
                current[signal].to_numpy(dtype=float)[positions[available]]
                - observed[valid][lag[available]]
            )
        result[f"{signal}_mean30s"] = mean
        result[f"{signal}_delta30s"] = delta
    return result
