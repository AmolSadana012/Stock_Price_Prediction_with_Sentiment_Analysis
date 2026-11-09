"""Chronological splitting and leakage-safe scaling/window generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Market_historical_data.preprocessing.features import TARGET_COLUMN


@dataclass(frozen=True)
class WindowSet:
    """Model inputs, raw-return targets, and their target dates."""

    x: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex


def split_boundaries(row_count: int) -> tuple[int, int]:
    """Return exclusive chronological boundaries for a 70:15:15 split."""
    if row_count < 3:
        raise ValueError("At least three feature rows are required for chronological splitting.")
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    if train_end < 1 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError(f"Cannot create non-empty 70:15:15 splits from {row_count} rows.")
    return train_end, validation_end


def create_scaled_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    lookback: int = 30,
    target_column: str = TARGET_COLUMN,
) -> tuple[MinMaxScaler, dict[str, WindowSet]]:
    """Fit the scaler on training rows only, then create target-date splits.

    Validation/test windows may include earlier observations from the preceding
    chronological period, as they would be available when forecasting forward.
    Targets remain in return units and are never included in the scaler fit.
    """
    if lookback < 1:
        raise ValueError("lookback must be a positive integer.")
    missing = sorted(set(feature_columns + [target_column]).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature frame is missing columns: {missing}")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Feature frame must have a unique chronological index.")

    train_end, validation_end = split_boundaries(len(frame))
    scaler = MinMaxScaler()
    scaler.fit(frame.iloc[:train_end][feature_columns])
    scaled = scaler.transform(frame[feature_columns]).astype(np.float32)
    targets = frame[target_column].to_numpy(dtype=np.float32)
    date_values = pd.DatetimeIndex(frame.index)

    result: dict[str, WindowSet] = {}
    periods = {
        "train": (lookback, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(frame)),
    }
    for name, (first_target, end_target) in periods.items():
        target_indices = np.arange(max(first_target, lookback), end_target)
        if target_indices.size == 0:
            raise ValueError(
                f"The {name} split has no samples for a {lookback}-day lookback."
            )
        windows = np.stack(
            [scaled[index - lookback : index] for index in target_indices]
        )
        result[name] = WindowSet(
            x=windows,
            y=targets[target_indices],
            dates=date_values[target_indices],
        )
    return scaler, result
