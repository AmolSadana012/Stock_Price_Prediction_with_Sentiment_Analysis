"""Validation and cleaning for downloaded OHLCV time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Market_historical_data.data.download import OHLCV_COLUMNS


def clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, unique, internally consistent OHLCV rows.

    Rows with missing/non-finite values, non-positive prices, negative volume,
    or impossible high/low relationships are excluded rather than imputed.
    For duplicate dates, the last source row is retained.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = sorted(set(OHLCV_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {missing}")

    result = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result.index.name = "Date"
    result = result.loc[result.index.notna()].sort_index(kind="stable")
    result = result.loc[~result.index.duplicated(keep="last")]

    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=list(OHLCV_COLUMNS))

    prices = ("Open", "High", "Low", "Close", "Adj Close")
    valid_prices = (result.loc[:, prices] > 0).all(axis=1)
    valid_volume = result["Volume"] >= 0
    valid_range = (
        (result["High"] >= result["Low"])
        & (result["High"] >= result[["Open", "Close"]].max(axis=1))
        & (result["Low"] <= result[["Open", "Close"]].min(axis=1))
    )
    result = result.loc[valid_prices & valid_volume & valid_range]

    if result.empty:
        raise ValueError("No valid OHLCV rows remain after cleaning.")
    if not result.index.is_monotonic_increasing or result.index.has_duplicates:
        raise RuntimeError("OHLCV cleaning failed to produce a unique chronological index.")
    return result
