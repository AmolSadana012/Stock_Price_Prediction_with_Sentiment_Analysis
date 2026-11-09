"""Causal technical indicators and next-day return target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Market_historical_data.data.download import OHLCV_COLUMNS

TARGET_COLUMN = "Daily_Return"


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Calculate trailing indicators and a one-day return target.

    All rolling and exponential indicators use present/past observations only.
    The returned feature rows omit the initial indicator warm-up period.
    """
    missing = sorted(set(OHLCV_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Clean OHLCV data is missing required columns: {missing}")
    result = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    close = result["Close"]

    result[TARGET_COLUMN] = close.pct_change(fill_method=None)
    result["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    result["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    result["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    result["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()

    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    average_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    relative_strength = average_gain / average_loss
    result["RSI_14"] = 100 - 100 / (1 + relative_strength)
    result.loc[(average_loss == 0) & (average_gain > 0), "RSI_14"] = 100.0
    result.loc[(average_loss == 0) & (average_gain == 0), "RSI_14"] = 50.0

    result["MACD"] = result["EMA_12"] - result["EMA_26"]
    result["MACD_Signal"] = result["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]
    result["Volatility_20"] = result[TARGET_COLUMN].rolling(
        window=20, min_periods=20
    ).std()

    feature_columns = [
        *OHLCV_COLUMNS,
        TARGET_COLUMN,
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Volatility_20",
    ]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    if result.empty:
        raise ValueError("Not enough valid observations to calculate all features.")
    return result, feature_columns
