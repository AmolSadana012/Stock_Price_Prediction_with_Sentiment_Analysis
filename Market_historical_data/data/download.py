"""Download historical OHLCV data from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TICKERS = ("AAPL", "MSFT", "AMZN")
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"  # Yahoo Finance treats end as exclusive.
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def download_history(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Fetch one ticker's unadjusted OHLCV history, including adjusted close."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required to download market data. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc

    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker} ({start} to {end}).")

    if isinstance(frame.columns, pd.MultiIndex):
        ticker_level = next(
            (
                level
                for level in range(frame.columns.nlevels)
                if ticker in frame.columns.get_level_values(level)
            ),
            None,
        )
        if ticker_level is not None:
            frame = frame.xs(ticker, axis=1, level=ticker_level, drop_level=True)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

    missing = sorted(set(OHLCV_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            f"Yahoo Finance data for {ticker} is missing required columns: {missing}"
        )

    frame = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame.index.name = "Date"
    frame = frame.loc[frame.index.notna()]
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no valid dates for {ticker}.")
    return frame


def download_all(
    output_dir: Path,
    tickers: tuple[str, ...] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """Download each ticker and persist its raw response as a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = download_history(ticker, start=start, end=end)
        frame.to_csv(output_dir / f"{ticker}_raw.csv", index_label="Date")
        result[ticker] = frame
    return result
