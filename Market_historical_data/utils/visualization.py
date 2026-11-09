"""Save data and model diagnostic visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Market_historical_data.models.training import TrainingResult
from Market_historical_data.preprocessing.splitting import WindowSet


def plot_price_and_indicators(features: pd.DataFrame, path: Path) -> None:
    """Plot adjusted market close and selected causal indicators."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(features.index, features["Close"], label="Close")
    axes[0].plot(features.index, features["SMA_20"], label="SMA 20", alpha=0.8)
    axes[0].plot(features.index, features["SMA_50"], label="SMA 50", alpha=0.8)
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[1].plot(features.index, features["RSI_14"], color="tab:purple")
    axes[1].axhline(70, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("RSI (14)")
    axes[2].plot(features.index, features["MACD"], label="MACD")
    axes[2].plot(features.index, features["MACD_Signal"], label="Signal")
    axes[2].set_ylabel("MACD")
    axes[2].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test_predictions(
    windows: WindowSet, predictions: np.ndarray, path: Path
) -> None:
    """Plot actual versus model-predicted next-day test returns."""
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(windows.dates, windows.y, label="Actual return", linewidth=1)
    axis.plot(windows.dates, predictions, label="Predicted return", linewidth=1)
    axis.set_ylabel("Next-day return")
    axis.set_xlabel("Target date")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_losses(result: TrainingResult, path: Path) -> None:
    """Plot observed training and validation losses."""
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(result.train_losses) + 1)
    axis.plot(epochs, result.train_losses, label="Training MSE")
    axis.plot(epochs, result.validation_losses, label="Validation MSE")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE (return units)")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
