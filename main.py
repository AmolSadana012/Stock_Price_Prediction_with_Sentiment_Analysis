"""Run the historical market-data modelling branch end to end."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Market_historical_data.data.cleaning import clean_ohlcv
from Market_historical_data.data.download import (
    END_DATE,
    START_DATE,
    TICKERS,
    download_history,
)
from Market_historical_data.models.training import (
    TrainingConfig,
    build_model,
    config_dict,
    extract_representations,
    predict,
    predict_direction,
    set_seed,
    train_model,
)
from Market_historical_data.utils.visualization import (
    plot_price_and_indicators,
    plot_test_predictions,
    plot_training_losses,
)
from Market_historical_data.preprocessing.features import engineer_features
from Market_historical_data.preprocessing.splitting import (
    WindowSet,
    create_scaled_windows,
    split_boundaries,
)

LOGGER = logging.getLogger("market_pipeline")
ROOT = Path(__file__).resolve().parent / "Market_historical_data"
MODEL_TYPES = ("lstm", "gru", "hybrid_lstm_gru")


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    predicted_direction: np.ndarray | None = None,
) -> dict[str, float | int]:
    if actual.size != predicted.size or actual.size == 0:
        raise ValueError("Metrics require non-empty, equally sized actual and predicted arrays.")
    errors = predicted - actual
    predicted_up = predicted > 0 if predicted_direction is None else predicted_direction > 0
    return {
        "samples": int(actual.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "directional_accuracy": float(np.mean((actual > 0) == predicted_up)),
    }


def _representation_frame(
    ticker: str,
    model_type: str,
    split_windows: dict[str, WindowSet],
    model: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name, windows in split_windows.items():
        representation = extract_representations(model, windows, device)
        direction = predict_direction(model, windows, device)
        columns = [f"market_rep_{index:03d}" for index in range(representation.shape[1])]
        part = pd.DataFrame(representation, columns=columns)
        part.insert(0, "Split", split_name)
        part.insert(0, "Ticker", ticker)
        part.insert(0, "Date", windows.dates.strftime("%Y-%m-%d"))
        part.insert(1, "Model", model_type)
        part["Target_Return"] = windows.y
        part["Predicted_Return"] = predict(model, windows, device)
        part["Actual_Up"] = windows.y > 0
        part["Predicted_Up"] = direction.astype(bool)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _load_or_download_data(
    tickers: tuple[str, ...], raw_dir: Path
) -> dict[str, pd.DataFrame]:
    """Reuse saved Yahoo CSVs and download only tickers without a local cache."""
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}_raw.csv"
        if raw_path.exists():
            try:
                frame = pd.read_csv(raw_path, index_col="Date", parse_dates=["Date"])
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise RuntimeError(f"Could not read cached data at {raw_path}: {exc}") from exc
            data[ticker] = frame
            LOGGER.info("Reusing cached Yahoo Finance data for %s", ticker)
        else:
            frame = download_history(ticker)
            frame.to_csv(raw_path, index_label="Date")
            data[ticker] = frame
    return data


def _save_comparison_metrics(
    metrics: pd.DataFrame, outputs_dir: Path
) -> Path:
    """Add this run's scores to a comparison file without replacing prior runs."""
    comparison_dir = outputs_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / "model_comparison.csv"
    prior: list[pd.DataFrame] = []
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        prior.append(existing[existing["model"].isin(MODEL_TYPES)])

    if prior:
        combined = pd.concat([*prior, metrics], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["model", "ticker", "split"], keep="last"
        )
    else:
        combined = metrics.copy()
    comparison_columns = [
        "ticker",
        "model",
        "split",
        "samples",
        "mae",
        "rmse",
        "directional_accuracy",
    ]
    combined = combined.loc[:, comparison_columns]
    combined = combined.sort_values(["model", "ticker", "split"]).reset_index(drop=True)
    combined.to_csv(comparison_path, index=False)
    test_summary = (
        combined.loc[combined["split"] == "test"]
        .groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values("model")
    )
    test_summary.to_csv(comparison_dir / "final_summary.csv", index=False)
    return comparison_path


def run_pipeline(
    model_type: str = "lstm",
    epochs: int = 30,
    lookback: int = 30,
    seed: int = 42,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    classification_loss_weight: float = 0.1,
    return_loss_scale: float = 0.01,
    tickers: tuple[str, ...] = TICKERS,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Download, process, train, evaluate, and save market representations."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if lookback < 1:
        raise ValueError("lookback must be a positive integer.")
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    outputs_dir = root / "outputs"
    model_outputs_dir = outputs_dir / model_type
    model_dir = model_outputs_dir / "models"
    plot_dir = model_outputs_dir / "plots"
    prediction_dir = model_outputs_dir / "predictions"
    representation_dir = model_outputs_dir / "representations"
    metadata_dir = model_outputs_dir / "metadata"
    output_directories = [
        raw_dir,
        processed_dir,
        model_dir,
        plot_dir,
        prediction_dir,
        representation_dir,
        metadata_dir,
    ]
    loss_dir = model_outputs_dir / "losses"
    if model_type == "hybrid_lstm_gru":
        output_directories.append(loss_dir)
    for directory in output_directories:
        directory.mkdir(parents=True, exist_ok=True)

    downloaded = _load_or_download_data(tickers, raw_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using %s for model training", device)
    representation_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(tickers):
        LOGGER.info("Preparing %s", ticker)
        cleaned = clean_ohlcv(downloaded[ticker])
        clean_path = processed_dir / f"{ticker}_clean.csv"
        if not clean_path.exists():
            cleaned.to_csv(clean_path, index_label="Date")
        features, feature_columns = engineer_features(cleaned)
        features_path = processed_dir / f"{ticker}_features.csv"
        if not features_path.exists():
            features.to_csv(features_path, index_label="Date")

        train_end, validation_end = split_boundaries(len(features))
        split_windows_scaler = create_scaled_windows(
            features, feature_columns, lookback=lookback
        )
        scaler, windows = split_windows_scaler
        config = TrainingConfig(
            model_type=model_type,
            epochs=epochs,
            seed=seed + ticker_index,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            classification_loss_weight=classification_loss_weight,
            return_loss_scale=return_loss_scale,
        )
        set_seed(config.seed)
        model = build_model(len(feature_columns), config)
        result = train_model(
            model, windows["train"], windows["validation"], config, device
        )

        predictions_by_split = {
            split_name: predict(result.model, split_data, device)
            for split_name, split_data in windows.items()
        }
        directions_by_split = {
            split_name: predict_direction(result.model, split_data, device)
            for split_name, split_data in windows.items()
        }
        for split_name, split_data in windows.items():
            split_predictions = predictions_by_split[split_name]
            split_directions = directions_by_split[split_name]
            row: dict[str, object] = {
                "ticker": ticker,
                "model": model_type,
                "split": split_name,
                **_metrics(split_data.y, split_predictions, split_directions),
            }
            metrics_rows.append(row)
            pd.DataFrame(
                {
                    "Date": split_data.dates.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Model": model_type,
                    "Split": split_name,
                    "Target_Return": split_data.y,
                    "Predicted_Return": split_predictions,
                    "Actual_Up": split_data.y > 0,
                    "Predicted_Up": split_directions.astype(bool),
                }
            ).to_csv(
                prediction_dir / f"{ticker}_{split_name}_predictions.csv",
                index=False,
            )

        checkpoint = {
            "ticker": ticker,
            "model_type": model_type,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in result.model.state_dict().items()
            },
            "training_config": config_dict(config),
            "feature_columns": feature_columns,
            "target_column": "Daily_Return",
            "lookback": lookback,
            "scaler_data_min": scaler.data_min_.tolist(),
            "scaler_data_max": scaler.data_max_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scaler_min": scaler.min_.tolist(),
            "best_epoch": result.best_epoch,
        }
        torch.save(checkpoint, model_dir / f"{ticker}.pt")
        plot_price_and_indicators(features, plot_dir / f"{ticker}_indicators.png")
        plot_test_predictions(
            windows["test"],
            predictions_by_split["test"],
            plot_dir / f"{ticker}_test_predictions.png",
        )
        plot_training_losses(result, plot_dir / f"{ticker}_training_loss.png")
        if model_type == "hybrid_lstm_gru":
            pd.DataFrame(
                {
                    "epoch": np.arange(1, len(result.train_losses) + 1),
                    "train_combined_loss": result.train_losses,
                    "validation_combined_loss": result.validation_losses,
                }
            ).to_csv(loss_dir / f"{ticker}.csv", index=False)

        metadata = {
            "ticker": ticker,
            "requested_period": {"start": START_DATE, "end_exclusive": END_DATE},
            "observations_after_feature_warmup": len(features),
            "split_row_boundaries": {
                "train_end_exclusive": train_end,
                "validation_end_exclusive": validation_end,
            },
            "split_target_samples": {
                split_name: len(split_data.y) for split_name, split_data in windows.items()
            },
            "feature_columns": feature_columns,
            "lookback_days": lookback,
            "best_epoch": result.best_epoch,
            "device": str(device),
        }
        metadata["training_config"] = config_dict(config)
        with (metadata_dir / f"{ticker}_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        representation_frames.append(
            _representation_frame(ticker, model_type, windows, result.model, device)
        )
        LOGGER.info("%s completed; best epoch %d", ticker, result.best_epoch)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(model_outputs_dir / "metrics.csv", index=False)
    representations = pd.concat(representation_frames, ignore_index=True)
    representations.to_csv(
        representation_dir / f"{model_type}_market_representations.csv", index=False
    )
    comparison_path = _save_comparison_metrics(metrics, outputs_dir)
    LOGGER.info(
        "Saved %s results under %s; comparison metrics are in %s",
        model_type,
        outputs_dir,
        comparison_path,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LSTM, GRU, or hybrid LSTM-GRU market encoder."
    )
    parser.add_argument("--model", choices=MODEL_TYPES, default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--classification-loss-weight", type=float, default=0.1)
    parser.add_argument("--return-loss-scale", type=float, default=0.01)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        lookback=args.lookback,
        seed=args.seed,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        classification_loss_weight=args.classification_loss_weight,
        return_loss_scale=args.return_loss_scale,
    )


if __name__ == "__main__":
    main()
