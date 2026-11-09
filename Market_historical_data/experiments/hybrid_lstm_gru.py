"""Validation-tune and test a parallel LSTM-GRU return/direction model."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Market_historical_data.data.cleaning import clean_ohlcv
from Market_historical_data.data.download import TICKERS
from Market_historical_data.models.training import (
    TrainingConfig,
    build_model,
    extract_representations,
    predict,
    predict_direction,
    set_seed,
    train_model,
)
from Market_historical_data.preprocessing.features import engineer_features
from Market_historical_data.preprocessing.splitting import (
    WindowSet,
    create_scaled_windows,
)

LOGGER = logging.getLogger("hybrid_lstm_gru")
BASELINE_MODELS = ("lstm", "gru")
MODEL_TYPES = (*BASELINE_MODELS, "hybrid_lstm_gru")
CLASSIFICATION_WEIGHTS = (0.1, 0.5, 1.0)
LEARNING_RATES = (1e-3, 5e-4)


def _score(
    actual: np.ndarray,
    predicted: np.ndarray,
    predicted_up: np.ndarray,
) -> dict[str, float | int]:
    if actual.size == 0 or actual.size != predicted.size or actual.size != predicted_up.size:
        raise ValueError("Scores require non-empty, aligned return and direction arrays.")
    errors = predicted - actual
    return {
        "samples": int(actual.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "directional_accuracy": float(
            np.mean((actual > 0) == predicted_up.astype(bool))
        ),
    }


def _read_predictions(root: Path, model: str, ticker: str, split: str) -> pd.DataFrame:
    path = root / "outputs" / model / "predictions" / (
        f"{ticker}_{split}_predictions.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model} {split} predictions: {path}")
    return pd.read_csv(path, parse_dates=["Date"])


def _check_alignment(
    frame: pd.DataFrame,
    windows: WindowSet,
    model: str,
    ticker: str,
    split: str,
) -> None:
    dates_match = np.array_equal(
        frame["Date"].to_numpy(dtype="datetime64[ns]"),
        windows.dates.to_numpy(dtype="datetime64[ns]"),
    )
    targets_match = np.allclose(
        frame["Target_Return"].to_numpy(dtype=np.float64),
        windows.y.astype(np.float64),
        atol=2e-8,
        rtol=0,
    )
    if not dates_match or not targets_match:
        raise ValueError(
            f"{model} {ticker} {split} predictions do not match the current "
            "feature windows and targets."
        )


def _prepared_data(root: Path) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for ticker in TICKERS:
        raw_path = root / "data" / "raw" / f"{ticker}_raw.csv"
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"Expected cached Yahoo Finance data for {ticker}: {raw_path}"
            )
        raw = pd.read_csv(raw_path, index_col="Date", parse_dates=["Date"])
        cleaned = clean_ohlcv(raw)
        features, feature_columns = engineer_features(cleaned)
        scaler, windows = create_scaled_windows(
            features, feature_columns, lookback=30
        )
        prepared[ticker] = {
            "features": features,
            "feature_columns": feature_columns,
            "scaler": scaler,
            "windows": windows,
        }
    return prepared


def _candidate_configs(ticker_index: int) -> list[TrainingConfig]:
    """Predeclare a small validation-only learning-rate/loss-weight sweep."""
    return [
        TrainingConfig(
            model_type="hybrid_lstm_gru",
            learning_rate=learning_rate,
            seed=42 + ticker_index,
            classification_loss_weight=weight,
        )
        for weight in CLASSIFICATION_WEIGHTS
        for learning_rate in LEARNING_RATES
    ]


def _rank_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    ranked = scores.copy()
    ranked["direction_rank"] = ranked.groupby("ticker")[
        "directional_accuracy"
    ].rank(ascending=False, method="average")
    ranked["mae_rank"] = ranked.groupby("ticker")["mae"].rank(
        ascending=True, method="average"
    )
    ranked["rmse_rank"] = ranked.groupby("ticker")["rmse"].rank(
        ascending=True, method="average"
    )
    ranked["selection_score"] = (
        0.6 * ranked["direction_rank"]
        + 0.2 * ranked["mae_rank"]
        + 0.2 * ranked["rmse_rank"]
    )
    return (
        ranked.groupby("candidate", as_index=False)
        .agg(
            selection_score=("selection_score", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            learning_rate=("learning_rate", "first"),
            classification_loss_weight=("classification_loss_weight", "first"),
        )
        .sort_values(
            ["selection_score", "mae", "rmse", "candidate"],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def _candidate_name(config: TrainingConfig) -> str:
    return (
        f"lr{config.learning_rate:g}_"
        f"cls{config.classification_loss_weight:g}"
    )


def _run_validation_candidate(
    config: TrainingConfig,
    ticker: str,
    data: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(len(data["feature_columns"]), config)
    windows: dict[str, WindowSet] = data["windows"]
    result = train_model(
        model, windows["train"], windows["validation"], config, device
    )
    validation = windows["validation"]
    predictions = predict(result.model, validation, device)
    directions = predict_direction(result.model, validation, device)
    return {
        "candidate": _candidate_name(config),
        "ticker": ticker,
        "best_epoch": result.best_epoch,
        "learning_rate": config.learning_rate,
        "classification_loss_weight": config.classification_loss_weight,
        **_score(validation.y, predictions, directions),
    }


def _save_final_model(
    config: TrainingConfig,
    ticker: str,
    data: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    model = build_model(len(data["feature_columns"]), config)
    windows: dict[str, WindowSet] = data["windows"]
    result = train_model(
        model, windows["train"], windows["validation"], config, device
    )
    checkpoint_dir = output_dir / "models"
    prediction_dir = output_dir / "predictions"
    representation_dir = output_dir / "representations"
    loss_dir = output_dir / "losses"
    for directory in (checkpoint_dir, prediction_dir, representation_dir, loss_dir):
        directory.mkdir(parents=True, exist_ok=True)

    scaler = data["scaler"]
    torch.save(
        {
            "ticker": ticker,
            "model_type": config.model_type,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in result.model.state_dict().items()
            },
            "training_config": asdict(config),
            "feature_columns": data["feature_columns"],
            "target_column": "Daily_Return",
            "lookback": 30,
            "scaler_data_min": scaler.data_min_.tolist(),
            "scaler_data_max": scaler.data_max_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scaler_min": scaler.min_.tolist(),
            "best_epoch": result.best_epoch,
        },
        checkpoint_dir / f"{ticker}.pt",
    )

    prediction_rows: list[dict[str, Any]] = []
    representation_frames: list[pd.DataFrame] = []
    for split_name, split_windows in windows.items():
        predictions = predict(result.model, split_windows, device)
        directions = predict_direction(result.model, split_windows, device)
        prediction = pd.DataFrame(
            {
                "Date": split_windows.dates.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Model": config.model_type,
                "Split": split_name,
                "Target_Return": split_windows.y,
                "Predicted_Return": predictions,
                "Actual_Up": split_windows.y > 0,
                "Predicted_Up": directions.astype(bool),
            }
        )
        prediction.to_csv(
            prediction_dir / f"{ticker}_{split_name}_predictions.csv", index=False
        )
        prediction_rows.append(
            {
                "model": config.model_type,
                "ticker": ticker,
                "split": split_name,
                "start_date": split_windows.dates[0].strftime("%Y-%m-%d"),
                "end_date": split_windows.dates[-1].strftime("%Y-%m-%d"),
                **_score(split_windows.y, predictions, directions),
            }
        )

        representations = extract_representations(
            result.model, split_windows, device
        )
        representation_frame = pd.DataFrame(
            representations,
            columns=[
                f"market_rep_{index:03d}"
                for index in range(representations.shape[1])
            ],
        )
        representation_frame.insert(0, "Split", split_name)
        representation_frame.insert(0, "Ticker", ticker)
        representation_frame.insert(0, "Date", split_windows.dates.strftime("%Y-%m-%d"))
        representation_frame.insert(1, "Model", config.model_type)
        representation_frame["Target_Return"] = split_windows.y
        representation_frame["Predicted_Return"] = predictions
        representation_frame["Actual_Up"] = split_windows.y > 0
        representation_frame["Predicted_Up"] = directions.astype(bool)
        representation_frames.append(representation_frame)

    pd.DataFrame(
        {
            "epoch": np.arange(1, len(result.train_losses) + 1),
            "train_combined_loss": result.train_losses,
            "validation_combined_loss": result.validation_losses,
        }
    ).to_csv(loss_dir / f"{ticker}.csv", index=False)
    metadata = {
        "ticker": ticker,
        "model_type": config.model_type,
        "training_config": asdict(config),
        "best_epoch": result.best_epoch,
        "feature_columns": data["feature_columns"],
        "lookback_days": 30,
        "split_target_samples": {
            split_name: len(split_windows.y)
            for split_name, split_windows in windows.items()
        },
        "device": str(device),
    }
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / f"{ticker}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata, prediction_rows, pd.concat(representation_frames, ignore_index=True)


def _baseline_metrics_after_selection(
    root: Path, prepared: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Read baseline test predictions after hybrid settings have been frozen."""
    rows: list[dict[str, Any]] = []
    for ticker, data in prepared.items():
        windows: WindowSet = data["windows"]["test"]
        for model in BASELINE_MODELS:
            frame = _read_predictions(root, model, ticker, "test")
            _check_alignment(frame, windows, model, ticker, "test")
            predicted = frame["Predicted_Return"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "model": model,
                    "ticker": ticker,
                    "split": "test",
                    "start_date": windows.dates[0].strftime("%Y-%m-%d"),
                    "end_date": windows.dates[-1].strftime("%Y-%m-%d"),
                    **_score(
                        windows.y.astype(np.float64),
                        predicted,
                        predicted > 0,
                    ),
                }
            )
    return rows


def run_experiment(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on validation only and evaluate selected hybrid once on test data."""
    prepared = _prepared_data(root)
    for ticker, data in prepared.items():
        windows: dict[str, WindowSet] = data["windows"]
        for model in BASELINE_MODELS:
            validation_frame = _read_predictions(root, model, ticker, "validation")
            _check_alignment(
                validation_frame, windows["validation"], model, ticker, "validation"
            )

    outputs_dir = root / "outputs"
    model_output_dir = outputs_dir / "hybrid_lstm_gru"
    experiment_dir = outputs_dir / "experiments"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using %s for hybrid validation selection", device)

    candidate_rows: list[dict[str, Any]] = []
    configs_by_name: dict[str, TrainingConfig] = {}
    for ticker_index, ticker in enumerate(TICKERS):
        for config in _candidate_configs(ticker_index):
            name = _candidate_name(config)
            configs_by_name[name] = config
            LOGGER.info("Validation trial %s / %s", name, ticker)
            candidate_rows.append(
                _run_validation_candidate(
                    config, ticker, prepared[ticker], device
                )
            )
    validation_scores = pd.DataFrame(candidate_rows)
    validation_scores.to_csv(experiment_dir / "validation_candidates.csv", index=False)
    ranking = _rank_candidates(validation_scores)
    ranking.to_csv(experiment_dir / "validation_ranking.csv", index=False)
    selected_name = str(ranking.iloc[0]["candidate"])
    selected_config = configs_by_name[selected_name]
    selection = {
        "selected_candidate": selected_name,
        "selection_rule": (
            "Validation only: average per-ticker metric ranks; directional "
            "accuracy rank 60%, MAE rank 20%, RMSE rank 20%; ties use lower "
            "mean MAE, then RMSE."
        ),
        "training_config": asdict(selected_config),
        "test_used_for_tuning": False,
    }
    (experiment_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    # Test results are read only after validation configuration is frozen.
    baseline_test_rows = _baseline_metrics_after_selection(root, prepared)
    hybrid_metric_rows: list[dict[str, Any]] = []
    hybrid_test_rows: list[dict[str, Any]] = []
    representation_frames: list[pd.DataFrame] = []
    for ticker_index, ticker in enumerate(TICKERS):
        final_config = TrainingConfig(
            **{
                **asdict(selected_config),
                "seed": 42 + ticker_index,
            }
        )
        _, rows, representations = _save_final_model(
            final_config, ticker, prepared[ticker], device, model_output_dir
        )
        hybrid_metric_rows.extend(rows)
        representation_frames.append(representations)
        hybrid_test_rows.extend(
            row for row in rows if row["split"] == "test"
        )

    pd.DataFrame(hybrid_metric_rows).drop(
        columns=["start_date", "end_date"]
    ).to_csv(model_output_dir / "metrics.csv", index=False)
    pd.concat(representation_frames, ignore_index=True).to_csv(
        model_output_dir
        / "representations"
        / "hybrid_lstm_gru_market_representations.csv",
        index=False,
    )
    test_comparison = pd.DataFrame([*baseline_test_rows, *hybrid_test_rows])
    test_comparison.to_csv(experiment_dir / "test_comparison.csv", index=False)
    summary = (
        test_comparison.groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values("model")
    )
    summary.to_csv(experiment_dir / "test_summary.csv", index=False)
    comparison_dir = outputs_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = [
        pd.read_csv(outputs_dir / model / "metrics.csv") for model in MODEL_TYPES
    ]
    combined_metrics = pd.concat(model_metrics, ignore_index=True)
    combined_metrics = combined_metrics.drop_duplicates(
        subset=["model", "ticker", "split"], keep="last"
    ).sort_values(["model", "ticker", "split"])
    combined_metrics.to_csv(comparison_dir / "model_comparison.csv", index=False)
    (
        combined_metrics.loc[combined_metrics["split"] == "test"]
        .groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values("model")
        .to_csv(comparison_dir / "final_summary.csv", index=False)
    )
    LOGGER.info(
        "Saved hybrid artifacts under %s and experiment records under %s",
        model_output_dir,
        experiment_dir,
    )
    return ranking, test_comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    validation_ranking, test_comparison = run_experiment(
        Path(__file__).resolve().parents[1]
    )
    print("Validation candidate ranking:")
    print(validation_ranking.to_string(index=False))
    print("\nPer-ticker test comparison:")
    print(test_comparison.to_string(index=False))
    print("\nMean test comparison:")
    print(
        test_comparison.groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .to_string(index=False)
    )
