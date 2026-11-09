"""Focused tests for data integrity, leakage prevention, and market encoders."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from Market_historical_data.data.cleaning import clean_ohlcv
from Market_historical_data.data.download import OHLCV_COLUMNS
from Market_historical_data.experiments.hybrid_lstm_gru import (
    _rank_candidates,
    _read_predictions,
    _save_final_model,
)
from Market_historical_data.models.encoders import (
    GRUMarketEncoder,
    HybridLSTMGRUMarketEncoder,
    LSTMMarketEncoder,
)
from Market_historical_data.models.training import (
    TrainingConfig,
    build_model,
    set_seed,
    train_model,
)
from main import run_pipeline
from Market_historical_data.preprocessing.features import engineer_features
from Market_historical_data.preprocessing.splitting import (
    WindowSet,
    create_scaled_windows,
)


def sample_ohlcv(rows: int = 160) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = np.linspace(50, 90, rows) + np.sin(np.arange(rows) / 4)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Adj Close": close - 0.1,
            "Volume": np.arange(rows, dtype=float) + 1000,
        },
        index=dates,
    ).loc[:, list(OHLCV_COLUMNS)]


class DataCleaningTests(unittest.TestCase):
    def test_removes_duplicate_dates_and_invalid_rows(self) -> None:
        frame = sample_ohlcv(4)
        duplicate = frame.iloc[[1]].copy()
        duplicate.iloc[0, duplicate.columns.get_loc("Close")] = 56.0
        duplicate.iloc[0, duplicate.columns.get_loc("Open")] = 55.8
        duplicate.iloc[0, duplicate.columns.get_loc("High")] = 56.8
        duplicate.iloc[0, duplicate.columns.get_loc("Low")] = 55.2
        duplicate.iloc[0, duplicate.columns.get_loc("Adj Close")] = 55.9
        invalid = frame.iloc[[2]].copy()
        invalid.iloc[0, invalid.columns.get_loc("High")] = 1.0
        data = pd.concat([frame, duplicate, invalid])

        cleaned = clean_ohlcv(data)

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned.index.nunique(), len(cleaned))
        self.assertEqual(cleaned.iloc[1]["Close"], 56.0)
        self.assertNotIn(invalid.index[0], cleaned.index)


class FeatureAndSplitTests(unittest.TestCase):
    def test_features_are_finite_and_have_expected_indicators(self) -> None:
        features, columns = engineer_features(clean_ohlcv(sample_ohlcv()))
        self.assertTrue(
            {"Daily_Return", "SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14",
             "MACD", "MACD_Signal", "MACD_Histogram", "Volatility_20"}.issubset(columns)
        )
        self.assertTrue(np.isfinite(features[columns].to_numpy()).all())
        self.assertTrue(features["RSI_14"].between(0, 100).all())

    def test_scaler_uses_training_rows_and_windows_target_next_day(self) -> None:
        features, columns = engineer_features(clean_ohlcv(sample_ohlcv()))
        original = features.copy()
        train_end = int(len(features) * 0.70)
        feature_name = columns[0]
        features.iloc[train_end:, features.columns.get_loc(feature_name)] += 1_000_000
        scaler, windows = create_scaled_windows(features, columns, lookback=30)

        self.assertAlmostEqual(scaler.data_max_[0], original[feature_name].iloc[:train_end].max())
        self.assertEqual(windows["train"].x.shape[1], 30)
        self.assertEqual(windows["train"].x.shape[2], len(columns))
        first_target_date = windows["train"].dates[0]
        target_position = features.index.get_loc(first_target_date)
        self.assertAlmostEqual(
            windows["train"].y[0],
            features["Daily_Return"].iloc[target_position],
            places=7,
        )
        self.assertLess(windows["train"].dates[-1], windows["validation"].dates[0])
        self.assertLess(windows["validation"].dates[-1], windows["test"].dates[0])


class HybridSelectionTests(unittest.TestCase):
    def test_candidate_ranking_compares_trials_per_ticker(self) -> None:
        rows = []
        for ticker in ("AAPL", "MSFT", "AMZN"):
            rows.extend(
                [
                    {
                        "candidate": "direction_better",
                        "ticker": ticker,
                        "directional_accuracy": 0.60,
                        "mae": 0.011,
                        "rmse": 0.016,
                        "learning_rate": 0.001,
                        "classification_loss_weight": 0.5,
                    },
                    {
                        "candidate": "error_better",
                        "ticker": ticker,
                        "directional_accuracy": 0.50,
                        "mae": 0.010,
                        "rmse": 0.015,
                        "learning_rate": 0.0005,
                        "classification_loss_weight": 0.1,
                    },
                ]
            )

        ranking = _rank_candidates(pd.DataFrame(rows))

        self.assertEqual(ranking.iloc[0]["candidate"], "direction_better")


class EncoderTests(unittest.TestCase):
    def test_seed_reproduces_encoder_initialization(self) -> None:
        config = TrainingConfig(hidden_size=16)
        set_seed(config.seed)
        first = build_model(input_size=5, config=config)
        set_seed(config.seed)
        second = build_model(input_size=5, config=config)
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values()
        ):
            self.assertTrue(torch.equal(first_value, second_value))

    def test_final_encoders_produce_predictions_and_representations(self) -> None:
        inputs = torch.randn(4, 30, 12)
        for model in (
            LSTMMarketEncoder(input_size=12, hidden_size=16),
            GRUMarketEncoder(input_size=12, hidden_size=64),
        ):
            prediction, representation = model(inputs)
            self.assertEqual(tuple(prediction.shape), (4,))
            expected_size = 64 if isinstance(model, GRUMarketEncoder) else 16
            self.assertEqual(tuple(representation.shape), (4, expected_size))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_hybrid_lstm_gru_outputs_and_branch_dimensions(self) -> None:
        model = HybridLSTMGRUMarketEncoder(input_size=12)
        returns, representation, direction_logits = model(torch.randn(5, 30, 12))

        self.assertEqual(tuple(returns.shape), (5,))
        self.assertEqual(tuple(representation.shape), (5, 64))
        self.assertEqual(tuple(direction_logits.shape), (5,))
        self.assertEqual(model.lstm.hidden_size, 64)
        self.assertEqual(model.gru.hidden_size, 64)
        self.assertEqual(model.fusion.in_features, 128)
        self.assertEqual(model.fusion.out_features, 64)
        self.assertEqual(model.return_head.out_features, 1)
        self.assertEqual(model.direction_head.out_features, 1)
        self.assertIsNot(model.return_head, model.direction_head)
        self.assertTrue(torch.isfinite(returns).all())
        self.assertTrue(torch.isfinite(direction_logits).all())

    def test_build_model_supports_only_final_architectures(self) -> None:
        inputs = torch.randn(3, 30, 12)
        for model_type, representation_size in (
            ("lstm", 16),
            ("gru", 64),
        ):
            model = build_model(
                input_size=12,
                config=TrainingConfig(
                    model_type=model_type,
                    hidden_size=representation_size,
                ),
            )
            prediction, representation = model(inputs)
            self.assertEqual(tuple(prediction.shape), (3,))
            self.assertEqual(tuple(representation.shape), (3, representation_size))
            if model_type == "gru":
                self.assertEqual(model.gru.num_layers, 1)
                self.assertEqual(model.gru.hidden_size, 64)

    def test_final_models_training_smoke(self) -> None:
        training = WindowSet(
            x=np.random.default_rng(4).random((8, 30, 12), dtype=np.float32),
            y=np.random.default_rng(5).normal(0, 0.01, 8).astype(np.float32),
            dates=pd.bdate_range("2022-01-03", periods=8),
        )
        validation = WindowSet(
            x=np.random.default_rng(6).random((4, 30, 12), dtype=np.float32),
            y=np.random.default_rng(7).normal(0, 0.01, 4).astype(np.float32),
            dates=pd.bdate_range("2022-02-01", periods=4),
        )
        for model_type, hidden_size in (
            ("lstm", 16),
            ("gru", 64),
            ("hybrid_lstm_gru", 64),
        ):
            config = TrainingConfig(
                model_type=model_type,
                epochs=1,
                batch_size=4,
                patience=1,
                hidden_size=hidden_size,
            )
            result = train_model(
                build_model(input_size=12, config=config),
                training,
                validation,
                config,
                torch.device("cpu"),
            )
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(len(result.train_losses), 1)
            self.assertTrue(np.isfinite(result.validation_losses[0]))


class OutputStructureTests(unittest.TestCase):
    def test_each_model_saves_independently_under_its_own_directory(self) -> None:
        ticker = "AAPL"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Market_historical_data"
            root.mkdir()
            cached_data = {ticker: sample_ohlcv(180)}
            checkpoint_hashes: dict[str, bytes] = {}
            with (
                patch("main._load_or_download_data", return_value=cached_data),
                patch("main.plot_price_and_indicators"),
                patch("main.plot_test_predictions"),
                patch("main.plot_training_losses"),
            ):
                for model_type in ("lstm", "gru", "hybrid_lstm_gru"):
                    run_pipeline(
                        model_type=model_type,
                        epochs=1,
                        lookback=30,
                        tickers=(ticker,),
                        root=root,
                    )
                    model_root = root / "outputs" / model_type
                    for folder in (
                        "models",
                        "predictions",
                        "representations",
                        "plots",
                        "metadata",
                    ):
                        self.assertTrue((model_root / folder).is_dir())
                    self.assertTrue((model_root / "metrics.csv").is_file())
                    self.assertTrue(
                        (
                            model_root
                            / "models"
                            / f"{ticker}.pt"
                        ).is_file()
                    )
                    self.assertTrue(
                        (
                            model_root
                            / "predictions"
                            / f"{ticker}_test_predictions.csv"
                        ).is_file()
                    )
                    self.assertTrue(
                        (
                            model_root
                            / "representations"
                            / f"{model_type}_market_representations.csv"
                        ).is_file()
                    )
                    if model_type != "hybrid_lstm_gru":
                        self.assertFalse((model_root / "losses").exists())
                    else:
                        self.assertTrue((model_root / "losses").is_dir())
                        self.assertTrue(
                            (model_root / "losses" / f"{ticker}.csv").is_file()
                        )
                    checkpoint_hashes[model_type] = (
                        model_root / "models" / f"{ticker}.pt"
                    ).read_bytes()

            for model_type, checkpoint in checkpoint_hashes.items():
                saved = (
                    root
                    / "outputs"
                    / model_type
                    / "models"
                    / f"{ticker}.pt"
                ).read_bytes()
                self.assertEqual(saved, checkpoint)

            comparison = pd.read_csv(
                root / "outputs" / "comparison" / "model_comparison.csv"
            )
            summary = pd.read_csv(
                root / "outputs" / "comparison" / "final_summary.csv"
            )
            self.assertEqual(
                set(comparison["model"]),
                {"lstm", "gru", "hybrid_lstm_gru"},
            )
            self.assertEqual(
                set(summary["model"]),
                {"lstm", "gru", "hybrid_lstm_gru"},
            )

    def test_hybrid_experiment_writer_uses_model_specific_paths(self) -> None:
        def windows(size: int, start: str, seed: int) -> WindowSet:
            return WindowSet(
                x=np.random.default_rng(seed).random((size, 30, 3), dtype=np.float32),
                y=np.random.default_rng(seed + 1).normal(0, 0.01, size).astype(
                    np.float32
                ),
                dates=pd.bdate_range(start, periods=size),
            )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Market_historical_data"
            root.mkdir()
            output_dir = root / "outputs" / "hybrid_lstm_gru"
            data = {
                "feature_columns": ["feature_a", "feature_b", "feature_c"],
                "scaler": MinMaxScaler().fit(np.zeros((2, 3))),
                "windows": {
                    "train": windows(8, "2022-01-03", 10),
                    "validation": windows(4, "2022-02-01", 20),
                    "test": windows(4, "2022-03-01", 30),
                },
            }
            config = TrainingConfig(
                model_type="hybrid_lstm_gru",
                epochs=1,
                batch_size=4,
                patience=1,
            )
            _, metric_rows, representations = _save_final_model(
                config,
                "AAPL",
                data,
                torch.device("cpu"),
                output_dir,
            )

            checkpoint = output_dir / "models" / "AAPL.pt"
            prediction = output_dir / "predictions" / "AAPL_test_predictions.csv"
            loss_history = output_dir / "losses" / "AAPL.csv"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(prediction.is_file())
            self.assertTrue(loss_history.is_file())
            self.assertEqual(len(metric_rows), 3)
            self.assertEqual(len(representations), 16)
            loaded = _read_predictions(root, "hybrid_lstm_gru", "AAPL", "test")
            self.assertEqual(len(loaded), 4)


if __name__ == "__main__":
    unittest.main()
