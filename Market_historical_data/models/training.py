"""Reproducible model training, evaluation, and representation extraction."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Market_historical_data.models.encoders import (
    GRUMarketEncoder,
    HybridLSTMGRUMarketEncoder,
    LSTMMarketEncoder,
)
from Market_historical_data.preprocessing.splitting import WindowSet


@dataclass(frozen=True)
class TrainingConfig:
    model_type: str = "lstm"
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 8
    seed: int = 42
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    classification_loss_weight: float = 0.5
    return_loss_scale: float = 0.01


@dataclass
class TrainingResult:
    model: nn.Module
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(input_size: int, config: TrainingConfig) -> nn.Module:
    """Construct the selected market encoder."""
    if config.model_type == "lstm":
        return LSTMMarketEncoder(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.model_type == "gru":
        return GRUMarketEncoder(
            input_size=input_size,
            hidden_size=config.hidden_size,
        )
    if config.model_type == "hybrid_lstm_gru":
        return HybridLSTMGRUMarketEncoder(
            input_size=input_size,
            branch_hidden_size=64,
            fusion_size=64,
        )
    raise ValueError(f"Unsupported model type: {config.model_type}")


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(batch_x)
            predictions = outputs[0]
            if len(outputs) == 3:
                if config.return_loss_scale <= 0 or config.classification_loss_weight < 0:
                    raise ValueError(
                        "Hybrid return scale must be positive and classification "
                        "loss weight must be non-negative."
                    )
                direction_logits = outputs[2]
                direction_targets = (batch_y > 0).to(dtype=direction_logits.dtype)
                regression_loss = nn.functional.smooth_l1_loss(
                    predictions / config.return_loss_scale,
                    batch_y / config.return_loss_scale,
                )
                classification_loss = nn.functional.binary_cross_entropy_with_logits(
                    direction_logits, direction_targets
                )
                loss = (
                    regression_loss
                    + config.classification_loss_weight * classification_loss
                )
            else:
                loss = loss_function(predictions, batch_y)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        sample_count += batch_x.size(0)
    if sample_count == 0:
        raise ValueError("Cannot calculate loss for an empty dataset.")
    return total_loss / sample_count


def train_model(
    model: nn.Module,
    training: WindowSet,
    validation: WindowSet,
    config: TrainingConfig,
    device: torch.device,
) -> TrainingResult:
    """Train with validation-based early stopping and restore the best weights."""
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    set_seed(config.seed)
    train_dataset = TensorDataset(
        torch.from_numpy(training.x), torch.from_numpy(training.y)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation.x), torch.from_numpy(validation.y)
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(config.epochs):
        train_loss = _mean_loss(
            model, train_loader, loss_function, device, config, optimizer
        )
        validation_loss = _mean_loss(
            model, validation_loader, loss_function, device, config
        )
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise FloatingPointError("Non-finite model loss encountered during training.")
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid best model state.")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainingResult(model, train_losses, validation_losses, best_epoch)


def predict(model: nn.Module, windows: WindowSet, device: torch.device) -> np.ndarray:
    """Predict next-day returns in original (unscaled) return units."""
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows.x), 512):
            batch = torch.from_numpy(windows.x[start : start + 512]).to(device)
            output = model(batch)[0]
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def extract_representations(
    model: nn.Module, windows: WindowSet, device: torch.device
) -> np.ndarray:
    """Extract the encoder's final-step market representation for each window."""
    model.eval()
    representations: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows.x), 512):
            batch = torch.from_numpy(windows.x[start : start + 512]).to(device)
            representation = model(batch)[1]
            representations.append(representation.cpu().numpy())
    return (
        np.concatenate(representations)
        if representations
        else np.empty((0, 0), dtype=np.float32)
    )


def predict_direction(
    model: nn.Module, windows: WindowSet, device: torch.device
) -> np.ndarray:
    """Classify positive predicted returns as UP and other returns as DOWN."""
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows.x), 512):
            batch = torch.from_numpy(windows.x[start : start + 512]).to(device)
            outputs = model(batch)
            scores = outputs[2] if len(outputs) == 3 else outputs[0]
            predictions.append((scores.cpu().numpy() > 0).astype(np.int8))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int8)


def config_dict(config: TrainingConfig) -> dict[str, object]:
    """Return JSON-serializable training configuration."""
    return asdict(config)
