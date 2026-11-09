"""PyTorch LSTM, GRU, and hybrid LSTM-GRU market encoders."""

import torch
from torch import nn


class LSTMMarketEncoder(nn.Module):
    """Encode a feature window with an LSTM and predict its next-day return."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.lstm(inputs)
        representation = encoded[:, -1, :]
        prediction = self.head(representation).squeeze(-1)
        return prediction, representation


class GRUMarketEncoder(nn.Module):
    """Encode a feature window with a 64-unit GRU and predict next-day return."""

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.gru(inputs)
        representation = encoded[:, -1, :]
        prediction = self.head(representation).squeeze(-1)
        return prediction, representation


class HybridLSTMGRUMarketEncoder(nn.Module):
    """Fuse parallel LSTM and GRU branches for return and direction prediction."""

    def __init__(
        self,
        input_size: int,
        branch_hidden_size: int = 64,
        fusion_size: int = 64,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=branch_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=branch_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fusion = nn.Linear(branch_hidden_size * 2, fusion_size)
        self.activation = nn.ReLU()
        self.return_head = nn.Linear(fusion_size, 1)
        self.direction_head = nn.Linear(fusion_size, 1)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lstm_sequence, _ = self.lstm(inputs)
        gru_sequence, _ = self.gru(inputs)
        combined = torch.cat((lstm_sequence[:, -1, :], gru_sequence[:, -1, :]), dim=1)
        representation = self.activation(self.fusion(combined))
        return (
            self.return_head(representation).squeeze(-1),
            representation,
            self.direction_head(representation).squeeze(-1),
        )
