"""Small LSTM encoder with a direct fixed-horizon coordinate decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    """Architecture dimensions for the 8-step to 12-step coordinate predictor."""

    observation_len: int = 8
    horizon_len: int = 12
    input_size: int = 2
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.observation_len != 8 or self.horizon_len != 12 or self.input_size != 2:
            raise ValueError("TrackVision LSTM requires 8 observed steps, 12 future steps, and 2D positions")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class LSTMTrajectoryPredictor(nn.Module):
    """Encode the observed positions and project the final state to 12 future points.

    The encoder is a batch-first LSTM. Its final sequence output passes through a linear
    projection to 24 normalized coordinate values, reshaped to ``(batch, 12, 2)``. This
    direct decoder avoids autoregressive feedback and keeps the model small.
    """

    def __init__(self, config: LSTMConfig | None = None) -> None:
        super().__init__()
        self.config = config or LSTMConfig()
        recurrent_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.Linear(
            self.config.hidden_size,
            self.config.horizon_len * self.config.input_size,
        )

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or tuple(observed.shape[1:]) != (
            self.config.observation_len,
            self.config.input_size,
        ):
            raise ValueError("observed must have shape (batch, 8, 2)")
        if observed.shape[0] == 0:
            raise ValueError("observed batch must not be empty")
        if not torch.isfinite(observed).all():
            raise ValueError("observed positions must be finite")
        encoded, _ = self.encoder(observed)
        decoded = self.decoder(encoded[:, -1, :])
        return decoded.reshape(
            observed.shape[0], self.config.horizon_len, self.config.input_size
        )
