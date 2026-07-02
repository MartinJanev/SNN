"""Classical RNN model using a GRU cell."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassicRNN(nn.Module):
    """GRU-based recurrent classifier for sequential formal-language inputs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_classes: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.classifier(out[-1])
