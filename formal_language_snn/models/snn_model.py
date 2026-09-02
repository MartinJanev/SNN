"""Spiking Neural Network using Leaky Integrate-and-Fire (LIF) neurons."""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn


class SpikingNet(nn.Module):
    """Two-layer LIF spiking network for sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_classes: int = 2,
        beta: float = 0.85,
        learn_beta: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=learn_beta)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=learn_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec: list[torch.Tensor] = []

        for t in range(time_steps):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)
