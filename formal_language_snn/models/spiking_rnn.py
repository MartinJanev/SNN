"""Recurrent spiking network: the control that separates spiking from feedforward.

``SpikingNet`` is strictly feedforward, so its only memory is passive membrane decay. That
makes every one of its failures ambiguous: it could be spiking that fails, or it could be
the absence of recurrence. This model exists to break that tie, so it is deliberately the
*same* network with exactly one thing changed -- the hidden layer's LIF neurons gain
recurrent connections (snntorch's ``RLeaky``, an all-to-all hidden-to-hidden weight matrix)
and feed their own spikes back at the next timestep.

Everything else is held identical to ``SpikingNet``: the same two-layer shape, the same
input and readout projections, the same beta, the same rate-coded output. The readout layer
stays a plain ``Leaky`` -- recurrence belongs in the hidden state, and a 2x2 recurrent
matrix over the class units would add a second uncontrolled difference for nothing.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn


class SpikingRNN(nn.Module):
    """Two-layer spiking network with a recurrent hidden LIF layer."""

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
        self.rlif1 = snn.RLeaky(
            beta=beta,
            learn_beta=learn_beta,
            all_to_all=True,
            linear_features=hidden_size,
        )
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=learn_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(0)
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec: list[torch.Tensor] = []

        for t in range(time_steps):
            cur = self.fc1(x[t])
            spk1, mem1 = self.rlif1(cur, spk1, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)
