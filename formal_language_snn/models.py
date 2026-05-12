from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn


class ClassicRNN(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 16, num_classes: int = 2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_out = out[-1]
        return self.fc(last_out)


class SpikingNet(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 16, num_classes: int = 2, beta: float = 0.85):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for step in range(time_steps):
            cur_x = x[step]
            cur_x = self.fc1(cur_x)
            spk1, mem1 = self.lif1(cur_x, mem1)

            cur_x = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur_x, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)
