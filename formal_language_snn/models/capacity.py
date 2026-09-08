"""Parameter counting and capacity matching across model families.

The published comparison ran every model at ``hidden_size=32``, which gives the GRU roughly
28x the trainable parameters of the feedforward SNN -- and by a factor that itself varies
with the alphabet size, so capacity is confounded with architecture differently in each
language. These helpers pick a per-model hidden size that equalises the parameter budget,
leaving the gating-versus-membrane-decay difference as the only one that remains.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

from .lstm import ClassicLSTM
from .rnn import ClassicRNN
from .snn_model import SpikingNet
from .spiking_rnn import SpikingRNN

BUILDERS: Dict[str, Callable[[int, int], nn.Module]] = {
    "rnn": lambda input_size, hidden: ClassicRNN(input_size, hidden),
    "lstm": lambda input_size, hidden: ClassicLSTM(input_size, hidden),
    "snn": lambda input_size, hidden: SpikingNet(input_size, hidden),
    "rsnn": lambda input_size, hidden: SpikingRNN(input_size, hidden),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_for(kind: str, input_size: int, hidden_size: int) -> int:
    """Trainable parameter count, without disturbing the seeded RNG.

    The count is obtained by building a throwaway model, and the binary search below builds
    a dozen or so of them per run. Those constructions draw from the global torch RNG, so
    without the fork the *number* of candidates tried would feed straight into the real
    models' initialisation -- and simply adding a family to BUILDERS would silently move
    every other model's numbers. Parameter counting is a measurement; it must not perturb
    the experiment it is measuring.
    """
    with torch.random.fork_rng(devices=[]):
        return count_parameters(BUILDERS[kind](input_size, hidden_size))


def matched_hidden_size(
    kind: str,
    input_size: int,
    target_params: int,
    *,
    max_hidden: int = 4096,
) -> int:
    """Smallest hidden size whose parameter count is closest to ``target_params``.

    Parameter count is monotone in hidden size for all three families, so a binary search
    finds the crossing point and the two neighbours are compared exactly.
    """
    low, high = 1, max_hidden
    while low < high:
        mid = (low + high) // 2
        if parameters_for(kind, input_size, mid) < target_params:
            low = mid + 1
        else:
            high = mid
    candidates = [h for h in (low - 1, low) if h >= 1]
    return min(candidates, key=lambda h: abs(parameters_for(kind, input_size, h) - target_params))


def matched_hidden_sizes(
    input_size: int,
    reference_kind: str = "rnn",
    reference_hidden: int = 32,
) -> Dict[str, int]:
    """Hidden size per model kind that matches ``reference_kind``'s parameter budget."""
    target = parameters_for(reference_kind, input_size, reference_hidden)
    return {kind: matched_hidden_size(kind, input_size, target) for kind in BUILDERS}
