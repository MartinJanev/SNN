"""Neural network models: Classical RNN (GRU) and Spiking Neural Network."""

from .rnn import ClassicRNN
from .snn_model import SpikingNet

__all__ = ["ClassicRNN", "SpikingNet"]
