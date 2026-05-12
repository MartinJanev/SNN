from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import random
from typing import Dict, List, Iterable

import torch
import torch.nn as nn
from snntorch import functional as SF

from .data import Sample, generate_dataset, word_to_tensor
from .models import ClassicRNN, SpikingNet


@dataclass(frozen=True)
class ExperimentConfig:
    train_pairs: int = 400
    test_pairs: int = 100
    train_min_n: int = 1
    train_max_n: int = 5
    test_min_n: int = 6
    test_max_n: int = 10
    hidden_size: int = 16
    beta: float = 0.85
    lr: float = 0.01
    epochs: int = 5
    seed: int = 42
    device: str | None = None
    # Choose which formal language to generate and train/test on
    # Options: "anbn" (default), "palindrome", "paren"
    language: str = "anbn"
    # Optionally provide an explicit alphabet (list/iterable of symbols). If None, a default
    # alphabet is chosen based on the language.
    alphabet: Iterable[str] | None = None
@dataclass(frozen=True)
class ExperimentResults:
    device: str
    history: Dict[str, List[float]]
    rnn_accuracy: float
    snn_accuracy: float
    rnn_accuracy_by_length: Dict[int, float]
    snn_accuracy_by_length: Dict[int, float]
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_name(config: ExperimentConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_snn(model: SpikingNet, inputs: torch.Tensor) -> torch.Tensor:
    spikes = model(inputs)
    return spikes.sum(dim=0)


def _accuracy_rnn(model: ClassicRNN, dataset: List[Sample], device: torch.device, alphabet: Iterable[str] | None = None) -> float:
    correct = 0
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            pred = model(inputs).argmax(dim=1).item()
            correct += int(pred == sample.label)
    return correct / len(dataset)


def _accuracy_snn(model: SpikingNet, dataset: List[Sample], device: torch.device, alphabet: Iterable[str] | None = None) -> float:
    correct = 0
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            pred = _predict_snn(model, inputs).argmax(dim=1).item()
            correct += int(pred == sample.label)
    return correct / len(dataset)


def _accuracy_by_length(model, dataset: List[Sample], device: torch.device, model_kind: str, alphabet: Iterable[str] | None = None) -> Dict[int, float]:
    buckets: Dict[int, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()

    with torch.no_grad():
        for sample in dataset:
            length = max(sample.a_count, sample.b_count)
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)

            if model_kind == "rnn":
                pred = model(inputs).argmax(dim=1).item()
            elif model_kind == "snn":
                pred = _predict_snn(model, inputs).argmax(dim=1).item()
            else:
                raise ValueError(f"Unknown model_kind: {model_kind}")

            buckets[length][1] += 1
            buckets[length][0] += int(pred == sample.label)

    return {length: correct / total for length, (correct, total) in sorted(buckets.items())}


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    # Phase 1: Setup - set seeds and device
    print("[Phase 1] Setup: seeding and device selection")
    set_seed(config.seed)
    device = _device_name(config)
    print(f"  Using device: {device}")

    # Phase 2: Data Generation - create train and test datasets for selected language
    print(f"[Phase 2] Data generation for language='{config.language}'")

    # Resolve the alphabet: use provided one or get default from language registry
    if config.alphabet is not None:
        alphabet_to_use = config.alphabet
    else:
        from . import languages as lang_module
        alphabet_to_use = lang_module.get_alphabet(config.language)

    train_data = generate_dataset(
        config.train_pairs,
        config.train_min_n,
        config.train_max_n,
        seed=config.seed,
        language=config.language,
        alphabet=alphabet_to_use,
    )
    test_data = generate_dataset(
        config.test_pairs,
        config.test_min_n,
        config.test_max_n,
        seed=config.seed + 1,
        language=config.language,
        alphabet=alphabet_to_use,
    )

    # Phase 3: Model Initialization - create RNN and SNN models and optimizers
    print("[Phase 3] Model initialization")
    input_size = len(alphabet_to_use)
    print(f"  Language: {config.language}, Input size: {input_size}")
    rnn_model = ClassicRNN(input_size=input_size, hidden_size=config.hidden_size).to(device)
    snn_model = SpikingNet(input_size=input_size, hidden_size=config.hidden_size, beta=config.beta).to(device)

    optimizer_rnn = torch.optim.Adam(rnn_model.parameters(), lr=config.lr)
    optimizer_snn = torch.optim.Adam(snn_model.parameters(), lr=config.lr)

    loss_rnn_fn = nn.CrossEntropyLoss()
    loss_snn_fn = SF.ce_rate_loss()

    history: Dict[str, List[float]] = {"rnn_loss": [], "snn_loss": []}

    # Phase 4: Training - iterate epochs and samples
    print("[Phase 4] Training start")
    for epoch_idx in range(config.epochs):
        print(f"  Epoch {epoch_idx + 1}/{config.epochs}...", end=" ", flush=True)
        rnn_model.train()
        snn_model.train()
        rnn_loss_total = 0.0
        snn_loss_total = 0.0

        for sample in train_data:
            # convert word to tensor using the resolved alphabet (ensures consistent input_size)
            inputs = word_to_tensor(sample.word, alphabet=alphabet_to_use).to(device)
            targets = torch.tensor([sample.label], dtype=torch.long, device=device)

            # Train RNN on the sequence
            optimizer_rnn.zero_grad()
            logits_rnn = rnn_model(inputs)
            loss_rnn = loss_rnn_fn(logits_rnn, targets)
            loss_rnn.backward()
            optimizer_rnn.step()
            rnn_loss_total += loss_rnn.item()

            # Train SNN on the same sequence
            optimizer_snn.zero_grad()
            spikes_snn = snn_model(inputs)
            loss_snn = loss_snn_fn(spikes_snn, targets)
            loss_snn.backward()
            optimizer_snn.step()
            snn_loss_total += loss_snn.item()

        rnn_avg = rnn_loss_total / len(train_data)
        snn_avg = snn_loss_total / len(train_data)
        history["rnn_loss"].append(rnn_avg)
        history["snn_loss"].append(snn_avg)
        print(f"RNN loss={rnn_avg:.6f} | SNN loss={snn_avg:.6f}")

    # Phase 5: Evaluation - compute accuracy and per-length breakdown
    print("[Phase 5] Evaluation on test set")
    rnn_accuracy = _accuracy_rnn(rnn_model, test_data, device, alphabet=alphabet_to_use)
    snn_accuracy = _accuracy_snn(snn_model, test_data, device, alphabet=alphabet_to_use)
    rnn_accuracy_by_length = _accuracy_by_length(rnn_model, test_data, device, "rnn", alphabet=alphabet_to_use)
    snn_accuracy_by_length = _accuracy_by_length(snn_model, test_data, device, "snn", alphabet=alphabet_to_use)

    results = ExperimentResults(
        device=str(device),
        history=history,
        rnn_accuracy=rnn_accuracy,
        snn_accuracy=snn_accuracy,
        rnn_accuracy_by_length=rnn_accuracy_by_length,
        snn_accuracy_by_length=snn_accuracy_by_length,
    )
    print("[Done] Experiment finished")
    return results

