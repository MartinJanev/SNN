from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
from snntorch import functional as SF

from .data import (
    Sample,
    generate_dataset,
    generate_difficulty_dataset,
    generate_stratified_dataset,
    get_language,
    length_bucket,
    word_length,
    word_to_tensor,
)
from .models import ClassicRNN, SpikingNet


@dataclass(frozen=True)
class ExperimentConfig:
    train_pairs: int = 400
    test_pairs: int = 100
    train_min_n: int = 1
    train_max_n: int = 10
    test_min_n: int = 1
    test_max_n: int = 80
    hidden_size: int = 32
    beta: float = 0.85
    learn_beta: bool = True
    lr: float = 0.01
    epochs: int = 5
    seed: int = 42
    device: str | None = None
    language: str = "anbn"
    alphabet: Iterable[str] | None = None
    output_dir: str | None = "outputs"
    stratified_test: bool = False
    difficulty_test: bool = False
    pairs_per_bucket: int = 50
    pairs_per_difficulty_cell: int = 5


@dataclass(frozen=True)
class ExperimentResults:
    device: str
    history: Dict[str, List[float]]
    rnn_accuracy: float
    snn_accuracy: float
    rnn_accuracy_by_length: Dict[int, float]
    snn_accuracy_by_length: Dict[int, float]
    rnn_accuracy_by_bucket: Dict[str, float] = field(default_factory=dict)
    snn_accuracy_by_bucket: Dict[str, float] = field(default_factory=dict)
    rnn_hard_neg_accuracy: float | None = None
    rnn_easy_neg_accuracy: float | None = None
    snn_hard_neg_accuracy: float | None = None
    snn_easy_neg_accuracy: float | None = None
    output_path: str | None = None


@dataclass
class AggregatedResults:
    num_seeds: int
    metrics: Dict[str, Dict[str, float]]


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
    return model(inputs).sum(dim=0)


def _predict(model, inputs: torch.Tensor, model_kind: str) -> int:
    if model_kind == "rnn":
        return model(inputs).argmax(dim=1).item()
    if model_kind == "snn":
        return _predict_snn(model, inputs).argmax(dim=1).item()
    raise ValueError(f"Unknown model_kind: {model_kind}")


def _accuracy(model, dataset: List[Sample], device: torch.device, model_kind: str, alphabet) -> float:
    if not dataset:
        return float("nan")
    correct = 0
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            correct += int(_predict(model, inputs, model_kind) == sample.label)
    return correct / len(dataset)


def _accuracy_by_length(model, dataset, device, model_kind, alphabet) -> Dict[int, float]:
    buckets: Dict[int, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            n = max(sample.a_count, sample.b_count)
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            pred = _predict(model, inputs, model_kind)
            buckets[n][1] += 1
            buckets[n][0] += int(pred == sample.label)
    return {n: c / t for n, (c, t) in sorted(buckets.items())}


def _accuracy_by_bucket(model, dataset, device, model_kind, alphabet) -> Dict[str, float]:
    buckets: Dict[str, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            bucket = length_bucket(len(sample.word))
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            pred = _predict(model, inputs, model_kind)
            buckets[bucket][1] += 1
            buckets[bucket][0] += int(pred == sample.label)
    return {b: c / t for b, (c, t) in sorted(buckets.items())}


def _accuracy_by_difficulty(model, dataset, device, model_kind, alphabet, difficulty: str) -> float | None:
    subset = [s for s in dataset if s.label == 0 and s.difficulty == difficulty]
    if not subset:
        return None
    return _accuracy(model, subset, device, model_kind, alphabet)


def _build_test_data(config: ExperimentConfig, alphabet, seed: int) -> List[Sample]:
    if config.difficulty_test:
        return generate_difficulty_dataset(
            config.pairs_per_difficulty_cell,
            seed=seed,
            language=config.language,
            alphabet=alphabet,
        )
    if config.stratified_test:
        return generate_stratified_dataset(
            config.pairs_per_bucket,
            seed=seed,
            language=config.language,
            alphabet=alphabet,
        )
    return generate_dataset(
        config.test_pairs,
        config.test_min_n,
        config.test_max_n,
        seed=seed,
        language=config.language,
        alphabet=alphabet,
    )


def _write_experiment_record(config, results, alphabet, output_dir) -> str | None:
    if not output_dir:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = output_path / f"{config.language}_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "language": config.language,
        "alphabet": list(alphabet),
        "device": results.device,
        "config": {
            "train_pairs": config.train_pairs,
            "test_pairs": config.test_pairs,
            "train_min_n": config.train_min_n,
            "train_max_n": config.train_max_n,
            "test_min_n": config.test_min_n,
            "test_max_n": config.test_max_n,
            "hidden_size": config.hidden_size,
            "beta": config.beta,
            "learn_beta": config.learn_beta,
            "lr": config.lr,
            "epochs": config.epochs,
            "seed": config.seed,
            "stratified_test": config.stratified_test,
            "difficulty_test": config.difficulty_test,
            "pairs_per_bucket": config.pairs_per_bucket,
            "pairs_per_difficulty_cell": config.pairs_per_difficulty_cell,
        },
        "history": results.history,
        "test_accuracy": {"rnn": results.rnn_accuracy, "snn": results.snn_accuracy},
        "accuracy_by_length": {
            "rnn": results.rnn_accuracy_by_length,
            "snn": results.snn_accuracy_by_length,
        },
        "accuracy_by_bucket": {
            "rnn": results.rnn_accuracy_by_bucket,
            "snn": results.snn_accuracy_by_bucket,
        },
        "difficulty_accuracy": {
            "rnn_hard": results.rnn_hard_neg_accuracy,
            "rnn_easy": results.rnn_easy_neg_accuracy,
            "snn_hard": results.snn_hard_neg_accuracy,
            "snn_easy": results.snn_easy_neg_accuracy,
        },
    }
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(file_path)


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    set_seed(config.seed)
    device = _device_name(config)
    alphabet = list(config.alphabet) if config.alphabet else get_language(config.language).alphabet

    train_data = generate_dataset(
        config.train_pairs,
        config.train_min_n,
        config.train_max_n,
        seed=config.seed,
        language=config.language,
        alphabet=alphabet,
    )
    test_data = _build_test_data(config, alphabet, config.seed + 1)

    input_size = len(alphabet)
    rnn_model = ClassicRNN(input_size=input_size, hidden_size=config.hidden_size).to(device)
    snn_model = SpikingNet(
        input_size=input_size,
        hidden_size=config.hidden_size,
        beta=config.beta,
        learn_beta=config.learn_beta,
    ).to(device)

    optimizer_rnn = torch.optim.Adam(rnn_model.parameters(), lr=config.lr)
    optimizer_snn = torch.optim.Adam(snn_model.parameters(), lr=config.lr)
    loss_rnn_fn = nn.CrossEntropyLoss()
    loss_snn_fn = SF.ce_rate_loss()
    history: Dict[str, List[float]] = {"rnn_loss": [], "snn_loss": []}

    for _ in range(config.epochs):
        rnn_model.train()
        snn_model.train()
        rnn_loss_total = 0.0
        snn_loss_total = 0.0
        for sample in train_data:
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            targets = torch.tensor([sample.label], dtype=torch.long, device=device)

            optimizer_rnn.zero_grad()
            loss_rnn = loss_rnn_fn(rnn_model(inputs), targets)
            loss_rnn.backward()
            optimizer_rnn.step()
            rnn_loss_total += loss_rnn.item()

            optimizer_snn.zero_grad()
            loss_snn = loss_snn_fn(snn_model(inputs), targets)
            loss_snn.backward()
            optimizer_snn.step()
            snn_loss_total += loss_snn.item()

        history["rnn_loss"].append(rnn_loss_total / len(train_data))
        history["snn_loss"].append(snn_loss_total / len(train_data))

    rnn_acc = _accuracy(rnn_model, test_data, device, "rnn", alphabet)
    snn_acc = _accuracy(snn_model, test_data, device, "snn", alphabet)

    results = ExperimentResults(
        device=str(device),
        history=history,
        rnn_accuracy=rnn_acc,
        snn_accuracy=snn_acc,
        rnn_accuracy_by_length=_accuracy_by_length(rnn_model, test_data, device, "rnn", alphabet),
        snn_accuracy_by_length=_accuracy_by_length(snn_model, test_data, device, "snn", alphabet),
        rnn_accuracy_by_bucket=_accuracy_by_bucket(rnn_model, test_data, device, "rnn", alphabet),
        snn_accuracy_by_bucket=_accuracy_by_bucket(snn_model, test_data, device, "snn", alphabet),
        rnn_hard_neg_accuracy=_accuracy_by_difficulty(rnn_model, test_data, device, "rnn", alphabet, "hard"),
        rnn_easy_neg_accuracy=_accuracy_by_difficulty(rnn_model, test_data, device, "rnn", alphabet, "easy"),
        snn_hard_neg_accuracy=_accuracy_by_difficulty(snn_model, test_data, device, "snn", alphabet, "hard"),
        snn_easy_neg_accuracy=_accuracy_by_difficulty(snn_model, test_data, device, "snn", alphabet, "easy"),
    )
    output_path = _write_experiment_record(config, results, alphabet, config.output_dir)
    return ExperimentResults(
        device=results.device,
        history=results.history,
        rnn_accuracy=results.rnn_accuracy,
        snn_accuracy=results.snn_accuracy,
        rnn_accuracy_by_length=results.rnn_accuracy_by_length,
        snn_accuracy_by_length=results.snn_accuracy_by_length,
        rnn_accuracy_by_bucket=results.rnn_accuracy_by_bucket,
        snn_accuracy_by_bucket=results.snn_accuracy_by_bucket,
        rnn_hard_neg_accuracy=results.rnn_hard_neg_accuracy,
        rnn_easy_neg_accuracy=results.rnn_easy_neg_accuracy,
        snn_hard_neg_accuracy=results.snn_hard_neg_accuracy,
        snn_easy_neg_accuracy=results.snn_easy_neg_accuracy,
        output_path=output_path,
    )


def _result_to_dict(results: ExperimentResults) -> Dict[str, Any]:
    return {
        "rnn_accuracy": results.rnn_accuracy,
        "snn_accuracy": results.snn_accuracy,
        "rnn_accuracy_by_bucket": results.rnn_accuracy_by_bucket,
        "snn_accuracy_by_bucket": results.snn_accuracy_by_bucket,
        "rnn_hard_neg_accuracy": results.rnn_hard_neg_accuracy,
        "rnn_easy_neg_accuracy": results.rnn_easy_neg_accuracy,
        "snn_hard_neg_accuracy": results.snn_hard_neg_accuracy,
        "snn_easy_neg_accuracy": results.snn_easy_neg_accuracy,
    }


def _mean_std(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return {"mean": mean, "std": 0.0}
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return {"mean": mean, "std": math.sqrt(var)}


def aggregate_results(records: List[Dict[str, Any]]) -> AggregatedResults:
    metrics: Dict[str, Dict[str, float]] = {}
    keys = [
        "rnn_accuracy",
        "snn_accuracy",
        "rnn_hard_neg_accuracy",
        "rnn_easy_neg_accuracy",
        "snn_hard_neg_accuracy",
        "snn_easy_neg_accuracy",
    ]
    for key in keys:
        stats = _mean_std([r.get(key) for r in records if r.get(key) is not None])
        metrics[key] = stats

    bucket_keys = set()
    for record in records:
        bucket_keys.update(record.get("rnn_accuracy_by_bucket", {}).keys())
    for bucket in sorted(bucket_keys):
        metrics[f"rnn_bucket_{bucket}"] = _mean_std(
            [r.get("rnn_accuracy_by_bucket", {}).get(bucket) for r in records]
        )
        metrics[f"snn_bucket_{bucket}"] = _mean_std(
            [r.get("snn_accuracy_by_bucket", {}).get(bucket) for r in records]
        )

    return AggregatedResults(num_seeds=len(records), metrics=metrics)


def run_multiseed_experiment(
    config: ExperimentConfig,
    num_seeds: int,
    seed_base: int = 42,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i in range(num_seeds):
        seed = seed_base + i
        run_config = replace(config, seed=seed)
        if not quiet:
            print(f"[seed {seed}] running {run_config.language}...")
        results = run_experiment(run_config)
        records.append(_result_to_dict(results))
    return records
