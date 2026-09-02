from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import defaultdict
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from snntorch import functional as SF

from .data import (
    Sample,
    generate_dataset,
    generate_difficulty_dataset,
    generate_stratified_dataset,
    length_bucket,
    word_to_tensor,
)
from .languages import get_language
from .models import ClassicRNN, SpikingNet, ClassicLSTM

MODEL_KINDS = ("rnn", "snn", "lstm")


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
    difficulty_min_n: int = 1
    difficulty_max_n: int = 10


@dataclass(frozen=True)
class ExperimentResults:
    device: str
    history: Dict[str, List[float]]
    rnn_accuracy: float
    snn_accuracy: float
    lstm_accuracy: float
    rnn_accuracy_by_length: Dict[int, float]
    snn_accuracy_by_length: Dict[int, float]
    lstm_accuracy_by_length: Dict[int, float]
    rnn_accuracy_by_bucket: Dict[str, float] = field(default_factory=dict)
    snn_accuracy_by_bucket: Dict[str, float] = field(default_factory=dict)
    lstm_accuracy_by_bucket: Dict[str, float] = field(default_factory=dict)
    rnn_hard_neg_accuracy: float | None = None
    rnn_easy_neg_accuracy: float | None = None
    snn_hard_neg_accuracy: float | None = None
    snn_easy_neg_accuracy: float | None = None
    lstm_hard_neg_accuracy: float | None = None
    lstm_easy_neg_accuracy: float | None = None
    rnn_strategy_neg_accuracy: Dict[int, float] = field(default_factory=dict)
    snn_strategy_neg_accuracy: Dict[int, float] = field(default_factory=dict)
    lstm_strategy_neg_accuracy: Dict[int, float] = field(default_factory=dict)
    difficulty_counts: Dict[str, int] = field(default_factory=dict)
    strategy_counts: Dict[int, int] = field(default_factory=dict)
    compute_budget: Dict[str, int] = field(default_factory=dict)
    output_path: str | None = None


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    payload = asdict(config)
    alphabet = payload.get("alphabet")
    if alphabet is not None:
        payload["alphabet"] = list(alphabet)
    return payload


@dataclass
class AggregatedResults:
    num_seeds: int
    metrics: Dict[str, Dict[str, float]]
    counts: Dict[str, Dict[str, float]]


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


def _predict(model, inputs: torch.Tensor, model_kind: str) -> int | float | bool | Any:
    if model_kind == "rnn":
        return model(inputs).argmax(dim=1).item()
    if model_kind == "snn":
        return _predict_snn(model, inputs).argmax(dim=1).item()
    if model_kind == "lstm":
        return model(inputs).argmax(dim=1).item()
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


def _accuracy_by_strategy(model, dataset, device, model_kind, alphabet) -> Dict[int, float]:
    buckets: Dict[int, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            if sample.label != 0 or sample.negative_strategy is None:
                continue
            strategy = int(sample.negative_strategy)
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            pred = _predict(model, inputs, model_kind)
            buckets[strategy][1] += 1
            buckets[strategy][0] += int(pred == sample.label)
    return {k: c / t for k, (c, t) in sorted(buckets.items()) if t > 0}


def _difficulty_and_strategy_counts(dataset: List[Sample]) -> Tuple[Dict[str, int], Dict[int, int]]:
    difficulty_counts: Dict[str, int] = defaultdict(int)
    strategy_counts: Dict[int, int] = defaultdict(int)
    for sample in dataset:
        if sample.label != 0:
            continue
        if sample.difficulty:
            difficulty_counts[sample.difficulty] += 1
        if sample.negative_strategy is not None:
            strategy_counts[int(sample.negative_strategy)] += 1
    return dict(sorted(difficulty_counts.items())), dict(sorted(strategy_counts.items()))


def _build_test_data(config: ExperimentConfig, seed: int) -> List[Sample]:
    if config.difficulty_test:
        return generate_difficulty_dataset(
            config.pairs_per_difficulty_cell,
            seed=seed,
            language=config.language,
            min_n=config.difficulty_min_n,
            max_n=config.difficulty_max_n,
        )
    if config.stratified_test:
        return generate_stratified_dataset(
            config.pairs_per_bucket,
            seed=seed,
            language=config.language,
        )
    return generate_dataset(
        config.test_pairs,
        config.test_min_n,
        config.test_max_n,
        seed=seed,
        language=config.language,
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
        "models": list(MODEL_KINDS),
        "config": config_to_dict(config),
        "history": results.history,
        "test_accuracy": {
            kind: getattr(results, f"{kind}_accuracy") for kind in MODEL_KINDS
        },
        "accuracy_by_length": {
            kind: getattr(results, f"{kind}_accuracy_by_length") for kind in MODEL_KINDS
        },
        "accuracy_by_bucket": {
            kind: getattr(results, f"{kind}_accuracy_by_bucket") for kind in MODEL_KINDS
        },
        "difficulty_accuracy": {
            f"{kind}_{diff}": getattr(results, f"{kind}_{diff}_neg_accuracy")
            for kind in MODEL_KINDS
            for diff in ("hard", "easy")
        },
        "strategy_accuracy": {
            kind: getattr(results, f"{kind}_strategy_neg_accuracy")
            for kind in MODEL_KINDS
        },
        "difficulty_counts": results.difficulty_counts,
        "strategy_counts": results.strategy_counts,
        "compute_budget": results.compute_budget,
    }
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(file_path)


def _format_run_header(config: ExperimentConfig) -> str:
    parts = [
        f"language={config.language}",
        f"seed={config.seed}",
        f"epochs={config.epochs}",
    ]
    if not config.learn_beta:
        parts.append(f"beta={config.beta}")
    return ", ".join(parts)


def run_experiment(config: ExperimentConfig, *, verbose: bool = True) -> ExperimentResults:
    set_seed(config.seed)
    device = _device_name(config)
    alphabet = list(config.alphabet) if config.alphabet else get_language(config.language).alphabet

    train_data = generate_dataset(
        config.train_pairs,
        config.train_min_n,
        config.train_max_n,
        seed=config.seed,
        language=config.language,
    )
    test_data = _build_test_data(config, config.seed + 1)

    if verbose:
        print(f"  training ({_format_run_header(config)})...", flush=True)

    input_size = len(alphabet)
    rnn_model = ClassicRNN(input_size=input_size, hidden_size=config.hidden_size).to(device)
    snn_model = SpikingNet(
        input_size=input_size,
        hidden_size=config.hidden_size,
        beta=config.beta,
        learn_beta=config.learn_beta,
    ).to(device)
    lstm_model = ClassicLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
    ).to(device)

    optimizer_rnn = torch.optim.Adam(rnn_model.parameters(), lr=config.lr)
    optimizer_snn = torch.optim.Adam(snn_model.parameters(), lr=config.lr)
    optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=config.lr)
    loss_rnn_fn = nn.CrossEntropyLoss()
    loss_snn_fn = SF.ce_rate_loss()
    loss_lstm_fn = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {"rnn_loss": [], "snn_loss": [], "lstm_loss": []}

    for epoch in range(1, config.epochs + 1):
        rnn_model.train()
        snn_model.train()
        lstm_model.train()
        rnn_loss_total = 0.0
        snn_loss_total = 0.0
        lstm_loss_total = 0.0
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

            optimizer_lstm.zero_grad()
            loss_lstm = loss_lstm_fn(lstm_model(inputs), targets)
            loss_lstm.backward()
            optimizer_lstm.step()
            lstm_loss_total += loss_lstm.item()

        history["rnn_loss"].append(rnn_loss_total / len(train_data))
        history["snn_loss"].append(snn_loss_total / len(train_data))
        history["lstm_loss"].append(lstm_loss_total / len(train_data))
        if verbose:
            print(
                f"    epoch {epoch}/{config.epochs}  "
                f"rnn={history['rnn_loss'][-1]:.4f}  "
                f"snn={history['snn_loss'][-1]:.4f}  "
                f"lstm={history['lstm_loss'][-1]:.4f}",
                flush=True,
            )

    rnn_acc = _accuracy(rnn_model, test_data, device, "rnn", alphabet)
    snn_acc = _accuracy(snn_model, test_data, device, "snn", alphabet)
    lstm_acc = _accuracy(lstm_model, test_data, device, "lstm", alphabet)

    if verbose:
        print(
            f"  eval  rnn={rnn_acc:.1%}  snn={snn_acc:.1%}  lstm={lstm_acc:.1%}",
            flush=True,
        )

    difficulty_counts, strategy_counts = _difficulty_and_strategy_counts(test_data)
    results = ExperimentResults(
        device=str(device),
        history=history,
        rnn_accuracy=rnn_acc,
        snn_accuracy=snn_acc,
        lstm_accuracy=lstm_acc,
        rnn_accuracy_by_length=_accuracy_by_length(rnn_model, test_data, device, "rnn", alphabet),
        snn_accuracy_by_length=_accuracy_by_length(snn_model, test_data, device, "snn", alphabet),
        lstm_accuracy_by_length=_accuracy_by_length(lstm_model, test_data, device, "lstm", alphabet),
        rnn_accuracy_by_bucket=_accuracy_by_bucket(rnn_model, test_data, device, "rnn", alphabet),
        snn_accuracy_by_bucket=_accuracy_by_bucket(snn_model, test_data, device, "snn", alphabet),
        lstm_accuracy_by_bucket=_accuracy_by_bucket(lstm_model, test_data, device, "lstm", alphabet),
        rnn_hard_neg_accuracy=_accuracy_by_difficulty(rnn_model, test_data, device, "rnn", alphabet, "hard"),
        rnn_easy_neg_accuracy=_accuracy_by_difficulty(rnn_model, test_data, device, "rnn", alphabet, "easy"),
        snn_hard_neg_accuracy=_accuracy_by_difficulty(snn_model, test_data, device, "snn", alphabet, "hard"),
        snn_easy_neg_accuracy=_accuracy_by_difficulty(snn_model, test_data, device, "snn", alphabet, "easy"),
        lstm_hard_neg_accuracy=_accuracy_by_difficulty(lstm_model, test_data, device, "lstm", alphabet, "hard"),
        lstm_easy_neg_accuracy=_accuracy_by_difficulty(lstm_model, test_data, device, "lstm", alphabet, "easy"),
        rnn_strategy_neg_accuracy=_accuracy_by_strategy(rnn_model, test_data, device, "rnn", alphabet),
        snn_strategy_neg_accuracy=_accuracy_by_strategy(snn_model, test_data, device, "snn", alphabet),
        lstm_strategy_neg_accuracy=_accuracy_by_strategy(lstm_model, test_data, device, "lstm", alphabet),
        difficulty_counts=difficulty_counts,
        strategy_counts=strategy_counts,
        compute_budget={
            "epochs": config.epochs,
            "train_samples_per_epoch": len(train_data),
            "test_samples": len(test_data),
            "updates_per_model": config.epochs * len(train_data),
            "total_model_updates": config.epochs * len(train_data) * len(MODEL_KINDS),
        },
    )
    output_path = _write_experiment_record(config, results, alphabet, config.output_dir)
    return replace(results, output_path=output_path)


def _result_to_dict(results: ExperimentResults, *, seed: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"seed": seed}
    for kind in MODEL_KINDS:
        record[f"{kind}_accuracy"] = getattr(results, f"{kind}_accuracy")
        record[f"{kind}_accuracy_by_bucket"] = getattr(results, f"{kind}_accuracy_by_bucket")
        for diff in ("hard", "easy"):
            record[f"{kind}_{diff}_neg_accuracy"] = getattr(results, f"{kind}_{diff}_neg_accuracy")
        for strategy, score in getattr(results, f"{kind}_strategy_neg_accuracy").items():
            record[f"{kind}_strategy_{strategy}_neg_accuracy"] = score
    for difficulty, count in results.difficulty_counts.items():
        record[f"difficulty_{difficulty}_count"] = float(count)
    for strategy, count in results.strategy_counts.items():
        record[f"strategy_{strategy}_count"] = float(count)
    for key, value in results.compute_budget.items():
        record[f"compute_{key}"] = float(value)
    return record


def _mean_std(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return {"mean": mean, "std": 0.0}
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return {"mean": mean, "std": math.sqrt(var)}


def _mean_std_ci(values: List[float], z: float = 1.96) -> Dict[str, float]:
    stats = _mean_std(values)
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {**stats, "n": 0, "ci_low": float("nan"), "ci_high": float("nan")}
    if len(clean) == 1:
        mean = clean[0]
        return {**stats, "n": 1, "ci_low": mean, "ci_high": mean}
    margin = z * stats["std"] / math.sqrt(len(clean))
    return {
        **stats,
        "n": len(clean),
        "ci_low": stats["mean"] - margin,
        "ci_high": stats["mean"] + margin,
    }


def _safe_float_list(records: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _two_sided_sign_test(diffs: List[float]) -> float:
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    pos = sum(1 for d in nonzero if d > 0)
    k = min(pos, n - pos)
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += math.comb(n, i)
    p = min(1.0, 2.0 * cumulative / (2 ** n))
    return p


def _paired_effect_size_dz(diffs: List[float]) -> float:
    if not diffs:
        return float("nan")
    mean = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        return float("nan")
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return mean / sd


def _holm_adjust(p_values: List[float]) -> List[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        candidate = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def _build_inference_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ci_metrics: Dict[str, Dict[str, float]] = {}
    for key in _aggregate_metric_keys(records):
        ci_metrics[key] = _mean_std_ci(_safe_float_list(records, key))

    comparisons: List[Dict[str, Any]] = []
    comparison_specs: List[Tuple[str, str, str]] = [
        ("overall", "snn", "rnn"),
        ("overall", "snn", "lstm"),
    ]
    for difficulty in ("hard", "easy"):
        comparison_specs.append((difficulty, "snn", "rnn"))
        comparison_specs.append((difficulty, "snn", "lstm"))

    raw_p_values: List[float] = []
    for metric, lhs, rhs in comparison_specs:
        if metric == "overall":
            lhs_key = f"{lhs}_accuracy"
            rhs_key = f"{rhs}_accuracy"
        else:
            lhs_key = f"{lhs}_{metric}_neg_accuracy"
            rhs_key = f"{rhs}_{metric}_neg_accuracy"
        paired: List[Tuple[float, float]] = []
        for record in records:
            lv = record.get(lhs_key)
            rv = record.get(rhs_key)
            if lv is None or rv is None:
                continue
            paired.append((float(lv), float(rv)))
        diffs = [l - r for l, r in paired]
        p_raw = _two_sided_sign_test(diffs)
        raw_p_values.append(p_raw)
        comparisons.append(
            {
                "metric": metric,
                "lhs": lhs,
                "rhs": rhs,
                "n": len(paired),
                "delta_mean": (sum(diffs) / len(diffs)) if diffs else float("nan"),
                "effect_size_dz": _paired_effect_size_dz(diffs),
                "p_value_sign_test": p_raw,
            }
        )

    p_adjusted = _holm_adjust(raw_p_values) if raw_p_values else []
    for comp, adj in zip(comparisons, p_adjusted):
        comp["p_value_holm"] = adj

    return {
        "confidence_intervals": ci_metrics,
        "paired_comparisons": comparisons,
        "multiple_comparisons": {"method": "holm_bonferroni"},
    }


def _aggregate_metric_keys(records: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for record in records:
        keys.update(
            key
            for key in record
            if key.endswith("_accuracy") and key not in {"seed"} and "_by_" not in key
        )
    return sorted(keys)


def aggregate_results(records: List[Dict[str, Any]]) -> AggregatedResults:
    metrics: Dict[str, Dict[str, float]] = {}
    for key in _aggregate_metric_keys(records):
        stats = _mean_std([r.get(key) for r in records if r.get(key) is not None])
        metrics[key] = stats

    bucket_keys = set()
    for record in records:
        for kind in MODEL_KINDS:
            bucket_keys.update(record.get(f"{kind}_accuracy_by_bucket", {}).keys())
    for bucket in sorted(bucket_keys):
        for kind in MODEL_KINDS:
            metrics[f"{kind}_bucket_{bucket}"] = _mean_std(
                [r.get(f"{kind}_accuracy_by_bucket", {}).get(bucket) for r in records]
            )
    count_keys = sorted(
        {
            key
            for record in records
            for key in record
            if key.endswith("_count") or key.startswith("compute_")
        }
    )
    counts = {key: _mean_std(_safe_float_list(records, key)) for key in count_keys}
    return AggregatedResults(num_seeds=len(records), metrics=metrics, counts=counts)


def build_seedagg_payload(
        *,
        experiment: str,
        config: ExperimentConfig,
        records: List[Dict[str, Any]],
        num_seeds: int,
        seed_base: int,
        **extra: Any,
) -> Dict[str, Any]:
    agg = aggregate_results(records)
    return {
        "experiment": experiment,
        "language": config.language,
        "num_seeds": num_seeds,
        "seed_base": seed_base,
        "models": list(MODEL_KINDS),
        "config": config_to_dict(config),
        "metrics": agg.metrics,
        "counts": agg.counts,
        "stats": _build_inference_stats(records),
        **extra,
    }


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
            print(
                f"[{i + 1}/{num_seeds}] seed {seed}  {_format_run_header(run_config)}",
                flush=True,
            )
        results = run_experiment(run_config, verbose=not quiet)
        records.append(_result_to_dict(results, seed=seed))
    return records
