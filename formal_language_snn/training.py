from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import defaultdict
from itertools import combinations
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

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
from .models import ClassicRNN, SpikingNet, ClassicLSTM, SpikingRNN
from .models.capacity import count_parameters, matched_hidden_sizes


@dataclass(frozen=True)
class ModelSpec:
    """Everything that differs between model families, one row per family.

    ``build`` takes the input width, the hidden size and the config (only the spiking
    families read anything from it). ``make_loss`` is the loss constructor. ``logits``
    turns a forward pass into a ``[1, num_classes]`` score tensor -- the rate-coded
    families sum their spike train over time, the gated ones already return scores.
    The training loop feeds the raw forward pass to the loss, not ``logits``, because
    snntorch's rate losses do their own reduction over the time axis.
    """

    build: Callable[[int, int, "ExperimentConfig"], nn.Module]
    make_loss: Callable[[], Callable[..., torch.Tensor]]
    logits: Callable[[nn.Module, torch.Tensor], torch.Tensor]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "rnn": ModelSpec(
        build=lambda input_size, hidden, config: ClassicRNN(
            input_size=input_size, hidden_size=hidden
        ),
        make_loss=nn.CrossEntropyLoss,
        logits=lambda model, inputs: model(inputs),
    ),
    "snn": ModelSpec(
        build=lambda input_size, hidden, config: SpikingNet(
            input_size=input_size,
            hidden_size=hidden,
            beta=config.beta,
            learn_beta=config.learn_beta,
        ),
        make_loss=SF.ce_rate_loss,
        logits=lambda model, inputs: model(inputs).sum(dim=0),
    ),
    "lstm": ModelSpec(
        build=lambda input_size, hidden, config: ClassicLSTM(
            input_size=input_size, hidden_size=hidden
        ),
        make_loss=nn.CrossEntropyLoss,
        logits=lambda model, inputs: model(inputs),
    ),
    # Appended last on purpose: MODEL_SPECS order is RNG-consumption order, so adding the
    # recurrent spiking control here leaves the other three models' numbers untouched.
    "rsnn": ModelSpec(
        build=lambda input_size, hidden, config: SpikingRNN(
            input_size=input_size,
            hidden_size=hidden,
            beta=config.beta,
            learn_beta=config.learn_beta,
        ),
        make_loss=SF.ce_rate_loss,
        logits=lambda model, inputs: model(inputs).sum(dim=0),
    ),
}

# Declaration order is construction order, and construction order consumes the seeded RNG,
# so appending to MODEL_SPECS is safe but reordering it changes every number.
MODEL_KINDS = tuple(MODEL_SPECS)

# The families whose membrane decay beta is a real hyperparameter. Experiment 2 sweeps beta,
# which has no analogue in a gated model, so its sweep points train only these -- the gated
# models still appear in every exp2 table and figure, as reference lines from the baseline
# block rather than as ten discarded retrainings per language.
SPIKING_KINDS = ("snn", "rsnn")


@dataclass(frozen=True)
class ExperimentConfig:
    train_pairs: int = 400
    test_pairs: int = 100
    train_min_n: int = 1
    train_max_n: int = 10
    train_max_word_len: int | None = None
    test_min_n: int = 1
    test_max_n: int = 80
    hidden_size: int = 32
    # Which model families this run trains. Defaults to all of them; an experiment that
    # sweeps a factor only some families have narrows it, and the roster actually trained
    # is recorded in every output file so a narrowed run cannot be mistaken for a full one.
    models: Tuple[str, ...] = MODEL_KINDS
    match_capacity: bool = False
    beta: float = 0.85
    learn_beta: bool = True
    lr: float = 0.01
    epochs: int = 5
    val_fraction: float = 0.1
    patience: int = 3
    grad_clip: float | None = 1.0
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
    difficulty_min_word_len: int | None = None
    difficulty_max_word_len: int | None = None


@dataclass(frozen=True)
class ExperimentResults:
    device: str
    history: Dict[str, List[float]]
    # Every per-model field below is keyed by model kind, so adding a model is a
    # MODEL_SPECS entry rather than a new field on each of these.
    accuracy: Dict[str, float]
    accuracy_by_length: Dict[str, Dict[int, float]]
    models: Tuple[str, ...] = MODEL_KINDS
    accuracy_by_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # [kind]["hard"|"easy"] -> accuracy on that difficulty's negatives, or None if absent.
    difficulty_accuracy: Dict[str, Dict[str, float | None]] = field(default_factory=dict)
    strategy_neg_accuracy: Dict[str, Dict[int, float]] = field(default_factory=dict)
    difficulty_counts: Dict[str, int] = field(default_factory=dict)
    strategy_counts: Dict[int, int] = field(default_factory=dict)
    compute_budget: Dict[str, int] = field(default_factory=dict)
    epochs_trained: Dict[str, int] = field(default_factory=dict)
    model_capacity: Dict[str, Dict[str, int]] = field(default_factory=dict)
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


def _predict(model, inputs: torch.Tensor, model_kind: str) -> int:
    return MODEL_SPECS[model_kind].logits(model, inputs).argmax(dim=1).item()


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
            # Key on the realised length, matching _accuracy_by_bucket. Keying on the
            # generator's n instead made the two disagree for reber (length 2n +/- 2) and
            # for every negative whose mutation changes length.
            n = len(sample.word)
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
            min_word_len=config.difficulty_min_word_len,
            max_word_len=config.difficulty_max_word_len,
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
        "models": list(results.models),
        "model_capacity": results.model_capacity,
        "config": config_to_dict(config),
        "history": results.history,
        "test_accuracy": results.accuracy,
        "accuracy_by_length": results.accuracy_by_length,
        "accuracy_by_bucket": results.accuracy_by_bucket,
        "difficulty_accuracy": {
            f"{kind}_{diff}": results.difficulty_accuracy[kind][diff]
            for kind in results.models
            for diff in ("hard", "easy")
        },
        "strategy_accuracy": results.strategy_neg_accuracy,
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
        max_word_len=config.train_max_word_len,
    )
    test_data = _build_test_data(config, config.seed + 1)

    if verbose:
        print(f"  training ({_format_run_header(config)})...", flush=True)

    input_size = len(alphabet)
    # At a shared hidden size the GRU carries ~28x the SNN's parameters, by a factor that
    # itself varies with alphabet size -- so capacity differs both between models and
    # between languages. match_capacity instead equalises the parameter budget against the
    # GRU at config.hidden_size, leaving architecture as the only remaining difference.
    roster = tuple(config.models)
    unknown = [kind for kind in roster if kind not in MODEL_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown model kind(s) {unknown}; known kinds are {list(MODEL_SPECS)}"
        )
    if config.match_capacity:
        hidden = matched_hidden_sizes(input_size, "rnn", config.hidden_size)
    else:
        hidden = {kind: config.hidden_size for kind in roster}

    models = {
        kind: MODEL_SPECS[kind].build(input_size, hidden[kind], config).to(device)
        for kind in roster
    }
    model_capacity = {
        "hidden_size": {kind: hidden[kind] for kind in roster},
        "parameters": {kind: count_parameters(models[kind]) for kind in roster},
    }

    optimizers = {k: torch.optim.Adam(m.parameters(), lr=config.lr) for k, m in models.items()}
    losses = {k: MODEL_SPECS[k].make_loss() for k in models}
    history: Dict[str, List[float]] = {f"{k}_loss": [] for k in models}

    # Hold out a validation slice for early stopping. It is drawn from the training set, so it
    # shares the training length range -- stopping must never be able to see the extrapolation
    # regime it will later be tested on. train_data is already shuffled and class-balanced.
    val_size = int(len(train_data) * config.val_fraction) if config.val_fraction > 0 else 0
    val_data = train_data[:val_size]
    fit_data = train_data[val_size:] if val_size else train_data

    # Every model gets the same optimiser, learning rate, gradient clip, epoch cap and patience;
    # only the epoch each one *stops* at differs, and that is recorded and reported. Training all
    # models for a fixed number of epochs instead would leave the baselines open to the charge of
    # being undertrained, which is the objection this exists to close.
    best_state = {k: None for k in models}
    best_val = {k: -1.0 for k in models}
    stale = {k: 0 for k in models}
    epochs_trained = {k: 0 for k in models}
    active = [k for k in roster]

    for epoch in range(1, config.epochs + 1):
        if not active:
            break
        for kind in active:
            models[kind].train()
        totals = {k: 0.0 for k in models}
        for sample in fit_data:
            inputs = word_to_tensor(sample.word, alphabet=alphabet).to(device)
            targets = torch.tensor([sample.label], dtype=torch.long, device=device)
            for kind in active:
                optimizers[kind].zero_grad()
                loss = losses[kind](models[kind](inputs), targets)
                loss.backward()
                if config.grad_clip:
                    nn.utils.clip_grad_norm_(models[kind].parameters(), config.grad_clip)
                optimizers[kind].step()
                totals[kind] += loss.item()

        for kind in models:
            history[f"{kind}_loss"].append(
                totals[kind] / len(fit_data) if kind in active else float("nan")
            )
            if kind not in active:
                continue
            epochs_trained[kind] = epoch
            if not val_data:
                continue
            score = _accuracy(models[kind], val_data, device, kind, alphabet)
            if score > best_val[kind]:
                best_val[kind] = score
                best_state[kind] = {
                    name: tensor.detach().clone()
                    for name, tensor in models[kind].state_dict().items()
                }
                stale[kind] = 0
            else:
                stale[kind] += 1
        active = [k for k in active if stale[k] <= config.patience]

        if verbose:
            cells = "  ".join(
                f"{k}={history[f'{k}_loss'][-1]:.4f}"
                + (f"/{best_val[k]:.2f}" if val_data and best_val[k] >= 0 else "")
                for k in models
            )
            print(f"    epoch {epoch}/{config.epochs}  {cells}", flush=True)

    # Evaluate the best checkpoint, not the last one -- otherwise patience epochs of
    # overfitting past the optimum are what gets reported.
    for kind, state in best_state.items():
        if state is not None:
            models[kind].load_state_dict(state)

    accuracy = {
        kind: _accuracy(models[kind], test_data, device, kind, alphabet) for kind in roster
    }

    if verbose:
        print(
            "  eval  " + "  ".join(f"{k}={accuracy[k]:.1%}" for k in roster),
            flush=True,
        )

    difficulty_counts, strategy_counts = _difficulty_and_strategy_counts(test_data)
    results = ExperimentResults(
        models=tuple(roster),
        model_capacity=model_capacity,
        device=str(device),
        history=history,
        accuracy=accuracy,
        accuracy_by_length={
            kind: _accuracy_by_length(models[kind], test_data, device, kind, alphabet)
            for kind in roster
        },
        accuracy_by_bucket={
            kind: _accuracy_by_bucket(models[kind], test_data, device, kind, alphabet)
            for kind in roster
        },
        difficulty_accuracy={
            kind: {
                diff: _accuracy_by_difficulty(
                    models[kind], test_data, device, kind, alphabet, diff
                )
                for diff in ("hard", "easy")
            }
            for kind in roster
        },
        strategy_neg_accuracy={
            kind: _accuracy_by_strategy(models[kind], test_data, device, kind, alphabet)
            for kind in roster
        },
        difficulty_counts=difficulty_counts,
        strategy_counts=strategy_counts,
        epochs_trained=epochs_trained,
        compute_budget={
            "epoch_cap": config.epochs,
            "train_samples_per_epoch": len(fit_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "total_model_updates": sum(epochs_trained.values()) * len(fit_data),
        },
    )
    output_path = _write_experiment_record(config, results, alphabet, config.output_dir)
    return replace(results, output_path=output_path)


def _result_to_dict(results: ExperimentResults, *, seed: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"seed": seed}
    for kind in results.models:
        record[f"{kind}_accuracy"] = results.accuracy[kind]
        record[f"{kind}_accuracy_by_bucket"] = results.accuracy_by_bucket[kind]
        for diff in ("hard", "easy"):
            record[f"{kind}_{diff}_neg_accuracy"] = results.difficulty_accuracy[kind][diff]
        for strategy, score in results.strategy_neg_accuracy[kind].items():
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

    # Every ordered-by-roster pair of models, on every metric -- not just the SNN against the
    # two baselines. Reporting only the comparisons involving one model leaves a reviewer
    # unable to check whether the baselines differ from each other, and it understates the
    # Holm family, which makes the surviving p-values look stronger than they are.
    kinds = _record_model_kinds(records)
    comparisons: List[Dict[str, Any]] = []
    comparison_specs: List[Tuple[str, str, str]] = [
        (metric, lhs, rhs)
        for metric in ("overall", "hard", "easy")
        for lhs, rhs in combinations(kinds, 2)
    ]

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
        # The Holm family is stated explicitly so the correction can be reported, and
        # audited, rather than inferred from the length of the list above.
        "multiple_comparisons": {
            "method": "holm_bonferroni",
            "family_size": len(raw_p_values),
            "family": (
                "all unordered model pairs x {overall, hard, easy} negatives, "
                "within one language's seed block"
            ),
            "models": list(kinds),
            "metrics": ["overall", "hard", "easy"],
            "test": "two-sided exact sign test over paired seeds",
        },
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


def _record_model_kinds(records: List[Dict[str, Any]]) -> Tuple[str, ...]:
    """Model kinds present in these seed records, in MODEL_SPECS order."""
    present = {
        key[: -len("_accuracy_by_bucket")]
        for record in records
        for key in record
        if key.endswith("_accuracy_by_bucket")
    }
    return tuple(kind for kind in MODEL_SPECS if kind in present)


def aggregate_results(records: List[Dict[str, Any]]) -> AggregatedResults:
    metrics: Dict[str, Dict[str, float]] = {}
    for key in _aggregate_metric_keys(records):
        stats = _mean_std([r.get(key) for r in records if r.get(key) is not None])
        metrics[key] = stats

    # Read the roster back off the records rather than assuming MODEL_KINDS, so aggregating
    # a narrowed run does not invent empty rows for models it never trained.
    kinds = _record_model_kinds(records)
    bucket_keys = set()
    for record in records:
        for kind in kinds:
            bucket_keys.update(record.get(f"{kind}_accuracy_by_bucket", {}).keys())
    for bucket in sorted(bucket_keys):
        for kind in kinds:
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
        "models": list(config.models),
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
