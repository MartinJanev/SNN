from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT
from .training import ExperimentConfig, run_experiment


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def _coerce_value(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_overrides(config_dict: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected section.key=value")
        path_str, value_str = override.split("=", 1)
        keys = [k for k in path_str.split(".") if k]
        cur = config_dict
        for key in keys[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[keys[-1]] = _coerce_value(value_str)
    return config_dict


def _get_nested(config_dict: dict, keys: list[str], default):
    cur = config_dict
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def config_from_dict(config_dict: dict) -> ExperimentConfig:
    alphabet = _get_nested(config_dict, ["experiment", "alphabet"], None)
    return ExperimentConfig(
        language=_get_nested(config_dict, ["experiment", "language"], "anbn"),
        alphabet=list(alphabet) if alphabet else None,
        output_dir=_get_nested(config_dict, ["experiment", "output_dir"], "outputs"),
        train_pairs=_get_nested(config_dict, ["training", "train_pairs"], 400),
        test_pairs=_get_nested(config_dict, ["training", "test_pairs"], 100),
        train_min_n=_get_nested(config_dict, ["training", "train_min_n"], 1),
        train_max_n=_get_nested(config_dict, ["training", "train_max_n"], 10),
        test_min_n=_get_nested(config_dict, ["training", "test_min_n"], 1),
        test_max_n=_get_nested(config_dict, ["training", "test_max_n"], 80),
        epochs=_get_nested(config_dict, ["training", "epochs"], 5),
        lr=_get_nested(config_dict, ["training", "learning_rate"], 0.01),
        seed=_get_nested(config_dict, ["training", "seed"], 42),
        hidden_size=_get_nested(config_dict, ["model", "hidden_size"], 32),
        beta=_get_nested(config_dict, ["model", "beta"], 0.85),
        learn_beta=_get_nested(config_dict, ["model", "learn_beta"], True),
        device=_get_nested(config_dict, ["device", "device"], None),
        stratified_test=_get_nested(config_dict, ["experiment", "stratified_test"], False),
        difficulty_test=_get_nested(config_dict, ["experiment", "difficulty_test"], False),
        pairs_per_bucket=_get_nested(config_dict, ["experiment", "pairs_per_bucket"], 50),
        pairs_per_difficulty_cell=_get_nested(
            config_dict, ["experiment", "pairs_per_difficulty_cell"], 5
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare RNN and SNN models on formal languages.",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, e.g. --set training.epochs=10",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_dict = load_config(args.config)
    config_dict = _apply_overrides(config_dict, args.set)
    config = config_from_dict(config_dict)
    results = run_experiment(config)
    print(f"RNN accuracy: {results.rnn_accuracy * 100:.2f}%")
    print(f"SNN accuracy: {results.snn_accuracy * 100:.2f}%")
    if results.output_path:
        print(f"Saved: {results.output_path}")
