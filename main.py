from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from formal_language_snn.training import ExperimentConfig, run_experiment


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare RNN and SNN models on formal languages.",
        epilog="Config can be set in config.yaml; CLI args override config file settings.",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )

    # Experiment parameters
    parser.add_argument("--language", type=str, help="Formal language to test")
    parser.add_argument("--alphabet", type=str, help="Optional explicit alphabet")
    parser.add_argument("--train-pairs", type=int, help="Number of training pairs")
    parser.add_argument("--test-pairs", type=int, help="Number of testing pairs")
    parser.add_argument("--train-min-n", type=int, help="Minimum n for training")
    parser.add_argument("--train-max-n", type=int, help="Maximum n for training")
    parser.add_argument("--test-min-n", type=int, help="Minimum n for testing")
    parser.add_argument("--test-max-n", type=int, help="Maximum n for testing")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--hidden-size", type=int, help="Hidden layer size")
    parser.add_argument("--beta", type=float, help="SNN beta (decay factor)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Compute device (cpu/cuda/mps)")

    return parser


def merge_config_and_args(config_dict: dict, args: argparse.Namespace) -> dict:
    """Merge config file with command-line args (CLI takes precedence)."""
    # Extract nested config sections
    exp_config = config_dict.get("experiment", {})
    train_config = config_dict.get("training", {})
    model_config = config_dict.get("model", {})
    device_config = config_dict.get("device", {})

    # Build merged config, with CLI args overriding file values
    merged = {
        "language": args.language or exp_config.get("language", "anbn"),
        "alphabet": args.alphabet or exp_config.get("alphabet"),
        "train_pairs": args.train_pairs or train_config.get("train_pairs", 400),
        "test_pairs": args.test_pairs or train_config.get("test_pairs", 100),
        "train_min_n": args.train_min_n or train_config.get("train_min_n", 1),
        "train_max_n": args.train_max_n or train_config.get("train_max_n", 5),
        "test_min_n": args.test_min_n or train_config.get("test_min_n", 6),
        "test_max_n": args.test_max_n or train_config.get("test_max_n", 10),
        "epochs": args.epochs or train_config.get("epochs", 5),
        "lr": args.lr or train_config.get("learning_rate", 0.01),
        "hidden_size": args.hidden_size or model_config.get("hidden_size", 16),
        "beta": args.beta or model_config.get("beta", 0.85),
        "seed": args.seed or train_config.get("seed", 42),
        "device": args.device or device_config.get("device"),
    }

    return merged


def main() -> None:
    args = build_parser().parse_args()

    # Load config file
    try:
        config_dict = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a configs/default.yaml file or specify one with --config")
        return

    # Merge config file with CLI args (CLI takes precedence)
    merged = merge_config_and_args(config_dict, args)

    print("=" * 70)
    print(f"Starting Experiment: RNN vs SNN on formal language '{merged['language']}'")
    print("=" * 70)

    config = ExperimentConfig(
        train_pairs=merged["train_pairs"],
        test_pairs=merged["test_pairs"],
        train_min_n=merged["train_min_n"],
        train_max_n=merged["train_max_n"],
        test_min_n=merged["test_min_n"],
        test_max_n=merged["test_max_n"],
        hidden_size=merged["hidden_size"],
        beta=merged["beta"],
        lr=merged["lr"],
        epochs=merged["epochs"],
        seed=merged["seed"],
        device=merged["device"],
        language=merged["language"],
        alphabet=list(merged["alphabet"]) if merged["alphabet"] else None,
    )

    print("\n[CONFIG]")
    print(f"  Language: {config.language}")
    print(f"  Train set: {config.train_pairs} pairs, n in [{config.train_min_n}, {config.train_max_n}]")
    print(f"  Test set:  {config.test_pairs} pairs, n in [{config.test_min_n}, {config.test_max_n}]")
    print(f"  Hidden size: {config.hidden_size} | Beta (SNN): {config.beta} | LR: {config.lr} | Epochs: {config.epochs}")
    print()

    print("[RUNNING]")
    results = run_experiment(config)

    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)

    print(f"\n[DEVICE] {results.device}")

    print(f"\n[TRAINING HISTORY]")
    for idx, (rnn_loss, snn_loss) in enumerate(zip(results.history["rnn_loss"], results.history["snn_loss"]), start=1):
        print(f"  Epoch {idx}: RNN loss={rnn_loss:.4f} | SNN loss={snn_loss:.4f}")

    print(f"\n[TEST ACCURACY (longer words)]")
    print(f"  RNN: {results.rnn_accuracy * 100:.2f}%")
    print(f"  SNN: {results.snn_accuracy * 100:.2f}%")

    print(f"\n[ACCURACY BY WORD LENGTH]")
    all_lengths = sorted(set(results.rnn_accuracy_by_length) | set(results.snn_accuracy_by_length))
    for length in all_lengths:
        rnn_acc = results.rnn_accuracy_by_length.get(length, float("nan"))
        snn_acc = results.snn_accuracy_by_length.get(length, float("nan"))
        print(f"  n={length}: RNN={rnn_acc * 100:.2f}% | SNN={snn_acc * 100:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
