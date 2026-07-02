from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def _bucket_midpoint(label: str) -> float:
    if "-" not in label:
        return 0.0
    lo, hi = label.split("-", 1)
    return (int(lo) + int(hi)) / 2


def plot_length_accuracy(
    aggregated: Dict,
    output_path: Path,
    title: str = "Accuracy vs word length",
) -> None:
    buckets = sorted(
        {k.replace("rnn_bucket_", "").replace("snn_bucket_", "")
         for k in aggregated["metrics"] if "_bucket_" in k},
        key=_bucket_midpoint,
    )
    rnn_means = [aggregated["metrics"][f"rnn_bucket_{b}"]["mean"] for b in buckets]
    snn_means = [aggregated["metrics"][f"snn_bucket_{b}"]["mean"] for b in buckets]
    rnn_stds = [aggregated["metrics"][f"rnn_bucket_{b}"]["std"] for b in buckets]
    snn_stds = [aggregated["metrics"][f"snn_bucket_{b}"]["std"] for b in buckets]
    xs = [_bucket_midpoint(b) for b in buckets]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, rnn_means, yerr=rnn_stds, label="RNN", marker="o")
    ax.errorbar(xs, snn_means, yerr=snn_stds, label="SNN", marker="s")
    ax.set_xlabel("Word length (midpoint of bucket)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_sweep(aggregated: Dict, output_path: Path, title: str = "Beta sweep") -> None:
    betas = aggregated.get("betas", [])
    rnn = aggregated["metrics"]["rnn_accuracy"]["mean"]
    snn_means = [aggregated["metrics"][f"snn_beta_{b}"]["mean"] for b in betas]
    snn_stds = [aggregated["metrics"][f"snn_beta_{b}"]["std"] for b in betas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(rnn, color="C0", linestyle="--", label=f"RNN baseline ({rnn:.2f})")
    ax.errorbar(betas, snn_means, yerr=snn_stds, label="SNN", marker="o")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_breakdown(
    aggregated: Dict,
    output_path: Path,
    title: str = "Hard vs easy negative accuracy",
) -> None:
    labels = ["hard", "easy"]
    rnn_means = [aggregated["metrics"][f"rnn_{d}_neg_accuracy"]["mean"] for d in labels]
    snn_means = [aggregated["metrics"][f"snn_{d}_neg_accuracy"]["mean"] for d in labels]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], rnn_means, width, label="RNN")
    ax.bar([i + width / 2 for i in x], snn_means, width, label="SNN")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy on negatives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
