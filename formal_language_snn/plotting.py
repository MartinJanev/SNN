from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

MODEL_COLORS = {"rnn": "#1f77b4", "snn": "#ff7f0e", "lstm": "#2ca02c"}
MODEL_MARKERS = {"rnn": "o", "snn": "s", "lstm": "^"}
MODEL_LINESTYLES = {"rnn": "-", "snn": "-", "lstm": "--"}
MODELS = ("rnn", "snn", "lstm")
DIFFICULTY_LABELS = ("hard", "easy")


def _bucket_bounds(label: str) -> tuple[int, int]:
    lo, hi = label.split("-", 1)
    return int(lo), int(hi)


def _bucket_midpoint(label: str) -> float:
    lo, hi = _bucket_bounds(label)
    return (lo + hi) / 2


def _bucket_label(metric_key: str) -> str:
    return metric_key.split("_bucket_", 1)[1]


def _is_length_bucket(label: str) -> bool:
    if "-" not in label:
        return False
    lo, hi = label.split("-", 1)
    return lo.isdigit() and hi.isdigit()


def _default_title(aggregated: Dict, prefix: str) -> str:
    return f"{prefix}: {aggregated.get('language', 'unknown')}"


def _subtitle(aggregated: Dict) -> str:
    n = aggregated.get("num_seeds", 1)
    return f"{n} seed{'s' if n != 1 else ''}"


def _apply_common_axes(ax, *, ylim: tuple[float, float] = (0, 1), chance_line: bool = True) -> None:
    ax.set_ylim(*ylim)
    if chance_line:
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="_nolegend_")
    ax.grid(True, alpha=0.3)


def _resolve_title(aggregated: Dict, prefix: str, title: str | None, default_title: str) -> str:
    if title is None or title == default_title:
        return _default_title(aggregated, prefix)
    return title


def _add_subtitle(fig, aggregated: Dict) -> None:
    fig.text(0.5, 0.01, _subtitle(aggregated), ha="center", fontsize=9, color="gray")


def _length_buckets(metrics: Dict) -> List[str]:
    buckets = sorted(
        {
            _bucket_label(k)
            for k in metrics
            if "_bucket_" in k and _is_length_bucket(_bucket_label(k)) and _bucket_label(k) != "other"
        },
        key=_bucket_midpoint,
    )
    return buckets


def _beta_metric_key(metrics: Dict, beta: float) -> str:
    candidates = [
        f"snn_beta_{beta}",
        f"snn_beta_{beta:.1f}",
        f"snn_beta_{round(beta, 1)}",
    ]
    for key in candidates:
        if key in metrics:
            return key
    for key in metrics:
        if not key.startswith("snn_beta_"):
            continue
        suffix = key.removeprefix("snn_beta_")
        try:
            if abs(float(suffix) - beta) < 1e-6:
                return key
        except ValueError:
            continue
    raise KeyError(f"No metric for beta={beta}")


def _plot_length_accuracy_ax(ax, aggregated: Dict, *, show_legend: bool = True) -> None:
    metrics = aggregated["metrics"]
    buckets = _length_buckets(metrics)
    xs = list(range(len(buckets)))

    for model in MODELS:
        means = [metrics[f"{model}_bucket_{b}"]["mean"] for b in buckets]
        stds = [metrics[f"{model}_bucket_{b}"]["std"] for b in buckets]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            label=model.upper(),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            capsize=3,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Word length bucket")
    ax.set_ylabel("Accuracy")
    _apply_common_axes(ax)
    if show_legend:
        ax.legend()


def plot_length_accuracy(
    aggregated: Dict,
    output_path: Path,
    title: str = "Accuracy vs word length",
) -> None:
    resolved_title = _resolve_title(aggregated, "Exp1", title, "Accuracy vs word length")
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_length_accuracy_ax(ax, aggregated)
    ax.set_title(resolved_title)
    _add_subtitle(fig, aggregated)
    fig.subplots_adjust(bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_length_accuracy_panel(aggregated_list: Sequence[Dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(aggregated_list), figsize=(4.5 * len(aggregated_list), 4.5), sharey=True)
    if len(aggregated_list) == 1:
        axes = [axes]
    for ax, aggregated in zip(axes, aggregated_list):
        _plot_length_accuracy_ax(ax, aggregated, show_legend=False)
        ax.set_title(_default_title(aggregated, "Exp1"))
    axes[0].legend(loc="lower left")
    fig.suptitle("Accuracy vs word length", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_beta_sweep_ax(ax, aggregated: Dict, *, show_legend: bool = True) -> None:
    betas = aggregated.get("betas", [])
    metrics = aggregated["metrics"]
    rnn = metrics["rnn_accuracy"]["mean"]
    lstm = metrics["lstm_accuracy"]["mean"]
    snn_learned = metrics["snn_accuracy"]["mean"]
    snn_means = [metrics[_beta_metric_key(metrics, b)]["mean"] for b in betas]
    snn_stds = [metrics[_beta_metric_key(metrics, b)]["std"] for b in betas]

    ax.axhline(
        rnn,
        color=MODEL_COLORS["rnn"],
        linestyle="--",
        label=f"RNN baseline ({rnn:.2f})",
    )
    ax.axhline(
        lstm,
        color=MODEL_COLORS["lstm"],
        linestyle="--",
        label=f"LSTM baseline ({lstm:.2f})",
    )
    ax.axhline(
        snn_learned,
        color=MODEL_COLORS["snn"],
        linestyle=":",
        label=f"SNN learned β ({snn_learned:.2f})",
    )
    ax.errorbar(
        betas,
        snn_means,
        yerr=snn_stds,
        label="SNN fixed β",
        color=MODEL_COLORS["snn"],
        marker=MODEL_MARKERS["snn"],
        linestyle=MODEL_LINESTYLES["snn"],
        capsize=3,
    )
    ax.set_xlabel("Membrane decay β")
    ax.set_ylabel("Accuracy")
    if betas:
        ax.set_xlim(min(betas), max(betas))
    _apply_common_axes(ax)
    if show_legend:
        ax.legend(fontsize=8)


def plot_beta_sweep(aggregated: Dict, output_path: Path, title: str = "Beta sweep") -> None:
    resolved_title = _resolve_title(aggregated, "Exp2", title, "Beta sweep")
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_beta_sweep_ax(ax, aggregated)
    ax.set_title(resolved_title)
    _add_subtitle(fig, aggregated)
    fig.subplots_adjust(bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_sweep_panel(aggregated_list: Sequence[Dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(aggregated_list), figsize=(4.5 * len(aggregated_list), 4.5), sharey=True)
    if len(aggregated_list) == 1:
        axes = [axes]
    for ax, aggregated in zip(axes, aggregated_list):
        _plot_beta_sweep_ax(ax, aggregated, show_legend=False)
        ax.set_title(_default_title(aggregated, "Exp2"))
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("SNN membrane decay β sweep", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_difficulty_breakdown_ax(ax, aggregated: Dict, *, show_legend: bool = True) -> None:
    metrics = aggregated["metrics"]
    labels = list(DIFFICULTY_LABELS)
    x = np.arange(len(labels))
    width = 0.25
    offsets = (-width, 0.0, width)

    for model, offset in zip(MODELS, offsets):
        means = [metrics[f"{model}_{d}_neg_accuracy"]["mean"] for d in labels]
        stds = [metrics[f"{model}_{d}_neg_accuracy"]["std"] for d in labels]
        has_err = any(s > 0 for s in stds)
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=model.upper(),
            color=MODEL_COLORS[model],
            yerr=stds if has_err else None,
            capsize=3 if has_err else 0,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, mean in zip(bars, means):
            if mean == 0:
                bar.set_hatch("//")
                bar.set_height(0.02)
                ax.annotate(
                    "0",
                    xy=(bar.get_x() + bar.get_width() / 2, 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy on negatives")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    if show_legend:
        ax.legend()


def plot_difficulty_breakdown(
    aggregated: Dict,
    output_path: Path,
    title: str = "Hard vs easy negative accuracy",
) -> None:
    resolved_title = _resolve_title(aggregated, "Exp3", title, "Hard vs easy negative accuracy")
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_difficulty_breakdown_ax(ax, aggregated)
    ax.set_title(resolved_title)
    _add_subtitle(fig, aggregated)
    fig.subplots_adjust(bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_breakdown_panel(aggregated_list: Sequence[Dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(aggregated_list), figsize=(4 * len(aggregated_list), 4.5), sharey=True)
    if len(aggregated_list) == 1:
        axes = [axes]
    for ax, aggregated in zip(axes, aggregated_list):
        _plot_difficulty_breakdown_ax(ax, aggregated, show_legend=False)
        ax.set_title(_default_title(aggregated, "Exp3"))
    axes[0].legend(loc="upper right")
    fig.suptitle("Hard vs easy negative accuracy", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
