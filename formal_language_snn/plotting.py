from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import strategy_difficulty_label, strategy_plot_order

plt.rcParams.update(
    {
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# Single-column paper panel (2x2, plus one centered row for a 5th language).
PANEL_COL_WIDTH = 5.0
PANEL_DPI = 300
PANEL_STYLE = {
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.markersize": 4,
    "lines.linewidth": 1.0,
    "errorbar.capsize": 2,
}
PANEL_HSPACE = 0.10
PANEL_WSPACE = 0.30
PANEL_LEGEND_BOTTOM = 0.04

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


def _panel_figsize(n: int) -> tuple[float, float]:
    if n <= 4:
        width = PANEL_COL_WIDTH
        return width, width * 0.95
    # 3-across top row needs a bit more width; 2 plot rows only.
    width = PANEL_COL_WIDTH * 1.55
    return width, width * 0.72


def _make_panel_axes(n: int):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=_panel_figsize(n))
    # Extra bottom GridSpec row reserves legend space (avoids crushing under x-labels).
    margins = dict(left=0.08, right=0.99, top=0.92, bottom=PANEL_LEGEND_BOTTOM)
    if n <= 4:
        gs = GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=[1, 1, 0.22],
            hspace=PANEL_HSPACE,
            wspace=PANEL_WSPACE,
            **margins,
        )
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        legend_ax = fig.add_subplot(gs[2, :])
        legend_ax.axis("off")
        for ax in axes[n:]:
            ax.set_visible(False)
        fig._panel_legend_ax = legend_ax  # type: ignore[attr-defined]
        return fig, axes[:n]

    # 5 languages: 3 on top, 2 on bottom spanning full width (6-col grid).
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1, 1, 0.06, 0.28],
        hspace=PANEL_HSPACE,
        wspace=PANEL_WSPACE,
        **margins,
    )
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]
    spacer = fig.add_subplot(gs[2, :])
    spacer.axis("off")
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis("off")
    fig._panel_legend_ax = legend_ax  # type: ignore[attr-defined]
    return fig, axes[:n]


def _panel_bottom_indices(n: int) -> set[int]:
    """Axes that sit on the bottom plot row (keep x-labels)."""
    if n <= 4:
        return {i for i in (2, 3) if i < n}
    if n == 5:
        return {3, 4}
    return {n - 1}


def _add_panel_legend(fig, axes, *, ncol: int) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    legend_ax = getattr(fig, "_panel_legend_ax", None)
    if legend_ax is not None:
        legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=ncol,
            frameon=False,
            columnspacing=0.8,
            handletextpad=0.4,
        )
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.4,
    )


def _save_panel(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _pair_by_language(left: Sequence[Dict], right: Sequence[Dict]) -> list[tuple[Dict, Dict]]:
    right_by_lang = {p["language"]: p for p in right}
    pairs: list[tuple[Dict, Dict]] = []
    for payload in sorted(left, key=lambda p: p.get("language", "")):
        lang = payload.get("language")
        if lang in right_by_lang:
            pairs.append((payload, right_by_lang[lang]))
    return pairs


def _make_regime_pair_axes(n_languages: int):
    """One row per language; left=in_range, right=extrapolation."""
    from matplotlib.gridspec import GridSpec

    width = PANEL_COL_WIDTH * 2.05
    height = max(3.0, 2.05 * n_languages)
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(
        n_languages,
        2,
        figure=fig,
        hspace=0.28,
        wspace=0.16,
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.10,
    )
    axes = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(n_languages)]
    return fig, axes


def _n_range_label(payload: Dict | None, *, fallback: str = "?") -> str:
    """Human label from seedagg n_range, e.g. 'n=1–10'."""
    if not payload:
        return fallback
    n_range = payload.get("n_range") or {}
    min_n = n_range.get("min_n")
    max_n = n_range.get("max_n")
    if min_n is None or max_n is None:
        cfg = payload.get("config") or {}
        min_n = cfg.get("difficulty_min_n", min_n)
        max_n = cfg.get("difficulty_max_n", max_n)
    if min_n is None or max_n is None:
        return fallback
    return f"n={min_n}–{max_n}"


def _add_regime_column_headers(fig, labels: tuple[str, str]) -> None:
    fig.text(0.31, 0.955, labels[0], ha="center", va="top", fontsize=10, fontweight="bold")
    fig.text(0.71, 0.955, labels[1], ha="center", va="top", fontsize=10, fontweight="bold")



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
    with plt.rc_context(PANEL_STYLE):
        n = len(aggregated_list)
        fig, axes = _make_panel_axes(n)
        bottom = _panel_bottom_indices(n)
        for i, (ax, aggregated) in enumerate(zip(axes, aggregated_list)):
            _plot_length_accuracy_ax(ax, aggregated, show_legend=False)
            # Language label inside axes (bottom) so it does not inflate row gaps.
            ax.text(
                0.03,
                0.04,
                _default_title(aggregated, "Exp1"),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
            if i in bottom:
                ax.tick_params(axis="x", rotation=0, labelsize=5.5)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
        fig.suptitle("Accuracy vs word length", y=0.98, fontsize=9)
        _add_panel_legend(fig, axes, ncol=3)
        _save_panel(fig, output_path)


def _plot_beta_sweep_ax(ax, aggregated: Dict, *, show_legend: bool = True, short_labels: bool = False) -> None:
    betas = aggregated.get("betas", [])
    metrics = aggregated["metrics"]
    rnn = metrics["rnn_accuracy"]["mean"]
    lstm = metrics["lstm_accuracy"]["mean"]
    snn_learned = metrics["snn_accuracy"]["mean"]
    snn_means = [metrics[_beta_metric_key(metrics, b)]["mean"] for b in betas]
    snn_stds = [metrics[_beta_metric_key(metrics, b)]["std"] for b in betas]

    rnn_label = "RNN" if short_labels else f"RNN baseline ({rnn:.2f})"
    lstm_label = "LSTM" if short_labels else f"LSTM baseline ({lstm:.2f})"
    learned_label = r"SNN learned $\beta$" if short_labels else f"SNN learned β ({snn_learned:.2f})"
    fixed_label = r"SNN fixed $\beta$" if short_labels else "SNN fixed β"

    ax.axhline(
        rnn,
        color=MODEL_COLORS["rnn"],
        linestyle="--",
        label=rnn_label,
    )
    ax.axhline(
        lstm,
        color=MODEL_COLORS["lstm"],
        linestyle="--",
        label=lstm_label,
    )
    ax.axhline(
        snn_learned,
        color=MODEL_COLORS["snn"],
        linestyle=":",
        label=learned_label,
    )
    ax.errorbar(
        betas,
        snn_means,
        yerr=snn_stds,
        label=fixed_label,
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
    with plt.rc_context(PANEL_STYLE):
        n = len(aggregated_list)
        fig, axes = _make_panel_axes(n)
        bottom = _panel_bottom_indices(n)
        for i, (ax, aggregated) in enumerate(zip(axes, aggregated_list)):
            _plot_beta_sweep_ax(ax, aggregated, show_legend=False, short_labels=True)
            ax.text(
                0.03,
                0.97,
                _default_title(aggregated, "Exp2"),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                fontweight="bold",
            )
            if i not in bottom:
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
        fig.suptitle("SNN membrane decay β sweep", y=0.98, fontsize=9)
        _add_panel_legend(fig, axes, ncol=4)
        _save_panel(fig, output_path)


def _plot_difficulty_breakdown_ax(ax, aggregated: Dict, *, show_legend: bool = True) -> None:
    metrics = aggregated["metrics"]
    ci = aggregated.get("stats", {}).get("confidence_intervals", {})
    labels = list(DIFFICULTY_LABELS)
    x = np.arange(len(labels))
    width = 0.25
    offsets = (-width, 0.0, width)

    for model, offset in zip(MODELS, offsets):
        means = [metrics[f"{model}_{d}_neg_accuracy"]["mean"] for d in labels]
        lowers = []
        uppers = []
        for diff in labels:
            key = f"{model}_{diff}_neg_accuracy"
            if key in ci and "ci_low" in ci[key] and "ci_high" in ci[key]:
                lo = max(0.0, means[labels.index(diff)] - ci[key]["ci_low"])
                hi = max(0.0, ci[key]["ci_high"] - means[labels.index(diff)])
            else:
                std = metrics[key]["std"]
                lo = std
                hi = std
            lowers.append(lo)
            uppers.append(hi)
        has_err = any((l > 0 or u > 0) for l, u in zip(lowers, uppers))
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=model.upper(),
            color=MODEL_COLORS[model],
            yerr=[lowers, uppers] if has_err else None,
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


def plot_difficulty_delta(
    aggregated: Dict,
    output_path: Path,
    title: str = "Hard-easy negative accuracy delta",
) -> None:
    metrics = aggregated["metrics"]
    ci = aggregated.get("stats", {}).get("confidence_intervals", {})
    xs = np.arange(len(MODELS))
    deltas = []
    yerr = [[], []]
    for model in MODELS:
        hard_key = f"{model}_hard_neg_accuracy"
        easy_key = f"{model}_easy_neg_accuracy"
        hard = metrics[hard_key]["mean"]
        easy = metrics[easy_key]["mean"]
        delta = easy - hard
        deltas.append(delta)
        hard_std = metrics[hard_key]["std"]
        easy_std = metrics[easy_key]["std"]
        std_delta = (hard_std ** 2 + easy_std ** 2) ** 0.5
        if hard_key in ci and easy_key in ci:
            n = max(int(ci[hard_key].get("n", 0)), int(ci[easy_key].get("n", 0)))
            if n > 1:
                margin = 1.96 * std_delta / (n ** 0.5)
            else:
                margin = 0.0
        else:
            margin = std_delta
        yerr[0].append(margin)
        yerr[1].append(margin)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(xs, deltas, color=[MODEL_COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.5)
    ax.errorbar(xs, deltas, yerr=yerr, fmt="none", color="black", capsize=4)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([m.upper() for m in MODELS])
    ax.set_ylabel("Easy - hard accuracy")
    ax.set_ylim(-0.2, 1.0)
    for bar, value in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    resolved_title = _resolve_title(aggregated, "Exp3", title, "Hard-easy negative accuracy delta")
    ax.set_title(resolved_title)
    ax.grid(True, axis="y", alpha=0.3)
    _add_subtitle(fig, aggregated)
    fig.subplots_adjust(bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_breakdown_ax(ax, aggregated: Dict, *, show_legend: bool = True) -> None:
    metrics = aggregated["metrics"]
    available_ids = {
        int(k.split("_strategy_", 1)[1].split("_", 1)[0])
        for k in metrics
        if "_strategy_" in k and k.endswith("_neg_accuracy")
    }
    language = aggregated.get("language", "")
    strategy_ids = [sid for sid in strategy_plot_order(language) if sid in available_ids]
    if not strategy_ids:
        strategy_ids = sorted(available_ids)
    if not strategy_ids:
        ax.text(0.5, 0.5, "No strategy metrics", ha="center", va="center")
        return

    x = np.arange(len(strategy_ids))
    width = 0.25
    offsets = (-width, 0.0, width)
    for model, offset in zip(MODELS, offsets):
        means = [
            metrics.get(f"{model}_strategy_{sid}_neg_accuracy", {}).get("mean", float("nan"))
            for sid in strategy_ids
        ]
        ax.bar(
            x + offset,
            means,
            width,
            label=model.upper(),
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([strategy_difficulty_label(language, sid) for sid in strategy_ids])
    ax.set_ylabel("Accuracy on negatives")
    ax.set_xlabel("Negative strategy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    if show_legend:
        ax.legend()


def plot_strategy_breakdown(
    aggregated: Dict,
    output_path: Path,
    title: str = "Negative strategy breakdown",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_strategy_breakdown_ax(ax, aggregated)
    resolved_title = _resolve_title(aggregated, "Exp3", title, "Negative strategy breakdown")
    ax.set_title(resolved_title)
    _add_subtitle(fig, aggregated)
    fig.subplots_adjust(bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_breakdown_panel(aggregated_list: Sequence[Dict], output_path: Path) -> None:
    with plt.rc_context(PANEL_STYLE):
        fig, axes = _make_panel_axes(len(aggregated_list))
        for ax, aggregated in zip(axes, aggregated_list):
            _plot_difficulty_breakdown_ax(ax, aggregated, show_legend=False)
            ax.set_title(_default_title(aggregated, "Exp3"), pad=3)
        fig.suptitle("Hard vs easy negative accuracy", y=0.99, fontsize=9)
        _add_panel_legend(fig, axes, ncol=3)
        _save_panel(fig, output_path)


def plot_strategy_breakdown_panel(
    aggregated_list: Sequence[Dict],
    output_path: Path,
    *,
    suptitle: str = "Negative strategy breakdown",
) -> None:
    with plt.rc_context(PANEL_STYLE):
        fig, axes = _make_panel_axes(len(aggregated_list))
        for ax, aggregated in zip(axes, aggregated_list):
            _plot_strategy_breakdown_ax(ax, aggregated, show_legend=False)
            ax.set_title(_default_title(aggregated, "Exp3"), pad=3)
        fig.suptitle(suptitle, y=0.99, fontsize=9)
        _add_panel_legend(fig, axes, ncol=3)
        _save_panel(fig, output_path)


def plot_difficulty_breakdown_regime_panel(
    in_range_list: Sequence[Dict],
    extrapolation_list: Sequence[Dict],
    output_path: Path,
    *,
    suptitle: str = "Hard vs easy negative accuracy",
) -> None:
    pairs = _pair_by_language(in_range_list, extrapolation_list)
    if not pairs:
        return
    # Slightly larger type than the cramped single-regime panel.
    style = {**PANEL_STYLE, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    with plt.rc_context(style):
        fig, axes = _make_regime_pair_axes(len(pairs))
        for row, (left, right) in enumerate(pairs):
            lang = left.get("language", "unknown")
            _plot_difficulty_breakdown_ax(axes[row][0], left, show_legend=False)
            _plot_difficulty_breakdown_ax(axes[row][1], right, show_legend=False)
            axes[row][0].set_title(lang, pad=3)
            axes[row][1].set_title(lang, pad=3)
            if row < len(pairs) - 1:
                axes[row][0].set_xlabel("")
                axes[row][1].set_xlabel("")
        _add_regime_column_headers(
            fig,
            (_n_range_label(pairs[0][0], fallback="In-range"), _n_range_label(pairs[0][1], fallback="Extrapolation")),
        )
        fig.suptitle(suptitle, y=0.995, fontsize=11)
        _add_panel_legend(fig, [axes[0][0]], ncol=3)
        _save_panel(fig, output_path)


def plot_strategy_breakdown_regime_panel(
    in_range_list: Sequence[Dict],
    extrapolation_list: Sequence[Dict],
    output_path: Path,
    *,
    suptitle: str = "Negative strategy breakdown",
) -> None:
    pairs = _pair_by_language(in_range_list, extrapolation_list)
    if not pairs:
        return
    style = {**PANEL_STYLE, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    with plt.rc_context(style):
        fig, axes = _make_regime_pair_axes(len(pairs))
        for row, (left, right) in enumerate(pairs):
            lang = left.get("language", "unknown")
            _plot_strategy_breakdown_ax(axes[row][0], left, show_legend=False)
            _plot_strategy_breakdown_ax(axes[row][1], right, show_legend=False)
            axes[row][0].set_title(lang, pad=3)
            axes[row][1].set_title(lang, pad=3)
            if row < len(pairs) - 1:
                axes[row][0].set_xlabel("")
                axes[row][1].set_xlabel("")
        _add_regime_column_headers(
            fig,
            (_n_range_label(pairs[0][0], fallback="In-range"), _n_range_label(pairs[0][1], fallback="Extrapolation")),
        )
        fig.suptitle(suptitle, y=0.995, fontsize=11)
        _add_panel_legend(fig, [axes[0][0]], ncol=3)
        _save_panel(fig, output_path)


def _seedagg_plot_stem(seedagg_path: Path) -> Path:
    stem = seedagg_path.stem
    if stem.endswith("_seedagg"):
        stem = stem[: -len("_seedagg")]
    return seedagg_path.parent / stem


def discover_exp1_seedagg_files(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("exp1_length_*_seedagg.json"))


def plot_exp1_seedagg_files(
    paths: Sequence[Path],
    out_root: Path | None = None,
    *,
    panel_only: bool = False,
) -> list[Path]:
    if not paths:
        return []

    out_dir = out_root or Path(paths[0]).parent
    written: list[Path] = []
    payloads: list[Dict] = []

    for path in paths:
        data = json.loads(Path(path).read_text())
        payloads.append(data)
        if not panel_only:
            out = _seedagg_plot_stem(Path(path)).with_suffix(".png")
            plot_length_accuracy(data, out, title=f"Exp1: {data['language']}")
            written.append(out)

    if len(payloads) > 1:
        panel_path = out_dir / "exp1_length_all.png"
        plot_length_accuracy_panel(payloads, panel_path)
        written.append(panel_path)

    return written


def discover_exp2_seedagg_files(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("exp2_beta_*_seedagg.json"))


def plot_exp2_seedagg_files(
    paths: Sequence[Path],
    out_root: Path | None = None,
    *,
    panel_only: bool = False,
) -> list[Path]:
    if not paths:
        return []

    out_dir = out_root or Path(paths[0]).parent
    written: list[Path] = []
    payloads: list[Dict] = []

    for path in paths:
        data = json.loads(Path(path).read_text())
        payloads.append(data)
        if not panel_only:
            out = _seedagg_plot_stem(Path(path)).with_suffix(".png")
            plot_beta_sweep(data, out, title=f"Exp2: {data['language']}")
            written.append(out)

    if len(payloads) > 1:
        panel_path = out_dir / "exp2_beta_all.png"
        plot_beta_sweep_panel(payloads, panel_path)
        written.append(panel_path)

    return written


EXP3_REGIMES = ("in_range", "extrapolation")


def graft_exp3_block(parent: Dict, block: Dict) -> Dict:
    return {
        "language": parent["language"],
        "regime": parent.get("regime"),
        "num_seeds": parent["num_seeds"],
        "metrics": block["metrics"],
        "stats": block.get("stats", {}),
    }


def plot_exp3_block_suite(parent: Dict, block: Dict, stem: Path, title: str) -> list[Path]:
    # ponytail: skip delta plots — hard/easy + strategy panels cover Exp3 for the paper
    payload = graft_exp3_block(parent, block)
    paths = [
        stem.with_suffix(".png"),
        stem.parent / f"{stem.name}_strategy.png",
    ]
    plot_difficulty_breakdown(payload, paths[0], title=title)
    plot_strategy_breakdown(payload, paths[1], title=f"{title} strategy")
    return paths


def discover_exp3_seedagg_files(exp_dir: Path) -> list[Path]:
    regime_paths = sorted(
        path
        for regime in EXP3_REGIMES
        for path in exp_dir.glob(f"*_{regime}_seedagg.json")
    )
    if regime_paths:
        return regime_paths
    return sorted(
        path
        for path in exp_dir.glob("exp3_difficulty_*_seedagg.json")
        if not any(f"_{regime}_" in path.name for regime in EXP3_REGIMES)
    )


def plot_exp3_seedagg_files(
    paths: Sequence[Path],
    out_root: Path | None = None,
    *,
    panel_only: bool = False,
) -> list[Path]:
    if not paths:
        return []

    out_dir = out_root or paths[0].parent
    written: list[Path] = []
    main_panels: dict[str, list[Dict]] = {regime: [] for regime in EXP3_REGIMES}
    control_panels: dict[str, list[Dict]] = {regime: [] for regime in EXP3_REGIMES}
    sensitivity_panels: dict[str, dict[int, list[Dict]]] = {
        regime: {} for regime in EXP3_REGIMES
    }

    for path in paths:
        parent = json.loads(Path(path).read_text())

        language = parent["language"]
        regime = parent.get("regime", "in_range")
        stem = out_dir / f"exp3_difficulty_{language}_{regime}"

        if not panel_only:
            written.extend(
                plot_exp3_block_suite(parent, parent, stem, f"Exp3 ({regime}): {language}")
            )

        main_panels.setdefault(regime, []).append(parent)

        control_block = parent.get("controls", {}).get("fixed_beta")
        if control_block:
            control_plot = graft_exp3_block(parent, control_block)
            control_panels.setdefault(regime, []).append(control_plot)
            if not panel_only:
                written.extend(
                    plot_exp3_block_suite(
                        parent,
                        control_block,
                        out_dir / f"exp3_difficulty_{language}_{regime}_control_fixed_beta",
                        f"Exp3 ({regime}): {language} — fixed beta",
                    )
                )

        for sens_block in parent.get("sensitivity", []):
            hidden_size = int(sens_block["hidden_size"])
            sens_plot = graft_exp3_block(parent, sens_block)
            sensitivity_panels.setdefault(regime, {}).setdefault(hidden_size, []).append(sens_plot)
            if not panel_only:
                written.extend(
                    plot_exp3_block_suite(
                        parent,
                        sens_block,
                        out_dir / f"exp3_difficulty_{language}_{regime}_hidden_{hidden_size}",
                        f"Exp3 ({regime}): {language} — hidden {hidden_size}",
                    )
                )

    def _write_exp3_panels(payloads: list[Dict], stem: str, *, strategy_title: str) -> None:
        if len(payloads) <= 1:
            return
        hard_easy = out_dir / f"{stem}.png"
        strategy = out_dir / f"{stem}_strategy.png"
        plot_difficulty_breakdown_panel(payloads, hard_easy)
        plot_strategy_breakdown_panel(payloads, strategy, suptitle=strategy_title)
        written.extend([hard_easy, strategy])

    def _write_combined_regime_panels(
        by_regime: dict[str, list[Dict]],
        stem: str,
        *,
        hard_easy_title: str,
        strategy_title: str,
    ) -> None:
        left = by_regime.get("in_range", [])
        right = by_regime.get("extrapolation", [])
        if left and right and _pair_by_language(left, right):
            hard_easy = out_dir / f"{stem}.png"
            strategy = out_dir / f"{stem}_strategy.png"
            plot_difficulty_breakdown_regime_panel(
                left, right, hard_easy, suptitle=hard_easy_title
            )
            plot_strategy_breakdown_regime_panel(
                left, right, strategy, suptitle=strategy_title
            )
            written.extend([hard_easy, strategy])
            return
        # Fallback when only one regime exists.
        for regime, payloads in by_regime.items():
            _write_exp3_panels(
                payloads,
                f"{stem}_{regime}",
                strategy_title=f"{strategy_title} ({regime})",
            )

    # Combined in_range | extrapolation panels (paper figures).
    _write_combined_regime_panels(
        main_panels,
        "exp3_difficulty_all",
        hard_easy_title="Hard vs easy negative accuracy",
        strategy_title="Negative strategy breakdown",
    )
    _write_combined_regime_panels(
        control_panels,
        "exp3_difficulty_all_control_fixed_beta",
        hard_easy_title="Hard vs easy negative accuracy (fixed β)",
        strategy_title="Negative strategy breakdown (fixed β)",
    )

    hidden_sizes = sorted(
        {
            hidden
            for by_hidden in sensitivity_panels.values()
            for hidden in by_hidden
        }
    )
    for hidden_size in hidden_sizes:
        by_regime = {
            regime: sensitivity_panels.get(regime, {}).get(hidden_size, [])
            for regime in EXP3_REGIMES
        }
        _write_combined_regime_panels(
            by_regime,
            f"exp3_difficulty_all_hidden_{hidden_size}",
            hard_easy_title=f"Hard vs easy negative accuracy (hidden {hidden_size})",
            strategy_title=f"Negative strategy breakdown (hidden {hidden_size})",
        )

    return written


EXPERIMENT_IDS = ("exp1", "exp2", "exp3")
EXPERIMENT_DIRS = {
    "exp1": "exp1_length",
    "exp2": "exp2_beta",
    "exp3": "exp3_difficulty",
}


def discover_experiment_seedagg_files(experiment: str, exp_dir: Path) -> list[Path]:
    if experiment == "exp1":
        return discover_exp1_seedagg_files(exp_dir)
    if experiment == "exp2":
        return discover_exp2_seedagg_files(exp_dir)
    if experiment == "exp3":
        return discover_exp3_seedagg_files(exp_dir)
    raise ValueError(f"Unknown experiment: {experiment}")


def plot_experiment_seedagg_files(
    experiment: str,
    paths: Sequence[Path],
    out_root: Path | None = None,
    *,
    panel_only: bool = False,
) -> list[Path]:
    if experiment == "exp1":
        return plot_exp1_seedagg_files(paths, out_root, panel_only=panel_only)
    if experiment == "exp2":
        return plot_exp2_seedagg_files(paths, out_root, panel_only=panel_only)
    if experiment == "exp3":
        return plot_exp3_seedagg_files(paths, out_root, panel_only=panel_only)
    raise ValueError(f"Unknown experiment: {experiment}")
