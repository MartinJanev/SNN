#!/usr/bin/env python3
"""Export aggregated experiment metrics as TSV or Overleaf-ready LaTeX tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formal_language_snn.data import (
    LENGTH_BUCKETS,
    strategy_difficulty_label,
    strategy_plot_order,
)
from formal_language_snn.paths import PROJECT_ROOT

# Keep local so this script does not import matplotlib via plotting.py.
EXPERIMENT_IDS = ("exp1", "exp2", "exp3")
EXPERIMENT_DIRS = {
    "exp1": "exp1_length",
    "exp2": "exp2_beta",
    "exp3": "exp3_difficulty",
}

MODELS = ("rnn", "snn", "lstm", "rsnn")
# GRU rather than "RNN": with a recurrent spiking model in the table, "RNN" no longer
# picks out one row. RSNN is the recurrent spiking control.
MODEL_LABELS = {"rnn": "GRU", "snn": "SNN", "lstm": "LSTM", "rsnn": "RSNN"}
BUCKET_LABELS = [f"{lo}-{hi}" for lo, hi in LENGTH_BUCKETS]


def fit_to_textwidth(tabular: str) -> str:
    """Wrap a tabular so a wide one cannot overflow the LNCS text block.

    The per-language x per-model tables are one column per cell, so they grow with the
    roster: exp1 is 1 + 5 languages x 4 models = 21 columns. \resizebox is what keeps
    that inside \textwidth without hand-tuning font sizes per table.
    """
    return "\\resizebox{\\textwidth}{!}{%\n" + tabular.rstrip("\n") + "\n}\n"

# Paper-facing labels / order for the overall-accuracy stats table.
PAPER_LANGUAGE_ORDER = ("anbn", "balanced_parens", "palindrome", "reber", "even_a")
PAPER_LANGUAGE_TEX = {
    "anbn": r"$a^n b^n$",
    "balanced_parens": r"$\text{Dyck-1}$",
    "palindrome": r"$\text{Palindrome}$",
    "reber": r"$\text{Regular}$",
    "even_a": r"$\text{Parity}$",
}
ALPHA = 0.05


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def n_range_latex(payload: dict | None, *, fallback: str = "?") -> str:
    """LaTeX column header from seedagg n_range, e.g. '$n=1$--$10$'."""
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
    return f"$n={min_n}$--${max_n}$"


def fmt_cell(stats: dict[str, float] | None, *, digits: int = 3) -> str:
    if not stats or "mean" not in stats:
        return "---"
    mean = float(stats["mean"])
    std = float(stats.get("std", 0.0))
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def fmt_pct_cell(stats: dict[str, float] | None, *, digits: int = 2) -> str:
    if not stats or "mean" not in stats:
        return "---"
    mean = 100.0 * float(stats["mean"])
    std = 100.0 * float(stats.get("std", 0.0))
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def fmt_pct_ci(stats: dict[str, float] | None, *, digits: int = 2) -> str:
    if not stats or "ci_low" not in stats or "ci_high" not in stats:
        return "---"
    lo = 100.0 * float(stats["ci_low"])
    hi = 100.0 * float(stats["ci_high"])
    return f"$[{lo:.{digits}f}, {hi:.{digits}f}]$"


def fmt_p_value(p: float | None) -> str:
    if p is None or p != p:  # NaN
        return "---"
    if p < 0.001:
        return r"$< 0.001$"
    return f"${p:.3f}$"


def _two_sided_sign_test(diffs: list[float]) -> float:
    import math

    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    pos = sum(1 for d in nonzero if d > 0)
    k = min(pos, n - pos)
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += math.comb(n, i)
    return min(1.0, 2.0 * cumulative / (2**n))


def _holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        candidate = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def _metric(metrics: dict, key: str) -> dict[str, float] | None:
    return metrics.get(key)


def _language_label(language: str) -> str:
    return latex_escape(language.replace("_", " "))


def _sort_payloads(payloads: list[dict]) -> list[dict]:
    return sorted(payloads, key=lambda p: p.get("language", ""))


def metrics_from(payload: dict, block: str | None = None) -> dict:
    """Top-level metrics, or nested block metrics (e.g. controls.fixed_beta)."""
    if block is None:
        return payload.get("metrics", {})
    if block == "fixed_beta":
        return payload.get("controls", {}).get("fixed_beta", {}).get("metrics", {})
    raise ValueError(f"Unknown metrics block: {block}")


def render_exp1(payloads: list[dict]) -> str:
    payloads = _sort_payloads(payloads)
    languages = [p["language"] for p in payloads]
    n_models = len(MODELS)
    col_spec = "l" + "c" * (len(languages) * n_models)

    lines = [
        "% Exp1: length generalization (mean ± std over seeds)",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header_langs = ["Bucket"]
    for lang in languages:
        header_langs.append(rf"\multicolumn{{{n_models}}}{{c}}{{{_language_label(lang)}}}")
    lines.append(" & ".join(header_langs) + r" \\")

    cmidrules = []
    col = 2
    for _ in languages:
        end = col + n_models - 1
        cmidrules.append(rf"\cmidrule(lr){{{col}-{end}}}")
        col = end + 1
    lines.append(" ".join(cmidrules))

    header_models = [""]
    for _ in languages:
        header_models.extend(MODEL_LABELS[m] for m in MODELS)
    lines.append(" & ".join(header_models) + r" \\")
    lines.append(r"\midrule")

    rows: list[tuple[str, str]] = [("Overall", "{model}_accuracy")]
    for label in BUCKET_LABELS:
        rows.append((label.replace("-", "--"), f"{{model}}_bucket_{label}"))

    for row_label, key_tmpl in rows:
        cells = [row_label]
        for payload in payloads:
            for model in MODELS:
                cells.append(fmt_cell(_metric(payload.get("metrics", {}), key_tmpl.format(model=model))))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines[0] + "\n" + fit_to_textwidth("\n".join(lines[1:]))


def _swept_models(payloads: list[dict]) -> tuple[str, ...]:
    """Models that carry a beta sweep in these payloads, in canonical MODELS order."""
    keys = {k for p in payloads for k in p.get("metrics", {})}
    return tuple(m for m in MODELS if any(k.startswith(f"{m}_beta_") for k in keys))


def _beta_stats(metrics: dict, beta: float, model: str) -> dict | None:
    """Sweep entry for one model at one beta, tolerating float formatting drift in the key."""
    prefix = f"{model}_beta_"
    for key in (f"{prefix}{beta}", f"{prefix}{beta:g}", f"{prefix}{beta:.1f}"):
        stats = _metric(metrics, key)
        if stats:
            return stats
    for mkey, mstats in metrics.items():
        if not mkey.startswith(prefix):
            continue
        try:
            if abs(float(mkey[len(prefix):]) - beta) < 1e-9:
                return mstats
        except ValueError:
            continue
    return None


def render_exp2(payloads: list[dict]) -> str:
    payloads = _sort_payloads(payloads)
    languages = [p["language"] for p in payloads]
    col_spec = "l" + "c" * len(languages)

    betas: list[float] = []
    for payload in payloads:
        for b in payload.get("betas") or []:
            if b not in betas:
                betas.append(float(b))
    betas = sorted(betas)

    swept = _swept_models(payloads)

    lines = [
        "% Exp2: membrane decay beta sweep (mean ± std over seeds)",
        "% Gated baselines are constant reference rows: beta has no analogue in them, so they",
        "% come from the baseline block rather than being retrained at each sweep point.",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(["Model / $\\beta$"] + [_language_label(l) for l in languages]) + r" \\",
        r"\midrule",
    ]

    # Every model gets a row, swept or not -- that is what keeps the gated baselines in the
    # table even though only the spiking families have a beta to sweep.
    for model in MODELS:
        label = MODEL_LABELS[model]
        if model in swept:
            label = rf"{label} (learned $\beta$)"
        cells = [label]
        for payload in payloads:
            cells.append(fmt_cell(_metric(payload.get("metrics", {}), f"{model}_accuracy")))
        lines.append(" & ".join(cells) + r" \\")

    for model in swept:
        if not betas:
            break
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{len(languages) + 1}}}{{l}}{{\textit{{{MODEL_LABELS[model]}, fixed $\beta$}}}} \\"
        )
        for beta in betas:
            cells = [f"$\\beta={beta:g}$"]
            for payload in payloads:
                cells.append(fmt_cell(_beta_stats(payload.get("metrics", {}), beta, model)))
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _pair_exp3_by_language(
    in_range: list[dict], extrapolation: list[dict]
) -> list[tuple[dict, dict | None]]:
    """Pair in_range with extrapolation by language."""
    extrap_by_lang = {p["language"]: p for p in extrapolation}
    return [
        (payload, extrap_by_lang.get(payload["language"]))
        for payload in _sort_payloads(in_range)
    ]


def render_exp3_hard_easy(
    pairs: list[tuple[dict, dict | None]],
    *,
    block: str | None = None,
    comment: str = "Exp3: hard vs easy negative accuracy",
) -> str:
    languages = [left["language"] for left, _ in pairs]
    n_regimes = 2
    col_spec = "l" + "c" * (len(languages) * n_regimes)

    lines = [
        f"% {comment} (mean ± std over seeds)",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header_langs = ["Model / difficulty"]
    for lang in languages:
        header_langs.append(rf"\multicolumn{{{n_regimes}}}{{c}}{{{_language_label(lang)}}}")
    lines.append(" & ".join(header_langs) + r" \\")

    cmidrules = []
    col = 2
    for _ in languages:
        end = col + n_regimes - 1
        cmidrules.append(rf"\cmidrule(lr){{{col}-{end}}}")
        col = end + 1
    lines.append(" ".join(cmidrules))

    header_regimes = [""]
    # Use n ranges from the first language pair (same across languages).
    left0, right0 = pairs[0]
    left_label = n_range_latex(left0, fallback="in-range")
    right_label = n_range_latex(right0, fallback="extrap.")
    for _ in languages:
        header_regimes.extend([left_label, right_label])
    lines.append(" & ".join(header_regimes) + r" \\")
    lines.append(r"\midrule")

    for model in MODELS:
        for difficulty in ("hard", "easy"):
            cells = [f"{MODEL_LABELS[model]} ({difficulty})"]
            key = f"{model}_{difficulty}_neg_accuracy"
            for left, right in pairs:
                cells.append(fmt_cell(_metric(metrics_from(left, block), key)))
                if right is not None:
                    cells.append(fmt_cell(_metric(metrics_from(right, block), key)))
                else:
                    cells.append("---")
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines[0] + "\n" + fit_to_textwidth("\n".join(lines[1:]))


def _strategy_ids_for(language: str, metrics: dict) -> list[int]:
    available = {
        int(k.split("_strategy_", 1)[1].split("_", 1)[0])
        for k in metrics
        if "_strategy_" in k and k.endswith("_neg_accuracy")
    }
    ordered = [sid for sid in strategy_plot_order(language) if sid in available]
    return ordered or sorted(available)


def render_exp3_strategy(
    pairs: list[tuple[dict, dict | None]],
    *,
    block: str | None = None,
    comment: str = "Exp3: negative strategy breakdown",
) -> str:
    """One tabular per language (strategies differ across languages)."""
    chunks: list[str] = [f"% {comment} (mean ± std over seeds)"]
    for left, right in pairs:
        language = left["language"]
        left_m = metrics_from(left, block)
        right_m = metrics_from(right, block) if right is not None else {}
        strategy_ids = _strategy_ids_for(language, left_m) or _strategy_ids_for(language, right_m)
        if not strategy_ids:
            continue

        left_label = n_range_latex(left, fallback="in-range")
        right_label = n_range_latex(right, fallback="extrap.")
        col_spec = "ll" + "c" * 2
        lines = [
            f"% language={language}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            rf"\multicolumn{{2}}{{c}}{{{_language_label(language)}}} & "
            + left_label
            + " & "
            + right_label
            + r" \\",
            r"\midrule",
        ]
        for model in MODELS:
            for sid in strategy_ids:
                label = strategy_difficulty_label(language, sid)
                key = f"{model}_strategy_{sid}_neg_accuracy"
                cells = [
                    MODEL_LABELS[model],
                    label,
                    fmt_cell(_metric(left_m, key)),
                    fmt_cell(_metric(right_m, key)) if right is not None else "---",
                ]
                lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
        chunks.append("\n".join(lines))
    return "\n".join(chunks).rstrip() + "\n"


def write_exp3_tables(pairs: list[tuple[dict, dict | None]]) -> list[tuple[str, str]]:
    """Return (filename, latex) for the paper-backed Exp3 tables."""
    return [
        (
            "exp3_difficulty.tex",
            render_exp3_hard_easy(
                pairs,
                block=None,
                comment="Exp3: hard vs easy negative accuracy (main, learned β)",
            ),
        ),
        (
            "exp3_difficulty_strategy.tex",
            render_exp3_strategy(
                pairs,
                block=None,
                comment="Exp3: negative strategy breakdown (main, learned β)",
            ),
        ),
        (
            "exp3_difficulty_control_fixed_beta.tex",
            render_exp3_hard_easy(
                pairs,
                block="fixed_beta",
                comment="Exp3: hard vs easy negative accuracy (fixed β control)",
            ),
        ),
        (
            "exp3_difficulty_control_fixed_beta_strategy.tex",
            render_exp3_strategy(
                pairs,
                block="fixed_beta",
                comment="Exp3: negative strategy breakdown (fixed β control)",
            ),
        ),
        (
            "exp3_statistical_accuracy.tex",
            render_statistical_accuracy(pairs),
        ),
    ]


def _ci_for(payload: dict, key: str) -> dict[str, float] | None:
    return payload.get("stats", {}).get("confidence_intervals", {}).get(key)


def _overall_comparison(payload: dict, rhs: str, lhs: str = "snn") -> dict | None:
    """The overall-accuracy comparison for this pair, in whichever order it was stored.

    Comparisons are emitted over unordered model pairs, so the SNN is not always the
    left-hand side. The sign test is two-sided, so the p-value does not depend on the
    orientation; only ``delta_mean`` does, and this table does not use it.
    """
    for comp in payload.get("stats", {}).get("paired_comparisons", []):
        if comp.get("metric") != "overall":
            continue
        if {comp.get("lhs"), comp.get("rhs")} == {lhs, rhs}:
            return comp
    return None


def _load_per_seed_overall(
    language: str,
    *,
    min_n: int,
    max_n: int,
    hidden_size: int,
    learn_beta: bool = True,
) -> list[dict[str, float]]:
    """Paired overall test_accuracy rows from individual Exp3 run JSONs."""
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / "exp3_difficulty"
    by_seed: dict[int, dict[str, float]] = {}
    for path in exp_dir.glob("*.json"):
        if "seedagg" in path.name:
            continue
        data = _load_json(path)
        cfg = data.get("config") or {}
        if data.get("language") != language:
            continue
        if not cfg.get("difficulty_test"):
            continue
        if int(cfg.get("difficulty_min_n", -1)) != min_n:
            continue
        if int(cfg.get("difficulty_max_n", -1)) != max_n:
            continue
        if int(cfg.get("hidden_size", -1)) != hidden_size:
            continue
        if bool(cfg.get("learn_beta", True)) != learn_beta:
            continue
        seed = cfg.get("seed")
        acc = data.get("test_accuracy") or {}
        if seed is None or not all(k in acc for k in MODELS):
            continue
        by_seed[int(seed)] = {k: float(acc[k]) for k in MODELS}
    return [by_seed[s] for s in sorted(by_seed)]


def _sign_tests_from_rows(
    rows: list[dict[str, float]], *, rhs: str = "rnn"
) -> tuple[float, float]:
    """Return (raw p, placeholder holm) for SNN vs rhs; Holming done across languages later."""
    diffs = [row["snn"] - row[rhs] for row in rows]
    return _two_sided_sign_test(diffs), float("nan")


def _n_range_from_payload(payload: dict) -> tuple[int, int] | None:
    n_range = payload.get("n_range") or {}
    min_n = n_range.get("min_n")
    max_n = n_range.get("max_n")
    if min_n is None or max_n is None:
        cfg = payload.get("config") or {}
        min_n = cfg.get("difficulty_min_n")
        max_n = cfg.get("difficulty_max_n")
    if min_n is None or max_n is None:
        return None
    return int(min_n), int(max_n)


def render_statistical_accuracy(
    pairs: list[tuple[dict, dict | None]],
    *,
    compare_rhs: str = "rnn",
) -> str:
    """Paper-style overall accuracy table on the Exp3 extrapolation regime."""
    # Prefer extrapolation payload when present; else fall back to left.
    by_lang: dict[str, dict] = {}
    for left, right in pairs:
        payload = right if right is not None else left
        by_lang[payload["language"]] = payload

    languages = [lang for lang in PAPER_LANGUAGE_ORDER if lang in by_lang]
    languages.extend(sorted(lang for lang in by_lang if lang not in languages))

    # Collect raw sign-test p-values (SNN vs compare_rhs), then Holm across languages.
    raw_ps: list[float | None] = []
    for lang in languages:
        payload = by_lang[lang]
        baked = _overall_comparison(payload, compare_rhs)
        if baked is not None:
            raw_ps.append(float(baked["p_value_sign_test"]))
            continue
        n_range = _n_range_from_payload(payload)
        cfg = payload.get("config") or {}
        hidden = int(cfg.get("hidden_size", 32))
        if n_range is None:
            raw_ps.append(None)
            continue
        rows = _load_per_seed_overall(
            lang, min_n=n_range[0], max_n=n_range[1], hidden_size=hidden
        )
        if not rows:
            raw_ps.append(None)
            continue
        p_raw, _ = _sign_tests_from_rows(rows, rhs=compare_rhs)
        raw_ps.append(p_raw)

    known = [(i, p) for i, p in enumerate(raw_ps) if p is not None]
    adjusted_map: dict[int, float] = {}
    if known:
        idxs, vals = zip(*known)
        for idx, adj in zip(idxs, _holm_adjust(list(vals))):
            adjusted_map[idx] = adj

    lines = [
        "% Exp3 extrapolation: overall string recognition accuracy (mean ± SD, %)",
        f"% Sign test / Holm: SNN vs {compare_rhs.upper()} overall accuracy across languages",
        r"\begin{tabular}{l" + "c" * len(MODELS) + r"ccc}",
        r"\toprule",
        r"\textbf{Language} & "
        + " & ".join(rf"\textbf{{{MODEL_LABELS[m]}}}" for m in MODELS)
        + r" & \textbf{95\% CI (SNN)} & \textbf{Sign Test ($p$)} & \textbf{Holm Sign} \\",
        r"\midrule",
    ]

    for i, lang in enumerate(languages):
        payload = by_lang[lang]
        metrics = metrics_from(payload)
        ci = _ci_for(payload, "snn_accuracy")
        p_raw = raw_ps[i]
        p_holm = adjusted_map.get(i)
        holm_label = "---"
        if p_holm is not None:
            holm_label = "Valid" if p_holm < ALPHA else "N.S."
        cells = [
            PAPER_LANGUAGE_TEX.get(lang, _language_label(lang)),
            *(fmt_pct_cell(_metric(metrics, f"{m}_accuracy")) for m in MODELS),
            fmt_pct_ci(ci),
            fmt_p_value(p_raw),
            holm_label,
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


RENDERERS = {
    "exp1": ("exp1_length.tex", render_exp1),
    "exp2": ("exp2_beta.tex", render_exp2),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _filter_languages(payloads: list[dict], languages: list[str] | None) -> list[dict]:
    if not languages:
        return payloads
    wanted = set(languages)
    return [p for p in payloads if p.get("language") in wanted]


def _filter_pairs(
    pairs: list[tuple[dict, dict | None]], languages: list[str] | None
) -> list[tuple[dict, dict | None]]:
    if not languages:
        return pairs
    wanted = set(languages)
    return [(a, b) for a, b in pairs if a.get("language") in wanted]


def discover_payloads(experiment: str, *, regime: str = "in_range") -> list[dict]:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT_DIRS[experiment]
    if experiment == "exp1":
        paths = sorted(exp_dir.glob("exp1_length_*_seedagg.json"))
    elif experiment == "exp2":
        paths = sorted(exp_dir.glob("exp2_beta_*_seedagg.json"))
    elif experiment == "exp3":
        tagged = sorted(exp_dir.glob(f"exp3_difficulty_*_{regime}_seedagg.json"))
        if tagged:
            paths = tagged
        else:
            paths = [
                p
                for p in sorted(exp_dir.glob("exp3_difficulty_*_seedagg.json"))
                if not any(f"_{r}_" in p.name for r in ("in_range", "extrapolation"))
            ]
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return [_load_json(p) for p in paths]


def discover_exp3_pairs() -> list[tuple[dict, dict | None]]:
    in_range = discover_payloads("exp3", regime="in_range")
    extrapolation = discover_payloads("exp3", regime="extrapolation")
    if in_range:
        return _pair_exp3_by_language(in_range, extrapolation)
    # Fallback: single-regime files only
    return [(p, None) for p in _sort_payloads(extrapolation or [])]


def print_tsv(data: dict) -> None:
    print(f"# {data.get('experiment', 'experiment')}  language={data.get('language', '?')}")
    print(f"# seeds={data.get('num_seeds', '?')}  seed_base={data.get('seed_base', '?')}")
    config = data.get("config")
    if config:
        print("# config")
        for key, value in sorted(config.items()):
            print(f"#   {key}={value}")
    print("metric\tmean\tstd")
    for key, stats in sorted(data.get("metrics", {}).items()):
        print(f"{key}\t{stats['mean']:.4f}\t{stats['std']:.4f}")
    counts = data.get("counts", {})
    if counts:
        print("\ncount\tmean\tstd")
        for key, stats in sorted(counts.items()):
            print(f"{key}\t{stats['mean']:.2f}\t{stats['std']:.2f}")
    paired = data.get("stats", {}).get("paired_comparisons", [])
    if paired:
        print("\npaired_comparison\tmetric\tdelta_mean\teffect_size_dz\tp_sign\tp_holm")
        for comp in paired:
            label = f"{comp['lhs']}_vs_{comp['rhs']}"
            print(
                f"{label}\t{comp['metric']}\t{comp['delta_mean']:.4f}\t"
                f"{comp['effect_size_dz']:.4f}\t{comp['p_value_sign_test']:.4f}\t{comp['p_value_holm']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export seed-aggregated metrics as Overleaf LaTeX tables (or TSV)."
    )
    parser.add_argument(
        "--experiment",
        choices=[*EXPERIMENT_IDS, "all"],
        default=None,
        help="Which experiment to export (default: all when --input is omitted)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Legacy: single seedagg JSON path (implies TSV unless --format latex)",
    )
    parser.add_argument("--languages", nargs="*", default=None, help="Optional language filter")
    parser.add_argument(
        "--format",
        choices=("latex", "tsv"),
        default="latex",
        help="Output format (default: latex)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/tables",
        help="Directory for .tex files (default: outputs/experiments/tables)",
    )
    parser.add_argument(
        "--regime",
        choices=("in_range", "extrapolation"),
        default="in_range",
        help="Exp3 regime for TSV / legacy single-file mode (LaTeX Exp3 always uses both)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    # Legacy single-file mode
    if args.input:
        in_path = Path(args.input)
        if not in_path.is_absolute():
            in_path = PROJECT_ROOT / in_path
        data = _load_json(in_path)
        if args.format == "tsv":
            print_tsv(data)
            return
        exp = args.experiment
        if exp is None or exp == "all":
            name = str(data.get("experiment", in_path.name))
            if "exp1" in name or "length" in name:
                exp = "exp1"
            elif "exp2" in name or "beta" in name:
                exp = "exp2"
            else:
                exp = "exp3"
        out_dir.mkdir(parents=True, exist_ok=True)
        if exp == "exp3":
            pairs = _filter_pairs([(data, None)], args.languages)
            for filename, tex in write_exp3_tables(pairs):
                out_path = out_dir / filename
                out_path.write_text(tex)
                print(tex, end="")
                print(f"% Wrote {out_path}", file=sys.stderr)
            return
        filename, renderer = RENDERERS[exp]
        payloads = _filter_languages([data], args.languages)
        tex = renderer(payloads)
        out_path = out_dir / filename
        out_path.write_text(tex)
        print(tex, end="")
        print(f"% Wrote {out_path}", file=sys.stderr)
        return

    experiments = list(EXPERIMENT_IDS) if args.experiment in (None, "all") else [args.experiment]
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        if exp == "exp3":
            pairs = _filter_pairs(discover_exp3_pairs(), args.languages)
            if not pairs:
                print("% No seedagg files found for exp3", file=sys.stderr)
                continue
            if args.format == "tsv":
                for left, right in pairs:
                    print_tsv(left)
                    print()
                    if right is not None:
                        print_tsv(right)
                        print()
                continue
            for filename, tex in write_exp3_tables(pairs):
                out_path = out_dir / filename
                out_path.write_text(tex)
                print(tex, end="")
                print()
                print(f"% Wrote {out_path}", file=sys.stderr)
            continue

        payloads = _filter_languages(discover_payloads(exp, regime=args.regime), args.languages)
        if not payloads:
            print(f"% No seedagg files found for {exp}", file=sys.stderr)
            continue

        if args.format == "tsv":
            for payload in _sort_payloads(payloads):
                print_tsv(payload)
                print()
            continue

        filename, renderer = RENDERERS[exp]
        tex = renderer(payloads)
        out_path = out_dir / filename
        out_path.write_text(tex)
        print(tex, end="")
        if len(experiments) > 1:
            print()
        print(f"% Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
