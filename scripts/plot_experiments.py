#!/usr/bin/env python3
"""Plot aggregated experiment JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formal_language_snn.paths import PROJECT_ROOT
from formal_language_snn.plotting import (
    plot_beta_sweep,
    plot_beta_sweep_panel,
    plot_difficulty_breakdown,
    plot_difficulty_breakdown_panel,
    plot_length_accuracy,
    plot_length_accuracy_panel,
)

EXP_PREFIX = {"exp1": "Exp1", "exp2": "Exp2", "exp3": "Exp3"}
EXP_DIR = {
    "exp1": "exp1_length",
    "exp2": "exp2_beta",
    "exp3": "exp3_difficulty",
}
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs/experiments"


def _title_from_data(data: dict, experiment: str) -> str:
    prefix = EXP_PREFIX[experiment]
    language = data.get("language", "unknown")
    return f"{prefix}: {language}"


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _discover_seedagg_files(experiment: str) -> list[Path]:
    exp_dir = EXPERIMENTS_ROOT / EXP_DIR[experiment]
    if not exp_dir.is_dir():
        return []
    return sorted(exp_dir.glob("*_seedagg.json"))


def _plot_single(experiment: str, data: dict, out: Path) -> None:
    title = _title_from_data(data, experiment)
    if experiment == "exp1":
        plot_length_accuracy(data, out, title=title)
    elif experiment == "exp2":
        plot_beta_sweep(data, out, title=title)
    else:
        plot_difficulty_breakdown(data, out, title=title)


def _plot_panel(experiment: str, data_list: list[dict], out: Path) -> None:
    if experiment == "exp1":
        plot_length_accuracy_panel(data_list, out)
    elif experiment == "exp2":
        plot_beta_sweep_panel(data_list, out)
    else:
        plot_difficulty_breakdown_panel(data_list, out)


def _process_experiment(
    experiment: str,
    inputs: list[Path] | None,
    *,
    panel_only: bool = False,
    panel_output: Path | None = None,
) -> None:
    paths = sorted(inputs) if inputs else _discover_seedagg_files(experiment)
    if not paths:
        print(f"No *_seedagg.json files found for {experiment}", flush=True)
        return

    data_list = [_load_json(p) for p in paths]

    if not panel_only:
        for path, data in zip(paths, data_list):
            out = path.with_suffix(".png")
            _plot_single(experiment, data, out)
            print(f"Wrote {out}")

    if len(data_list) > 1:
        out = panel_output or (paths[0].parent / f"{EXP_DIR[experiment]}_all.png")
        _plot_panel(experiment, data_list, out)
        print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregated *_seedagg.json results. "
        "With no --input, auto-discovers seedagg files under outputs/experiments/.",
    )
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "exp3", "all"],
        default="all",
        help="Which experiment to plot (default: all)",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Explicit seedagg JSON path(s); if omitted, auto-discover per experiment",
    )
    parser.add_argument("--output", default=None, help="Output path for --panel-only mode")
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Only write the combined multi-language panel (skip per-language PNGs)",
    )
    args = parser.parse_args()

    experiments = list(EXP_DIR) if args.experiment == "all" else [args.experiment]
    explicit_inputs = [_resolve_path(raw) for raw in args.input] if args.input else None
    panel_output = _resolve_path(args.output) if args.output else None

    if explicit_inputs and len(experiments) > 1:
        parser.error("Pass --experiment when using explicit --input paths")

    for experiment in experiments:
        inputs = explicit_inputs if explicit_inputs else None
        _process_experiment(
            experiment,
            inputs,
            panel_only=args.panel,
            panel_output=panel_output if len(experiments) == 1 else None,
        )


if __name__ == "__main__":
    main()
