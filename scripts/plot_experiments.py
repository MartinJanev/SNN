#!/usr/bin/env python3
"""Plot aggregated experiment JSON results from disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formal_language_snn.paths import PROJECT_ROOT
from formal_language_snn.plotting import (
    EXPERIMENT_DIRS,
    EXPERIMENT_IDS,
    discover_experiment_seedagg_files,
    plot_experiment_seedagg_files,
)

EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs/experiments"


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _process_experiment(
    experiment: str,
    inputs: list[Path] | None,
    *,
    panel_only: bool = False,
    out_root: Path | None = None,
) -> None:
    exp_dir = EXPERIMENTS_ROOT / EXPERIMENT_DIRS[experiment]
    paths = sorted(inputs) if inputs else discover_experiment_seedagg_files(experiment, exp_dir)
    if not paths:
        print(f"No seedagg files found for {experiment} under {exp_dir}", flush=True)
        return

    written = plot_experiment_seedagg_files(
        experiment,
        paths,
        out_root or paths[0].parent,
        panel_only=panel_only,
    )
    for path in written:
        print(f"Wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot whatever *_seedagg.json results exist under outputs/experiments/. "
        "Discovers all available languages (and Exp3 regimes/blocks) automatically.",
    )
    parser.add_argument(
        "--experiment",
        choices=[*EXPERIMENT_IDS, "all"],
        default="all",
        help="Which experiment to plot (default: all)",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Explicit seedagg JSON path(s); if omitted, auto-discover per experiment",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same directory as seedagg files)",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Only write combined multi-language panels (skip per-language PNGs)",
    )
    args = parser.parse_args()

    experiments = list(EXPERIMENT_IDS) if args.experiment == "all" else [args.experiment]
    explicit_inputs = [_resolve_path(raw) for raw in args.input] if args.input else None
    out_root = _resolve_path(args.output) if args.output else None

    if explicit_inputs and len(experiments) > 1:
        parser.error("Pass --experiment when using explicit --input paths")

    for experiment in experiments:
        inputs = explicit_inputs if explicit_inputs else None
        _process_experiment(
            experiment,
            inputs,
            panel_only=args.panel,
            out_root=out_root if len(experiments) == 1 else None,
        )


if __name__ == "__main__":
    main()
