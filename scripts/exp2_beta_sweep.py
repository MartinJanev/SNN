#!/usr/bin/env python3
"""Experiment 2: SNN beta sweep with fixed learn_beta=False."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formal_language_snn.cli import config_from_dict, load_config
from formal_language_snn.data import EXPERIMENT_LANGUAGES
from formal_language_snn.paths import PROJECT_ROOT
from formal_language_snn.plotting import plot_exp2_seedagg_files
from formal_language_snn.training import (
    SPIKING_KINDS,
    aggregate_results,
    build_seedagg_payload,
    run_multiseed_experiment,
)

DEFAULT_BETAS = [i * 0.1 for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2: SNN beta sweep")
    parser.add_argument("--config", default="configs/experiments/exp2_beta.yaml")
    parser.add_argument("--languages", nargs="*", default=list(EXPERIMENT_LANGUAGES))
    parser.add_argument("--betas", nargs="*", type=float, default=DEFAULT_BETAS)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    num_seeds = 1 if args.pilot else args.num_seeds
    base_config = config_from_dict(load_config(args.config))
    out_root = PROJECT_ROOT / "outputs/experiments/exp2_beta"
    out_root.mkdir(parents=True, exist_ok=True)

    seedagg_paths: list[Path] = []
    for language in args.languages:
        print(f"=== exp2 {language}: baseline ({num_seeds} seeds) ===", flush=True)
        config = replace(base_config, language=language)
        # The baseline block trains the full roster once. It is what the gated models'
        # constant reference lines are drawn from, so they stay in every exp2 table and
        # figure without being retrained at each sweep point, where beta means nothing.
        baseline_config = replace(config, learn_beta=True)
        baseline_records = run_multiseed_experiment(baseline_config, num_seeds, args.seed_base)

        metrics = dict(aggregate_results(baseline_records).metrics)
        swept = ", ".join(SPIKING_KINDS)
        for beta in args.betas:
            print(f"=== exp2 {language}: {swept} beta={beta} ({num_seeds} seeds) ===", flush=True)
            beta_config = replace(
                config,
                beta=beta,
                learn_beta=False,
                models=SPIKING_KINDS,
            )
            sweep_records = run_multiseed_experiment(beta_config, num_seeds, args.seed_base)
            sweep_metrics = aggregate_results(sweep_records).metrics
            for kind in SPIKING_KINDS:
                metrics[f"{kind}_beta_{beta}"] = sweep_metrics[f"{kind}_accuracy"]

        payload = build_seedagg_payload(
            experiment="exp2_beta",
            config=config,
            records=baseline_records,
            num_seeds=num_seeds,
            seed_base=args.seed_base,
            betas=args.betas,
            metrics=metrics,
        )
        out_file = out_root / f"exp2_beta_{language}_seedagg.json"
        out_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Wrote {out_file}")
        seedagg_paths.append(out_file)

    if args.plot and seedagg_paths:
        for path in plot_exp2_seedagg_files(seedagg_paths, out_root):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
