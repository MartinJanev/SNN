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
from formal_language_snn.plotting import plot_beta_sweep
from formal_language_snn.training import aggregate_results, run_multiseed_experiment

DEFAULT_BETAS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


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

    for language in args.languages:
        rnn_config = replace(base_config, language=language, learn_beta=True)
        rnn_records = run_multiseed_experiment(rnn_config, num_seeds, args.seed_base, quiet=True)
        rnn_agg = aggregate_results(rnn_records)

        metrics = dict(rnn_agg.metrics)
        for beta in args.betas:
            beta_config = replace(
                base_config,
                language=language,
                beta=beta,
                learn_beta=False,
            )
            snn_records = run_multiseed_experiment(beta_config, num_seeds, args.seed_base, quiet=True)
            snn_agg = aggregate_results(snn_records)
            metrics[f"snn_beta_{beta}"] = snn_agg.metrics["snn_accuracy"]

        payload = {
            "experiment": "exp2_beta",
            "language": language,
            "betas": args.betas,
            "num_seeds": num_seeds,
            "metrics": metrics,
        }
        out_file = out_root / f"exp2_beta_{language}_seedagg.json"
        out_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Wrote {out_file}")
        if args.plot:
            plot_beta_sweep(payload, out_root / f"exp2_beta_{language}.png", title=f"Exp2: {language}")


if __name__ == "__main__":
    main()
