#!/usr/bin/env python3
"""Experiment 3: hard vs easy negative difficulty analysis."""

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
from formal_language_snn.plotting import plot_difficulty_breakdown, plot_difficulty_breakdown_panel
from formal_language_snn.training import build_seedagg_payload, run_multiseed_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: negative difficulty breakdown")
    parser.add_argument("--config", default="configs/experiments/exp3_difficulty.yaml")
    parser.add_argument("--languages", nargs="*", default=list(EXPERIMENT_LANGUAGES))
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    num_seeds = 1 if args.pilot else args.num_seeds
    base_config = config_from_dict(load_config(args.config))
    out_root = PROJECT_ROOT / "outputs/experiments/exp3_difficulty"
    out_root.mkdir(parents=True, exist_ok=True)

    payloads: list[dict] = []
    for language in args.languages:
        print(f"=== exp3 {language} ({num_seeds} seeds) ===", flush=True)
        config = replace(
            base_config,
            language=language,
            stratified_test=False,
            difficulty_test=True,
        )
        records = run_multiseed_experiment(config, num_seeds, args.seed_base)
        payload = build_seedagg_payload(
            experiment="exp3_difficulty",
            config=config,
            records=records,
            num_seeds=num_seeds,
            seed_base=args.seed_base,
        )
        out_file = out_root / f"exp3_difficulty_{language}_seedagg.json"
        out_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Wrote {out_file}")
        payloads.append(payload)
        if args.plot:
            plot_difficulty_breakdown(
                payload,
                out_root / f"exp3_difficulty_{language}.png",
                title=f"Exp3: {language}",
            )

    if args.plot and len(payloads) > 1:
        panel_path = out_root / "exp3_difficulty_all.png"
        plot_difficulty_breakdown_panel(payloads, panel_path)
        print(f"Wrote {panel_path}")


if __name__ == "__main__":
    main()
