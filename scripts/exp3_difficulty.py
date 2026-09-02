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
from formal_language_snn.plotting import plot_exp3_seedagg_files
from formal_language_snn.training import build_seedagg_payload, run_multiseed_experiment


def _parse_hidden_sizes(raw: str | None, default: int) -> list[int]:
    if not raw:
        return [max(8, default // 2), default * 2]
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted({v for v in values if v > 0 and v != default})


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: negative difficulty breakdown")
    parser.add_argument("--config", default="configs/experiments/exp3_difficulty.yaml")
    parser.add_argument("--languages", nargs="*", default=list(EXPERIMENT_LANGUAGES))
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--run-extrapolation", action="store_true")
    parser.add_argument("--extrapolation-min-n", type=int, default=11)
    parser.add_argument("--extrapolation-max-n", type=int, default=40)
    parser.add_argument(
        "--sensitivity-hidden-sizes",
        default=None,
        help="Comma-separated hidden sizes for Exp3 stability checks (default: half/double base)",
    )
    args = parser.parse_args()

    num_seeds = 1 if args.pilot else args.num_seeds
    base_config = config_from_dict(load_config(args.config))
    sensitivity_sizes = _parse_hidden_sizes(args.sensitivity_hidden_sizes, base_config.hidden_size)
    out_root = PROJECT_ROOT / "outputs/experiments/exp3_difficulty"
    out_root.mkdir(parents=True, exist_ok=True)

    regimes: list[tuple[str, int, int]] = [("in_range", base_config.difficulty_min_n, base_config.difficulty_max_n)]
    if args.run_extrapolation:
        regimes.append(("extrapolation", args.extrapolation_min_n, args.extrapolation_max_n))

    training_blocks = ["main", "control_fixed_beta"]
    training_blocks.extend(f"sensitivity_hidden_{hidden_size}" for hidden_size in sensitivity_sizes)
    runs_per_regime = len(training_blocks) * num_seeds
    print(
        f"Training blocks per language/regime: {', '.join(training_blocks)} "
        f"({runs_per_regime} full trainings each)",
        flush=True,
    )

    seedagg_paths: list[Path] = []
    for language in args.languages:
        print(f"=== exp3 {language} ({num_seeds} seeds) ===", flush=True)
        for regime_name, min_n, max_n in regimes:
            print(f"  regime={regime_name} n={min_n}-{max_n}", flush=True)
            config = replace(
                base_config,
                language=language,
                stratified_test=False,
                difficulty_test=True,
                difficulty_min_n=min_n,
                difficulty_max_n=max_n,
            )
            print("  block=main", flush=True)
            records = run_multiseed_experiment(config, num_seeds, args.seed_base)
            payload = build_seedagg_payload(
                experiment="exp3_difficulty",
                config=config,
                records=records,
                num_seeds=num_seeds,
                seed_base=args.seed_base,
                regime=regime_name,
                n_range={"min_n": min_n, "max_n": max_n},
                controls={},
            )

            control_cfg = replace(config, learn_beta=False)
            print("  block=control_fixed_beta (learn_beta=false)", flush=True)
            control_records = run_multiseed_experiment(control_cfg, num_seeds, args.seed_base)
            control_payload = build_seedagg_payload(
                experiment="exp3_difficulty_control_fixed_beta",
                config=control_cfg,
                records=control_records,
                num_seeds=num_seeds,
                seed_base=args.seed_base,
                regime=regime_name,
            )
            payload["controls"]["fixed_beta"] = {
                "description": "Compute-matched control with fixed SNN beta (learn_beta=false).",
                "metrics": control_payload["metrics"],
                "counts": control_payload["counts"],
                "stats": control_payload["stats"],
                "config": control_payload["config"],
            }

            payload["sensitivity"] = []
            for hidden_size in sensitivity_sizes:
                sens_cfg = replace(config, hidden_size=hidden_size)
                print(f"  block=sensitivity_hidden_{hidden_size}", flush=True)
                sens_records = run_multiseed_experiment(sens_cfg, num_seeds, args.seed_base)
                sens_payload = build_seedagg_payload(
                    experiment="exp3_difficulty_sensitivity",
                    config=sens_cfg,
                    records=sens_records,
                    num_seeds=num_seeds,
                    seed_base=args.seed_base,
                    regime=regime_name,
                )
                payload["sensitivity"].append(
                    {
                        "hidden_size": hidden_size,
                        "metrics": sens_payload["metrics"],
                        "stats": sens_payload["stats"],
                    }
                )

            out_file = out_root / f"exp3_difficulty_{language}_{regime_name}_seedagg.json"
            out_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"Wrote {out_file}")
            seedagg_paths.append(out_file)

            # Backward-compatible canonical filename for existing tooling.
            if regime_name == "in_range":
                canonical = out_root / f"exp3_difficulty_{language}_seedagg.json"
                canonical.write_text(json.dumps(payload, indent=2, sort_keys=True))
                print(f"Wrote {canonical}")

    if args.plot and seedagg_paths:
        for path in plot_exp3_seedagg_files(seedagg_paths, out_root):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
