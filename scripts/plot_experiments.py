#!/usr/bin/env python3
"""Plot aggregated experiment JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_language_snn.paths import PROJECT_ROOT
from formal_language_snn.plotting import (
    plot_beta_sweep,
    plot_difficulty_breakdown,
    plot_length_accuracy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["exp1", "exp2", "exp3"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    data = json.loads(in_path.read_text())
    out = Path(args.output) if args.output else in_path.with_suffix(".png")

    if args.experiment == "exp1":
        plot_length_accuracy(data, out)
    elif args.experiment == "exp2":
        plot_beta_sweep(data, out)
    else:
        plot_difficulty_breakdown(data, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
