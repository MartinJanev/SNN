#!/usr/bin/env python3
"""Export aggregated experiment metrics as a plain-text table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_language_snn.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    data = json.loads(in_path.read_text())

    print(f"# {data.get('experiment', 'experiment')}  language={data.get('language', '?')}")
    print(f"# seeds={data.get('num_seeds', '?')}")
    print("metric\tmean\tstd")
    for key, stats in sorted(data.get("metrics", {}).items()):
        print(f"{key}\t{stats['mean']:.4f}\t{stats['std']:.4f}")


if __name__ == "__main__":
    main()
