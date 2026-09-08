#!/usr/bin/env python3
"""Assert that only the factor under study varies.

Every model, language and run must agree on everything except the thing an experiment
manipulates. This script fails loudly when they do not, because an asymmetric run matrix
looks identical to a symmetric one in the output files.

Usage:
    python scripts/check_symmetry.py                 # design checks only
    python scripts/check_symmetry.py --outputs       # also audit outputs/experiments/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formal_language_snn.data import (  # noqa: E402
    EXPERIMENT_LANGUAGES,
    NEGATIVE_DIFFICULTY,
    generate_difficulty_dataset,
    generate_pair_with_meta,
)
from formal_language_snn.languages import NUM_STRATEGIES, get_language  # noqa: E402
from formal_language_snn.models import ClassicLSTM, ClassicRNN, SpikingNet  # noqa: E402
from formal_language_snn.training import MODEL_KINDS  # noqa: E402
from formal_language_snn.models.capacity import (  # noqa: E402
    BUILDERS,
    matched_hidden_sizes,
    parameters_for,
)

PROBE_NS = (3, 6, 12, 25)
SAMPLES_PER_CELL = 400
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_difficulty_ratio() -> None:
    section("hard/easy ratio is identical across languages")
    ratios = set()
    for name in EXPERIMENT_LANGUAGES:
        mapping = NEGATIVE_DIFFICULTY[name]
        hard = sorted(s for s, d in mapping.items() if d == "hard")
        easy = sorted(s for s, d in mapping.items() if d == "easy")
        ratios.add((len(hard), len(easy)))
        check(
            len(hard) + len(easy) == NUM_STRATEGIES,
            f"{name}: {len(hard) + len(easy)} strategies, expected {NUM_STRATEGIES}",
        )
        print(f"  {name:<17} hard={hard} easy={easy}")
    check(len(ratios) == 1, f"hard:easy ratio differs across languages: {sorted(ratios)}")
    print(f"  ratio {'consistent' if len(ratios) == 1 else 'INCONSISTENT'}: {sorted(ratios)}")


def check_difficulty_is_earned() -> None:
    section("declared difficulty matches the strings actually generated")
    print(f"  {'language':<17}{'strat':>6}{'declared':>10}{'surface-matched':>17}")
    for name in EXPERIMENT_LANGUAGES:
        lang = get_language(name)
        rng = random.Random(20260905)
        for strategy in range(NUM_STRATEGIES):
            matched = total = 0
            for n in PROBE_NS:
                for _ in range(SAMPLES_PER_CELL // len(PROBE_NS)):
                    neg = lang.generate_negative(rng, n, strategy=strategy)
                    if lang.is_member(neg):
                        continue
                    total += 1
                    matched += lang.respects_surface_statistics(neg, n)
            frac = matched / max(1, total)
            declared = lang.difficulty_of(strategy)
            check(
                (frac > 0.9) == (declared == "hard"),
                f"{name} strategy {strategy}: declared {declared} but "
                f"{frac:.0%} of its negatives keep the surface statistics",
            )
            print(f"  {name:<17}{strategy:>6}{declared:>10}{frac:>16.0%}")


def check_recorded_strategy() -> None:
    section("recorded strategy label matches the mutation applied")
    for name in EXPERIMENT_LANGUAGES:
        lang = get_language(name)
        rng = random.Random(11)
        agree = total = 0
        for _ in range(2000):
            n = rng.randint(2, 25)
            _, neg = generate_pair_with_meta(name, n, rng)  # strategy=None path
            total += 1
            expected = lang.difficulty_of(neg.negative_strategy)
            surface = lang.respects_surface_statistics(neg.word, n)
            agree += (expected == "hard") == surface
        rate = agree / total
        check(rate > 0.98, f"{name}: label agrees with the applied mutation only {rate:.1%} of the time")
        print(f"  {name:<17}{rate:>7.1%} of unlabelled draws carry a truthful difficulty tag")


def check_cell_balance() -> None:
    section("difficulty test set is balanced, identically, for every language")
    shapes = set()
    for name in EXPERIMENT_LANGUAGES:
        negatives = [s for s in generate_difficulty_dataset(5, seed=1, language=name) if s.label == 0]
        by_class = Counter(s.difficulty for s in negatives)
        by_strategy = Counter(s.negative_strategy for s in negatives)
        shapes.add((len(negatives), tuple(sorted(by_class.values())), tuple(sorted(by_strategy.values()))))
        check(
            len(set(by_class.values())) == 1,
            f"{name}: hard/easy cells differ in size {dict(by_class)}",
        )
        print(f"  {name:<17}{len(negatives):>4} negatives  per class {dict(by_class)}")
    check(len(shapes) == 1, f"difficulty test set shape differs across languages: {shapes}")


def check_length_matching() -> None:
    section("negative length matches its positive, per strategy")
    print(f"  {'language':<17}{'strat':>6}{'same length':>13}")
    for name in EXPERIMENT_LANGUAGES:
        lang = get_language(name)
        rng = random.Random(5)
        for strategy in range(NUM_STRATEGIES):
            same = total = 0
            for n in PROBE_NS:
                for _ in range(100):
                    pos, neg = lang.generate_pair(rng, n, strategy=strategy)
                    total += 1
                    same += len(pos) == len(neg)
            frac = same / total
            declared = lang.difficulty_of(strategy)
            if declared == "hard":
                check(
                    frac > 0.99,
                    f"{name} strategy {strategy} is declared hard but only {frac:.0%} of its "
                    "negatives match their positive's length -- length alone separates the classes",
                )
            note = "" if frac > 0.9 else "   <- length differs by design (easy)"
            print(f"  {name:<17}{strategy:>6}{frac:>12.0%}{note}")


def check_model_capacity(reference_hidden: int = 32) -> None:
    section(f"model capacity per language (GRU at hidden_size={reference_hidden} is the budget)")
    # Every trainable model family must have a capacity builder, or match_capacity crashes
    # mid-run -- after the overnight matrix has already started.
    missing = [kind for kind in MODEL_KINDS if kind not in BUILDERS]
    check(
        not missing,
        f"model kinds {missing} are trainable but have no capacity builder in models/capacity.py",
    )
    print(f"  {'language':<17}{'|Sigma|':>8}{'shared-width SNN:GRU':>22}   matched hidden / params")
    for name in EXPERIMENT_LANGUAGES:
        k = len(get_language(name).alphabet)
        gru = params(ClassicRNN(k, reference_hidden))
        snn = params(SpikingNet(k, reference_hidden))
        hidden = matched_hidden_sizes(k, "rnn", reference_hidden)
        matched = {kind: parameters_for(kind, k, hidden[kind]) for kind in hidden}
        cells = "  ".join(f"{kind}={hidden[kind]}/{matched[kind]}" for kind in hidden)
        print(f"  {name:<17}{k:>8}{f'{gru / snn:.0f}x':>22}   {cells}")
        spread = max(matched.values()) / min(matched.values())
        check(
            spread <= 1.10,
            f"{name}: parameter-matched models still differ by {spread:.2f}x "
            f"({matched}) -- widen the search or accept the gap explicitly",
        )
    print("\n  Set model.match_capacity: true so runs use the matched sizes; leaving it false")
    print("  reproduces the published comparison, in which the GRU carries ~28x the SNN's weights.")


def check_outputs() -> None:
    section("run matrix in outputs/experiments")
    files = sorted(glob.glob(str(ROOT / "outputs/experiments/*/*_seedagg.json")))
    if not files:
        print("  no *_seedagg.json found; skipping")
        return
    by_experiment: dict[str, list[dict]] = {}
    for path in files:
        data = json.loads(Path(path).read_text())
        by_experiment.setdefault(Path(path).parent.name, []).append(data)

    for experiment, payloads in by_experiment.items():
        languages = {p["language"] for p in payloads}
        seeds = {p.get("num_seeds") for p in payloads}
        knobs = {
            (
                p.get("config", {}).get("epochs"),
                p.get("config", {}).get("train_min_n"),
                p.get("config", {}).get("train_max_n"),
                p.get("config", {}).get("train_pairs"),
                p.get("config", {}).get("hidden_size"),
            )
            for p in payloads
        }
        missing = set(EXPERIMENT_LANGUAGES) - languages
        check(not missing, f"{experiment}: missing languages {sorted(missing)}")
        check(len(seeds) == 1, f"{experiment}: seed count varies across languages: {sorted(seeds)}")
        check(len(knobs) == 1, f"{experiment}: training config varies across languages: {sorted(knobs)}")
        print(f"  {experiment:<18}{len(languages)} languages, seeds={sorted(seeds)}, config variants={len(knobs)}")

    all_knobs = {
        (p.get("config", {}).get("epochs"), p.get("config", {}).get("hidden_size"))
        for payloads in by_experiment.values()
        for p in payloads
    }
    check(
        len(all_knobs) == 1,
        f"epochs/hidden_size differ between experiments: {sorted(all_knobs)} -- "
        "results from different experiments are then not comparable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check experimental symmetry")
    parser.add_argument("--outputs", action="store_true", help="also audit existing result files")
    args = parser.parse_args()

    check_difficulty_ratio()
    check_difficulty_is_earned()
    check_recorded_strategy()
    check_cell_balance()
    check_length_matching()
    check_model_capacity()
    if args.outputs:
        check_outputs()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} asymmetr{'y' if len(FAILURES) == 1 else 'ies'} found:")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("All symmetry checks passed.")


if __name__ == "__main__":
    main()
