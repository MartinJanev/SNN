from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Tuple

import torch

from .languages import NUM_STRATEGIES, get_language

EXPERIMENT_LANGUAGES = ("anbn", "balanced_parens", "palindrome", "reber", "even_a")

LENGTH_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1, 10),
    (11, 20),
    (21, 40),
    (41, 80),
    (81, 160),
)

# Derived from each language's declared HARD_STRATEGIES rather than hand-maintained, so the
# hard/easy ratio is identical (2:3) for every language by construction. A negative is "hard"
# when it still satisfies every surface statistic that positives share -- same length, same
# symbol counts -- and can therefore only be rejected by tracking real structure.
# `scripts/check_symmetry.py` verifies each declaration against the strings actually generated.
NEGATIVE_DIFFICULTY = {
    name: {s: get_language(name).difficulty_of(s) for s in range(NUM_STRATEGIES)}
    for name in EXPERIMENT_LANGUAGES
}


def strategy_plot_order(language: str) -> list[int]:
    difficulty_map = NEGATIVE_DIFFICULTY.get(language.lower(), {})
    hard_ids = sorted(sid for sid, diff in difficulty_map.items() if diff == "hard")
    easy_ids = sorted(sid for sid, diff in difficulty_map.items() if diff == "easy")
    return hard_ids + easy_ids


def strategy_difficulty_label(language: str, strategy_id: int) -> str:
    difficulty_map = NEGATIVE_DIFFICULTY.get(language.lower(), {})
    if strategy_id not in difficulty_map:
        return f"S{strategy_id}"
    diff = difficulty_map[strategy_id]
    group = [sid for sid, tag in sorted(difficulty_map.items()) if tag == diff]
    index = group.index(strategy_id) + 1
    return f"{'H' if diff == 'hard' else 'E'}{index}"


def length_bucket(word_len: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= word_len <= hi:
            return f"{lo}-{hi}"
    return "other"


@dataclass(frozen=True)
class Sample:
    word: str
    label: int
    a_count: int
    b_count: int
    negative_strategy: int | None = None
    difficulty: str | None = None


def generate_pair_with_meta(
    language: str,
    n: int,
    rng: random.Random,
    strategy: int | None = None,
) -> Tuple[Sample, Sample]:
    lang = get_language(language)
    # Draw the strategy here and pass it down, so the id recorded on the Sample is the
    # mutation that was actually applied. Letting generate_negative pick its own and then
    # drawing a second id independently made the label agree with reality 1 time in 5.
    neg_strategy = rng.randrange(NUM_STRATEGIES) if strategy is None else strategy % NUM_STRATEGIES
    pos_word, neg_word = lang.generate_pair(rng, n, strategy=neg_strategy)
    difficulty = NEGATIVE_DIFFICULTY.get(language, {}).get(neg_strategy)
    pos = Sample(word=pos_word, label=1, a_count=n, b_count=n)
    neg = Sample(
        word=neg_word,
        label=0,
        a_count=n,
        b_count=n,
        negative_strategy=neg_strategy,
        difficulty=difficulty,
    )
    return pos, neg


def generate_dataset(
    num_pairs: int,
    min_n: int,
    max_n: int,
    seed: int | None = None,
    language: str = "anbn",
    max_word_len: int | None = None,
) -> List[Sample]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if min_n <= 0:
        raise ValueError("min_n must be positive")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    rng = random.Random(seed)
    language = language.lower()

    dataset: List[Sample] = []
    for _ in range(num_pairs):
        # Redraw until both members fit under the cap. Capping n instead would not be
        # enough: n is only a target, and reber overshoots it by up to +2 while several
        # negative strategies change the length -- so a "training" word could land in an
        # extrapolation bucket and quietly make the length claim untestable.
        for _attempt in range(100):
            n = rng.randint(min_n, max_n)
            pos, neg = generate_pair_with_meta(language, n, rng)
            if max_word_len is None or max(len(pos.word), len(neg.word)) <= max_word_len:
                break
        else:
            raise RuntimeError(
                f"{language}: could not draw a pair under max_word_len={max_word_len} "
                f"with n in [{min_n}, {max_n}]"
            )
        dataset.extend([pos, neg])

    rng.shuffle(dataset)
    return dataset


def generate_stratified_dataset(
    pairs_per_bucket: int,
    seed: int | None = None,
    language: str = "anbn",
) -> List[Sample]:
    rng = random.Random(seed)
    language = language.lower()

    dataset: List[Sample] = []
    for lo, hi in LENGTH_BUCKETS:
        min_n = max(1, (lo + 1) // 2)
        max_n = hi // 2
        for _ in range(pairs_per_bucket):
            n = rng.randint(min_n, max_n)
            pos, neg = generate_pair_with_meta(language, n, rng)
            dataset.extend([pos, neg])

    rng.shuffle(dataset)
    return dataset


def generate_difficulty_dataset(
    pairs_per_difficulty_cell: int,
    seed: int | None = None,
    language: str = "anbn",
    min_n: int = 1,
    max_n: int = 10,
    min_word_len: int | None = None,
    max_word_len: int | None = None,
) -> List[Sample]:
    if pairs_per_difficulty_cell <= 0:
        raise ValueError("pairs_per_difficulty_cell must be positive")
    if min_n <= 0:
        raise ValueError("min_n must be positive")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    rng = random.Random(seed)
    language = language.lower()

    difficulty_map = NEGATIVE_DIFFICULTY.get(language, {})
    strategies_by_difficulty: dict[str, list[int]] = {"hard": [], "easy": []}
    for strategy, difficulty in difficulty_map.items():
        strategies_by_difficulty.setdefault(difficulty, []).append(strategy)

    # Emit the same number of pairs for "hard" as for "easy", and the same number for every
    # strategy within a class. Iterating pairs_per_cell x |strategies| instead would make the
    # easy column rest on 1.5x the samples of the hard column, since the split is 2:3.
    sizes = [len(v) for v in strategies_by_difficulty.values() if v]
    pairs_per_class = pairs_per_difficulty_cell
    for size in sizes:
        pairs_per_class *= size

    dataset: List[Sample] = []
    for difficulty, strategies in strategies_by_difficulty.items():
        if not strategies:
            continue
        pairs_per_strategy = pairs_per_class // len(strategies)
        for _ in range(pairs_per_strategy):
            for strategy in strategies:
                # Bound the realised length, not just n. Several strategies lengthen the
                # negative (anbn varies a count by up to n//2), so an "in-range" cell drawn
                # at n<=20 could still emit a 47-symbol word the model never trained on.
                for _attempt in range(200):
                    n = rng.randint(min_n, max_n)
                    pos, neg = generate_pair_with_meta(language, n, rng, strategy=strategy)
                    longest = max(len(pos.word), len(neg.word))
                    shortest = min(len(pos.word), len(neg.word))
                    if (max_word_len is None or longest <= max_word_len) and (
                        min_word_len is None or shortest >= min_word_len
                    ):
                        break
                else:
                    raise RuntimeError(
                        f"{language} strategy {strategy}: no pair with word length in "
                        f"[{min_word_len}, {max_word_len}] for n in [{min_n}, {max_n}]"
                    )
                neg = Sample(
                    word=neg.word,
                    label=0,
                    a_count=n,
                    b_count=n,
                    negative_strategy=strategy,
                    difficulty=difficulty,
                )
                dataset.extend([pos, neg])

    rng.shuffle(dataset)
    return dataset


def word_to_tensor(word: str, alphabet: Iterable[str] | None = None) -> torch.Tensor:
    if not word:
        raise ValueError("word must not be empty")

    symbols = list(word)
    alphabet_list = sorted(set(symbols)) if alphabet is None else list(alphabet)
    idx_map = {s: i for i, s in enumerate(alphabet_list)}

    spike_sequence = []
    for symbol in symbols:
        if symbol not in idx_map:
            raise ValueError(f"Unsupported symbol: {symbol!r} for alphabet {alphabet_list}")
        vec = [0.0] * len(alphabet_list)
        vec[idx_map[symbol]] = 1.0
        spike_sequence.append(vec)

    return torch.tensor(spike_sequence, dtype=torch.float32).unsqueeze(1)
