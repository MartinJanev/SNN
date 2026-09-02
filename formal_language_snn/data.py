from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Tuple

import torch

from .languages import get_language

EXPERIMENT_LANGUAGES = ("anbn", "balanced_parens", "palindrome", "reber", "even_a")

LENGTH_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1, 10),
    (11, 20),
    (21, 40),
    (41, 80),
    (81, 160),
)

NEGATIVE_DIFFICULTY = {
    "anbn": {0: "hard", 1: "hard", 2: "easy", 3: "easy", 4: "easy"},
    "palindrome": {0: "hard", 1: "easy", 2: "easy", 3: "easy", 4: "hard"},
    "balanced_parens": {0: "easy", 1: "easy", 2: "hard", 3: "easy", 4: "easy"},
    "reber": {0: "hard", 1: "hard", 2: "easy", 3: "easy", 4: "easy"},
    "even_a": {0: "hard", 1: "hard", 2: "easy", 3: "easy", 4: "easy"},
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
    pos_word, neg_word = lang.generate_pair(rng, n, strategy=strategy)
    neg_strategy = strategy if strategy is not None else rng.randrange(5)
    difficulty = NEGATIVE_DIFFICULTY.get(language, {}).get(neg_strategy % 5)
    pos = Sample(word=pos_word, label=1, a_count=n, b_count=n)
    neg = Sample(
        word=neg_word,
        label=0,
        a_count=n,
        b_count=n,
        negative_strategy=neg_strategy % 5,
        difficulty=difficulty,
    )
    return pos, neg


def generate_dataset(
    num_pairs: int,
    min_n: int,
    max_n: int,
    seed: int | None = None,
    language: str = "anbn",
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
        n = rng.randint(min_n, max_n)
        pos, neg = generate_pair_with_meta(language, n, rng)
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

    dataset: List[Sample] = []
    for difficulty, strategies in strategies_by_difficulty.items():
        if not strategies:
            continue
        for _ in range(pairs_per_difficulty_cell):
            for strategy in strategies:
                n = rng.randint(min_n, max_n)
                pos, neg = generate_pair_with_meta(language, n, rng, strategy=strategy)
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
