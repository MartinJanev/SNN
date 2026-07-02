from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Tuple

import torch

from .languages import get_language as _get_language
from .languages import list_languages as _list_languages

EXPERIMENT_LANGUAGES = ("anbn", "balanced_parens", "palindrome")

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
}


def get_language(name: str):
    return _get_language(name)


def list_languages():
    return _list_languages()


def word_length(n: int) -> int:
    return 2 * n


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
    alphabet: Iterable[str] | None = None,
) -> List[Sample]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if min_n <= 0:
        raise ValueError("min_n must be positive")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    rng = random.Random(seed)
    language = language.lower()
    lang = get_language(language)
    if alphabet is None:
        alphabet = lang.alphabet

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
    alphabet: Iterable[str] | None = None,
) -> List[Sample]:
    rng = random.Random(seed)
    language = language.lower()
    lang = get_language(language)
    if alphabet is None:
        alphabet = lang.alphabet

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
    alphabet: Iterable[str] | None = None,
) -> List[Sample]:
    rng = random.Random(seed)
    language = language.lower()
    lang = get_language(language)
    if alphabet is None:
        alphabet = lang.alphabet

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
                n = rng.randint(1, 10)
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
