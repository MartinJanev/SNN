from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Iterable

import torch

from . import languages


@dataclass(frozen=True)
class Sample:
    word: str
    label: int
    a_count: int
    b_count: int



def generate_dataset(
    num_pairs: int,
    min_n: int,
    max_n: int,
    seed: int | None = None,
    language: str = "anbn",
    alphabet: Iterable[str] | None = None,
) -> List[Sample]:
    """Generate a balanced dataset (positive + negative) for the selected formal language.

    Supported languages and their default alphabets:
    - "anbn": language a^n b^n (context-free, counting constraint)
    - "anbncn": language a^n b^n c^n (context-sensitive, three-way counting)
    - "palindrome": even-length palindromes (context-free)
    - "paren": balanced parentheses (context-free, classic example)
    - "equal_ab": strings with equal counts of 'a' and 'b' in any order (context-free)
    - "ends_with_abb": strings ending with 'abb' (regular language)
    - "repeat_ab": (ab)^n repetition (regular language)
    - "ww": strings of form w || w (non-context-free)
    - "prime_a": a^p where p is prime >= n (non-regular unary language)
    - "alternating": no two consecutive identical symbols (regular language)

    This returns 2 * num_pairs samples (each pair: one positive, one negative), shuffled.
    """

    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if min_n <= 0:
        raise ValueError("min_n must be positive")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    rng = random.Random(seed)
    dataset: List[Sample] = []

    # Normalize language name and get generator + default alphabet
    language = language.lower()
    try:
        generator = languages.get_generator(language)
        if alphabet is None:
            alphabet = languages.get_alphabet(language)
    except ValueError as e:
        raise ValueError(str(e)) from e

    for _ in range(num_pairs):
        n = rng.randint(min_n, max_n)

        # Call generator with rng, n, and alphabet keyword argument (if supported)
        pos, neg = generator(rng, n, alphabet=alphabet)

        dataset.append(Sample(word=pos, label=1, a_count=n, b_count=n))
        dataset.append(Sample(word=neg, label=0, a_count=n, b_count=n))

    rng.shuffle(dataset)
    return dataset


def word_to_tensor(word: str, alphabet: Iterable[str] | None = None) -> torch.Tensor:
    """Convert a word to a spike tensor with shape [time_steps, batch_size=1, features=len(alphabet)].

    If alphabet is None, it is inferred from the symbols present in the word (sorted), but for
    consistent model input sizes it's recommended to provide a fixed alphabet.
    """

    if not word:
        raise ValueError("word must not be empty")

    symbols = list(word)
    if alphabet is None:
        alphabet_list = sorted(set(symbols))
    else:
        alphabet_list = list(alphabet)

    # build one-hot encodings according to alphabet_list
    idx_map = {s: i for i, s in enumerate(alphabet_list)}

    spike_sequence = []
    for symbol in symbols:
        if symbol not in idx_map:
            raise ValueError(f"Unsupported symbol: {symbol!r} for alphabet {alphabet_list}")
        vec = [0.0] * len(alphabet_list)
        vec[idx_map[symbol]] = 1.0
        spike_sequence.append(vec)

    return torch.tensor(spike_sequence, dtype=torch.float32).unsqueeze(1)
