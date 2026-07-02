"""Language: a^n b^n (context-free)."""

from __future__ import annotations

import random
from typing import List

from . import FormalLanguage, REGISTRY


class AnBn(FormalLanguage):
    name = "anbn"
    description = "a^n b^n – equal number of a's followed by equal number of b's (Context-Free)"
    chomsky_class = "Context-Free"

    @property
    def alphabet(self) -> List[str]:
        return ["a", "b"]

    def is_member(self, word: str) -> bool:
        if not word:
            return False
        i = 0
        while i < len(word) and word[i] == "a":
            i += 1
        if i == 0 or i == len(word):
            return False
        return word[i:] == "b" * (len(word) - i) and i == len(word) - i

    def generate_positive(self, rng: random.Random, n: int) -> str:
        n = max(1, n)
        return "a" * n + "b" * n

    def generate_negative(self, rng: random.Random, n: int, strategy: int | None = None) -> str:
        n = max(1, n)
        strategy = rng.randrange(5) if strategy is None else strategy % 5

        if strategy == 0:
            delta = rng.choice([1, 2, max(1, n // 2)])
            m = n + (delta if rng.random() < 0.5 else -delta)
            m = max(1, m)
            if m == n:
                m = n + 1
            return "a" * m + "b" * n

        if strategy == 1:
            delta = rng.choice([1, 2, max(1, n // 2)])
            m = n + (delta if rng.random() < 0.5 else -delta)
            m = max(1, m)
            if m == n:
                m = n + 1
            return "a" * n + "b" * m

        if strategy == 2:
            return "b" * n + "a" * n

        if strategy == 3:
            chars = list("a" * n + "b" * n)
            for _ in range(20):
                rng.shuffle(chars)
                candidate = "".join(chars)
                if not self.is_member(candidate):
                    return candidate
            return "b" + "a" * n + "b" * (n - 1)

        length = rng.randint(max(1, 2 * n - 2), 2 * n + 2)
        return "".join(rng.choice(self.alphabet) for _ in range(length))


REGISTRY.register(AnBn())
