"""Language: even count of a's modulo 2 (regular / modular counting)."""

from __future__ import annotations

import random
from math import comb
from typing import List

from . import FormalLanguage, REGISTRY


def _uniform_ab_string(rng: random.Random, length: int, a_parity: str) -> str:
    """Sample uniformly among strings of given length with even/odd a-count."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return ""

    if a_parity == "even":
        ks = list(range(0, length + 1, 2))
    elif a_parity == "odd":
        ks = list(range(1, length + 1, 2))
    else:
        raise ValueError("a_parity must be 'even' or 'odd'")

    weights = [comb(length, k) for k in ks]
    k = rng.choices(ks, weights=weights, k=1)[0]
    positions = rng.sample(range(length), k)
    chars = ["b"] * length
    for p in positions:
        chars[p] = "a"
    return "".join(chars)


class EvenA(FormalLanguage):
    name = "even_a"
    description = "Strings over {a,b} with an even number of a's: count(a) ≡ 0 (mod 2) (Regular)"
    chomsky_class = "Regular"

    @property
    def alphabet(self) -> List[str]:
        return ["a", "b"]

    def is_member(self, word: str) -> bool:
        return word.count("a") % 2 == 0

    def generate_positive(self, rng: random.Random, n: int) -> str:
        n = max(1, n)
        return _uniform_ab_string(rng, 2 * n, "even")

    def generate_negative(
        self, rng: random.Random, n: int, strategy: int | None = None
    ) -> str:
        n = max(1, n)
        strategy = rng.randrange(5) if strategy is None else strategy % 5
        pos = self.generate_positive(rng, n)

        if strategy == 0:
            chars = list(pos)
            idx = rng.randrange(len(chars))
            chars[idx] = "b" if chars[idx] == "a" else "a"
            return "".join(chars)

        if strategy == 1:
            return pos + "a"

        if strategy == 2:
            return _uniform_ab_string(rng, 2 * n, "odd")

        if strategy == 3:
            length = 2 * n + rng.choice([-2, 2])
            length = max(1, length)
            return _uniform_ab_string(rng, length, "odd")

        # strategy 4
        odd_len = max(1, 2 * n - 1)
        return "a" * odd_len


if __name__ == "__main__":
    lang = EvenA()
    rng = random.Random(0)
    for n in (1, 5, 10, 20):
        for _ in range(25):
            pos = lang.generate_positive(rng, n)
            assert lang.is_member(pos), pos
            assert len(pos) == 2 * n, (n, len(pos), pos)
            assert pos.count("a") % 2 == 0
            for s in range(5):
                neg = lang.generate_negative(rng, n, strategy=s)
                assert not lang.is_member(neg), (s, neg)
                assert neg.count("a") % 2 == 1
    print("even_a self-check ok")
else:
    REGISTRY.register(EvenA())
