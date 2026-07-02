"""Language: Balanced Parentheses – Dyck language over {(, )}."""

from __future__ import annotations

import random
from typing import List

from . import FormalLanguage, REGISTRY


class BalancedParens(FormalLanguage):
    name = "balanced_parens"
    description = "Balanced parentheses (Dyck language) – n pairs of ( and ) (Context-Free)"
    chomsky_class = "Context-Free"

    @property
    def alphabet(self) -> List[str]:
        return ["(", ")"]

    def is_member(self, word: str) -> bool:
        if not word or len(word) % 2 != 0:
            return False
        depth = 0
        for ch in word:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
            else:
                return False
        return depth == 0

    def _random_balanced(self, rng: random.Random, n: int) -> str:
        seq: List[str] = []
        opens = n
        closes = n
        while opens > 0 or closes > 0:
            depth = seq.count("(") - seq.count(")")
            can_close = closes > 0 and depth > 0
            can_open = opens > 0
            if can_open and can_close:
                if rng.random() < 0.55:
                    seq.append("(")
                    opens -= 1
                else:
                    seq.append(")")
                    closes -= 1
            elif can_open:
                seq.append("(")
                opens -= 1
            else:
                seq.append(")")
                closes -= 1
        return "".join(seq)

    def generate_positive(self, rng: random.Random, n: int) -> str:
        return self._random_balanced(rng, max(1, n))

    def generate_negative(self, rng: random.Random, n: int, strategy: int | None = None) -> str:
        n = max(1, n)
        strategy = rng.randrange(5) if strategy is None else strategy % 5

        if strategy == 0:
            total = 2 * n
            opens = rng.randint(0, total)
            while opens == n:
                opens = rng.randint(0, total)
            closes = total - opens
            chars = ["("] * opens + [")"] * closes
            rng.shuffle(chars)
            return "".join(chars)

        if strategy == 1:
            suffix = self._random_balanced(rng, max(1, n - 1))
            return ")" + suffix + "("

        if strategy == 2:
            valid = self._random_balanced(rng, n)
            lst = list(valid)
            idx = rng.randrange(len(lst))
            lst[idx] = ")" if lst[idx] == "(" else "("
            return "".join(lst)

        if strategy == 3:
            valid = self._random_balanced(rng, n)
            chars = list(valid)
            rng.shuffle(chars)
            return "".join(chars)

        valid = self._random_balanced(rng, n)
        return ")" + valid


REGISTRY.register(BalancedParens())
