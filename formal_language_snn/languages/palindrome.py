"""Language: Even-Length Palindromes over {a, b, c}."""

from __future__ import annotations

import random
from typing import List

from . import FormalLanguage, REGISTRY


class EvenPalindrome(FormalLanguage):
    name = "palindrome"
    description = "Even-length palindromes over {a,b,c}: s = reverse(s) (Context-Free)"
    chomsky_class = "Context-Free"

    @property
    def alphabet(self) -> List[str]:
        return ["a", "b", "c"]

    def is_member(self, word: str) -> bool:
        if not word or len(word) % 2 != 0:
            return False
        return word == word[::-1]

    def generate_positive(self, rng: random.Random, n: int) -> str:
        n = max(1, n)
        half = [rng.choice(self.alphabet) for _ in range(n)]
        return "".join(half) + "".join(reversed(half))

    def generate_negative(self, rng: random.Random, n: int, strategy: int | None = None) -> str:
        n = max(1, n)
        strategy = rng.randrange(5) if strategy is None else strategy % 5

        if strategy == 0:
            half = [rng.choice(self.alphabet) for _ in range(n)]
            pos = "".join(half) + "".join(reversed(half))
            lst = list(pos)
            idx = rng.randint(n, 2 * n - 1)
            mirror = 2 * n - 1 - idx
            alternatives = [s for s in self.alphabet if s != lst[idx]]
            lst[idx] = rng.choice(alternatives)
            candidate = "".join(lst)
            if not self.is_member(candidate):
                return candidate
            lst[mirror] = rng.choice([s for s in self.alphabet if s != lst[mirror]])
            return "".join(lst)

        if strategy == 1:
            length = 2 * n
            for _ in range(50):
                s = "".join(rng.choice(self.alphabet) for _ in range(length))
                if not self.is_member(s):
                    return s
            chars = list(s)
            chars[0] = rng.choice([x for x in self.alphabet if x != chars[0]])
            return "".join(chars)

        if strategy == 2:
            w = [rng.choice(self.alphabet) for _ in range(n)]
            candidate = "".join(w) * 2
            if not self.is_member(candidate):
                return candidate
            lst = list(candidate)
            lst[-1] = rng.choice([s for s in self.alphabet if s != lst[-1]])
            return "".join(lst)

        if strategy == 3:
            half = [rng.choice(self.alphabet) for _ in range(n)]
            palindrome = "".join(half) + "".join(reversed(half))
            shifted = palindrome[1:] + rng.choice(self.alphabet)
            if not self.is_member(shifted):
                return shifted
            return palindrome[1:] + rng.choice([s for s in self.alphabet if s != palindrome[0]])

        half = [rng.choice(self.alphabet) for _ in range(n)]
        pos = list("".join(half) + "".join(reversed(half)))
        if n >= 2:
            i = rng.randrange(n)
            j = rng.randrange(n)
            while j == i:
                j = rng.randrange(n)
            mi, mj = 2 * n - 1 - i, 2 * n - 1 - j
            pos[mi], pos[mj] = pos[mj], pos[mi]
            pos[mi] = rng.choice([s for s in self.alphabet if s != pos[mi]])
        else:
            pos[0] = rng.choice([s for s in self.alphabet if s != pos[0]])
        return "".join(pos)


REGISTRY.register(EvenPalindrome())
