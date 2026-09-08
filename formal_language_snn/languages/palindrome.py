"""Language: Even-Length Palindromes over {a, b, c}."""

from __future__ import annotations

import random
from typing import List

from . import FormalLanguage, REGISTRY


class EvenPalindrome(FormalLanguage):
    name = "palindrome"
    description = "Even-length palindromes over {a,b,c}: s = reverse(s) (Context-Free)"
    chomsky_class = "Context-Free"
    # 2 = w w (repeat instead of mirror), 4 = two symbols transposed. Both keep length 2n
    # and leave every symbol count even, which is the surface signature of a palindrome.
    HARD_STRATEGIES = (2, 4)

    @property
    def alphabet(self) -> List[str]:
        # Three symbols, matching the manuscript. A 26-symbol alphabet made the task easier,
        # not harder: a random even-length string is almost never a palindrome, so negatives
        # were rejectable on surface statistics alone, and it inflated the input width to 26.
        return ["a", "b", "c"]

    def is_member(self, word: str) -> bool:
        if not word or len(word) % 2 != 0:
            return False
        return word == word[::-1]

    def respects_surface_statistics(self, word: str, n: int) -> bool:
        if len(word) != 2 * max(1, n):
            return False
        counts: dict[str, int] = {}
        for ch in word:
            counts[ch] = counts.get(ch, 0) + 1
        return all(v % 2 == 0 for v in counts.values())

    def generate_positive(self, rng: random.Random, n: int) -> str:
        n = max(1, n)
        half = [rng.choice(self.alphabet) for _ in range(n)]
        return "".join(half) + "".join(reversed(half))

    def generate_negative(
        self,
        rng: random.Random,
        n: int,
        strategy: int | None = None,
        positive: str | None = None,
    ) -> str:
        n = max(1, n)
        strategy = rng.randrange(5) if strategy is None else strategy % 5
        base = positive if positive is not None else self.generate_positive(rng, n)

        if strategy == 0:
            lst = list(base)
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
            # Reject-sample until the string is both a non-member and genuinely *easy*: over a
            # 3-symbol alphabet a random even-length string lands on all-even symbol counts
            # often enough (~14 % of draws) to look like a palindrome by surface statistics,
            # which would put an unrejectable-by-counting word in the easy column.
            length = 2 * n
            for _ in range(50):
                s = "".join(rng.choice(self.alphabet) for _ in range(length))
                if not self.is_member(s) and not self.respects_surface_statistics(s, n):
                    return s
            # Fall back to an explicit count violation: swap one symbol for another, which
            # makes two counts odd.
            chars = list(base)
            idx = rng.randrange(length)
            chars[idx] = rng.choice([x for x in self.alphabet if x != chars[idx]])
            return "".join(chars)

        if strategy == 2:
            # w+w is a palindrome exactly when w is, so redraw w rather than flipping a
            # symbol: a flip would change two counts and quietly demote this to an easy
            # negative. Over {a,b,c} that happened ~4 % of the time.
            for _ in range(50):
                w = "".join(rng.choice(self.alphabet) for _ in range(n))
                candidate = w * 2
                if not self.is_member(candidate):
                    return candidate
            return self._count_breaking_fallback(rng, base)

        if strategy == 3:
            # Drop the first symbol and append a *different* one, so the multiset always
            # changes. Appending any symbol at random left the counts untouched whenever it
            # matched the dropped one -- 1 draw in 3 over {a,b,c}, which made a third of this
            # "easy" cell unrejectable by counting.
            palindrome = base
            tail = rng.choice([c for c in self.alphabet if c != palindrome[0]])
            shifted = palindrome[1:] + tail
            if not self.is_member(shifted):
                return shifted
            return self._count_breaking_fallback(rng, base)

        # Strategy 4: transpose two differing symbols. The multiset -- and therefore every
        # symbol count -- is preserved exactly, so only mirror symmetry is violated. If this
        # base admits no such transposition, draw a fresh positive rather than breaking the
        # counts; only n == 1 is genuinely impossible.
        for attempt in range(50):
            chars = list(base if attempt == 0 else self.generate_positive(rng, n))
            differing = [
                (i, j)
                for i in range(len(chars))
                for j in range(i + 1, len(chars))
                if chars[i] != chars[j]
            ]
            if not differing:
                continue
            i, j = rng.choice(differing)
            chars[i], chars[j] = chars[j], chars[i]
            candidate = "".join(chars)
            if not self.is_member(candidate):
                return candidate
        # At n == 1 every length-2 word with even symbol counts is "xx", which is a member,
        # so no count-preserving non-member exists at all.
        return self._count_breaking_fallback(rng, base)

    def _count_breaking_fallback(self, rng: random.Random, base: str) -> str:
        """Last resort: swap one symbol, which changes two counts."""
        chars = list(base)
        idx = rng.randrange(len(chars))
        chars[idx] = rng.choice([c for c in self.alphabet if c != chars[idx]])
        return "".join(chars)


REGISTRY.register(EvenPalindrome())
