"""Formal language generators and registry.

This module defines the supported formal languages and their corresponding
positive/negative example generators. Each generator function takes:
  - rng: random.Random instance for reproducibility
  - n: parameter controlling language complexity (length, repetition count, etc.)
  - alphabet: (optional, only if needed) the symbol alphabet to draw from

And returns a tuple (positive_example, negative_example) of strings.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable


# ============================================================================
# Language Generators
# ============================================================================


def _gen_anbn(rng: random.Random, n: int, **kwargs) -> tuple[str, str]:
    """Language a^n b^n: n 'a's followed by n 'b's."""
    pos = ("a" * n) + ("b" * n)
    # negative: different counts
    n_a = rng.randint(1, max(1, n + 1))
    n_b = rng.randint(1, max(1, n + 2))
    neg = ("a" * n_a) + ("b" * n_b)
    return pos, neg


def _gen_anbncn(rng: random.Random, n: int, **kwargs) -> tuple[str, str]:
    """Language a^n b^n c^n: context-sensitive, requires counting across three symbols."""
    pos = ("a" * n) + ("b" * n) + ("c" * n)
    # negative: differing counts
    n_a = rng.randint(1, max(1, n + 1))
    n_b = rng.randint(1, max(1, n + 1))
    n_c = rng.randint(1, max(1, n + 1))
    while n_a == n_b == n_c:
        n_c = rng.randint(1, max(1, n + 2))
    neg = ("a" * n_a) + ("b" * n_b) + ("c" * n_c)
    return pos, neg


def _gen_palindrome(rng: random.Random, n: int, alphabet: Iterable[str]) -> tuple[str, str]:
    """Even-length palindromes over the given alphabet."""
    half = [rng.choice(list(alphabet)) for _ in range(n)]
    pos = "".join(half + list(reversed(half)))
    # negative: alter one random position
    neg_list = list(pos)
    if neg_list:
        idx = rng.randrange(len(neg_list))
        alternatives = [s for s in alphabet if s != neg_list[idx]]
        if alternatives:
            neg_list[idx] = rng.choice(alternatives)
    neg = "".join(neg_list)
    return pos, neg


def _gen_balanced_parentheses(rng: random.Random, n: int, **kwargs) -> tuple[str, str]:
    """Balanced parentheses: n pairs of '(' and ')'."""
    seq = []
    open_rem = n
    close_rem = n
    stack = 0
    while open_rem > 0 or close_rem > 0:
        # if only opens remain, must open; if no opens left, must close
        if open_rem > 0 and (stack == 0 or rng.random() < 0.6):
            seq.append("(")
            stack += 1
            open_rem -= 1
        else:
            if stack > 0:
                seq.append(")")
                stack -= 1
                close_rem -= 1
            else:
                # forced to open because stack is empty
                seq.append("(")
                stack += 1
                open_rem -= 1

    pos = "".join(seq)
    # negative: shuffle or flip to break balance
    neg_list = list(pos)
    if neg_list:
        if rng.random() < 0.5:
            i = rng.randrange(len(neg_list))
            neg_list[i] = ")" if neg_list[i] == "(" else "("
        else:
            i, j = sorted(rng.sample(range(len(neg_list)), k=min(2, len(neg_list))))
            neg_list[i], neg_list[j] = neg_list[j], neg_list[i]
    neg = "".join(neg_list)
    return pos, neg


def _gen_equal_counts_ab(rng: random.Random, n: int, **kwargs) -> tuple[str, str]:
    """Strings with equal counts of 'a' and 'b' in any order (context-free)."""
    chars = ["a"] * n + ["b"] * n
    rng.shuffle(chars)
    pos = "".join(chars)
    # negative: unbalanced
    m_b = rng.randint(0, max(1, n * 2))
    while m_b == n:
        m_b = rng.randint(0, max(1, n * 2))
    neg_chars = ["a"] * n + ["b"] * m_b
    rng.shuffle(neg_chars)
    neg = "".join(neg_chars)
    return pos, neg


def _gen_ends_with_abb(rng: random.Random, n: int, alphabet: Iterable[str]) -> tuple[str, str]:
    """Strings that end with 'abb' (regular language property)."""
    core = [rng.choice(list(alphabet)) for _ in range(max(0, n))]
    pos = "".join(core) + "abb"
    # negative: different suffix
    neg_list = list(pos)
    if len(neg_list) >= 3:
        neg_list[-1] = "a" if neg_list[-1] != "a" else "b"
    neg = "".join(neg_list)
    return pos, neg


def _gen_repeat_ab(rng: random.Random, n: int, **kwargs) -> tuple[str, str]:
    """Strings of form (ab)^n (regular language)."""
    pos = "ab" * n
    neg = list(pos)
    if neg:
        i = rng.randrange(len(neg))
        neg[i] = "a" if neg[i] != "a" else "b"
    return pos, "".join(neg)


def _gen_ww(rng: random.Random, n: int, alphabet: Iterable[str]) -> tuple[str, str]:
    """Strings of form w || w: concatenation of a word with itself (non-context-free)."""
    half = [rng.choice(list(alphabet)) for _ in range(n)]
    pos = "".join(half + half)
    neg_list = list(pos)
    if len(neg_list) >= 1:
        # change one char in second half
        idx = n + (rng.randrange(max(1, n)) if n > 0 else 0)
        if idx < len(neg_list):
            neg_list[idx] = rng.choice([s for s in alphabet if s != neg_list[idx]])
    neg = "".join(neg_list)
    return pos, neg


def _next_prime_ge(x: int) -> int:
    """Find the smallest prime >= x."""
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0:
                return False
            i += 2
        return True

    p = max(2, x)
    while not is_prime(p):
        p += 1
    return p


def _gen_prime_a(rng: random.Random, n: int, **kwargs) -> tuple[str, str]:
    """Strings a^p where p is prime >= n (non-regular, unary alphabet)."""
    p = _next_prime_ge(max(2, n))
    pos = "a" * p
    neg = "a" * (p + 1)
    return pos, neg


def _gen_alternating(rng: random.Random, n: int, alphabet: Iterable[str]) -> tuple[str, str]:
    """Strings with no two consecutive identical symbols (regular)."""
    symbols = list(alphabet)
    if len(symbols) < 2:
        symbols = ["a", "b"]
    s = [rng.choice(symbols)]
    for _ in range(1, n):
        choices = [c for c in symbols if c != s[-1]]
        s.append(rng.choice(choices))
    pos = "".join(s)
    # negative: create a double (adjacent identical symbols)
    neg_list = list(pos)
    if neg_list:
        idx = rng.randrange(len(neg_list))
        neg_list[idx] = neg_list[idx - 1] if idx > 0 else neg_list[idx]
    neg = "".join(neg_list)
    return pos, neg


# ============================================================================
# Language Registry
# ============================================================================


# Type alias for language generator function signature
LanguageGenerator = Callable[[random.Random, int], tuple[str, str]]


class LanguageRegistry:
    """Registry of supported formal languages and their properties."""

    def __init__(self):
        self._generators: dict[str, LanguageGenerator] = {}
        self._alphabets: dict[str, list[str]] = {}
        self._descriptions: dict[str, str] = {}

        # Register all languages
        self._register(
            "anbn",
            _gen_anbn,
            ["a", "b"],
            "Language a^n b^n (context-free, counting constraint)",
        )
        self._register(
            "anbncn",
            _gen_anbncn,
            ["a", "b", "c"],
            "Language a^n b^n c^n (context-sensitive, three-way counting)",
        )
        self._register(
            "palindrome",
            _gen_palindrome,
            ["a", "b"],
            "Even-length palindromes (context-free)",
        )
        self._register(
            "paren",
            _gen_balanced_parentheses,
            ["(", ")"],
            "Balanced parentheses (context-free, classic example)",
        )
        self._register(
            "equal_ab",
            _gen_equal_counts_ab,
            ["a", "b"],
            "Strings with equal counts of 'a' and 'b' in any order (context-free)",
        )
        self._register(
            "ends_with_abb",
            _gen_ends_with_abb,
            ["a", "b"],
            "Strings ending with 'abb' (regular language)",
        )
        self._register(
            "repeat_ab",
            _gen_repeat_ab,
            ["a", "b"],
            "(ab)^n repetition (regular language)",
        )
        self._register(
            "ww",
            _gen_ww,
            ["a", "b"],
            "Strings of form w || w (non-context-free)",
        )
        self._register(
            "prime_a",
            _gen_prime_a,
            ["a"],
            "a^p where p is prime >= n (non-regular unary language)",
        )
        self._register(
            "alternating",
            _gen_alternating,
            ["a", "b"],
            "No two consecutive identical symbols (regular language)",
        )

    def _register(
        self,
        name: str,
        generator: LanguageGenerator,
        alphabet: list[str],
        description: str,
    ) -> None:
        """Register a language."""
        self._generators[name] = generator
        self._alphabets[name] = alphabet
        self._descriptions[name] = description

    def get_generator(self, language: str) -> LanguageGenerator:
        """Get the generator function for a language."""
        language = language.lower()
        if language not in self._generators:
            raise ValueError(
                f"Unknown language: {language!r}. Supported: {', '.join(sorted(self._generators.keys()))}"
            )
        return self._generators[language]

    def get_alphabet(self, language: str) -> list[str]:
        """Get the default alphabet for a language."""
        language = language.lower()
        if language not in self._alphabets:
            raise ValueError(f"Unknown language: {language!r}")
        return self._alphabets[language]

    def get_description(self, language: str) -> str:
        """Get a description of the language."""
        language = language.lower()
        if language not in self._descriptions:
            raise ValueError(f"Unknown language: {language!r}")
        return self._descriptions[language]

    def list_languages(self) -> dict[str, str]:
        """Return all supported languages and their descriptions."""
        return {name: self._descriptions[name] for name in sorted(self._generators.keys())}


# Global registry instance
_REGISTRY = LanguageRegistry()


# Public API
def get_generator(language: str) -> LanguageGenerator:
    """Get the generator function for a language."""
    return _REGISTRY.get_generator(language)


def get_alphabet(language: str) -> list[str]:
    """Get the default alphabet for a language."""
    return _REGISTRY.get_alphabet(language)


def get_description(language: str) -> str:
    """Get a description of a language."""
    return _REGISTRY.get_description(language)


def list_languages() -> dict[str, str]:
    """Return all supported languages and their descriptions."""
    return _REGISTRY.list_languages()

