"""Formal language definitions: abstract base + registry."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


#: Every language exposes exactly this many negative-mutation strategies, split into
#: HARD_STRATEGIES and the remainder, so the hard/easy ratio is identical across the suite.
NUM_STRATEGIES = 5


class FormalLanguage(ABC):
    name: str
    description: str
    chomsky_class: str

    #: Strategy ids whose negatives still satisfy every surface statistic that positives
    #: share, so they can only be rejected by tracking real structure. Exactly two per
    #: language; the remaining three are "easy" (detectable from length/symbol counts alone).
    HARD_STRATEGIES: Tuple[int, ...] = ()

    @property
    @abstractmethod
    def alphabet(self) -> List[str]:
        ...

    @abstractmethod
    def respects_surface_statistics(self, word: str, n: int) -> bool:
        """Does ``word`` satisfy the length and symbol-count statistics shared by all
        positives at this ``n``? Negatives that do are "hard" by construction; negatives
        that do not can be rejected by a counting heuristic and are "easy"."""
        ...

    def difficulty_of(self, strategy: int) -> str:
        return "hard" if strategy % NUM_STRATEGIES in self.HARD_STRATEGIES else "easy"

    @abstractmethod
    def is_member(self, word: str) -> bool:
        ...

    @abstractmethod
    def generate_positive(self, rng: random.Random, n: int) -> str:
        ...

    @abstractmethod
    def generate_negative(
        self,
        rng: random.Random,
        n: int,
        strategy: int | None = None,
        positive: str | None = None,
    ) -> str:
        """Corrupt ``positive`` (the string this negative is paired with) when given.

        Mutating an independently drawn positive instead leaves a length difference between
        the two members of a pair, which a model can exploit without learning anything."""
        ...

    def generate_pair(
        self,
        rng: random.Random,
        n: int,
        strategy: int | None = None,
    ) -> Tuple[str, str]:
        pos = self.generate_positive(rng, n)
        if not self.is_member(pos):
            raise AssertionError(
                f"[{self.name}] Bug: generate_positive returned non-member: {pos!r}"
            )

        for _ in range(100):
            # Always corrupt this pair's own positive, so the two differ structurally and
            # not merely in length. Retries redraw the mutation, never the target length.
            neg = self.generate_negative(rng, n, strategy=strategy, positive=pos)
            if not self.is_member(neg):
                return pos, neg

        raise RuntimeError(
            f"[{self.name}] Could not generate a valid negative for n={n} after 100 tries."
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class LanguageRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, FormalLanguage] = {}

    def register(self, lang: FormalLanguage) -> None:
        key = lang.name.lower()
        if key in self._registry:
            raise ValueError(f"Language already registered: {key!r}")
        self._registry[key] = lang

    def get(self, name: str) -> FormalLanguage:
        key = name.lower()
        if key not in self._registry:
            supported = ", ".join(sorted(self._registry))
            raise ValueError(f"Unknown language {name!r}. Supported: {supported}")
        return self._registry[key]


REGISTRY = LanguageRegistry()


def get_language(name: str) -> FormalLanguage:
    return REGISTRY.get(name)


from . import anbn  # noqa: E402, F401
from . import balanced_parens  # noqa: E402, F401
from . import even_a  # noqa: E402, F401
from . import palindrome  # noqa: E402, F401
from . import reber  # noqa: E402, F401
