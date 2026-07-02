"""Formal language definitions: abstract base + registry."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class FormalLanguage(ABC):
    name: str
    description: str
    chomsky_class: str

    @property
    @abstractmethod
    def alphabet(self) -> List[str]:
        ...

    @abstractmethod
    def is_member(self, word: str) -> bool:
        ...

    @abstractmethod
    def generate_positive(self, rng: random.Random, n: int) -> str:
        ...

    @abstractmethod
    def generate_negative(self, rng: random.Random, n: int, strategy: int | None = None) -> str:
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
            neg = self.generate_negative(rng, n, strategy=strategy)
            if not self.is_member(neg):
                return pos, neg
            n_try = rng.randint(1, max(1, n))
            neg = self.generate_negative(rng, n_try, strategy=strategy)
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

    def list_all(self) -> Dict[str, str]:
        return {k: v.description for k, v in sorted(self._registry.items())}


REGISTRY = LanguageRegistry()


def get_language(name: str) -> FormalLanguage:
    return REGISTRY.get(name)


def list_languages() -> Dict[str, str]:
    return REGISTRY.list_all()


from . import anbn  # noqa: E402, F401
from . import balancesd_parens  # noqa: E402, F401
from . import palindrome  # noqa: E402, F401
