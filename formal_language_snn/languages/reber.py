"""Language: Reber grammar (regular / finite-state automaton)."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from . import FormalLanguage, REGISTRY

# Classic 6-state Reber FSA after the initial B.
TRANSITIONS: Dict[int, Dict[str, Optional[int]]] = {
    1: {"T": 2, "P": 3},
    2: {"S": 2, "X": 4},
    3: {"T": 3, "V": 5},
    4: {"X": 3, "S": 6},
    5: {"P": 4, "V": 6},
    6: {"E": None},
}

# Minimum symbols still required to reach and include terminal E from each state.
_MIN_TO_E: Dict[int, int] = {6: 1, 5: 2, 4: 2, 3: 2, 2: 3, 1: 4}

ALPHABET = ["B", "E", "P", "S", "T", "V", "X"]


def _walk_states(word: str) -> List[Optional[int]]:
    """Return FSA state after each prefix (state after word[:i] for i in 1..len)."""
    states: List[Optional[int]] = []
    if not word or word[0] != "B":
        return states

    state = 1
    for ch in word[1:]:
        if state is None:
            states.append(None)
            continue
        nxt = TRANSITIONS.get(state, {}).get(ch)
        state = nxt
        states.append(state)
    return states


def generate_reber_string(
    rng: random.Random,
    min_length: int,
    max_length: int,
    *,
    max_attempts: int = 500,
) -> str:
    """Random walk from B until E; retry until len(word) in [min_length, max_length]."""
    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid length bounds")

    for _ in range(max_attempts):
        result = ["B"]
        state = 1
        while state is not None:
            choices = list(TRANSITIONS[state].items())
            remaining_min = _MIN_TO_E[state]
            cur_len = len(result)
            if cur_len + remaining_min > max_length:
                choices = [(c, s) for c, s in choices if s != state or remaining_min == 1]
                if not choices:
                    choices = list(TRANSITIONS[state].items())
            elif cur_len + remaining_min < min_length:
                loops = [(c, s) for c, s in choices if s == state]
                if loops:
                    choices = loops
            ch, nxt = rng.choice(choices)
            result.append(ch)
            state = nxt
        word = "".join(result)
        if min_length <= len(word) <= max_length:
            return word

    raise RuntimeError(
        f"Could not generate Reber string in [{min_length}, {max_length}] "
        f"after {max_attempts} attempts"
    )


def _legal_next(state: Optional[int]) -> set[str]:
    if state is None:
        return set()
    return set(TRANSITIONS.get(state, {}))


def _mutate_transition(
    rng: random.Random, word: str, *, middle_third: bool = False
) -> str:
    pos_word = word
    length = len(pos_word)
    if length < 2:
        return pos_word + "T"

    if middle_third and length >= 3:
        lo = length // 3
        hi = max(lo + 1, (2 * length) // 3)
        idx = rng.randrange(lo, hi)
    else:
        idx = rng.randrange(1, length)

    state_before = 1 if idx == 1 else _walk_states(pos_word)[idx - 2]
    legal = _legal_next(state_before)
    illegal = [c for c in ALPHABET if c not in legal and c != pos_word[idx]]
    if not illegal:
        illegal = [c for c in ALPHABET if c != pos_word[idx]]
    replacement = rng.choice(illegal)

    chars = list(pos_word)
    chars[idx] = replacement
    return "".join(chars)


class ReberGrammar(FormalLanguage):
    name = "reber"
    description = "Reber grammar – strings from a finite-state transition graph (Regular)"
    chomsky_class = "Regular"
    # 0, 1 = one illegal transition substituted inside an otherwise well-formed walk:
    # the string still starts with B, ends with E and has a plausible length.
    HARD_STRATEGIES = (0, 1)

    @property
    def alphabet(self) -> List[str]:
        return list(ALPHABET)

    def is_member(self, word: str) -> bool:
        if not word or word[0] != "B" or word[-1] != "E":
            return False
        state = 1
        for ch in word[1:]:
            if state is None:
                return False
            state = TRANSITIONS.get(state, {}).get(ch)
        return state is None

    def respects_surface_statistics(self, word: str, n: int) -> bool:
        n = max(1, n)
        if len(word) < 2 or word[0] != "B" or word[-1] != "E":
            return False
        if any(ch not in ALPHABET for ch in word):
            return False
        return max(5, 2 * n - 2) <= len(word) <= 2 * n + 2

    def generate_positive(self, rng: random.Random, n: int) -> str:
        n = max(1, n)
        min_length = max(5, 2 * n - 2)
        max_length = max(min_length, 2 * n + 2)
        return generate_reber_string(rng, min_length, max_length)

    def generate_negative(
        self,
        rng: random.Random,
        n: int,
        strategy: int | None = None,
        positive: str | None = None,
    ) -> str:
        n = max(1, n)
        strategy = rng.randrange(5) if strategy is None else strategy % 5
        # Reber walks vary in length within [2n-2, 2n+2], so drawing a fresh positive here
        # left the negative a different length from its pair four times in five.
        pos = positive if positive is not None else self.generate_positive(rng, n)

        if strategy == 0:
            return _mutate_transition(rng, pos, middle_third=False)

        if strategy == 1:
            return _mutate_transition(rng, pos, middle_third=True)

        if strategy == 2:
            length = len(pos)
            return "".join(rng.choice(self.alphabet) for _ in range(length))

        if strategy == 3:
            chars = list(pos)
            non_e = [c for c in self.alphabet if c != "E"]
            chars[-1] = rng.choice(non_e)
            return "".join(chars)

        # strategy 4: fail start/end guard
        if rng.random() < 0.5:
            chars = list(pos)
            chars[0] = rng.choice([c for c in self.alphabet if c != "B"])
            return "".join(chars)
        chars = list(pos)
        chars[-1] = rng.choice([c for c in self.alphabet if c != "E"])
        return "".join(chars)


if __name__ == "__main__":
    lang = ReberGrammar()
    rng = random.Random(0)
    for n in (1, 5, 10, 20):
        min_len = max(5, 2 * n - 2)
        max_len = max(min_len, 2 * n + 2)
        for _ in range(25):
            pos = lang.generate_positive(rng, n)
            assert lang.is_member(pos), pos
            assert min_len <= len(pos) <= max_len, (n, len(pos), pos)
            for s in range(5):
                neg = lang.generate_negative(rng, n, strategy=s)
                assert not lang.is_member(neg), (s, neg)
    print("reber self-check ok")
else:
    REGISTRY.register(ReberGrammar())
