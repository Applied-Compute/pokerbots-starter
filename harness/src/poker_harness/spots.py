"""Which hands give the trained seat a real decision. Deck order depends only on the seed, so a task's hole cards
are known before it is played; trash hands are folded by every sample and teach GRPO nothing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import HandRunner, codes

from .prompts import TableRules

RANKS = "23456789TJQKA"
HIGH_CARD_POINTS = {"A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0, **{r: (RANKS.index(r) + 2) / 2 for r in "23456789T"}}


@dataclass
class _Seat:
    name: str


ONE_HAND = TableRules(total_hands=1)


def hole_cards(seed: int, seat: int, rules: TableRules = ONE_HAND, seats: int = 10) -> list[str]:
    """The seat's hole cards in hand 1 of a table with this seed (button on the last seat), as the harness deals."""
    runner = HandRunner(
        players={i: _Seat(f"seat{i}") for i in range(seats)},
        stacks=dict.fromkeys(range(seats), rules.starting_stack),
        button_seat=seats - 1,
        hand_number=1,
        total_hands=rules.total_hands,
        seed=seed,
        small_blind=rules.small_blind,
        big_blind=rules.big_blind,
    )
    runner._deal()
    return codes(runner.state.hole_cards[runner.order.index(seat)])


def chen_score(cards: list[str]) -> int:
    """Bill Chen's preflop hand score: high card, pair, suitedness and gap, rounded up."""
    (r1, s1), (r2, s2) = sorted(((c[0], c[1]) for c in cards), key=lambda c: RANKS.index(c[0]), reverse=True)
    if r1 == r2:
        return math.ceil(max(5.0, 2 * HIGH_CARD_POINTS[r1]))
    score = HIGH_CARD_POINTS[r1] + (2 if s1 == s2 else 0)
    gap = RANKS.index(r1) - RANKS.index(r2) - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and RANKS.index(r1) < RANKS.index("Q"):
        score += 1
    return math.ceil(score)
