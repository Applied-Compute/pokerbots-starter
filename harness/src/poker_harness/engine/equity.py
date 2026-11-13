"""True equity: the share of the pot a hand wins against the opponents' actual hole cards, over sampled runouts.

Ported from Vals' poker-agent engine (poker-agent-benchmark-service 0.1.0) and used with Vals' permission; behavior is
unchanged, and tests/test_engine_parity.py checks it hand for hand.
"""

from __future__ import annotations

import random

from pokerkit import Card, Deck, StandardHighHand, calculate_equities

EQUITY_SAMPLES = 300


def _cards(codes: list[str]) -> list[Card]:
    return list(Card.parse("".join(codes)))


def true_equity(hole: list[str], opponents: list[list[str]], board: list[str]) -> float:
    """Share of the pot this hand wins against the opponents' actual hole cards over random board runouts."""
    if not opponents:
        return 1.0
    random.seed(repr((hole, opponents, board)))  # pokerkit samples runouts from the global RNG; seeded so grades repeat
    equities = calculate_equities(
        [[_cards(hole)], *[[_cards(h)] for h in opponents]],
        _cards(board),
        2,
        5,
        Deck.STANDARD,
        (StandardHighHand,),
        sample_count=EQUITY_SAMPLES,
    )
    return round(equities[0], 4)
