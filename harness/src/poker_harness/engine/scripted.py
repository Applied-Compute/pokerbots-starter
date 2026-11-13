"""Scripted seats: a seeded random player and fixed archetypes, for tests and local games without a model.

Ported from Vals' poker-agent engine (poker-agent-benchmark-service 0.1.0) and used with Vals' permission; behavior is
unchanged, and tests/test_engine_parity.py checks it hand for hand.
"""

from __future__ import annotations

import random

from .table import Decision, DecisionContext

ARCHETYPES = ("random", "maniac", "nit", "calling_station")


def _strong(hole_cards: list[str]) -> bool:
    """Pairs of tens or better, and AK/AQ."""
    ranks = {card[0] for card in hole_cards}
    return (len(ranks) == 1 and ranks <= set("AKQJT")) or ranks in ({"A", "K"}, {"A", "Q"})


class ScriptedPlayer:
    """`random` mixes legal actions; `maniac` raises big; `nit` plays only strong hands; `calling_station` calls."""

    def __init__(self, name: str, archetype: str, seed: int = 0) -> None:
        if archetype not in ARCHETYPES:
            raise ValueError(f"unknown archetype {archetype!r}")
        self.name = name
        self.archetype = archetype
        self.rng = random.Random(f"{name}:{seed}")

    def _choose(self, ctx: DecisionContext) -> Decision:
        legal = ctx.legal
        passive = Decision("check" if legal.can_check else "call")
        if self.archetype == "calling_station":
            return passive
        if self.archetype == "maniac":
            if legal.raise_type and self.rng.random() < 0.7:
                return Decision(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, ctx.pot * 2)))
            return passive
        if self.archetype == "nit":
            if not _strong(ctx.hole_cards):
                return Decision("check" if legal.can_check else "fold")
            if legal.raise_type and ctx.street == "preflop":
                return Decision(legal.raise_type, legal.min_raise_to)
            return passive
        roll = self.rng.random()
        if legal.raise_type and roll < 0.15:
            top = min(legal.max_raise_to, legal.min_raise_to * 3)
            return Decision(legal.raise_type, self.rng.randint(legal.min_raise_to, top))
        if legal.can_fold and roll > 0.6:
            return Decision("fold")
        return passive

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        decision = self._choose(ctx)
        decision.model_response = {
            "model_name": self.name,
            "provider": "scripted",
            "latency_seconds": 0.0,
            "in_tokens": 0,
            "out_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "reasoning": "",
            "fallback_used": False,
            "error": None,
        }
        return decision
