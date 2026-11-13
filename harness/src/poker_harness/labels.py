"""Evaluator-only labels added to finished game logs; models never see them.

`fold_regret_bb` on a fold is what calling would have been worth, in big blinds, against the hands still live at
that moment: equity x (pot + call) - call, floored at 0. Equity uses the opponents' actual hole cards over sampled
runouts (the Vals `true_equity`), so a fold of a hand that was ahead is costly and a fold of a beaten hand is free.
"""

from __future__ import annotations

from typing import Any

from poker_agent_benchmark_service.evaluation import true_equity


def fold_regret_bb(action: dict[str, Any], hole: list[str], opponents: list[list[str]], big_blind: int) -> float:
    call = action["call_amount"]
    if not opponents or call <= 0:
        return 0.0
    equity = true_equity(hole, opponents, action["board"])
    return round(max(0.0, equity * (action["pot_before"] + call) - call) / big_blind, 3)


def label_folds(log: dict[str, Any], seats: set[str]) -> None:
    """Add `fold_regret_bb` (and the equity inputs) to every fold by a player in `seats`, in place."""
    big_blind = log["session_info"]["big_blind"]
    for hand in log["hands"]:
        starting = hand["starting_state"]
        live = set(starting)
        for action in hand["actions"]:
            name = action["player"]
            if name in seats and action["action"]["type"] == "fold":
                opponents = [starting[o]["hole_cards"] for o in sorted(live) if o != name]
                action["fold_regret_bb"] = fold_regret_bb(action, starting[name]["hole_cards"], opponents, big_blind)
                action["fold_live_opponents"] = len(opponents)
            if action["action"]["type"] == "fold":
                live.discard(name)
