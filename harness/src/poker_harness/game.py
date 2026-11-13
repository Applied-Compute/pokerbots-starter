"""Play one table on the hand engine (`engine/`) with no cross-hand history; table talk only with `rules.table_talk`.

The log keeps the Vals game-log shape, so `evaluation.grade`, `metrics.tally` and `rating.rate` work on it unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from .engine import HandRunner, Player

from .prompts import MAX_RAW_TALK_CHARS, TableRules, sanitize_talk

DEFAULT_RULES = TableRules()


def _next_live_seat(after: int, stacks: dict[int, int]) -> int:
    seats = sorted(stacks)
    ordered = [s for s in seats if s > after] + [s for s in seats if s <= after]
    return next(s for s in ordered if stacks[s] > 0)


async def play_session(
    game_id: str,
    seed: int,
    players: Sequence[Player],
    rules: TableRules = DEFAULT_RULES,
    on_hand: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Returns the game log. With `rules.reset_stacks`, every hand starts at `starting_stack` and nobody busts;
    `final_stacks` is then the starting stack plus the session's net chips."""
    names = [p.name for p in players]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate players at one table: {names}")
    by_seat = dict(enumerate(players))
    start = rules.starting_stack
    stacks = dict.fromkeys(by_seat, start)
    net = dict.fromkeys(by_seat, 0)
    no_history = dict.fromkeys(by_seat, "")
    hands: list[dict[str, Any]] = []
    eliminated: dict[str, int] = {}
    start_time = datetime.now(UTC).isoformat()
    button = max(by_seat)
    timestep = 0

    def log(complete: bool) -> dict[str, Any]:
        return {
            "game_id": game_id,
            "hands": hands,
            "session_info": {
                "player_names": names,
                "starting_stacks": [start] * len(names),
                "small_blind": rules.small_blind,
                "big_blind": rules.big_blind,
                "seed": seed,
                "start_time": start_time,
                "end_time": datetime.now(UTC).isoformat(),
                "total_hands": len(hands),
                "reset_stacks": rules.reset_stacks,
                "table_talk": rules.table_talk,
                "history": "current_hand",
            },
            "final_stacks": {by_seat[s].name: start + net[s] for s in by_seat},
            "eliminated_on_hand": dict(eliminated),
            "complete": complete,
        }

    for hand_number in range(1, rules.total_hands + 1):
        if rules.reset_stacks:
            stacks = dict.fromkeys(by_seat, start)
        elif sum(s > 0 for s in stacks.values()) < 2:
            break
        if hand_number > 1:
            button = _next_live_seat(button, stacks)
        runner = HandRunner(
            players=by_seat,
            stacks=stacks,
            button_seat=button,
            hand_number=hand_number,
            total_hands=rules.total_hands,
            seed=seed,
            small_blind=rules.small_blind,
            big_blind=rules.big_blind,
        )
        record, ending = await runner.play(no_history, timestep)
        for action in record["actions"]:
            if not rules.table_talk:  # talk off: nothing a player sends as talk reaches the log
                action["say"] = ""
                action["model_response"].pop("say", None)
            else:  # talk on: the log keeps the sanitized line and, for review, the raw one
                raw = action["say"] if isinstance(action["say"], str) else ""
                action["say"] = sanitize_talk(raw)
                if raw:
                    action["model_response"].setdefault("say_raw", raw[:MAX_RAW_TALK_CHARS])
        timestep += len(record["actions"])
        hands.append(record)
        for seat, player in by_seat.items():
            net[seat] += ending[seat] - stacks[seat]
            if not rules.reset_stacks and stacks[seat] > 0 and ending[seat] == 0:
                eliminated[player.name] = hand_number
        stacks = ending
        if on_hand:
            on_hand(log(complete=False))
    return log(complete=True)
