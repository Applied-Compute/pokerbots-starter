"""Play one table on the hand engine (`engine/`) with no cross-hand history; table talk only with `rules.table_talk`.

The log keeps the Vals game-log shape, so `evaluation.grade`, `metrics.tally` and `rating.rate` work on it unchanged.

Official-rules tables (`rules.ranking` set; see the `TableRules` docstring) also log, per hand: `blinds`,
`stacks_before` (each dealt seat's stack before the blinds; the engine's `starting_state[...].starting_stack` is off
when an all-in blind settles first), `blinds_posted`, `pot_won` and `eliminated`; per action: `put_in` (chips that
action added; the engine's `chip_change` on the action that ends a hand includes the award); and per game:
`net_chips` and `ranking` (`placements`). Research tables (`ranking` None) log exactly as before.

At scratchpad tables every seat dealt into a hand gets a cleanup turn after it, and everything private sits in
`log["private"][player]` (scratchpads, cleanup prompts and replies, update_scratchpad arguments) and in each action's
`model_response["private"]` (the model's own text). Nothing private is anywhere else, so a runner that publishes logs
strips those two keys.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pokerkit import (
    BetCollection,
    BlindOrStraddlePosting,
    CheckingOrCalling,
    ChipsPushing,
    CompletionBettingOrRaisingTo,
    Folding,
)

from .engine import HandRunner, Player
from .prompts import MAX_RAW_TALK_CHARS, TableRules, sanitize_talk

logger = logging.getLogger("poker_harness.game")

DEFAULT_RULES = TableRules()


def _next_live_seat(after: int, stacks: dict[int, int]) -> int:
    seats = sorted(stacks)
    ordered = [s for s in seats if s > after] + [s for s in seats if s <= after]
    return next(s for s in ordered if stacks[s] > 0)


def _chip_flow(runner: HandRunner, record: dict[str, Any]) -> None:
    """Add `blinds_posted`, `pot_won` and each action's `put_in` from the engine's PokerKit operations."""
    names = [runner.players[seat].name for seat in runner.order]
    bets = [0] * len(names)
    posted: dict[str, int] = {}
    won: dict[str, int] = {}
    put_in: list[int] = []
    for op in runner.state.operations:
        if isinstance(op, BlindOrStraddlePosting):
            posted[names[op.player_index]] = posted.get(names[op.player_index], 0) + op.amount
            bets[op.player_index] += op.amount
        elif isinstance(op, CheckingOrCalling):
            put_in.append(op.amount)
            bets[op.player_index] += op.amount
        elif isinstance(op, CompletionBettingOrRaisingTo):
            put_in.append(op.amount - bets[op.player_index])
            bets[op.player_index] = op.amount
        elif isinstance(op, Folding):
            put_in.append(0)
        elif isinstance(op, BetCollection):
            bets = [0] * len(names)
        elif isinstance(op, ChipsPushing):
            for i, amount in enumerate(op.amounts):
                if amount:
                    won[names[i]] = won.get(names[i], 0) + amount
    record["blinds_posted"], record["pot_won"] = posted, won
    if len(put_in) == len(record["actions"]):
        for action, chips in zip(record["actions"], put_in, strict=True):
            action["put_in"] = chips
    else:  # never expected: one PokerKit operation per engine action
        logger.warning(
            "hand %s: %d actions but %d action operations", record["hand_number"], len(record["actions"]), len(put_in)
        )


def placements(log: dict[str, Any], mode: str = "elimination") -> dict[str, int]:
    """Each player's place, 1 = best; tied players share a place (1, 1, 3).

    "elimination": players still in at the end rank above everyone who busted, by chips; busted players rank by the
    hand they busted on (later is better), and players who bust on the same hand by their stack before it.
    "chips": by final chips only."""
    names = log["session_info"]["player_names"]
    chips = log["final_stacks"]
    busted = log.get("eliminated_on_hand", {})
    hands = {h["hand_number"]: h for h in log["hands"]}

    def key(name: str) -> tuple:
        if mode == "chips":
            return (chips[name],)
        if mode != "elimination":
            raise ValueError(f"unknown ranking mode {mode!r}; use 'elimination' or 'chips'")
        if name not in busted:
            return (1, 0, chips[name])
        hand = hands.get(busted[name], {})
        before = (hand.get("stacks_before") or {}).get(name)
        if before is None:  # logs from before `stacks_before`
            before = hand.get("starting_state", {}).get(name, {}).get("starting_stack", 0)
        return (0, busted[name], before)

    ordered = sorted(names, key=key, reverse=True)
    places: dict[str, int] = {}
    for i, name in enumerate(ordered):
        tied = i and key(name) == key(ordered[i - 1])
        places[name] = places[ordered[i - 1]] if tied else i + 1
    return places


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
    official = rules.ranking is not None
    # Multi-hand talk tables without the after-hand turn: each seat's next hand opens with what it missed of the last.
    recap = rules.table_talk and not rules.scratchpad and rules.total_hands > 1
    timestep = 0

    def log(complete: bool) -> dict[str, Any]:
        out = {
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
        if official:
            out["net_chips"] = {by_seat[s].name: net[s] for s in by_seat}
            out["ranking"] = {"mode": rules.ranking, "places": placements(out, rules.ranking)}
        if rules.scratchpad:  # everything a published log must not contain; see the module docstring
            out["private"] = {
                p.name: {
                    "scratchpad": getattr(p, "scratchpad", ""),
                    "hands": {str(n): v for n, v in sorted(getattr(p, "private", {}).items())},
                }
                for p in players
            }
        return out

    for hand_number in range(1, rules.total_hands + 1):
        if rules.reset_stacks:
            stacks = dict.fromkeys(by_seat, start)
        elif sum(s > 0 for s in stacks.values()) < 2:
            break
        if hand_number > 1:
            button = _next_live_seat(button, stacks)
        small_blind, big_blind = rules.small_blind, rules.big_blind
        stacks_before = {by_seat[s].name: stacks[s] for s in sorted(by_seat) if stacks[s] > 0}
        info = {"hand_number": hand_number, "button_seat": button, "stacks": stacks_before}
        for player in players:
            if hasattr(player, "begin_hand"):
                player.begin_hand(info)
        runner = HandRunner(
            players=by_seat,
            stacks=stacks,
            button_seat=button,
            hand_number=hand_number,
            total_hands=rules.total_hands,
            seed=seed,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        record, ending = await runner.play(no_history, timestep)
        if official or rules.scratchpad or recap:
            record["blinds"] = {"small_blind": small_blind, "big_blind": big_blind}
            record["stacks_before"] = stacks_before
            _chip_flow(runner, record)
        for action in record["actions"]:
            if not rules.table_talk:  # talk off: nothing a player sends as talk reaches the log
                action["say"] = ""
                action["model_response"].pop("say", None)
            else:  # talk on: the log keeps the sanitized line and, for review, the raw one
                raw = action["say"] if isinstance(action["say"], str) else ""
                action["say"] = sanitize_talk(raw)
                response = action["model_response"]
                target = response.setdefault("private", {}) if rules.scratchpad else response
                if raw and "say_raw" not in response:
                    target.setdefault("say_raw", raw[:MAX_RAW_TALK_CHARS])
        timestep += len(record["actions"])
        hands.append(record)
        busted = []
        for seat, player in by_seat.items():
            net[seat] += ending[seat] - stacks[seat]
            if not rules.reset_stacks and stacks[seat] > 0 and ending[seat] == 0:
                eliminated[player.name] = hand_number
                busted.append(player.name)
        if official or rules.scratchpad or recap:
            record["eliminated"] = busted
        if rules.scratchpad or recap:
            summary = {
                **{k: record[k] for k in ("hand_number", "actions", "final_board", "showdown", "shown_hole_cards")},
                "starting_state": record["starting_state"],
                "pot_won": record["pot_won"],
                "stacks": {by_seat[s].name: ending[s] for s in sorted(by_seat)},
                "eliminated": busted,
            }
        if recap:  # no model call: the seat keeps the rest of the hand for its next first message
            for seat in runner.order:
                if hasattr(by_seat[seat], "hand_over"):
                    by_seat[seat].hand_over(summary)
        if rules.scratchpad:  # every seat dealt in gets the after-hand turn, concurrently
            dealt = [by_seat[s] for s in runner.order if hasattr(by_seat[s], "cleanup")]
            results = await asyncio.gather(*(p.cleanup(summary) for p in dealt), return_exceptions=True)
            for player, result in zip(dealt, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("cleanup for %s failed: %s", player.name, result)
        stacks = ending
        if on_hand:
            on_hand(log(complete=False))
    return log(complete=True)
