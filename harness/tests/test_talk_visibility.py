"""Every seat sees every other seat's move and table talk since its own last turn, at every talk table: in the hand's
later messages, in the after-hand turn at scratchpad tables, and otherwise at the start of its next hand."""

import json
import random
import re

import pytest

from poker_harness import Completion, ModelSeat
from poker_harness.completions import table_rules, with_preset
from poker_harness.game import play_session

SEATS = 6


def _call(name: str, arguments: dict, i: int) -> dict:
    return {"id": f"{name}-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


async def _play(seed: int, overrides: dict) -> tuple[dict, list[dict]]:
    """A game where every action says a unique tag; returns the log and every model call with what it newly showed."""
    rng = random.Random(seed)
    rules = table_rules(with_preset({"preset": "event", "seed": seed, "total_hands": 8, **overrides}))
    calls: list[dict] = []
    conversation: dict[str, list[dict]] = {}

    def bot(name: str):
        async def complete(messages: list[dict]) -> Completion:
            before = conversation.get(name, [])
            continued = bool(before) and messages[: len(before)] == before
            new = messages[len(before) :] if continued else messages
            conversation[name] = list(messages)
            shown = "\n".join(str(m["content"]) for m in new if m["role"] in ("user", "tool"))
            last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            if "Now call update_scratchpad" in last_user:
                calls.append({"seat": name, "shown": shown})
                return Completion(text="", tool_calls=[_call("update_scratchpad", {"contents": "n"}, len(calls))])
            legal = last_user.split("Legal actions:")[-1]
            raise_ = re.search(r"(bet|raise) to (\d+)-", legal)
            roll = rng.random()
            if roll < 0.2 and "check" not in legal:
                arguments: dict = {"action": "fold"}
            elif roll < 0.45 and raise_:
                arguments = {"action": raise_[1], "amount": int(raise_[2])}
            else:
                arguments = {"action": "check" if "check" in legal else "call"}
            tag = f"tag{len(calls)}x"
            calls.append({"seat": name, "shown": shown, "tag": tag})
            call = _call("submit_action", arguments | {"say": f"Hey {tag}!"}, len(calls))
            return Completion(text="", tool_arguments=call["function"]["arguments"], tool_calls=[call])

        return complete

    players = [ModelSeat(f"seat{i}", bot(f"seat{i}"), rules) for i in range(SEATS)]
    return await play_session(f"g{seed}", seed, players, rules), calls


@pytest.mark.parametrize(
    "overrides",
    [{}, {"scratchpad": False}, {"scratchpad": False, "ranking": None, "reset_stacks": True}],
    ids=["event", "event-no-scratchpad", "talk-reset-stacks"],
)
@pytest.mark.parametrize("seed", [1, 2])
async def test_every_seat_sees_all_moves_and_talk_since_its_last_turn(overrides: dict, seed: int) -> None:
    log, calls = await _play(seed, overrides)
    actions = [a for hand in log["hands"] for a in hand["actions"]]
    assert len(log["hands"]) > 1 and len(actions) == sum("tag" in c for c in calls)
    assert all(a["say"] == f"hey {c['tag']}!" for a, c in zip(actions, (c for c in calls if "tag" in c), strict=True))
    for i, call in enumerate(calls):
        previous = max((j for j in range(i) if calls[j]["seat"] == call["seat"]), default=-1)
        since = [c for c in calls[previous + 1 : i] if "tag" in c and c["seat"] != call["seat"]]
        missing = [
            c["tag"]
            for c in since
            if f"{c['seat']} (seat {c['seat'][4:]}) " not in call["shown"]
            or f'says: "hey {c["tag"]}!"' not in call["shown"]
        ]
        assert not missing, f"{call['seat']} call {i} never saw {missing}"


async def test_single_hand_talk_tables_are_unchanged() -> None:
    """One-hand tasks (the 6max-event datasets) have no next hand, so no recap and no extra log keys."""
    log, _ = await _play(3, {"scratchpad": False, "ranking": None, "reset_stacks": True, "total_hands": 1})
    assert "pot_won" not in log["hands"][0] and "eliminated" not in log["hands"][0]
