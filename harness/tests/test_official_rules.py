"""The official game rules, one test per rule: carry-over and busting, ranking, afk on invalid moves, and the
scratchpad (persistence, the one retry, cut-offs, single calls, cleanup content and privacy)."""

import itertools
import json
import re
from collections.abc import Callable
from typing import Any

from poker_harness.engine.table import DecisionContext, LegalActions, SeatView

from poker_harness.completions import table_rules, with_preset
from poker_harness.game import placements, play_session
from poker_harness.prompts import TableRules
from poker_harness.seat import TOO_EARLY, Completion, ModelSeat

OFFICIAL = TableRules(
    total_hands=30,
    reset_stacks=False,
    seats=3,
    ranking="elimination",
    strict_actions=True,
    max_completion_tokens=5000,
)
SCRATCH = TableRules(**{**OFFICIAL.__dict__, "scratchpad": True, "cleanup_max_completion_tokens": 10000})
CANARY = "CANARY-7f3a"
SUBMIT = "submit_action"
BAD = "BAD"  # stands for malformed update_scratchpad arguments


def tool_call(name: str, arguments: Any, i: int = 0) -> dict:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": f"{name}-{i}", "type": "function", "function": {"name": name, "arguments": text}}


def submit(action: str, amount: int | None = None, text: str = "thinking") -> Completion:
    call = tool_call(SUBMIT, {"action": action} | ({"amount": amount} if amount is not None else {}))
    return Completion(text=text, tool_arguments=call["function"]["arguments"], tool_calls=[call])


def update(*contents: Any, text: str = "noting", truncated: bool = False) -> Completion:
    """update_scratchpad calls with these contents; `BAD` sends unparseable arguments."""
    calls = [
        tool_call("update_scratchpad", "{oops" if c == BAD else {"contents": c}, i) for i, c in enumerate(contents)
    ]
    return Completion(text=text, tool_calls=calls, truncated=truncated)


def shove(messages: list[dict]) -> Completion:
    """All in whenever a bet or raise is legal, else call (a check when free)."""
    legal = re.search(r"(bet|raise) to \d+-(\d+)", last_user(messages).split("Legal actions:")[-1])
    return submit(legal[1], int(legal[2])) if legal else submit("call")


def last_user(messages: list[dict]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def bot(
    act: Callable[[list[dict]], Completion] = lambda m: submit("call"),
    clean: Callable[[list[dict], bool], Completion] | None = None,
):
    """A scripted model: `act` on action turns, `clean(messages, retry)` on cleanup turns (default: no update)."""

    async def complete(messages: list[dict]) -> Completion:
        if "is over" in last_user(messages):
            return (clean or (lambda m, r: Completion(text="nothing to note")))(
                messages, messages[-1]["role"] == "tool"
            )
        return act(messages)

    return complete


def seats(n: int, rules: TableRules, **policies) -> list[ModelSeat]:
    return [ModelSeat(f"seat{i}", policies.get(f"seat{i}", bot()), rules) for i in range(n)]


# Carry-over, busting and ranking


async def test_chips_carry_over_and_busted_seats_sit_out_until_one_player_has_everything() -> None:
    rules = TableRules(**{**OFFICIAL.__dict__, "starting_stack": 6})  # 3 big blinds each: calling stations bust fast
    players = [ModelSeat(f"seat{i}", bot(shove), rules) for i in range(3)]
    log = await play_session("g", 7, players, rules)
    hands, final = log["hands"], log["final_stacks"]
    assert sum(final.values()) == 18 and log["complete"]
    assert len(hands) < 30 and max(final.values()) == 18  # it ended because one player had every chip
    for before, after in itertools.pairwise(hands):
        assert after["stacks_before"] == {n: s for n, s in before["ending_stacks"].items() if s > 0}
        assert set(after["starting_state"]) == set(after["stacks_before"])  # busted seats are not dealt in
    busted = log["eliminated_on_hand"]
    assert len(busted) == 2 and all(name in hands[h - 1]["eliminated"] for name, h in busted.items())
    places = log["ranking"]["places"]
    winner = max(final, key=final.get)
    assert places[winner] == 1 and sorted(places.values()) in ([1, 2, 3], [1, 2, 2])
    later, earlier = sorted(busted, key=busted.get, reverse=True)
    assert places[later] <= places[earlier] and log["net_chips"] == {n: s - 6 for n, s in final.items()}
    for hand in hands:  # blinds 1/2 all game; chips are conserved and every pot is won
        assert hand["blinds"] == {"small_blind": 1, "big_blind": 2}
        assert sum(hand["stacks_before"].values()) == 18 == sum(hand["ending_stacks"].values())
        put_in = sum(a["put_in"] for a in hand["actions"]) + sum(hand["blinds_posted"].values())
        assert put_in >= sum(hand["pot_won"].values()) > 0  # an uncalled part of a shove goes back


async def test_a_short_stack_posts_its_blind_all_in_for_less() -> None:
    rules = TableRules(**{**OFFICIAL.__dict__, "starting_stack": 1, "total_hands": 1})
    log = await play_session("g", 3, seats(3, rules), rules)
    hand = log["hands"][0]
    assert hand["blinds_posted"] == {"seat0": 1, "seat1": 1} and hand["stacks_before"]["seat1"] == 1


def test_placements_rank_survivors_by_chips_then_busted_seats_in_reverse_elimination_order() -> None:
    log = {
        "session_info": {"player_names": ["a", "b", "c", "d", "e"]},
        "final_stacks": {"a": 300, "b": 300, "c": 0, "d": 0, "e": 0},
        "eliminated_on_hand": {"c": 10, "d": 10, "e": 3},
        "hands": [
            {"hand_number": 3, "stacks_before": {"e": 40}},
            {"hand_number": 10, "stacks_before": {"c": 50, "d": 20}},
        ],
    }
    assert placements(log) == {"a": 1, "b": 1, "c": 3, "d": 4, "e": 5}
    assert placements(log, "chips") == {"a": 1, "b": 1, "c": 3, "d": 3, "e": 3}


# Afk: invalid moves and the token cap


def context(legal: LegalActions, hand_actions: list | None = None) -> DecisionContext:
    views = [SeatView(i, f"seat{i}", 200, 0, "") for i in range(3)]
    return DecisionContext(1, 30, "preflop", 0, 2, 0, ["Ah", "Kd"], [], 3, views, hand_actions or [], legal)


FACING_BET = LegalActions(
    can_fold=True, can_check=False, call_amount=2, raise_type="raise", min_raise_to=4, max_raise_to=200
)
FREE = LegalActions(can_fold=False, can_check=True, call_amount=0, raise_type="bet", min_raise_to=2, max_raise_to=200)


async def afk_of(reply: Completion, legal: LegalActions, rules: TableRules = OFFICIAL) -> tuple[str, dict]:
    async def complete(messages):
        return reply

    decision = await ModelSeat("seat0", complete, rules).decide(context(legal), "")
    return decision.type, decision.model_response


async def test_invalid_moves_and_cut_replies_are_played_check_else_fold() -> None:
    cases = [
        (submit("check"), FACING_BET, "fold", "check is not legal facing a bet"),
        (submit("raise", 500), FACING_BET, "fold", "raise to 500 is outside 4-200"),
        (submit("raise"), FACING_BET, "fold", "raise without an amount"),
        (submit("bet", 1), FREE, "check", "bet to 1 is outside 2-200"),
        (Completion(text="no tool call"), FREE, "check", None),
        (
            Completion(
                text="", tool_calls=submit("call").tool_calls, tool_arguments='{"action":"call"}', truncated=True
            ),
            FACING_BET,
            "fold",
            "reply hit the 5000-token cap",
        ),
    ]
    for reply, legal, applied, reason in cases:
        kind, response = await afk_of(reply, legal)
        assert kind == applied and response["afk"] and response["format_error"], (reply, response)
        assert reason is None or response["afk_reason"] == reason
    kind, response = await afk_of(submit("raise", 500), FACING_BET, TableRules())  # research tables: clamped later
    assert kind == "raise" and not response["format_error"] and "afk" not in response


async def test_update_scratchpad_during_a_hand_is_an_invalid_move_and_its_call_is_rejected() -> None:
    replies = iter([update("too early"), submit("call")])
    seen: list[list[dict]] = []

    async def complete(messages):
        seen.append(list(messages))
        return next(replies)

    seat = ModelSeat("seat0", complete, SCRATCH)
    decision = await seat.decide(context(FACING_BET), "")
    assert (
        decision.type == "fold" and decision.model_response["afk_reason"] == "update_scratchpad called during the hand"
    )
    own = {
        "player": "seat0",
        "seat": 0,
        "street": "preflop",
        "action": {"type": "fold"},
        "call_amount": 2,
        "model_response": decision.model_response,
    }
    await seat.decide(context(FACING_BET, [own]), "")
    assert {"role": "tool", "tool_call_id": "update_scratchpad-0", "content": TOO_EARLY} in seen[1]


# The scratchpad


async def test_the_scratchpad_persists_across_hands_and_opens_every_hand() -> None:
    blocks: dict[str, list[str]] = {}

    def player(name: str):
        def act(messages):
            blocks.setdefault(name, []).append(messages[1]["content"])
            return submit("call")

        def clean(messages, retry):
            hand = re.search(r"## Hand (\d+) is over", last_user(messages))[1]
            return update(f"{name} after hand {hand}")

        return bot(act, clean)

    rules = TableRules(**{**SCRATCH.__dict__, "total_hands": 3})
    players = [ModelSeat(f"seat{i}", player(f"seat{i}"), rules) for i in range(3)]
    log = await play_session("g", 11, players, rules)
    assert all(p.scratchpad == f"{p.name} after hand 3" for p in players)
    for name, seen in blocks.items():
        assert "(empty)" in seen[0] and any(f"{name} after hand 1" in b for b in seen[1:])
    private = log["private"]["seat0"]["hands"]
    assert (
        private["2"]["scratchpad_before"] == "seat0 after hand 1"
        and private["2"]["scratchpad_after"] == "seat0 after hand 2"
    )


SUMMARY = {
    "hand_number": 1,
    "actions": [],
    "final_board": [],
    "showdown": False,
    "shown_hole_cards": {},
    "starting_state": {"seat0": {"seat": 0, "hole_cards": ["Ah", "Kd"]}},
    "pot_won": {"seat1": 3},
    "stacks": {"seat0": 199, "seat1": 201},
    "eliminated": [],
}


async def cleanup_with(*replies: Completion, old: str = "old notes") -> tuple[ModelSeat, dict, list[list[dict]]]:
    pending, seen = list(replies), []

    async def clean(messages):
        seen.append(list(messages))
        return pending.pop(0)

    async def never(messages):
        raise AssertionError("action turns use complete; cleanup has its own caller")

    seat = ModelSeat("seat0", never, SCRATCH, cleanup_complete=clean)
    seat.scratchpad = old
    record = await seat.cleanup(SUMMARY)
    return seat, record, seen


async def test_a_too_long_or_malformed_update_gets_one_more_try_then_is_cut_or_dropped() -> None:
    long, longer = "x" * 6000, "y" * 7000
    seat, record, seen = await cleanup_with(update(long), update("fixed"))
    assert seat.scratchpad == "fixed" and record["result"] == "updated" and len(seen) == 2
    rejection = seen[1][-1]
    assert rejection["role"] == "tool" and "has 6000 characters; the limit is 5000" in rejection["content"]
    seat, record, _ = await cleanup_with(update(long), update(longer))
    assert seat.scratchpad == "y" * 5000 and record["result"] == "cut"
    seat, _, _ = await cleanup_with(update(long), update("BAD"))  # the retry is malformed: the first update, cut
    assert seat.scratchpad == "x" * 5000
    seat, record, seen = await cleanup_with(update("BAD"), update("BAD"))
    assert seat.scratchpad == "old notes" and record["result"] == "unchanged" and "malformed" in seen[1][-1]["content"]
    seat, _, _ = await cleanup_with(update("BAD"), update("good"))
    assert seat.scratchpad == "good"
    seat, record, _ = await cleanup_with(update({"not": "a string"}), Completion(text="gave up"))
    assert seat.scratchpad == "old notes" and len(record["attempts"]) == 2


async def test_no_call_a_cut_reply_or_extra_calls_get_no_retry() -> None:
    seat, record, seen = await cleanup_with(Completion(text="I'll keep it"))
    assert seat.scratchpad == "old notes" and record["error"] == "no update_scratchpad call" and len(seen) == 1
    seat, record, seen = await cleanup_with(update("new", truncated=True))
    assert seat.scratchpad == "old notes" and record["error"] == "reply hit the 10000-token cap" and len(seen) == 1
    both = update("first", "second")
    both.tool_calls.append(submit("raise", 10).tool_calls[0])
    seat, record, seen = await cleanup_with(both)
    assert seat.scratchpad == "first" and record["attempts"][0]["ignored_calls"] == 2 and len(seen) == 1


async def test_cleanup_reports_the_rest_of_the_hand_to_an_early_folder_and_the_ending_to_a_seat_at_showdown() -> None:
    rules = TableRules(**{**SCRATCH.__dict__, "total_hands": 1})
    players = seats(3, rules, seat2=bot(lambda m: submit("fold")))  # the button acts first three-handed, and folds
    log = await play_session("g", 5, players, rules)
    hand = log["hands"][0]
    prompts = {name: log["private"][name]["hands"]["1"]["cleanup"]["prompt"] for name in ("seat0", "seat1", "seat2")}
    assert hand["showdown"] and hand["actions"][0]["player"] == "seat2"
    early, late = prompts["seat2"], prompts["seat1"]
    assert "Since your last action:" in early and all(
        s in early for s in ("preflop: seat0", "flop:", "turn:", "river:")
    )
    assert "Since your last action:\n  (no other actions)" in late  # the big blind acts last on the river
    assert "river: seat1" in prompts["seat0"] and "flop:" not in prompts["seat0"].split("Since your last action:")[1]
    for name, cards in hand["shown_hole_cards"].items():
        assert all(f"{name} showed {' '.join(cards)}" in p for p in prompts.values())
    hidden = [n for n in hand["starting_state"] if n not in hand["shown_hole_cards"]]
    for name in hidden:  # folded and mucked hands stay hidden from everyone
        cards = " ".join(hand["starting_state"][name]["hole_cards"])
        assert all(cards not in p for p in prompts.values())
    for p in prompts.values():
        assert "Pot: " in p and "Stacks now: seat0" in p and "call update_scratchpad exactly once" in p


async def test_a_seat_that_never_acted_gets_its_cards_and_the_whole_hand_in_cleanup() -> None:
    rules = TableRules(**{**SCRATCH.__dict__, "total_hands": 1})
    fold = bot(lambda m: submit("fold"))
    log = await play_session("g", 5, seats(3, rules, seat2=fold, seat0=fold), rules)  # the big blind gets a walk
    prompt = log["private"]["seat1"]["hands"]["1"]["cleanup"]["prompt"]
    cards = " ".join(log["hands"][0]["starting_state"]["seat1"]["hole_cards"])
    assert "You did not act this hand" in prompt and cards in prompt and "preflop: seat2 (seat 2) fold" in prompt


async def test_private_content_appears_only_in_the_private_sections() -> None:
    def clean(messages, retry):
        return update(f"{CANARY} pad", text=f"cleanup reasoning {CANARY}")

    rules = TableRules(**{**SCRATCH.__dict__, "total_hands": 2, "table_talk": True})
    act = bot(lambda m: submit("call", text=f"decision reasoning {CANARY}"), clean)
    players = [ModelSeat(f"seat{i}", act, rules) for i in range(3)]
    log = await play_session("g", 9, players, rules)
    assert CANARY in json.dumps(log)
    public = json.loads(json.dumps(log))
    del public["private"]
    for hand in public["hands"]:
        for action in hand["actions"]:
            assert CANARY in json.dumps(action["model_response"].pop("private"))
    assert CANARY not in json.dumps(public)
    for player in players:  # the scratchpad block and the cleanup turn are withheld from the transcript
        assert CANARY in json.dumps(player.messages) and "SCRATCHPAD" not in json.dumps(player.public_messages())
        assert "is over" not in json.dumps(player.public_messages())


async def test_an_event_game_stops_at_its_30_hand_cap() -> None:
    rules = table_rules(with_preset({"preset": "event", "seed": 1, "seats": 3, "scratchpad": False}))
    log = await play_session("g", 13, seats(3, rules), rules)  # calling stations at 100 BB: nobody busts in 30 hands
    assert rules.total_hands == 30 and len(log["hands"]) == 30 and not log["eliminated_on_hand"]
    assert min(log["ranking"]["places"].values()) == 1 and sum(log["net_chips"].values()) == 0
