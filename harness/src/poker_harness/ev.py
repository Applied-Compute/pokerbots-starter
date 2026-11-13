"""Expected-value labels for finished hands; models never see them.

Use `redraw=False` (the default): opponents keep the hands they were dealt, and only an all-in's runout is averaged.
That cuts the standard deviation of all-in hands by about 40% without moving the mean. Redrawing opponents with the heuristic
ranges below cuts more variance but biases the mean upward when the ranges are looser than the opponents really play;
calibrate them against your opponents' showdown hands before using it.

Realized chips over a whole hand are dominated by card luck: who held what at showdown and how an all-in board ran
out. `label_ev` replaces the trained seat's result in hands that reach a showdown with its expected pot share:

- **Showdown, betting finished on the river:** the board is fixed. Each live opponent's hole cards are redrawn
  `variants` times from hands consistent with its actions (`consistent`), and the hero's share is averaged.
- **All-in before the river:** the same redraws, and each one also averages over `runouts` completions of the board
  from the cards unseen when the last chip went in (the all-in EV of `poker_spots`, which also redraws opponents).
- **No showdown:** someone folded out; the result involves no card luck and stays as dealt.

The consistency model is a heuristic for LLM opponents, whose policy cannot be replayed to test a
candidate hand the way `poker_spots` tests its deterministic bots. Preflop it maps each opponent's strongest action
to a minimum Chen score; postflop, a bet or raise needs a pair using a hole card or a four-flush, and a call also
accepts two overcards. An opponent with no consistent hand within `MAX_TRIES` keeps the hand it was dealt.

Stacks reset to the same size every hand, so everyone who reaches showdown has put in the same amount and there are
no side pots; hands where that does not hold keep the realized result and say so.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from pokerkit import Card, StandardHighHand

from .spots import chen_score

RANKS = "23456789TJQKA"
DECK = [r + s for r in RANKS for s in "cdhs"]
BOARD_AT = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
MAX_TRIES = 300
BIG_BET_CHIPS = 60  # a preflop raise or call of at least this many chips (30 BB) means a very strong range


def _preflop_min_chen(actions: list[dict[str, Any]]) -> int:
    pre = [a for a in actions if a["street"] == "preflop"]
    kinds = {a["action"]["type"] for a in pre}
    big = any(a["action"].get("amount", 0) >= BIG_BET_CHIPS or -a["chip_change"] >= BIG_BET_CHIPS for a in pre)
    if big and kinds & {"raise", "call", "bet"}:
        return 8
    if "raise" in kinds:
        return 6
    if "call" in kinds and any(a["call_amount"] > 2 for a in pre if a["action"]["type"] == "call"):
        return 5
    if "call" in kinds:
        return 3
    return -10  # checked the big blind or completed the small blind: any two cards


def _pairs_or_draws(hole: list[str], board: list[str], allow_overcards: bool) -> bool:
    board_ranks = {c[0] for c in board}
    if hole[0][0] == hole[1][0] or any(c[0] in board_ranks for c in hole):
        return True
    for suit in "cdhs":
        if sum(c[1] == suit for c in hole + board) >= 4 and any(c[1] == suit for c in hole):
            return True
    if allow_overcards and board:
        top = max(RANKS.index(c[0]) for c in board)
        return all(RANKS.index(c[0]) > top for c in hole)
    return False


def consistent(hole: list[str], actions: list[dict[str, Any]]) -> bool:
    """Whether an opponent holding `hole` plausibly took `actions` (its own actions in this hand)."""
    if chen_score(hole) < _preflop_min_chen(actions):
        return False
    for a in actions:
        if a["street"] == "preflop":
            continue
        kind = a["action"]["type"]
        if kind in ("bet", "raise") and not _pairs_or_draws(hole, a["board"], allow_overcards=False):
            return False
        if kind == "call" and not _pairs_or_draws(hole, a["board"], allow_overcards=True):
            return False
    return True


def _share(hands: dict[str, list[str]], board: list[str], hero: str) -> float:
    """Hero's fraction of the pot on a complete board (split pots share)."""
    cards = {
        name: StandardHighHand.from_game(list(Card.parse("".join(h))), list(Card.parse("".join(board))))
        for name, h in hands.items()
    }
    best = max(cards.values())
    winners = [name for name, hand in cards.items() if hand == best]
    return 1.0 / len(winners) if hero in winners else 0.0


BLINDS_BY_POSITION = {0: 1, 1: 2}  # small and big blind, in chips, for the 1/2 game


def _contributions(hand: dict[str, Any]) -> dict[str, int]:
    """Chips each player put in. A seat's stack after its last action is pre-payout, except for the action that
    ends the hand: the engine pays the pot before recording it, so that stack is rebuilt from the stack before the
    action minus what the action committed (a call's price; a check or fold commits nothing)."""
    stack = {
        name: info["starting_stack"] - BLINDS_BY_POSITION.get(info.get("position", -1), 0)
        for name, info in hand["starting_state"].items()
    }
    actions = hand["actions"]
    for i, a in enumerate(actions):
        if i == len(actions) - 1:
            before = stack[a["player"]]
            committed = min(a["call_amount"], before) if a["action"]["type"] == "call" else 0
            stack[a["player"]] = before - committed
        else:
            stack[a["player"]] = a["stack_after"]
    return {name: info["starting_stack"] - stack[name] for name, info in hand["starting_state"].items()}


def hand_ev(
    hand: dict[str, Any],
    hero: str,
    big_blind: int,
    variants: int = 8,
    runouts: int = 32,
    seed: str = "",
    redraw: bool = True,
) -> dict[str, Any]:
    """The hero's realized and expected result for one hand, in big blinds, and how the expectation was taken.

    With `redraw=False` the opponents keep the hands they were dealt, so only an all-in's runout is averaged."""
    start = hand["starting_state"][hero]["starting_stack"]
    realized = (hand["ending_stacks"][hero] - start) / big_blind
    folded = {a["player"] for a in hand["actions"] if a["action"]["type"] == "fold"}
    live = [name for name in hand["starting_state"] if name not in folded]
    out = {"realized_bb": realized, "ev_bb": realized, "method": "no-showdown"}
    if hero not in live or len(live) < 2:
        return out
    contrib = _contributions(hand)
    if len({contrib[name] for name in live}) > 1:
        return out | {"method": "side-pot"}
    pot = sum(contrib.values())
    last_street = hand["actions"][-1]["street"]
    able = [name for name in live if hand["starting_state"][name]["starting_stack"] - contrib[name] > 0]
    allin_before_river = len(able) <= 1 and last_street != "river"
    known_board = hand["final_board"][: BOARD_AT[last_street]] if allin_before_river else hand["final_board"]
    rng = random.Random(f"ev:{seed}:{hand['hand_number']}:{hero}")
    hero_hole = hand["starting_state"][hero]["hole_cards"]
    shares, fallbacks = [], 0
    for _ in range(variants):
        used = set(hero_hole) | set(known_board)
        holes = {hero: hero_hole}
        for name in live:
            if name == hero:
                continue
            mine = [a for a in hand["actions"] if a["player"] == name]
            available = [c for c in DECK if c not in used]
            chosen = None if redraw else hand["starting_state"][name]["hole_cards"]
            for _try in range(MAX_TRIES if redraw else 0):
                candidate = rng.sample(available, 2)
                if consistent(candidate, mine):
                    chosen = candidate
                    break
            if redraw and (chosen is None or set(chosen) & used):
                dealt = hand["starting_state"][name]["hole_cards"]
                chosen = dealt if not set(dealt) & used else rng.sample(available, 2)
                fallbacks += 1
            holes[name] = chosen
            used |= set(chosen)
        if allin_before_river:
            rest = [c for c in DECK if c not in used]
            need = 5 - len(known_board)
            for _r in range(runouts):
                shares.append(_share(holes, known_board + rng.sample(rest, need), hero))
        else:
            shares.append(_share(holes, known_board, hero))
    share = statistics.fmean(shares)
    ev = (share * pot - contrib[hero]) / big_blind
    return out | {
        "ev_bb": round(ev, 3),
        "method": "allin" if allin_before_river else "showdown",
        "ev_share": round(share, 4),
        "pot_bb": pot / big_blind,
        "redraw_fallbacks": fallbacks,
    }


def label_ev(
    log: dict[str, Any], seats: set[str], variants: int = 8, runouts: int = 32, redraw: bool = False
) -> dict[str, Any]:
    """Add `ev` = {seat: hand_ev(...)} to every hand for the seats in `seats`, in place; returns the log so a process
    pool can run it."""
    big_blind = log["session_info"]["big_blind"]
    for hand in log["hands"]:
        hand["ev"] = {
            seat: hand_ev(hand, seat, big_blind, variants, runouts, seed=log["game_id"], redraw=redraw)
            for seat in seats
            if seat in hand["starting_state"]
        }
    return log
