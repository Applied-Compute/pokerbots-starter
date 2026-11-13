"""One hand of no-limit Texas Hold'em on top of PokerKit: dealing, betting, side pots and showdown.

Ported from Vals' poker-agent engine (poker-agent-benchmark-service 0.1.0) and used with Vals' permission; behavior is
unchanged, and tests/test_engine_parity.py checks it hand for hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from pokerkit import Automation, Card, Deck, HoleCardsShowingOrMucking, Mode, NoLimitTexasHoldem, State

ActionType = Literal["fold", "check", "call", "bet", "raise"]
STREETS = ("preflop", "flop", "turn", "river")

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.RUNOUT_COUNT_SELECTION,
)


@dataclass(frozen=True)
class LegalActions:
    """`raise_type` is None when the seat cannot bet or raise; the range is then empty."""

    can_fold: bool
    can_check: bool
    call_amount: int
    raise_type: Literal["bet", "raise"] | None = None
    min_raise_to: int = 0
    max_raise_to: int = 0


@dataclass
class Decision:
    type: ActionType
    amount: int | None = None
    model_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeatView:
    seat: int
    name: str
    stack: int
    bet: int
    status: Literal["", "folded", "all-in", "out"]


@dataclass(frozen=True)
class DecisionContext:
    """Everything a player may see when it is asked to act."""

    hand_number: int
    total_hands: int
    street: str
    seat: int
    position: int
    button_seat: int
    hole_cards: list[str]
    board: list[str]
    pot: int
    seats: list[SeatView]
    hand_actions: list[dict[str, Any]]
    legal: LegalActions


class Player(Protocol):
    name: str

    async def decide(self, ctx: DecisionContext, history: str) -> Decision: ...


def codes(cards: Any) -> list[str]:
    return [repr(card) for card in cards]


def seeded_deck(seed: int, hand_number: int) -> list[Card]:
    """Deck order depends only on (seed, hand number), so a reused seed replays the same cards."""
    deck = list(Deck.STANDARD)
    random.Random(f"{seed}:{hand_number}").shuffle(deck)
    return deck


def legal_actions(state: State) -> LegalActions:
    call_amount = state.checking_or_calling_amount or 0
    legal = LegalActions(can_fold=call_amount > 0, can_check=call_amount == 0, call_amount=call_amount)
    if not state.can_complete_bet_or_raise_to():
        return legal
    facing_bet = state.street_index == 0 or max(state.bets) > 0
    return replace(
        legal,
        raise_type="raise" if facing_bet else "bet",
        min_raise_to=state.min_completion_betting_or_raising_to_amount or 0,
        max_raise_to=state.max_completion_betting_or_raising_to_amount or 0,
    )


def resolve(decision: Decision, legal: LegalActions) -> tuple[Decision, str | None]:
    """Map a requested action onto a legal one. Returns the action to apply and an error if it was changed."""
    kind = decision.type
    if kind in ("bet", "raise"):
        if legal.raise_type is None:
            fallback = "check" if legal.can_check else "call"
            return Decision(fallback), f"{kind} not legal; applied {fallback}"
        if decision.amount is None:
            return Decision(
                legal.raise_type, legal.min_raise_to
            ), f"{kind} without amount; applied min {legal.min_raise_to}"
        amount = min(max(decision.amount, legal.min_raise_to), legal.max_raise_to)
        errors = []
        if kind != legal.raise_type:
            errors.append(f"{kind} not legal; applied {legal.raise_type}")
        if amount != decision.amount:
            errors.append(f"{kind} to {decision.amount} clamped to {amount}")
        return Decision(legal.raise_type, amount), "; ".join(errors) or None
    if kind == "check" and not legal.can_check:
        return Decision("fold"), "check not legal facing a bet; applied fold"
    if kind == "call" and legal.can_check:
        return Decision("check"), None
    if kind == "fold" and not legal.can_fold:
        return Decision("check"), "fold not needed; applied check"
    return Decision(kind), None


class HandRunner:
    def __init__(
        self,
        *,
        players: dict[int, Player],
        stacks: dict[int, int],
        button_seat: int,
        hand_number: int,
        total_hands: int,
        seed: int,
        small_blind: int,
        big_blind: int,
    ) -> None:
        self.players = players
        self.stacks = stacks
        self.button_seat = button_seat
        self.hand_number = hand_number
        self.total_hands = total_hands
        self.deck = seeded_deck(seed, hand_number)
        live = sorted(seat for seat, stack in stacks.items() if stack > 0)
        self.order = [s for s in live if s > button_seat] + [
            s for s in live if s <= button_seat
        ]  # SB first, button last
        self.state = NoLimitTexasHoldem.create_state(
            AUTOMATIONS,
            True,
            0,
            (small_blind, big_blind),
            big_blind,
            tuple(stacks[s] for s in self.order),
            len(self.order),
            mode=Mode.CASH_GAME,
        )

    def _deal(self) -> None:
        state = self.state
        while True:
            if state.can_deal_hole():
                state.deal_hole(self.deck.pop())
            elif state.can_burn_card():
                state.burn_card(self.deck.pop())
            elif state.can_deal_board():
                state.deal_board(tuple(self.deck.pop() for _ in range(state.board_dealing_count or 0)))
            else:
                return

    def _board(self) -> list[str]:
        return codes(self.state.get_board_cards(0))

    def _seat_views(self, folded: set[int]) -> list[SeatView]:
        views = []
        for seat, player in sorted(self.players.items()):
            if seat not in self.order:
                views.append(SeatView(seat, player.name, 0, 0, "out"))
                continue
            i = self.order.index(seat)
            stack, bet = self.state.stacks[i], self.state.bets[i]
            status = "folded" if seat in folded else "all-in" if stack == 0 else ""
            views.append(SeatView(seat, player.name, stack, bet, status))
        return views

    def _apply(self, decision: Decision) -> None:
        if decision.type == "fold":
            self.state.fold()
        elif decision.type in ("check", "call"):
            self.state.check_or_call()
        else:
            self.state.complete_bet_or_raise_to(decision.amount)

    async def play(self, history_for: dict[int, str], timestep: int) -> tuple[dict[str, Any], dict[int, int]]:
        """Returns the hand record and every seat's stack after the hand."""
        state = self.state
        self._deal()
        starting_state = {
            self.players[seat].name: {
                "hole_cards": codes(state.hole_cards[i]),
                "position": i,
                "seat": seat,
                "starting_stack": state.stacks[i] + state.bets[i],
            }
            for i, seat in enumerate(self.order)
        }
        actions: list[dict[str, Any]] = []
        folded: set[int] = set()
        while True:
            self._deal()  # an all-in runout can finish the hand here
            if not state.status:
                break
            i = state.actor_index
            if i is None:
                raise RuntimeError(f"hand {self.hand_number} stalled with no actor")
            seat = self.order[i]
            player = self.players[seat]
            legal = legal_actions(state)
            pot_before = state.total_pot_amount
            stack_before = state.stacks[i]
            street = STREETS[state.street_index or 0]
            board = self._board()
            ctx = DecisionContext(
                hand_number=self.hand_number,
                total_hands=self.total_hands,
                street=street,
                seat=seat,
                position=i,
                button_seat=self.button_seat,
                hole_cards=codes(state.hole_cards[i]),
                board=board,
                pot=pot_before,
                seats=self._seat_views(folded),
                hand_actions=actions,
                legal=legal,
            )
            requested = await player.decide(ctx, history_for[seat])
            applied, error = resolve(requested, legal)
            self._apply(applied)
            if applied.type == "fold":
                folded.add(seat)
            response = dict(requested.model_response)
            if error:
                response.update(fallback_used=True, error=error)
            action: dict[str, Any] = {"type": applied.type}
            if applied.amount is not None:
                action["amount"] = applied.amount
            actions.append(
                {
                    "timestep": timestep,
                    "player": player.name,
                    "seat": seat,
                    "street": street,
                    "action": action,
                    "chip_change": state.stacks[i] - stack_before,
                    "stack_after": state.stacks[i],
                    "pot_before": pot_before,
                    "call_amount": legal.call_amount,
                    "board": board,
                    "say": response.get("say") or "",
                    "model_response": response,
                }
            )
            timestep += 1

        ending = dict(self.stacks) | {seat: state.stacks[i] for i, seat in enumerate(self.order)}
        showdown = len(self.order) - len(folded) > 1
        shown = [op for op in state.operations if isinstance(op, HoleCardsShowingOrMucking) and op.hole_cards]
        record = {
            "hand_number": self.hand_number,
            "button_seat": self.button_seat,
            "starting_state": starting_state,
            "actions": actions,
            "ending_stacks": {self.players[s].name: ending[s] for s in sorted(self.players)},
            "final_board": self._board(),
            "showdown": showdown,
            "shown_hole_cards": {
                self.players[self.order[op.player_index]].name: codes(op.hole_cards) for op in shown if showdown
            },
        }
        return record, ending
