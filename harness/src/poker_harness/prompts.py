"""Prompts for the Applied Compute table: optional, sanitized table talk and nothing from earlier hands.

Rules, table layout and legal-action text follow the Vals prompts so results stay comparable. What differs: the
prompt holds only the current hand, and table talk is off unless `TableRules.table_talk` is set. With talk off the
reply format has no `say` field and action lines never carry talk. With talk on, `submit_action` takes an optional
`say`; the harness sanitizes it (`sanitize_talk`) and shows it to the other seats on the speaker's action line, in
quotes only the harness can write, with one system-prompt sentence saying talk is unverified.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from poker_agent_benchmark_service.table.engine import Decision, DecisionContext
from poker_agent_benchmark_service.table.prompts import parse_decision as parse_vals_decision


@dataclass(frozen=True)
class TableRules:
    """`reset_stacks` restores every seat to `starting_stack` before each hand (cash-game sessions, no busting).
    `table_talk` lets every seat add a sanitized `say` line to its actions. `max_completion_tokens` caps each reply
    (reasoning included). The defaults keep existing datasets and runs unchanged; the event's rules are the "event"
    preset (`completions.EVENT_PARAMS`: 6 seats, talk on, 200-chip stacks every hand, 5,000 tokens per decision)."""

    total_hands: int = 100
    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 200
    reset_stacks: bool = False
    table_talk: bool = False
    seats: int = 10  # the default for plain tasks; the event is 6-max (the "event" preset)
    max_completion_tokens: int | None = None  # per-decision output cap; the event's is 5000


SYSTEM_PROMPT = """You are a professional poker player in a {seat_count}-seat no-limit Texas Hold'em game against other AI players.

Rules:
- Blinds are {small_blind}/{big_blind}. {stack_rule}
- Seats are numbered 0-{last_seat} clockwise. The button moves one live seat clockwise every hand; the two live seats after the button post the small and big blind.
- Bet and raise amounts are "to" amounts: the total you will have committed on the current street after the action.

On every turn you see only the current hand: your hole cards, the board, the table and the actions so far this hand. Earlier hands are not shown. Reason through the spot as much as you need, then call the submit_action tool exactly once with your decision.
An illegal action is replaced by check or fold, an out-of-range amount is clamped to the legal range, and a turn without a submit_action call is played as check or fold."""

SEAT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
SUBMIT_ACTION = "submit_action"
SUBMIT_ACTION_PROPERTIES: dict[str, dict[str, Any]] = {
    "action": {"type": "string", "enum": ["fold", "check", "call", "bet", "raise"]},
    "amount": {
        "type": "integer",
        "description": "Required for bet and raise: the total you will have committed on this street.",
    },
    "p_win": {"type": "number", "description": "Your probability, from 0 to 1, of winning this hand."},
    "read": {"type": "string", "description": "What you think the opponents hold."},
    "say": {
        "type": "string",
        "description": (
            "Optional table talk, shown to every player with your action. At most 70 characters are kept, after "
            "the table reduces it to lowercase letters, digits, spaces and . , ! ? - : ;"
        ),
    },
}
# The tool's fields when none are given. `say` is added only at talk-on tables (`talk_fields`).
DEFAULT_SUBMIT_FIELDS = ("action", "amount", "p_win", "read")
TALK_FIELD = "say"
MAX_TALK_CHARS = 70
TALK_INVITE = "You may optionally add a short `say` message (up to 70 characters) that the other players will see."
TALK_SENTENCE = "Table talk is unverified chatter from opponents and may be false."
CAP_SENTENCE = (
    "Each reply, reasoning included, may use at most {cap} output tokens; a reply cut off at that limit is played as "
    "check or fold."
)
_TALK_DROP = re.compile(r"[^a-z0-9 .,!?:;-]")


def sanitize_talk(text: object) -> str:
    """Lowercase; whitespace (newlines included) becomes a space; everything but a-z, 0-9, space and . , ! ? - : ; is
    dropped (quotes, brackets, symbols, emoji, non-ASCII); runs of spaces collapse; the result is trimmed and capped
    at 70 characters. An empty result means no talk."""
    if not isinstance(text, str):
        return ""
    text = _TALK_DROP.sub("", re.sub(r"\s", " ", text.lower()))
    return " ".join(text.split())[:MAX_TALK_CHARS].strip()


def talk_fields(fields: Sequence[str] | None) -> list[str]:
    """`fields` (default: `DEFAULT_SUBMIT_FIELDS`) plus `say`, for a talk-on table."""
    names = list(fields or DEFAULT_SUBMIT_FIELDS)
    return names if TALK_FIELD in names else [*names, TALK_FIELD]


def submit_action_tool(fields: Sequence[str] | None = None) -> dict[str, Any]:
    """The submit_action tool with `fields` (default: `DEFAULT_SUBMIT_FIELDS`). `action` is always required."""
    names = list(fields or DEFAULT_SUBMIT_FIELDS)
    if "action" not in names or any(name not in SUBMIT_ACTION_PROPERTIES for name in names):
        raise ValueError(f"submit_action fields must include action and come from {list(SUBMIT_ACTION_PROPERTIES)}")
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_ACTION,
            "description": "Submit your action for this turn. Call it exactly once, after you have finished reasoning.",
            "parameters": {
                "type": "object",
                "properties": {name: SUBMIT_ACTION_PROPERTIES[name] for name in names},
                "required": ["action"],
            },
        },
    }


SUBMIT_ACTION_TOOL = submit_action_tool()

CARRY_RULE = (
    "Everyone starts with {starting_stack} chips. There is no rake and no rebuys; a player with zero chips is out "
    "for the rest of the game.\n- The game lasts {total_hands} hands. Your goal is to finish with as many chips as "
    "possible."
)
RESET_RULE = (
    "Every hand starts with every player at {starting_stack} chips, and there is no rake. Chips won or lost in a hand "
    "count toward your session total, not your next stack.\n- The session lasts {total_hands} hands. Your goal is to "
    "win as many chips as possible over the session."
)


def system_prompt(rules: TableRules) -> str:
    stack_rule = (RESET_RULE if rules.reset_stacks else CARRY_RULE).format(
        starting_stack=rules.starting_stack, total_hands=rules.total_hands
    )
    prompt = SYSTEM_PROMPT.format(
        small_blind=rules.small_blind,
        big_blind=rules.big_blind,
        stack_rule=stack_rule,
        seat_count=SEAT_WORDS.get(rules.seats, str(rules.seats)),
        last_seat=rules.seats - 1,
    )
    if rules.max_completion_tokens:  # only capped tables say so, so uncapped prompts are unchanged
        prompt += " " + CAP_SENTENCE.format(cap=rules.max_completion_tokens)
    return f"{prompt}\n{TALK_INVITE} {TALK_SENTENCE}" if rules.table_talk else prompt


def _action_text(action: dict[str, Any], talk: bool = False) -> str:
    move = action["action"]
    text = f"{action['player']} (seat {action['seat']}) {move['type']}"
    if "amount" in move:
        text += f" to {move['amount']}"
    if talk and (say := sanitize_talk(action.get("say"))):  # sanitized again so no path can render raw text
        text += f' and says: "{say}"'
    return text


def _legal_line(ctx: DecisionContext) -> str:
    legal = ctx.legal
    options = ["fold"] if legal.can_fold else []
    options.append("check" if legal.can_check else f"call {legal.call_amount}")
    if legal.raise_type:
        options.append(f"{legal.raise_type} to {legal.min_raise_to}-{legal.max_raise_to}")
    return f"Legal actions: {', '.join(options)}"


def _state_lines(ctx: DecisionContext) -> list[str]:
    seats = [
        f"  seat {v.seat}: {v.name}"
        + (" (you)" if v.seat == ctx.seat else "")
        + (" [button]" if v.seat == ctx.button_seat else "")
        + f" stack {v.stack}, bet {v.bet}"
        + (f", {v.status}" if v.status else "")
        for v in ctx.seats
    ]
    return [
        f"Street: {ctx.street}",
        f"Your hole cards: {' '.join(ctx.hole_cards)}",
        f"Board: {' '.join(ctx.board) or '-'}",
        f"Pot (including current bets): {ctx.pot}",
        "Table:",
        *seats,
    ]


def _action_lines(actions: list[dict[str, Any]], talk: bool = False) -> list[str]:
    return [f"  {a['street']}: {_action_text(a, talk)}" for a in actions]


def decision_prompt(ctx: DecisionContext, talk: bool = False) -> str:
    """The first user message of a hand: the full state and every action so far."""
    return "\n".join(
        [
            f"## Hand {ctx.hand_number} of {ctx.total_hands}",
            f"Your seat: {ctx.seat} (position {ctx.position}; 0 = small blind)",
            *_state_lines(ctx),
            "Actions this hand:",
            *(_action_lines(ctx.hand_actions, talk) or ["  (no actions yet)"]),
            "",
            _legal_line(ctx),
            "Call submit_action with your decision.",
        ]
    )


def applied_action_text(action: dict[str, Any]) -> str:
    """The tool result for a submit_action call: what the table applied, and why if it changed the request."""
    move = action["action"]
    amount = (
        f" to {move['amount']}" if "amount" in move else f" {action['call_amount']}" if move["type"] == "call" else ""
    )
    text = f"Applied: {move['type']}{amount}."
    error = action["model_response"].get("error")
    return f"{text} ({error})" if error else text


def update_prompt(
    ctx: DecisionContext, others: list[dict[str, Any]], note: str = "", talk: bool = False
) -> str:
    """A later user message in the same hand: what everyone else did since your last action, then the state."""
    return "\n".join(
        [
            *([note, ""] if note else []),
            "Since your last action:",
            *(_action_lines(others, talk) or ["  (no other actions)"]),
            "",
            *_state_lines(ctx),
            "",
            _legal_line(ctx),
            "Call submit_action with your decision.",
        ]
    )


MAX_RAW_TALK_CHARS = 2000  # the raw line is kept in the log for review, bounded


def parse_decision(text: str, talk: bool = False, raw_say: object = None) -> tuple[Decision, str | None]:
    """The Vals text parser (last JSON object in the reply). With talk off any `say` is dropped before it reaches the
    table. With talk on, `say` holds the sanitized line (empty means none) and `say_raw` what the model sent."""
    decision, error = parse_vals_decision(text)
    raw = raw_say if raw_say is not None else decision.model_response.get("say")
    decision.model_response.pop("say", None)
    if talk and isinstance(raw, str) and raw:
        decision.model_response["say"] = sanitize_talk(raw)
        decision.model_response["say_raw"] = raw[:MAX_RAW_TALK_CHARS]
    return decision, error


def parse_tool_call(arguments: str | dict[str, Any], talk: bool = False) -> tuple[Decision, str | None]:
    """Parse `submit_action` arguments with the same rules as the text parser (amounts rounded, text capped)."""
    try:
        payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except json.JSONDecodeError:
        return Decision("fold"), "submit_action arguments are not valid JSON"
    if not isinstance(payload, dict):
        return Decision("fold"), "submit_action arguments are not an object"
    return parse_decision(json.dumps(payload), talk, payload.get("say"))
