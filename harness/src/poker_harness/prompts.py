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

from .engine import Decision, DecisionContext
from .engine import parse_decision as parse_vals_decision


@dataclass(frozen=True)
class TableRules:
    """`reset_stacks` restores every seat to `starting_stack` before each hand (cash-game sessions, no busting).
    `table_talk` lets every seat add a sanitized `say` line to its actions. `max_completion_tokens` caps each reply
    (reasoning included). The defaults keep existing datasets and runs unchanged; the official rules are the "event"
    preset (`completions.EVENT_PARAMS`).

    Official-rules fields (all off by default):
    - `total_hands` is the game's length in hands (the event: 60, ten orbits of six); a carry-over game
      (`reset_stacks=False`) also ends when one player has all the chips. Blinds stay `small_blind`/`big_blind`.
    - `ranking`: "elimination" (players still in at the end rank above every busted player, by chips; busted
      players rank by when they busted, later is better) or "chips" (final chips only). Setting it also adds the
      official log fields (`game.play_session`); None keeps research logs as they were.
    - `strict_actions`: an invalid `submit_action` (illegal action, missing or out-of-range amount) is played as
      check if legal, else fold, instead of being mapped to the nearest legal action.
    - `scratchpad`: each seat keeps up to `scratchpad_chars` characters of private notes for the whole game, shown at
      the start of every hand and rewritten with `update_scratchpad` in a cleanup turn after each hand, whose replies
      may use `cleanup_max_completion_tokens` (the event: 10,000)."""

    total_hands: int = 100
    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 200
    reset_stacks: bool = False
    table_talk: bool = False
    seats: int = 10  # the default for existing (10-max) datasets; the event is 6-max, 5-max a research format
    max_completion_tokens: int | None = None  # per-decision output cap; the event's is 5000
    ranking: str | None = None  # None: research logs; the event's is "elimination"
    strict_actions: bool = False
    scratchpad: bool = False
    scratchpad_chars: int = 5000
    cleanup_max_completion_tokens: int | None = None  # the cleanup turn's output cap; the event's is 10000


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
LENIENT_SENTENCE = (
    "An illegal action is replaced by check or fold, an out-of-range amount is clamped to the legal range, and a turn "
    "without a submit_action call is played as check or fold."
)
STRICT_SENTENCE = (
    "A turn without a valid submit_action call (an illegal action, a missing or out-of-range amount, unreadable "
    "arguments, an update_scratchpad call, or a reply cut off at the token limit) is played as check if checking is "
    "legal, else fold, with no retry."
)
GAME_RULE = (
    "Everyone starts with {starting_stack} chips. There is no rake and no rebuys; a player with zero chips is out "
    "for the rest of the game.\n- The game lasts {total_hands} hands, or ends sooner when one player has all the "
    "chips. Players still in at the end rank above everyone who busted, by their chips; players who busted rank by "
    "when they busted (later ranks higher). Your goal is the best rank."
)
SCRATCHPAD_RULE = (
    "You have a private scratchpad: up to {limit} characters of notes that only you can see. It starts empty and "
    "lasts for the whole game, and it is shown to you at the start of every hand. After each hand you were dealt into "
    "you get a summary of how it ended and are asked to call the update_scratchpad tool exactly once with the "
    "scratchpad's full new contents, which replace the old ones; that reply may use up to {cleanup_cap} output "
    "tokens. An update over {limit} characters or with malformed arguments is rejected with the reason and you get "
    "one more try; if that fails too, a too-long update is cut to {limit} characters and a malformed one changes "
    "nothing. update_scratchpad works only after a hand: calling it during a hand makes that turn invalid. If you "
    "don't call it, or your reply is cut off, the scratchpad stays as it was."
)
UPDATE_SCRATCHPAD = "update_scratchpad"
SCRATCHPAD_OPEN, SCRATCHPAD_CLOSE = "<<<SCRATCHPAD", "SCRATCHPAD>>>"
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
    carry = GAME_RULE if rules.ranking == "elimination" else CARRY_RULE
    stack_rule = (RESET_RULE if rules.reset_stacks else carry).format(
        starting_stack=rules.starting_stack, total_hands=rules.total_hands
    )
    prompt = SYSTEM_PROMPT.format(
        small_blind=rules.small_blind,
        big_blind=rules.big_blind,
        stack_rule=stack_rule,
        seat_count=SEAT_WORDS.get(rules.seats, str(rules.seats)),
        last_seat=rules.seats - 1,
    )
    if rules.strict_actions:
        prompt = prompt.replace(LENIENT_SENTENCE, STRICT_SENTENCE)
    if rules.scratchpad:
        cleanup_cap = rules.cleanup_max_completion_tokens or rules.max_completion_tokens or "unlimited"
        prompt += "\n" + SCRATCHPAD_RULE.format(limit=rules.scratchpad_chars, cleanup_cap=cleanup_cap)
    if rules.max_completion_tokens:  # only capped tables say so, so uncapped prompts are unchanged
        cap = CAP_SENTENCE.format(cap=rules.max_completion_tokens)
        prompt += " " + (cap.replace("Each reply", "Each action turn's reply", 1) if rules.scratchpad else cap)
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


def update_scratchpad_tool(limit: int = 5000) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": UPDATE_SCRATCHPAD,
            "description": (
                f"Replace your private scratchpad with `contents` (at most {limit} characters; longer is cut off). "
                "Call it exactly once, only when asked after a hand; during a hand it makes your turn invalid."
            ),
            "parameters": {
                "type": "object",
                "properties": {"contents": {"type": "string", "description": "The scratchpad's full new contents."}},
                "required": ["contents"],
            },
        },
    }


def table_tools(fields: Sequence[str] | None = None, scratchpad: bool = False, limit: int = 5000) -> list[dict]:
    """The tools every seat gets: submit_action, plus update_scratchpad at scratchpad tables."""
    return [submit_action_tool(fields), *([update_scratchpad_tool(limit)] if scratchpad else [])]


def scratchpad_block(contents: str, limit: int) -> str:
    """The private message that opens each hand at scratchpad tables (withheld by `ModelSeat.public_messages`)."""
    return "\n".join(
        [
            "## Your scratchpad (private: only you can see it)",
            SCRATCHPAD_OPEN,
            contents or "(empty)",
            SCRATCHPAD_CLOSE,
            f"{len(contents)}/{limit} characters. You can rewrite it after this hand.",
        ]
    )


def cleanup_prompt(
    summary: dict[str, Any], seat: str, since: list[dict[str, Any]], fresh: bool, talk: bool, limit: int
) -> str:
    """The after-hand message: what this seat hasn't seen yet, how the hand ended, then the request to update the
    scratchpad. `fresh` (the seat never acted this hand) shows its cards and every action instead."""
    lines = [f"## Hand {summary['hand_number']} is over"]
    if fresh:
        mine = summary["starting_state"].get(seat, {})
        lines += [
            (
                f"You did not act this hand. Your seat: {mine.get('seat')}; "
                f"your hole cards: {' '.join(mine.get('hole_cards', [])) or '-'}."
            ),
            "Everything that happened this hand:",
        ]
    else:
        lines.append("Since your last action:")
    lines += _action_lines(since, talk) or ["  (no other actions)"]
    shown = summary.get("shown_hole_cards") or {}
    lines += [
        f"Final board: {' '.join(summary.get('final_board') or []) or '-'}",
        (
            "Showdown: "
            + "; ".join(f"{name} showed {' '.join(cards)}" for name, cards in shown.items())
            + ". Mucked and folded hands stay hidden."
            if summary.get("showdown") and shown
            else "No showdown: hands that were not shown stay hidden."
        ),
        "Pot: " + (", ".join(f"{name} won {amount}" for name, amount in summary["pot_won"].items()) or "-") + ".",
        "Stacks now: "
        + ", ".join(f"{name} {stack}" + (" (out)" if stack == 0 else "") for name, stack in summary["stacks"].items())
        + ".",
        *([f"Out of the game this hand: {', '.join(summary['eliminated'])}."] if summary.get("eliminated") else []),
        "",
        (
            f"Now call update_scratchpad exactly once with your scratchpad's full new contents (at most {limit} "
            "characters; they replace the old ones and are shown to you at the start of every later hand). Reason first "
            "if you like. submit_action does nothing now. Without a call the scratchpad stays as it is."
        ),
    ]
    return "\n".join(lines)


def scratchpad_rejection(reason: str, limit: int) -> str:
    """The tool result for a first update_scratchpad attempt that is too long or malformed: one more try."""
    return (
        f"Rejected: {reason}. Call update_scratchpad once more with at most {limit} characters. This is your last "
        f"try: if it fails too, a too-long update is cut to {limit} characters and a malformed one changes nothing."
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
