"""A model seat behind any async chat-completion function, such as an OpenAI-compatible AC2 deployment.

Each hand is one conversation per seat: the first user message holds the full state, the model reasons and calls
`submit_action`, the table answers the call with what it applied, and the next user message lists what everyone
else did since, then the new state. Echoing each assistant turn back lets the AC2 relay record the hand as one
episode instead of one per decision.

With `rules.table_talk`, other seats' sanitized talk appears on their action lines and this seat's own `say` is
sanitized into `model_response["say"]` (the raw text goes to `say_raw` for the log).

At scratchpad tables (`rules.scratchpad`) each hand's conversation opens with the seat's private scratchpad, and after
the hand `cleanup` sends one more message (what the seat hasn't seen yet and how the hand ended) and applies the
seat's single `update_scratchpad` call. Everything private (the scratchpad, cleanup turns, the model's own text) goes
to `self.private` and to `model_response["private"]`, never anywhere else; `public_messages` withholds it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .engine import Decision, DecisionContext

from .prompts import (
    SUBMIT_ACTION,
    UPDATE_SCRATCHPAD,
    TableRules,
    applied_action_text,
    cleanup_prompt,
    decision_prompt,
    parse_decision,
    parse_tool_call,
    scratchpad_block,
    scratchpad_rejection,
    system_prompt,
    update_prompt,
)

Message = dict[str, Any]
PRIVATE_RESPONSE_KEYS = ("reasoning", "raw_output", "read", "say_raw")  # the model's own text, at scratchpad tables
TOO_EARLY = "Rejected: update_scratchpad works only after the hand ends; this turn was played as check or fold."


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("function", {}).get("name", ""))


def invalid_action(decision: Decision, ctx: DecisionContext) -> str | None:
    """Why `decision` is not a legal action as given (`rules.strict_actions`), or None. Bet and raise are one action
    under two names, a call with nothing to call is a check and a fold that could check is applied as a check."""
    legal = ctx.legal
    if decision.type in ("bet", "raise"):
        if legal.raise_type is None:
            return f"{decision.type} is not legal here"
        if decision.amount is None:
            return f"{decision.type} without an amount"
        if not legal.min_raise_to <= decision.amount <= legal.max_raise_to:
            return f"{decision.type} to {decision.amount} is outside {legal.min_raise_to}-{legal.max_raise_to}"
    if decision.type == "check" and not legal.can_check:
        return "check is not legal facing a bet"
    return None


@dataclass
class Completion:
    """`tool_arguments` holds the last `submit_action` call's arguments; `tool_calls` are the raw calls to echo."""

    text: str
    tool_arguments: str | dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    in_tokens: int | None = None
    out_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    truncated: bool = False  # the reply ended at the output-token cap (chat finish_reason "length" or Responses
    # status "incomplete" for max_output_tokens); it is played as a format error even if it holds an action


Complete = Callable[[list[Message]], Awaitable[str | Completion]]


@dataclass
class ModelSeat:
    """Asks `complete(messages)` once per decision. Errors, timeouts and unparseable replies check or fold."""

    name: str
    complete: Complete
    rules: TableRules
    model: str | None = None
    timeout_seconds: float | None = None
    cleanup_complete: Complete | None = None  # the after-hand turn's caller (its own token cap); default `complete`
    turn: int = field(default=0, init=False)
    messages: list[Message] = field(default_factory=list, init=False)
    _hand: int | None = field(default=None, init=False)
    _seen: int = field(default=0, init=False)
    _pending_calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    scratchpad: str = field(default="", init=False)
    private: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)  # per hand number
    _hand_info: dict[str, Any] = field(default_factory=dict, init=False)
    _private_messages: set[int] = field(default_factory=set, init=False)  # indexes into `messages`

    def begin_hand(self, info: dict[str, Any]) -> None:
        """Called by `play_session` before each hand: number, blinds, orbit and stacks."""
        self._hand_info = info

    def public_messages(self) -> list[Message]:
        """This hand's conversation without the scratchpad block and the cleanup turn."""
        return [m for i, m in enumerate(self.messages) if i not in self._private_messages]

    def _open_hand(self, hand_number: int) -> None:
        self._hand = hand_number
        self.messages = [{"role": "system", "content": self.system}]
        self._private_messages = set()
        if self.rules.scratchpad:
            self._private_messages.add(len(self.messages))
            self.messages.append(
                {"role": "user", "content": scratchpad_block(self.scratchpad, self.rules.scratchpad_chars)}
            )
            self.private.setdefault(hand_number, {})["scratchpad_before"] = self.scratchpad

    def _answer_calls(self, own: dict[str, Any] | None) -> str:
        """Tool results for the last turn's calls; returns the applied-action text."""
        applied = applied_action_text(own) if own else "Applied: nothing."
        submits = [c for c in self._pending_calls if _tool_name(c) == SUBMIT_ACTION]
        for call in self._pending_calls:
            name = _tool_name(call)
            if name == SUBMIT_ACTION:
                used = call is submits[-1]
                content = applied if used else "Ignored: one submit_action per turn; the last call was used."
            elif name == UPDATE_SCRATCHPAD:
                content = TOO_EARLY
            else:
                content = f"Ignored: unknown tool {name}."
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        return applied

    @property
    def system(self) -> str:
        return system_prompt(self.rules)

    def _next_messages(self, ctx: DecisionContext) -> list[Message]:
        """Start the hand's conversation, or answer the last call and add what happened since."""
        if ctx.hand_number != self._hand:
            self._open_hand(ctx.hand_number)
            prompt = decision_prompt(ctx, self.rules.table_talk)
            self.messages.append({"role": "user", "content": prompt})
        else:
            new = ctx.hand_actions[self._seen :]
            own = new[0] if new and new[0]["player"] == self.name else None
            others = new[1:] if own else new
            submitted = any(_tool_name(c) == SUBMIT_ACTION for c in self._pending_calls)
            applied = self._answer_calls(own)
            note = "" if submitted else f"Your last turn made no usable submit_action call. {applied}"
            prompt = update_prompt(ctx, others, note, self.rules.table_talk)
            if self.messages[-1]["role"] == "user":  # the last call failed outright; fold its turn into this one
                self.messages[-1] = {"role": "user", "content": f"{self.messages[-1]['content']}\n\n{prompt}"}
            else:
                self.messages.append({"role": "user", "content": prompt})
        self._pending_calls = []
        self._seen = len(ctx.hand_actions)
        return list(self.messages)

    async def _ask(self, messages: list[Message]) -> Completion:
        pending = self.complete(messages)
        reply = await (asyncio.wait_for(pending, self.timeout_seconds) if self.timeout_seconds else pending)
        return reply if isinstance(reply, Completion) else Completion(text=reply)

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        del history  # earlier hands are never shown
        self.turn += 1
        model = self.model or self.name
        messages = self._next_messages(ctx)
        started = time.perf_counter()
        try:
            reply, error = await self._ask(messages), None
        except Exception as exc:  # noqa: BLE001 - any failure checks or folds and is recorded, never stops the table
            reply, error = Completion(text=""), f"{type(exc).__name__}: {str(exc)[:300]}"
        if not error:
            assistant: Message = {"role": "assistant", "content": reply.text}
            if reply.tool_calls:
                assistant["tool_calls"] = reply.tool_calls
            self.messages.append(assistant)
            self._pending_calls = [
                c for c in reply.tool_calls if self.rules.scratchpad or _tool_name(c) == SUBMIT_ACTION
            ]
        if error:
            decision, parse_error = Decision("fold"), None
        elif reply.truncated:  # the cap is a hard limit: a cut reply never counts, whatever it holds
            cap = self.rules.max_completion_tokens
            decision, parse_error = Decision("fold"), f"reply hit the {cap or 'output'}-token cap"
        elif self.rules.scratchpad and any(_tool_name(c) == UPDATE_SCRATCHPAD for c in reply.tool_calls):
            decision, parse_error = Decision("fold"), "update_scratchpad called during the hand"
        elif reply.tool_arguments is not None:
            decision, parse_error = parse_tool_call(reply.tool_arguments, self.rules.table_talk)
        else:  # no tool call: accept a JSON action in the text, as the Vals format does
            decision, parse_error = parse_decision(reply.text, self.rules.table_talk)
        error = error or parse_error
        if not error and self.rules.strict_actions:
            error = invalid_action(decision, ctx)
        response: dict[str, Any] = {
            "model_name": model,
            "provider": model.split("/", 1)[0],
            "turn": self.turn,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "in_tokens": reply.in_tokens,
            "out_tokens": reply.out_tokens,
            "reasoning_tokens": reply.reasoning_tokens or 0,
            "cost": reply.cost,
            "fallback_used": False,
            "error": None,
            **decision.model_response,
        }
        response["reasoning"] = response.get("reasoning") or reply.text[-2000:]
        response["submitted_by_tool"] = reply.tool_arguments is not None and not reply.truncated
        if reply.truncated:
            response["truncated"] = True
        # A failed call or unusable reply; the engine's later legal-action adjustments only set `fallback_used`.
        response["format_error"] = bool(error)
        if error:
            response.update(fallback_used=True, error=error, raw_output=reply.text[-2000:])
            if self.rules.strict_actions:  # the official rules' forced default action
                response.update(afk=True, afk_reason=error)
            decision = Decision("check" if ctx.legal.can_check else "fold")
        if self.rules.scratchpad:  # the model's own text may quote its scratchpad: keep it in the private section
            response["private"] = {k: response.pop(k) for k in PRIVATE_RESPONSE_KEYS if k in response}
        decision.model_response = response
        return decision

    def _update(self, reply: Completion, error: str | None) -> dict[str, Any]:
        """One cleanup reply judged: "ok", "too_long" (with contents), "malformed", "none", "cut" or "error"."""
        calls = [c for c in reply.tool_calls if _tool_name(c) == UPDATE_SCRATCHPAD]
        attempt: dict[str, Any] = {
            "output": reply.text,
            "update_calls": [c["function"].get("arguments") for c in calls],
            "ignored_calls": len(reply.tool_calls) - min(len(calls), 1),
            "truncated": reply.truncated,
            "out_tokens": reply.out_tokens,
        }
        if error:
            return attempt | {"outcome": "error", "error": error}
        if reply.truncated:
            cap = self.rules.cleanup_max_completion_tokens or self.rules.max_completion_tokens
            return attempt | {"outcome": "cut", "error": f"reply hit the {cap or 'output'}-token cap"}
        if not calls:
            return attempt | {"outcome": "none", "error": "no update_scratchpad call"}
        arguments = calls[0]["function"].get("arguments")
        try:
            payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            contents = payload["contents"]
            if not isinstance(contents, str):
                raise TypeError("contents is not a string")
        except (ValueError, TypeError, KeyError) as exc:
            reason = f"malformed update_scratchpad arguments ({type(exc).__name__}: {str(exc)[:200]})"
            return attempt | {"outcome": "malformed", "error": reason, "call_id": calls[0]["id"]}
        limit = self.rules.scratchpad_chars
        if len(contents) > limit:
            reason = f"your update has {len(contents)} characters; the limit is {limit}"
            return attempt | {"outcome": "too_long", "error": reason, "contents": contents, "call_id": calls[0]["id"]}
        return attempt | {"outcome": "ok", "contents": contents}

    async def _cleanup_reply(self) -> tuple[Completion, str | None]:
        complete = self.cleanup_complete or self.complete
        try:
            pending = complete(list(self.messages))
            reply = await (asyncio.wait_for(pending, self.timeout_seconds) if self.timeout_seconds else pending)
            return (reply if isinstance(reply, Completion) else Completion(text=reply)), None
        except Exception as exc:  # noqa: BLE001 - a failed cleanup leaves the scratchpad unchanged
            return Completion(text=""), f"{type(exc).__name__}: {str(exc)[:300]}"

    def _record_reply(self, reply: Completion) -> None:
        assistant: Message = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            assistant["tool_calls"] = reply.tool_calls
        self.messages.append(assistant)

    async def cleanup(self, summary: dict[str, Any]) -> dict[str, Any]:
        """The after-hand turn at scratchpad tables: report what this seat hasn't seen and how the hand ended, then
        apply its first `update_scratchpad` call. A first update that is too long or malformed is rejected with the
        reason and the seat gets one more try; if that fails too, a too-long update is cut to `scratchpad_chars` and a
        malformed one changes nothing. Extra calls and submit_action are ignored; no call, a failed call or a reply
        cut at the token cap leaves the scratchpad unchanged. Every message of this turn is private."""
        hand = summary["hand_number"]
        actions = summary["actions"]
        fresh = self._hand != hand
        if fresh:  # dealt in but never asked to act (e.g. a walk): start the hand's conversation here
            self._open_hand(hand)
            since = actions
        else:
            own = actions[self._seen] if self._seen < len(actions) else None
            own = own if own and own["player"] == self.name else None
            self._answer_calls(own)
            since = actions[self._seen + 1 :] if own else actions[self._seen :]
        self._pending_calls = []
        limit = self.rules.scratchpad_chars
        prompt = cleanup_prompt(summary, self.name, since, fresh, self.rules.table_talk, limit)
        first_private = len(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        reply, error = await self._cleanup_reply()
        self._record_reply(reply)
        attempts = [self._update(reply, error)]
        if attempts[0]["outcome"] in ("too_long", "malformed"):  # one more try, in the same turn
            for call in reply.tool_calls:
                rejected = call["id"] == attempts[0]["call_id"]
                content = scratchpad_rejection(attempts[0]["error"], limit) if rejected else "Ignored."
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            reply, error = await self._cleanup_reply()
            self._record_reply(reply)
            attempts.append(self._update(reply, error))
        self._private_messages.update(range(first_private, len(self.messages)))
        last = attempts[-1]
        if last["outcome"] == "ok":
            self.scratchpad, result = last["contents"], "updated"
        elif last["outcome"] == "too_long":
            self.scratchpad, result = last["contents"][:limit], "cut"
        elif attempts[0]["outcome"] == "too_long":  # the retry failed otherwise: the first update, cut
            self.scratchpad, result = attempts[0]["contents"][:limit], "cut"
        else:
            result = "unchanged"
        record = {
            "prompt": prompt,
            "attempts": attempts,
            "result": result,
            "error": None if result == "updated" else last.get("error"),
        }
        self.private.setdefault(hand, {}).update(cleanup=record, scratchpad_after=self.scratchpad)
        return record
