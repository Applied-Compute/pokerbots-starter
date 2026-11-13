"""A model seat behind any async chat-completion function, such as an OpenAI-compatible AC2 deployment.

Each hand is one conversation per seat: the first user message holds the full state, the model reasons and calls
`submit_action`, the table answers the call with what it applied, and the next user message lists what everyone
else did since, then the new state. Echoing each assistant turn back lets the AC2 relay record the hand as one
episode instead of one per decision.

With `rules.table_talk`, other seats' sanitized talk appears on their action lines and this seat's own `say` is
sanitized into `model_response["say"]` (the raw text goes to `say_raw` for the log).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .engine import Decision, DecisionContext

from .prompts import (
    SUBMIT_ACTION,
    TableRules,
    applied_action_text,
    decision_prompt,
    parse_decision,
    parse_tool_call,
    system_prompt,
    update_prompt,
)

Message = dict[str, Any]


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
    turn: int = field(default=0, init=False)
    messages: list[Message] = field(default_factory=list, init=False)
    _hand: int | None = field(default=None, init=False)
    _seen: int = field(default=0, init=False)
    _pending_calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def system(self) -> str:
        return system_prompt(self.rules)

    def _next_messages(self, ctx: DecisionContext) -> list[Message]:
        """Start the hand's conversation, or answer the last call and add what happened since."""
        if ctx.hand_number != self._hand:
            self._hand = ctx.hand_number
            self.messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": decision_prompt(ctx, self.rules.table_talk)},
            ]
        else:
            new = ctx.hand_actions[self._seen :]
            own = new[0] if new and new[0]["player"] == self.name else None
            others = new[1:] if own else new
            applied = applied_action_text(own) if own else "Applied: nothing."
            for i, call in enumerate(self._pending_calls):
                used = i == len(self._pending_calls) - 1
                content = applied if used else "Ignored: one submit_action per turn; the last call was used."
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            note = "" if self._pending_calls else f"Your last turn made no usable submit_action call. {applied}"
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
            self._pending_calls = [c for c in reply.tool_calls if c.get("function", {}).get("name") == SUBMIT_ACTION]
        if error:
            decision, parse_error = Decision("fold"), None
        elif reply.truncated:  # the cap is a hard limit: a cut reply never counts, whatever it holds
            cap = self.rules.max_completion_tokens
            decision, parse_error = Decision("fold"), f"reply hit the {cap or 'output'}-token cap"
        elif reply.tool_arguments is not None:
            decision, parse_error = parse_tool_call(reply.tool_arguments, self.rules.table_talk)
        else:  # no tool call: accept a JSON action in the text, as the Vals format does
            decision, parse_error = parse_decision(reply.text, self.rules.table_talk)
        error = error or parse_error
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
            decision = Decision("check" if ctx.legal.can_check else "fold")
        decision.model_response = response
        return decision
