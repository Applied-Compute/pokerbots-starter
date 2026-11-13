"""Model calls and table rules shared by the harness server and the native AC2 environment (no web framework).

`relay_completion` sends a seat's conversation to an OpenAI-compatible chat-completions endpoint (the rollout relay or
the AC2 gateway); `responses_completion` uses the Responses API, which OpenAI reasoning models need for tools with reasoning.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

import httpx

from .prompts import SUBMIT_ACTION, TableRules, table_tools
from .seat import Completion

logger = logging.getLogger("poker_harness.completions")

# The official rules (final, 2026-11-13): 6-max with sanitized table talk; 200 chips each that carry over within the
# game (no resets or rebuys; a seat at 0 is out) at blinds 1/2 all game; a game is 30 hands (five full orbits) or ends
# sooner when one player has all the chips, ranked by final chips with busted seats below every survivor in reverse
# elimination order; an invalid submit_action or a reply cut at 5,000 output tokens is played as check, else fold,
# with no retry; a 5,000-character private scratchpad per player, rewritten in a cleanup turn after each hand whose
# reply may use 10,000 output tokens; thinking on. `env_params.preset = "event"` applies them; any key the task sets
# itself wins. Plain tasks keep the old defaults (10 seats, talk off).
EVENT_PARAMS: dict[str, Any] = {
    "seats": 6,
    "table_talk": True,
    "reset_stacks": False,
    "starting_stack": 200,
    "small_blind": 1,
    "big_blind": 2,
    "total_hands": 30,
    "ranking": "elimination",
    "strict_actions": True,
    "scratchpad": True,
    "scratchpad_chars": 5000,
    "max_completion_tokens": 5000,
    "cleanup_max_completion_tokens": 10000,
    "thinking": True,
}
PRESETS: dict[str, dict[str, Any]] = {"event": EVENT_PARAMS}


def with_preset(params: dict[str, Any]) -> dict[str, Any]:
    """`params` over its named preset's values (`env_params.preset`); unchanged without one."""
    name = params.get("preset")
    if name is None:
        return params
    if name not in PRESETS:
        raise ValueError(f"unknown env_params.preset {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name] | params


RETRYABLE_STATUS = {408, 409, 425, 429}
DEFAULT_DECISION_TIMEOUT_SECONDS = 300.0
GATEWAY_URL = "https://inference.appliedcompute.com/v1"


class RelayError(Exception):
    pass


def relay_completion(client: httpx.AsyncClient, base_url: str, api_key: str, params: dict[str, Any], retries: int = 2):
    """A `complete(messages)` that sends the seat's conversation as one chat completion to `base_url`."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    passthrough = ("reasoning_effort", "max_completion_tokens", "max_tokens", "temperature", "chat_template_kwargs")
    tools = _tools(params)
    extra = {k: params[k] for k in passthrough if k in params}
    if "thinking" in params and "chat_template_kwargs" not in params:  # Qwen-style chat templates' switch
        extra["chat_template_kwargs"] = {"enable_thinking": bool(params["thinking"])}

    async def complete(messages: list[dict[str, Any]]) -> Completion:
        # The relay serves the eval's configured model; this field only labels the request.
        body = {
            "model": params.get("model_label", "policy"),
            "messages": messages,
            "tools": tools,
            **extra,
        }
        for attempt in range(retries + 1):
            try:
                response = await client.post(url, json=body, headers=headers)
            except httpx.TransportError:
                if attempt == retries:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if response.is_success:
                break
            error = RelayError(f"relay returned HTTP {response.status_code}: {response.text[:300]}")
            retryable = response.status_code in RETRYABLE_STATUS or response.status_code >= 500
            if not retryable or attempt == retries:
                logger.warning("%s", error)
                raise error
            await asyncio.sleep(2**attempt)
        data = response.json()
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        message = data["choices"][0]["message"]
        raw_calls = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]},
            }
            for c in message.get("tool_calls") or []
        ]
        calls = [c["function"] for c in raw_calls if c["function"]["name"] == SUBMIT_ACTION]
        return Completion(
            text=message.get("content") or "",
            tool_arguments=calls[-1]["arguments"] if calls else None,
            tool_calls=raw_calls,
            in_tokens=usage.get("prompt_tokens"),
            out_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            truncated=data["choices"][0].get("finish_reason") == "length",
        )

    return complete


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat messages (system, user, assistant with tool_calls, tool) as Responses API input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}
            )
            continue
        if message.get("content"):
            items.append({"role": role, "content": message["content"]})
        for call in message.get("tool_calls") or []:
            function = call["function"]
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            )
    return items


def responses_completion(
    client: httpx.AsyncClient, base_url: str, api_key: str, params: dict[str, Any], retries: int = 2
):
    """A `complete(messages)` for models that need the Responses API for tools with reasoning, such as OpenAI reasoning models."""
    url = f"{base_url.rstrip('/')}/responses"
    headers = {"Authorization": f"Bearer {api_key}"}
    tools = [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
        }
        for tool in _tools(params)
    ]
    extra: dict[str, Any] = (
        {"reasoning": {"effort": params["reasoning_effort"]}} if "reasoning_effort" in params else {}
    )
    if "max_completion_tokens" in params:  # the Responses API's name for the same cap
        extra["max_output_tokens"] = int(params["max_completion_tokens"])

    async def complete(messages: list[dict[str, Any]]) -> Completion:
        body = {"model": params["model_label"], "input": to_responses_input(messages), "tools": tools, **extra}
        for attempt in range(retries + 1):
            try:
                response = await client.post(url, json=body, headers=headers)
            except httpx.TransportError:
                if attempt == retries:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if response.is_success:
                break
            error = RelayError(f"responses endpoint returned HTTP {response.status_code}: {response.text[:300]}")
            if not (response.status_code in RETRYABLE_STATUS or response.status_code >= 500) or attempt == retries:
                logger.warning("%s", error)
                raise error
            await asyncio.sleep(2**attempt)
        data = response.json()
        text = "".join(
            part.get("text", "")
            for item in data.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        calls = [
            {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            for item in data.get("output", [])
            if item.get("type") == "function_call"
        ]
        submits = [c["function"] for c in calls if c["function"]["name"] == SUBMIT_ACTION]
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            tool_arguments=submits[-1]["arguments"] if submits else None,
            tool_calls=calls,
            in_tokens=usage.get("input_tokens"),
            out_tokens=usage.get("output_tokens"),
            reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            truncated=data.get("status") == "incomplete"
            and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens",
        )

    return complete


def _tools(params: dict[str, Any]) -> list[dict[str, Any]]:
    limit = int(params.get("scratchpad_chars", 5000))
    return table_tools(params.get("submit_fields"), bool(params.get("scratchpad")), limit)


def cleanup_params(params: dict[str, Any]) -> dict[str, Any]:
    """A seat's completion params for the after-hand cleanup turn: the cleanup cap instead of the action cap."""
    cap = params.get("cleanup_max_completion_tokens")
    return params | {"max_completion_tokens": cap} if cap else params


def table_rules(params: dict[str, Any]) -> TableRules:
    return TableRules(
        total_hands=int(params.get("total_hands", 3)),
        small_blind=int(params.get("small_blind", 1)),
        big_blind=int(params.get("big_blind", 2)),
        starting_stack=int(params.get("starting_stack", 200)),
        reset_stacks=bool(params.get("reset_stacks", True)),
        table_talk=bool(params.get("table_talk", False)),
        seats=int(params.get("seats", 10)),
        max_completion_tokens=int(params["max_completion_tokens"]) if params.get("max_completion_tokens") else None,
        ranking=params.get("ranking"),
        strict_actions=bool(params.get("strict_actions", False)),
        scratchpad=bool(params.get("scratchpad", False)),
        scratchpad_chars=int(params.get("scratchpad_chars", 5000)),
        cleanup_max_completion_tokens=(
            int(params["cleanup_max_completion_tokens"]) if params.get("cleanup_max_completion_tokens") else None
        ),
    )


def rules_output(rules: TableRules) -> dict[str, Any]:
    """The rules as reported to the grader. `table_talk` and `seats` appear only when not the default."""
    defaults = {
        "table_talk": False,
        "seats": 10,
        "max_completion_tokens": None,
        "ranking": None,
        "strict_actions": False,
        "scratchpad": False,
        "scratchpad_chars": 5000,
        "cleanup_max_completion_tokens": None,
    }  # left out at their defaults, so 10-max talk-off outputs are unchanged
    return {k: v for k, v in asdict(rules).items() if k not in defaults or v != defaults[k]}


def opponent_params(params: dict[str, Any]) -> dict[str, Any]:
    """Opponent seats call `opponent_model` (an AC2 inference deployment or external endpoint) on the gateway.

    `opponent_thinking` sets Qwen's chat-template thinking switch and is sent only when the task sets it;
    `submit_fields` trims the submit_action tool for every seat.
    """
    opponent: dict[str, Any] = {"model_label": params["opponent_model"]}
    if "submit_fields" in params:
        opponent["submit_fields"] = params["submit_fields"]
    if "opponent_reasoning_effort" in params:
        opponent["reasoning_effort"] = params["opponent_reasoning_effort"]
    if "opponent_thinking" in params:
        opponent["chat_template_kwargs"] = {"enable_thinking": bool(params["opponent_thinking"])}
    if "thinking" in params and "opponent_thinking" not in params:
        opponent["chat_template_kwargs"] = {"enable_thinking": bool(params["thinking"])}
    # the table's rules bind every seat
    for key in ("max_completion_tokens", "cleanup_max_completion_tokens", "scratchpad", "scratchpad_chars"):
        if key in params:
            opponent[key] = params[key]
    return opponent
