"""The Vals text parser: read an action from the last JSON object in a reply, for replies without a tool call.

Ported from Vals' poker-agent engine (poker-agent-benchmark-service 0.1.0) and used with Vals' permission; behavior is
unchanged, and tests/test_engine_parity.py checks it hand for hand.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from .table import ActionType, Decision

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")
MAX_TEXT = 500
_ACTIONS = ("fold", "check", "call", "bet", "raise")


def parse_decision(text: str) -> tuple[Decision, str | None]:
    """Parse the last JSON action object in a model reply. An unparseable reply is a fold."""
    for match in reversed(_JSON_OBJECT.findall(text)):
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            continue
        kind = str(payload.get("action", "")).strip().lower()
        if kind not in _ACTIONS:
            continue
        try:
            amount = round(float(payload["amount"])) if payload.get("amount") is not None else None
        except (TypeError, ValueError, OverflowError):
            amount = None
        response: dict[str, Any] = {
            key: value[:MAX_TEXT] if isinstance(value := payload.get(key), str) else ""
            for key in ("reasoning", "say", "read")
        }
        p_win = payload.get("p_win")
        if isinstance(p_win, int | float) and not isinstance(p_win, bool) and 0 <= p_win <= 1:
            response["p_win"] = float(p_win)
        return Decision(cast(ActionType, kind), amount, response), None
    return Decision("fold"), "no valid action JSON in reply"
