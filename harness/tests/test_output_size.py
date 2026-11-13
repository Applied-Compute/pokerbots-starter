"""A full scratchpad game's collect_outputs fits through the connector (1 MiB) with a margin."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from poker_harness.server import MAX_OUTPUT_BYTES, _bounded, _size, create_app

CONNECTOR_LIMIT = 1024 * 1024


def verbose_relay() -> FastAPI:
    """Worst case: every reply carries ~13.5k characters of text, and every cleanup writes a 4,990-character pad."""
    relay = FastAPI()
    calls = [0]

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        calls[0] += 1
        last_user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
        if "is over" in last_user:
            name, arguments = "update_scratchpad", {"contents": f"{calls[0]:06d}" + "n" * 4984}
        else:
            name, arguments = "submit_action", {"action": "call"}
        function = {"name": name, "arguments": json.dumps(arguments)}
        message = {
            "role": "assistant",
            "content": "r" * 13_500,
            "tool_calls": [{"id": "c", "type": "function", "function": function}],
        }
        return {"choices": [{"message": message}], "usage": {"completion_tokens": 4000}}

    return relay


async def outputs_of(app: FastAPI, params: dict) -> tuple[dict, int]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        body = {
            "ac_api_url": "http://relay/job/trace/v1",
            "ac_api_key": "k",
            "task": {"input": [], "env_params": params},
        }
        rollout_id = (await client.post("/submit", json=body)).json()["rollout_id"]
        for _ in range(3000):
            status = (await client.post("/get_status", json={"rollout_id": rollout_id})).json()
            if status["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.05)
        assert status["status"] == "completed", status
        response = await client.post("/collect_outputs", json={"rollout_id": rollout_id})
    return response.json(), len(response.content)


@pytest.mark.parametrize("hands", [30, 60])
async def test_a_full_scratchpad_game_fits_through_the_connector(tmp_path: Path, hands: int) -> None:
    app = create_app(relay_transport=httpx.ASGITransport(app=verbose_relay()), runs_dir=tmp_path)
    outputs, size = await outputs_of(app, {"preset": "event", "seed": 3, "total_hands": hands})  # every seat a policy
    log = outputs["log"]
    assert len(log["hands"]) == hands and size < MAX_OUTPUT_BYTES < CONNECTOR_LIMIT, size
    for seat in log["private"].values():  # compact: the final pad and outcome counts, no per-hand history
        assert set(seat) <= {"scratchpad", "scratchpad_chars", "cleanups"} and seat["cleanups"]["turns"] >= hands
        assert seat["scratchpad_chars"] == 4990
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())  # the full private history stays on disk
    first = next(iter(stored["private"].values()))["hands"]["1"]
    assert first["cleanup"]["attempts"][0]["update_calls"] and len(first["scratchpad_after"]) == 4990


def test_an_oversized_log_is_trimmed_in_stages_until_it_fits() -> None:
    """Far past what any game produces: 5,000 actions with large responses still come back under the limit."""
    response = {
        "model_name": "m",
        "format_error": False,
        "error": "e" * 1000,
        "latency_seconds": 1.0,
        "notes": "x" * 500,
    }
    action = {
        "timestep": 0,
        "player": "seat0",
        "seat": 0,
        "street": "flop",
        "action": {"type": "call"},
        "board": ["Ah"] * 5,
    }
    hands = [
        {"hand_number": n, "actions": [action | {"model_response": dict(response)} for _ in range(50)]}
        for n in range(100)
    ]
    log = {"hands": hands, "private": {"seat0": {"scratchpad": "p" * 5000, "hands": {}}}}
    trimmed = _bounded(log)
    assert _size(trimmed) <= MAX_OUTPUT_BYTES and len(trimmed["hands"]) == 100
    assert trimmed["hands"][0]["actions"][0]["model_response"] == {"format_error": False}
    assert trimmed["private"]["seat0"]["scratchpad_chars"] == 5000 and "hands" not in trimmed["private"]["seat0"]
    small = {"hands": [{"hand_number": 1, "actions": [action | {"model_response": {"reasoning": "kept"}}]}]}
    assert _bounded(small) is small  # research logs that fit come back unchanged
