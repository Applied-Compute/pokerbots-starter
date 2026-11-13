"""The event's rules (6-max, table talk on, 200 chips every hand, 5,000 output tokens per decision) as an opt-in preset."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from test_server import fake_relay, fake_responses, play

from poker_harness.completions import responses_completion, rules_output, table_rules, with_preset
from poker_harness.prompts import TALK_INVITE, system_prompt
from poker_harness.server import create_app


def test_the_event_preset_sets_the_event_rules_and_explicit_keys_win() -> None:
    rules = table_rules(with_preset({"seed": 1, "preset": "event"}))
    assert (rules.seats, rules.table_talk, rules.reset_stacks, rules.max_completion_tokens) == (6, True, True, 5000)
    prompt = system_prompt(rules)
    assert "six-seat" in prompt and "numbered 0-5" in prompt and TALK_INVITE in prompt
    assert "Every hand starts with every player at 200 chips" in prompt  # cash-game stacks
    assert "at most 5000 output tokens" in prompt and "played as check or fold" in prompt
    assert table_rules(with_preset({"preset": "event", "seats": 10})).seats == 10
    with pytest.raises(ValueError, match="unknown env_params.preset"):
        with_preset({"preset": "tournament"})


def test_tasks_without_a_preset_keep_the_old_defaults() -> None:
    rules = table_rules(with_preset({"seed": 1}))
    assert (rules.seats, rules.table_talk, rules.reset_stacks, rules.max_completion_tokens) == (10, False, True, None)
    assert "output tokens" not in system_prompt(rules) and "max_completion_tokens" not in rules_output(rules)


async def test_an_event_table_caps_every_seat_and_resets_stacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    relay_requests: list[dict] = []
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay(relay_requests)),
        opponent_transport=httpx.ASGITransport(app=fake_responses(opponent_requests)),
        runs_dir=tmp_path,
    )
    params = {
        "preset": "event",
        "seed": 21,
        "total_hands": 3,
        "policy_seats": [1],
        "opponent_model": "my-project/opponent",
        "opponent_api": "responses",
    }
    status, outputs = await play(app, {"input": [], "env_params": params})
    hands = outputs["log"]["hands"]
    assert status["status"] == "completed" and len(hands) == 3 and len(hands[0]["starting_state"]) == 6
    rules = {k: outputs["rules"][k] for k in ("seats", "table_talk", "reset_stacks", "max_completion_tokens")}
    assert rules == {"seats": 6, "table_talk": True, "reset_stacks": True, "max_completion_tokens": 5000}
    assert relay_requests and all(r["body"]["max_completion_tokens"] == 5000 for r in relay_requests)
    assert opponent_requests and all(r["max_output_tokens"] == 5000 for r in opponent_requests)
    stacks = [{v["starting_stack"] for v in h["starting_state"].values()} for h in hands]
    assert stacks == [{200}] * 3, "every hand starts at 200 chips"


def capped_relay(requests: list[dict]) -> FastAPI:
    """A relay whose replies end at the token cap, though each still carries a complete submit_action call."""
    relay = FastAPI()

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        requests.append(await request.json())
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "submit_action", "arguments": '{"action": "raise", "amount": 20}'},
        }
        message = {"role": "assistant", "content": "Long reasoning ...", "tool_calls": [call]}
        return {"choices": [{"message": message, "finish_reason": "length"}], "usage": {"completion_tokens": 5000}}

    return relay


async def test_a_reply_cut_at_the_cap_is_a_format_error_even_with_an_action(tmp_path: Path) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=capped_relay(requests)), runs_dir=tmp_path)
    params = {"preset": "event", "seed": 4, "total_hands": 1}  # every seat on the capped relay
    status, outputs = await play(app, {"input": [], "env_params": params})
    actions = outputs["log"]["hands"][0]["actions"]
    assert status["status"] == "completed" and actions and requests[0]["max_completion_tokens"] == 5000
    for action in actions:
        response = action["model_response"]
        assert response["format_error"] and response["truncated"] and not response["submitted_by_tool"]
        assert response["error"] == "reply hit the 5000-token cap" and action["action"]["type"] in ("check", "fold")


async def test_a_responses_reply_cut_at_the_cap_is_marked_truncated() -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        call = {"type": "function_call", "call_id": "f1", "name": "submit_action", "arguments": '{"action": "call"}'}
        body = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [call]}
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        complete = responses_completion(
            client, "http://gw/v1", "k", {"model_label": "opponent", "max_completion_tokens": 5000}
        )
        reply = await complete([{"role": "user", "content": "act"}])
    assert reply.truncated and reply.tool_arguments is not None and sent[0]["max_output_tokens"] == 5000
