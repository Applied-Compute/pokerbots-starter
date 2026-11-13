"""The official rules as the "event" preset, end to end through the harness server."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from test_server import fake_responses, play

from poker_harness.completions import EVENT_PARAMS, responses_completion, rules_output, table_rules, with_preset
from poker_harness.prompts import TALK_INVITE, system_prompt, table_tools, talk_fields
from poker_harness.server import create_app


def test_the_event_preset_is_the_official_rules_and_explicit_keys_win() -> None:
    rules = table_rules(with_preset({"seed": 1, "preset": "event"}))
    assert (rules.seats, rules.table_talk, rules.reset_stacks, rules.starting_stack) == (6, True, False, 200)
    assert (rules.small_blind, rules.big_blind, rules.total_hands, rules.ranking) == (1, 2, 30, "elimination")
    assert rules.strict_actions and rules.scratchpad and rules.scratchpad_chars == 5000
    assert (rules.max_completion_tokens, rules.cleanup_max_completion_tokens) == (5000, 10000)
    assert EVENT_PARAMS["thinking"] is True
    prompt = system_prompt(rules)
    assert "six-seat" in prompt and "numbered 0-5" in prompt and TALK_INVITE in prompt and "Blinds are 1/2." in prompt
    assert "a player with zero chips is out" in prompt and "The game lasts 30 hands" in prompt
    assert "played as check if checking is legal, else fold, with no retry" in prompt
    assert "private scratchpad" in prompt and "up to 10000 output tokens" in prompt and "at most 5000 output" in prompt
    assert table_rules(with_preset({"preset": "event", "seats": 10})).seats == 10
    with pytest.raises(ValueError, match="unknown env_params.preset"):
        with_preset({"preset": "tournament"})


def test_tasks_without_a_preset_keep_the_old_defaults() -> None:
    rules = table_rules(with_preset({"seed": 1}))
    assert (rules.seats, rules.table_talk, rules.reset_stacks, rules.max_completion_tokens) == (10, False, True, None)
    assert not (rules.scratchpad or rules.strict_actions or rules.ranking)
    assert "output tokens" not in system_prompt(rules) and "scratchpad" not in system_prompt(rules)
    assert set(rules_output(rules)) == {"total_hands", "small_blind", "big_blind", "starting_stack", "reset_stacks"}


def recording_relay(requests: list[dict]) -> FastAPI:
    """Submits call (a check when free) on action turns and writes a short note on cleanup turns."""
    relay = FastAPI()

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        requests.append(body)
        last_user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
        if "is over" in last_user:
            name, arguments = "update_scratchpad", {"contents": f"note {len(requests)}"}
        else:
            name, arguments = "submit_action", {"action": "call"}
        function = {"name": name, "arguments": json.dumps(arguments)}
        call = {"id": f"c{len(requests)}", "type": "function", "function": function}
        return {"choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": [call]}}], "usage": {}}

    return relay


async def test_an_event_table_carries_chips_and_caps_action_and_cleanup_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    relay_requests: list[dict] = []
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=recording_relay(relay_requests)),
        opponent_transport=httpx.ASGITransport(app=fake_responses(opponent_requests)),
        runs_dir=tmp_path,
    )
    params = {
        "preset": "event",
        "seed": 21,
        "total_hands": 2,
        "policy_seats": [1],
        "opponent_model": "my-project/opponent",
        "opponent_api": "responses",
    }
    status, outputs = await play(app, {"input": [], "env_params": params})
    log, hands = outputs["log"], outputs["log"]["hands"]
    assert status["status"] == "completed" and len(hands) == 2 and len(hands[0]["starting_state"]) == 6
    rules = outputs["rules"]
    assert (rules["seats"], rules["table_talk"], rules["reset_stacks"], rules["scratchpad"]) == (6, True, False, True)
    assert hands[1]["stacks_before"] == {n: s for n, s in hands[0]["ending_stacks"].items() if s > 0}  # carry-over
    assert log["ranking"]["mode"] == "elimination" and sum(log["net_chips"].values()) == 0
    cleanups = [b for b in relay_requests if "is over" in b["messages"][-1]["content"]]
    actions = [b for b in relay_requests if b not in cleanups]
    assert actions and all(b["max_completion_tokens"] == 5000 for b in actions)
    assert len(cleanups) == 2 and all(b["max_completion_tokens"] == 10000 for b in cleanups)  # seat1, both hands
    assert all(b["chat_template_kwargs"] == {"enable_thinking": True} for b in relay_requests)
    tools = [[t["function"]["name"] for t in b["tools"]] for b in relay_requests]
    assert all(names == ["submit_action", "update_scratchpad"] for names in tools)
    assert {r["max_output_tokens"] for r in opponent_requests} == {5000, 10000}
    assert all(len(r["tools"]) == 2 for r in opponent_requests)
    assert log["private"]["seat1"]["scratchpad"].startswith("note ")


def capped_relay(requests: list[dict]) -> FastAPI:
    """A relay whose replies end at the token cap, though each still carries a complete call."""
    relay = FastAPI()

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        requests.append(await request.json())
        arguments = '{"action": "raise", "amount": 20}'
        call = {"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": arguments}}
        message = {"role": "assistant", "content": "Long reasoning ...", "tool_calls": [call]}
        return {"choices": [{"message": message, "finish_reason": "length"}], "usage": {"completion_tokens": 5000}}

    return relay


async def test_a_reply_cut_at_the_cap_is_played_afk_even_with_an_action(tmp_path: Path) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=capped_relay(requests)), runs_dir=tmp_path)
    params = {"preset": "event", "seed": 4, "total_hands": 1}  # every seat on the capped relay
    status, outputs = await play(app, {"input": [], "env_params": params})
    actions = outputs["log"]["hands"][0]["actions"]
    assert status["status"] == "completed" and actions and requests[0]["max_completion_tokens"] == 5000
    for action in actions:
        response = action["model_response"]
        assert response["format_error"] and response["truncated"] and response["afk"]
        assert not response["submitted_by_tool"] and action["action"]["type"] in ("check", "fold")
        assert response["error"] == "reply hit the 5000-token cap"
    for seat in outputs["log"]["private"].values():  # cleanup replies were cut too: nothing changes
        assert seat["scratchpad"] == "" and seat["cleanups"] == {
            "turns": 1,
            "updated": 0,
            "cut": 0,
            "unchanged": 1,
            "retried": 0,
        }


async def test_a_responses_reply_cut_at_the_cap_is_marked_truncated() -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        call = {"type": "function_call", "call_id": "f1", "name": "submit_action", "arguments": '{"action": "call"}'}
        body = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [call]}
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        params = {"model_label": "opponent", "max_completion_tokens": 5000}
        reply = await responses_completion(client, "http://gw/v1", "k", params)([{"role": "user", "content": "act"}])
    assert reply.truncated and reply.tool_arguments is not None and sent[0]["max_output_tokens"] == 5000


def test_the_opsd_builder_lists_the_tools_the_harness_gives_every_seat() -> None:
    """opsd/tools.json (what the OPSD builder puts in env_params.tools) must match the event table's tools exactly."""
    recorded = json.loads((Path(__file__).resolve().parents[2] / "opsd" / "tools.json").read_text())
    rules = table_rules(with_preset({"preset": "event", "seed": 1}))
    tools = table_tools(talk_fields(["action", "amount"]), scratchpad=rules.scratchpad, limit=rules.scratchpad_chars)
    assert recorded == [{"type": "function", **t["function"]} for t in tools]
