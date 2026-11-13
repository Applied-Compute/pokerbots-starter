import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from poker_harness.completions import rules_output, table_rules
from poker_harness.prompts import submit_action_tool, system_prompt
from poker_harness.server import create_app

TASK = {"input": [{"role": "user", "content": "play"}], "env_params": {"seed": 123, "total_hands": 2}}


def fake_relay(requests: list[dict], reply: str | None = None) -> FastAPI:
    relay = FastAPI()

    @relay.post("/job/trace/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        requests.append({"body": body, "auth": request.headers.get("authorization")})
        assert body["tools"][0]["function"]["name"] == "submit_action"
        if reply is not None:  # a reply with no tool call
            message = {"role": "assistant", "content": reply}
        else:
            arguments = json.dumps({"action": "call", "p_win": 0.2, "say": "fold now"})
            message = {
                "role": "assistant",
                "content": "Pot odds are fine, so I call.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": arguments}}
                ],
            }
        return {
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": 600,
                "completion_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        }

    return relay


async def play(app: FastAPI, task: dict = TASK) -> tuple[dict, dict]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://harness") as client:
        body = {"ac_api_url": "http://relay/job/trace/v1", "ac_api_key": "relay-key", "task": task}
        submitted = await client.post("/submit", json=body, headers={"Idempotency-Key": "gw_1"})
        rollout_id = submitted.json()["rollout_id"]
        again = await client.post("/submit", json=body, headers={"Idempotency-Key": "gw_1"})
        assert again.json()["rollout_id"] == rollout_id
        for _ in range(500):
            status = (await client.post("/get_status", json={"rollout_id": rollout_id})).json()
            if status["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.05)
        outputs = (await client.post("/collect_outputs", json={"rollout_id": rollout_id})).json()
    return status, outputs


async def test_table_plays_every_seat_through_the_relay(tmp_path: Path) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    status, outputs = await play(app)
    log = outputs["log"]
    actions = [a for h in log["hands"] for a in h["actions"]]

    assert status == {"status": "completed", "annotations": {"hands_played": 2}}
    assert log["complete"] and len(log["hands"]) == 2 and outputs["rules"]["reset_stacks"]
    assert len(requests) == len(actions) and {r["auth"] for r in requests} == {"Bearer relay-key"}
    assert {a["player"] for a in actions} <= {f"seat{i}" for i in range(10)}
    assert not any(a["model_response"]["fallback_used"] for a in actions)
    assert all(a["say"] == "" for a in actions) and not any(
        "fold now" in r["body"]["messages"][1]["content"] for r in requests
    )
    assert actions[0]["model_response"]["in_tokens"] == 600 and actions[0]["model_response"]["reasoning_tokens"] == 30
    assert all(a["model_response"]["submitted_by_tool"] for a in actions)
    assert actions[0]["model_response"]["reasoning"] == "Pot odds are fine, so I call."
    saved = json.loads((tmp_path / f"{log['game_id']}.json").read_text())
    assert saved["relay_job_id"] is None  # the fake relay URL names no job


async def test_unparseable_replies_fall_back_but_finish(tmp_path: Path) -> None:
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([], reply="I fold?")), runs_dir=tmp_path)
    status, outputs = await play(app)
    actions = [a for h in outputs["log"]["hands"] for a in h["actions"]]
    assert status["status"] == "completed" and all(a["model_response"]["fallback_used"] for a in actions)
    assert all(a["model_response"]["format_error"] for a in actions)


async def test_missing_seed_is_rejected(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        body = {"ac_api_url": "http://relay/v1", "ac_api_key": "k", "task": {"input": [], "env_params": {}}}
        assert (await client.post("/submit", json=body)).status_code == 422


async def test_token_is_required_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POKER_HARNESS_TOKEN", "secret")
    app = create_app(runs_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness") as client:
        assert (await client.post("/get_status", json={"rollout_id": "x"})).status_code == 401
        authed = await client.post("/get_status", json={"rollout_id": "x"}, headers={"Authorization": "Bearer secret"})
        assert authed.status_code == 404


async def test_policy_seat_uses_relay_and_opponents_use_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    relay_requests: list[dict] = []
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay(relay_requests)),
        opponent_transport=httpx.ASGITransport(app=fake_relay(opponent_requests)),
        runs_dir=tmp_path,
    )
    params = {
        "seed": 99,
        "total_hands": 1,
        "policy_seats": [3],
        "opponent_model": "vals-poker/vanilla",
        "opponent_thinking": False,
    }
    status, outputs = await play(app, {"input": [], "env_params": params})
    actions = outputs["log"]["hands"][0]["actions"]
    policy_actions = [a for a in actions if a["player"] == "seat3"]

    assert status["status"] == "completed" and outputs["policy_seats"] == ["seat3"]
    assert len(relay_requests) == len(policy_actions) >= 1
    assert len(opponent_requests) == len(actions) - len(policy_actions)
    assert {r["auth"] for r in relay_requests} == {"Bearer relay-key"}
    assert {r["auth"] for r in opponent_requests} == {"Bearer gateway-key"}
    assert all(r["body"]["model"] == "vals-poker/vanilla" for r in opponent_requests)
    assert all(r["body"]["chat_template_kwargs"] == {"enable_thinking": False} for r in opponent_requests)
    assert all(a["model_response"]["provider"] == "ac2" for a in policy_actions)


async def test_opponents_without_a_gateway_key_fail_the_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AC2_API_KEY", raising=False)
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)
    params = {"seed": 1, "total_hands": 1, "policy_seats": [0], "opponent_model": "vals-poker/vanilla"}
    status, _ = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "failed" and "AC2_API_KEY" in status["error_message"]


async def test_opponent_thinking_switch_is_only_sent_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    opponent_requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay([])),
        opponent_transport=httpx.ASGITransport(app=fake_relay(opponent_requests)),
        runs_dir=tmp_path,
    )
    params = {"seed": 7, "total_hands": 1, "policy_seats": [0], "opponent_model": "my-project/opponent"}
    status, _ = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed" and opponent_requests
    assert all("chat_template_kwargs" not in r["body"] for r in opponent_requests)


def fake_responses(requests: list[dict]) -> FastAPI:
    api = FastAPI()

    @api.post("/job/trace/v1/responses")
    async def responses(request: Request) -> dict:
        body = await request.json()
        requests.append(body)
        assert body["tools"][0]["name"] == "submit_action"
        n = len(requests)
        return {
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [{"type": "output_text", "text": "Calling."}]},
                {
                    "type": "function_call",
                    "call_id": f"fc{n}",
                    "name": "submit_action",
                    "arguments": '{"action": "call"}',
                },
            ],
            "usage": {"input_tokens": 700, "output_tokens": 90, "output_tokens_details": {"reasoning_tokens": 60}},
        }

    return api


async def test_responses_opponents_carry_the_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AC2_API_KEY", "gateway-key")
    monkeypatch.setenv("POKER_OPPONENT_BASE_URL", "http://relay/job/trace/v1")
    requests: list[dict] = []
    app = create_app(
        relay_transport=httpx.ASGITransport(app=fake_relay([])),
        opponent_transport=httpx.ASGITransport(app=fake_responses(requests)),
        runs_dir=tmp_path,
    )
    params = {
        "seed": 99,
        "total_hands": 1,
        "policy_seats": [3],
        "opponent_model": "my-project/opponent",
        "opponent_api": "responses",
    }
    status, outputs = await play(app, {"input": [], "env_params": params})
    opponents = [a for a in outputs["log"]["hands"][0]["actions"] if a["player"] != "seat3"]
    assert status["status"] == "completed" and len(requests) == len(opponents)
    assert all(a["model_response"]["submitted_by_tool"] and not a["model_response"]["format_error"] for a in opponents)
    continued = [r for r in requests if any(i.get("type") == "function_call_output" for i in r["input"])]
    assert continued, "an opponent acting twice should send its earlier call and the table's answer back"
    first = continued[0]["input"]
    call = next(i for i in first if i.get("type") == "function_call")
    answer = next(i for i in first if i.get("type") == "function_call_output")
    assert answer["call_id"] == call["call_id"] and answer["output"].startswith("Applied:")


async def test_submit_fields_trim_the_tool_for_every_seat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        "seed": 99,
        "total_hands": 1,
        "policy_seats": [3],
        "opponent_model": "my-project/opponent",
        "opponent_api": "responses",
        "submit_fields": ["action", "amount"],
    }
    status, _ = await play(app, {"input": [], "env_params": params})
    assert status["status"] == "completed" and relay_requests and opponent_requests
    assert all(
        set(r["body"]["tools"][0]["function"]["parameters"]["properties"]) == {"action", "amount"}
        for r in relay_requests
    )
    assert all(set(r["tools"][0]["parameters"]["properties"]) == {"action", "amount"} for r in opponent_requests)


def test_submit_action_tool_defaults_to_every_field_and_rejects_unknown_ones() -> None:
    assert set(submit_action_tool()["function"]["parameters"]["properties"]) == {"action", "amount", "p_win", "read"}
    with pytest.raises(ValueError):
        submit_action_tool(["amount"])


async def test_ev_labels_are_added_only_when_asked(tmp_path: Path) -> None:
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)
    params = {"seed": 5, "total_hands": 1, "policy_seats": [2], "ev_labels": True}
    status, outputs = await play(app, {"input": [], "env_params": params})
    hand = outputs["log"]["hands"][0]
    assert status["status"] == "completed" and list(hand["ev"]) == ["seat2"]
    assert {"ev_bb", "realized_bb", "method"} <= set(hand["ev"]["seat2"])

    plain = {"seed": 5, "total_hands": 1, "policy_seats": [2]}
    _, outputs = await play(
        create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path),
        {"input": [], "env_params": plain},
    )
    assert "ev" not in outputs["log"]["hands"][0]


async def test_a_five_seat_table_plays_and_labels(tmp_path: Path) -> None:
    """Any table size from 2 to 10 works: five seats, the prompt says so, and EV labels still work."""
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    params = {"seed": 7, "total_hands": 1, "seats": 5, "policy_seats": [3], "ev_labels": True}
    status, outputs = await play(app, {"input": [], "env_params": params})
    log, hand = outputs["log"], outputs["log"]["hands"][0]
    assert status["status"] == "completed" and len(hand["starting_state"]) == 5 and hand["button_seat"] == 4
    assert outputs["rules"]["seats"] == 5
    system = requests[0]["body"]["messages"][0]["content"]
    assert "five-seat" in system and "numbered 0-4" in system
    assert "seat3" in hand["ev"]
    assert log["session_info"]["player_names"] == [f"seat{i}" for i in range(5)]


async def test_a_six_seat_table_is_the_event_format(tmp_path: Path) -> None:
    """6-max, the event: six seats, every seat's prompt says six seats numbered 0-5 (no ten-seat text), labels work."""
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    params = {"seed": 11, "total_hands": 1, "seats": 6, "policy_seats": [2], "ev_labels": True}
    status, outputs = await play(app, {"input": [], "env_params": params})
    hand = outputs["log"]["hands"][0]
    assert status["status"] == "completed" and len(hand["starting_state"]) == 6 and hand["button_seat"] == 5
    assert outputs["rules"]["seats"] == 6 and outputs["log"]["session_info"]["player_names"] == [
        f"seat{i}" for i in range(6)
    ]
    for request in requests:
        system = request["body"]["messages"][0]["content"]
        assert "six-seat" in system and "numbered 0-5" in system
        assert "ten-seat" not in system and "0-9" not in system and "nine" not in system
    assert "seat2" in hand["ev"]


def test_ten_seat_rules_report_and_prompt_as_before() -> None:
    rules = table_rules({})
    assert "seats" not in rules_output(rules) and "ten-seat" in system_prompt(rules) and "0-9" in system_prompt(rules)
