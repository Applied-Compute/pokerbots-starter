import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from test_server import TASK, fake_relay, play

from poker_harness.grading import (
    FAILURE_SCORE,
    MAX_SCORE_BYTES,
    REWARD_OFFSET,
    GradingError,
    bounded,
    check_grader_params,
    grade_log,
)
from poker_harness.server import create_app


def harness(tmp_path: Path) -> FastAPI:
    return create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)


async def grade(client: httpx.AsyncClient, body: dict) -> tuple[dict, httpx.Response]:
    headers = {"Idempotency-Key": f"{body['rollout_id']}:grading"}
    assert (await client.post("/submit_grading", json=body, headers=headers)).is_success
    for _ in range(200):
        status = (await client.post("/get_grading_status", json={"rollout_id": body["rollout_id"]})).json()
        if status["status"] not in ("queued", "running"):
            break
        await asyncio.sleep(0.02)
    return status, await client.post("/collect_score", json={"rollout_id": body["rollout_id"]})


async def played(app: FastAPI) -> str:
    """Play TASK as gateway rollout gw_1; the harness keeps its id."""
    await play(app)
    return "gw_1"


def grading_body(rollout_id: str, status: str = "completed", **task: object) -> dict:
    return {"rollout_id": rollout_id, "rollout_status": status, "task": {"input": TASK["input"], **task}}


async def test_graded_table_scores_and_replays_are_idempotent(tmp_path: Path) -> None:
    app = harness(tmp_path)
    rollout_id = await played(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        body = grading_body(rollout_id, env_params=TASK["env_params"])
        status, score = await grade(client, body)
        replay_status, replay_score = await grade(client, body)  # AC2 replays submit_grading after a hiccup

    assert status["status"] == "completed" and replay_status == status
    reply = score.json()
    assert math.isfinite(reply["score"]) and replay_score.json() == reply
    # Every seat is a policy seat in TASK, so their chips net to zero.
    assert reply["score"] == REWARD_OFFSET and reply["artifacts"]["ev_method"] == "realized"
    assert status["annotations"] == {"score": REWARD_OFFSET}


async def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    app = harness(tmp_path)
    rollout_id = await played(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        extra_body = grading_body(rollout_id) | {"surprise": 1}
        assert (await client.post("/submit_grading", json=extra_body)).status_code == 422
        extra_task = grading_body(rollout_id, oops=True)
        assert (await client.post("/submit_grading", json=extra_task)).status_code == 422
        status_extra = {"rollout_id": rollout_id, "x": 1}
        assert (await client.post("/get_grading_status", json=status_extra)).status_code == 422
        wrong_key = {"Idempotency-Key": "other:grading"}
        assert (
            await client.post("/submit_grading", json=grading_body(rollout_id), headers=wrong_key)
        ).status_code == 422
        # Unknown grader params fail the grading (AC2 drops the sample) rather than score with defaults.
        status, score = await grade(client, grading_body(rollout_id, grader_params={"lambda": 1.0}))
    assert status["status"] == "failed" and "unknown grader_params" in status["error_message"]
    assert score.status_code == 409


async def test_failed_rollouts_score_zero_but_unscoreable_ones_are_dropped(tmp_path: Path) -> None:
    app = harness(tmp_path)
    rollout_id = await played(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as client:
        failed_status, failed_score = await grade(client, grading_body("gw_never_played", status="failed"))
        lost_status, lost_score = await grade(client, grading_body("gw_lost_on_restart"))
        assert (await client.post("/collect_score", json={"rollout_id": "gw_unknown"})).status_code == 404
        assert (await client.post("/get_grading_status", json={"rollout_id": rollout_id})).status_code == 404

    # A rollout AC2 reports as failed is a harness-scored failure: 0, below every real hand, and it still trains.
    assert failed_status["status"] == "completed" and failed_score.json()["score"] == FAILURE_SCORE
    # A completed rollout the harness can't find has no log to score: the grading fails and AC2 drops the sample.
    assert lost_status["status"] == "failed" and lost_score.status_code == 409


def hand(seat_stacks: dict, actions: list, ev: dict | None = None) -> dict:
    return {
        "hand_number": 1,
        "starting_state": {s: {"starting_stack": 200} for s in seat_stacks},
        "actions": actions,
        "ev": ev or {},
    }


def log_for(actions: list, final: dict, **labels: dict) -> dict:
    names = ["seat0", "seat1"]
    return {
        "session_info": {"big_blind": 2, "player_names": names, "starting_stacks": [200, 200]},
        "final_stacks": final,
        "hands": [hand(dict.fromkeys(names), actions, **labels)],
    }


def action(player: str, kind: str, format_error: bool = False) -> dict:
    return {"player": player, "action": {"type": kind}, "model_response": {"format_error": format_error}}


def test_reward_is_ev_in_20_bb_units_shifted_by_six() -> None:
    ev = {"seat0": {"ev_bb": 30.0, "realized_bb": 40.0, "method": "allin"}}
    log = log_for([action("seat0", "call")], {"seat0": 280, "seat1": 120}, ev=ev)
    reply = grade_log(log, ["seat0"])
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 30.0 / 20.0)
    assert reply["artifacts"]["ev_bb"] == 30.0 and reply["artifacts"]["realized_bb"] == 40.0

    # Without EV labels the reward is the chips actually won.
    plain = log_for([action("seat0", "call")], {"seat0": 240, "seat1": 160})
    assert grade_log(plain, ["seat0"])["score"] == pytest.approx(REWARD_OFFSET + 20.0 / 20.0)
    assert grade_log(plain, ["seat0"])["artifacts"]["ev_method"] == "realized"


def test_rewards_order_failures_below_every_real_hand() -> None:
    """Every legitimate outcome scores strictly above 0; a format error scores exactly 0."""
    outcomes = [-1000.0, -100.0, -5.0, 0.0, 5.0, 100.0, 1000.0]
    scores = []
    for ev_bb in outcomes:
        ev = {"seat0": {"ev_bb": ev_bb, "realized_bb": ev_bb, "method": "allin"}}
        scores.append(grade_log(log_for([action("seat0", "call")], {"seat0": 200, "seat1": 200}, ev=ev), ["seat0"]))
    values = [s["score"] for s in scores]
    assert all(v > FAILURE_SCORE for v in values) and values == sorted(values)
    assert min(values) == 1.0 and max(values) == 11.0


def test_rewards_are_clipped_nonnegative_and_format_errors_score_zero() -> None:
    huge = {"seat0": {"ev_bb": 400.0, "realized_bb": 400.0, "method": "allin"}}
    win = log_for([action("seat0", "raise")], {"seat0": 400, "seat1": 0}, ev=huge)
    assert grade_log(win, ["seat0"])["score"] == 11.0
    broken = log_for([action("seat0", "fold", format_error=True)], {"seat0": 199, "seat1": 201}, ev=huge)
    assert grade_log(broken, ["seat0"])["score"] == FAILURE_SCORE == 0.0
    lose = {"seat0": {"ev_bb": -400.0, "realized_bb": -400.0, "method": "allin"}}
    worst = log_for([action("seat0", "call")], {"seat0": 0, "seat1": 400}, ev=lose)
    assert grade_log(worst, ["seat0"])["score"] == 1.0 > FAILURE_SCORE
    with pytest.raises(GradingError):
        grade_log({"hands": []}, ["seat0"])


def test_grader_params_and_reply_size_are_bounded() -> None:
    check_grader_params(None)
    check_grader_params({})
    with pytest.raises(GradingError):
        check_grader_params({"outcome_weight": 0.5})
    reply = bounded({"score": 1.0, "reasoning": "r" * MAX_SCORE_BYTES, "artifacts": {"blob": "x" * MAX_SCORE_BYTES}})
    assert (
        len(json.dumps(reply)) <= MAX_SCORE_BYTES and reply["score"] == 1.0 and reply["artifacts"] == {"trimmed": True}
    )


def test_the_extension_points_change_the_reward_but_failures_stay_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teams replace `hand_reward` and extend `GRADER_PARAMS`; a format error still scores exactly 0."""
    import poker_harness.grading as grading

    monkeypatch.setattr(grading, "GRADER_PARAMS", frozenset({"scale"}))
    monkeypatch.setattr(grading, "hand_reward", lambda result, params: params.get("scale", 1.0) * result.realized_bb)
    params = grading.check_grader_params({"scale": 0.5})
    ev = {"seat0": {"ev_bb": 2.0, "realized_bb": 4.0, "method": "allin"}}
    log = log_for([action("seat0", "call")], {"seat0": 208, "seat1": 192}, ev=ev)
    assert grading.grade_log(log, ["seat0"], params)["score"] == pytest.approx(REWARD_OFFSET + 2.0)
    broken = log_for([action("seat0", "call", format_error=True)], {"seat0": 208, "seat1": 192}, ev=ev)
    assert grading.grade_log(broken, ["seat0"], params)["score"] == FAILURE_SCORE

