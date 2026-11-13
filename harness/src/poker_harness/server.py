"""Harness API for the AC2 Harness Connector: one table per rollout.

`ac2 harness connect --upstream http://127.0.0.1:<port>` forwards three POSTs here: `submit` starts a table,
`get_status` reports it, and `collect_outputs` returns the game log. When a run names no grader, AC2 grades through
the Grader API here instead (`submit_grading`, `get_grading_status`, `collect_score`), scored by the sample
grader in `grading.py`. Each policy decision is one chat completion against the rollout's relay URL, so AC2 records every call in the rollout's trace.

`env_params.table_talk` (default false) turns on table talk: every seat's `submit_action` gains an optional `say`,
sanitized and shown to the other seats.

A server keeps its tables in memory, so scale out by sharding (`shard_router.py` sends every call for a rollout to
the same server), not with `uvicorn --workers`.

    uv run uvicorn poker_harness.server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .completions import (
    DEFAULT_DECISION_TIMEOUT_SECONDS,
    GATEWAY_URL,
    cleanup_params,
    opponent_params,
    relay_completion,
    responses_completion,
    rules_output,
    table_rules,
    with_preset,
)
from .ev import label_ev
from .game import play_session
from .grading import GradingError, bounded, check_grader_params, failure_score, grade_log
from .labels import label_folds
from .prompts import TableRules, talk_fields
from .seat import ModelSeat

logger = logging.getLogger("poker_harness.server")

MAX_OUTPUT_BYTES = 900_000  # the connector rejects upstream responses over 1 MiB


class HarnessTask(BaseModel):
    input: list[Any]
    env_params: dict[str, Any] | None = None


class SubmitRequest(BaseModel):
    ac_api_url: str = Field(min_length=1)
    ac_api_key: str = Field(min_length=1, repr=False)
    task: HarnessTask
    rollout_id: str | None = None


class RolloutRequest(BaseModel):
    rollout_id: str = Field(min_length=1)


class GradingTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[Any]
    env_params: dict[str, Any] | None = None
    grader_params: dict[str, Any] | None = None


class SubmitGradingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: str = Field(min_length=1, max_length=255)
    rollout_status: Literal["completed", "failed", "truncated"]
    task: GradingTask


class GradingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: str = Field(min_length=1, max_length=255)


@dataclass
class Grading:
    status: str = "queued"
    error: str | None = None
    score: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class Rollout:
    id: str
    rules: TableRules
    policy_seats: list[str] = field(default_factory=list)
    status: str = "queued"
    error: str | None = None
    log: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None


def _bounded(log: dict[str, Any]) -> dict[str, Any]:
    """Drop free text from model responses if the log would not fit through the connector."""
    if len(json.dumps(log)) <= MAX_OUTPUT_BYTES:
        return log
    trimmed = json.loads(json.dumps(log))
    for hand in trimmed["hands"]:
        for action in hand["actions"]:
            for key in ("reasoning", "read", "raw_output", "private"):
                action["model_response"].pop(key, None)
    if len(json.dumps(trimmed)) > MAX_OUTPUT_BYTES:  # scratchpad tables: drop the cleanup turns' text, keep the pads
        for seat in (trimmed.get("private") or {}).values():
            for hand in seat.get("hands", {}).values():
                for key in ("prompt", "output", "update_calls"):
                    (hand.get("cleanup") or {}).pop(key, None)
    return trimmed


def create_app(
    relay_transport: httpx.AsyncBaseTransport | None = None,
    runs_dir: Path | None = None,
    opponent_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """`relay_transport` and `opponent_transport` replace the network for tests; `runs_dir` keeps every game log.

    With `env_params.policy_seats` and `env_params.opponent_model`, only the policy seats call the rollout's relay
    (the model AC2 evaluates or trains); every other seat calls `opponent_model` at `POKER_OPPONENT_BASE_URL` (default: the AC2 gateway) with
    `POKER_OPPONENT_API_KEY` (default: `AC2_API_KEY`). Without them, every seat calls the relay.
    """
    token = os.environ.get("POKER_HARNESS_TOKEN")
    opponent_key = os.environ.get("POKER_OPPONENT_API_KEY") or os.environ.get("AC2_API_KEY")
    opponent_url = os.environ.get("POKER_OPPONENT_BASE_URL", GATEWAY_URL)
    runs = runs_dir or Path(os.environ.get("POKER_HARNESS_RUNS", "runs"))
    rollouts: dict[str, Rollout] = {}
    gradings: dict[str, Grading] = {}
    pools: list[ProcessPoolExecutor] = []

    def label_pool() -> ProcessPoolExecutor:
        """All-in EV labels are CPU-bound, so they run off the event loop."""
        if not pools:
            workers = max(1, min(3, (os.cpu_count() or 2) - 1))
            pools.append(ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")))
        return pools[0]

    app = FastAPI(title="poker harness")
    timeout = httpx.Timeout(DEFAULT_DECISION_TIMEOUT_SECONDS, connect=30.0)

    def authorize(authorization: str | None) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid harness token")

    async def run_table(rollout: Rollout, request: SubmitRequest) -> None:
        params = with_preset(request.task.env_params or {})  # `preset: "event"` applies the event rules
        if rollout.rules.table_talk:  # every seat's tool gains `say`
            params = params | {"submit_fields": talk_fields(params.get("submit_fields"))}
        rollout.status = "running"
        try:
            seats = int(params.get("seats", 10))
            policy = {int(i) for i in params.get("policy_seats", range(seats))}
            if params.get("opponent_model") and not opponent_key and len(policy) < seats:
                raise RuntimeError("opponent seats need POKER_OPPONENT_API_KEY or AC2_API_KEY in the harness environment")
            decision_timeout = float(params.get("decision_timeout_seconds", DEFAULT_DECISION_TIMEOUT_SECONDS))
            opponent_timeout = float(params.get("opponent_timeout_seconds", decision_timeout))
            async with (
                httpx.AsyncClient(timeout=timeout, transport=relay_transport) as relay_client,
                httpx.AsyncClient(timeout=timeout, transport=opponent_transport) as opponent_client,
            ):
                policy_complete = relay_completion(relay_client, request.ac_api_url, request.ac_api_key, params)
                opponent_api = responses_completion if params.get("opponent_api") == "responses" else relay_completion
                opponent_complete = (
                    opponent_api(opponent_client, opponent_url, opponent_key or "", opponent_params(params))
                    if params.get("opponent_model")
                    else policy_complete
                )
                # The after-hand cleanup turn has its own output cap (`cleanup_max_completion_tokens`).
                policy_cleanup = relay_completion(
                    relay_client, request.ac_api_url, request.ac_api_key, cleanup_params(params)
                )
                opponent_cleanup = (
                    opponent_api(
                        opponent_client, opponent_url, opponent_key or "", cleanup_params(opponent_params(params))
                    )
                    if params.get("opponent_model")
                    else policy_cleanup
                )
                players = [
                    ModelSeat(
                        f"seat{i}",
                        policy_complete if i in policy else opponent_complete,
                        rollout.rules,
                        model="ac2/policy" if i in policy else f"opponent/{params.get('opponent_model', 'policy')}",
                        timeout_seconds=decision_timeout if i in policy else opponent_timeout,
                        cleanup_complete=policy_cleanup if i in policy else opponent_cleanup,
                    )
                    for i in range(seats)
                ]
                rollout.policy_seats = [f"seat{i}" for i in sorted(policy)]
                rollout.log = await play_session(rollout.id, int(params["seed"]), players, rollout.rules)
                label_folds(rollout.log, set(rollout.policy_seats))
                if params.get("ev_labels"):  # all-in EV for the policy seats, scored by grading.py
                    rollout.log = await asyncio.get_running_loop().run_in_executor(
                        label_pool(),
                        label_ev,
                        rollout.log,
                        set(rollout.policy_seats),
                        int(params.get("ev_variants", 8)),
                        int(params.get("ev_runouts", 32)),
                        bool(params.get("ev_redraw", False)),
                    )
            # The relay URL names the AC2 job the hand belongs to; keep it so concurrent runs can be told apart.
            job = re.search(r"/(job_[0-9a-f]+)/", request.ac_api_url)
            rollout.log["relay_job_id"] = job[1] if job else None
            runs.mkdir(parents=True, exist_ok=True)
            (runs / f"{rollout.id}.json").write_text(json.dumps(rollout.log, indent=1))
            rollout.status = "completed"
        except Exception as exc:  # reported to AC2 as a failed rollout
            logger.exception("rollout %s failed", rollout.id)
            rollout.status, rollout.error = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "rollouts": {s: sum(r.status == s for r in rollouts.values()) for s in ("running",)}}

    @app.post("/submit")
    async def submit(
        request: SubmitRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, str]:
        authorize(authorization)
        rollout_id = request.rollout_id or idempotency_key or f"table_{uuid.uuid4().hex}"
        if rollout_id in rollouts:
            return {"rollout_id": rollout_id}
        params = with_preset(request.task.env_params or {})  # `preset: "event"` applies the event rules
        if "seed" not in params:
            raise HTTPException(status_code=422, detail="env_params.seed is required")
        rollout = Rollout(id=rollout_id, rules=table_rules(params))
        rollouts[rollout_id] = rollout
        rollout.task = asyncio.create_task(run_table(rollout, request))
        logger.info("rollout %s: %d hands, relay %s", rollout_id, rollout.rules.total_hands, request.ac_api_url)
        return {"rollout_id": rollout_id}

    def lookup(rollout_id: str) -> Rollout:
        rollout = rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="unknown rollout")
        return rollout

    @app.post("/get_status")
    async def get_status(request: RolloutRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        rollout = lookup(request.rollout_id)
        status: dict[str, Any] = {"status": rollout.status}
        if rollout.error:
            status["error_message"] = rollout.error[:2000]
        if rollout.log:
            status["annotations"] = {"hands_played": len(rollout.log["hands"])}
        return status

    @app.post("/collect_outputs")
    async def collect_outputs(
        request: RolloutRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        rollout = lookup(request.rollout_id)
        log = _bounded(rollout.log) if rollout.log else None
        return {
            "log": log,
            "rules": rules_output(rollout.rules),
            "policy_seats": rollout.policy_seats,
            "error": rollout.error,
        }

    async def run_grading(grading: Grading, request: SubmitGradingRequest) -> None:
        grading.status = "running"
        try:
            params = check_grader_params(request.task.grader_params)  # the sample grader: grading.py
            rollout = rollouts.get(request.rollout_id)
            if request.rollout_status != "completed":
                grading.score = failure_score(f"AC2 reported the rollout {request.rollout_status}.")
            elif rollout is None:  # e.g. the harness restarted since the table was played: nothing to score
                raise GradingError("unknown rollout; the harness has no log for it")
            else:
                if rollout.task is not None:
                    await asyncio.shield(rollout.task)
                if rollout.status != "completed" or not rollout.log:
                    grading.score = failure_score(f"The table ended {rollout.status}: {rollout.error}")
                else:
                    grading.score = bounded(grade_log(rollout.log, rollout.policy_seats, params))
            grading.status = "completed"
        except Exception as exc:  # AC2 drops the sample (GRADER_ERROR)
            logger.exception("grading %s failed", request.rollout_id)
            grading.status, grading.error = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"

    @app.post("/submit_grading")
    async def submit_grading(
        request: SubmitGradingRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, str]:
        """One grading per rollout: a replay (same rollout_id) returns the grading already started."""
        authorize(authorization)
        if idempotency_key is not None and idempotency_key != f"{request.rollout_id}:grading":
            raise HTTPException(status_code=422, detail="Idempotency-Key must be '<rollout_id>:grading'")
        if request.rollout_id not in gradings:
            grading = Grading()
            gradings[request.rollout_id] = grading
            grading.task = asyncio.create_task(run_grading(grading, request))
        return {"rollout_id": request.rollout_id}

    def lookup_grading(rollout_id: str) -> Grading:
        grading = gradings.get(rollout_id)
        if grading is None:
            raise HTTPException(status_code=404, detail="unknown grading")
        return grading

    @app.post("/get_grading_status")
    async def get_grading_status(
        request: GradingRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        grading = lookup_grading(request.rollout_id)
        status: dict[str, Any] = {"status": grading.status}
        if grading.error:
            status["error_message"] = grading.error[:2000]
        if grading.score is not None:
            status["annotations"] = {"score": grading.score["score"]}
        return status

    @app.post("/collect_score")
    async def collect_score(
        request: GradingRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        grading = lookup_grading(request.rollout_id)
        if grading.status != "completed" or grading.score is None:
            raise HTTPException(status_code=409, detail=f"grading is {grading.status}")
        return grading.score

    return app


app = create_app()
