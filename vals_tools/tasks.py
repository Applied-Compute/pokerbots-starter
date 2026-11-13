"""One-hand tasks for the self-hosted poker harness. The harness reads everything from `env_params`.

Each task is one hand of 6-max no-limit hold'em with sanitized table talk. The model being evaluated or trained plays
`policy_seats`; every other seat is played by `opponent_model`, an OpenAI-compatible model the harness calls on the
AC2 gateway (an external endpoint in your project). The deck depends only on `seed`, so two runs on the same dataset
play the same deals and can be compared hand by hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ac2.runtime import Message, Task

# The event's rules are the harness's "event" preset: 6 seats, table talk, 60-hand games with chips carried over (no
# rebuys; a busted seat is out), invalid moves played as check or fold, a private scratchpad with an after-hand
# cleanup turn, Qwen thinking on, and 5,000 output tokens per decision (10,000 for the cleanup turn). A task's own keys
# win over the preset; here every task is a single hand, so the scratchpad starts blank and its update is unused.
SEATS = 6
TABLE_RULES: dict[str, Any] = {
    "preset": "event",
    # With table talk on, the harness adds an optional, sanitized `say` field to these.
    "submit_fields": ["action", "amount"],
    # The event has no time limit. This is only a safety net so a stuck call can't hang a training rollout.
    "decision_timeout_seconds": 600,
}
# Each split draws its own seeds, so no deal appears in two splits. `select` is for picking checkpoints (training
# sidecars); `eval` is held out for the verdict. Never judge a checkpoint on the deals you picked it on.
SPLITS = {
    "train": (2000, 6_610_001),
    "select": (500, 6_620_002),
    "eval": (3000, 6_630_003),
    "smoke": (4, 6_640_004),
}


@dataclass(frozen=True)
class Opponents:
    """How the non-policy seats are played. `api` is "responses" for models that need the Responses API for tool
    calls with reasoning (such as OpenAI reasoning models), else "chat"."""

    model: str
    api: str = "responses"
    reasoning_effort: str | None = "medium"
    timeout_seconds: int = 60

    def env_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"opponent_model": self.model, "opponent_timeout_seconds": self.timeout_seconds}
        if self.api == "responses":
            params["opponent_api"] = "responses"
        if self.reasoning_effort:
            params["opponent_reasoning_effort"] = self.reasoning_effort
        return params


def hand_task(seed: int, policy_seat: int, opponents: Opponents, seats: int = SEATS) -> Task:
    """One hand with the policy in `policy_seat`. The button starts on the last seat, so seat 0 is the small blind,
    seat 1 the big blind and the last seat the button. `ev_labels` replaces all-in luck with expected value."""
    return Task(
        input=[Message(role="user", content=f"Play one hand of {seats}-seat no-limit hold'em from seat {policy_seat}.")],
        env_params=TABLE_RULES
        | {
            "seed": seed,
            "seats": seats,
            "total_hands": 1,
            "policy_seats": [policy_seat],
            "ev_labels": True,
            "ev_runouts": 64,
            **opponents.env_params(),
        },
    )


def hand_tasks(count: int, base_seed: int, opponents: Opponents, seats: int = SEATS) -> list[Task]:
    """Fresh 48-bit seeds; the policy seat cycles through every position so each is equally represented."""
    rng = random.Random(base_seed)
    return [hand_task(rng.getrandbits(48), i % seats, opponents, seats) for i in range(count)]


def table_smoke_tasks(seats: int = SEATS) -> list[Task]:
    """Two 3-hand tables where the evaluated model plays every seat: no opponent endpoint needed."""
    rng = random.Random(6_650_005)
    return [
        Task(
            input=[Message(role="user", content=f"Play a {seats}-seat no-limit hold'em table for 3 hands.")],
            env_params=TABLE_RULES | {"seed": rng.getrandbits(48), "seats": seats, "total_hands": 3},
        )
        for _ in range(2)
    ]


def datasets(prefix: str, opponents: Opponents, seats: int = SEATS) -> dict[str, list[Task]]:
    out = {f"{prefix}-{split}": hand_tasks(n, seed, opponents, seats) for split, (n, seed) in SPLITS.items()}
    out[f"{prefix}-table-smoke"] = table_smoke_tasks(seats)
    return out
