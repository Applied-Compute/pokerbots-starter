"""Sample grader: how the harness scores each rollout for AC2.

How it is wired
---------------
A run that names no grader is graded by the harness itself, through AC2's Grader API (routes in `server.py`):

    POST /submit_grading      start grading one rollout; AC2 may repeat it, so it is idempotent
    POST /get_grading_status  queued | running | completed | failed
    POST /collect_score       the reply `grade_log` builds below: {"score", "reasoning", "artifacts"}

`server.py` calls `check_grader_params` with the task's `grader_params`, then `grade_log` with the finished game log,
or `failure_score` when the rollout never produced one. Nothing in this module talks to the network.

What it scores (deliberately naive)
-----------------------------------
The trained seat's result in big blinds. With `env_params.ev_labels`, an all-in before the river counts its expected
share of the pot instead of the cards that came (`ev.py`); otherwise it is the chips actually won.

    score = 6 + clip(hand_reward, -5, +5),  hand_reward = clip(EV, -100 BB, +100 BB) / 1 BB

So every real hand scores in [1, 11]. Using 1 BB per reward point (not 20) makes typical single-hand pots
actually use that band, so GRPO sees real within-group gaps. Model-caused failures score exactly 0: a format
error (no usable `submit_action`, or a reply cut off at the token cap), or a rollout AC2 reports as failed or
truncated. Rewards are never negative, a failure is always worse than any real hand, and failed samples still
train instead of being dropped. The +6 is a shift, not a rescale, so group-relative advantages (GRPO) are unchanged.

Extension points (search for "EXTENSION POINT")
-----------------------------------------------
1. `hand_reward`: the raw reward for a played hand. Replace it with your own.
2. `GRADER_PARAMS`: per-task knobs your reward reads from the dataset's `grader_params`.
3. `artifacts`: anything you add to the reply is stored with each score row on AC2, for analysis.
Keep failures at 0 and real hands above it, or a model can learn that failing pays.

This grader scores chips only. On single-hand tasks the scratchpad starts blank and its after-hand update is never
read, so nothing here rewards or exercises scratchpad use; how to train that is up to you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

REWARD_OFFSET = 6.0  # shifts the clipped reward [-5, +5] to [1, 11]
REWARD_FLOOR, REWARD_CEILING = -5.0, 5.0
FAILURE_SCORE = 0.0  # every model-caused failure
EV_CLIP_BB = 100.0  # one hand can't swing the reward by more than a full stack
EV_UNIT_BB = 1.0  # 1 BB of EV = 1 reward point so typical hands use most of [1, 11]
MAX_SCORE_BYTES = 1 << 20  # AC2 rejects collect_score replies over 1 MiB

# EXTENSION POINT 2: the `grader_params` keys your reward accepts. Unknown keys fail the grading, so a typo in a
# dataset can't silently change the reward.
GRADER_PARAMS: frozenset[str] = frozenset()


class GradingError(ValueError):
    """The log can't be scored. AC2 drops the sample (GRADER_ERROR) instead of training on a made-up reward."""


@dataclass
class SeatResult:
    """What the trained seat did over one game log."""

    ev_bb: float  # EV in big blinds (all-in luck removed when the log has EV labels)
    realized_bb: float  # chips actually won, in big blinds
    method: str  # how ev_bb was computed: "realized", "allin", "showdown", ...
    decisions: int
    format_errors: int
    folded: bool
    hands: int


def check_grader_params(grader_params: dict[str, Any] | None) -> dict[str, Any]:
    """The task's `grader_params`, after refusing any key not in `GRADER_PARAMS`."""
    params = dict(grader_params or {})
    unknown = set(params) - GRADER_PARAMS
    if unknown:
        raise GradingError(f"unknown grader_params: {sorted(unknown)}")
    return params


def seat_result(log: dict[str, Any], policy_seats: list[str]) -> SeatResult:
    """Sum the trained seat's results over every hand in the log."""
    policy = set(policy_seats)
    if not log.get("hands") or not policy:
        raise GradingError("the rollout has no hands or no policy seats")
    info = log["session_info"]
    big_blind = info["big_blind"]
    start = dict(zip(info["player_names"], info["starting_stacks"], strict=True))
    seats = [(hand, seat) for hand in log["hands"] for seat in policy if seat in hand["starting_state"]]
    actions = [a for hand in log["hands"] for a in hand["actions"] if a["player"] in policy]
    labels = [hand.get("ev", {}).get(seat) for hand, seat in seats]
    if labels and all(label is not None for label in labels):
        ev_bb = sum(label["ev_bb"] for label in labels)
        realized_bb = sum(label["realized_bb"] for label in labels)
        method = labels[0]["method"]
    else:  # no EV labels on the task: score the chips actually won
        realized_bb = ev_bb = sum(log["final_stacks"][seat] - start[seat] for seat in policy) / big_blind
        method = "realized"
    return SeatResult(
        ev_bb=ev_bb,
        realized_bb=realized_bb,
        method=method,
        decisions=len(actions),
        format_errors=sum(bool(a["model_response"].get("format_error")) for a in actions),
        folded=any(a["action"]["type"] == "fold" for a in actions),
        hands=len(log["hands"]),
    )


def hand_reward(result: SeatResult, params: dict[str, Any]) -> float:
    """EXTENSION POINT 1: the raw reward for a game the seat played without failing, before clipping and the +6.

    The default is EV in units of 1 BB, clipped to +-100 BB. `result` has everything the log says about the trained
    seat; read more from the log in `seat_result` if you need it. `params` is the task's checked `grader_params`.
    """
    del params  # the default reward takes no parameters
    return max(-EV_CLIP_BB, min(EV_CLIP_BB, result.ev_bb)) / EV_UNIT_BB


def failure_score(reason: str) -> dict[str, Any]:
    """The reply for a rollout that failed before producing a log (AC2 reported it failed or truncated)."""
    return {"score": FAILURE_SCORE, "reasoning": reason, "artifacts": {"failure": True}}


def grade_log(
    log: dict[str, Any], policy_seats: list[str], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score one finished game for its policy seats, as a `collect_score` reply."""
    result = seat_result(log, policy_seats)
    if result.format_errors:  # model-caused failure: exactly 0, below every real hand
        score = FAILURE_SCORE
    else:
        score = REWARD_OFFSET + max(REWARD_FLOOR, min(REWARD_CEILING, hand_reward(result, params or {})))
    return {
        "score": score,
        "reasoning": f"Policy seat {'/'.join(sorted(policy_seats))}: EV {result.ev_bb:+.1f} BB "
        f"(realized {result.realized_bb:+.1f}), {result.method}, {result.format_errors} format errors.",
        # EXTENSION POINT 3: add any per-rollout numbers you want to analyze later.
        "artifacts": {
            "ev_bb": result.ev_bb,
            "realized_bb": result.realized_bb,
            "ev_method": result.method,
            "policy_decisions": result.decisions,
            "policy_format_errors": result.format_errors,
            "folded": result.folded,
            "hands": result.hands,
        },
    }


def bounded(score: dict[str, Any]) -> dict[str, Any]:
    """Drop artifacts, then truncate reasoning, until the reply fits AC2's 1 MiB limit."""
    if len(json.dumps(score)) <= MAX_SCORE_BYTES:
        return score
    trimmed = {"score": score["score"], "reasoning": score.get("reasoning", ""), "artifacts": {"trimmed": True}}
    overflow = len(json.dumps(trimmed)) - MAX_SCORE_BYTES
    if overflow > 0:
        trimmed["reasoning"] = trimmed["reasoning"][: max(0, len(trimmed["reasoning"]) - overflow - 16)]
    return trimmed
