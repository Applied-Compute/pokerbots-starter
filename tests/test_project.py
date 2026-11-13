import importlib.util
import json
import sys
from pathlib import Path

import pytest
from ac2.runtime import FunctionCall, Message, Reasoning

from vals_tools.score import summarize
from vals_tools.tasks import SPLITS, Opponents, datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "opsd")]

import build_dataset  # noqa: E402
import train as opsd_train  # noqa: E402

OPPONENT = Opponents("my-project/opponent")


def test_splits_share_no_deal_and_cycle_every_seat() -> None:
    built = datasets("p", OPPONENT)
    seeds = [t.env_params["seed"] for name, tasks in built.items() if not name.endswith("table-smoke") for t in tasks]
    assert len(seeds) == len(set(seeds)) == sum(n for n, _ in SPLITS.values())
    train = built["p-train"]
    assert {t.env_params["policy_seats"][0] for t in train[:6]} == set(range(6))
    params = train[0].env_params
    assert params["seats"] == 6 and params["opponent_model"] == "my-project/opponent" and params["ev_labels"]
    assert params["opponent_api"] == "responses" and params["opponent_reasoning_effort"] == "medium"
    assert params["preset"] == "event" and built["p-table-smoke"][0].env_params["preset"] == "event"
    assert params["total_hands"] == 1 and "chat_template_kwargs" not in params  # thinking comes from the preset
    assert "policy_seats" not in built["p-table-smoke"][0].env_params  # every seat is the evaluated model


def row(ev: float) -> dict:
    return {"ev_bb": ev, "realized_bb": ev, "policy_format_errors": 0}


def test_summary_pairs_by_deal() -> None:
    rows = {"a": row(-0.5), "b": row(4.0), "c": row(2.0)}
    base = {"a": row(-3.0), "b": row(1.0), "c": row(1.0)}
    out = summarize(rows, base)
    assert out["hands"] == 3 and out["ev_bb_per_hand"] == pytest.approx(5.5 / 3, abs=1e-3)
    assert out["vs_base"]["paired_hands"] == 3 and out["vs_base"]["diff"] == pytest.approx(6.5 / 3, abs=1e-3)


def test_opsd_builder_makes_one_hinted_task_per_clean_decision() -> None:
    system = Message(role="system", content="rules")
    user = Message(role="user", content="state")
    good = [Reasoning(content="think"), FunctionCall(call_id="c1", name="submit_action", arguments='{"action":"fold"}')]
    bad = [Message(role="assistant", content="no tool call")]

    def span(start: int, reply: list) -> dict:
        dump = lambda items: json.dumps([i.model_dump(mode="json") for i in items])  # noqa: E731
        return {
            "name": "get_completion",
            "status": "OK",
            "start_time": start,
            "input": dump([system, user]),
            "output": dump(reply),
            "tokens_out": 10,
        }

    tasks, stats = build_dataset.decision_tasks({"trace_id": "t", "spans": [span(1, good), span(2, bad)]}, "hint", 100)
    assert stats == {"decisions": 2, "kept": 1, "cleanup": 0, "format": 1, "truncated": 0}
    (task,) = tasks
    (selection,) = task.opsd_hint_selections
    assert len(task.input) == 4 and selection.rollout_prefix_len == 2
    assert selection.mutation.item_boundary == 1 and selection.mutation.action == "append_to_previous_item"
    assert [t["name"] for t in task.env_params["tools"]] == ["submit_action", "update_scratchpad"]
    assert set(task.env_params["tools"][0]["parameters"]["properties"]) == {"action", "amount", "say"}
    assert "hint" not in json.dumps([i.model_dump(mode="json") for i in task.input])  # only the teacher sees it


def test_training_configs_build() -> None:
    spec = importlib.util.spec_from_file_location("rl_train", ROOT / "rl" / "train.py")
    grpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grpo)
    config = grpo.run_config("n").to_json()
    text = json.dumps(config, default=str)
    assert "dr-grpo" in text and "vals-poker-6max-train" in text and "vals-poker-6max-select" in text
    opsd = json.dumps(opsd_train.run_config("n", "d").to_json(), default=str)
    assert "offline" in opsd and '"num_train_steps": 20' in opsd
