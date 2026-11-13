"""Score a poker eval in big blinds per hand, paired per deal against a base eval on the same dataset.

    uv run python -m vals_tools.score --project <project> --eval eval_new --base eval_base
    uv run python -m vals_tools.score --project <project> --run train_xxx --step 20

Reads the sample grader's score artifacts (`ev_bb`, `realized_bb`, `policy_format_errors`), not the shifted reward.
Both evals must use the same dataset: the same deals, so the difference can be paired hand by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from ac2.sdk import Client

COLUMNS = ("job_id", "generated_step", "datapoint_id", "artifacts", "parent_score_id", "created_at")


def score_rows(client: Client, eval_id: str | None = None, run: str | None = None, step: int | None = None) -> dict:
    """Top-level harness scores keyed by task id; the latest score wins on retries."""
    rows: dict[str, dict] = {}
    query = client.analytics.scores(run_ids=[eval_id or run]).select(*COLUMNS)
    for row in sorted(query.iter_rows(), key=lambda r: str(r["created_at"])):
        if row["parent_score_id"]:
            continue
        job = str(row["job_id"])
        if eval_id and job != eval_id:
            continue
        if run and (not job.startswith("eval_") or str(row["generated_step"]) != str(step)):
            continue
        artifacts = row["artifacts"] if isinstance(row["artifacts"], dict) else json.loads(row["artifacts"] or "{}")
        if "ev_bb" in artifacts:
            rows[str(row["datapoint_id"])] = artifacts
    return rows


def _ci95(values: list[float]) -> float:
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")


def summarize(rows: dict, base: dict | None = None) -> dict[str, Any]:
    if not rows:
        return {"hands": 0}
    ev = [r["ev_bb"] for r in rows.values()]
    out: dict[str, Any] = {
        "hands": len(ev),
        "ev_bb_per_hand": round(statistics.fmean(ev), 3),
        "ci95": round(_ci95(ev), 3),
        "format_error_hands": round(statistics.fmean(r["policy_format_errors"] > 0 for r in rows.values()), 4),
    }
    if base:
        diffs = [rows[k]["ev_bb"] - base[k]["ev_bb"] for k in rows if k in base]
        if diffs:
            out["vs_base"] = {
                "paired_hands": len(diffs),
                "diff": round(statistics.fmean(diffs), 3),
                "ci95": round(_ci95(diffs), 3),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-poker")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--eval", help="a standalone eval")
    target.add_argument("--run", help="a training run; with --step, its sidecar eval at that step")
    parser.add_argument("--step", type=int)
    parser.add_argument("--base", help="the base model's eval on the same dataset")
    args = parser.parse_args()

    client = Client(project=args.project)
    rows = score_rows(client, eval_id=args.eval, run=args.run, step=args.step)
    base = score_rows(client, eval_id=args.base) if args.base else None
    print(json.dumps(summarize(rows, base), indent=1))


if __name__ == "__main__":
    main()
