"""Run a BYOH eval through the self-hosted poker harness. No project grader: the harness grades each hand.

    uv run python -m vals_tools.eval --dataset vals-poker-6max-table-smoke --model <catalog-model>   # harness smoke
    uv run python -m vals_tools.eval --dataset vals-poker-6max-smoke --deployment <project>/<deployment>
    uv run python -m vals_tools.eval --dataset vals-poker-6max-eval --deployment <project>/<deployment>

Run a 4-hand smoke before any large eval, and only send traffic to a deployment after it has answered one request.
"""

from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, EvalConfig
from ac2.sdk.byoh import InferenceModelConfiguration

GRADING_TIMEOUT_S = 900


def eval_config(
    dataset: str,
    model: str | None = None,
    deployment: str | None = None,
    api_type: str = "responses",
    max_parallel: int = 32,
    cluster_id: str | None = None,
) -> EvalConfig:
    if (model is None) == (deployment is None):
        raise ValueError("pass exactly one of model or deployment")
    seat = (
        InferenceModelConfiguration(deployment=deployment)
        if deployment
        else ModelConfiguration(model=model, api_type=api_type)
    )
    return EvalConfig(
        custom_harness=CustomHarnessConfig(model=seat, grading_timeout_seconds=GRADING_TIMEOUT_S),
        dataset=dataset,
        name=f"vals-poker-{dataset}",
        description=f"Poker hands on {dataset}, harness-side grading.",
        num_samples=1,
        max_parallel=max_parallel,
        on_error="record",
        cluster_id=cluster_id,
        tags={"benchmark": "vals-poker", "dataset": dataset, "seat_model": deployment or model},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-poker")
    parser.add_argument("--dataset", default="vals-poker-6max-smoke")
    seat = parser.add_mutually_exclusive_group(required=True)
    seat.add_argument("--model", help="an AC2 catalog model")
    seat.add_argument("--deployment", help="an AC2 inference deployment, e.g. a served checkpoint")
    parser.add_argument("--api-type", default="responses", help="for --model: responses or completions")
    parser.add_argument("--max-parallel", type=int, default=32, help="concurrent hands on your harness")
    parser.add_argument("--cluster-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = eval_config(
        args.dataset, args.model, args.deployment, args.api_type, args.max_parallel, args.cluster_id
    )
    if args.dry_run:
        print(config)
        return
    run = asyncio.run(Client(project=args.project).eval.run(config))
    print(getattr(run, "eval_id", run))


if __name__ == "__main__":
    main()
