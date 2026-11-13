"""Offline OPSD: distill the hinted teacher into a student that never sees the hint.

Offline mode replays each stored reply from `build_dataset.py`: the student scores it on the plain harness prompt,
the frozen teacher (the base model) on the same prompt plus the hint, and the loss is their reverse KL on the reply's
tokens. No environment, grader or reward runs in training, and no inference engine is needed.

Training has no poker eval: after the run, serve a checkpoint and play it through the harness, which never shows the
hint, so that eval measures the student without it.

    uv run python opsd/train.py --dataset pokerbots-opsd-train --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from ac2.sdk import Client
from ac2.sdk.run_config import (
    EvalConfig,
    ExperimentConfig,
    InternalConfig,
    InternalLaunchConfig,
    OPSDConfig,
    RolloutConfig,
    RunConfig,
    StepConfig,
    TasksConfig,
    TrainDatasetConfig,
    TrainerConfig,
)

MODEL = "Qwen/Qwen3.6-35B-A3B"
STEPS = 20
BATCH = 64  # decisions per step
LEARNING_RATE = 2e-6
MAX_RESPONSE, MAX_TOTAL = 5_000, 32_768  # the event's per-decision cap, reasoning included


def run_config(
    name: str, dataset: str, steps: int = STEPS, lr: float = LEARNING_RATE, cluster_id: str | None = None
) -> RunConfig:
    return RunConfig(
        model=MODEL,
        name=name,
        description=f"Offline OPSD of a prompted strategy hint into {MODEL} (student without the hint), {dataset}.",
        tags={"benchmark": "pokerbots", "method": "opsd", "dataset": dataset},
        experiment_cfg=ExperimentConfig(
            method=OPSDConfig(rollout_mode="offline"),
            step=StepConfig(
                num_train_steps=steps,
                problem_batch_size=BATCH,
                samples_per_problem=1,
                checkpoint_interval=10,
                keep_last_checkpoints=2,
            ),
            tasks=TasksConfig(
                datasets=(TrainDatasetConfig(ac2_dataset_name=dataset),),
                # Qwen thinking is on at the event; render the replayed replies the same way.
                apply_chat_template_kwargs={"enable_thinking": True},
            ),
            trainer=TrainerConfig(learning_rate=lr),
            rollout=RolloutConfig(n_inference_replicas=0, max_response_len=MAX_RESPONSE, max_total_len=MAX_TOTAL),
            eval=EvalConfig(mode="off"),
        ),
        # W&B is off because AC2 refuses `wandb_enabled` for customer orgs today; set it if your org allows it.
        internal_cfg=InternalConfig(launch=InternalLaunchConfig(cluster_id=cluster_id, tensorboard_enabled=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="pokerbots")
    parser.add_argument("--dataset", required=True, help="hinted dataset from build_dataset.py")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--cluster-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    name = f"pokerbots-opsd-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(name, args.dataset, args.steps, args.lr, args.cluster_id)
    if args.dry_run:
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
