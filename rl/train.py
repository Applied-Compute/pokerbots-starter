"""GRPO on 6-max poker hands through the self-hosted harness, graded harness-side.

The harness plays each hand (the trained seat through AC2's rollout relay, the other five seats through your
opponent endpoint) and scores it with the plain EV reward: 6 + clip(EV / 20 BB, -5, +5), failures 0. The run names
no grader, so AC2 calls the harness's Grader API. Checkpoints every 10 steps; a sidecar eval plays the `select` deals
at each checkpoint so you can pick one. Judge the pick on `eval` afterwards, never on `select`.

    uv run python rl/train.py --project <project> --dry-run
    uv run python rl/train.py --project <project>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from ac2.sdk import Client, TrainingCustomHarnessConfig
from ac2.sdk.run_config import (
    EvalConfig,
    EvalDatasetConfig,
    ExperimentConfig,
    InternalConfig,
    InternalLaunchConfig,
    RLConfig,
    RolloutConfig,
    RunConfig,
    StepConfig,
    TasksConfig,
    TrainDatasetConfig,
    TrainerConfig,
)

MODEL = "Qwen/Qwen3.6-35B-A3B"
PREFIX = "pokerbots"
IPB, SPI = 16, 4  # hands per step x samples per hand: 64 rollouts per step
STEPS = 60
LEARNING_RATE = 2e-6
KL_COEF = 0.01
# The event caps each decision at 5,000 output tokens, reasoning included; train with the same cap so the model
# learns to fit its reasoning and the tool call inside it.
MAX_RESPONSE, MAX_TOTAL = 5_000, 32_768
GRADING_TIMEOUT_S = 900


def run_config(
    name: str,
    prefix: str = PREFIX,
    steps: int = STEPS,
    lr: float = LEARNING_RATE,
    concurrency: int | None = None,
    cluster_id: str | None = None,
) -> RunConfig:
    return RunConfig(
        model=MODEL,
        name=name,
        description=f"GRPO on {prefix}-train through the self-hosted poker harness, EV reward, harness-side grading.",
        tags={"benchmark": "pokerbots", "method": "grpo", "dataset": f"{prefix}-train"},
        experiment_cfg=ExperimentConfig(
            method=RLConfig(advantage="dr-grpo", use_kl_loss=True, kl_loss_coef=KL_COEF),
            step=StepConfig(
                num_train_steps=steps,
                problem_batch_size=IPB,
                samples_per_problem=SPI,
                checkpoint_interval=10,
                keep_last_checkpoints=6,
            ),
            tasks=TasksConfig(
                datasets=(TrainDatasetConfig(ac2_dataset_name=f"{prefix}-train"),),
                custom_harness=TrainingCustomHarnessConfig(grading_timeout_seconds=GRADING_TIMEOUT_S),
                # Qwen thinking is on at the event; train with it on.
                apply_chat_template_kwargs={"enable_thinking": True},
            ),
            trainer=TrainerConfig(learning_rate=lr),
            rollout=RolloutConfig(
                n_inference_replicas=1,
                max_response_len=MAX_RESPONSE,
                max_total_len=MAX_TOTAL,
                # Concurrent hands on your harness; size it to what your harness can serve.
                global_sampling_concurrency=concurrency or IPB * SPI,
            ),
            eval=EvalConfig(
                datasets=(EvalDatasetConfig(ac2_dataset_name=f"{prefix}-select"),),
                interval=10,
                eval_before_train=True,  # step 0 is base on the same deals: the in-run comparison
                samples_per_problem=1,
                global_sampling_concurrency=32,
                mode="sidecar",
                n_inference_replicas=1,
            ),
        ),
        # W&B is off because AC2 refuses `wandb_enabled` for customer orgs today; set it if your org allows it.
        internal_cfg=InternalConfig(launch=InternalLaunchConfig(cluster_id=cluster_id, tensorboard_enabled=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="pokerbots")
    parser.add_argument("--prefix", default=PREFIX, help="dataset prefix from rl/make_dataset.py")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--concurrency", type=int, help="cap on concurrent hands (default IPB x SPI)")
    parser.add_argument("--cluster-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    name = f"pokerbots-grpo-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(name, args.prefix, args.steps, args.lr, args.concurrency, args.cluster_id)
    if args.dry_run:
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
