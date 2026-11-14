"""GRPO on single 6-max hands through your self-hosted harness, graded by the harness's sample grader.

Every setting this script sends is a flag below. A flag left unset is not sent, so AC2 uses its own default for it
(learning rate, sampling concurrency, ...). Two kinds of setting are the exception. Token budgets: in BYOH
training, `max_response_len` is a budget for the whole hand (every decision's reasoning, kept inline, plus every
observation and the cleanup turn), and AC2's default of 8,192 runs out after a couple of long decisions, so the recipe
sets 24,576. Replica counts: AC2's
default for Qwen3.6-35B-A3B is 4 inference and 4 eval replicas (2 GPUs each), 20 GPUs per run with training, more
than a team's 10-GPU quota, so the recipe sets 1 of each: 8 GPUs. `--dry-run` prints the
settings this launch sets and the exact configuration it would submit; nothing else is overridden.

    uv run python rl/train.py --dry-run
    uv run python rl/train.py
    uv run python rl/train.py --lr 1e-6 --max-response-len 10000 --dry-run   # override any AC2 default explicitly

The run names no grader, so AC2 asks your harness to grade every rollout. At step 0 and every checkpoint it also
plays `<prefix>-select` as an in-run eval; judge your pick on `<prefix>-eval` afterwards.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

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


def _set(**values: Any) -> dict[str, Any]:
    """Only the values that were given: anything None is left to AC2's default."""
    return {key: value for key, value in values.items() if value is not None}


def run_config(args: argparse.Namespace, name: str) -> RunConfig:
    harness = TrainingCustomHarnessConfig(**_set(grading_timeout_seconds=args.grading_timeout))
    eval_cfg = (
        EvalConfig(mode="off")
        if args.no_eval
        else EvalConfig(
            **_set(
                datasets=(EvalDatasetConfig(ac2_dataset_name=args.eval_dataset or f"{args.prefix}-select"),),
                interval=args.eval_every,
                eval_before_train=True,  # step 0 plays the same deals with the untrained model: the comparison
                samples_per_problem=1,
                mode="sidecar",
                n_inference_replicas=args.eval_replicas,
                global_sampling_concurrency=args.eval_concurrency,
            )
        )
    )
    launch = _set(cluster_id=args.cluster_id, tensorboard_enabled=args.tensorboard or None)
    return RunConfig(
        model=args.model,
        name=name,
        description=f"GRPO on {args.prefix}-train through the self-hosted poker harness, harness-side grading.",
        tags={"benchmark": "pokerbots", "method": "grpo", "dataset": f"{args.prefix}-train"},
        experiment_cfg=ExperimentConfig(
            method=RLConfig(
                **_set(advantage=args.advantage, use_kl_loss=args.kl_coef > 0 or None, kl_loss_coef=args.kl_coef or None)
            ),
            step=StepConfig(
                **_set(
                    num_train_steps=args.steps,
                    problem_batch_size=args.hands_per_step,
                    samples_per_problem=args.samples_per_hand,
                    checkpoint_interval=args.checkpoint_every,
                    keep_last_checkpoints=args.keep_checkpoints,
                )
            ),
            tasks=TasksConfig(
                datasets=(TrainDatasetConfig(ac2_dataset_name=f"{args.prefix}-train"),),
                custom_harness=harness,
                **_set(apply_chat_template_kwargs={"enable_thinking": True} if args.thinking else None),
            ),
            trainer=TrainerConfig(**_set(learning_rate=args.lr)),
            rollout=RolloutConfig(
                **_set(
                    max_response_len=args.max_response_len,
                    max_total_len=args.max_total_len,
                    global_sampling_concurrency=args.concurrency,
                    n_inference_replicas=args.inference_replicas,
                )
            ),
            eval=eval_cfg,
        ),
        **_set(internal_cfg=InternalConfig(launch=InternalLaunchConfig(**launch)) if launch else None),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--prefix", default="pokerbots", help="dataset prefix from rl/make_dataset.py")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="base model (arena entries must fine-tune this one)")
    p.add_argument("--dry-run", action="store_true", help="print the settings and configuration, submit nothing")
    g = p.add_argument_group("set by default (the GRPO recipe); change them with these flags")
    g.add_argument("--steps", type=int, default=60, help="training steps (default 60)")
    g.add_argument("--hands-per-step", type=int, default=16, help="tasks per step (default 16)")
    g.add_argument("--samples-per-hand", type=int, default=16, help="attempts per task, the GRPO group (default 16)")
    g.add_argument("--advantage", default="dr-grpo", help="advantage estimator (default dr-grpo)")
    g.add_argument("--kl-coef", type=float, default=0.01, help="KL penalty toward the base model; 0 turns it off")
    g.add_argument("--checkpoint-every", type=int, default=10, help="steps between checkpoints (default 10)")
    g.add_argument("--eval-every", type=int, default=10, help="steps between in-run evals (default 10)")
    g.add_argument("--no-eval", action="store_true", help="no in-run eval")
    g.add_argument("--max-response-len", type=int, default=24_576,
                   help="token budget for a whole hand: every reply (reasoning included), observation and the cleanup "
                        "turn after the task's first prompt (default 24576; AC2's own default is 8192)")
    g.add_argument("--max-total-len", type=int, default=32_768,
                   help="prompt plus that budget (default 32768)")
    g.add_argument("--inference-replicas", type=int, default=1,
                   help="inference replicas for rollouts, 2 GPUs each (default 1; AC2's own default is 4)")
    g.add_argument("--eval-replicas", type=int, default=1,
                   help="replicas for the in-run eval, 2 GPUs each (default 1; AC2's own default is 4)")
    g.add_argument("--no-thinking", dest="thinking", action="store_false",
                   help="don't render prompts with Qwen thinking on (the event has it on)")
    g.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", help="don't record TensorBoard logs")
    a = p.add_argument_group("left to AC2's defaults unless you set them")
    a.add_argument("--lr", type=float, help="learning rate")
    a.add_argument("--concurrency", type=int, help="concurrent hands on your harness")
    a.add_argument("--keep-checkpoints", type=int, help="checkpoints to keep")
    a.add_argument("--eval-dataset", help="in-run eval dataset (default <prefix>-select)")
    a.add_argument("--eval-concurrency", type=int, help="concurrent hands in the in-run eval")
    a.add_argument("--grading-timeout", type=float, help="seconds AC2 waits for your harness to grade a rollout")
    a.add_argument("--cluster-id")
    return p


def explicit_settings(p: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, Any]:
    skip = {"project", "dry_run", "help"} | (
        {"eval_every", "eval_dataset", "eval_concurrency", "eval_replicas"} if args.no_eval else set()
    )
    return {
        action.dest: getattr(args, action.dest)
        for action in p._actions  # noqa: SLF001
        if action.dest not in skip and getattr(args, action.dest, None) not in (None, False)
    }


def gpu_estimate(args: argparse.Namespace) -> str:
    """GPUs this run holds while it trains, from AC2's per-model defaults (docs: Training configuration)."""
    if args.model != "Qwen/Qwen3.6-35B-A3B":
        return "GPUs: see AC2's Training configuration docs (Default GPUs) for this model"
    inference = 2 * (args.inference_replicas or 4)
    evals = 0 if args.no_eval else 2 * (args.eval_replicas or 4)
    return f"GPUs while running: about {4 + inference + evals} (4 training + {inference} rollout inference + {evals} eval)"


def main() -> None:
    p = parser()
    args = p.parse_args()
    name = f"pokerbots-grpo-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(args, name)
    if args.dry_run:
        print("This launch sets (everything else is AC2's default):")
        for key, value in explicit_settings(p, args).items():
            print(f"  {key} = {value}")
        print(gpu_estimate(args))
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
