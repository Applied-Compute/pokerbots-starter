"""Offline OPSD: distill the hinted teacher into a student that never sees the hint.

Offline mode replays each stored reply from `build_dataset.py`: the student scores it on the plain harness prompt,
the frozen teacher (the base model) on the same prompt plus the hint, and the loss is their reverse KL on the reply's
tokens. No environment, grader or reward runs in training, and no inference engine is needed.

Every setting this script sends is a flag below. A flag left unset is not sent, so AC2 uses its own default for it
(learning rate, response and context length, ...). Offline mode itself requires one sample per task, no inference
replicas and no in-run eval, so those are always set. `--dry-run` prints the settings and the exact configuration.

Training has no poker eval: after the run, serve a checkpoint and play it through the harness, which never shows the
hint, so that eval measures the student without it.

    uv run python opsd/train.py --dataset pokerbots-opsd-train --dry-run
    uv run python opsd/train.py --dataset pokerbots-opsd-train --lr 1e-6 --dry-run   # override an AC2 default
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

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



def _set(**values: Any) -> dict[str, Any]:
    """Only the values that were given: anything None is left to AC2's default."""
    return {key: value for key, value in values.items() if value is not None}


def run_config(args: argparse.Namespace, name: str) -> RunConfig:
    launch = _set(cluster_id=args.cluster_id, tensorboard_enabled=args.tensorboard or None)
    return RunConfig(
        model=args.model,
        name=name,
        description=f"Offline OPSD of a prompted strategy hint into {args.model} (student without the hint), "
        f"{args.dataset}.",
        tags={"benchmark": "pokerbots", "method": "opsd", "dataset": args.dataset},
        experiment_cfg=ExperimentConfig(
            method=OPSDConfig(rollout_mode="offline"),
            step=StepConfig(
                samples_per_problem=1,  # required by offline mode: it replays one stored reply per task
                **_set(
                    num_train_steps=args.steps,
                    problem_batch_size=args.batch,
                    checkpoint_interval=args.checkpoint_every,
                    keep_last_checkpoints=args.keep_checkpoints,
                ),
            ),
            tasks=TasksConfig(
                datasets=(TrainDatasetConfig(ac2_dataset_name=args.dataset),),
                **_set(apply_chat_template_kwargs={"enable_thinking": True} if args.thinking else None),
            ),
            trainer=TrainerConfig(**_set(learning_rate=args.lr)),
            rollout=RolloutConfig(
                n_inference_replicas=0,  # required by offline mode: nothing is sampled
                **_set(max_response_len=args.max_response_len, max_total_len=args.max_total_len),
            ),
            eval=EvalConfig(mode="off"),  # required here: the poker eval runs through the harness afterwards
        ),
        **_set(internal_cfg=InternalConfig(launch=InternalLaunchConfig(**launch)) if launch else None),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="pokerbots")
    p.add_argument("--dataset", required=True, help="hinted dataset from build_dataset.py")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="base model (arena entries must fine-tune this one)")
    p.add_argument("--dry-run", action="store_true", help="print the settings and configuration, submit nothing")
    g = p.add_argument_group("set by default (the OPSD recipe); change them with these flags")
    g.add_argument("--steps", type=int, default=20, help="training steps (default 20)")
    g.add_argument("--batch", type=int, default=64, help="decisions per step (default 64)")
    g.add_argument("--checkpoint-every", type=int, default=10, help="steps between checkpoints (default 10)")
    g.add_argument("--no-thinking", dest="thinking", action="store_false",
                   help="don't render with Qwen thinking on (the event has it on)")
    g.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", help="don't record TensorBoard logs")
    a = p.add_argument_group("left to AC2's defaults unless you set them")
    a.add_argument("--lr", type=float, help="learning rate")
    a.add_argument("--max-response-len", type=int, help="per-reply token limit")
    a.add_argument("--max-total-len", type=int, help="prompt plus reply token limit")
    a.add_argument("--keep-checkpoints", type=int, help="checkpoints to keep")
    a.add_argument("--cluster-id")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    name = f"pokerbots-opsd-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(args, name)
    if args.dry_run:
        print("This launch sets (everything else is AC2's default):")
        skip = {"project", "dry_run", "help"}
        for action in p._actions:  # noqa: SLF001
            value = getattr(args, action.dest, None)
            if action.dest not in skip and value not in (None, False):
                print(f"  {action.dest} = {value}")
        print("  samples_per_problem = 1, n_inference_replicas = 0, eval = off (required by offline mode)")
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
