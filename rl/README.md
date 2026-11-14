# RL: GRPO on single 6-max hands

[`train.py`](train.py) trains Qwen3.6-35B-A3B on `pokerbots-train` through your self-hosted harness:

- **Rollouts:** each hand is played by the harness; the trained seat calls the run's inference engine through the
  rollout relay and the five opponents call your opponent endpoint.
- **Reward:** harness-side grading, `6 + clip(EV / 20 BB, -5, +5)`, failures 0 (see the [main README](../README.md)).
  The run names no grader.
- **Method:** Dr. GRPO (`--advantage dr-grpo`) with a small KL penalty (`--kl-coef 0.01`), 16 hands per step
  (`--hands-per-step`) with 4 attempts each (`--samples-per-hand`), 60 steps (`--steps`).
- **Replicas:** 1 rollout and 1 eval replica (`--inference-replicas`, `--eval-replicas`), about 8 GPUs in all.
  AC2's own default for this model is 4 of each, 20 GPUs, more than a team's 10-GPU quota.
- **Everything else is AC2's default** (learning rate, response and context length, sampling concurrency,
  checkpoints kept) unless you pass the matching flag. `--dry-run` lists every setting the launch sends,
  then the exact configuration.
- **Scratchpad:** tasks use the event preset, so every rollout ends with the after-hand `update_scratchpad` turn.
  In a single-hand task that update is never read, and the grader ignores it.
- **Token budget:** `--max-response-len 24576` (and `--max-total-len 32768`). In training this is a budget for the
  whole hand: every reply's reasoning, every observation and the cleanup turn. AC2's default (8,192) runs out after a
  couple of long decisions, and the rest of the hand is played as check or fold. Raise it for longer hands.
- **Checkpoints and eval:** a checkpoint every 10 steps (`--checkpoint-every`), and an in-run eval on
  `pokerbots-select` at step 0 (base) and every 10 steps (`--eval-every`, or `--no-eval`).

## Run

With the harness and connector running and the datasets uploaded:

```bash
uv run python rl/train.py --project <project> --dry-run     # print the resolved config
uv run python -m vals_tools.eval --project <project> --dataset pokerbots-smoke --model <catalog-model>   # harness smoke first
uv run python rl/train.py --project <project>
```

`--concurrency` caps concurrent hands on your harness (default 64, one step). `--steps`, `--lr` and `--prefix`
change the obvious things.

## Cost and capacity

- **GPUs:** about 8 while running: 4 to train, 2 for the rollout replica, 2 for the in-run eval replica.
- **Opponent calls:** every hand makes several opponent calls per seat, so 64 hands a step is a few hundred opponent
  calls a step. Check your opponent endpoint's price and rate limits before a long run.
- **Harness:** one shard handles a few dozen concurrent hands. Run enough shards (`SHARDS=`) for the run's
  concurrency plus its sidecar eval.

## Judge it

1. Pick a checkpoint from the sidecar on `-select`, compared with step 0.
2. Serve that checkpoint and base the same way (as inference deployments) and evaluate both on
   `pokerbots-eval`.
3. `uv run python -m vals_tools.score --project <project> --eval <checkpoint-eval> --base <base-eval>`.
   The difference from base is paired per deal; its 95% CI tells you whether it is more than noise.
