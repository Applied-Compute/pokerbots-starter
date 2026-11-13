# Offline OPSD: distill a strategy hint

At the event the harness owns the system prompt, so any strategy you want the model to follow has to be in its
weights. Offline OPSD (on-policy self-distillation) puts it there: the teacher is the base model with a hint in its
system prompt, the student is the same model without it, and training pulls the student's token distribution
toward the teacher's on replies the base model actually wrote.

[`hint.py`](hint.py) holds a deliberately generic tight-aggressive hint. Writing a better one is the point.

## Run

1. **Play base on the train deals** through the harness, served as an inference deployment:
   ```bash
   uv run python -m vals_tools.eval --dataset vals-poker-6max-train --deployment <project>/<base-deployment>
   ```
2. **Build the hinted dataset** from that eval's traces ([`build_dataset.py`](build_dataset.py)):
   ```bash
   uv run python opsd/build_dataset.py --eval <eval-id> --dataset vals-poker-6max-opsd-train --dry-run
   uv run python opsd/build_dataset.py --eval <eval-id> --dataset vals-poker-6max-opsd-train
   ```
3. **Train** ([`train.py`](train.py)): 20 steps of 64 decisions at lr 2e-6, a checkpoint every 10.
   ```bash
   uv run python opsd/train.py --dataset vals-poker-6max-opsd-train --dry-run
   uv run python opsd/train.py --dataset vals-poker-6max-opsd-train
   ```
4. **Evaluate the student without the hint:** serve a checkpoint and run it through the harness on
   `vals-poker-6max-eval`. The harness never shows the hint, so this measures what the weights learned.

## The `HintedTask` contract

There is no trace-to-`HintedTask` helper, so `build_dataset.py` builds the tasks itself. Each policy decision
becomes one task:

- `input`: the decision's prompt followed by the reply base wrote (its reasoning, text and `submit_action` call);
- `opsd_hint_selections`: one `HintSelection` with `rollout_prefix_len` at the end of the prompt and
  `HintMutation(item_boundary=1, action="append_to_previous_item")`, which appends the hint to the system message
  for the teacher only;
- `env_params.tools`: the tool schemas the model saw, `submit_action` and `update_scratchpad` ([`tools.json`](tools.json),
  generated from the harness and checked by a harness test). **Training rejects a transcript with tool calls unless
  its tools are listed here.**

The dataset is created with `required_types=["hinted"]`. Only clean decisions are kept: one `submit_action` call
and a reply under the token cap. The after-hand `update_scratchpad` turns are skipped.

## What offline mode can and can't do

- It replays base's own tokens, so it can only reweight behavior base already produces. A hint asking for something
  base never writes needs `rollout_mode="one_step"`, where the student samples its own reply.
- It is cheap: no inference engines and no opponent calls during training. Opponent calls happen only in steps 1
  and 4.

## Judge it

Compare the student with base on `vals-poker-6max-eval`, both served the same way, paired per deal, with
`python -m vals_tools.score`.
