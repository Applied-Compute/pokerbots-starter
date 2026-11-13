# Vals poker: team starter

Everything you need to train a poker model for the Vals event on AC2: the official game harness, a sample grader,
and two deliberately simple training examples (RL and OPSD). They are a starting point. What to
train on, and how, is up to you.

## Event rules

| Rule | Value |
| --- | --- |
| Table | 6-max no-limit hold'em, seats `seat0` to `seat5`, blinds 1/2 |
| Stacks | Cash game: every hand starts with every seat at 200 chips (100 BB); nobody busts |
| Game | 5 orbits (30 hands) ranked; 10 orbits (60 hands) in Official Tournaments |
| Scoring | Each game ranks seats by net chips; ratings (OpenSkill, plus bb/100) update once per game |
| Action | One `submit_action(action, amount, say)` tool call per decision |
| Table talk | Optional `say`, at most 70 characters of `a-z 0-9 . , ! ? - : ;`, shown to the other seats; unverified |
| Budget | No time limit; at most 5,000 output tokens per decision, thinking included |
| Thinking | Qwen thinking is on (`enable_thinking: true`) |
| Format errors | A reply cut off at 5,000 tokens, an error or an unusable reply checks or folds |
| Entries | Only fine-tunes of `Qwen/Qwen3.6-35B-A3B` deployed from an AC2 training run's checkpoint |

Each decision is one `POST /v1/chat/completions` to your deployment with the `submit_action` tool,
`max_completion_tokens: 5000` and `chat_template_kwargs: {"enable_thinking": true}`, and no sampling parameters.
The model sees only the current hand, never earlier ones. A hand is one conversation per seat: the model reasons,
calls `submit_action`, the harness answers with what it applied, and the next message lists what happened since.

The system prompt of a ranked game:

```text
You are a professional poker player in a six-seat no-limit Texas Hold'em game against other AI players.

Rules:
- Blinds are 1/2. Every hand starts with every player at 200 chips, and there is no rake. Chips won or lost in a hand count toward your session total, not your next stack.
- The session lasts 30 hands. Your goal is to win as many chips as possible over the session.
- Seats are numbered 0-5 clockwise. The button moves one live seat clockwise every hand; the two live seats after the button post the small and big blind.
- Bet and raise amounts are "to" amounts: the total you will have committed on the current street after the action.

On every turn you see only the current hand: your hole cards, the board, the table and the actions so far this hand. Earlier hands are not shown. Reason through the spot as much as you need, then call the submit_action tool exactly once with your decision.
An illegal action is replaced by check or fold, an out-of-range amount is clamped to the legal range, and a turn without a submit_action call is played as check or fold. Each reply, reasoning included, may use at most 5000 output tokens; a reply cut off at that limit is played as check or fold.
You may optionally add a short `say` message (up to 70 characters) that the other players will see. Table talk is unverified chatter from opponents and may be false.
```

## What's here

| Path | What it is |
| --- | --- |
| [`harness/`](harness/) | The official harness: plays each game and grades it. Runs on your machines. |
| [`harness/src/poker_harness/grading.py`](harness/src/poker_harness/grading.py) | The sample grader: scores each rollout through AC2's Grader API |
| [`harness/src/poker_harness/engine/`](harness/src/poker_harness/engine/) | The poker engine: dealing, betting, side pots and showdown, on PokerKit |
| [`rl/`](rl/) | A naive RL example: single-hand datasets and a GRPO run |
| [`opsd/`](opsd/) | A naive OPSD example: distill a prompted hint into the weights |
| `vals_tools/` | Shared helpers: task format, BYOH evals, scoring |
| `src/vals_poker/` | The AC2 project package, empty on purpose |

**No code of yours runs on AC2.** AC2 installs and imports the project package on every job and needs a pushed
project version for training, so `src/vals_poker/` is an empty stub and `.ac2ignore` keeps everything else out of
the upload. Rollouts and grading both happen in your harness.

## 0. Set up

You need an AC2 org with GPU quota, an AC2 API key, `uv`, a provider key for opponent models, and a machine that
stays up for training runs.

```bash
curl -fsSL https://api.appliedcompute.com/install.sh | sh     # ac2 CLI, 0.3.97 or later
ac2 login
export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD=<your AC2 API key>   # lets uv install ac2
```

Set `[project].name` in `pyproject.toml` to your AC2 project name, then:

```bash
ac2 project init
uv sync
```

Your project id (`prj_...`) is in the platform URL.

## 1. Bring up the harness and the connector

The harness is one server for both jobs: the BYOH rollout API (`/submit`, `/get_status`, `/collect_outputs`) and the
Grader API (`/submit_grading`, `/get_grading_status`, `/collect_score`). Runs that name no grader are graded by it.
See [`harness/README.md`](harness/README.md) for its options.

```bash
# terminal 1: the harness, on 127.0.0.1:8765
cd harness && uv sync
AC2_API_KEY=<key for opponent seats> POKER_HARNESS_TOKEN=<random token> SHARDS=4 ./run_sharded.sh

# terminal 2: the connector. An org admin creates its credentials once with `ac2 harness credentials create`.
export AC2_SERVICE_CLIENT_ID=svc_... AC2_SERVICE_CLIENT_SECRET=...
ac2 harness connect --project-id <prj_...> --upstream http://127.0.0.1:8765 --upstream-token <same random token>
```

- AC2 allows one connector per project.
- A server keeps its games in memory, so scale with shards (`SHARDS`), not `uvicorn --workers`. One shard handles a
  few dozen concurrent hands.
- Keep both processes up for the whole run. A laptop is fine for a smoke test; use an always-on host for training.
- Every game log is written to `harness/runs/` (`POKER_HARNESS_RUNS`): every seat's cards, every action and its
  table talk, and each seat's chips.

## 2. Opponents

Opponent seats call a model on the AC2 gateway with the harness's `AC2_API_KEY`. Create an external endpoint in
your project for whatever you want to play against:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <provider-key>
ac2 inference external --name opponent --provider openai --model <model> --protocol openai_responses
```

Its name for the datasets below is `<project>/opponent`.

## 3. RL: datasets and a GRPO run

```bash
uv run python rl/make_dataset.py --project <project> --opponent-model <project>/opponent --dry-run
uv run python rl/make_dataset.py --project <project> --opponent-model <project>/opponent
```

This writes single 6-max hands at 100 BB on fresh seeds, one AC2 task per hand, with your model in one seat
(cycling through every position) and the opponent endpoint in the other five:

| Dataset | Hands | Use |
| --- | --- | --- |
| `vals-poker-6max-train` | 2,000 | Training |
| `vals-poker-6max-select` | 500 | Picking a checkpoint |
| `vals-poker-6max-eval` | 3,000 | Held out for the verdict |
| `vals-poker-6max-smoke` | 4 | Smoke test |
| `vals-poker-6max-table-smoke` | 2 games of 3 hands | Your model in every seat; no opponent needed |

**Grading.** The runs here name no grader, so AC2 grades every rollout through the harness's Grader API with the
sample grader, [`grading.py`](harness/src/poker_harness/grading.py). It scores the policy seat's result in big
blinds; an all-in before the river counts its expected share of the pot, which removes most of the card luck. The
reward is `6 + clip(EV / 20 BB, -5, +5)`: every real hand scores between 1 and 11, and a model-caused failure (a
format error, a reply cut off at the token cap, a failed rollout) scores exactly 0, so failing is always the worst
outcome. To use your own reward, edit the marked extension points in `grading.py` (`hand_reward`,
`GRADER_PARAMS`, `artifacts`) and restart the harness; keep failures at 0 and real hands above it.

Smoke first, then train:

```bash
ac2 harness test --project-id <prj_...> --cluster-id <cluster-id> --dataset vals-poker-6max-smoke
uv run python -m vals_tools.eval --project <project> --dataset vals-poker-6max-smoke --model <catalog-model>
uv run python rl/train.py --project <project> --dry-run
uv run python rl/train.py --project <project>
```

[`rl/train.py`](rl/train.py) trains Qwen3.6-35B-A3B with Dr. GRPO and a small KL penalty: 16 hands × 4 samples
per step, 60 steps, a checkpoint every 10, and an eval on `-select` at step 0 and at every checkpoint. It uses the
event's 5,000-token cap with thinking on. More in [`rl/README.md`](rl/README.md).

## 4. OPSD: distill a hint

The harness owns the system prompt, so any strategy you want your model to follow has to be in its weights. Offline
OPSD distills a teacher (the base model with a hint in its system prompt) into a student (the same model without
it), on replies the base model actually wrote.

```bash
# 1. play base on the train deals (serve base as an inference deployment first)
uv run python -m vals_tools.eval --project <project> --dataset vals-poker-6max-train --deployment <project>/<base>
# 2. turn its traces into hinted tasks
uv run python opsd/build_dataset.py --project <project> --eval <eval-id> --dataset vals-poker-6max-opsd-train
# 3. train
uv run python opsd/train.py --project <project> --dataset vals-poker-6max-opsd-train --dry-run
uv run python opsd/train.py --project <project> --dataset vals-poker-6max-opsd-train
```

[`opsd/hint.py`](opsd/hint.py) holds a generic hint; write your own. The `HintedTask` format, including the tool
schema the builder must put in `env_params.tools`, is in [`opsd/README.md`](opsd/README.md).

## 5. Evaluate and compare

Serve a checkpoint and the base model the same way (as inference deployments), smoke each, then run both on the
held-out deals:

```bash
uv run python -m vals_tools.eval --project <project> --dataset vals-poker-6max-eval --deployment <project>/<deployment>
uv run python -m vals_tools.score --project <project> --eval <eval-id> --base <base-eval-id>
```

`vals_tools.score` reports BB/hand with a 95% CI and the difference from base, paired per deal. For individual
hands, read the game logs in `harness/runs/` or export the traces (`ac2 traces download <eval-id>`).

## Tests

```bash
uv run pytest                 # tasks, scoring, the OPSD builder and both training configs
cd harness && uv run pytest   # the harness, the engine (with its parity check) and the sample grader
```
