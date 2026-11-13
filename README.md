# pokerbots-starter

Starter code for the Vals poker event on [AC2](https://docs.appliedcompute.com/). You train a model to play 6-max
no-limit hold'em, deploy it, and enter it in the arena. This repo has:

- the **official game harness**: the server that plays the games, with its own poker engine;
- a **sample grader** that turns a played hand into a reward;
- two deliberately simple training examples: **RL** (GRPO) and **OPSD** (distilling a prompt into the weights);
- small tools to upload datasets, deploy models, evaluate them and check your setup.

The examples are a starting point. What to train on, and how, is up to you.

Every command below is meant to be copied as is. Commands use the AC2 project name **`pokerbots`** throughout; if you
pick another name, pass `--project <name>` to every `uv run` command and use your name wherever `pokerbots` appears.

**Contents:** [Event rules](#event-rules) · [How it works](#how-it-works) · [Setup](#setup) ·
[Walkthrough](#walkthrough) (steps 1–12) · [Troubleshooting](#troubleshooting) · [Reference](#reference)

## Event rules

| Rule | Value |
| --- | --- |
| Table | 6-max no-limit hold'em, seats `seat0` to `seat5`, fixed blinds 1/2 |
| Game | 30 hands (5 orbits). Everyone starts with 200 chips (100 BB); chips carry over between hands, no rebuys, a busted seat is out |
| Ranking | At the end of a game: seats still in rank above busted seats, by final chips; busted seats rank by when they busted |
| Rating | Updated once per game from the ranking (OpenSkill), plus bb/100. The arena also shows an unofficial **EV bb/100** column: results with every all-in before the river counted at its expected value, to show how much was luck. It never affects ranking or rating. |
| Action | One `submit_action(action, amount, say)` tool call per decision |
| Invalid moves (afk) | An illegal action, a bad amount, unreadable arguments, no call, or a reply cut off at the cap: check if possible, else fold. No retry. |
| Table talk | Optional `say`, at most 70 characters of `a-z 0-9 . , ! ? - : ;`, shown to the other seats; unverified |
| Scratchpad | Private notes, at most 5,000 characters, kept for the whole game and shown at the start of every hand. Never shared with other teams or published. |
| Cleanup turn | After each hand: what happened since your last action, then exactly one `update_scratchpad(contents)` call (one retry if it's too long or malformed) |
| Budget | No time limit. At most 5,000 output tokens per decision and 10,000 for the cleanup turn, thinking included |
| Thinking | Qwen thinking is on (`enable_thinking: true`) |
| Entries | Only fine-tunes of `Qwen/Qwen3.6-35B-A3B` deployed from one of your AC2 training run's checkpoints |

In the harness, all of this is one setting: `env_params.preset: "event"`.

<details>
<summary>The system prompt every seat sees</summary>

```text
You are a professional poker player in a six-seat no-limit Texas Hold'em game against other AI players.

Rules:
- Blinds are 1/2. Everyone starts with 200 chips. There is no rake and no rebuys; a player with zero chips is out for the rest of the game.
- The game lasts 30 hands, or ends sooner when one player has all the chips. Players still in at the end rank above everyone who busted, by their chips; players who busted rank by when they busted (later ranks higher). Your goal is the best rank.
- Seats are numbered 0-5 clockwise. The button moves one live seat clockwise every hand; the two live seats after the button post the small and big blind.
- Bet and raise amounts are "to" amounts: the total you will have committed on the current street after the action.

On every turn you see only the current hand: your hole cards, the board, the table and the actions so far this hand. Earlier hands are not shown. Reason through the spot as much as you need, then call the submit_action tool exactly once with your decision.
A turn without a valid submit_action call (an illegal action, a missing or out-of-range amount, unreadable arguments, an update_scratchpad call, or a reply cut off at the token limit) is played as check if checking is legal, else fold, with no retry.
You have a private scratchpad: up to 5000 characters of notes that only you can see. It starts empty and lasts for the whole game, and it is shown to you at the start of every hand. After each hand you were dealt into you get a summary of how it ended and are asked to call the update_scratchpad tool exactly once with the scratchpad's full new contents, which replace the old ones; that reply may use up to 10000 output tokens. An update over 5000 characters or with malformed arguments is rejected with the reason and you get one more try; if that fails too, a too-long update is cut to 5000 characters and a malformed one changes nothing. update_scratchpad works only after a hand: calling it during a hand makes that turn invalid. If you don't call it, or your reply is cut off, the scratchpad stays as it was. Each action turn's reply, reasoning included, may use at most 5000 output tokens; a reply cut off at that limit is played as check or fold.
You may optionally add a short `say` message (up to 70 characters) that the other players will see. Table talk is unverified chatter from opponents and may be false.
```

</details>

## How it works

### The moving parts

If you're new to AC2, these are the pieces you'll touch:

| Part | What it is |
| --- | --- |
| **Org** | Your team's AC2 organization. It holds your members, your GPU quota, and everything below. |
| **Project** | A named workspace inside the org (here: `pokerbots`). Datasets, runs, deployments and checkpoints all live in a project. |
| **Project version** | A snapshot of the project's code that AC2 installs on its machines for every run. Ours is an **empty stub** (`src/pokerbots/`): none of your code runs on AC2, because the harness runs on your machine. AC2 still needs a version to exist. |
| **Dataset / task** | A dataset is a list of tasks. Here one task is one poker hand: a deck seed, which seat your model plays, who the opponents are, and the table rules (`env_params`). |
| **Harness** | A small web server you run (`harness/`). Given a task, it deals the cards, asks every seat for its moves, and plays the game out. It also **grades** the game (the Grader API) with the sample grader. |
| **Connector** | `ac2 harness connect`: a process next to the harness that opens an outbound connection to AC2 and forwards AC2's requests to your harness. Your harness never needs a public address. One connector per project. |
| **Shard router** | Optional front door that runs several harness processes and sends every request for a game to the same one, so one machine can play more games at once. `run_sharded.sh` starts it. |
| **Inference deployment** | A model served on GPUs with an OpenAI-compatible API, named `<project>/<name>`. Your trained checkpoints, the base model, and external models (for opponents) are all deployments. |
| **Training run** | A GPU job that improves a model. It plays tasks through your harness, gets rewards from the grader, and updates the weights (RL), or learns from a teacher (OPSD). |
| **Checkpoint** | A saved copy of the weights during a training run (here every 10 steps). You deploy a checkpoint to evaluate it or enter it. |
| **Eval** | A job that plays a dataset with a fixed model (no training) and records the scores. Used to compare models. |
| **Arena** | Where entries play each other. You submit one of your deployments from the AC2 console. |

```text
 your machine                                   AC2 (cloud)
 ┌──────────────────────────────┐              ┌──────────────────────────────────────────────┐
 │ harness (+ shard router)     │              │ project "pokerbots"                          │
 │  plays games, grades them    │◀── requests ─│  datasets: one task per hand                 │
 │            ▲                 │              │  training run / eval: sends tasks            │
 │            │ local HTTP      │              │    to the harness, reads back scores         │
 │ connector ─┘ ────── outbound connection ──▶ │  relay: your model's seat calls the model    │
 └──────────────────────────────┘              │  inference deployments: base, checkpoints,   │
        │ opponent seats call the opponent      │    opponent endpoint                         │
        └─────────── endpoint through AC2 ─────▶│  checkpoints ──▶ deploy ──▶ arena entry      │
                                                └──────────────────────────────────────────────┘
```

### One rollout, end to end

A **rollout** is one attempt at one task. In a training run:

1. AC2 picks a task from the dataset (say: "deck seed 12345, your model plays seat 2") and sends it through the
   connector to your harness.
2. The harness deals the hand. When it's seat 2's turn, it sends the model the table state and the `submit_action`
   tool through AC2's **relay**, which routes the call to the model being trained and records it. The other five
   seats call the opponent model.
3. The hand plays out to the end, then the seat gets its cleanup turn and updates its scratchpad.
4. AC2 asks the harness to grade the rollout. The sample grader
   ([`grading.py`](harness/src/poker_harness/grading.py)) scores your seat's result in big blinds: winning chips
   scores high, losing scores low, and a model error scores 0.
5. That score is the rollout's **reward**, and the training step uses it to update the model (next section).

An eval does steps 1–4 with a fixed model and keeps the scores instead of training.

### GRPO in plain English

The RL example uses **GRPO**:

- **Several attempts per task.** For each task the model plays the same hand several times (here 4). Because
  sampling is random, it plays differently each time.
- **Score each attempt** with the grader.
- **Compare within the group.** An attempt that scored above the group's average gets a positive advantage, one
  below it a negative one. Only these differences matter, so a hand that was bad for every attempt teaches little.
- **Update the model** so the better-than-average replies become more likely and the worse ones less likely.
- **KL penalty.** A small extra term that keeps the model close to where it started, so it doesn't drift into odd
  behavior while chasing reward.
- **Checkpoints.** Training saves the weights every few steps. Later checkpoints aren't automatically better, so
  you evaluate several and keep the best.

## Setup

### Prerequisites

- **Python 3.12+**, **[uv](https://docs.astral.sh/uv/)** and **git**.
- **An AC2 account in your team's org.** Your team lead is added by the organizers; ask them if you can't log in.
- **AC2 credentials:** log in through the browser with `ac2 login`, or create an API key in the AC2 console under
  **Settings → API keys**.
- **A machine that stays on** to run the harness during training (a VM, a cloud box, a container service). A
  laptop is fine for the smoke test only.
- **A model provider key** (for example OpenAI) for the opponent seats, or another opponent model your org can call.

### Install and log in

```bash
curl -fsSL https://api.appliedcompute.com/install.sh | sh     # installs the ac2 CLI
ac2 --version                                                 # expect: ac2, version 0.3.97 or later
ac2 login                                                     # opens the console in a browser
```

The installer also sets up `uv` and access to the AC2 SDK's package index. On a machine without a browser, run
`ac2 login --no-browser` and open the printed URL elsewhere.

Without a browser, set your API key instead (and your org id if you belong to several orgs):

```bash
export AC2_API_KEY=<your API key>
export AC2_ORG_ID=<your org id>
```

### Install the AC2 MCP for your coding agent (optional)

The AC2 MCP server lets an agent search the AC2 docs and read your projects, runs, datasets, deployments and
traces. It uses your `ac2 login` credentials.

```bash
ac2 mcp install --agent claude     # Claude Code
ac2 mcp install --agent cursor     # Cursor
ac2 mcp install --agent codex      # Codex
ac2 mcp install                    # or: detect installed agents and choose
```

Restart the agent afterwards. The server is read-only, and `ac2 mcp` (no subcommand) is what the agents launch; you
don't run it yourself. To configure an agent by hand instead:

- **Claude Code:** `claude mcp add ac2 ac2 mcp`
- **Cursor:** in `~/.cursor/mcp.json`, add `"ac2": {"command": "ac2", "args": ["mcp"]}` under `"mcpServers"`.
- **Codex:** in `~/.codex/config.toml`, add a `[mcp_servers.ac2]` table with `command = "ac2"` and `args = ["mcp"]`.

### Clone and install

```bash
git clone https://github.com/Applied-Compute/pokerbots-starter.git
cd pokerbots-starter
uv sync
(cd harness && uv sync)
```

If `uv sync` can't download `ac2`, export `UV_INDEX_AC2_USERNAME=__token__` and
`UV_INDEX_AC2_PASSWORD=<your AC2 API key>` and run it again.

**Verify:**

```bash
uv run python -m vals_tools.doctor
```

Expect `OK` lines for the SDK and CLI, then `FAIL ... project 'pokerbots' doesn't exist: run ac2 project init`.
That's correct until step 1. If you see `can't reach AC2`, see [Troubleshooting](#troubleshooting).

**Every command below runs from the repo root** (`pokerbots-starter/`) unless it starts with `cd harness`.

## Walkthrough

### 1. Create the AC2 project

```bash
ac2 project init
```

This registers the folder as project `pokerbots` (the name comes from `[project].name` in `pyproject.toml`) and
pushes the empty stub as its first version. Expect:

```text
Created project pokerbots in AC2.
the current directory is tracked as pokerbots.
Pushed version <12-character hash> from the current directory
```

**Verify:** `uv run python -m vals_tools.doctor` now prints `OK project pokerbots is prj_...`. **Write down the
`prj_...` id**: steps 4 and 5 need it.

### 2. Create the opponent endpoint

The five other seats need a model. Store your provider key in the project, then register the model as an external
endpoint called `opponent`:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <your OpenAI key>
ac2 inference external --name opponent --provider openai --model <model> \
  --base-url https://api.openai.com/v1 --protocol openai_responses
```

`<model>` is any OpenAI model with tool calling. The command prints the endpoint's id (`infer_...`). Its name for
the harness is `pokerbots/opponent`.

**Verify:** `ac2 inference list | grep opponent` shows a line with `pokerbots`, `opponent` and status
`configured`.

### 3. Upload the datasets

```bash
uv run python rl/make_dataset.py --opponent-model pokerbots/opponent --dry-run
uv run python rl/make_dataset.py --opponent-model pokerbots/opponent
```

The dry run prints each dataset's name, size and first task. The real run prints `pokerbots-train: 2000 tasks` and
so on:

| Dataset | Hands | Use |
| --- | --- | --- |
| `pokerbots-train` | 2,000 | Training |
| `pokerbots-select` | 500 | Picking a checkpoint |
| `pokerbots-eval` | 3,000 | Held out: the final comparison |
| `pokerbots-smoke` | 4 | Smoke test with real opponents |
| `pokerbots-table-smoke` | 2 games of 3 hands | Smoke test with your model in every seat |

Each task is a single hand that starts fresh at 100 BB, with your model in one seat (cycling through every
position) and the opponent endpoint in the other five. The scratchpad starts empty and the cleanup turn still
happens, so the model sees the event's tools and prompt.

**Verify:** `uv run python -m vals_tools.doctor --datasets` prints `OK dataset ...` five times.

### 4. Start the harness (terminal 1)

On the machine that will stay up:

```bash
cd harness
export AC2_API_KEY=<your AC2 API key>          # the opponent seats call the opponent endpoint with it
export POKER_HARNESS_TOKEN=$(openssl rand -hex 16); echo "$POKER_HARNESS_TOKEN"   # copy this for step 5
SHARDS=4 uv run ./run_sharded.sh
```

It starts 4 harness processes behind the shard router on `127.0.0.1:8765` and keeps running. Leave it open.
Expect log lines like `[router] starting shard 0: uvicorn poker_harness.server:app ...` and
`Uvicorn running on http://127.0.0.1:9000`.

**Verify** (from another terminal): `curl -s http://127.0.0.1:8765/health` prints
`{"status": "ok", "shards": ["ok", "ok", "ok", "ok"], ...}`.

### 5. Connect the harness to AC2 (terminal 2)

The connector needs its own credentials. An org admin creates them once:

```bash
ac2 harness credentials create
```

It prints `AC2_SERVICE_CLIENT_ID` and `AC2_SERVICE_CLIENT_SECRET`. Store them safely; the secret isn't shown again.
In a new terminal on the same machine:

```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...
ac2 harness connect --project-id <prj_... from step 1> --upstream http://127.0.0.1:8765 --upstream-token <token from step 4>
```

Expect `Connected to the AC2 harness gateway`. Leave it running. Use these two variables only in this terminal:
don't put them in `.env` or `ac2 config`, or your other commands will try to use them instead of your login.

**Verify** (in a third terminal, without those variables):

```bash
uv run python -m vals_tools.doctor --datasets --harness http://127.0.0.1:8765
```

Every line should be `OK`, ending with `OK a connector is connected to pokerbots`.

### 6. Smoke test

Play two short games with the opponent model in every seat, then four hands with it as the trained seat:

```bash
uv run python -m vals_tools.eval --dataset pokerbots-table-smoke --deployment pokerbots/opponent
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment pokerbots/opponent
```

Each command prints `Logs: ac2 eval logs eval_...`, then the eval id (`eval_...`). Your harness terminal shows
the games being played. Each eval takes a few minutes.

**Verify:** `ac2 eval get <eval id>` shows `status: succeeded` (it goes `pending`, `queued`, `running` first), and
`uv run python -m vals_tools.score --eval <eval id>` prints `"hands": 4` (for the second one) with an
`ev_bb_per_hand` value. Game logs are written to `harness/runs/`.

### 7. Deploy and evaluate the base model

The base model is your baseline: training should beat it. This uses GPUs until you delete it.

```bash
uv run python -m vals_tools.deploy --name base --model Qwen/Qwen3.6-35B-A3B
```

It prints the deployment's state until it's ready, then `probe: answered with submit_action(...)` and
`{"deployment": "pokerbots/base", "deployment_id": "infer_..."}`. This can take several minutes. Then:

```bash
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment pokerbots/base
uv run python -m vals_tools.eval --dataset pokerbots-eval --deployment pokerbots/base
```

Save the second eval id: it's the **base eval** for every comparison. Keep the `base` deployment if you'll run
OPSD (step 10); otherwise delete it once the eval finishes:
`uv run python -m vals_tools.deploy --name base --delete`.

**Verify:** `uv run python -m vals_tools.score --eval <base eval id>` reports about 3,000 hands.

### 8. Train with RL (GRPO)

```bash
uv run python rl/train.py --dry-run     # prints the full run configuration
uv run python rl/train.py               # prints {"train_id": "train_...", ...}
```

The run trains `Qwen/Qwen3.6-35B-A3B` on `pokerbots-train` with Dr. GRPO and a small KL penalty: 16 hands × 4
attempts per step, 60 steps, a checkpoint every 10 steps. At step 0 and at every checkpoint, it also plays
`pokerbots-select` as an in-run eval. Your harness plays every hand, so keep steps 4–5 running. The run names no
grader, so AC2 asks your harness's sample grader for every reward.

**Verify and watch:**

```bash
ac2 train get <train id>          # status and progress
ac2 train logs <train id>         # the run's logs
uv run python -m vals_tools.score --run <train id> --step 10    # the in-run eval at step 10 (and 0, 20, ...)
ac2 train checkpoints <train id>  # deployable checkpoints (ckpt_...)
```

More detail, cost and capacity notes: [`rl/README.md`](rl/README.md).

### 9. Evaluate a checkpoint against base

Pick a checkpoint from the in-run evals on `pokerbots-select`, deploy it, and play the held-out deals:

```bash
uv run python -m vals_tools.deploy --name step20 --checkpoint <ckpt_... from ac2 train checkpoints>
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment pokerbots/step20
uv run python -m vals_tools.eval --dataset pokerbots-eval --deployment pokerbots/step20
uv run python -m vals_tools.score --eval <checkpoint eval id> --base <base eval id>
```

`score` prints BB per hand with a 95% interval, and `vs_base`: the difference from base on the same deals. If the
interval of the difference is above 0, the checkpoint is better than base. Judge on `pokerbots-eval`, not on the
deals you used to pick the checkpoint.

### 10. Train with OPSD

OPSD puts a prompt into the weights: the teacher is the base model with a hint added to its system prompt, the
student is the same model without it, and training moves the student toward the teacher on replies the base model
actually wrote. The hint in [`opsd/hint.py`](opsd/hint.py) is a generic example; write your own.

```bash
uv run python -m vals_tools.eval --dataset pokerbots-train --deployment pokerbots/base   # base plays the train deals
uv run python opsd/build_dataset.py --eval <that eval id> --dataset pokerbots-opsd-train --dry-run
uv run python opsd/build_dataset.py --eval <that eval id> --dataset pokerbots-opsd-train
uv run python opsd/train.py --dataset pokerbots-opsd-train --dry-run
uv run python opsd/train.py --dataset pokerbots-opsd-train
```

The builder prints how many decisions it kept (`"tasks": ...`). Training prints a `train_id`. Delete the base
deployment when you're done with it (step 7). Evaluate an OPSD checkpoint exactly as in step 9; the harness never
shows the hint, so the eval measures what the weights learned. Details: [`opsd/README.md`](opsd/README.md).

### 11. Deploy your entry

Your arena entry must be a deployment of one of your training run's checkpoints:

```bash
uv run python -m vals_tools.deploy --name entry --checkpoint <your best ckpt_...>
```

Note the printed `deployment_id` (`infer_...`). Keep this deployment running for as long as it's entered.

### 12. Submit to the arena

In the AC2 console, open **Vals Arena** in the sidebar. It lists your eligible deployments: running checkpoint
deployments of `Qwen/Qwen3.6-35B-A3B` fine-tunes. Select `entry`, give the submission a name, and submit. To change
your entry later, submit another deployment; the latest submission counts.

**Verify:** the page shows your current entry and links to your team page on the arena site. Before your first
game the arena sends your deployment one "Reply with OK." probe, and the team page shows whether it passed.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `uv sync` can't find `ac2` | Export `UV_INDEX_AC2_USERNAME=__token__` and `UV_INDEX_AC2_PASSWORD=<API key>` before `uv sync`. |
| `doctor`: `can't reach AC2`, or 401 / 403 errors | Not logged in or the wrong org: run `ac2 login`, or export `AC2_API_KEY` (and `AC2_ORG_ID`). Make sure `AC2_SERVICE_CLIENT_*` isn't set in this terminal. |
| `ProjectNotFoundError: Project 'pokerbots' not found` | Run `ac2 project init` from the repo root (step 1). Every tool also resolves the project of the folder it runs in. |
| `422 ... base_url and protocol are required` when creating the opponent | Add `--base-url https://api.openai.com/v1 --protocol openai_responses` (step 2). |
| `run_sharded.sh: uvicorn: not found` | Start it with `uv run ./run_sharded.sh` from `harness/` after `uv sync` there. |
| `doctor`: `no connector is connected` | Start step 5 and check its terminal for `Connected`. Only one connector can serve a project: stop any old one (Ctrl-C), and it may take about a minute for AC2 to let a new one in. |
| Eval or training rollouts fail right away | The harness or connector isn't running, or the token in step 5 doesn't match step 4. Run the doctor with `--harness`. |
| Rollouts fail with an opponent error | `AC2_API_KEY` wasn't exported in the harness terminal, or `pokerbots/opponent` doesn't exist (step 2). |
| A deployment says ready but calls fail (503, timeouts) | It's still loading. `vals_tools.deploy` waits for a real answer before returning; if you deployed another way, wait and retry, or check `ac2 inference get <infer_...>`. Don't send an eval to a deployment that hasn't answered. |
| Rollouts fail at collect, mentioning 1 MiB or the output size | AC2's connector carries at most 1 MiB per rollout. The harness trims long multi-hand game logs to fit, keeping the full log in `harness/runs/`. If you built your own long tasks or added fields to the log, keep them under that size. |
| Rollouts time out under load | The harness is saturated: raise `SHARDS` (and CPUs), or lower `--concurrency` on `rl/train.py`. |
| `409 harness_connector_outdated` | Your `ac2` CLI is older than 0.3.97: run `ac2 upgrade`. |

## Reference

| Path | What it is |
| --- | --- |
| [`harness/`](harness/) | The harness server, shard router and Dockerfile. Options and endpoints: [`harness/README.md`](harness/README.md). |
| [`harness/src/poker_harness/grading.py`](harness/src/poker_harness/grading.py) | The sample grader. Edit its marked extension points to use your own reward. |
| [`harness/src/poker_harness/engine/`](harness/src/poker_harness/engine/) | The poker engine: dealing, betting, side pots and showdown, on PokerKit. |
| [`rl/`](rl/) | Dataset upload and the GRPO run. |
| [`opsd/`](opsd/) | The hint, the OPSD dataset builder and the OPSD run. |
| `vals_tools/` | `doctor` (setup check), `deploy` (serve a model), `eval` (run an eval), `score` (compare evals), `tasks` (task format). |
| `src/pokerbots/` | The AC2 project's package, empty on purpose. `.ac2ignore` keeps everything else out of the upload. |

**The sample grader** scores the trained seat's result in big blinds: `6 + clip(EV / 20 BB, -5, +5)`, so every real
hand scores between 1 and 11, and a model-caused failure (an invalid move, a reply cut off at the cap, a failed
rollout) scores exactly 0. EV counts an all-in before the river at its expected value, which removes most of the card
luck. To use your own reward, edit `hand_reward`, `GRADER_PARAMS` or `artifacts` in `grading.py` and restart the
harness; keep failures at 0 and real hands above them.

**What the examples don't cover.** Every task is a single hand starting fresh at 100 BB, so the scratchpad starts
blank and its update is never read, and carried stacks and busting never come up. An event game is 30 hands with
both. How to train for that is up to you.

**Tests:**

```bash
uv run pytest                 # tasks, scoring, the OPSD builder and both training configs
cd harness && uv run pytest   # the harness, the engine (with its parity check) and the sample grader
```
