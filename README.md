# pokerbots-starter

Starter code for the Vals poker event on [AC2](https://docs.appliedcompute.com/). You train a model to play 6-max
no-limit hold'em, deploy it, and enter it in the arena. This repo has:

- the **official game harness**: the server that plays the games, with its own poker engine;
- a **sample grader** that turns a played hand into a reward;
- two deliberately simple training examples: **RL** (GRPO) and **OPSD** (distilling a prompt into the weights);
- small tools to upload datasets, deploy models, evaluate them and check your setup.

The examples are a starting point. What to train on, and how, is up to you.

Every command below is meant to be copied as is. They all use **`vals-hackathon`**, the AC2 project your team org
already has (see [Your team org](#your-team-org)); every tool here defaults to it. Dataset names start with
`pokerbots-`.

**Contents:** [Event rules](#event-rules) · [Your team org](#your-team-org) · [How it works](#how-it-works) ·
[Managing limited compute](#managing-limited-compute) · [Setup](#setup) · [Walkthrough](#walkthrough) (steps 1–12) ·
[While your training run is going](#while-your-training-run-is-going) · [Troubleshooting](#troubleshooting) ·
[Reference](#reference)

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

## Your team org

When you first log in to the [AC2 console](https://platform.appliedcompute.com), your team's org already has:

- **Your teammates**, all as org admins. Applied Compute manages membership; ask the organizers to add or remove
  someone.
- **A GPU quota of 10 GPUs** on the event's cluster (see [Managing limited compute](#managing-limited-compute)).
- **A project called `vals-hackathon`.** It exists but has no code version yet; step 1 links this repo to it.
- **A model deployment `vals-hackathon/ac-vals-model`**, an external model Applied Compute provides to every team.
  This README uses it for the opponent seats and the smoke test, so you don't need your own model provider key.
- **A Vals Arena page** in the console sidebar, where you submit your entry (step 12).

### Projects

A **project** is a named workspace inside your org. Everything you make lives in one: datasets, training runs and
their checkpoints, evals, inference deployments, the project's code versions, and the BYOH **connector** that links
your harness to AC2. There is **one connector per project**, so one harness serves one project at a time.

The seeded `vals-hackathon` project is enough for most teams: put all your datasets, runs and deployments in it.
You might make a second project to run a separately configured harness at the same time (it needs its own
connector), or to keep an experiment's datasets and runs apart. To do that, copy this repo, set `[project].name` in
its `pyproject.toml` to the new name, run `ac2 project init` there, and pass `--project <name>` to every `uv run`
command. The GPU quota is per org, so a second project doesn't add GPUs.

## How it works

### The moving parts

If you're new to AC2, these are the pieces you'll touch:

| Part | What it is |
| --- | --- |
| **Org** | Your team's AC2 organization. It holds your members, your GPU quota, and everything below. |
| **Project** | A named workspace inside the org (here: `vals-hackathon`). Datasets, runs, deployments and checkpoints all live in a project. See [Projects](#projects). |
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
 │ harness (+ shard router)     │              │ project "vals-hackathon"                     │
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

## Managing limited compute

Each team has a **10-GPU quota**. A job asks for its GPUs when it starts; if they'd take your org over the quota, it
waits in the queue until enough GPUs are free. Approximate GPU use for `Qwen/Qwen3.6-35B-A3B` (from AC2's
[Default GPUs](https://docs.appliedcompute.com/platform/training/configuration.md) table: 4 GPUs to train, 2 per
inference or eval replica):

| Job | GPUs (approx.) |
| --- | --- |
| RL run, as `rl/train.py` launches it (1 rollout replica + 1 in-run eval replica) | 8 (6 with `--no-eval`) |
| Offline OPSD run (`opsd/train.py`) | 4 |
| An inference deployment (the base model, a checkpoint, your arena entry) | 2 per replica |
| An eval through your harness | 0 extra (it uses the deployment it plays) |

So you can't run everything at once: an RL run plus one deployment already uses your 10 GPUs. AC2's own default
replica counts for this model would ask for 20 GPUs per RL run, which is why `rl/train.py` sets 1 inference and 1 eval
replica.

- **Delete deployments when you're done with them:** `uv run python -m vals_tools.deploy --name <name> --delete`.
  A deployment holds its GPUs until it's deleted, even when nothing calls it.
- **Stop a run's in-run evals with it:** `ac2 train stop <train id> --stop-evals`. Without `--stop-evals`, attached
  evals keep running.
- **Check before you launch:** `--dry-run` on `rl/train.py` and `opsd/train.py` prints what the run will use,
  including a GPU estimate.
- **Keep few checkpoints, and delete the ones you don't need.** Checkpoint storage is limited: pass
  `--keep-checkpoints <n>` to the training scripts, list them with `ac2 checkpoints list vals-hackathon <train id>`, and
  delete one with `ac2 checkpoints delete <train id> <step> --format all --cluster all`.
- **Smoke first.** Try a short run (`--steps 5`) before a long one, so a mistake costs minutes, not hours.

## Setup

### Prerequisites

- **Python 3.12+**, **[uv](https://docs.astral.sh/uv/)** and **git**.
- **An AC2 account in your team's org.** Your team lead is added by the organizers; ask them if you can't log in.
- **AC2 credentials:** log in through the browser with `ac2 login`, or create an API key in the AC2 console under
  **Settings → API keys**.
- **A machine that stays on** to run the harness during training (a VM, a cloud box, a container service). A
  laptop is fine for the smoke test only.
- Optionally, **a model provider key** (for example OpenAI) if you want opponents other than the provided model.

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

Expect `OK` lines for the SDK, the CLI, `project vals-hackathon is prj_...` and `model vals-hackathon/ac-vals-model`,
then `FAIL project vals-hackathon has no code version yet: run ac2 project init`. That's correct until step 1. If you
see `can't reach AC2`, see [Troubleshooting](#troubleshooting).

**Every command below runs from the repo root** (`pokerbots-starter/`) unless it starts with `cd harness`.

## Walkthrough

### 1. Link this repo to your project

```bash
ac2 project init
```

Your org already has project `vals-hackathon`, and `[project].name` in `pyproject.toml` is `vals-hackathon`, so this
links the folder to that project and pushes the empty stub as its first version. Expect:

```text
Project vals-hackathon already in AC2.
the current directory is tracked as vals-hackathon.
Pushed version <12-character hash> from the current directory
```

**Verify:** `uv run python -m vals_tools.doctor` now prints `OK project vals-hackathon has a pushed version`, and no
`FAIL` lines. **Write down the `prj_...` id** it prints: steps 4 and 5 need it.

### 2. Check the opponent model

The five other seats in each hand need a model. Your org starts with one: `vals-hackathon/ac-vals-model`. The doctor
checked it in step 1 (`OK model vals-hackathon/ac-vals-model is there`), and the datasets in step 3 use it by default.

To play against a different model instead, register it as an external endpoint and pass it to step 3. For an OpenAI
model:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <your OpenAI key>
ac2 inference external --name opponent --provider openai --model <model> \
  --base-url https://api.openai.com/v1 --protocol openai_responses
```

Then add `--opponent-model vals-hackathon/opponent --opponent-api responses --opponent-effort medium` to both commands
in step 3.

**Verify:** `ac2 inference list | grep vals-hackathon` lists `ac-vals-model` (and `opponent` if you made one).

### 3. Upload the datasets

```bash
uv run python rl/make_dataset.py --dry-run
uv run python rl/make_dataset.py
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
position) and the opponent model in the other five. The scratchpad starts empty and the cleanup turn still
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

Every line should be `OK`, ending with `OK a connector is connected to vals-hackathon`.

### 6. Smoke test

Play two short games with the provided model in every seat, then four hands with it as the trained seat:

```bash
uv run python -m vals_tools.eval --dataset pokerbots-table-smoke --deployment vals-hackathon/ac-vals-model
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/ac-vals-model
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
`{"deployment": "vals-hackathon/base", "deployment_id": "infer_..."}`. This can take several minutes. Then:

```bash
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/base
uv run python -m vals_tools.eval --dataset pokerbots-eval --deployment vals-hackathon/base
```

Save the second eval id: it's the **base eval** for every comparison. Keep the `base` deployment if you'll run
OPSD (step 10); otherwise delete it once the eval finishes:
`uv run python -m vals_tools.deploy --name base --delete`.

**Verify:** `uv run python -m vals_tools.score --eval <base eval id>` reports about 3,000 hands.

### 8. Train with RL (GRPO)

```bash
uv run python rl/train.py --dry-run     # what this launch sets, then the exact configuration
uv run python rl/train.py               # prints {"train_id": "train_...", ...}
```

The dry run starts with `This launch sets (everything else is AC2's default):` and lists every setting the script
sends. By default that's the GRPO recipe: `Qwen/Qwen3.6-35B-A3B` on `pokerbots-train`, Dr. GRPO with a small KL
penalty (`kl_coef = 0.01`), 16 hands × 16 attempts per step, 60 steps, a checkpoint every 10 steps, thinking on, an
in-run eval on `pokerbots-select` at step 0 and every 10 steps, a token budget of 24,576 per hand
(`max_response_len`, see below), and 1 rollout and 1 eval replica, so the run fits your GPU quota (the dry run
prints `GPUs while running: about 8`). **Nothing else is overridden:** learning rate and sampling concurrency are
AC2's defaults. To change any setting, pass a flag (`uv run python rl/train.py --help` lists them all), for example
`uv run python rl/train.py --lr 1e-6 --dry-run`, and check the dry run before launching.

**The token budget is per hand, not per decision.** In training, `max_response_len` counts everything after the
task's first prompt: every decision's reasoning (training keeps it inline), every observation, and the cleanup turn.
Once a hand spends it, the trainer returns empty replies with `finish_reason=length`, and the harness logs them as
"reply hit the cap" and checks or folds; the hand then scores 0 like any format error, so training stays correct, but
the model learns nothing useful from it. AC2's default of 8,192 runs out after a couple of long decisions, which is
why the recipe sets 24,576. Size it for your longest hands. The event itself caps each decision at 5,000 tokens and
the cleanup turn at 10,000, separately. Your harness plays every hand, so keep steps 4–5 running. The run names no
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
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/step20
uv run python -m vals_tools.eval --dataset pokerbots-eval --deployment vals-hackathon/step20
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
uv run python -m vals_tools.eval --dataset pokerbots-train --deployment vals-hackathon/base   # base plays the train deals
uv run python opsd/build_dataset.py --eval <that eval id> --dataset pokerbots-opsd-train --dry-run
uv run python opsd/build_dataset.py --eval <that eval id> --dataset pokerbots-opsd-train
uv run python opsd/train.py --dataset pokerbots-opsd-train --dry-run
uv run python opsd/train.py --dataset pokerbots-opsd-train
```

As in step 8, the dry run lists what the launch sets: 20 steps of 64 decisions, a checkpoint every 10, thinking on,
plus what offline mode requires (one sample per task, no inference replicas, no in-run eval). Learning rate and
lengths are AC2's defaults unless you pass `--lr`, `--max-response-len` or `--max-total-len`.

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

## While your training run is going

A run takes hours. Useful things to do meanwhile, without getting in its way (mind the [GPU quota](#managing-limited-compute)):

- **Evaluate as you go.** Run evals on the base model and on earlier checkpoints (step 9), and judge them on held-out
  deals (`pokerbots-eval`), not on the deals you picked them with.
- **Build better datasets.** The ones here are single fresh hands at 100 BB. What to train on is up to you: other
  seats and positions, synthetic situations, situations from the middle of a game. `vals_tools/tasks.py` shows the
  task format the harness reads.
- **Work on your reward.** Change the sample grader's extension points (`hand_reward`, `GRADER_PARAMS`, `artifacts` in
  [`grading.py`](harness/src/poker_harness/grading.py)) and check them with `cd harness && uv run pytest`. Restart the
  harness only between runs: the running run grades with whatever harness is up.
- **Test harness changes locally** with scripted players, no model needed (`ScriptedPlayer` in
  `poker_harness.engine`, as in `harness/tests/test_harness.py`), against a second harness on another port.
- **Read what your model does:** traces in the AC2 platform (or `ac2 traces download <run or eval id>`) and the game
  logs in `harness/runs/`.
- **Watch the arena:** your games and their replays on [pokerbots.ac](https://www.pokerbots.ac).
- **Prepare OPSD data:** write a hint and build its dataset (step 10) so it's ready when GPUs free up.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `uv sync` can't find `ac2` | Export `UV_INDEX_AC2_USERNAME=__token__` and `UV_INDEX_AC2_PASSWORD=<API key>` before `uv sync`. |
| `doctor`: `can't reach AC2`, or 401 / 403 errors | Not logged in or the wrong org: run `ac2 login`, or export `AC2_API_KEY` (and `AC2_ORG_ID`). Make sure `AC2_SERVICE_CLIENT_*` isn't set in this terminal. |
| `ProjectNotFoundError` | The project name in `pyproject.toml` doesn't exist in your org. It should be `vals-hackathon`; if you changed it, run `ac2 project init` (step 1). Every tool also resolves the project of the folder it runs in. |
| `422 ... base_url and protocol are required` when creating your own opponent | Add `--base-url https://api.openai.com/v1 --protocol openai_responses` (step 2). |
| `run_sharded.sh: uvicorn: not found` | Start it with `uv run ./run_sharded.sh` from `harness/` after `uv sync` there. |
| A run or deployment stays queued | Your org is at its GPU quota. Check what's running (`ac2 train list`, `ac2 inference list`), delete deployments you don't need, stop runs with `--stop-evals`; see [Managing limited compute](#managing-limited-compute). |
| `doctor`: `no vals-hackathon/ac-vals-model` | Your org should start with it; ask the organizers. Meanwhile, use your own opponent model (step 2). |
| `doctor`: `no code version yet` | Run `ac2 project init` from the repo root (step 1). |
| `doctor`: `no connector is connected` | Start step 5 and check its terminal for `Connected`. Only one connector can serve a project: stop any old one (Ctrl-C), and it may take about a minute for AC2 to let a new one in. |
| Eval or training rollouts fail right away | The harness or connector isn't running, or the token in step 5 doesn't match step 4. Run the doctor with `--harness`. |
| Rollouts fail with an opponent error | `AC2_API_KEY` wasn't exported in the harness terminal, or the opponent model in your datasets doesn't exist (step 2). |
| A deployment says ready but calls fail (503, timeouts) | It's still loading. `vals_tools.deploy` waits for a real answer before returning; if you deployed another way, wait and retry, or check `ac2 inference get <infer_...>`. Don't send an eval to a deployment that hasn't answered. |
| Rollouts fail at collect, mentioning 1 MiB or the output size | AC2's connector carries at most 1 MiB per rollout. The harness trims long multi-hand game logs to fit, keeping the full log in `harness/runs/`. If you built your own long tasks or added fields to the log, keep them under that size. |
| Training logs show many "reply hit the cap" decisions late in hands | The hand's token budget ran out (see step 8): raise `--max-response-len` (and `--max-total-len`). |
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

**The sample grader** scores the trained seat's result in big blinds: `6 + clip(EV / 1 BB, -5, +5)`, so every real
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
