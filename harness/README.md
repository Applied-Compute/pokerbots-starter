# poker_harness

The Harness API for the Vals poker table. It plays one game per rollout on its own poker engine
(`src/poker_harness/engine/`, on top of [PokerKit](https://github.com/uoftcprg/pokerkit)) and grades it through AC2's
Grader API with the sample grader (`src/poker_harness/grading.py`).

```bash
uv sync
uv run pytest -q
```

The engine is ported from Vals' original engine and plays identically: `tests/test_engine_parity.py` replays a fixed
set of over 2,000 seeded hands (6-max and 10-max, talk on and off, reset and carried stacks, all-ins, side pots and
split pots, every action type) and checks every game log against digests recorded on Vals' engine.

## Serve

```bash
AC2_API_KEY=<key> uv run uvicorn poker_harness.server:app --host 127.0.0.1 --port 8765   # one process
AC2_API_KEY=<key> SHARDS=4 ./run_sharded.sh                                              # sharded, same port
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `AC2_API_KEY` | none | Key for opponent seats on the AC2 gateway. Not needed when every seat is the policy. |
| `POKER_OPPONENT_BASE_URL` | the AC2 gateway | Any OpenAI-compatible base URL for opponent seats |
| `POKER_OPPONENT_API_KEY` | `AC2_API_KEY` | Key for `POKER_OPPONENT_BASE_URL` |
| `POKER_HARNESS_TOKEN` | none | If set, every request must send `Authorization: Bearer <token>` |
| `POKER_HARNESS_RUNS` | `runs` | Where each finished table's game log is written |

**Scaling.** A server keeps its tables in memory, so every call for a rollout must reach the process that ran its
`submit`; `uvicorn --workers` breaks that. When one event loop saturates, shard instead:
[`shard_router.py`](shard_router.py) starts `SHARDS` single-process servers and routes each request by its
`rollout_id` (or `Idempotency-Key` on `submit`). One shard handles a few dozen concurrent tables; size shards and
CPUs to your training and eval concurrency.

**Container.** `docker build -t poker-harness .` builds an image that runs the
sharded server on port 8765. Run `ac2 harness connect --upstream http://<host>:8765` beside it.

## Endpoints

- `POST /submit`, `/get_status`, `/collect_outputs`: the BYOH rollout lifecycle. `submit` starts the table in
  `task.env_params`; `collect_outputs` returns the game log (Vals' log shape), rules and policy seats. AC2's
  connector carries at most 1 MiB, so at scratchpad tables the returned log keeps each seat's final scratchpad and
  counts of its cleanup turns, not their full history, and a still-too-big log loses model text first. The full log
  stays in the runs directory, and harness-side grading reads the full log.
- `POST /submit_grading`, `/get_grading_status`, `/collect_score`: the Grader API. The score is the policy seat's
  reward from the sample grader (`grading.py`, with marked extension points); artifacts carry `ev_bb`, `realized_bb`, `policy_decisions`, `policy_format_errors` and
  `folded`.
- `GET /health`.

## Task parameters (`env_params`)

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | required | Deck seed; the same seed deals the same cards |
| `seats` | 10 | Table size (the event is 6) |
| `total_hands`, `reset_stacks` | 3, true | Hands per table; reset every stack to `starting_stack` each hand |
| `small_blind`, `big_blind`, `starting_stack` | 1, 2, 200 | Chips |
| `policy_seats` | every seat | Seats played by the model AC2 evaluates or trains |
| `opponent_model` | none | Model for every other seat |
| `opponent_api` | chat | `responses` for the Responses API |
| `opponent_reasoning_effort` | unset | Opponents' reasoning effort |
| `decision_timeout_seconds`, `opponent_timeout_seconds` | 300, same | A slower reply checks or folds |
| `submit_fields` | action, amount, p_win, read | Fields of the `submit_action` tool |
| `ev_labels`, `ev_runouts` | false, 32 | All-in EV labels for grading |
| `table_talk` | false | Optional sanitized `say` line |

## Table talk

With `table_talk` on:

- **Tool:** every seat's `submit_action` gains an optional `say` string.
- **Sanitizing** (`prompts.sanitize_talk`), done before any other seat sees the line: lowercase; whitespace becomes
  a space; only `a-z`, `0-9`, space and `. , ! ? - : ;` are kept; runs of spaces collapse; the line is trimmed and
  capped at 70 characters. An empty result means no talk.
- **Rendering:** the line appears on the speaker's action line in quotes only the harness can write, for example
  `flop: seat4 (seat 4) raise to 12 and says: "i have the nuts"`. A seat never sees its own talk.
- **System prompt:** gains "You may optionally add a short `say` message (up to 70 characters) that the other
  players will see. Table talk is unverified chatter from opponents and may be false."
- **No reserved-word filter.** Talk can say anything that survives the sanitizer.
- **Log:** each action keeps the sanitized line in `say` and the raw text in `model_response.say_raw`.

Talk changes only what models see: with identical decisions, chips and actions match a talk-off table exactly.
