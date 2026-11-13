#!/bin/sh
# Single-process harness shards behind the rollout-id router (shard_router.py), listening on $PORT (default 8765).
# One shard handles a few dozen concurrent tables; add shards (and CPUs) for more.
export SHARDS="${SHARDS:-4}" SHARD_BASE_PORT="${SHARD_BASE_PORT:-9000}"
export SHARD_COMMAND="${SHARD_COMMAND:-uvicorn poker_harness.server:app --host 127.0.0.1 --port {port}}"
exec uvicorn shard_router:app --app-dir "$(dirname "$0")" --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
