"""Run N copies of a single-process harness server and route each request to one by its rollout id.

A harness that keeps rollouts in memory must serve every call for a rollout from the process that ran `submit`, so it
can't run several uvicorn workers. When one event loop saturates, shard instead: this router starts `SHARD_COMMAND`
(with `{port}` filled in) on `SHARD_BASE_PORT + i` for i in range(SHARDS) and proxies each request to shard
hash(rollout_id) % SHARDS. The key is the JSON body's `rollout_id`, else the `Idempotency-Key` header (the connector
names a new rollout only in that header on `submit`), so `submit`, `get_status`, `collect_outputs` and the Grader API
calls for one rollout all reach the same shard. `/health` sums the shards' running counts. Dead shards are restarted.

    SHARDS=4 SHARD_BASE_PORT=9000 SHARD_COMMAND='uvicorn poker_harness.server:app --host 127.0.0.1 --port {port}' \
        uvicorn shard_router:app --host 127.0.0.1 --port 8765

`run_sharded.sh` does the same with defaults.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time

import httpx

SHARDS = int(os.environ.get("SHARDS", "4"))
BASE_PORT = int(os.environ.get("SHARD_BASE_PORT", "9000"))
COMMAND = os.environ.get("SHARD_COMMAND", "")
HOP_HEADERS = {b"host", b"content-length", b"connection", b"transfer-encoding"}

_children: dict[int, subprocess.Popen] = {}
_client: httpx.AsyncClient | None = None


def _spawn(shard: int) -> None:
    argv = shlex.split(COMMAND.format(port=BASE_PORT + shard))
    print(f"[router] starting shard {shard}: {' '.join(argv)}", flush=True)
    _children[shard] = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)


def _watch() -> None:
    while True:
        time.sleep(2)
        for shard, child in list(_children.items()):
            if child.poll() is not None:
                print(f"[router] shard {shard} exited with {child.returncode}; restarting", flush=True)
                _spawn(shard)


def _shard(key: str | None) -> int:
    if not key:
        return 0
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big") % SHARDS


async def _read_body(receive) -> bytes:
    body = bytearray()
    while True:
        message = await receive()
        body.extend(message.get("body", b""))
        if not message.get("more_body"):
            return bytes(body)


async def _respond(send, status: int, body: bytes, content_type: bytes = b"application/json") -> None:
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", content_type)]})
    await send({"type": "http.response.body", "body": body})


async def _health(send) -> None:
    assert _client is not None
    running, statuses = 0, []
    for shard in range(SHARDS):
        try:
            reply = (await _client.get(f"http://127.0.0.1:{BASE_PORT + shard}/health", timeout=30)).json()
            running += sum((reply.get("rollouts") or {}).values())
            statuses.append(reply.get("status"))
        except (httpx.HTTPError, ValueError):
            statuses.append("down")
    ok = all(status == "ok" for status in statuses)
    payload = {"status": "ok" if ok else "degraded", "shards": statuses, "rollouts": {"running": running}}
    await _respond(send, 200 if ok else 503, json.dumps(payload).encode())


async def app(scope, receive, send) -> None:
    global _client
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                for shard in range(SHARDS):
                    _spawn(shard)
                threading.Thread(target=_watch, daemon=True).start()
                limits = httpx.Limits(max_connections=2000, max_keepalive_connections=200)
                _client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10), limits=limits)
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                for child in _children.values():
                    child.terminate()
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["type"] != "http":
        return
    if scope["path"] == "/health":
        await _health(send)
        return
    body = await _read_body(receive)
    headers = [(k.decode(), v.decode()) for k, v in scope["headers"] if k.lower() not in HOP_HEADERS]
    key = None
    if body:
        try:
            parsed = json.loads(body)
            key = parsed.get("rollout_id") if isinstance(parsed, dict) else None
        except ValueError:
            key = None
    if not key:
        key = next((v for k, v in headers if k.lower() == "idempotency-key"), None)
    url = f"http://127.0.0.1:{BASE_PORT + _shard(key)}{scope['path']}"
    if scope.get("query_string"):
        url += "?" + scope["query_string"].decode()
    assert _client is not None
    try:
        reply = await _client.request(scope["method"], url, content=body, headers=headers)
    except httpx.HTTPError as exc:
        await _respond(send, 503, json.dumps({"detail": f"shard unavailable: {type(exc).__name__}"}).encode())
        return
    out = [(k.encode(), v.encode()) for k, v in reply.headers.items() if k.lower().encode() not in HOP_HEADERS]
    await send({"type": "http.response.start", "status": reply.status_code, "headers": out})
    await send({"type": "http.response.body", "body": reply.content})
