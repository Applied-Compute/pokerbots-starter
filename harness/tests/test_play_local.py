"""play_local.py: one local event game against a deployment, reporting failed decisions."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import play_local  # noqa: E402


def gateway(calls_tool: bool) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        if not calls_tool:
            return {"choices": [{"finish_reason": "stop", "message": {"content": "OK"}}], "usage": {}}
        last = str(body["messages"][-1].get("content", ""))
        name, arguments = ("update_scratchpad", {"contents": "n"}) if "is over" in last else ("submit_action", {"action": "call"})
        call = {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [call]}}], "usage": {}}

    return app


def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls_tool: bool, *extra: str) -> None:
    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.ASGITransport(app=gateway(calls_tool))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(play_local.httpx, "AsyncClient", Client)
    monkeypatch.setattr(play_local, "GATEWAY_URL", "http://gw/v1")
    monkeypatch.setenv("AC2_API_KEY", "k")
    argv = ["play_local.py", "--model", "p/entry", "--hands", "3", "--seed", "5", "--out", str(tmp_path), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    play_local.main()


def test_a_tool_calling_deployment_plays_a_clean_game(monkeypatch, tmp_path, capsys) -> None:
    run(monkeypatch, tmp_path, True)
    out = capsys.readouterr().out
    assert "3 hands" in out and ", 0 failed" in out and list(tmp_path.glob("local-5.json"))


def test_a_deployment_that_never_calls_the_tool_is_flagged(monkeypatch, tmp_path) -> None:
    with pytest.raises(SystemExit, match="every decision failed"):
        run(monkeypatch, tmp_path, False, "--opponents", "bots")
