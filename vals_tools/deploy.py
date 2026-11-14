"""Serve a model as an AC2 inference deployment and wait until it really answers.

    uv run python -m vals_tools.deploy --name base --model Qwen/Qwen3.6-35B-A3B     # the base model, for comparisons
    uv run python -m vals_tools.deploy --name step20 --checkpoint ckpt_xxx            # a training checkpoint
    uv run python -m vals_tools.deploy --name step20 --delete                          # tear it down when you're done

A training checkpoint is first converted to serving weights (`promote`), which can take several minutes. AC2 can
report a deployment ready before it actually serves, so after it says ready this sends one real chat completion with
the `submit_action` tool and retries with backoff until it answers. Only then does it print the deployment's name
(`<project>/<name>`, for `vals_tools.eval --deployment`) and id (`infer_...`, for the arena). Deployments hold GPUs
until you delete them.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx
from ac2.sdk import Client
from ac2.sdk.model import ModelConfig

STATE_POLL_S = 10  # ask AC2 for the deployment's state at most once every 10 s
PROBE_BACKOFF_S = (10, 120)  # first wait, cap; doubles after every failed probe
PROBE_GIVE_UP_S = 20 * 60
INFERENCE_URL = "https://inference.appliedcompute.com/v1"
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_action",
        "description": "Submit your action for this turn.",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["fold", "check", "call", "bet", "raise"]}},
            "required": ["action"],
        },
    },
}
PROBE_MESSAGES = [{"role": "user", "content": "You hold 7c 2d preflop facing a big raise. Call submit_action."}]


def promote(client: Client, checkpoint: str) -> None:
    """Convert a training checkpoint to serving weights, unless that already happened."""
    state = str(client.checkpoints.promotion_status(checkpoint).state)
    if state in ("missing", "none", "not_found", "failed"):
        state = str(client.checkpoints.promote(checkpoint).state)
    while state in ("pending", "queued", "converting", "running"):
        print(f"checkpoint {checkpoint}: {state}", flush=True)
        time.sleep(30)
        state = str(client.checkpoints.promotion_status(checkpoint).state)
    if "fail" in state or "error" in state:
        raise SystemExit(f"converting {checkpoint} to serving weights ended {state}")


def wait_until_serving(client: Client, deployment_id: str, model: str) -> None:
    """AC2's state first (polled gently), then one real completion that must return a tool call."""
    while True:
        state = str(client.inference.get(deployment_id).state)
        print(f"deployment {deployment_id}: {state}", flush=True)
        if state in ("running", "ready", "healthy"):
            break
        if state in ("failed", "stopped", "deleted"):
            raise SystemExit(f"deployment {deployment_id} is {state}")
        time.sleep(STATE_POLL_S)
    wait, started = PROBE_BACKOFF_S[0], time.monotonic()
    headers = {"Authorization": f"Bearer {os.environ.get('AC2_API_KEY', '')}"}
    body = {"model": model, "messages": PROBE_MESSAGES, "tools": [PROBE_TOOL], "max_completion_tokens": 5000}
    while True:
        try:
            reply = httpx.post(f"{INFERENCE_URL}/chat/completions", json=body, headers=headers, timeout=300)
            calls = reply.json()["choices"][0]["message"].get("tool_calls") if reply.status_code == 200 else None
            if calls:
                print(f"probe: answered with {calls[0]['function']['name']}({calls[0]['function']['arguments']})")
                return
            print(f"probe: HTTP {reply.status_code}, no tool call yet", flush=True)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            print(f"probe: {type(exc).__name__}", flush=True)
        if time.monotonic() - started > PROBE_GIVE_UP_S:
            raise SystemExit(f"{model} never answered; check `ac2 inference get {deployment_id}`")
        time.sleep(wait)
        wait = min(wait * 2, PROBE_BACKOFF_S[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--name", required=True, help="deployment name in the project, e.g. base or step20")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="a catalog model, e.g. Qwen/Qwen3.6-35B-A3B")
    source.add_argument("--checkpoint", help="a training checkpoint id (ckpt_...), from `ac2 train checkpoints`")
    source.add_argument("--delete", action="store_true", help="delete the deployment with this name")
    parser.add_argument("--cluster-id")
    args = parser.parse_args()

    client = Client(project=args.project)
    model_name = f"{args.project}/{args.name}"
    if args.delete:
        matches = [d for d in client.inference.list() if getattr(d, "name", None) == args.name]
        for deployment in matches:
            client.inference.delete(deployment.id)
            print(f"deleted {deployment.id} ({model_name})")
        if not matches:
            print(f"no deployment named {args.name} in {args.project}")
        return
    if args.checkpoint:
        promote(client, args.checkpoint)
        config = ModelConfig(checkpoint=args.checkpoint)
    else:
        config = ModelConfig(model=args.model)
    print(f"{model_name} holds GPUs (2 for Qwen3.6-35B-A3B) until you delete it: --name {args.name} --delete")
    deployment_id = client.inference.deploy(name=args.name, model=config, cluster=args.cluster_id)
    print(f"deploying {model_name} as {deployment_id}", flush=True)
    wait_until_serving(client, deployment_id, model_name)
    print(json.dumps({"deployment": model_name, "deployment_id": deployment_id}))


if __name__ == "__main__":
    main()
