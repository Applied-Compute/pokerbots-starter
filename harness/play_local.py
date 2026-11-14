"""Play one event game on your machine against a deployed model, to check it before you enter it in the arena.

    uv run python play_local.py --model vals-hackathon/entry                    # your deployment vs the house model
    uv run python play_local.py --model vals-hackathon/entry --opponents bots   # vs scripted bots (no opponent calls)
    uv run python play_local.py --model vals-hackathon/entry --hands 6          # a shorter game

The game runs under the event's rules (the "event" preset: 6 seats, talk, carried stacks, the scratchpad and its
cleanup turn, 5,000 tokens per decision and 10,000 per cleanup turn, thinking on), with your model in seat 0. Every
seat calls its model directly through the AC2 inference API with your `AC2_API_KEY`; nothing goes through a
connector, and no AC2 job runs. At the end it prints how many of your model's decisions failed (no usable
`submit_action` call, an illegal move, or a reply cut off at the cap) and writes the full game log.

The arena's entry probe only checks that your deployment answers a plain text request. A deployment that can't call
`submit_action` passes the probe and then fails every decision: its seats check or fold, and the arena pauses it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from collections import Counter
from pathlib import Path

import httpx

from poker_harness.completions import (
    GATEWAY_URL,
    cleanup_params,
    opponent_params,
    relay_completion,
    table_rules,
    with_preset,
)
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat

HOUSE_MODEL = "vals-hackathon/ac-vals-model"


async def play(args: argparse.Namespace) -> dict:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    seed = args.seed if args.seed is not None else random.getrandbits(48)
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    if args.hands:
        params["total_hands"] = args.hands
    rules = table_rules(params)
    if rules.table_talk:  # every seat's tool gains `say`, as on the harness server
        params["submit_fields"] = talk_fields(params["submit_fields"])
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        mine = params | {"model_label": args.model}
        players = [
            ModelSeat(
                "seat0",
                relay_completion(client, GATEWAY_URL, key, mine),
                rules,
                model=args.model,
                cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(mine)),
            )
        ]
        for i in range(1, rules.seats):
            if args.opponents == "bots":
                players.append(ScriptedPlayer(f"seat{i}", random.Random(seed + i).choice(["random", "calling_station"]), seed))
            else:
                theirs = opponent_params(params | {"opponent_model": args.opponent_model})
                players.append(
                    ModelSeat(
                        f"seat{i}",
                        relay_completion(client, GATEWAY_URL, key, theirs),
                        rules,
                        model=args.opponent_model,
                        cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(theirs)),
                    )
                )
        return await play_session(f"local-{seed}", seed, players, rules, on_hand=lambda log: print(
            f"hand {len(log['hands'])} done", flush=True
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="your deployment, <project>/<name>")
    parser.add_argument("--opponents", choices=["house", "bots"], default="house")
    parser.add_argument("--opponent-model", default=HOUSE_MODEL, help="model for the other seats with --opponents house")
    parser.add_argument("--hands", type=int, help="hands in the game (default: the event's 30)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    log = asyncio.run(play(args))
    mine = [a for hand in log["hands"] for a in hand["actions"] if a["player"] == "seat0"]
    errors = Counter(a["model_response"].get("error") or "?" for a in mine if a["model_response"].get("format_error"))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{log['game_id']}.json"
    path.write_text(json.dumps(log, indent=1))
    big_blind = log["session_info"]["big_blind"]
    print(f"\n{len(log['hands'])} hands, {len(mine)} decisions by {args.model}, {sum(errors.values())} failed")
    for error, count in errors.most_common():
        print(f"  {count} x {error}")
    print(f"seat0 net: {(log['final_stacks']['seat0'] - 200) / big_blind:+.1f} BB; log: {path}")
    if mine and sum(errors.values()) == len(mine):
        raise SystemExit("every decision failed: don't submit this deployment until it calls submit_action")


if __name__ == "__main__":
    main()
