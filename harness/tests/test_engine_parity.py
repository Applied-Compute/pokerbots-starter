"""`poker_harness.engine` plays exactly like Vals' original engine: byte-identical game logs on a fixed set of games,
including full games under the event's rules (carried stacks, strict actions, the scratchpad).

The recorded digests (golden/engine_parity.json) were made on Vals' engine, so this runs anywhere. With
VALS_ENGINE_SRC set, the second test also plays every game on Vals' engine live and compares the full logs.
"""

import json
import os
import subprocess
import sys

import pytest
from engine_parity import GOLDEN, coverage, digest, games, play, vals_engine


RECORDED = json.loads(GOLDEN.read_text())
SPECS = games()


def test_the_recorded_games_cover_every_case() -> None:
    covered = RECORDED["coverage"]
    assert RECORDED["engine"] == "vals poker-agent 0.1.0" and len(RECORDED["digests"]) == len(SPECS)
    assert covered["hands"] >= 2000
    assert {s["seats"] for s in SPECS} == {6, 10} and {s["talk"] for s in SPECS} == {False, True}
    assert {s["reset"] for s in SPECS} == {False, True}
    for key in ("action_fold", "action_check", "action_call", "action_bet", "action_raise"):
        assert covered[key] > 100, key
    for key in ("all_in_hands", "side_pot_hands", "split_pot_hands", "fallbacks", "talk_lines", "busted_seats"):
        assert covered[key] > 10, key
    assert sum(1 for s in SPECS if s.get("event")) >= 5  # 60-hand event-rule games: busting, strict, scratchpad
    for key in ("cleanup_updated", "cleanup_unchanged", "cleanup_attempt_too_long", "cleanup_attempt_malformed"):
        assert covered[key] > 10, key


def test_the_engine_matches_vals_engine_on_every_recorded_game() -> None:
    texts = [play(spec) for spec in SPECS]
    mismatched = [spec["seed"] for spec, text, want in zip(SPECS, texts, RECORDED["digests"], strict=True)
                  if digest(text) != want]
    assert not mismatched, f"games that differ from Vals' engine: {mismatched}"
    assert coverage([json.loads(t) for t in texts]) == RECORDED["coverage"]


@pytest.mark.skipif(vals_engine() is None, reason="set VALS_ENGINE_SRC to compare against Vals' engine live")
def test_the_engine_matches_vals_engine_live() -> None:
    original = vals_engine()
    assert original is not None
    runner, scripted = original
    for spec in SPECS:
        assert play(spec) == play(spec, runner, scripted), spec["seed"]


# Recorded on Vals' engine with PYTHONHASHSEED=0. PokerKit samples runouts from card sets whose order follows Python's
# hash seed, so true equity (in both engines) repeats only for a fixed hash seed; the check runs in a child process.
EQUITY_CASES = [
    (["As", "Ad"], [["Kc", "Kd"]], []),
    (["7c", "2h"], [["As", "Ad"], ["Kc", "Qc"]], ["Ks", "7d", "2c"]),
    (["Jh", "Th"], [["9c", "9d"]], ["8h", "7h", "2c", "3d"]),
    (["Qs", "Qh"], [["Ac", "Kc"], ["Jd", "Jc"], ["5s", "5h"]], ["2d"]),
]
VALS_EQUITIES = [0.79, 0.6367, 0.3733, 0.43]


def test_true_equity_matches_vals_engine() -> None:
    code = (
        "import json, sys; from poker_harness.engine import true_equity; "
        "print(json.dumps([true_equity(*c) for c in json.loads(sys.argv[1])]))"
    )
    env = os.environ | {"PYTHONHASHSEED": "0"}
    out = subprocess.run([sys.executable, "-c", code, json.dumps(EQUITY_CASES)], env=env, capture_output=True, text=True,
                         check=True)
    assert json.loads(out.stdout) == VALS_EQUITIES
