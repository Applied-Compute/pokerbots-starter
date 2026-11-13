import asyncio

from poker_harness.engine import ScriptedPlayer

from poker_harness import TableRules, play_session
from poker_harness.spots import chen_score, hole_cards


def test_hole_cards_match_what_the_table_deals() -> None:
    for seed in (11, 12345, 2**40 + 7):
        players = [ScriptedPlayer(f"seat{i}", "calling_station", seed) for i in range(10)]
        log = asyncio.run(play_session("g", seed, players, TableRules(total_hands=1, reset_stacks=True)))
        dealt = log["hands"][0]["starting_state"]
        assert all(hole_cards(seed, i) == dealt[f"seat{i}"]["hole_cards"] for i in range(10))


def test_chen_score_orders_classic_hands() -> None:
    assert chen_score(["As", "Ad"]) == 20 and chen_score(["As", "Ks"]) == 12
    assert chen_score(["2c", "2d"]) == 5 and chen_score(["7c", "2h"]) == -1
    assert chen_score(["Ts", "9s"]) > chen_score(["Tc", "9d"])
