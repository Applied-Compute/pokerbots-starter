from poker_harness.ev import consistent, hand_ev


def act(
    player: str, street: str, kind: str, stack_after: int, board: list[str], call: int = 0, amount=None, change=0
) -> dict:
    move = {"type": kind} | ({"amount": amount} if amount is not None else {})
    return {
        "player": player,
        "street": street,
        "action": move,
        "stack_after": stack_after,
        "board": board,
        "call_amount": call,
        "chip_change": change,
    }


def hand(hole: dict[str, list[str]], actions: list[dict], ending: dict[str, int], board: list[str]) -> dict:
    return {
        "hand_number": 1,
        "starting_state": {n: {"hole_cards": h, "starting_stack": 200} for n, h in hole.items()},
        "actions": actions,
        "ending_stacks": ending,
        "final_board": board,
    }


HOLES = {"h": ["As", "Ad"], "v": ["Kc", "Kh"], "x": ["7c", "2d"]}


def test_no_showdown_keeps_the_realized_result() -> None:
    h = hand(
        HOLES,
        [
            act("x", "preflop", "fold", 200, []),
            act("h", "preflop", "raise", 194, [], amount=6, change=-6),
            act("v", "preflop", "fold", 198, [], call=4),
        ],
        {"h": 202, "v": 198, "x": 200},
        [],
    )
    ev = hand_ev(h, "h", big_blind=2)
    assert ev["method"] == "no-showdown" and ev["ev_bb"] == ev["realized_bb"] == 1.0


def test_preflop_all_in_is_scored_by_equity_not_the_runout() -> None:
    board = ["Kd", "7h", "2c", "9s", "3h"]  # kings win as dealt
    h = hand(
        HOLES,
        [
            act("x", "preflop", "fold", 200, []),
            act("v", "preflop", "raise", 0, [], amount=200, change=-200),
            act("h", "preflop", "call", 0, [], call=200, change=-200),
        ],
        {"h": 0, "v": 400, "x": 200},
        board,
    )
    ev = hand_ev(h, "h", big_blind=2, variants=8, runouts=32)
    assert ev["method"] == "allin" and ev["realized_bb"] == -100.0
    assert ev["ev_bb"] > 40  # aces against a strong shoving range win most of a 200 BB pot


def test_river_showdown_redraws_opponents_but_keeps_the_board() -> None:
    board = ["Ks", "8d", "4c", "2h", "9c"]
    acts = [
        act("x", "preflop", "fold", 200, []),
        act("v", "preflop", "call", 198, [], call=2, change=-2),
        act("h", "preflop", "check", 198, []),
        act("h", "flop", "check", 198, board[:3]),
        act("v", "flop", "bet", 194, board[:3], amount=4, change=-4),
        act("h", "flop", "call", 194, board[:3], call=4, change=-4),
        act("h", "turn", "check", 194, board[:4]),
        act("v", "turn", "check", 194, board[:4]),
        act("h", "river", "check", 194, board),
        act("v", "river", "check", 194, board),
    ]
    h = hand(HOLES, acts, {"h": 194, "v": 206, "x": 200}, board)
    ev = hand_ev(h, "h", big_blind=2, variants=16)
    assert ev["method"] == "showdown" and ev["realized_bb"] == -3.0
    assert -3.0 < ev["ev_bb"] <= 3.0 and 0 < ev["ev_share"] <= 1  # aces beat many hands that bet a pair or draw


def test_consistency_model_follows_actions() -> None:
    raise_big = [act("v", "preflop", "raise", 0, [], amount=200, change=-200)]
    assert consistent(["As", "Ad"], raise_big) and not consistent(["7c", "2d"], raise_big)
    flop_bet = [
        act("v", "preflop", "call", 198, [], call=2),
        act("v", "flop", "bet", 190, ["Ks", "8d", "4c"], amount=8),
    ]
    assert consistent(["Kc", "Qd"], flop_bet) and not consistent(["7c", "6d"], flop_bet)
