from poker_harness.labels import fold_regret_bb, label_folds


def action(player: str, kind: str, call: int = 0, pot: int = 3, board: list[str] | None = None) -> dict:
    return {"player": player, "action": {"type": kind}, "call_amount": call, "pot_before": pot, "board": board or []}


def test_folding_the_best_hand_costs_and_folding_a_beaten_hand_is_free() -> None:
    aces = fold_regret_bb(action("p", "fold", call=4, pot=10), ["As", "Ad"], [["7c", "2h"]], big_blind=2)
    assert aces > 3  # ~87% x 14 - 4 = ~8 chips, about 4 BB; sampled (hash-seed dependent), so leave room
    trash = fold_regret_bb(action("p", "fold", call=40, pot=50), ["7c", "2h"], [["As", "Ad"]], big_blind=2)
    assert trash == 0.0


def test_label_folds_only_marks_policy_folds_against_live_opponents() -> None:
    log = {
        "session_info": {"big_blind": 2},
        "hands": [
            {
                "starting_state": {
                    "p": {"hole_cards": ["Ks", "Kd"]},
                    "a": {"hole_cards": ["9c", "8c"]},
                    "b": {"hole_cards": ["Qh", "Jh"]},
                },
                "actions": [action("a", "fold", call=2), action("b", "raise"), action("p", "fold", call=6, pot=9)],
            }
        ],
    }
    label_folds(log, {"p"})
    acts = log["hands"][0]["actions"]
    assert "fold_regret_bb" not in acts[0]
    assert acts[2]["fold_live_opponents"] == 1 and acts[2]["fold_regret_bb"] > 0
