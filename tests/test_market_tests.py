import numpy as np
import pandas as pd

import market_tests as mt


def _data(n=8):
    return pd.DataFrame({"GAME_ID": [f"g{i}" for i in range(n)], "PLAYER_ID": 1,
                         "GAME_DATE": pd.date_range("2026-01-01", periods=n), "IS_HOME": 1, "tz_shift": 2.0,
                         "games_in_season": 30, "PTS": [10, 12, 8, 10, 12, 8, 30, 10.0], "REB": 5.0, "AST": 5.0})


def _pred(game):
    return pd.DataFrame({"stat": ["PTS"], "GAME_ID": [game], "PLAYER_ID": [1], "p_mid": [0.5],
                         "tip": ["2026-01-08T01:00:00+00:00"]})


def test_last_dev_uses_only_previous_games():
    # g6 zdobywa 30 pkt: widać to dopiero w g7; w g6 ostatni mecz to 8 pkt, poniżej średniej
    before, after = (mt.add_features(_pred(g), _data())["last_dev"].iloc[0] for g in ("g6", "g7"))
    assert before < 0 and after > 1.5


def test_east_home_counts_only_home_eastward_trips():
    d = _data().assign(IS_HOME=0)
    assert mt.add_features(_pred("g7"), d)["east_home"].iloc[0] == 0
    assert mt.add_features(_pred("g7"), _data())["east_home"].iloc[0] == 2


def test_early_tip_is_in_eastern_time():
    p = _pred("g7").assign(tip="2026-01-07T20:00:00+00:00")    # 15:00 ET
    assert mt.add_features(p, _data())["early_tip"].iloc[0] == 1
    assert mt.add_features(_pred("g7"), _data())["early_tip"].iloc[0] == 0


def test_game_ci_widens_with_level():
    rng = np.random.default_rng(0)
    v, g = rng.normal(size=400), np.repeat(np.arange(100), 4)
    lo, hi = mt.game_ci(v, g, rng, 0.8), mt.game_ci(v, g, rng, 0.996)
    assert hi[0] < lo[0] and hi[1] > lo[1]
