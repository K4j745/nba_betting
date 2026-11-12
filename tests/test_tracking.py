import numpy as np
import pandas as pd

import tracking as tk


def _games(n=12, reb=None, chances=None):
    reb = reb if reb is not None else [5.0] * n
    chances = chances if chances is not None else [10.0] * n
    return pd.DataFrame({"PLAYER_ID": 1, "GAME_ID": [f"{i:010d}" for i in range(n)],
                         "GAME_DATE": pd.date_range("2026-01-01", periods=n), "MIN": 30.0,
                         "REB": reb, "AST": 4.0, "REB_CHANCES": chances, "POTENTIAL_AST": 8.0})


CONV = {"REB": 0.5, "AST": 0.5}


def test_features_use_only_previous_games():
    # mecz 10: 20 szans i 15 zbiórek — widać to dopiero w meczu 11
    g = _games(reb=[5.0] * 10 + [15.0, 5.0], chances=[10.0] * 10 + [20.0, 10.0])
    f = tk.rolling_features(g, CONV)
    assert f.loc[10, "REB_10"] == 5.0 and f.loc[10, "xREB_10"] == 5.0
    assert f.loc[11, "REB_10"] == 6.0 and f.loc[11, "xREB_10"] > 5.0
    assert np.isnan(f.loc[0, "xREB_10"])


def test_gap_positive_when_results_lag_chances():
    # szans tyle samo, ale ostatnio mniej zbiórek niż wynika ze skuteczności → gap > 0
    g = _games(n=40, reb=[5.0] * 30 + [3.0] * 10)
    f = tk.rolling_features(g, CONV)
    assert f["gap_REB"].iloc[-1] > 0
    assert abs(f["gap_REB"].iloc[20]) < 1e-9


def test_conversion_shrinks_to_league_mean():
    # 5 meczów 10/10: skuteczność 1,0 ściągnięta w stronę średniej 0,5
    g = _games(n=6, reb=[10.0] * 6)
    f = tk.rolling_features(g, CONV)
    assert abs(f.loc[5, "xREB_10"] - 8.125) < 1e-9      # (50 + 30·0,5) / (50 + 30) · 10


def test_trend_detects_volume_jump():
    g = _games(n=25, chances=[10.0] * 20 + [20.0] * 5)
    f = tk.rolling_features(g, CONV)
    assert f["trend_REB"].iloc[20] == 0 and f["trend_REB"].iloc[24] > 0.3


def test_attach_fills_missing_and_sets_line():
    f = tk.rolling_features(_games(), CONV)
    pred = pd.DataFrame({"stat": ["REB", "REB", "AST"], "GAME_ID": ["0000000011", "0000000099", "0000000011"],
                         "PLAYER_ID": 1, "line": [4.5, 4.5, 3.5], "p_mid": 0.5})
    m = tk.attach(pred, f)
    assert m.loc[0, "line_REB"] == 0.5 and m.loc[0, "has_tracking"] == 1
    assert m.loc[1, "gap_REB"] == 0 and m.loc[1, "has_tracking"] == 0
    assert np.isnan(m.loc[2, "gap_REB"]) and m.loc[2, "line_AST"] == 0.5


def test_merge_measures_keeps_first_duplicate():
    a = pd.DataFrame({"PLAYER_ID": [1, 2], "MIN": [30, 20], "REB_CHANCES": [10, 5]})
    b = pd.DataFrame({"PLAYER_ID": [1, 3], "MIN": [99, 10], "POTENTIAL_AST": [7, 1]})
    m = tk.merge_measures({"Rebounding": a, "Passing": b}).set_index("PLAYER_ID")
    assert m.loc[1, "MIN"] == 30 and m.loc[3, "MIN"] == 10 and m.loc[1, "POTENTIAL_AST"] == 7


def test_league_conversion_uses_only_earlier_games():
    g = _games(n=4, reb=[5.0, 5.0, 10.0, 10.0])
    assert tk.league_conversion(g, pd.Timestamp("2026-01-03"))["REB"] == 0.5


def test_missing_rebounding_row_means_zero_chances():
    logs = pd.DataFrame({"PLAYER_ID": [1, 2], "GAME_ID": ["1", "2"], "GAME_DATE": pd.to_datetime(["2026-01-01"] * 2),
                         "MIN": [3.0, 3.0], "REB": 0.0, "AST": 0.0})
    tr = pd.DataFrame({"PLAYER_ID": [1], "GAME_DATE": pd.to_datetime(["2026-01-01"]), "POTENTIAL_AST": [1.0],
                       "REB_CHANCES": [np.nan]})
    g = tk.player_games(logs, tr).set_index("PLAYER_ID")
    assert g.loc[1, "REB_CHANCES"] == 0 and np.isnan(g.loc[2, "REB_CHANCES"])
