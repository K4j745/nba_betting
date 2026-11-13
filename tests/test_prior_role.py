import numpy as np
import pandas as pd

import prior_role as pr


def _data():
    rows = []
    # sezon 2024-25: zawodnik 1 w drużynie 10 (12 meczów, 8 zbiórek), potem 2025-26 w drużynie 20
    for i in range(12):
        rows.append(dict(GAME_ID=f"a{i}", GAME_DATE=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                         PLAYER_ID=1, SEASON="2024-25", TEAM_ID=10, MIN=30, PTS=20, REB=8, AST=4,
                         games_in_season=i, min_avg_5=30))
    for i in range(3):
        rows.append(dict(GAME_ID=f"b{i}", GAME_DATE=pd.Timestamp("2025-11-01") + pd.Timedelta(days=i),
                         PLAYER_ID=1, SEASON="2025-26", TEAM_ID=20, MIN=20, PTS=10, REB=4, AST=2,
                         games_in_season=i, min_avg_5=20))
    return pd.DataFrame(rows)


def test_season_summary_shifts_to_next_season():
    s = pr.season_summary(_data()).set_index(["PLAYER_ID", "SEASON"])
    row = s.loc[(1, "2025-26")]
    assert row["prev_REB"] == 8 and row["prev_TEAM"] == 10 and row["n_prev"] == 12
    assert (1, "2026-27") in s.index          # podsumowanie 2025-26 trafia do 2026-27


def test_add_features_prev_gap_role_and_age():
    data = _data()
    pred = pd.DataFrame({"GAME_ID": ["b1"], "PLAYER_ID": [1.0], "stat": ["REB"], "line": [4.0],
                         "p_mid": [0.5], "game_date": [pd.Timestamp("2025-11-02")]})
    ages = pd.DataFrame({"PLAYER_ID": [1], "SEASON": ["2025-26"], "AGE": [30.0]})
    m = pr.add_features(pred, data, ages).iloc[0]
    assert np.isclose(m["prev_gap"], (8 - 4) / 2)       # (średnia − linia) / √linia
    assert m["prev_gap_early"] == m["prev_gap"]          # mecz nr 1 sezonu
    assert m["team_changed"] == 1 and m["age"] == 3
    assert m["min_delta"] == 20 - 30
