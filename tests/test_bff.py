import pandas as pd

import bff


def _games():
    rows = []
    for i, day in enumerate(pd.date_range("2026-01-01", periods=10)):
        mate_plays = i % 2 == 0
        rows.append({"GAME_ID": f"g{i}", "GAME_DATE": day, "PLAYER_ID": 1, "PTS": 20 if mate_plays else 30})
        if mate_plays:
            rows.append({"GAME_ID": f"g{i}", "GAME_DATE": day, "PLAYER_ID": 2, "PTS": 10})
    return pd.DataFrame(rows)


def test_effect_is_shrunk_by_games_without_mate():
    # mecze g0..g8: z kolegą 5, bez niego 4 (g9 poza okresem kolegi): (30 − 20) · 4 / 9
    assert abs(bff.pair_effect(_games(), 1, 2, "PTS", pd.Timestamp("2026-02-01")) - 40 / 9) < 1e-9


def test_effect_ignores_games_from_the_day_of_the_line():
    assert bff.pair_effect(_games(), 1, 2, "PTS", pd.Timestamp("2026-01-02")) == 0


def test_unknown_mate_has_no_effect():
    assert bff.pair_effect(_games(), 1, 99, "PTS", pd.Timestamp("2026-02-01")) == 0
