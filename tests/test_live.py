# tests/test_live.py
"""Mapowanie propsów z Odds API na wiersze przyszłych meczów."""
import pandas as pd

from history import team_game_table, player_game_table
from live import props_to_rows, live_features
from tests.test_history import _logs, LAL, BOS


def _games():
    p, t = _logs()
    tg = team_game_table(t)
    return player_game_table(p, tg), tg


def test_prop_maps_team_opponent_and_home():
    games, _ = _games()
    props = [{"player_name": "Test Player", "home_team": "Boston Celtics",
              "away_team": "Los Angeles Lakers", "line": 25.5}]
    rows = props_to_rows(props, games, "2025-01-09")
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["TEAM_ID"] == LAL and r["OPP_ID"] == BOS and r["IS_HOME"] == 0


def test_prop_for_other_teams_game_is_skipped():
    games, _ = _games()
    props = [{"player_name": "Test Player", "home_team": "Miami Heat",
              "away_team": "Chicago Bulls", "line": 25.5}]
    assert props_to_rows(props, games, "2025-01-09").empty


def test_live_features_use_only_history():
    games, tg = _games()
    props = [{"player_name": "Test Player", "home_team": "Los Angeles Lakers",
              "away_team": "Boston Celtics", "line": 25.5}]
    hist, live = live_features(props, games, tg, "2025-01-09")
    assert len(hist) == 4 and len(live) == 1
    assert live.iloc[0]["pts_avg_3"] == 30.0          # (20 + 30 + 40) / 3
    assert live.iloc[0]["IS_HOME"] == 1
