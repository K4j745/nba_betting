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


def _games_with_teammate():
    """Zawodnik 7 (z propsem) + kolega 8 z rotacji (30 min), oba w LAL."""
    p, t = _logs()
    mate = p.copy()
    mate["PLAYER_ID"], mate["PLAYER_NAME"], mate["PTS"] = 8, "Mate Player", 12
    tg = team_game_table(t)
    return player_game_table(pd.concat([p, mate], ignore_index=True), tg), tg


PROPS = [{"player_name": "Test Player", "home_team": "Los Angeles Lakers",
          "away_team": "Boston Celtics", "line": 25.5}]


def test_injury_report_drives_live_absences():
    games, tg = _games_with_teammate()
    none_out = pd.DataFrame(columns=["team", "name_key"])
    _, live = live_features(PROPS, games, tg, "2025-01-09", out=none_out)
    assert len(live) == 1                              # wiersze zastępcze usunięte
    assert live.iloc[0]["n_out"] == 0                  # kolega bez propsa NIE jest „nieobecny”

    mate_out = pd.DataFrame({"team": ["LAL"], "name_key": ["mate player"]})
    _, live = live_features(PROPS, games, tg, "2025-01-09", out=mate_out)
    assert live.iloc[0]["n_out"] == 1
    assert live.iloc[0]["vac_min"] == 30.0
    assert live.iloc[0]["vac_pts"] == 12.0
