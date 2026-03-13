# tests/conftest.py
import pytest
import pandas as pd
import sqlite3
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


# ── Dane testowe ──────────────────────────────────────────────────────────────

@pytest.fixture
def fake_game_log_df():
    """Symuluje output PlayerGameLog.get_data_frames()[0]"""
    today = datetime(2026, 3, 5)
    rows = []
    for i in range(20):
        date = today - timedelta(days=i * 2)
        rows.append({
            "GAME_DATE":  date.strftime("%b %d, %Y"),
            "GAME_ID":    f"002250{800 - i}",
            "MATCHUP":    "SAS vs. PHI" if i % 2 == 0 else "SAS @ BOS",
            "PTS":        20 + (i % 10),
            "REB":        8 + (i % 5),
            "AST":        3 + (i % 4),
            "MIN":        "32:00",
            "FGA":        15,
            "FGM":        8,
            "FTA":        6,
            "FTM":        5,
            "WL":         "W" if i % 3 != 0 else "L",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_usage_df():
    """Symuluje output PlayerDashboardByGeneralSplits (Usage)"""
    return pd.DataFrame([{"USG_PCT": 0.284, "PCT_FGA": 0.22}])


@pytest.fixture
def fake_advanced_df():
    """Symuluje PlayerDashboardByGeneralSplits (Advanced)"""
    return pd.DataFrame([{
        "PTS": 24.2, "REB": 10.4, "AST": 3.7,
        "MIN": 33.1, "FGA": 16.2, "FGM": 8.8,
        "FTA": 5.9,  "FTM": 4.7,  "GP": 58,
        "USG_PCT": 0.284,
    }])


@pytest.fixture
def fake_scoreboard_games_df():
    return pd.DataFrame([
        {"GAME_ID": "0022501001", "HOME_TEAM_ID": 1610612759,
         "VISITOR_TEAM_ID": 1610612755, "GAME_STATUS_TEXT": "7:30 pm ET"},
        {"GAME_ID": "0022501002", "HOME_TEAM_ID": 1610612747,
         "VISITOR_TEAM_ID": 1610612756, "GAME_STATUS_TEXT": "10:00 pm ET"},
    ])


@pytest.fixture
def fake_scoreboard_linescore_df():
    return pd.DataFrame([
        {"GAME_ID": "0022501001", "TEAM_ID": 1610612759, "TEAM_ABBREVIATION": "SAS"},
        {"GAME_ID": "0022501001", "TEAM_ID": 1610612755, "TEAM_ABBREVIATION": "PHI"},
        {"GAME_ID": "0022501002", "TEAM_ID": 1610612747, "TEAM_ABBREVIATION": "LAL"},
        {"GAME_ID": "0022501002", "TEAM_ID": 1610612756, "TEAM_ABBREVIATION": "PHX"},
    ])


@pytest.fixture
def fake_team_advanced_df():
    return pd.DataFrame([
        {"TEAM_ID": 1610612755, "DEF_RATING": 113.8, "PACE": 99.2,
         "TEAM_NAME": "Philadelphia 76ers"},
        {"TEAM_ID": 1610612759, "DEF_RATING": 108.4, "PACE": 97.1,
         "TEAM_NAME": "San Antonio Spurs"},
    ])


@pytest.fixture
def fake_ptdefend_guards_df():
    return pd.DataFrame([
        {"TEAM_ID": 1610612755, "CLOSE_DEF_PERSON_ID": 1, "PLAYER_PTS": 18.4},
        {"TEAM_ID": 1610612755, "CLOSE_DEF_PERSON_ID": 2, "PLAYER_PTS": 17.9},
        {"TEAM_ID": 1610612759, "CLOSE_DEF_PERSON_ID": 3, "PLAYER_PTS": 15.2},
    ])


@pytest.fixture
def in_memory_fetcher(tmp_path):
    """
    NBADataFetcher korzystający z tymczasowego SQLite (tmp_path).
    Każdy test dostaje czysty stan — zero resztek między testami.
    """
    # Importujemy po patchowaniu żeby uniknąć side-effectów przy imporcie
    from data_fetcher import NBADataFetcher
    db_path = str(tmp_path / "test.db")
    return NBADataFetcher(db_path=db_path)
