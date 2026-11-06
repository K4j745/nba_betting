# tests/test_injuries.py
"""Raport kontuzji NBA: nazwy plików, nazwiska, filtr Out/Doubtful."""
from datetime import datetime

import pandas as pd

from injuries import ET, _player_key, out_players, report_names


def test_report_names_new_and_old_format():
    assert report_names(datetime(2026, 10, 29, 17, 30, tzinfo=ET)) == ["2026-10-29_05_30PM"]
    assert report_names(datetime(2025, 12, 1, 17, 0, tzinfo=ET)) == ["2025-12-01_05_00PM", "2025-12-01_05PM"]


def test_player_key_reorders_and_normalizes():
    assert _player_key("Porter Jr., Michael") == "michael porter"
    assert _player_key("Williams III, Robert") == "robert williams"
    assert _player_key("Tomlin, Nae'Qwan") == "nae qwan tomlin"


def test_out_players_only_out_and_doubtful_for_day():
    r = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-10-29"] * 4 + ["2026-10-30"]),
        "team": ["CLE", "CLE", "ATL", "ATL", "ATL"],
        "name_key": ["a", "b", "c", "d", "e"],
        "status": ["Out", "Questionable", "Doubtful", "Available", "Out"],
    })
    o = out_players(r, "2026-10-29")
    assert sorted(o["name_key"]) == ["a", "c"]
