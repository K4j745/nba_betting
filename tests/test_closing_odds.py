# tests/test_closing_odds.py
"""Drugi run (kursy zamknięcia): okno czasowe, brak duplikatów, budżet, priorytet kuponu."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from odds_fetcher import OddsFetcher, _month_end

NOW = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)   # 11 dni do końca listopada


def _event(eid, minutes, home="Boston Celtics", away="Los Angeles Lakers"):
    t = (NOW + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": t}


def _props(event, *_):
    return [
        {"player_name": "LeBron James", "stat_type": "player_points", "line": 24.5, "side": s,
         "odds": 1.9, "home_team": event["home_team"], "away_team": event["away_team"],
         "game_time": event["commence_time"], "bookmaker": "pinnacle", "source": "odds_api"}
        for s in ("over", "under")
    ]


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    f = OddsFetcher(db_path=str(tmp_path / "t.db"), nba_player_names=["LeBron James"])
    calls = []

    def event_props(event, *a):
        calls.append(event["id"])
        f.api_client._requests_remaining -= 1
        return _props(event)

    monkeypatch.setattr(f.api_client, "get_event_props_raw", event_props)
    f.api_client._requests_remaining = 400
    f._calls = calls
    return f


def test_month_end():
    assert _month_end(NOW.date()).day == 30
    assert _month_end(datetime(2027, 2, 3).date()).day == 28


def test_window_and_no_duplicates(fetcher, monkeypatch):
    events = [_event("soon", 10), _event("due", 60), _event("later", 120)]
    monkeypatch.setattr(fetcher.api_client, "get_events", lambda *a: events)
    s = fetcher.fetch_closing(now=NOW)
    assert fetcher._calls == ["due"] and s["fetched"] == 1
    with sqlite3.connect(fetcher.db_path) as c:
        src = c.execute("SELECT DISTINCT source FROM odds_history").fetchall()
    assert src == [("odds_api_close",)]
    fetcher.fetch_closing(now=NOW + timedelta(minutes=30))   # ten sam mecz nadal w oknie
    assert fetcher._calls == ["due"]


def test_budget_reserve_and_priority(fetcher, monkeypatch):
    events = [_event("a", 50, home="Miami Heat", away="Orlando Magic"), _event("b", 55)]
    monkeypatch.setattr(fetcher.api_client, "get_events", lambda *a: events)
    # rezerwa = 11 dni × 15 = 165; limit 166 → stać nas na dokładnie jeden mecz
    fetcher.api_client._requests_remaining = 166
    s = fetcher.fetch_closing(now=NOW, priority_teams=frozenset({"Boston Celtics"}))
    assert s["reserve"] == 165
    assert fetcher._calls == ["b"]            # mecz z picku pierwszy
    assert s["skipped_budget"] == 1
