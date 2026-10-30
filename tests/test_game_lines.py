# tests/test_game_lines.py
"""Spread/total z Odds API (krok 4c): parsowanie i zapis do game_lines."""
import sqlite3

from odds_fetcher import OddsAPIClient, OddsFetcher

EVENT = {
    "home_team": "Boston Celtics", "away_team": "Los Angeles Lakers",
    "commence_time": "2026-10-31T23:30:00Z",
    "bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "spreads", "outcomes": [
            {"name": "Boston Celtics", "price": 1.91, "point": -6.5},
            {"name": "Los Angeles Lakers", "price": 1.95, "point": 6.5}]},
        {"key": "totals", "outcomes": [
            {"name": "Over", "price": 1.90, "point": 228.5},
            {"name": "Under", "price": 1.96, "point": 228.5}]},
    ]}],
}


def test_game_lines_parsing(monkeypatch):
    client = OddsAPIClient(api_key="test")
    monkeypatch.setattr(client, "_get", lambda path, params: [EVENT])
    rows = {r["market"]: r for r in client.get_game_lines_raw()}
    assert rows["spreads"]["point"] == -6.5          # spread gospodarza
    assert rows["spreads"]["price_a"] == 1.91
    assert rows["totals"]["point"] == 228.5
    assert rows["totals"]["price_b"] == 1.96


def test_fetch_game_lines_saves_rows(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    f = OddsFetcher(db_path=db)
    monkeypatch.setattr(f.api_client, "_get", lambda path, params: [EVENT])
    assert f.fetch_game_lines() == 2
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM game_lines").fetchone()[0] == 2
