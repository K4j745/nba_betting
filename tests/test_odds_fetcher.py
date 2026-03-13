# tests/test_odds_fetcher.py
import pytest
import pandas as pd
import sqlite3
from unittest.mock import MagicMock, patch
from odds_fetcher import (
    OddsFetcher, OddsAPIClient, PlayerNameMatcher,
    american_to_decimal, decimal_in_range, _pair_over_under,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

NBA_NAMES = [
    "Victor Wembanyama",
    "Jayson Tatum",
    "Shai Gilgeous-Alexander",
    "LeBron James",
    "Stephen Curry",
]

@pytest.fixture
def fetcher(tmp_path):
    return OddsFetcher(
        db_path=str(tmp_path / "test.db"),
        nba_player_names=NBA_NAMES,
        odds_min=1.40,
        odds_max=1.90,
    )

@pytest.fixture
def raw_api_outcomes():
    """Symuluje surowy output Odds API (osobne over/under)."""
    return [
        {"player_name": "Victor Wembanyama", "stat_type": "player_points",
         "line": 25.5, "side": "over", "odds": 1.83,
         "home_team": "SAS", "away_team": "PHI",
         "game_time": "2026-03-05T23:00:00Z",
         "bookmaker": "pinnacle", "source": "odds_api"},
        {"player_name": "Victor Wembanyama", "stat_type": "player_points",
         "line": 25.5, "side": "under", "odds": 1.87,
         "home_team": "SAS", "away_team": "PHI",
         "game_time": "2026-03-05T23:00:00Z",
         "bookmaker": "pinnacle", "source": "odds_api"},
        {"player_name": "Jayson Tatum", "stat_type": "player_points",
         "line": 28.5, "side": "over", "odds": 2.10,  # poza zakresem!
         "home_team": "BOS", "away_team": "MIL",
         "game_time": "2026-03-05T23:30:00Z",
         "bookmaker": "bet365", "source": "odds_api"},
        {"player_name": "Jayson Tatum", "stat_type": "player_points",
         "line": 28.5, "side": "under", "odds": 1.70,
         "home_team": "BOS", "away_team": "MIL",
         "game_time": "2026-03-05T23:30:00Z",
         "bookmaker": "bet365", "source": "odds_api"},
    ]


# ── Testy konwersji kursów ─────────────────────────────────────────────────────

class TestOddsUtils:
    def test_american_plus_to_decimal(self):
        assert american_to_decimal(150) == pytest.approx(2.5, 0.01)

    def test_american_minus_to_decimal(self):
        assert american_to_decimal(-110) == pytest.approx(1.909, 0.01)

    def test_decimal_in_range_true(self):
        assert decimal_in_range(1.65) is True

    def test_decimal_in_range_too_low(self):
        assert decimal_in_range(1.30) is False

    def test_decimal_in_range_too_high(self):
        assert decimal_in_range(2.10) is False

    def test_decimal_on_boundary(self):
        assert decimal_in_range(1.40) is True
        assert decimal_in_range(1.90) is True


# ── Testy fuzzy matchingu ──────────────────────────────────────────────────────

class TestPlayerNameMatcher:
    def setup_method(self):
        self.matcher = PlayerNameMatcher(NBA_NAMES)

    def test_exact_match(self):
        assert self.matcher.match("Jayson Tatum") == "Jayson Tatum"

    def test_hyphen_variation(self):
        # "Shai Gilgeous Alexander" (bez myślnika) → canonical z myślnikiem
        result = self.matcher.match("Shai Gilgeous Alexander")
        assert result == "Shai Gilgeous-Alexander"

    def test_partial_name(self):
        # Samo nazwisko bez imienia = niższy score (~74) — obniżamy próg dla tego case'u
        result = self.matcher.match("Wembanyama", threshold=70)
        assert result == "Victor Wembanyama"

    def test_no_match_returns_none(self):
        result = self.matcher.match("XYZ NonExistentPlayer99")
        assert result is None

    def test_case_insensitive(self):
        result = self.matcher.match("lebron james")
        assert result == "LeBron James"

    def test_empty_canonical_list(self):
        matcher = PlayerNameMatcher([])
        assert matcher.match("LeBron James") is None


# ── Testy _pair_over_under ─────────────────────────────────────────────────────

class TestPairOverUnder:
    def test_pairs_correctly(self, raw_api_outcomes):
        paired = _pair_over_under(raw_api_outcomes[:2])  # tylko Wemby
        assert len(paired) == 1
        assert paired[0]["over_odds"]  == 1.83
        assert paired[0]["under_odds"] == 1.87

    def test_discards_incomplete_pairs(self):
        # Tylko over, brak under → powinno być odrzucone
        incomplete = [
            {"player_name": "LeBron James", "stat_type": "player_points",
             "line": 22.5, "side": "over", "odds": 1.75,
             "home_team": "LAL", "away_team": "PHX",
             "game_time": "", "bookmaker": "pinnacle", "source": "odds_api"},
        ]
        paired = _pair_over_under(incomplete)
        assert len(paired) == 0

    def test_multiple_players(self, raw_api_outcomes):
        paired = _pair_over_under(raw_api_outcomes)
        # Wemby + Tatum = 2 pary
        assert len(paired) == 2
        players = {p["player_name"] for p in paired}
        assert "Victor Wembanyama" in players
        assert "Jayson Tatum" in players


# ── Testy filtrowania ─────────────────────────────────────────────────────────

class TestFilterOdds:
    def test_filters_out_of_range(self, fetcher, raw_api_outcomes):
        paired = _pair_over_under(raw_api_outcomes)
        # Tatum: over=2.10 (za wysoki), under=1.70 (OK) → prop przechodzi
        # Wemby: over=1.83, under=1.87 → oba OK
        filtered = fetcher._filter_odds(paired)
        assert len(filtered) == 2  # oba przechodzą bo under Tatuma = 1.70

    def test_rejects_all_out_of_range(self, fetcher):
        props = [{
            "player_name": "Stephen Curry",
            "stat_type": "player_points",
            "line": 30.5,
            "over_odds": 2.50,   # za wysoki
            "under_odds": 1.25,  # za niski
            "home_team": "GSW", "away_team": "LAL",
            "game_time": "", "bookmaker": "bet365", "source": "odds_api",
        }]
        assert fetcher._filter_odds(props) == []


# ── Testy zapisu do SQLite ────────────────────────────────────────────────────

class TestSaveToDb:
    def test_saves_props(self, fetcher, raw_api_outcomes):
        paired = _pair_over_under(raw_api_outcomes)
        fetcher._save_to_db(paired)

        with fetcher._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM odds_history"
            ).fetchone()[0]
        assert count == len(paired)

    def test_saves_correct_player_name(self, fetcher, raw_api_outcomes):
        paired = _pair_over_under(raw_api_outcomes[:2])  # Wemby
        fetcher._save_to_db(paired)

        with fetcher._conn() as conn:
            row = conn.execute(
                "SELECT player_name, line, over_odds FROM odds_history LIMIT 1"
            ).fetchone()
        assert row[0] == "Victor Wembanyama"
        assert row[1] == 25.5
        assert row[2] == 1.83


# ── Testy głównej metody (z mockiem API) ──────────────────────────────────────

class TestGetPlayerProps:

    @patch("odds_fetcher.asyncio.run", return_value=[])  # Playwright → nic
    @patch.object(OddsAPIClient, "get_player_props_raw")
    def test_falls_back_to_odds_api(
        self, mock_api, mock_playwright, fetcher, raw_api_outcomes
    ):
        mock_api.return_value = raw_api_outcomes
        props = fetcher.get_player_props(use_scraping=True)
        # Playwright zwrócił [] → powinien użyć Odds API
        mock_api.assert_called_once()
        assert len(props) > 0

    @patch("odds_fetcher.asyncio.run", return_value=[])
    @patch.object(OddsAPIClient, "get_player_props_raw", return_value=[])
    def test_returns_empty_when_both_fail(
        self, mock_api, mock_playwright, fetcher
    ):
        result = fetcher.get_player_props()
        assert result == []

    @patch("odds_fetcher.asyncio.run", return_value=[])
    @patch.object(OddsAPIClient, "get_player_props_raw")
    def test_props_have_required_keys(
        self, mock_api, mock_playwright, fetcher, raw_api_outcomes
    ):
        mock_api.return_value = raw_api_outcomes
        props = fetcher.get_player_props()

        required = {"player_name", "stat_type", "line",
                    "over_odds", "under_odds", "home_team", "away_team"}
        for prop in props:
            assert required.issubset(prop.keys())

    @patch("odds_fetcher.asyncio.run", return_value=[])
    @patch.object(OddsAPIClient, "get_player_props_raw")
    def test_props_saved_to_db(
        self, mock_api, mock_playwright, fetcher, raw_api_outcomes
    ):
        mock_api.return_value = raw_api_outcomes
        fetcher.get_player_props()

        with fetcher._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM odds_history"
            ).fetchone()[0]
        assert count > 0


# ── Test line movement ─────────────────────────────────────────────────────────

class TestLineMovement:
    def test_returns_dataframe(self, fetcher, raw_api_outcomes):
        paired = _pair_over_under(raw_api_outcomes[:2])
        fetcher._save_to_db(paired)
        df = fetcher.get_line_movement("Victor Wembanyama", hours_back=24)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_empty_for_unknown_player(self, fetcher):
        df = fetcher.get_line_movement("NonExistent Player")
        assert df.empty
