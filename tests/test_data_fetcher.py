# tests/test_data_fetcher.py
import pytest
import pandas as pd
import time
from unittest.mock import MagicMock, patch, PropertyMock


# ─────────────────────────────────────────────────────────────────────────────
# Helper: buduje mock endpointu nba_api
# ─────────────────────────────────────────────────────────────────────────────
def make_endpoint_mock(*dataframes):
    """
    nba_api endpoints zwracają dane przez .get_data_frames() → lista DataFrames.
    Ta funkcja buduje mock z odpowiednim return_value.
    """
    mock_ep = MagicMock()
    mock_ep.get_data_frames.return_value = list(dataframes)
    # Dla ScoreboardV2 — dostęp przez atrybuty .game_header, .line_score
    mock_ep.game_header.get_data_frame.return_value = dataframes[0] if dataframes else pd.DataFrame()
    mock_ep.line_score.get_data_frame.return_value = dataframes[1] if len(dataframes) > 1 else pd.DataFrame()
    return mock_ep


# ─────────────────────────────────────────────────────────────────────────────
# TEST: SQLiteCache — zapis, odczyt, TTL
# ─────────────────────────────────────────────────────────────────────────────
class TestSQLiteCache:

    def test_cache_miss_on_empty_db(self, tmp_path):
        from data_fetcher import SQLiteCache
        cache = SQLiteCache(str(tmp_path / "test.db"))
        result = cache.get("nonexistent_key", ttl=3600)
        assert result is None

    def test_cache_set_and_get(self, tmp_path):
        from data_fetcher import SQLiteCache
        cache = SQLiteCache(str(tmp_path / "test.db"))
        df = pd.DataFrame({"PTS": [24, 31, 18], "REB": [9, 12, 7]})
        cache.set("test_key", df)
        result = cache.get("test_key", ttl=3600)
        pd.testing.assert_frame_equal(result, df)

    def test_cache_miss_when_expired(self, tmp_path):
        """TTL=0 powinno zawsze zwracać None (dane już 'przeterminowane')."""
        from data_fetcher import SQLiteCache
        cache = SQLiteCache(str(tmp_path / "test.db"))
        df = pd.DataFrame({"PTS": [20]})
        cache.set("expired_key", df)
        # TTL=0 → każdy wpis jest od razu przeterminowany
        result = cache.get("expired_key", ttl=0)
        assert result is None

    def test_cache_overwrites_old_entry(self, tmp_path):
        from data_fetcher import SQLiteCache
        cache = SQLiteCache(str(tmp_path / "test.db"))
        df_v1 = pd.DataFrame({"PTS": [10]})
        df_v2 = pd.DataFrame({"PTS": [99]})
        cache.set("key", df_v1)
        cache.set("key", df_v2)
        result = cache.get("key", ttl=3600)
        assert int(result["PTS"].iloc[0]) == 99

    def test_all_tables_created(self, tmp_path):
        """Weryfikuje że _init_db() tworzy wszystkie 4 tabele."""
        import sqlite3
        from data_fetcher import SQLiteCache
        db_path = str(tmp_path / "test.db")
        SQLiteCache(db_path)
        with sqlite3.connect(db_path) as conn:
            tables = {
                row[0] for row in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        assert {"cache", "odds_history", "coupons", "results"}.issubset(tables)


# ─────────────────────────────────────────────────────────────────────────────
# TEST: get_todays_games()
# ─────────────────────────────────────────────────────────────────────────────
class TestGetTodaysGames:

    @patch("data_fetcher._rand_sleep")  # blokujemy sleep — testy mają być szybkie
    @patch("data_fetcher.scoreboardv2.ScoreboardV2")
    def test_returns_correct_game_count(
        self, mock_scoreboard, mock_sleep,
        in_memory_fetcher, fake_scoreboard_games_df, fake_scoreboard_linescore_df
    ):
        mock_ep = make_endpoint_mock(fake_scoreboard_games_df, fake_scoreboard_linescore_df)
        mock_scoreboard.return_value = mock_ep

        games = in_memory_fetcher.get_todays_games()

        assert len(games) == 2

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.scoreboardv2.ScoreboardV2")
    def test_game_structure(
        self, mock_scoreboard, mock_sleep,
        in_memory_fetcher, fake_scoreboard_games_df, fake_scoreboard_linescore_df
    ):
        mock_ep = make_endpoint_mock(fake_scoreboard_games_df, fake_scoreboard_linescore_df)
        mock_scoreboard.return_value = mock_ep

        games = in_memory_fetcher.get_todays_games()
        game = games[0]

        # Weryfikujemy że wszystkie klucze są obecne
        required_keys = {"game_id", "home_team", "away_team",
                         "home_team_id", "away_team_id", "game_time_et", "status"}
        assert required_keys.issubset(game.keys())

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.scoreboardv2.ScoreboardV2")
    def test_team_abbreviations_correct(
        self, mock_scoreboard, mock_sleep,
        in_memory_fetcher, fake_scoreboard_games_df, fake_scoreboard_linescore_df
    ):
        mock_ep = make_endpoint_mock(fake_scoreboard_games_df, fake_scoreboard_linescore_df)
        mock_scoreboard.return_value = mock_ep

        games = in_memory_fetcher.get_todays_games()
        # Pierwszy mecz: SAS (home) vs PHI (away)
        assert games[0]["home_team"] == "SAS"
        assert games[0]["away_team"] == "PHI"

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.scoreboardv2.ScoreboardV2")
    def test_empty_scoreboard_returns_empty_list(
        self, mock_scoreboard, mock_sleep, in_memory_fetcher
    ):
        mock_ep = make_endpoint_mock(pd.DataFrame(), pd.DataFrame())
        mock_scoreboard.return_value = mock_ep

        games = in_memory_fetcher.get_todays_games()
        assert games == []

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.scoreboardv2.ScoreboardV2")
    def test_second_call_uses_cache(
        self, mock_scoreboard, mock_sleep,
        in_memory_fetcher, fake_scoreboard_games_df, fake_scoreboard_linescore_df
    ):
        """API powinno być wywołane TYLKO raz — drugi call trafia w cache."""
        mock_ep = make_endpoint_mock(fake_scoreboard_games_df, fake_scoreboard_linescore_df)
        mock_scoreboard.return_value = mock_ep

        in_memory_fetcher.get_todays_games()
        in_memory_fetcher.get_todays_games()

        mock_scoreboard.assert_called_once()  # ← kluczowy assert!


# ─────────────────────────────────────────────────────────────────────────────
# TEST: get_player_recent_stats()
# ─────────────────────────────────────────────────────────────────────────────
class TestGetPlayerRecentStats:

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_returns_correct_columns(
        self, mock_gamelog, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_game_log_df, fake_usage_df
    ):
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]
        mock_dashboard.return_value.get_data_frames.return_value = [fake_usage_df]

        result = in_memory_fetcher.get_player_recent_stats(1641705, last_n=5)

        expected_cols = {"PTS", "REB", "AST", "is_home",
                         "days_rest", "is_back_to_back", "opponent_team_id"}
        assert expected_cols.issubset(set(result.columns))

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_respects_last_n_param(
        self, mock_gamelog, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_game_log_df, fake_usage_df
    ):
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]
        mock_dashboard.return_value.get_data_frames.return_value = [fake_usage_df]

        result = in_memory_fetcher.get_player_recent_stats(1641705, last_n=5)
        assert len(result) == 5

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_is_home_parsed_correctly(
        self, mock_gamelog, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_game_log_df, fake_usage_df
    ):
        """'SAS vs. PHI' = home=1, 'SAS @ BOS' = home=0"""
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]
        mock_dashboard.return_value.get_data_frames.return_value = [fake_usage_df]

        result = in_memory_fetcher.get_player_recent_stats(1641705, last_n=20)

        # Mecze z "vs." powinny mieć is_home=1
        home_games = result[result["is_home"] == 1]
        away_games = result[result["is_home"] == 0]
        assert len(home_games) > 0
        assert len(away_games) > 0

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_back_to_back_detection(
        self, mock_gamelog, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_usage_df
    ):
        """Mecz po 1 dniu odpoczynku → is_back_to_back=1."""
        from datetime import datetime, timedelta
        # Konstruujemy mecze z jawnym B2B
        dates = [
            datetime(2026, 3, 5),
            datetime(2026, 3, 4),  # B2B z 5 marca
            datetime(2026, 3, 2),
        ]
        rows = [
            {"GAME_DATE": d.strftime("%b %d, %Y"), "GAME_ID": f"002250{i}",
             "MATCHUP": "SAS vs. PHI", "PTS": 25, "REB": 8, "AST": 3,
             "MIN": "33:00", "FGA": 15, "FGM": 8, "FTA": 5, "FTM": 4, "WL": "W"}
            for i, d in enumerate(dates)
        ]
        b2b_df = pd.DataFrame(rows)

        mock_gamelog.return_value.get_data_frames.return_value = [b2b_df]
        mock_dashboard.return_value.get_data_frames.return_value = [fake_usage_df]

        result = in_memory_fetcher.get_player_recent_stats(9999, last_n=3)
        # Mecz z 4 marca (1 dzień po 3 marca) powinien być B2B
        b2b_rows = result[result["is_back_to_back"] == 1]
        assert len(b2b_rows) >= 1

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_usg_pct_applied(
        self, mock_gamelog, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_game_log_df
    ):
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]
        mock_dashboard.return_value.get_data_frames.return_value = [
            pd.DataFrame([{"USG_PCT": 0.312}])
        ]
        result = in_memory_fetcher.get_player_recent_stats(1641705, last_n=5)
        assert (result["USG_PCT"] == 0.312).all()

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_empty_game_log_returns_empty_df(
        self, mock_gamelog, mock_dashboard, mock_sleep, in_memory_fetcher
    ):
        mock_gamelog.return_value.get_data_frames.return_value = [pd.DataFrame()]
        result = in_memory_fetcher.get_player_recent_stats(9999)
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# TEST: get_player_season_avg()
# ─────────────────────────────────────────────────────────────────────────────
class TestGetPlayerSeasonAvg:

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    def test_returns_pts_reb_ast(
        self, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_advanced_df
    ):
        # Oba wywołania (Base + Advanced) zwracają ten sam fake df
        mock_dashboard.return_value.get_data_frames.return_value = [fake_advanced_df]

        result = in_memory_fetcher.get_player_season_avg(1641705)

        assert pytest.approx(result["PTS"], 0.1) == 24.2
        assert pytest.approx(result["USG_PCT"], 0.001) == 0.284

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits")
    def test_cached_on_second_call(
        self, mock_dashboard, mock_sleep,
        in_memory_fetcher, fake_advanced_df
    ):
        mock_dashboard.return_value.get_data_frames.return_value = [fake_advanced_df]
        in_memory_fetcher.get_player_season_avg(1641705)
        in_memory_fetcher.get_player_season_avg(1641705)
        # 2 wywołania API (Base + Advanced) tylko raz — potem cache
        assert mock_dashboard.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# TEST: get_player_vs_opponent()
# ─────────────────────────────────────────────────────────────────────────────
class TestGetPlayerVsOpponent:

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_filters_by_opponent(
        self, mock_gamelog, mock_sleep, in_memory_fetcher, fake_game_log_df
    ):
        """Powinny wrócić tylko mecze vs PHI (MATCHUP: 'SAS vs. PHI')."""
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]

        phi_id = 1610612755  # Philadelphia
        result = in_memory_fetcher.get_player_vs_opponent(1641705, phi_id, last_n=10)

        # Wszystkie mecze muszą zawierać 'PHI' w MATCHUP
        if not result.empty and "MATCHUP" in result.columns:
            assert result["MATCHUP"].str.contains("PHI").all()

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_respects_last_n(
        self, mock_gamelog, mock_sleep, in_memory_fetcher, fake_game_log_df
    ):
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]
        result = in_memory_fetcher.get_player_vs_opponent(1641705, 1610612755, last_n=3)
        assert len(result) <= 3

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.playergamelog.PlayerGameLog")
    def test_fetches_3_seasons(
        self, mock_gamelog, mock_sleep, in_memory_fetcher, fake_game_log_df
    ):
        """Metoda powinna wywołać API 3 razy (3 sezony wstecz)."""
        mock_gamelog.return_value.get_data_frames.return_value = [fake_game_log_df]
        in_memory_fetcher.get_player_vs_opponent(1641705, 1610612755)
        assert mock_gamelog.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# TEST: get_team_defensive_stats()
# ─────────────────────────────────────────────────────────────────────────────
class TestGetTeamDefensiveStats:

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.leaguedashptdefend.LeagueDashPtDefend")
    @patch("data_fetcher.leaguedashteamstats.LeagueDashTeamStats")
    def test_returns_def_rating_and_pace(
        self, mock_team_stats, mock_ptdefend, mock_sleep,
        in_memory_fetcher, fake_team_advanced_df, fake_ptdefend_guards_df
    ):
        mock_team_stats.return_value.get_data_frames.return_value = [fake_team_advanced_df]
        mock_ptdefend.return_value.get_data_frames.return_value = [fake_ptdefend_guards_df]

        result = in_memory_fetcher.get_team_defensive_stats(1610612755)

        assert pytest.approx(result["def_rating"], 0.1) == 113.8
        assert pytest.approx(result["pace"], 0.1) == 99.2

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.leaguedashptdefend.LeagueDashPtDefend")
    @patch("data_fetcher.leaguedashteamstats.LeagueDashTeamStats")
    def test_returns_all_position_keys(
        self, mock_team_stats, mock_ptdefend, mock_sleep,
        in_memory_fetcher, fake_team_advanced_df, fake_ptdefend_guards_df
    ):
        mock_team_stats.return_value.get_data_frames.return_value = [fake_team_advanced_df]
        mock_ptdefend.return_value.get_data_frames.return_value = [fake_ptdefend_guards_df]

        result = in_memory_fetcher.get_team_defensive_stats(1610612755)

        for pos in ["PG", "SG", "SF", "PF", "C"]:
            assert f"pts_allowed_{pos}" in result, f"Brak klucza pts_allowed_{pos}"

    @patch("data_fetcher._rand_sleep")
    @patch("data_fetcher.leaguedashptdefend.LeagueDashPtDefend")
    @patch("data_fetcher.leaguedashteamstats.LeagueDashTeamStats")
    def test_unknown_team_returns_empty_dict(
        self, mock_team_stats, mock_ptdefend, mock_sleep,
        in_memory_fetcher, fake_team_advanced_df
    ):
        mock_team_stats.return_value.get_data_frames.return_value = [fake_team_advanced_df]
        # team_id=9999 nie istnieje w fake_team_advanced_df
        result = in_memory_fetcher.get_team_defensive_stats(9999)
        assert result == {}
