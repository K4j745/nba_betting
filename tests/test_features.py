# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from features import (
    build_feature_vector,
    compute_rolling_features,
    compute_h2h_features,
    compute_game_context,
    compute_defensive_features,
    compute_market_features,
    remove_overround,
    MODEL_FEATURE_COLS,
    _parse_minutes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_recent_df():
    """20 meczów z rosnącymi punktami (symulacja dobrej formy)."""
    today = datetime(2026, 3, 5)
    rows = []
    for i in range(20):
        rows.append({
            "GAME_DATE":      today - timedelta(days=i * 2),
            "GAME_ID":        f"00225{i:04d}",
            "PTS":            29.5 - i * 0.5,   # trend rosnący #dokonano zmiany z 20.0 + i *0.5 z powodu błędu przy teście kompilacji
            "REB":            8.0,
            "AST":            3.0,
            "MIN":            "33:00",
            "FGA":            15,  "FGM": 8,
            "FTA":            5,   "FTM": 4,
            "USG_PCT":        0.284,
            "opponent_team_id": 1610612755,
            "is_home":        i % 2,
            "days_rest":      2 if i > 0 else 3,
            "is_back_to_back": 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_vs_df():
    """5 meczów vs konkretny rywal."""
    today = datetime(2026, 3, 5)
    return pd.DataFrame([
        {"GAME_DATE": today - timedelta(days=30 * i),
         "PTS": 22.0 + i, "REB": 9, "AST": 3,
         "MIN": "34:00", "MATCHUP": "SAS vs. PHI"}
        for i in range(5)
    ])


@pytest.fixture
def sample_team_def():
    return {
        "team_id":    1610612755,
        "def_rating": 113.8,
        "pace":        99.2,
        "pts_allowed_PG": 18.4,
        "pts_allowed_SG": 18.4,
        "pts_allowed_SF": 17.9,
        "pts_allowed_PF": 17.9,
        "pts_allowed_C":  14.6,
    }


@pytest.fixture
def mock_fetcher(sample_recent_df, sample_vs_df, sample_team_def):
    """Mockowany NBADataFetcher — bez żadnych wywołań API."""
    fetcher = MagicMock()
    fetcher.get_player_recent_stats.return_value  = sample_recent_df
    fetcher.get_player_vs_opponent.return_value   = sample_vs_df
    fetcher.get_team_defensive_stats.return_value = sample_team_def
    return fetcher


# ── Testy remove_overround ────────────────────────────────────────────────────

class TestRemoveOverround:
    def test_probs_sum_to_one(self):
        p_over, p_under = remove_overround(1.83, 1.87)
        assert abs(p_over + p_under - 1.0) < 0.0001

    def test_symmetric_odds_give_50_50(self):
        p_over, p_under = remove_overround(1.90, 1.90)
        assert p_over  == pytest.approx(0.5, abs=0.001)
        assert p_under == pytest.approx(0.5, abs=0.001)

    def test_favoured_side_higher_prob(self):
        # under=1.70 (niższy kurs = faworyt) → p_under powinno być > 0.5
        p_over, p_under = remove_overround(2.10, 1.70)
        assert p_under > p_over

    def test_removes_margin(self):
        # Raw: 1/1.83 + 1/1.87 = 1.081 → po normalizacji suma = 1.0
        p_over, p_under = remove_overround(1.83, 1.87)
        assert p_over + p_under == pytest.approx(1.0, abs=0.0001)


# ── Testy _parse_minutes ──────────────────────────────────────────────────────

class TestParseMinutes:
    def test_colon_format(self):
        assert _parse_minutes("33:30") == pytest.approx(33.5, abs=0.01)

    def test_decimal_format(self):
        assert _parse_minutes("33.5") == pytest.approx(33.5, abs=0.01)

    def test_integer_string(self):
        assert _parse_minutes("33") == pytest.approx(33.0, abs=0.01)

    def test_invalid_returns_zero(self):
        assert _parse_minutes("N/A") == 0.0


# ── Testy compute_rolling_features ───────────────────────────────────────────

class TestComputeRollingFeatures:
    def test_returns_all_keys(self, sample_recent_df):
        result = compute_rolling_features(sample_recent_df, line=25.5)
        expected_keys = {
            "pts_avg_3", "pts_avg_5", "pts_avg_10", "pts_avg_20",
            "pts_std_10", "pts_trend", "pct_over_line_historical",
            "min_avg_5", "usg_pct_avg_5",
        }
        assert expected_keys.issubset(result.keys())

    def test_avg_3_uses_last_3_games(self, sample_recent_df):
        result  = compute_rolling_features(sample_recent_df, line=25.5)
        df_sorted = sample_recent_df.sort_values("GAME_DATE")
        expected  = float(df_sorted["PTS"].tail(3).mean())
        assert result["pts_avg_3"] == pytest.approx(expected, abs=0.01)

    def test_trend_positive_for_hot_player(self, sample_recent_df):
        """Rosnące punkty → trend pozytywny (avg3 > avg10)."""
        result = compute_rolling_features(sample_recent_df, line=25.5)
        assert result["pts_trend"] > 0

    def test_pct_over_line_correct(self):
        """Gracz zawsze powyżej linii 15 → pct = 1.0."""
        df = pd.DataFrame({
            "GAME_DATE": pd.date_range("2026-01-01", periods=10, freq="2D"),
            "PTS": [20.0] * 10,
            "MIN": ["30:00"] * 10,
            "USG_PCT": [0.25] * 10,
        })
        result = compute_rolling_features(df, line=15.0)
        assert result["pct_over_line_historical"] == 1.0

    def test_empty_df_returns_defaults(self):
        result = compute_rolling_features(pd.DataFrame(), line=25.5)
        assert result["pct_over_line_historical"] == 0.5
        assert result["pts_avg_10"] == 0.0

    def test_usg_pct_averaged_over_5(self, sample_recent_df):
        result = compute_rolling_features(sample_recent_df, line=25.5)
        assert result["usg_pct_avg_5"] == pytest.approx(0.284, abs=0.001)


# ── Testy compute_h2h_features ───────────────────────────────────────────────

class TestComputeH2HFeatures:
    def test_correct_avg(self, sample_vs_df):
        result = compute_h2h_features(sample_vs_df, line=25.5, last_n=5)
        expected_avg = float(sample_vs_df.head(5)["PTS"].mean())
        assert result["pts_avg_vs_opponent"] == pytest.approx(expected_avg, abs=0.01)

    def test_pct_over_correct(self):
        df = pd.DataFrame({"PTS": [30.0, 20.0, 28.0, 18.0, 32.0]})
        result = compute_h2h_features(df, line=25.5)
        # 30, 28, 32 > 25.5 → 3/5 = 0.6
        assert result["pct_over_vs_opponent"] == pytest.approx(0.6, abs=0.001)

    def test_empty_df_returns_defaults(self):
        result = compute_h2h_features(pd.DataFrame(), line=25.5)
        assert result["pct_over_vs_opponent"] == 0.5
        assert result["h2h_games_available"]  == 0

    def test_games_available_capped_at_last_n(self, sample_vs_df):
        result = compute_h2h_features(sample_vs_df, line=25.5, last_n=3)
        assert result["h2h_games_available"] == 3


# ── Testy compute_defensive_features ─────────────────────────────────────────

class TestComputeDefensiveFeatures:
    def test_pg_uses_correct_key(self, sample_team_def):
        result = compute_defensive_features(sample_team_def, "PG")
        assert result["opp_pts_allowed_to_position"] == pytest.approx(18.4, abs=0.01)

    def test_center_uses_c_key(self, sample_team_def):
        result = compute_defensive_features(sample_team_def, "C")
        assert result["opp_pts_allowed_to_position"] == pytest.approx(14.6, abs=0.01)

    def test_empty_dict_returns_defaults(self):
        result = compute_defensive_features({})
        assert result["opp_def_rating"] == 110.0
        assert result["opp_pace"]       == 99.0

    def test_returns_def_rating_and_pace(self, sample_team_def):
        result = compute_defensive_features(sample_team_def, "SG")
        assert result["opp_def_rating"] == pytest.approx(113.8, abs=0.01)
        assert result["opp_pace"]       == pytest.approx(99.2,  abs=0.01)


# ── Testy compute_market_features ────────────────────────────────────────────

class TestComputeMarketFeatures:
    def test_line_preserved(self):
        result = compute_market_features(25.5, 1.83, 1.87)
        assert result["line"] == 25.5

    def test_implied_probs_correct(self):
        result = compute_market_features(25.5, 1.83, 1.87)
        assert result["over_implied_prob"]  + result["under_implied_prob"] == pytest.approx(1.0, abs=0.0001)

    def test_margin_positive(self):
        result = compute_market_features(25.5, 1.83, 1.87)
        assert result["market_margin"] > 0


# ── Testy build_feature_vector (integracyjny) ─────────────────────────────────

class TestBuildFeatureVector:
    def test_returns_dataframe(self, mock_fetcher):
        df = build_feature_vector(
            player_id=1641705, opponent_team_id=1610612755,
            line=25.5, over_odds=1.83, under_odds=1.87,
            is_home=1, player_position="C", fetcher=mock_fetcher,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_contains_all_model_features(self, mock_fetcher):
        df = build_feature_vector(
            player_id=1641705, opponent_team_id=1610612755,
            line=25.5, over_odds=1.83, under_odds=1.87,
            is_home=1, fetcher=mock_fetcher,
        )
        for col in MODEL_FEATURE_COLS:
            assert col in df.columns, f"Brak kolumny: {col}"

    def test_no_nan_values(self, mock_fetcher):
        """Feature vector nie może zawierać NaN — XGBoost to obsłuży, ale sygnalizuje problem."""
        df = build_feature_vector(
            player_id=1641705, opponent_team_id=1610612755,
            line=25.5, over_odds=1.83, under_odds=1.87,
            is_home=0, fetcher=mock_fetcher,
        )
        model_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
        assert df[model_cols].isna().sum().sum() == 0, \
            f"NaN w kolumnach: {df[model_cols].isna().sum()[df[model_cols].isna().sum() > 0].to_dict()}"

    def test_is_home_propagated(self, mock_fetcher):
        df = build_feature_vector(
            player_id=1641705, opponent_team_id=1610612755,
            line=25.5, over_odds=1.83, under_odds=1.87,
            is_home=1, fetcher=mock_fetcher,
        )
        assert int(df["is_home"].iloc[0]) == 1

    def test_calls_all_fetcher_methods(self, mock_fetcher):
        build_feature_vector(
            player_id=1641705, opponent_team_id=1610612755,
            line=25.5, over_odds=1.83, under_odds=1.87,
            is_home=1, fetcher=mock_fetcher,
        )
        mock_fetcher.get_player_recent_stats.assert_called_once_with(1641705, last_n=20)
        mock_fetcher.get_player_vs_opponent.assert_called_once_with(1641705, 1610612755, last_n=10)
        mock_fetcher.get_team_defensive_stats.assert_called_once_with(1610612755)

    def test_empty_fetcher_data_no_crash(self):
        """Jeśli fetcher zwraca puste dane — brak exception, sensowne defaulty."""
        empty_fetcher = MagicMock()
        empty_fetcher.get_player_recent_stats.return_value  = pd.DataFrame()
        empty_fetcher.get_player_vs_opponent.return_value   = pd.DataFrame()
        empty_fetcher.get_team_defensive_stats.return_value = {}

        df = build_feature_vector(
            player_id=9999, opponent_team_id=9999,
            line=20.0, over_odds=1.80, under_odds=1.80,
            is_home=0, fetcher=empty_fetcher,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
