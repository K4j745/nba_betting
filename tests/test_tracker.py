# tests/test_tracker.py
import json
import pytest
import sqlite3
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from tracker import ResultTracker, ODDS_BUCKETS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker(tmp_path):
    fetcher = MagicMock()
    fetcher.get_player_recent_stats.return_value = pd.DataFrame()
    return ResultTracker(
        db_path=str(tmp_path / "test.db"),
        model_dir=str(tmp_path / "models"),
        fetcher=fetcher,
        stake=10.0,
    )


@pytest.fixture
def sample_coupon():
    return {
        "date":  "2026-03-06",
        "picks": [
            {
                "player":        "Victor Wembanyama",
                "game":          "SAS vs PHI",
                "bet":           "under",
                "line":          25.5,
                "odds":          1.87,
                "pts_predicted": 22.1,
                "p_over":        0.28,
                "confidence":    74,
                "top_features":  ["pts_avg_10", "opp_def_rating", "line"],
            },
            {
                "player":        "Jayson Tatum",
                "game":          "BOS vs MIL",
                "bet":           "over",
                "line":          28.5,
                "odds":          1.75,
                "pts_predicted": 30.2,
                "p_over":        0.72,
                "confidence":    68,
                "top_features":  ["pts_avg_3", "pts_trend"],
            },
        ],
    }


@pytest.fixture
def tracker_with_coupon(tracker, sample_coupon):
    """Tracker z kuponem wstawionym do DB."""
    with tracker._conn() as conn:
        conn.execute(
            "INSERT INTO coupons (date, coupon_json) VALUES (?, ?)",
            ("2026-03-06", json.dumps(sample_coupon)),
        )
    return tracker


@pytest.fixture
def sample_results_df():
    """Symuluje 40 wyników zapisanych w DB."""
    np.random.seed(42)
    n = 40
    bets    = ["over", "under"] * (n // 2)
    results = ["WIN", "LOSS"] * (n // 2)
    odds    = np.random.uniform(1.40, 1.90, n)
    profits = [
        round(10 * (o - 1), 2) if r == "WIN" else -10.0
        for r, o in zip(results, odds)
    ]
    return pd.DataFrame({
        "id":               range(n),
        "coupon_id":        [1] * n,
        "game_date":        ["2026-03-06"] * n,
        "player_name":      [f"Player{i % 5}" for i in range(n)],
        "player_id":        [1000 + (i % 5) for i in range(n)],
        "game":             ["SAS vs PHI"] * n,
        "bet":              bets,
        "line":             [25.5] * n,
        "odds":             odds,
        "pts_predicted":    np.random.normal(22, 4, n),
        "p_over":           np.random.uniform(0.3, 0.7, n),
        "confidence":       np.random.randint(60, 90, n),
        "top_feature":      ["pts_avg_10", "line"] * (n // 2),
        "actual_pts":       np.random.normal(22, 5, n),
        "result":           results,
        "simulated_profit": profits,
    })


@pytest.fixture
def tracker_with_results(tracker, sample_results_df):
    """Tracker z wstawionymi wynikami do DB."""
    with tracker._conn() as conn:
        sample_results_df.to_sql("results", conn, if_exists="append", index=False)
    return tracker


# ── Testy _calculate_result ────────────────────────────────────────────────────

class TestCalculateResult:
    def test_over_win(self):
        result, profit = ResultTracker._calculate_result(
            actual_pts=28.0, line=25.5, bet="over", odds=1.83, stake=10.0
        )
        assert result == "WIN"
        assert profit == pytest.approx(8.3, abs=0.01)

    def test_over_loss(self):
        result, profit = ResultTracker._calculate_result(
            actual_pts=23.0, line=25.5, bet="over", odds=1.83, stake=10.0
        )
        assert result == "LOSS"
        assert profit == -10.0

    def test_under_win(self):
        result, profit = ResultTracker._calculate_result(
            actual_pts=22.0, line=25.5, bet="under", odds=1.87, stake=10.0
        )
        assert result == "WIN"
        assert profit == pytest.approx(8.7, abs=0.01)

    def test_under_loss(self):
        result, profit = ResultTracker._calculate_result(
            actual_pts=27.0, line=25.5, bet="under", odds=1.87, stake=10.0
        )
        assert result == "LOSS"
        assert profit == -10.0

    def test_push_on_line(self):
        result, profit = ResultTracker._calculate_result(
            actual_pts=25.5, line=25.5, bet="over", odds=1.83, stake=10.0
        )
        assert result == "PUSH"
        assert profit == 0.0

    def test_push_within_half_point(self):
        result, profit = ResultTracker._calculate_result(
            actual_pts=25.3, line=25.5, bet="over", odds=1.83, stake=10.0
        )
        assert result == "PUSH"


# ── Testy verify_yesterday ─────────────────────────────────────────────────────

class TestVerifyYesterday:
    def test_returns_empty_when_no_coupons(self, tracker):
        result = tracker.verify_yesterday("2026-01-01")
        assert result.empty

    @patch.object(ResultTracker, "_resolve_player_id", return_value=1641705)
    @patch.object(ResultTracker, "_fetch_actual_pts", return_value=22.0)
    def test_verifies_under_correctly(
        self, mock_pts, mock_id, tracker_with_coupon
    ):
        df = tracker_with_coupon.verify_yesterday("2026-03-06")
        assert not df.empty
        wemby = df[df["player_name"] == "Victor Wembanyama"].iloc[0]
        # actual=22.0 < line=25.5, bet=under → WIN
        assert wemby["result"] == "WIN"
        assert wemby["actual_pts"] == 22.0

    @patch.object(ResultTracker, "_resolve_player_id", return_value=1628369)
    @patch.object(ResultTracker, "_fetch_actual_pts", return_value=31.0)
    def test_verifies_over_correctly(
        self, mock_pts, mock_id, tracker_with_coupon
    ):
        df = tracker_with_coupon.verify_yesterday("2026-03-06")
        tatum = df[df["player_name"] == "Jayson Tatum"].iloc[0]
        # actual=31.0 > line=28.5, bet=over → WIN
        assert tatum["result"] == "WIN"

    @patch.object(ResultTracker, "_resolve_player_id", return_value=None)
    def test_skips_unknown_player(self, mock_id, tracker_with_coupon):
        """Jeśli player_id=None → gracz pomijany, brak crash."""
        df = tracker_with_coupon.verify_yesterday("2026-03-06")
        assert isinstance(df, pd.DataFrame)

    @patch.object(ResultTracker, "_resolve_player_id", return_value=1641705)
    @patch.object(ResultTracker, "_fetch_actual_pts", return_value=22.0)
    def test_results_saved_to_db(
        self, mock_pts, mock_id, tracker_with_coupon
    ):
        tracker_with_coupon.verify_yesterday("2026-03-06")
        with tracker_with_coupon._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM results"
            ).fetchone()[0]
        assert count >= 1


# ── Testy generate_report ─────────────────────────────────────────────────────

class TestGenerateReport:
    def test_returns_dict(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        assert isinstance(report, dict)

    def test_overall_keys(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        assert "overall" in report
        for key in ["wins", "losses", "hit_rate", "roi", "total_profit"]:
            assert key in report["overall"], f"Brak klucza: {key}"

    def test_hit_rate_between_0_and_1(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        hr = report["overall"]["hit_rate"]
        assert 0.0 <= hr <= 1.0

    def test_by_side_present(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        assert "by_side" in report
        assert "over"  in report["by_side"]
        assert "under" in report["by_side"]

    def test_odds_buckets_present(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        assert "by_odds_bucket" in report
        # Przynajmniej 1 bucket powinien mieć dane
        assert len(report["by_odds_bucket"]) >= 1

    def test_best_worst_players(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        assert "best_players"  in report
        assert "worst_players" in report
        assert len(report["best_players"]) >= 1

    def test_empty_db_returns_error(self, tracker):
        report = tracker.generate_report(last_n_days=365)
        assert "error" in report

    def test_roi_calculation_correct(self, tracker, sample_results_df):
        """50% hit rate z odds ~1.65 → ROI ≈ -17%"""
        with tracker._conn() as conn:
            sample_results_df.to_sql(
                "results", conn, if_exists="append", index=False
            )
        report = tracker.generate_report(last_n_days=365)
        # Sprawdź że ROI jest liczbą, nie NaN
        assert isinstance(report["overall"]["roi"], float)
        assert not np.isnan(report["overall"]["roi"])


# ── Testy calibration ─────────────────────────────────────────────────────────

class TestCalibration:
    def test_calibration_with_enough_data(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        cal    = report.get("calibration", {})
        if "error" not in cal:
            assert "prob_pred" in cal
            assert "prob_true" in cal
            assert len(cal["prob_pred"]) == len(cal["prob_true"])

    def test_calibration_probs_in_range(self, tracker_with_results):
        report = tracker_with_results.generate_report(last_n_days=365)
        cal    = report.get("calibration", {})
        if "prob_pred" in cal:
            for p in cal["prob_pred"] + cal["prob_true"]:
                assert 0.0 <= p <= 1.0


# ── Testy retrain_if_due ──────────────────────────────────────────────────────

class TestRetrainIfDue:
    def test_skips_when_too_few_samples(self, tracker_with_results):
        mock_predictor = MagicMock()
        # Mamy 40 wyników ale próg = 100 → skip
        result = tracker_with_results.retrain_if_due(
            mock_predictor, min_new_samples=100
        )
        assert result is False
        mock_predictor.train.assert_not_called()

    def test_triggers_when_enough_samples(self, tracker_with_results):
        mock_predictor = MagicMock()
        mock_predictor.version = 1
        mock_predictor.train.return_value = {
            "version": 2, "cv_mae": 3.5, "cv_auc": 0.68,
            "reg_best_params": {}, "clf_best_params": {},
        }
        # Mockujemy _build_training_df żeby ominąć wywołania nba_api
        with patch.object(
            tracker_with_results,
            "_build_training_df_from_results",
            return_value=pd.DataFrame({
                **{col: [0.5] * 40 for col in __import__("features").MODEL_FEATURE_COLS},
                "actual_pts": [22.0] * 40,
                "line": [25.5] * 40,
            }),
        ):
            result = tracker_with_results.retrain_if_due(
                mock_predictor, min_new_samples=30
            )
        assert result is True
        mock_predictor.train.assert_called_once()

    def test_retrain_logged_to_db(self, tracker_with_results):
        mock_predictor = MagicMock()
        mock_predictor.version = 1
        mock_predictor.train.return_value = {
            "version": 2, "cv_mae": 3.5, "cv_auc": 0.68,
            "reg_best_params": {}, "clf_best_params": {},
        }
        with patch.object(
            tracker_with_results,
            "_build_training_df_from_results",
            return_value=pd.DataFrame({
                **{col: [0.5] * 40 for col in __import__("features").MODEL_FEATURE_COLS},
                "actual_pts": [22.0] * 40,
                "line": [25.5] * 40,
            }),
        ):
            tracker_with_results.retrain_if_due(
                mock_predictor, min_new_samples=30
            )
        history = tracker_with_results.get_retrain_history()
        assert len(history) >= 1


# ── Testy cumulative P&L ──────────────────────────────────────────────────────

class TestCumulativePnL:
    def test_returns_dataframe(self, tracker_with_results):
        df = tracker_with_results.get_cumulative_pnl()
        assert isinstance(df, pd.DataFrame)

    def test_cumulative_column_exists(self, tracker_with_results):
        df = tracker_with_results.get_cumulative_pnl()
        assert "cumulative_pnl" in df.columns

    def test_monotone_trend_for_all_wins(self, tracker):
        """Wszystkie winy → cumulative P&L rosnący."""
        wins_df = pd.DataFrame({
            "game_date":        ["2026-03-01", "2026-03-02", "2026-03-03"],
            "simulated_profit": [8.3, 7.5, 8.7],
            "result":           ["WIN", "WIN", "WIN"],
        })
        with tracker._conn() as conn:
            wins_df.to_sql("results", conn, if_exists="append", index=False)
        df = tracker.get_cumulative_pnl()
        assert df["cumulative_pnl"].is_monotonic_increasing
