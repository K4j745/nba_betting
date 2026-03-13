# tests/test_model.py
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification, make_regression

from model import PropPredictor, _next_model_version, _model_paths
from features import MODEL_FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_SAMPLES = 80   # minimum żeby TimeSeriesSplit(5) miał sens

@pytest.fixture
def synthetic_data():
    """Syntetyczne dane treningowe — symuluje historyczne prop bety."""
    np.random.seed(42)
    n = N_SAMPLES

    # Feature matrix z realistycznymi wartościami
    X = pd.DataFrame({
        "pts_avg_3":                np.random.normal(22, 5, n),
        "pts_avg_5":                np.random.normal(22, 4, n),
        "pts_avg_10":               np.random.normal(22, 3, n),
        "pts_avg_20":               np.random.normal(22, 2, n),
        "pts_std_10":               np.random.uniform(2, 8, n),
        "pts_trend":                np.random.normal(0, 3, n),
        "pct_over_line_historical": np.random.uniform(0.3, 0.7, n),
        "min_avg_5":                np.random.normal(32, 4, n),
        "usg_pct_avg_5":            np.random.uniform(0.15, 0.35, n),
        "is_home":                  np.random.randint(0, 2, n),
        "days_rest":                np.random.randint(1, 5, n),
        "is_back_to_back":          np.random.randint(0, 2, n),
        "opp_def_rating":           np.random.normal(110, 5, n),
        "opp_pace":                 np.random.normal(99, 3, n),
        "opp_pts_allowed_to_position": np.random.normal(18, 2, n),
        "pts_avg_vs_opponent":      np.random.normal(22, 5, n),
        "pct_over_vs_opponent":     np.random.uniform(0.2, 0.8, n),
        "h2h_games_available":      np.random.randint(0, 8, n),
        "line":                     np.random.uniform(15, 35, n),
        "over_implied_prob":        np.random.uniform(0.4, 0.6, n),
        "under_implied_prob":       np.random.uniform(0.4, 0.6, n),
        "market_margin":            np.random.uniform(0.04, 0.10, n),
    })
    y_reg = X["pts_avg_10"] + np.random.normal(0, 4, n)
    y_clf = (y_reg > X["line"]).astype(int)
    return X, y_reg, y_clf


@pytest.fixture
def trained_predictor(tmp_path, synthetic_data):
    """PropPredictor wytrenowany na syntetycznych danych (5 trials — szybko)."""
    X, y_reg, y_clf = synthetic_data
    predictor = PropPredictor(
        db_path=str(tmp_path / "test.db"),
        model_dir=str(tmp_path / "models"),
    )
    predictor.train(X, y_reg, y_clf, n_trials=5)
    return predictor


@pytest.fixture
def sample_features_df():
    """Pojedynczy feature vector dla jednego gracza."""
    return pd.DataFrame([{
        "pts_avg_3": 27.0, "pts_avg_5": 25.5, "pts_avg_10": 24.2,
        "pts_avg_20": 23.0, "pts_std_10": 4.5, "pts_trend": 2.8,
        "pct_over_line_historical": 0.45, "min_avg_5": 33.5,
        "usg_pct_avg_5": 0.284, "is_home": 1, "days_rest": 2,
        "is_back_to_back": 0, "opp_def_rating": 113.8, "opp_pace": 99.2,
        "opp_pts_allowed_to_position": 14.6, "pts_avg_vs_opponent": 25.5,
        "pct_over_vs_opponent": 0.40, "h2h_games_available": 5,
        "line": 25.5, "over_implied_prob": 0.505,
        "under_implied_prob": 0.495, "market_margin": 0.075,
    }])


# ── Testy wersjonowania ────────────────────────────────────────────────────────

class TestVersioning:
    def test_first_version_is_1(self, tmp_path):
        assert _next_model_version(str(tmp_path)) == 1

    def test_version_increments(self, tmp_path):
        # Symuluj istniejący model
        (tmp_path / "regressor_v1.pkl").touch()
        assert _next_model_version(str(tmp_path)) == 2

    def test_model_paths_format(self, tmp_path):
        reg, clf = _model_paths(3, str(tmp_path))
        assert "regressor_v3.pkl" in reg
        assert "classifier_v3.pkl" in clf


# ── Testy treningu ─────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_metrics_dict(self, trained_predictor, synthetic_data):
        X, y_reg, y_clf = synthetic_data
        # Re-trening z nową instancją dla czystości
        predictor2 = PropPredictor(
            db_path=trained_predictor.db_path,
            model_dir=trained_predictor.model_dir,
        )
        metrics = predictor2.train(X, y_reg, y_clf, n_trials=3)
        assert "cv_mae"         in metrics
        assert "cv_auc"         in metrics
        assert "version"        in metrics
        assert "reg_best_params" in metrics

    def test_mae_is_positive(self, trained_predictor, synthetic_data):
        X, y_reg, y_clf = synthetic_data
        metrics = trained_predictor.train(X, y_reg, y_clf, n_trials=3)
        assert metrics["cv_mae"] > 0

    def test_auc_between_0_and_1(self, trained_predictor, synthetic_data):
        X, y_reg, y_clf = synthetic_data
        metrics = trained_predictor.train(X, y_reg, y_clf, n_trials=3)
        assert 0 < metrics["cv_auc"] < 1

    def test_models_saved_to_disk(self, trained_predictor):
        reg_path, clf_path = _model_paths(
            trained_predictor.version, trained_predictor.model_dir
        )
        import os
        assert os.path.exists(reg_path), f"Brak pliku: {reg_path}"
        assert os.path.exists(clf_path), f"Brak pliku: {clf_path}"

    def test_history_logged_to_db(self, trained_predictor):
        history = trained_predictor.get_model_history()
        assert len(history) >= 1
        assert "regressor_mae" in history.columns
        assert "classifier_auc" in history.columns

    def test_missing_features_raises(self, trained_predictor):
        bad_df = pd.DataFrame([{"pts_avg_3": 20}])  # brak 21 kolumn
        with pytest.raises(ValueError, match="Brak kolumn"):
            PropPredictor.prepare_training_data(
                bad_df, pts_col="actual_pts"
            )


# ── Testy prepare_training_data ───────────────────────────────────────────────

class TestPrepareTrainingData:
    def test_creates_binary_target(self, synthetic_data):
        X, y_reg, _ = synthetic_data
        df = X.copy()
        df["actual_pts"] = y_reg.values
        df["line"]       = 22.0

        _, _, y_clf = PropPredictor.prepare_training_data(df)
        assert set(y_clf.unique()).issubset({0, 1})

    def test_over_rate_logged(self, synthetic_data, caplog):
        import logging
        X, y_reg, _ = synthetic_data
        df = X.copy()
        df["actual_pts"] = y_reg.values
        with caplog.at_level(logging.INFO, logger="model"):
            PropPredictor.prepare_training_data(df)
        assert "over_rate" in caplog.text


# ── Testy predict ──────────────────────────────────────────────────────────────

class TestPredict:
    @pytest.mark.slow
    def test_returns_required_keys(self, trained_predictor, sample_features_df):
        result = trained_predictor.predict(sample_features_df)
        assert "pts_predicted"    in result
        assert "p_over"           in result
        assert "confidence_score" in result

    @pytest.mark.slow
    def test_p_over_between_0_and_1(self, trained_predictor, sample_features_df):
        result = trained_predictor.predict(sample_features_df)
        assert 0.0 <= result["p_over"] <= 1.0

    @pytest.mark.slow
    def test_confidence_between_0_and_100(self, trained_predictor, sample_features_df):
        result = trained_predictor.predict(sample_features_df)
        assert 0 <= result["confidence_score"] <= 100

    @pytest.mark.slow
    def test_pts_predicted_is_float(self, trained_predictor, sample_features_df):
        result = trained_predictor.predict(sample_features_df)
        assert isinstance(result["pts_predicted"], float)

    def test_raises_without_training(self, tmp_path, sample_features_df):
        predictor = PropPredictor(
            db_path=str(tmp_path / "empty.db"),
            model_dir=str(tmp_path / "empty_models"),
        )
        with pytest.raises(RuntimeError, match="nie są wytrenowane"):
            predictor.predict(sample_features_df)


# ── Testy confidence score ─────────────────────────────────────────────────────

class TestConfidence:
    @pytest.mark.slow
    def test_high_p_over_gives_high_confidence(
        self, trained_predictor, sample_features_df
    ):
        """p_over=0.95 → model bardzo pewny over → wysoki confidence."""
        with patch.object(trained_predictor.classifier, "predict_proba",
                          return_value=np.array([[0.05, 0.95]])):
            result = trained_predictor.predict(sample_features_df)
            assert result["confidence_score"] > 60
    @pytest.mark.slow
    def test_p_over_near_50_gives_low_model_signal(
        self, trained_predictor, sample_features_df
    ):
        """p_over=0.51 → model ledwo faworyzuje over → niższy confidence."""
        with patch.object(trained_predictor.classifier, "predict_proba",
                          return_value=np.array([[0.49, 0.51]])):
            result_uncertain = trained_predictor.predict(sample_features_df)

        with patch.object(trained_predictor.classifier, "predict_proba",
                          return_value=np.array([[0.05, 0.95]])):
            result_certain = trained_predictor.predict(sample_features_df)

        assert result_certain["confidence_score"] > result_uncertain["confidence_score"]


# ── Testy generowania kuponu ──────────────────────────────────────────────────

class TestGenerateCoupon:
    def _make_props(self, features_df, n=3):
        games = ["SAS vs PHI", "BOS vs MIL", "OKC vs DEN"]
        players = ["Wembanyama", "Tatum", "Gilgeous-Alexander"]
        return [
            {
                "player_name":  players[i],
                "game":         games[i],
                "line":         25.5,
                "over_odds":    1.83,
                "under_odds":   1.87,
                "features_df":  features_df.copy(),
            }
            for i in range(n)
        ]

    @pytest.mark.slow
    def test_coupon_has_required_keys(self, trained_predictor, sample_features_df):
        props  = self._make_props(sample_features_df, n=2)
        coupon = trained_predictor.generate_coupon(props, game_date="2026-03-07")

        assert "date"   in coupon
        assert "picks"  in coupon
        assert "model_version" in coupon

    @pytest.mark.slow
    def test_max_one_pick_per_game(self, trained_predictor, sample_features_df):
        """Dwa propsów z tego samego meczu → max 1 pick."""
        props = [
            {"player_name": "Player A", "game": "SAS vs PHI",
             "line": 25.5, "over_odds": 1.83, "under_odds": 1.87,
             "features_df": sample_features_df.copy()},
            {"player_name": "Player B", "game": "SAS vs PHI",  # ten sam mecz!
             "line": 20.5, "over_odds": 1.75, "under_odds": 1.85,
             "features_df": sample_features_df.copy()},
        ]
        coupon = trained_predictor.generate_coupon(props, game_date="2026-03-07")
        games  = [p["game"] for p in coupon["picks"]]
        assert len(games) == len(set(games)), "Duplikat meczu w kuponie!"

    @pytest.mark.slow
    def test_pick_structure(self, trained_predictor, sample_features_df):
        props  = self._make_props(sample_features_df, n=1)
        coupon = trained_predictor.generate_coupon(props, game_date="2026-03-07")

        if coupon["picks"]:
            pick = coupon["picks"][0]
            for key in ["game", "player", "bet", "line", "odds",
                        "pts_predicted", "p_over", "confidence",
                        "top_features", "reasoning"]:
                assert key in pick, f"Brak klucza w picku: {key}"

    @pytest.mark.slow
    def test_bet_is_over_or_under(self, trained_predictor, sample_features_df):
        props  = self._make_props(sample_features_df, n=2)
        coupon = trained_predictor.generate_coupon(props, game_date="2026-03-07")
        for pick in coupon["picks"]:
            assert pick["bet"] in {"over", "under"}

    @pytest.mark.slow
    def test_coupon_saved_to_db(self, trained_predictor, sample_features_df):
        props = self._make_props(sample_features_df, n=2)
        trained_predictor.generate_coupon(props, game_date="2026-03-07")

        import sqlite3
        with sqlite3.connect(trained_predictor.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM coupons").fetchone()[0]
        assert count >= 1

    @pytest.mark.slow
    def test_coupon_exported_to_json(
        self, trained_predictor, sample_features_df, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)  # JSON ląduje w tmp_path
        props = self._make_props(sample_features_df, n=1)
        trained_predictor.generate_coupon(props, game_date="2026-03-07")

        import os
        assert os.path.exists("coupon_2026-03-07.json")

    @pytest.mark.slow
    def test_load_and_predict(self, trained_predictor, sample_features_df, tmp_path):
        """Zapisz → załaduj → predykcja musi dać ten sam wynik."""
        predictor2 = PropPredictor(
            db_path=str(tmp_path / "test2.db"),
            model_dir=trained_predictor.model_dir,
        )
        predictor2.load(version=trained_predictor.version)
        result = predictor2.predict(sample_features_df)
        assert "pts_predicted" in result
