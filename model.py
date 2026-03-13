# nba_betting/model.py
import json
import logging
import os
import sqlite3
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
from xgboost import XGBClassifier, XGBRegressor

from config import (
    DB_PATH, MODEL_DIR, ODDS_MIN, ODDS_MAX,
    CONFIDENCE_THRESHOLD, STAKE_PER_PICK,
)
from features import MODEL_FEATURE_COLS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Optuna — wyłącz domyślny verbose logging ──────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — wersjonowanie modeli
# ─────────────────────────────────────────────────────────────────────────────
def _next_model_version(model_dir: str) -> int:
    """Zwraca kolejny numer wersji na podstawie plików w katalogu models/."""
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.glob("regressor_v*.pkl"))
    if not existing:
        return 1
    versions = [int(f.stem.split("_v")[1]) for f in existing]
    return max(versions) + 1


def _model_paths(version: int, model_dir: str) -> tuple[str, str]:
    return (
        os.path.join(model_dir, f"regressor_v{version}.pkl"),
        os.path.join(model_dir, f"classifier_v{version}.pkl"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective functions
# ─────────────────────────────────────────────────────────────────────────────
def _regressor_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Minimalizuje MAE regressora przez TimeSeriesSplit CV.
    TimeSeriesSplit zamiast random CV — dane mają strukturę czasową.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state":      42,
        "verbosity":         0,
    }
    model = XGBRegressor(**params)
    tscv  = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    return -scores.mean()   # minimalizujemy MAE


def _classifier_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Maksymalizuje ROC-AUC klasyfikatora przez TimeSeriesSplit CV.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        "random_state":      42,
        "verbosity":         0,
    }
    model  = XGBClassifier(**params)
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
    return scores.mean()    # maksymalizujemy ROC-AUC


# ─────────────────────────────────────────────────────────────────────────────
# PropPredictor — główna klasa Modułu 4
# ─────────────────────────────────────────────────────────────────────────────
class PropPredictor:
    """
    Dwa modele:
      Model A — XGBRegressor  → pts_predicted (ile pkt zdobędzie gracz)
      Model B — XGBClassifier → p_over (prawdopodobieństwo przekroczenia linii)

    Trening przez Optuna (Bayesian search).
    SHAP do interpretacji ważności cech.
    Generowanie kuponu z confidence_score.
    """

    def __init__(
        self,
        db_path:   str = DB_PATH,
        model_dir: str = MODEL_DIR,
    ):
        self.db_path   = db_path
        self.model_dir = model_dir
        self.regressor:  Optional[XGBRegressor]  = None
        self.classifier: Optional[XGBClassifier] = None
        self.version = 0
        self._init_db()
        logger.info("PropPredictor initialized (model_dir=%s)", model_dir)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS model_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    version         INTEGER,
                    trained_at      TEXT,
                    n_samples       INTEGER,
                    regressor_mae   REAL,
                    classifier_auc  REAL,
                    best_reg_params TEXT,
                    best_clf_params TEXT
                );

                CREATE TABLE IF NOT EXISTS coupons (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        TEXT,
                    coupon_json TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                );
            """)
        logger.debug("PropPredictor DB initialized")

    # ── Dane treningowe ───────────────────────────────────────────────────────
    @staticmethod
    def prepare_training_data(
        df: pd.DataFrame,
        line_col: str = "line",
        pts_col:  str = "actual_pts",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Przygotowuje dane do treningu z historycznego DataFrame.

        Wymaga kolumn: MODEL_FEATURE_COLS + actual_pts + line
        Zwraca: X (features), y_reg (punkty), y_clf (over=1/under=0)

        Użycie:
            X, y_reg, y_clf = PropPredictor.prepare_training_data(historical_df)
            predictor.train(X, y_reg, y_clf)
        """
        missing = [c for c in MODEL_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Brak kolumn w danych: {missing}")

        X     = df[MODEL_FEATURE_COLS].copy()
        y_reg = df[pts_col].astype(float)
        y_clf = (df[pts_col] > df[line_col]).astype(int)

        logger.info(
            "Dane treningowe: %d próbek | over_rate=%.1f%%",
            len(X), y_clf.mean() * 100,
        )
        return X, y_reg, y_clf

    # ── Trening ───────────────────────────────────────────────────────────────
    def train(
        self,
        X:        pd.DataFrame,
        y_reg:    pd.Series,
        y_clf:    pd.Series,
        n_trials: int = 50,
    ) -> dict:
        """
        Trenuje oba modele z Optuna hyperparameter tuning.

        n_trials: liczba prób Optuna (więcej = lepiej, ale wolniej)
                  prod: 100+, dev/test: 20-30

        Zwraca słownik z metrykami i najlepszymi parametrami.
        """
        self.version = _next_model_version(self.model_dir)
        logger.info(
            "START treningu v%d | próbek=%d | n_trials=%d",
            self.version, len(X), n_trials,
        )

        # ── Model A: Regressor ────────────────────────────────────────────────
        logger.info("Optuna tuning: XGBRegressor (%d trials)...", n_trials)
        reg_study = optuna.create_study(
            direction="minimize",
            study_name=f"regressor_v{self.version}",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        reg_study.optimize(
            lambda trial: _regressor_objective(trial, X, y_reg),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_reg_params = {**reg_study.best_params, "random_state": 42, "verbosity": 0}
        self.regressor = XGBRegressor(**best_reg_params)
        self.regressor.fit(X, y_reg)

        # Metryki regressora (in-sample MAE — train CV był main metric)
        y_reg_pred   = self.regressor.predict(X)
        regressor_mae = float(mean_absolute_error(y_reg, y_reg_pred))
        cv_mae       = reg_study.best_value

        logger.info(
            "Regressor v%d: CV_MAE=%.2f  InSample_MAE=%.2f  best_params=%s",
            self.version, cv_mae, regressor_mae,
            {k: round(v, 4) if isinstance(v, float) else v
             for k, v in best_reg_params.items()},
        )

        # ── Model B: Classifier ───────────────────────────────────────────────
        logger.info("Optuna tuning: XGBClassifier (%d trials)...", n_trials)
        clf_study = optuna.create_study(
            direction="maximize",
            study_name=f"classifier_v{self.version}",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        clf_study.optimize(
            lambda trial: _classifier_objective(trial, X, y_clf),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_clf_params = {**clf_study.best_params, "random_state": 42, "verbosity": 0}
        self.classifier = XGBClassifier(**best_clf_params)
        self.classifier.fit(X, y_clf)

        # Metryki klasyfikatora
        y_clf_proba  = self.classifier.predict_proba(X)[:, 1]
        classifier_auc = float(roc_auc_score(y_clf, y_clf_proba))
        cv_auc         = clf_study.best_value

        logger.info(
            "Classifier v%d: CV_AUC=%.3f  InSample_AUC=%.3f",
            self.version, cv_auc, classifier_auc,
        )

        # ── Zapisz modele ─────────────────────────────────────────────────────
        reg_path, clf_path = _model_paths(self.version, self.model_dir)
        joblib.dump(self.regressor,  reg_path)
        joblib.dump(self.classifier, clf_path)
        logger.info("Modele zapisane: %s | %s", reg_path, clf_path)

        # ── Zapisz historię metryk ────────────────────────────────────────────
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO model_history
                    (version, trained_at, n_samples,
                     regressor_mae, classifier_auc,
                     best_reg_params, best_clf_params)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.version,
                    datetime.now().isoformat(),
                    len(X),
                    round(cv_mae, 4),
                    round(cv_auc, 4),
                    json.dumps(best_reg_params),
                    json.dumps(best_clf_params),
                ),
            )

        metrics = {
            "version":        self.version,
            "n_samples":      len(X),
            "cv_mae":         round(cv_mae,         3),
            "cv_auc":         round(cv_auc,         3),
            "insample_mae":   round(regressor_mae,  3),
            "insample_auc":   round(classifier_auc, 3),
            "reg_best_params": best_reg_params,
            "clf_best_params": best_clf_params,
        }
        return metrics

    # ── SHAP ──────────────────────────────────────────────────────────────────
    def explain(
        self,
        X: pd.DataFrame,
        model_type: str = "classifier",
        max_display: int = 10,
    ) -> pd.DataFrame:
        """
        Wyświetla SHAP summary plot i zwraca DataFrame z mean |SHAP| per feature.

        model_type: 'classifier' lub 'regressor'

        Użycie:
            shap_df = predictor.explain(X_train)
            print(shap_df.head(10))
        """
        model = self.classifier if model_type == "classifier" else self.regressor
        if model is None:
            raise RuntimeError(f"Model '{model_type}' nie jest wytrenowany. Uruchom train() najpierw.")

        logger.info("Obliczam SHAP values dla %s (%d próbek)...", model_type, len(X))
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Dla klasyfikatora shap_values to 1D array (prawdopodobieństwo klasy 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Summary plot
        print(f"\n{'='*60}")
        print(f"SHAP Feature Importance — {model_type.upper()}")
        print(f"{'='*60}")
        shap.summary_plot(
            shap_values, X,
            plot_type="bar",
            max_display=max_display,
            show=True,
        )

        # Zwróć jako DataFrame
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_shap})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return shap_df

    def get_top_features(
        self,
        features_df: pd.DataFrame,
        n: int = 3,
    ) -> list[str]:
        """
        Zwraca n najważniejszych cech dla konkretnej predykcji (SHAP per-sample).
        Używane do pola 'top_features' w kuponie.
        """
        if self.classifier is None:
            return list(features_df.columns[:n])

        X = features_df[MODEL_FEATURE_COLS]
        explainer   = shap.TreeExplainer(self.classifier)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        importance = np.abs(shap_values[0])
        top_idx    = importance.argsort()[::-1][:n]
        return [MODEL_FEATURE_COLS[i] for i in top_idx]

    # ── Predykcja ─────────────────────────────────────────────────────────────
    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Generuje predykcję dla jednego gracza.

        Wymaga: features_df z kolumnami MODEL_FEATURE_COLS

        Zwraca:
            {
                pts_predicted:    float,  # przewidywana liczba punktów
                p_over:           float,  # P(over) od klasyfikatora
                confidence_score: int,    # 0–100, kombinacja 4 sygnałów
            }
        """
        if self.regressor is None or self.classifier is None:
            raise RuntimeError("Modele nie są wytrenowane. Uruchom train() lub load().")

        X = features_df[MODEL_FEATURE_COLS]

        pts_predicted = float(self.regressor.predict(X)[0])
        p_over        = float(self.classifier.predict_proba(X)[0, 1])

        confidence_score = self._compute_confidence(features_df, pts_predicted, p_over)

        logger.info(
            "Predykcja: pts_predicted=%.1f p_over=%.3f confidence=%d",
            pts_predicted, p_over, confidence_score,
        )
        return {
            "pts_predicted":    round(pts_predicted, 1),
            "p_over":           round(p_over,        3),
            "confidence_score": confidence_score,
        }

    def _compute_confidence(
        self,
        features_df: pd.DataFrame,
        pts_predicted: float,
        p_over: float,
    ) -> int:
        """
        Oblicza confidence_score 0–100 jako ważoną sumę 4 sygnałów:

        1. MODEL (40%)     — odległość p_over od 0.5 (im dalej, tym pewniej)
        2. FORMA (20%)     — czy pts_trend idzie w stronę predykcji
        3. HISTORICAL (20%)— czy pct_over_line_historical zgadza się z p_over
        4. MARKET (20%)    — czy implied_prob rynku potwierdza predykcję

        Skala: 0–100 (int)
        """
        row = features_df.iloc[0]

        # 1. Sygnał modelu (p_over odległość od 0.5)
        model_signal = min(abs(p_over - 0.5) * 200, 100)  # 0.5→0, 1.0→100

        # 2. Sygnał formy (trend w dobrą stronę?)
        trend       = float(row.get("pts_trend",       0))
        line        = float(row.get("line",             0))
        pts_avg_10  = float(row.get("pts_avg_10",       0))
        predicting_over = p_over > 0.5
        if predicting_over:
            form_signal = min(max((pts_avg_10 - line) / max(line, 1) * 200 + 50, 0), 100)
        else:
            form_signal = min(max((line - pts_avg_10) / max(line, 1) * 200 + 50, 0), 100)

        # 3. Sygnał historyczny
        pct_hist = float(row.get("pct_over_line_historical", 0.5))
        if predicting_over:
            hist_signal = pct_hist * 100
        else:
            hist_signal = (1 - pct_hist) * 100

        # 4. Sygnał rynkowy (market implied prob zgadza się z naszą predykcją?)
        market_p = float(row.get("over_implied_prob", 0.5))
        if predicting_over:
            market_signal = market_p * 100
        else:
            market_signal = (1 - market_p) * 100

        # Ważona suma
        confidence = (
            model_signal   * 0.40 +
            form_signal    * 0.20 +
            hist_signal    * 0.20 +
            market_signal  * 0.20
        )

        return int(min(max(confidence, 0), 100))

    # ── Ładowanie modeli ──────────────────────────────────────────────────────
    def load(self, version: Optional[int] = None) -> int:
        """
        Ładuje modele z dysku.
        Jeśli version=None, ładuje najnowszą wersję.
        Zwraca załadowaną wersję.
        """
        if version is None:
            version = _next_model_version(self.model_dir) - 1
        if version < 1:
            raise FileNotFoundError("Brak zapisanych modeli. Uruchom train() najpierw.")

        reg_path, clf_path = _model_paths(version, self.model_dir)
        self.regressor  = joblib.load(reg_path)
        self.classifier = joblib.load(clf_path)
        self.version    = version
        logger.info("Załadowano modele v%d z %s", version, self.model_dir)
        return version

    # ── Generowanie kuponu ────────────────────────────────────────────────────
    def generate_coupon(
        self,
        props: list[dict],
        game_date: Optional[str] = None,
        stake: float = STAKE_PER_PICK,
        min_confidence: int = CONFIDENCE_THRESHOLD,
    ) -> dict:
        """
        Generuje kupon na podstawie listy propsów z feature vectorami.

        Każdy prop w liście musi zawierać:
            {
                player_name, game, line, over_odds, under_odds,
                features_df: pd.DataFrame,  # z build_feature_vector()
                player_position: str,
            }

        Filtrowanie:
            - confidence_score > min_confidence
            - kurs 1.40–1.90 dla wybranej strony (over/under)
            - max 1 pick per mecz (wybieramy najwyższy confidence)

        Zwraca dict kuponu gotowy do zapisu/eksportu JSON.
        """
        if game_date is None:
            game_date = date.today().isoformat()

        picks = []
        for prop in props:
            fv           = prop["features_df"]
            player_name  = prop["player_name"]
            game         = prop.get("game", "?")
            line         = float(prop["line"])
            over_odds    = float(prop["over_odds"])
            under_odds   = float(prop["under_odds"])

            try:
                prediction = self.predict(fv)
            except Exception as e:
                logger.warning("Predykcja nie powiodła się dla %s: %s", player_name, e)
                continue

            pts_predicted    = prediction["pts_predicted"]
            p_over           = prediction["p_over"]
            confidence_score = prediction["confidence_score"]

            # Wybierz stronę
            bet        = "over" if p_over > 0.5 else "under"
            bet_odds   = over_odds if bet == "over" else under_odds

            # Filtrowanie
            if confidence_score < min_confidence:
                logger.debug(
                    "ODRZUCONO %s: confidence=%d < %d",
                    player_name, confidence_score, min_confidence,
                )
                continue
            if not (ODDS_MIN <= bet_odds <= ODDS_MAX):
                logger.debug(
                    "ODRZUCONO %s: odds=%.2f poza zakresem",
                    player_name, bet_odds,
                )
                continue

            top_features = self.get_top_features(fv, n=3)
            reasoning    = self._build_reasoning(fv, bet, pts_predicted, line)

            picks.append({
                "game":          game,
                "player":        player_name,
                "bet":           bet,
                "line":          line,
                "odds":          bet_odds,
                "pts_predicted": pts_predicted,
                "p_over":        round(p_over, 3),
                "confidence":    confidence_score,
                "top_features":  top_features,
                "reasoning":     reasoning,
            })

        # Max 1 pick per mecz (najwyższy confidence)
        picks = self._deduplicate_by_game(picks)
        picks.sort(key=lambda x: x["confidence"], reverse=True)

        coupon = {
            "date":                    game_date,
            "model_version":           self.version,
            "picks":                   picks,
            "total_picks":             len(picks),
            "simulated_stake_per_pick": stake,
            "simulated_total_stake":   round(stake * len(picks), 2),
        }

        # Zapisz do SQLite
        self._save_coupon(coupon, game_date)

        # Eksport JSON
        json_path = f"coupon_{game_date}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coupon, f, indent=2, ensure_ascii=False)
        logger.info(
            "Kupon zapisany: %d picków | stake=%.1f | %s",
            len(picks), stake * len(picks), json_path,
        )
        return coupon

    @staticmethod
    def _deduplicate_by_game(picks: list[dict]) -> list[dict]:
        """Zostawia najwyższy confidence per mecz."""
        best: dict[str, dict] = {}
        for pick in picks:
            game = pick["game"]
            if game not in best or pick["confidence"] > best[game]["confidence"]:
                best[game] = pick
        return list(best.values())

    @staticmethod
    def _build_reasoning(
        fv: pd.DataFrame,
        bet: str,
        pts_predicted: float,
        line: float,
    ) -> str:
        """Generuje krótki opis logiki pickowego."""
        row        = fv.iloc[0]
        avg_10     = round(float(row.get("pts_avg_10", 0)), 1)
        trend      = round(float(row.get("pts_trend", 0)),  1)
        def_rating = round(float(row.get("opp_def_rating", 0)), 1)
        market_p   = round(float(row.get("over_implied_prob", 0.5)), 3)

        trend_str  = f"trend {'↑' if trend > 0 else '↓'}{abs(trend)}"
        market_str = f"market {'confirms' if (market_p > 0.5) == (bet == 'over') else 'disagrees'}"

        return (
            f"avg {avg_10} last 10, {trend_str}, "
            f"opp DEF_RTG={def_rating}, "
            f"model={pts_predicted}pts, {market_str}"
        )

    def _save_coupon(self, coupon: dict, game_date: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO coupons (date, coupon_json) VALUES (?, ?)",
                (game_date, json.dumps(coupon)),
            )
        logger.debug("Kupon zapisany do SQLite dla daty %s", game_date)

    # ── Historia metryk modelu ────────────────────────────────────────────────
    def get_model_history(self) -> pd.DataFrame:
        """Zwraca historię metryk wszystkich wersji modelu."""
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT version, trained_at, n_samples, "
                "regressor_mae, classifier_auc "
                "FROM model_history ORDER BY version",
                conn,
            )
        return df
