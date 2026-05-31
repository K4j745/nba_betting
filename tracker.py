# nba_betting/tracker.py
import json
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from config import DB_PATH, MODEL_DIR, STAKE_PER_PICK
from data_fetcher import NBADataFetcher
from features import MODEL_FEATURE_COLS, build_feature_vector
from model import PropPredictor, _next_model_version

logger = logging.getLogger(__name__)

# ── Progi kursów do bucket analysis ──────────────────────────────────────────
ODDS_BUCKETS = [
    (1.40, 1.60, "1.40–1.60"),
    (1.60, 1.75, "1.60–1.75"),
    (1.75, 1.90, "1.75–1.90"),
]


# ─────────────────────────────────────────────────────────────────────────────
# ResultTracker
# ─────────────────────────────────────────────────────────────────────────────
class ResultTracker:
    """
    Weryfikuje wyniki kuponów, generuje raporty i retrenuje model.

    Pipeline:
        1. verify_yesterday()  → pobiera box scores, zapisuje WIN/LOSS/PUSH
        2. generate_report()   → hit rate, ROI, calibration, top/worst players
        3. retrain_if_due()    → co 30 nowych wyników retrenuje model
    """

    def __init__(
        self,
        db_path:   str = DB_PATH,
        model_dir: str = MODEL_DIR,
        fetcher:   Optional[NBADataFetcher] = None,
        stake:     float = STAKE_PER_PICK,
    ):
        self.db_path   = db_path
        self.model_dir = model_dir
        self.stake     = stake
        self.fetcher   = fetcher or NBADataFetcher(db_path)
        self._init_db()
        logger.info("ResultTracker initialized (db=%s)", db_path)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS results (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    coupon_id        INTEGER NOT NULL,
                    game_date        TEXT    NOT NULL,
                    player_name      TEXT    NOT NULL,
                    player_id        INTEGER,
                    game             TEXT,
                    bet              TEXT    NOT NULL,   -- over/under
                    line             REAL    NOT NULL,
                    odds             REAL    NOT NULL,
                    pts_predicted    REAL,
                    p_over           REAL,
                    confidence       INTEGER,
                    top_feature      TEXT,              -- top_features[0]
                    actual_pts       REAL,
                    result           TEXT,              -- WIN/LOSS/PUSH
                    simulated_profit REAL,
                    verified_at      TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS retrain_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    version      INTEGER,
                    trigger_date TEXT,
                    n_new_samples INTEGER,
                    cv_mae       REAL,
                    cv_auc       REAL,
                    notes        TEXT
                );
            """)
            # Migrate old schema: add any columns missing from early versions
            existing = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
            migrations = {
                "player_name":     "TEXT NOT NULL DEFAULT ''",
                "game":            "TEXT",
                "odds":            "REAL NOT NULL DEFAULT 0",
                "pts_predicted":   "REAL",
                "p_over":          "REAL",
                "confidence":      "INTEGER",
                "top_feature":     "TEXT",
                "verified_at":     "TEXT DEFAULT (datetime('now'))",
            }
            for col, dtype in migrations.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE results ADD COLUMN {col} {dtype}")
                    logger.info("DB migration: added column results.%s", col)
        logger.debug("ResultTracker DB initialized")

    # ── Weryfikacja wyników ───────────────────────────────────────────────────
    def verify_yesterday(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Pobiera box scores z poprzedniego dnia i weryfikuje kupony.

        target_date: YYYY-MM-DD (domyślnie wczoraj)
        Zwraca DataFrame z wynikami (WIN/LOSS/PUSH).
        """
        if target_date is None:
            target_date = (date.today() - timedelta(days=1)).isoformat()

        logger.info("Weryfikacja wyników dla daty: %s", target_date)

        # Pobierz kupony z danego dnia
        with self._conn() as conn:
            coupons = pd.read_sql_query(
                "SELECT id, coupon_json FROM coupons WHERE date = ?",
                conn,
                params=(target_date,),
            )

        if coupons.empty:
            logger.warning("Brak kuponów dla daty %s", target_date)
            return pd.DataFrame()

        verified_rows = []
        for _, coup_row in coupons.iterrows():
            coupon    = json.loads(coup_row["coupon_json"])
            coupon_id = coup_row["id"]

            for pick in coupon.get("picks", []):
                player_name = pick["player"]
                player_id   = self._resolve_player_id(player_name)

                if player_id is None:
                    logger.warning("Nie znaleziono player_id dla: %s", player_name)
                    continue

                # Pobierz rzeczywiste punkty z box score
                actual_pts = self._fetch_actual_pts(player_id, target_date)
                if actual_pts is None:
                    logger.warning(
                        "Brak box score dla %s (%s)", player_name, target_date
                    )
                    continue

                # Wyznacz wynik
                result, profit = self._calculate_result(
                    actual_pts=actual_pts,
                    line=pick["line"],
                    bet=pick["bet"],
                    odds=pick["odds"],
                    stake=self.stake,
                )

                top_feature = (
                    pick["top_features"][0] if pick.get("top_features") else None
                )

                row = {
                    "coupon_id":       coupon_id,
                    "game_date":       target_date,
                    "player_name":     player_name,
                    "player_id":       player_id,
                    "game":            pick.get("game", ""),
                    "bet":             pick["bet"],
                    "line":            pick["line"],
                    "odds":            pick["odds"],
                    "pts_predicted":   pick.get("pts_predicted"),
                    "p_over":          pick.get("p_over"),
                    "confidence":      pick.get("confidence"),
                    "top_feature":     top_feature,
                    "actual_pts":      actual_pts,
                    "result":          result,
                    "simulated_profit": profit,
                }
                verified_rows.append(row)

                logger.info(
                    "%s | actual=%.1f vs line=%.1f | %s %s | profit=%.2f",
                    player_name, actual_pts, pick["line"],
                    pick["bet"].upper(), result, profit,
                )

        if not verified_rows:
            return pd.DataFrame()

        df = pd.DataFrame(verified_rows)
        self._save_results(df)
        return df

    def _resolve_player_id(self, player_name: str) -> Optional[int]:
        """Szuka player_id w lokalnym cache SQLite (dane z Modułu 1)."""
        try:
            from nba_api.stats.static import players as nba_players
            results = nba_players.find_players_by_full_name(player_name)
            return results[0]["id"] if results else None
        except Exception as e:
            logger.warning("_resolve_player_id error: %s", e)
            return None

    def _fetch_actual_pts(
        self, player_id: int, game_date: str
    ) -> Optional[float]:
        """
        Pobiera rzeczywiste punkty gracza z game logu dla danej daty.
        Najpierw sprawdza cache SQLite, potem nba_api.
        """
        # Sprawdź cache
        with self._conn() as conn:
            cached = conn.execute(
                """
                SELECT data FROM cache
                WHERE key = ? AND (unixepoch() - created_at) < 86400
                """,
                (f"box_{player_id}_{game_date}",),
            ).fetchone()

        if cached:
            return float(pickle_loads_safe(cached[0]))

        # Pobierz przez nba_api
        try:
            recent = self.fetcher.get_player_recent_stats(player_id, last_n=5)
            if recent.empty:
                return None
            # Szukamy meczu z konkretną datą
            recent["date_str"] = pd.to_datetime(
                recent["GAME_DATE"]
            ).dt.date.astype(str)
            match = recent[recent["date_str"] == game_date]
            if match.empty:
                return None
            return float(match.iloc[0]["PTS"])
        except Exception as e:
            logger.error("_fetch_actual_pts error: %s", e)
            return None

    @staticmethod
    def _calculate_result(
        actual_pts: float,
        line: float,
        bet: str,
        odds: float,
        stake: float,
    ) -> tuple[str, float]:
        """
        Wyznacza wynik (WIN/LOSS/PUSH) i profit.

        PUSH: |actual - line| < 0.5 (standardowy próg)
        WIN:  profit = stake * (odds - 1)
        LOSS: profit = -stake
        PUSH: profit = 0
        """
        if abs(actual_pts - line) < 0.5:
            return "PUSH", 0.0

        went_over = actual_pts > line
        if (bet == "over" and went_over) or (bet == "under" and not went_over):
            return "WIN", round(stake * (odds - 1), 2)
        return "LOSS", round(-stake, 2)

    def _save_results(self, df: pd.DataFrame):
        with self._conn() as conn:
            df.to_sql("results", conn, if_exists="append", index=False)
        logger.info("Zapisano %d wyników do SQLite", len(df))

    # ── Raportowanie ──────────────────────────────────────────────────────────
    def generate_report(self, last_n_days: int = 30) -> dict:
        """
        Generuje kompleksowy raport wydajności modelu.

        Zawiera:
        - Overall: hit rate, ROI, P&L
        - Over vs Under hit rate
        - Hit rate per bucket kursów
        - Hit rate per dominant feature
        - Top/worst gracze
        - Calibration data (p_over vs actual win rate)
        """
        since = (date.today() - timedelta(days=last_n_days)).isoformat()
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM results WHERE game_date >= ? AND result != 'PUSH'",
                conn,
                params=(since,),
            )

        if df.empty:
            logger.warning("Brak wyników do raportu (last_%dd)", last_n_days)
            return {"error": "Brak danych"}

        report = {
            "period":        f"ostatnie {last_n_days} dni",
            "generated_at":  datetime.now().isoformat(),
            "total_picks":   len(df),
        }

        # ── 1. Overall ────────────────────────────────────────────────────────
        wins   = (df["result"] == "WIN").sum()
        losses = (df["result"] == "LOSS").sum()
        total_stake  = len(df) * self.stake
        total_profit = df["simulated_profit"].sum()

        report["overall"] = {
            "wins":      int(wins),
            "losses":    int(losses),
            "hit_rate":  round(wins / len(df), 4) if len(df) > 0 else 0,
            "roi":       round(total_profit / total_stake, 4) if total_stake > 0 else 0,
            "total_profit": round(float(total_profit), 2),
            "total_stake":  round(float(total_stake),  2),
        }

        # ── 2. Over vs Under ──────────────────────────────────────────────────
        over_df  = df[df["bet"] == "over"]
        under_df = df[df["bet"] == "under"]
        report["by_side"] = {
            "over": {
                "n":        len(over_df),
                "hit_rate": round(
                    (over_df["result"] == "WIN").mean(), 4
                ) if len(over_df) > 0 else 0,
            },
            "under": {
                "n":        len(under_df),
                "hit_rate": round(
                    (under_df["result"] == "WIN").mean(), 4
                ) if len(under_df) > 0 else 0,
            },
        }

        # ── 3. Hit rate per bucket kursów ─────────────────────────────────────
        report["by_odds_bucket"] = {}
        for lo, hi, label in ODDS_BUCKETS:
            bucket = df[(df["odds"] >= lo) & (df["odds"] < hi)]
            if len(bucket) == 0:
                continue
            bucket_wins = (bucket["result"] == "WIN").sum()
            report["by_odds_bucket"][label] = {
                "n":        len(bucket),
                "hit_rate": round(bucket_wins / len(bucket), 4),
                "roi":      round(
                    bucket["simulated_profit"].sum() / (len(bucket) * self.stake), 4
                ),
            }

        # ── 4. Hit rate per dominant feature ─────────────────────────────────
        if "top_feature" in df.columns and df["top_feature"].notna().any():
            feature_stats = (
                df.groupby("top_feature")
                .apply(lambda g: pd.Series({
                    "n":        len(g),
                    "hit_rate": round((g["result"] == "WIN").mean(), 4),
                    "roi":      round(
                        g["simulated_profit"].sum() / (len(g) * self.stake), 4
                    ),
                }))
                .reset_index()
                .sort_values("hit_rate", ascending=False)
            )
            report["by_top_feature"] = feature_stats.to_dict("records")

        # ── 5. Top/worst gracze ───────────────────────────────────────────────
        player_stats = (
            df.groupby("player_name")
            .apply(lambda g: pd.Series({
                "n":        len(g),
                "hit_rate": round((g["result"] == "WIN").mean(), 4),
                "profit":   round(float(g["simulated_profit"].sum()), 2),
            }))
            .reset_index()
        )
        report["best_players"]  = (
            player_stats.sort_values("hit_rate", ascending=False).head(5).to_dict("records")
        )
        report["worst_players"] = (
            player_stats.sort_values("hit_rate", ascending=True).head(5).to_dict("records")
        )

        # ── 6. Calibration data ───────────────────────────────────────────────
        report["calibration"] = self._compute_calibration(df)

        self._print_report(report)
        return report

    @staticmethod
    def _compute_calibration(df: pd.DataFrame) -> dict:
        """
        Sprawdza czy p_over=0.7 faktycznie trafia 70% czasu.
        Używa sklearn calibration_curve.
        Zwraca: {prob_pred: [...], prob_true: [...], n_bins: 5}
        """
        valid = df[df["p_over"].notna() & df["result"].isin(["WIN", "LOSS"])].copy()
        if len(valid) < 10:
            return {"error": "Za mało danych do kalibracji (min. 10)"}

        # Dla betów "over": WIN = gracz faktycznie był over
        # Dla betów "under": WIN = gracz był under → p_over_actual = 0
        valid["y_true"] = (
            (valid["bet"] == "over") & (valid["result"] == "WIN")
        ).astype(int) | (
            (valid["bet"] == "under") & (valid["result"] == "LOSS")
        ).astype(int)

        prob_true, prob_pred = calibration_curve(
            valid["y_true"],
            valid["p_over"],
            n_bins=5,
            strategy="quantile",
        )
        return {
            "prob_pred": [round(float(p), 3) for p in prob_pred],
            "prob_true": [round(float(p), 3) for p in prob_true],
            "n_bins":    5,
            "n_samples": len(valid),
        }

    @staticmethod
    def _print_report(report: dict):
        """Ładny print raportu do konsoli/logów."""
        ov = report.get("overall", {})
        print("\n" + "="*60)
        print(f"RAPORT WYNIKÓW — {report.get('period', '')}")
        print("="*60)
        print(f"  Picks:     {report['total_picks']}")
        print(f"  Hit rate:  {ov.get('hit_rate', 0)*100:.1f}%  "
              f"({ov.get('wins', 0)}W / {ov.get('losses', 0)}L)")
        print(f"  ROI:       {ov.get('roi', 0)*100:.2f}%")
        print(f"  P&L:       {ov.get('total_profit', 0):+.2f} PLN\n")

        print("  Hit rate per kurs:")
        for bucket, stats in report.get("by_odds_bucket", {}).items():
            print(f"    {bucket}: {stats['hit_rate']*100:.1f}%  "
                  f"(n={stats['n']}, ROI={stats['roi']*100:.2f}%)")

        print("\n  Hit rate over vs under:")
        for side, stats in report.get("by_side", {}).items():
            print(f"    {side.upper()}: {stats['hit_rate']*100:.1f}% (n={stats['n']})")

        if "by_top_feature" in report:
            print("\n  Hit rate per top feature:")
            for row in report["by_top_feature"][:5]:
                print(f"    {row['top_feature']:<30} "
                      f"{row['hit_rate']*100:.1f}%  (n={row['n']})")

        print("\n  Najlepsi gracze:")
        for p in report.get("best_players", [])[:3]:
            print(f"    {p['player_name']:<22} {p['hit_rate']*100:.1f}%  "
                  f"profit={p['profit']:+.2f}")

        cal = report.get("calibration", {})
        if "prob_pred" in cal:
            print(f"\n  Kalibracja (n={cal['n_samples']}):")
            for pp, pt in zip(cal["prob_pred"], cal["prob_true"]):
                bar = "█" * int(pt * 20)
                print(f"    p_over={pp:.2f} → faktyczne={pt:.2f}  {bar}")

    # ── Calibration chart (Plotly) ────────────────────────────────────────────
    def plot_calibration(
        self,
        report: dict,
        save_path: str = "calibration_chart.png",
    ):
        """
        Generuje calibration chart (reliability diagram).
        Idealna kalibracja = linia diagonalna.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly niezainstalowany — pip install plotly")
            return

        cal = report.get("calibration", {})
        if "prob_pred" not in cal:
            logger.warning("Brak danych kalibracji w raporcie")
            return

        fig = go.Figure()

        # Idealna kalibracja
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray", "width": 1},
            name="Idealna kalibracja",
        ))

        # Nasza kalibracja
        fig.add_trace(go.Scatter(
            x=cal["prob_pred"],
            y=cal["prob_true"],
            mode="lines+markers",
            marker={"size": 10, "color": "#1f77b4"},
            line={"color": "#1f77b4", "width": 2},
            name=f"Model (n={cal['n_samples']})",
            text=[
                f"pred={pp:.2f}<br>actual={pt:.2f}"
                for pp, pt in zip(cal["prob_pred"], cal["prob_true"])
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

        fig.update_layout(
            title="Calibration Chart — NBA Player Props",
            xaxis_title="Predicted p_over",
            yaxis_title="Actual win rate",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            width=600, height=500,
            template="plotly_white",
        )
        fig.write_image(save_path)
        logger.info("Calibration chart zapisany: %s", save_path)

    # ── Retrenowanie ──────────────────────────────────────────────────────────
    def retrain_if_due(
        self,
        predictor:        PropPredictor,
        min_new_samples:  int = 30,
    ) -> bool:
        """
        Retrenuje model gdy zgromadzono >= min_new_samples nowych wyników
        od ostatniego treningu.

        Strategia:
        1. Pobierz wszystkie wyniki z DB jako nowe dane treningowe
        2. Połącz z istniejącymi danymi historycznymi (jeśli są)
        3. Retrenuj z n_trials=50
        4. Zaloguj metryki do retrain_log
        5. Zapisz nową wersję modelu

        Zwraca True jeśli retrenował, False jeśli za mało danych.
        """
        with self._conn() as conn:
            # Sprawdź ile nowych wyników od ostatniego treningu
            last_retrain = conn.execute(
                "SELECT MAX(trigger_date) FROM retrain_log"
            ).fetchone()[0]

            query = "SELECT * FROM results WHERE result IN ('WIN','LOSS')"
            params = ()
            if last_retrain:
                query  += " AND verified_at > ?"
                params  = (last_retrain,)

            new_results = pd.read_sql_query(query, conn, params=params)

        n_new = len(new_results)
        logger.info(
            "retrain_if_due: %d nowych wyników (próg=%d)",
            n_new, min_new_samples,
        )

        if n_new < min_new_samples:
            logger.info(
                "Za mało danych do retrenowania (%d/%d) — pomijam",
                n_new, min_new_samples,
            )
            return False

        # Zbuduj X, y z wyników — potrzebujemy feature vectors
        # W produkcji: dołącz oryginalne feature vectors z kuponów
        # Tu: używamy dostępnych kolumn jako proxy
        training_df = self._build_training_df_from_results(new_results)
        if training_df is None or len(training_df) < min_new_samples:
            logger.warning("Nie można zbudować danych treningowych z wyników")
            return False

        X, y_reg, y_clf = PropPredictor.prepare_training_data(training_df)

        logger.info(
            "Retrenowanie: %d próbek | poprzednia wersja: v%d",
            len(X), predictor.version,
        )
        metrics = predictor.train(X, y_reg, y_clf, n_trials=50)

        # Zapisz log retrenowania
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO retrain_log
                    (version, trigger_date, n_new_samples, cv_mae, cv_auc, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics["version"],
                    datetime.now().isoformat(),
                    n_new,
                    metrics["cv_mae"],
                    metrics["cv_auc"],
                    f"Auto-retrain po {n_new} nowych wynikach",
                ),
            )

        logger.info(
            "Retrenowanie zakończone: v%d | CV_MAE=%.2f | CV_AUC=%.3f",
            metrics["version"], metrics["cv_mae"], metrics["cv_auc"],
        )
        return True

    def _build_training_df_from_results(
        self, results_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Buduje DataFrame treningowy z tabeli results.
        Pobiera feature vectors na nowo z nba_api dla każdego gracza.
        W praktyce produkcyjnej lepiej cache'ować feature vectors przy
        generowaniu kuponu — tu robimy best-effort.
        """
        rows = []
        for _, row in results_df.iterrows():
            try:
                player_id = int(row["player_id"]) if row["player_id"] else None
                if player_id is None:
                    continue

                # Szukamy opponent_team_id z game log
                recent = self.fetcher.get_player_recent_stats(player_id, last_n=5)
                if recent.empty:
                    continue

                # Bierzemy mecz z danego dnia jeśli dostępny, albo ostatni
                game_match = recent[
                    recent["GAME_DATE"].dt.date.astype(str) == row["game_date"]
                ] if not recent.empty else pd.DataFrame()

                opp_id = (
                    int(game_match.iloc[0]["opponent_team_id"])
                    if not game_match.empty
                    else int(recent.iloc[0]["opponent_team_id"])
                )

                fv = build_feature_vector(
                    player_id=player_id,
                    opponent_team_id=opp_id,
                    line=float(row["line"]),
                    over_odds=float(row["odds"]) if row["bet"] == "over" else 1.75,
                    under_odds=float(row["odds"]) if row["bet"] == "under" else 1.75,
                    is_home=0,
                    fetcher=self.fetcher,
                )
                fv_row = fv.iloc[0].to_dict()
                fv_row["actual_pts"] = float(row["actual_pts"])
                rows.append(fv_row)

            except Exception as e:
                logger.warning(
                    "Pominięto %s przy budowaniu danych treningowych: %s",
                    row.get("player_name", "?"), e,
                )
                continue

        if not rows:
            return None

        return pd.DataFrame(rows)

    # ── Historia retrenowań ───────────────────────────────────────────────────
    def get_retrain_history(self) -> pd.DataFrame:
        """Zwraca historię retrenowań z metrykami."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT version, trigger_date, n_new_samples, "
                "cv_mae, cv_auc, notes "
                "FROM retrain_log ORDER BY version",
                conn,
            )

    def get_cumulative_pnl(self) -> pd.DataFrame:
        """Zwraca skumulowany P&L w czasie — do wykresu equity curve."""
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT game_date, simulated_profit FROM results "
                "WHERE result != 'PUSH' ORDER BY game_date",
                conn,
            )
        if df.empty:
            return df
        df["cumulative_pnl"] = df["simulated_profit"].cumsum()
        return df


# Helper — bezpieczny pickle.loads bez importu na top level
def pickle_loads_safe(data):
    import pickle
    try:
        return pickle.loads(data)
    except Exception:
        return None
