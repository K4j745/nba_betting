# nba_betting/train.py
"""
Skrypt do ręcznego treningu modelu na danych historycznych.

Użycie:
  python train.py                     # trening na danych z DB (domyślnie)
  python train.py --trials 100        # więcej prób Optuna
  python train.py --csv data.csv      # trening z pliku CSV

Format CSV:
  Musi zawierać kolumny z MODEL_FEATURE_COLS + actual_pts + line
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from config import DB_PATH, MODEL_DIR
from features import MODEL_FEATURE_COLS
from model import PropPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def load_from_db(db_path: str) -> pd.DataFrame:
    """Ładuje historyczne dane z SQLite (tabela results + feature cache)."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM results WHERE result IN ('WIN','LOSS')",
                conn,
            )
            logger.info("Załadowano %d historycznych wyników z DB", len(df))
            return df
        except Exception as e:
            logger.error("Błąd ładowania z DB: %s", e)
            return pd.DataFrame()


def generate_synthetic_data(n: int = 300) -> pd.DataFrame:
    """
    Generuje syntetyczne dane gdy brak historycznych.
    UWAGA: Służy tylko do testów architektury — nie do realnych predykcji.
    """
    logger.warning(
        "Używam SYNTETYCZNYCH danych treningowych — "
        "model nie będzie miał wartości predykcyjnej na realnych grach!"
    )
    np.random.seed(42)
    df = pd.DataFrame({
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
    df["actual_pts"] = df["pts_avg_10"] + np.random.normal(0, 4, n)
    return df


def main():
    parser = argparse.ArgumentParser(description="Trening modelu NBA Betting")
    parser.add_argument("--trials", type=int, default=50, help="Liczba prób Optuna")
    parser.add_argument("--csv",    type=str, default=None, help="Ścieżka do CSV")
    parser.add_argument("--synthetic", action="store_true",
                        help="Wymuś syntetyczne dane (debug)")
    args = parser.parse_args()

    # Załaduj dane
    if args.csv:
        logger.info("Ładuję dane z CSV: %s", args.csv)
        df = pd.read_csv(args.csv)
    elif args.synthetic:
        df = generate_synthetic_data(300)
    else:
        df = load_from_db(DB_PATH)
        if df.empty:
            logger.warning("Brak danych w DB — używam syntetycznych")
            df = generate_synthetic_data(300)

    logger.info("Dane treningowe: %d wierszy", len(df))

    # Sprawdź czy mamy wymagane kolumny
    missing = [c for c in MODEL_FEATURE_COLS if c not in df.columns]
    if missing and not args.synthetic:
        logger.error(
            "Brakuje kolumn w danych: %s\n"
            "Spróbuj: python train.py --synthetic",
            missing,
        )
        sys.exit(1)

    # Trening
    predictor = PropPredictor(db_path=DB_PATH, model_dir=MODEL_DIR)
    X, y_reg, y_clf = PropPredictor.prepare_training_data(df)

    logger.info("Trening z %d próbami Optuna...", args.trials)
    metrics = predictor.train(X, y_reg, y_clf, n_trials=args.trials)

    print("\n" + "=" * 50)
    print(f"  Model v{metrics['version']} wytrenowany!")
    print(f"  CV MAE:  {metrics['cv_mae']}")
    print(f"  CV AUC:  {metrics['cv_auc']}")
    print(f"  Próbek:  {metrics['n_samples']}")
    print("=" * 50)
    print(f"\n  Możesz teraz odpalić pipeline:")
    print(f"  python main.py --run-now\n")


if __name__ == "__main__":
    main()
