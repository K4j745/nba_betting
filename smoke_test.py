# nba_betting/smoke_test.py
import json, sqlite3, logging
import pandas as pd
from tracker import ResultTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s"
)

DB = "smoke_test.db"

# ── Wstaw testowy kupon do DB ──────────────────────────────────────────────
with sqlite3.connect(DB) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coupons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            coupon_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    coupon = {
        "date": "2026-03-06",
        "picks": [
            {"player": "Victor Wembanyama", "game": "SAS vs PHI",
             "bet": "under", "line": 25.5, "odds": 1.87,
             "pts_predicted": 22.1, "p_over": 0.28, "confidence": 74,
             "top_features": ["pts_avg_10", "opp_def_rating", "line"]},
            {"player": "Jayson Tatum", "game": "BOS vs MIL",
             "bet": "over", "line": 28.5, "odds": 1.75,
             "pts_predicted": 30.2, "p_over": 0.72, "confidence": 68,
             "top_features": ["pts_avg_3", "pts_trend"]},
        ]
    }
    conn.execute(
        "INSERT INTO coupons (date, coupon_json) VALUES (?, ?)",
        ("2026-03-06", json.dumps(coupon))
    )
    conn.commit()

# ── Wstaw wyniki ───────────────────────────────────────────────────────────
import numpy as np
from unittest.mock import patch

tracker = ResultTracker(db_path=DB, stake=10.0)

with (
    patch.object(tracker, "_resolve_player_id", return_value=1641705),
    patch.object(tracker, "_fetch_actual_pts",  return_value=22.0),
):
    results = tracker.verify_yesterday("2026-03-06")

print("\n=== WYNIKI WERYFIKACJI ===")
print(results[["player_name", "actual_pts", "line", "bet", "result", "simulated_profit"]])

# ── Dodaj więcej wyników do raportu ───────────────────────────────────────
np.random.seed(42)
n = 40
fake = pd.DataFrame({
    "coupon_id":        [1] * n,
    "game_date":        ["2026-03-06"] * n,
    "player_name":      [f"Player{i%5}" for i in range(n)],
    "player_id":        [1000 + i%5 for i in range(n)],
    "game":             ["SAS vs PHI"] * n,
    "bet":              ["over","under"] * 20,
    "line":             [25.5] * n,
    "odds":             np.random.uniform(1.45, 1.88, n),
    "pts_predicted":    np.random.normal(22, 3, n),
    "p_over":           np.random.uniform(0.3, 0.75, n),
    "confidence":       np.random.randint(62, 88, n),
    "top_feature":      ["pts_avg_10","line","opp_def_rating","pts_trend","pct_over_line_historical"] * 8,
    "actual_pts":       np.random.normal(22, 5, n),
    "result":           ["WIN","LOSS"] * 20,
    "simulated_profit": [8.3 if i%2==0 else -10.0 for i in range(n)],
})
with tracker._conn() as conn:
    fake.to_sql("results", conn, if_exists="append", index=False)

# ── Raport ─────────────────────────────────────────────────────────────────
report = tracker.generate_report(last_n_days=365)

# ── Cumulative P&L ────────────────────────────────────────────────────────
pnl = tracker.get_cumulative_pnl()
print(f"\nKońcowy P&L: {pnl['cumulative_pnl'].iloc[-1]:+.2f} PLN")

# ── Sprzątanie ─────────────────────────────────────────────────────────────
import os
os.remove(DB)
print("\n✓ Smoke test przeszedł — Moduł 5 działa poprawnie")
