# nba_betting/evaluate.py
"""
Krok 3: czy model punktów wie coś, czego nie wie rynek?

Dla każdej linii z odds_history (ostatni odczyt przed meczem, linia główna):
  - p_market  = prawdopodobieństwo over z kursów bukmachera po zdjęciu marży
  - p_model   = P(PTS > linia) z PointsModel trenowanego WYŁĄCZNIE na meczach
                sprzed dnia tej linii (walk-forward, osobny model na każdy dzień)
  - y         = czy zawodnik faktycznie przekroczył linię (z game logu)

Raport: log-loss / Brier modelu vs rynku (z bootstrapowym CI różnicy) i symulacja
płaskich zakładów przy progach EV. Wynik w reports/eval_<data>.md.

Użycie:
  python evaluate.py
"""
import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import DB_PATH
from features import remove_overround
from history import BASE_FEATURE_COLS, FEATURE_COLS, load_dataset, norm_name
from points_model import train_points_model

logger = logging.getLogger(__name__)

REPORT_DIR    = Path("reports")
EV_THRESHOLDS = (0.0, 0.03, 0.05, 0.10)
N_BOOT        = 2000
EPS           = 1e-6
# Zestawy cech porównywane w ablacji (każdy nowy krok 4 = nowy wpis)
FEATURE_SETS  = {
    "base":    BASE_FEATURE_COLS,
    "absence": FEATURE_COLS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Linie z bazy
# ─────────────────────────────────────────────────────────────────────────────
def load_lines(db_path: str = DB_PATH) -> pd.DataFrame:
    """Jedna linia na (zawodnik, mecz, bukmacher): ostatni odczyt, linia najbliższa kursowi 50/50."""
    with sqlite3.connect(db_path) as conn:
        o = pd.read_sql_query(
            "SELECT player_name, line, over_odds, under_odds, home_team, away_team, "
            "game_time, fetched_at, bookmaker FROM odds_history "
            "WHERE stat_type = 'player_points' AND over_odds > 1 AND under_odds > 1",
            conn,
        )
    o["game_time"]  = pd.to_datetime(o["game_time"], utc=True)
    o["fetched_at"] = pd.to_datetime(o["fetched_at"], utc=True)
    o = o[o["fetched_at"] < o["game_time"]]

    key = ["player_name", "game_time", "bookmaker"]
    o = o[o["fetched_at"] == o.groupby(key)["fetched_at"].transform("max")]
    o["imbalance"] = (1 / o["over_odds"] - 1 / o["under_odds"]).abs()
    o = o.sort_values("imbalance").drop_duplicates(key)

    o["game_date"] = o["game_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    fair = o.apply(lambda r: remove_overround(r["over_odds"], r["under_odds"]), axis=1)
    o["p_market"] = [f[0] for f in fair]
    o["margin"]   = 1 / o["over_odds"] + 1 / o["under_odds"] - 1
    return o.drop(columns="imbalance").reset_index(drop=True)


def attach_games(lines: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dopasowuje linię do meczu zawodnika z tego dnia (ET). Zwraca (dopasowane, niedopasowane)."""
    from rapidfuzz import process, fuzz

    g = games.copy()
    g["name_key"] = g["PLAYER_NAME"].map(norm_name)
    lines = lines.copy()
    lines["name_key"] = lines["player_name"].map(norm_name)

    # fuzzy tylko w obrębie zawodników grających tego dnia
    by_day = g.groupby("GAME_DATE")["name_key"].apply(lambda s: sorted(set(s))).to_dict()
    def resolve(row):
        cands = by_day.get(row["game_date"], [])
        if row["name_key"] in cands:
            return row["name_key"]
        hit = process.extractOne(row["name_key"], cands, scorer=fuzz.token_sort_ratio, score_cutoff=90)
        return hit[0] if hit else None
    lines["name_key"] = lines.apply(resolve, axis=1)

    m = lines.merge(g, left_on=["name_key", "game_date"], right_on=["name_key", "GAME_DATE"], how="left")
    matched = m[m["PTS"].notna()].copy()
    unmatched = m[m["PTS"].isna()][["player_name", "game_date", "bookmaker", "line"]]
    matched["y_over"] = (matched["PTS"] > matched["line"]).astype(int)
    return matched, unmatched


# ─────────────────────────────────────────────────────────────────────────────
# Predykcje walk-forward
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_p(matched: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Kolumny mu_<zestaw>, p_<zestaw> — model trenowany osobno na każdy dzień i zestaw cech."""
    out = []
    for d, block in matched.groupby("game_date"):
        block = block.copy()
        for name, cols in FEATURE_SETS.items():
            model = train_points_model(dataset, cutoff=d, features=cols)
            block[f"mu_{name}"] = model.mu(block)
            block[f"p_{name}"]  = model.p_over(block, block["line"])
        out.append(block)
    return pd.concat(out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metryki
# ─────────────────────────────────────────────────────────────────────────────
def _logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _boot_ci(values: np.ndarray, rng, stat=np.mean) -> tuple[float, float]:
    idx = rng.integers(0, len(values), (N_BOOT, len(values)))
    s = stat(values[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def score_book(df: pd.DataFrame, p_col: str, mu_col: str, rng) -> dict:
    y = df["y_over"].to_numpy()
    ll_m, ll_k = _logloss(df[p_col].to_numpy(), y), _logloss(df["p_market"].to_numpy(), y)
    br_m, br_k = (df[p_col] - y) ** 2, (df["p_market"] - y) ** 2
    diff = ll_m - ll_k     # < 0 → model lepszy od rynku
    lo, hi = _boot_ci(diff, rng)
    return {
        "n": int(len(df)),
        "over_rate": round(float(y.mean()), 3),
        "logloss_model": round(float(ll_m.mean()), 4),
        "logloss_market": round(float(ll_k.mean()), 4),
        "logloss_diff": round(float(diff.mean()), 4),
        "logloss_diff_ci95": [round(lo, 4), round(hi, 4)],
        "brier_model": round(float(br_m.mean()), 4),
        "brier_market": round(float(br_k.mean()), 4),
        "mae_points": round(float((df["PTS"] - df[mu_col]).abs().mean()), 3),
        "mae_line": round(float((df["PTS"] - df["line"]).abs().mean()), 3),
        "avg_margin": round(float(df["margin"].mean()), 4),
    }


def compare_sets(df: pd.DataFrame, a: str, b: str, rng) -> dict:
    """Sparowana różnica log-loss zestawu b względem a (< 0 → b lepszy)."""
    y = df["y_over"].to_numpy()
    diff = _logloss(df[f"p_{b}"].to_numpy(), y) - _logloss(df[f"p_{a}"].to_numpy(), y)
    mae_diff = (df["PTS"] - df[f"mu_{b}"]).abs() - (df["PTS"] - df[f"mu_{a}"]).abs()
    lo, hi = _boot_ci(diff, rng)
    mlo, mhi = _boot_ci(mae_diff.to_numpy(), rng)
    return {
        "logloss_diff": round(float(diff.mean()), 4), "logloss_diff_ci95": [round(lo, 4), round(hi, 4)],
        "mae_diff": round(float(mae_diff.mean()), 3), "mae_diff_ci95": [round(mlo, 3), round(mhi, 3)],
    }


def simulate_bets(df: pd.DataFrame, p_col: str, rng) -> list[dict]:
    """Płaska stawka 1 na stronę z wyższym EV, jeśli EV > próg."""
    ev_over  = df[p_col] * df["over_odds"] - 1
    ev_under = (1 - df[p_col]) * df["under_odds"] - 1
    side_over = ev_over >= ev_under
    ev   = np.where(side_over, ev_over, ev_under)
    win  = np.where(side_over, df["y_over"] == 1, df["y_over"] == 0)
    odds = np.where(side_over, df["over_odds"], df["under_odds"])
    pnl  = np.where(win, odds - 1, -1.0)
    rows = []
    for t in EV_THRESHOLDS:
        sel = ev > t
        n = int(sel.sum())
        if n == 0:
            rows.append({"ev_threshold": t, "bets": 0})
            continue
        lo, hi = _boot_ci(pnl[sel], rng) if n > 1 else (float("nan"), float("nan"))
        rows.append({
            "ev_threshold": t, "bets": n,
            "hit_rate": round(float(win[sel].mean()), 3),
            "roi": round(float(pnl[sel].mean()), 4),
            "roi_ci95": [round(lo, 4), round(hi, 4)],
        })
    return rows


def calibration_table(df: pd.DataFrame, p_col: str) -> list[dict]:
    bins = pd.cut(df[p_col], [0, .3, .4, .45, .5, .55, .6, .7, 1])
    t = df.groupby(bins, observed=True).agg(n=("y_over", "size"), p=(p_col, "mean"), freq=("y_over", "mean"))
    return [{"bin": str(i), "n": int(r.n), "p_mean": round(r.p, 3), "over_freq": round(r.freq, 3)}
            for i, r in t.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# Raport
# ─────────────────────────────────────────────────────────────────────────────
def run(db_path: str = DB_PATH) -> dict:
    rng = np.random.default_rng(42)
    dataset = load_dataset()
    lines = load_lines(db_path)
    matched, unmatched = attach_games(lines, dataset)
    logger.info("Linie: %d, dopasowane: %d, niedopasowane: %d", len(lines), len(matched), len(unmatched))
    pred = walk_forward_p(matched, dataset)

    report = {
        "generated": date.today().isoformat(),
        "feature_sets": {k: len(v) for k, v in FEATURE_SETS.items()},
        "lines_total": int(len(lines)),
        "lines_matched": int(len(matched)),
        "lines_unmatched": int(len(unmatched)),
        "game_dates": sorted(d.date().isoformat() for d in pred["game_date"].unique()),
        "books": {},
        "ablation": {},
    }
    names = list(FEATURE_SETS)
    for book, df in pred.groupby("bookmaker"):
        report["books"][book] = {
            name: {
                "scores": score_book(df, f"p_{name}", f"mu_{name}", rng),
                "bets": simulate_bets(df, f"p_{name}", rng),
                "calibration": calibration_table(df, f"p_{name}"),
            }
            for name in names
        }
        report["ablation"][book] = {
            f"{b}_vs_{names[0]}": compare_sets(df, names[0], b, rng) for b in names[1:]
        }

    # Line shopping: Pinnacle bez marży jako „model” dla Unibetu (ta sama linia)
    pin = pred[pred["bookmaker"] == "pinnacle"][["PLAYER_ID", "game_date", "line", "p_market"]]
    uni = pred[pred["bookmaker"] == "unibet"].merge(
        pin.rename(columns={"p_market": "p_pinnacle"}), on=["PLAYER_ID", "game_date", "line"])
    report["unibet_vs_pinnacle_fair"] = {
        "n_same_line": int(len(uni)),
        "bets": simulate_bets(uni, "p_pinnacle", rng) if len(uni) else [],
    }

    REPORT_DIR.mkdir(exist_ok=True)
    stem = REPORT_DIR / f"eval_{report['generated']}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pred.drop(columns=["name_key"], errors="ignore").to_csv(f"{stem}_predictions.csv", index=False)
    unmatched.to_csv(f"{stem}_unmatched.csv", index=False)
    logger.info("Raport: %s.json", stem)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    r = run()
    print(json.dumps(r, indent=2, ensure_ascii=False))
