# nba_betting/market_gap.py
"""
Różnica Kalshi − Pinnacle (punkty): czy Kalshi spóźnia się za ostrzejszym rynkiem?

Pinnacle i Kalshi mają inne progi (Kalshi co 5 pkt), więc porównujemy je na progu Pinnacle L:
cenę Kalshi interpolujemy po logicie między sąsiednimi progami drabinki.

    gap = logit p_Pinnacle(L) − logit p_Kalshi(L)

Na linii głównej Kalshi K przyjmujemy to samo przesunięcie: logit p = logit p_Kalshi(K) + gap.
Odczyt Pinnacle musi być z chwili ≤ ceny Kalshi (start − 1 h): inaczej „wiedzielibyśmy” więcej niż
w momencie zakładu. Jeden wcześniej ustalony test, bez strojenia progów; historia Pinnacle to
głównie X–XI 2026, więc nie ma z czego zostawić osobnej próby końcowej.

  1. który rynek jest ostrzejszy na tym samym progu L (log-loss),
  2. korekta Kalshi: logit p = α + β·logit p_Kalshi(K) + γ·gap, walk-forward i γ z CI,
  3. zakłady bez dopasowania: p Pinnacle na progu K, YES po ask / NO po 1 − bid, z prowizją.

Użycie:
  python market_gap.py        # → reports/market_gap_<data>.json
"""
import json
import logging
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from config import DB_PATH
from evaluate import attach_games
from evaluate_stats import _cluster_ci, _logloss, simulate
from features import remove_overround
from history import load_dataset, norm_name
from kalshi import load_history, main_lines

logger = logging.getLogger(__name__)

GAP_BINS = (-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf)


def pinnacle_reads(db_path: str = DB_PATH) -> pd.DataFrame:
    """Wszystkie odczyty Pinnacle dla punktów, z kursami bez marży."""
    with sqlite3.connect(db_path) as conn:
        o = pd.read_sql_query(
            "SELECT player_name, line AS pin_line, over_odds, under_odds, game_time, fetched_at FROM odds_history "
            "WHERE stat_type = 'player_points' AND bookmaker = 'pinnacle' AND over_odds > 1 AND under_odds > 1", conn)
    o["game_time"]  = pd.to_datetime(o["game_time"], utc=True)
    o["fetched_at"] = pd.to_datetime(o["fetched_at"], utc=True)
    o["game_date"]  = o["game_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    o["name_key"]   = o["player_name"].map(norm_name)
    o["p_pin"]      = [remove_overround(a, b)[0] for a, b in zip(o["over_odds"], o["under_odds"])]
    return o


def last_read_before(reads: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Dla każdej linii Kalshi ostatni odczyt Pinnacle ≤ snap_ts, z progu najbliższego 50/50."""
    m = lines.merge(reads[["game_date", "name_key", "pin_line", "p_pin", "fetched_at"]], on=["game_date", "name_key"])
    m = m[m["fetched_at"] <= m["snap_ts"]]
    key = ["game_date", "name_key"]
    m = m[m["fetched_at"] == m.groupby(key)["fetched_at"].transform("max")]
    return m.assign(_d=(m["p_pin"] - 0.5).abs()).sort_values("_d").drop_duplicates(key).drop(columns="_d")


def kalshi_at(ladder: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """p Kalshi na progu pin_line: interpolacja logitu po drabince; NaN poza zakresem progów."""
    steps = {k: (g["line"].to_numpy(), mm._logit(g["p_mid"].to_numpy()))
             for k, g in ladder.dropna(subset=["p_mid"]).sort_values("line").groupby(["game_date", "name_key"])}
    out = np.full(len(pairs), np.nan)
    for i, (d, n, x) in enumerate(zip(pairs["game_date"], pairs["name_key"], pairs["pin_line"])):
        xs, ys = steps.get((d, n), ((), ()))
        if len(xs) >= 2 and xs[0] <= x <= xs[-1]:
            out[i] = 1 / (1 + np.exp(-np.interp(x, xs, ys)))
    return out


def build(data: pd.DataFrame, hist: pd.DataFrame, reads: pd.DataFrame) -> pd.DataFrame:
    ladder = hist[hist["stat"] == "PTS"].copy()
    ladder["snap_ts"] = pd.to_datetime(ladder["snap_ts"], utc=True, format="ISO8601")
    pairs = last_read_before(reads, main_lines(ladder))
    pairs["p_k_at_pin"] = kalshi_at(ladder, pairs)
    pairs = pairs.dropna(subset=["p_k_at_pin"])
    m, _ = attach_games(pairs.rename(columns={"player": "player_name"}).drop(columns=["name_key"]).assign(bookmaker="kalshi"), data)
    m = m.sort_values("game_date").reset_index(drop=True)
    m["y_pin"] = (m["PTS"] > m["pin_line"]).astype(int)
    m["lm"]    = mm._logit(m["p_mid"])
    m["gap"]   = mm._logit(m["p_pin"]) - mm._logit(m["p_k_at_pin"])
    m["p_shift"] = 1 / (1 + np.exp(-(m["lm"] + m["gap"])))
    return m


def evaluate(m: pd.DataFrame, rng) -> dict:
    games = m["GAME_ID"].astype(str).to_numpy()
    y_pin, y = m["y_pin"].to_numpy(), m["y_over"].to_numpy()
    sharp = _logloss(m["p_pin"].to_numpy(), y_pin) - _logloss(m["p_k_at_pin"].to_numpy(), y_pin)
    p, _ = mm.walk_forward(m, ["lm", "gap"])
    ok = ~np.isnan(p)
    corr = _logloss(p[ok], y[ok]) - _logloss(m["p_mid"].to_numpy()[ok], y[ok])
    bins = pd.cut(m["gap"], GAP_BINS)
    return {
        "n": int(len(m)), "n_games": int(m["GAME_ID"].nunique()),
        "from": str(m["game_date"].min().date()), "to": str(m["game_date"].max().date()),
        "hours_pin_before_price": round(float(((m["snap_ts"] - m["fetched_at"]).dt.total_seconds() / 3600).median()), 2),
        "gap_abs_mean": round(float(m["gap"].abs().mean()), 3),
        "pinnacle_minus_kalshi_logloss": round(float(sharp.mean()), 4), "sharpness_ci95": _cluster_ci(sharp, games, rng),
        "coefs_full_sample": mm.full_fit_ci(m, ["lm", "gap"], rng),
        "corrected_n": int(ok.sum()), "corrected_diff": round(float(corr.mean()), 4),
        "corrected_ci95": _cluster_ci(corr, games[ok], rng),
        "bets_pinnacle_shift": simulate(m, "p_shift", rng),
        "by_gap": [{"gap": str(b), "n": int(len(g)), "over_rate": round(float(g["y_over"].mean()), 3),
                    "kalshi_mean": round(float(g["p_mid"].mean()), 3)}
                   for b, g in m.groupby(bins, observed=True)],
    }


def run() -> dict:
    rng = np.random.default_rng(0)
    m = build(load_dataset(), load_history(), pinnacle_reads())
    report = {"generated": date.today().isoformat(), **evaluate(m, rng)}
    out = mm.REPORT_DIR / f"market_gap_{report['generated']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(run(), indent=1, ensure_ascii=False))
