# nba_betting/daily.py
"""
Kupon dnia (główna strona dashboardu): jeden typ na mecz ze wszystkich modeli (PTS, REB, AST, FG3M).

Reguła (ustalona 2026-11-17 przed backtestem, wg opisu użytkownika):
  kandydaci = każda linia × strona (over/under) × bukmacher z dzisiejszych predykcji,
  wartość oczekiwana EV = p_strony · kurs − 1 ≥ EV_MIN i kurs w [ODDS_MIN, ODDS_MAX],
  w meczu wybieramy kandydata z NAJWIĘKSZĄ PEWNOŚCIĄ MODELU (p_strony); remis → wyższe EV.
  Ta sama linia u kilku bukmacherów: liczy się najlepszy kurs („line shopping”).

Backtest (python daily.py) na głównych liniach Kalshi z eval_stats (ceny ask/bid + prowizja):
reguła główna na dev i na holdoucie (bez wyboru reguły na holdoucie) + dla porównania na dev:
max EV oraz pewność przy EV ≥ 5%.

Użycie:
  python daily.py            # → reports/daily_<data>.json
"""
import json
import logging
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from config import DB_PATH
from evaluate_stats import _cluster_ci
from feature_lab import FINAL_FROM
from kalshi import kalshi_fee

logger = logging.getLogger(__name__)

STATS    = ("PTS", "REB", "AST", "FG3M")
EV_MIN   = 0.0
ODDS_MIN = 1.40
ODDS_MAX = 3.00
STAT_PL  = {"PTS": "pkt", "REB": "zb.", "AST": "as.", "FG3M": "3pt"}


def candidates(preds: pd.DataFrame) -> pd.DataFrame:
    """preds: game, player_name, stat, line, p_over, over_odds, under_odds[, bookmaker, mu, player_id].
    Zwraca jeden wiersz na linię × stronę (najlepszy kurs spośród bukmacherów)."""
    sides = []
    for side, p, odds in (("over", preds["p_over"], preds["over_odds"]),
                          ("under", 1 - preds["p_over"], preds["under_odds"])):
        sides.append(preds.assign(bet=side, p_side=p, odds=odds))
    c = pd.concat(sides, ignore_index=True).dropna(subset=["p_side", "odds"])
    c["ev"] = c["p_side"] * c["odds"] - 1
    key = ["game", "player_name", "stat", "line", "bet"]
    return c.sort_values("odds").drop_duplicates(key, keep="last").reset_index(drop=True)


def pick(c: pd.DataFrame, rule: str = "confidence", ev_min: float = EV_MIN) -> pd.DataFrame:
    """Jeden kandydat na mecz. rule: confidence (max p_strony, potem EV) albo ev (max EV)."""
    ok = c[(c["ev"] >= ev_min) & c["odds"].between(ODDS_MIN, ODDS_MAX)]
    order = ["p_side", "ev"] if rule == "confidence" else ["ev", "p_side"]
    return ok.sort_values(order, ascending=False).drop_duplicates("game").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Na żywo
# ─────────────────────────────────────────────────────────────────────────────
def build(game_date: str, db_path: str = DB_PATH) -> dict:
    """Kupon dnia z live_predictions (najnowszy przebieg każdej statystyki tego dnia)."""
    with sqlite3.connect(db_path) as conn:
        try:
            p = pd.read_sql_query("SELECT * FROM live_predictions WHERE game_date = ?", conn, params=(game_date,))
        except Exception:                       # brak tabeli (pierwszy przebieg)
            p = pd.DataFrame()
    if p.empty:
        return {"date": game_date, "picks": [], "n_candidates": 0, "rule": RULE_TEXT}
    last = p.groupby("stat")["created_at"].transform("max")
    p = p[p["created_at"] == last]
    c = candidates(p)
    picks = pick(c)
    out = [{
        "game": r.game, "player": r.player_name, "player_id": None if pd.isna(r.player_id) else int(r.player_id),
        "stat": r.stat, "bet": r.bet, "line": r.line, "odds": round(float(r.odds), 3),
        "bookmaker": r.bookmaker, "mu": None if pd.isna(r.mu) else round(float(r.mu), 1),
        "confidence": round(float(r.p_side), 3), "ev": round(float(r.ev), 4),
        "reasoning": f"model {r.mu:.1f} {STAT_PL.get(r.stat, '')} vs linia {r.line} ({r.bookmaker}), "
                     f"pewność {r.p_side:.0%}, EV {r.ev:+.1%}",
    } for r in picks.itertuples()]
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return {"date": game_date, "picks": out, "n_candidates": int(len(c)),
            "n_games": int(p["game"].nunique()), "rule": RULE_TEXT, "paper_trading": True}


def history(db_path: str = DB_PATH, actual_fns: dict | None = None, before: str | None = None) -> list[dict]:
    """Kupony dnia z poprzednich dni odtworzone z live_predictions + wynik z logów (paper trading, stawka 1)."""
    with sqlite3.connect(db_path) as conn:
        try:
            dates = [r[0] for r in conn.execute("SELECT DISTINCT game_date FROM live_predictions ORDER BY game_date")]
        except sqlite3.Error:
            return []
    if before:
        dates = [d for d in dates if d < before]
    if not dates:
        return []
    from live import actual_stat_fn            # import tu: live ładuje cały model
    actual_fns = {} if actual_fns is None else actual_fns
    rows = []
    for d in dates:
        for pk in build(d, db_path)["picks"]:
            st = pk["stat"]
            if st not in actual_fns:
                actual_fns[st] = actual_stat_fn(st)
            a = actual_fns[st](pk["player"], d)
            res = None if a is None else ("WIN" if (a > pk["line"]) == (pk["bet"] == "over") else "LOSS")
            rows.append({**pk, "date": d, "actual": a, "result": res,
                         "profit": None if res is None else round(pk["odds"] - 1 if res == "WIN" else -1.0, 3)})
    return rows


RULE_TEXT = (f"jeden typ na mecz: spośród linii z EV ≥ {EV_MIN:.0%} i kursem {ODDS_MIN}–{ODDS_MAX} "
             "największa pewność modelu")


# ─────────────────────────────────────────────────────────────────────────────
# Backtest na liniach Kalshi
# ─────────────────────────────────────────────────────────────────────────────
def kalshi_candidates(pred: pd.DataFrame) -> pd.DataFrame:
    """eval_stats predictions → format candidates (kurs = 1 / (cena + prowizja), jak w kalshi.to_props)."""
    ask, no = pred["ask"], 1 - pred["bid"]
    d = pred.assign(game=pred["GAME_ID"].astype(str),
                    over_odds=1 / (ask + ask.map(kalshi_fee)), under_odds=1 / (no + no.map(kalshi_fee)),
                    p_over=pred["p_base"])
    d = d[(pred["bid"] > 0) & (pred["ask"] > 0)]
    return candidates(d)


def backtest_rule(c: pd.DataFrame, rule: str, ev_min: float, rng) -> dict:
    picks = []
    for _, day in c.groupby("game_date"):
        picks.append(pick(day, rule, ev_min))
    p = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    if p.empty:
        return {"picks": 0}
    win = np.where(p["bet"] == "over", p["y_over"] == 1, p["y_over"] == 0)
    pnl = np.where(win, p["odds"] - 1, -1.0)
    out = {"picks": int(len(p)), "games": int(p["game"].nunique()),
           "hit_rate": round(float(win.mean()), 3), "mean_confidence": round(float(p["p_side"].mean()), 3),
           "mean_odds": round(float(p["odds"].mean()), 3), "roi": round(float(pnl.mean()), 4),
           "by_stat": p["stat"].value_counts().to_dict()}
    if len(p) >= 10:
        out["roi_ci95"] = _cluster_ci(pnl, p["game"].to_numpy(), rng)
    return out


def backtest() -> dict:
    rng = np.random.default_rng(0)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    pred = pred[pred["stat"].isin(STATS)].dropna(subset=["p_base", "y_over"])
    c = kalshi_candidates(pred)
    dev, hold = c[c["game_date"] < FINAL_FROM], c[c["game_date"] >= FINAL_FROM]
    report = {"generated": date.today().isoformat(), "predictions": mm.latest_predictions().name,
              "rule": RULE_TEXT, "stats": sorted(pred["stat"].unique()),
              "main": {"dev": backtest_rule(dev, "confidence", EV_MIN, rng),
                       "holdout": backtest_rule(hold, "confidence", EV_MIN, rng)},
              "alternatives_dev": {"max_ev": backtest_rule(dev, "ev", EV_MIN, rng),
                                   "confidence_ev5": backtest_rule(dev, "confidence", 0.05, rng)}}
    out = mm.REPORT_DIR / f"daily_{report['generated']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s → %s", out, report["main"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    backtest()
