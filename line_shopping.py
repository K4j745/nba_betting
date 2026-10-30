# nba_betting/line_shopping.py
"""
Krok 6: line shopping — czy różnice między bukmacherami dają przewagę?

Pinnacle (niska marża, ostre linie) traktujemy jako „prawdę rynkową” po zdjęciu marży.
  1. value_vs_pinnacle — zakład u innego bukmachera, gdy jego kurs bije uczciwą cenę Pinnacle
     o ≥ próg EV. Różne linie (np. 24.5 vs 25.5) przeliczamy przez rozkład normalny z σ(μ)
     modelu punktów. Wynik: ROI (CI po meczach) i CLV względem ostatniego odczytu Pinnacle.
  2. best_price — najlepszy over i najlepszy under spośród bukmacherów przy tej samej linii:
     ile marży zdejmuje wybór kursu, jak często zdarza się arbitraż (suma < 1).
  3. model_best_price — PointsModel (base) obstawiany po najlepszej dostępnej cenie
     zamiast u jednego bukmachera (predykcje z reports/eval_*_predictions.csv).

CLV (closing line value) = kurs wzięty × uczciwe prawdopodobieństwo z ostatniego odczytu
Pinnacle − 1. Dodatni średni CLV to najmniej zaszumiony sygnał przewagi; „zamknięcie” to tu
ostatni odczyt przed meczem (pipeline pobiera kursy kilka godzin przed meczem).

Użycie:
  python line_shopping.py        # → reports/line_shopping_<data>.json
"""
import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from config import DB_PATH
from features import remove_overround
from history import load_dataset
from evaluate import EV_THRESHOLDS, attach_games, simulate_bets

logger = logging.getLogger(__name__)

REFERENCE   = "pinnacle"
SIGMA_A     = 3.85     # σ(μ) = a + b·μ — typowe dopasowanie PointsModel (logi walidacji 2026-10)
SIGMA_B     = 0.19
MATCH_TOL   = pd.Timedelta("15min")   # odczyty z jednego uruchomienia pipeline'u
N_BOOT      = 2000
REPORT_DIR  = Path("reports")


# ─────────────────────────────────────────────────────────────────────────────
# Przeliczanie prawdopodobieństwa na inną linię
# ─────────────────────────────────────────────────────────────────────────────
def _sigma(mu):
    return SIGMA_A + SIGMA_B * mu


def p_over_at(p_ref: float, line_ref: float, line: float) -> float:
    """P(over `line`), gdy rynek daje P(over `line_ref`) = p_ref (rozkład N(μ, σ(μ)))."""
    if line == line_ref:
        return p_ref
    p_ref = min(max(p_ref, 1e-4), 1 - 1e-4)
    f = lambda mu: 1 - norm.cdf((line_ref - mu) / _sigma(mu)) - p_ref
    mu = 0.0 if f(0.0) >= 0 else brentq(f, 0.0, 120.0)   # σ(μ) > 0 tylko dla μ ≥ 0
    return float(1 - norm.cdf((line - mu) / _sigma(mu)))


# ─────────────────────────────────────────────────────────────────────────────
# Dane: wszystkie odczyty przed meczem
# ─────────────────────────────────────────────────────────────────────────────
def load_snapshots(db_path: str = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        o = pd.read_sql_query(
            "SELECT player_name, line, over_odds, under_odds, home_team, away_team, "
            "game_time, fetched_at, bookmaker FROM odds_history "
            "WHERE stat_type = 'player_points' AND over_odds > 1 AND under_odds > 1",
            conn,
        )
    o["game_time"] = pd.to_datetime(o["game_time"], utc=True, format="mixed")
    o["fetched_at"] = pd.to_datetime(o["fetched_at"], utc=True)
    o = o[o["fetched_at"] < o["game_time"]].copy()
    # kilka linii naraz (alternatywne) → linia najbliższa 50/50 w danym odczycie
    o["imbalance"] = (1 / o["over_odds"] - 1 / o["under_odds"]).abs()
    o = o.sort_values("imbalance").drop_duplicates(["player_name", "game_time", "bookmaker", "fetched_at"])
    fair = [remove_overround(a, b) for a, b in zip(o["over_odds"], o["under_odds"])]
    o["p_fair"] = [f[0] for f in fair]
    o["game_date"] = o["game_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return o.drop(columns="imbalance").sort_values("fetched_at").reset_index(drop=True)


def attach_reference(snaps: pd.DataFrame) -> pd.DataFrame:
    """
    Do każdego odczytu innego bukmachera dokleja:
      ref_*   — odczyt Pinnacle z tego samego uruchomienia (±MATCH_TOL),
      close_* — ostatni odczyt Pinnacle przed meczem.
    """
    pin = snaps[snaps["bookmaker"] == REFERENCE]
    other = snaps[snaps["bookmaker"] != REFERENCE]
    key = ["player_name", "game_time"]
    ref = pin[[*key, "fetched_at", "line", "p_fair"]].rename(
        columns={"line": "ref_line", "p_fair": "ref_p", "fetched_at": "ref_at"})
    m = pd.merge_asof(
        other.sort_values("fetched_at"), ref.sort_values("ref_at"),
        left_on="fetched_at", right_on="ref_at", by=key,
        tolerance=MATCH_TOL, direction="nearest",
    )
    close = pin.groupby(key).tail(1)[[*key, "fetched_at", "line", "p_fair"]].rename(
        columns={"line": "close_line", "p_fair": "close_p", "fetched_at": "close_at"})
    return m.merge(close, on=key, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Statystyka
# ─────────────────────────────────────────────────────────────────────────────
def _cluster_ci(values: np.ndarray, clusters: np.ndarray, rng) -> list[float]:
    """CI95 średniej z bootstrapu po klastrach (meczach)."""
    uc, inv = np.unique(clusters, return_inverse=True)
    s, c = np.bincount(inv, values, len(uc)), np.bincount(inv, minlength=len(uc))
    boots = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        boots.append((w @ s) / (w @ c))
    return [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)]


def value_vs_pinnacle(m: pd.DataFrame, rng) -> dict:
    """
    Pierwszy odczyt każdej linii u bukmachera ≠ Pinnacle, przy którym jest odczyt Pinnacle
    z tego samego uruchomienia. Strona z wyższym EV względem uczciwej ceny Pinnacle.
    """
    m = m.dropna(subset=["ref_p"]).copy()
    m = m.groupby(["player_name", "game_time", "bookmaker"]).head(1)
    p_ref = np.array([p_over_at(p, lr, l) for p, lr, l in zip(m["ref_p"], m["ref_line"], m["line"])])
    p_close = np.array([
        p_over_at(p, lc, l) if pd.notna(p) else np.nan
        for p, lc, l in zip(m["close_p"], m["close_line"], m["line"])
    ])
    ev_o, ev_u = p_ref * m["over_odds"] - 1, (1 - p_ref) * m["under_odds"] - 1
    over = (ev_o >= ev_u).to_numpy()
    m["side"] = np.where(over, "over", "under")
    m["ev"] = np.where(over, ev_o, ev_u)
    m["odds"] = np.where(over, m["over_odds"], m["under_odds"])
    m["clv"] = np.where(over, p_close, 1 - p_close) * m["odds"] - 1
    m["same_line"] = m["line"] == m["ref_line"]
    # CLV ma sens tylko, gdy po zakładzie był PÓŹNIEJSZY odczyt Pinnacle — inaczej
    # „zamknięcie” = odczyt referencyjny i CLV = EV z definicji (selekcja po EV > 0)
    m.loc[~(m["close_at"] > m["ref_at"] + MATCH_TOL), "clv"] = np.nan

    out = {"n_lines": len(m), "same_line_share": round(float(m["same_line"].mean()), 3),
           "by_threshold": []}
    for t in EV_THRESHOLDS:
        s = m[m["ev"] > t]
        row = {"ev_threshold": t, "bets": len(s)}
        if len(s) >= 2:
            clv = s["clv"].dropna()
            row.update({
                "mean_ev_at_bet": round(float(s["ev"].mean()), 4),
                "n_with_later_close": len(clv),
                "mean_ev_same_bets": round(float(s.loc[clv.index, "ev"].mean()), 4) if len(clv) else None,
                "mean_clv": round(float(clv.mean()), 4) if len(clv) else None,
                "clv_ci95": _cluster_ci(clv.to_numpy(), s.loc[clv.index, "game_time"].astype(str).to_numpy(), rng)
                if len(clv) >= 2 else None,
            })
        out["by_threshold"].append(row)
    return out, m


def realized_roi(bets: pd.DataFrame, rng) -> list[dict]:
    """ROI zakładów z value_vs_pinnacle na wynikach (bets z kolumnami PTS, side, odds, ev)."""
    win = np.where(bets["side"] == "over", bets["PTS"] > bets["line"], bets["PTS"] < bets["line"])
    push = bets["PTS"] == bets["line"]
    pnl = np.where(push, 0.0, np.where(win, bets["odds"] - 1, -1.0))
    rows = []
    for t in EV_THRESHOLDS:
        sel = (bets["ev"] > t).to_numpy()
        n = int(sel.sum())
        row = {"ev_threshold": t, "bets": n}
        if n >= 2:
            row.update({
                "hit_rate": round(float(win[sel].mean()), 3),
                "roi": round(float(pnl[sel].mean()), 4),
                "roi_ci95": _cluster_ci(pnl[sel], bets.loc[sel, "game_time"].astype(str).to_numpy(), rng),
            })
        rows.append(row)
    return rows


def best_price(snaps: pd.DataFrame) -> dict:
    """Ostatni odczyt każdego bukmachera; grupy (zawodnik, mecz, linia) z ≥ 2 bukmacherami."""
    last = snaps.groupby(["player_name", "game_time", "bookmaker"]).tail(1)
    g = last.groupby(["player_name", "game_time", "line"])
    agg = g.agg(n_books=("bookmaker", "nunique"), best_over=("over_odds", "max"),
                best_under=("under_odds", "max"))
    agg = agg[agg["n_books"] >= 2]
    if agg.empty:
        return {"n_groups": 0}
    book_margin = (1 / last["over_odds"] + 1 / last["under_odds"] - 1)
    best_margin = 1 / agg["best_over"] + 1 / agg["best_under"] - 1
    return {
        "n_groups": len(agg),
        "books": sorted(last["bookmaker"].unique().tolist()),
        "margin_per_book": {b: round(float(v), 4) for b, v in book_margin.groupby(last["bookmaker"]).mean().items()},
        "margin_best_of": round(float(best_margin.mean()), 4),
        "arbitrage_count": int((best_margin < 0).sum()),
        "arbitrage_max_profit": round(float(-best_margin.min()), 4) if (best_margin < 0).any() else 0.0,
    }


def model_best_price(rng) -> dict | None:
    """PointsModel (base) obstawiany u jednego bukmachera vs po najlepszej cenie."""
    preds = sorted(REPORT_DIR.glob("eval_*_predictions.csv"))
    if not preds:
        return None
    p = pd.read_csv(preds[-1])
    if "p_base" not in p:
        return None
    p = p.dropna(subset=["PTS", "p_base"])
    p["y_over"] = (p["PTS"] > p["line"]).astype(int)
    ev = np.maximum(p["p_base"] * p["over_odds"] - 1, (1 - p["p_base"]) * p["under_odds"] - 1)
    best = p.assign(_ev=ev).sort_values("_ev").groupby(["PLAYER_ID", "game_date"]).tail(1)
    out = {b: simulate_bets(d, "p_base", rng) for b, d in p.groupby("bookmaker")}
    out["best_price"] = simulate_bets(best, "p_base", rng)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def run(db_path: str = DB_PATH) -> dict:
    rng = np.random.default_rng(0)
    snaps = load_snapshots(db_path)
    value, bets = value_vs_pinnacle(attach_reference(snaps), rng)

    games = load_dataset()
    matched, _ = attach_games(bets, games)
    value["realized"] = realized_roi(matched, rng) if len(matched) else []
    value["n_with_result"] = len(matched)

    report = {
        "generated": date.today().isoformat(),
        "reference": REFERENCE,
        "n_snapshots": len(snaps),
        "snapshots_per_book": snaps["bookmaker"].value_counts().to_dict(),
        "value_vs_pinnacle": value,
        "best_price": best_price(snaps),
        "model_best_price": model_best_price(rng),
        "note": ("EV i CLV względem uczciwej ceny Pinnacle; różne linie przeliczone przez N(μ, σ(μ)). "
                 "CI95 z bootstrapu po meczach. „Zamknięcie” = ostatni odczyt przed meczem."),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"line_shopping_{report['generated']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    logger.info("Raport: %s", path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=float))
