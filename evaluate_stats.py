# nba_betting/evaluate_stats.py
"""
Ewaluacja modeli PTS / REB / AST względem cen Kalshi (darmowa historia od 12.2025).

Dla każdej statystyki:
  - linie: główna linia Kalshi na zawodnika i mecz (cena najbliższa 0,5), cena = środek
    bid/ask ≥ 1 h przed meczem (kalshi.py → data/raw/kalshi_lines.csv.gz),
  - model: PointsModel(target=stat) trenowany TYGODNIOWO na meczach sprzed tygodnia
    (walk-forward; tygodniowo zamiast dziennie, bo dni z liniami jest ~150),
  - zestawy cech: base i base+absence (absencje = rzeczywiste, jak w treningu — górna granica
    tego, co da raport kontuzji),
  - log-loss / Brier model vs rynek (CI z bootstrapu po meczach), MAE,
  - symulacja: kupno YES po ask albo NO po (1 − bid) z prowizją Kalshi, gdy EV > próg.

Użycie:
  python evaluate_stats.py          # → reports/eval_stats_<data>.json + _predictions.csv
"""
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import attach_games
from history import TARGETS, load_dataset
from kalshi import kalshi_fee, load_history, main_lines
from points_model import train_points_model

logger = logging.getLogger(__name__)

REPORT_DIR    = Path("reports")
EV_THRESHOLDS = (0.0, 0.03, 0.05, 0.10)
N_BOOT        = 2000
EPS           = 1e-6


def _logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _cluster_ci(values: np.ndarray, clusters: np.ndarray, rng) -> list[float]:
    uc, inv = np.unique(clusters, return_inverse=True)
    s, c = np.bincount(inv, values, len(uc)), np.bincount(inv, minlength=len(uc))
    b = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        b.append((w @ s) / (w @ c))
    return [round(float(np.percentile(b, q)), 4) for q in (2.5, 97.5)]


def walk_forward(lines: pd.DataFrame, data: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Kolumny mu_<set>, p_<set> dla każdej linii; model na tydzień (cutoff = poniedziałek)."""
    base, absn = TARGETS[stat]
    sets = {"base": base, "absence": base + absn}
    lines = lines.copy()
    lines["week"] = lines["game_date"] - pd.to_timedelta(lines["game_date"].dt.weekday, unit="D")
    out = []
    for week, chunk in lines.groupby("week"):
        chunk = chunk.copy()
        for name, cols in sets.items():
            m = train_points_model(data, cutoff=week, features=cols, target=stat)
            chunk[f"mu_{name}"] = m.mu(chunk)
            chunk[f"p_{name}"] = m.p_over(chunk, chunk["line"].to_numpy())
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


def simulate(df: pd.DataFrame, p_col: str, rng) -> list[dict]:
    """1 kontrakt na stronę z wyższym EV: YES po ask (over) albo NO po 1 − bid (under), z prowizją."""
    p = df[p_col].to_numpy()
    ask, bid = df["ask"].to_numpy(), df["bid"].to_numpy()
    fee_y = np.array([kalshi_fee(a) for a in ask])
    fee_n = np.array([kalshi_fee(1 - b) for b in bid])
    cost_y, cost_n = ask + fee_y, (1 - bid) + fee_n
    ev_y, ev_n = p / cost_y - 1, (1 - p) / cost_n - 1
    yes = ev_y >= ev_n
    ev = np.where(yes, ev_y, ev_n)
    cost = np.where(yes, cost_y, cost_n)
    win = np.where(yes, df["y_over"] == 1, df["y_over"] == 0)
    pnl = (win.astype(float) - cost) / cost                 # zwrot na zainwestowaną kwotę
    games = df["GAME_ID"].astype(str).to_numpy()
    rows = []
    for t in EV_THRESHOLDS:
        sel = ev > t
        n = int(sel.sum())
        row = {"ev_threshold": t, "bets": n}
        if n >= 2:
            row.update({"hit_rate": round(float(win[sel].mean()), 3),
                        "roi": round(float(pnl[sel].mean()), 4),
                        "roi_ci95": _cluster_ci(pnl[sel], games[sel], rng)})
        rows.append(row)
    return rows


def score(df: pd.DataFrame, p_col: str, mu_col: str, stat: str, rng) -> dict:
    y = df["y_over"].to_numpy()
    games = df["GAME_ID"].astype(str).to_numpy()
    diff = _logloss(df[p_col].to_numpy(), y) - _logloss(df["p_mid"].to_numpy(), y)
    return {
        "n": int(len(df)), "n_games": int(df["GAME_ID"].nunique()),
        "over_rate": round(float(y.mean()), 3),
        "market_p_mean": round(float(df["p_mid"].mean()), 3),
        "logloss_model": round(float(_logloss(df[p_col].to_numpy(), y).mean()), 4),
        "logloss_market": round(float(_logloss(df["p_mid"].to_numpy(), y).mean()), 4),
        "logloss_diff": round(float(diff.mean()), 4),
        "logloss_diff_ci95": _cluster_ci(diff, games, rng),
        "brier_model": round(float(((df[p_col] - y) ** 2).mean()), 4),
        "brier_market": round(float(((df["p_mid"] - y) ** 2).mean()), 4),
        "mae_model": round(float((df[stat] - df[mu_col]).abs().mean()), 3),
        "mae_line": round(float((df[stat] - df["line"]).abs().mean()), 3),
    }


def calibration(df: pd.DataFrame, p_col: str) -> list[dict]:
    bins = pd.cut(df[p_col], [0, .3, .4, .45, .5, .55, .6, .7, 1])
    g = df.groupby(bins, observed=True).agg(p=(p_col, "mean"), market=("p_mid", "mean"),
                                            actual=("y_over", "mean"), n=("y_over", "size"))
    return [{"bin": str(k), **{c: round(float(v), 3) for c, v in r.items()}} for k, r in g.iterrows()]


def run() -> dict:
    rng = np.random.default_rng(0)
    hist = load_history()
    if hist.empty:
        raise FileNotFoundError("Brak data/raw/kalshi_lines.csv.gz — uruchom: python kalshi.py --backfill")
    data = load_dataset()
    report = {"generated": date.today().isoformat(), "source": "kalshi (środek bid/ask ≥ 1 h przed meczem)",
              "stats": {}}
    preds = []
    for stat in ("PTS", "REB", "AST"):
        lines = main_lines(hist[hist["stat"] == stat]).rename(columns={"player": "player_name"})
        matched, unmatched = attach_games(lines.drop(columns=["name_key"]), data)
        matched["y_over"] = (matched[stat] > matched["line"]).astype(int)
        logger.info("%s: %d linii, dopasowane %d, niedopasowane %d", stat, len(lines), len(matched), len(unmatched))
        pred = walk_forward(matched, data, stat)
        preds.append(pred.assign(stat=stat))
        report["stats"][stat] = {
            "lines": len(lines), "matched": len(matched),
            "dates": [str(pred["game_date"].min().date()), str(pred["game_date"].max().date())],
            "sets": {s: {"scores": score(pred, f"p_{s}", f"mu_{s}", stat, rng),
                         "bets": simulate(pred, f"p_{s}", rng)} for s in ("base", "absence")},
            "calibration_base": calibration(pred, "p_base"),
        }
    REPORT_DIR.mkdir(exist_ok=True)
    stem = REPORT_DIR / f"eval_stats_{report['generated']}"
    Path(f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    cols = ["stat", "game_date", "player_name", "GAME_ID", "PLAYER_ID", "line", "bid", "ask", "p_mid",
            "mu_base", "p_base", "mu_absence", "p_absence", "y_over", "PTS", "REB", "AST"]
    pd.concat(preds)[cols].to_csv(f"{stem}_predictions.csv", index=False)
    logger.info("Raport: %s.json", stem)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    r = run()
    for st, v in r["stats"].items():
        for s, d in v["sets"].items():
            sc = d["scores"]
            print(f"{st} {s}: n={sc['n']} Δlog-loss={sc['logloss_diff']:+.4f} CI {sc['logloss_diff_ci95']} "
                  f"MAE {sc['mae_model']} vs linia {sc['mae_line']}")
