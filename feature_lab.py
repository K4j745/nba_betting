# nba_betting/feature_lab.py
"""
Laboratorium cech: jeden sposób oceny nowej grupy cech dla PTS / REB / AST.

  1. Holdout 2025-26 (trening do 2025-10-01): MAE względem base + absencje, sparowane CI po meczach,
     osobno dla top 7 drużyny wg minut (zakres projektu).
  2. foul_baiter: czy profil przewiduje rzuty osobiste (model FTA) i tabela baiter × agresywny rywal.
  3. Kalshi, walk-forward: Δ log-loss vs rynek i korekta rynku (γ, poza próbą).
  4. --scope: na top 7 trening tylko na top 7 oraz model dwuczęściowy minuty × produkcja na 36 min.

Zasady (wiele cech = fałszywe odkrycia):
  - absencje wyłącznie w punkcie 1; w punkcie 3 bez faktycznych absencji (wyciek),
  - próg istotności 0,05 / liczba testów; CI w raporcie jest zwykłe 95%, p-wartości nie liczymy,
  - linie od FINAL_FROM zostają nietknięte; --final dopiero dla cech, które przeszły dev.

Użycie:
  python feature_lab.py            # → reports/feature_lab_<data>.json (część dev)
  python feature_lab.py --final
  python feature_lab.py --scope    # → reports/feature_scope_<data>.json
"""
import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from evaluate import attach_games
from evaluate_stats import _cluster_ci, _logloss, walk_forward
from history import DECAY_FEATURE_COLS, FOUL_FEATURE_COLS, TARGETS, load_dataset
from kalshi import load_history, main_lines
from points_model import train_points_model

logger = logging.getLogger(__name__)

HOLDOUT_FROM = "2025-10-01"
FINAL_FROM   = pd.Timestamp("2026-04-01")
TOP_N        = 7
MIN_FOR_RATE = 10      # produkcja z krótszych występów to głównie szum


def groups(stat: str) -> dict[str, list[str]]:
    return {"foul": FOUL_FEATURE_COLS, "decay": DECAY_FEATURE_COLS[stat]}


def add_top_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Miejsce w drużynie wg minut z ostatnich 10 meczów (sprzed meczu)."""
    return df.assign(min_rank=df.groupby(["GAME_ID", "TEAM_ID"])["min_avg_10"].rank(ascending=False, method="first"))


def holdout(data: pd.DataFrame, stat: str, rng) -> dict:
    test = add_top_rank(data[(data["GAME_DATE"] >= HOLDOUT_FROM) & data[stat].notna() & (data["MIN"] > 0)])
    y, games = test[stat].to_numpy(), test["GAME_ID"].astype(str).to_numpy()
    top = (test["min_rank"] <= TOP_N).to_numpy()
    base, absn = TARGETS[stat]
    ref_cols = base + absn
    err = lambda cols: np.abs(train_points_model(data, HOLDOUT_FROM, features=cols, target=stat).mu(test) - y)
    ref = err(ref_cols)
    out = {"n": int(len(test)), "n_top7": int(top.sum()), "mae_ref": round(float(ref.mean()), 4),
           "mae_ref_top7": round(float(ref[top].mean()), 4), "sets": {}}
    for name, cols in groups(stat).items():
        d = err(ref_cols + cols) - ref
        out["sets"][name] = {"delta_mae": round(float(d.mean()), 4), "ci95": _cluster_ci(d, games, rng),
                             "delta_mae_top7": round(float(d[top].mean()), 4),
                             "ci95_top7": _cluster_ci(d[top], games[top], rng)}
    logger.info("%s holdout: %s", stat, out["sets"])
    return out


def foul_profile(data: pd.DataFrame, rng) -> dict:
    """Czy foul_baiter × agresywny rywal przewiduje FTA i punkty ponad model bazowy?"""
    test = add_top_rank(data[(data["GAME_DATE"] >= HOLDOUT_FROM) & data["FTA"].notna() & (data["MIN"] > 0)])
    test = test[test["min_rank"] <= TOP_N].dropna(subset=["fta_per36_10", "opp_fta_allowed"])
    base = TARGETS["PTS"][0]
    y, games = test["FTA"].to_numpy(), test["GAME_ID"].astype(str).to_numpy()
    mu0 = train_points_model(data, HOLDOUT_FROM, features=base, target="FTA").mu(test)
    mu1 = train_points_model(data, HOLDOUT_FROM, features=base + FOUL_FEATURE_COLS, target="FTA").mu(test)
    pts = train_points_model(data, HOLDOUT_FROM, features=base, target="PTS").mu(test)
    d = np.abs(mu1 - y) - np.abs(mu0 - y)
    baiter = test["fta_per36_10"] >= test["fta_per36_10"].quantile(0.75)
    aggr = test["opp_fta_allowed"] >= test["opp_fta_allowed"].quantile(0.75)
    t = test.assign(res_fta=y - mu0, res_pts=test["PTS"].to_numpy() - pts, baiter=baiter, aggressive=aggr)
    table = []
    for (b, a), g in t.groupby(["baiter", "aggressive"]):
        table.append({"baiter": bool(b), "aggressive_opp": bool(a), "n": int(len(g)),
                      "fta_mean": round(float(g["FTA"].mean()), 2),
                      "res_fta": round(float(g["res_fta"].mean()), 3),
                      "res_fta_ci95": _cluster_ci(g["res_fta"].to_numpy(), g["GAME_ID"].astype(str).to_numpy(), rng),
                      "res_pts": round(float(g["res_pts"].mean()), 3),
                      "res_pts_ci95": _cluster_ci(g["res_pts"].to_numpy(), g["GAME_ID"].astype(str).to_numpy(), rng)})
    return {"n": int(len(test)), "fta_delta_mae": round(float(d.mean()), 4), "fta_ci95": _cluster_ci(d, games, rng),
            "baiter_min_fta36": round(float(test["fta_per36_10"].quantile(0.75)), 2),
            "aggressive_min_fta100": round(float(test["opp_fta_allowed"].quantile(0.75)), 2), "table": table}


def vs_kalshi(data: pd.DataFrame, hist: pd.DataFrame, stat: str, final: bool, rng) -> dict:
    lines = main_lines(hist[hist["stat"] == stat]).rename(columns={"player": "player_name"}).assign(bookmaker="kalshi")
    m, _ = attach_games(lines.drop(columns=["name_key"]), data)
    m["y_over"] = (m[stat] > m["line"]).astype(int)
    base = TARGETS[stat][0]
    sets = {"base": base, **{k: base + v for k, v in groups(stat).items()}}
    pred = walk_forward(m, data, stat, sets=sets).reset_index(drop=True)
    pred["lm"] = mm._logit(pred["p_mid"])
    res = {"sample": "final" if final else f"dev (< {FINAL_FROM.date()})"}
    keep = (pred["game_date"] >= FINAL_FROM) if final else (pred["game_date"] < FINAL_FROM)
    for name in sets:
        pred["d"] = mm._logit(pred[f"p_{name}"]) - pred["lm"]
        p, _ = mm.walk_forward(pred, ["lm", "d"])          # historia zawsze ze wszystkiego sprzed tygodnia
        ok = keep.to_numpy() & ~np.isnan(p)
        s = pred[ok]
        y, games = s["y_over"].to_numpy(), s["GAME_ID"].astype(str).to_numpy()
        ll_m = _logloss(s["p_mid"].to_numpy(), y)
        raw, corr = _logloss(s[f"p_{name}"].to_numpy(), y) - ll_m, _logloss(p[ok], y) - ll_m
        res[name] = {"n": int(ok.sum()), "raw_diff": round(float(raw.mean()), 4), "raw_ci95": _cluster_ci(raw, games, rng),
                     "corrected_diff": round(float(corr.mean()), 4), "corrected_ci95": _cluster_ci(corr, games, rng),
                     "gamma": mm.full_fit_ci(pred[keep], ["lm", "d"], rng, n_boot=200)["d"]}
    logger.info("%s Kalshi: %s", stat, res)
    return res


def scope(data: pd.DataFrame, stat: str, rng) -> dict:
    """MAE na top 7 (holdout): trening na wszystkich vs tylko top 7 vs minuty × produkcja."""
    base, absn = TARGETS[stat]
    cols = base + absn + DECAY_FEATURE_COLS[stat]
    ranked = add_top_rank(data[data["MIN"] > 0]).assign(rate=lambda d: 36 * d[stat] / d["MIN"])
    test = ranked[(ranked["GAME_DATE"] >= HOLDOUT_FROM) & (ranked["min_rank"] <= TOP_N) & ranked[stat].notna()]
    y, games = test[stat].to_numpy(), test["GAME_ID"].astype(str).to_numpy()
    fit = lambda d, target: train_points_model(d, HOLDOUT_FROM, features=cols, target=target).mu(test)
    ref = np.abs(fit(ranked, stat) - y)
    two_part = fit(ranked, "MIN") * fit(ranked[ranked["MIN"] >= MIN_FOR_RATE], "rate") / 36
    out = {"n": int(len(test)), "mae_all_rows": round(float(ref.mean()), 4)}
    for name, mu in {"train_top7": fit(ranked[ranked["min_rank"] <= TOP_N], stat), "minutes_x_rate": two_part}.items():
        d = np.abs(mu - y) - ref
        out[name] = {"delta_mae": round(float(d.mean()), 4), "ci95": _cluster_ci(d, games, rng)}
    logger.info("%s zakres: %s", stat, out)
    return out


def run_scope() -> dict:
    rng = np.random.default_rng(0)
    data = load_dataset()
    report = {"generated": date.today().isoformat(), **{s: scope(data, s, rng) for s in TARGETS}}
    out = mm.REPORT_DIR / f"feature_scope_{report['generated']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


def run(final: bool = False) -> dict:
    rng = np.random.default_rng(0)
    data, hist = load_dataset(), load_history()
    report = {"generated": date.today().isoformat(), "final_from": str(FINAL_FROM.date()),
              "foul_profile": foul_profile(data, rng),
              "holdout": {s: holdout(data, s, rng) for s in TARGETS},
              "kalshi": {s: vs_kalshi(data, hist, s, final, rng) for s in TARGETS}}
    out = mm.REPORT_DIR / f"feature_lab_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("points_model").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--scope", action="store_true")
    args = ap.parse_args()
    run_scope() if args.scope else run(args.final)
