# nba_betting/prior_role.py
"""
Poprzedni sezon, wiek i zmiana roli jako cechy względem ceny Kalshi (PTS, REB, AST).

Cechy (point-in-time: poprzedni sezon jest zamknięty przed każdą linią bieżącego):
  prev_gap       — (średnia statystyki z poprzedniego sezonu − linia) / √linia; 0 bez ≥ MIN_PREV meczów
  prev_gap_early — prev_gap tylko w pierwszych EARLY_GAMES meczach sezonu (gdy bieżących danych jest mało)
  age            — wiek w sezonie − 27 (leaguedashplayerbiostats, jedno zapytanie na sezon)
  team_changed   — 1, gdy drużyna w meczu ≠ drużyna z największą liczbą meczów w poprzednim sezonie
  min_delta      — średnie minuty z 5 poprzednich meczów − średnie minuty poprzedniego sezonu (zmiana roli)

Ocena jak market_tests.py: logit p = α + β·logit p_Kalshi + γ·x, walk-forward tygodniowo, Δ log-loss
z CI z bootstrapu po meczach (Bonferroni: cechy × statystyki), selekcja na liniach sprzed FINAL_FROM.

Użycie:
  python prior_role.py            # → reports/prior_role_<data>.json
  python prior_role.py --final    # holdout, tylko gdy coś przeszło dev
"""
import argparse
import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from feature_lab import FINAL_FROM
from evaluate_stats import _cluster_ci
from history import RAW_DIR, TARGETS, load_dataset
from market_tests import evaluate
from points_model import train_points_model

logger = logging.getLogger(__name__)

FEATURES    = ("prev_gap", "prev_gap_early", "age", "team_changed", "min_delta")
STATS       = ("PTS", "REB", "AST")
MIN_PREV    = 10        # mecze w poprzednim sezonie, żeby jego średnia coś znaczyła
EARLY_GAMES = 20
AGE_CENTER  = 27


def bio_path(season: str):
    return RAW_DIR / f"bio_{season}.csv.gz"


def load_ages(seasons) -> pd.DataFrame:
    """PLAYER_ID, SEASON, AGE; brakujące sezony pobiera z stats.nba.com (lokalnie) i zapisuje w data/raw."""
    out = []
    for s in seasons:
        p = bio_path(s)
        if not p.exists():
            from nba_api.stats.endpoints import leaguedashplayerbiostats
            time.sleep(1.5)
            df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=s, per_mode_simple="PerGame", timeout=60).get_data_frames()[0]
            df[["PLAYER_ID", "AGE"]].assign(SEASON=s).to_csv(p, index=False, compression="gzip")
        out.append(pd.read_csv(p))
    return pd.concat(out, ignore_index=True)


def season_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Podsumowanie (zawodnik, sezon) przypisane do NASTĘPNEGO sezonu: średnie, minuty, główna drużyna."""
    played = data[data["MIN"] > 0]
    agg = played.groupby(["PLAYER_ID", "SEASON"]).agg(
        n_prev=("GAME_ID", "size"), prev_MIN=("MIN", "mean"),
        prev_PTS=("PTS", "mean"), prev_REB=("REB", "mean"), prev_AST=("AST", "mean")).reset_index()
    team = (played.groupby(["PLAYER_ID", "SEASON", "TEAM_ID"]).size().rename("n").reset_index()
            .sort_values("n").drop_duplicates(["PLAYER_ID", "SEASON"], keep="last")
            .rename(columns={"TEAM_ID": "prev_TEAM"})[["PLAYER_ID", "SEASON", "prev_TEAM"]])
    agg = agg.merge(team, on=["PLAYER_ID", "SEASON"])
    agg["SEASON"] = agg["SEASON"].map(lambda s: f"{int(s[:4]) + 1}-{(int(s[:4]) + 2) % 100:02d}")
    return agg


def add_features(pred: pd.DataFrame, data: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    d = data.sort_values("GAME_DATE")
    ctx = d[["GAME_ID", "PLAYER_ID", "SEASON", "TEAM_ID", "games_in_season", "min_avg_5"]].copy()
    ctx["GAME_ID"] = ctx["GAME_ID"].astype(str)
    m = pred.assign(GAME_ID=pred["GAME_ID"].astype(str)).merge(ctx, on=["GAME_ID", "PLAYER_ID"], how="left")
    m = m.merge(season_summary(data), on=["PLAYER_ID", "SEASON"], how="left")
    m = m.merge(ages, on=["PLAYER_ID", "SEASON"], how="left")
    has_prev = m["n_prev"].fillna(0) >= MIN_PREV
    prev = np.select([m["stat"] == st for st in STATS], [m[f"prev_{st}"] for st in STATS], np.nan)
    gap = (prev - m["line"]) / np.sqrt(m["line"].clip(lower=0.5))
    m["prev_gap"] = np.where(has_prev, gap, 0.0)
    m["prev_gap_early"] = np.where(m["games_in_season"].fillna(99) < EARLY_GAMES, m["prev_gap"], 0.0)
    m["age"] = (m["AGE"] - AGE_CENTER).fillna(0.0)
    m["team_changed"] = (has_prev & (m["TEAM_ID"] != m["prev_TEAM"])).astype(int)
    m["min_delta"] = np.where(has_prev, m["min_avg_5"] - m["prev_MIN"], 0.0)
    m["min_delta"] = m["min_delta"].fillna(0.0)
    m["lm"] = mm._logit(m["p_mid"])
    m.attrs["coverage"] = {"has_prev": round(float(has_prev.mean()), 3),
                           "age": round(float(m["AGE"].notna().mean()), 3),
                           "matched": round(float(m["SEASON"].notna().mean()), 3)}
    return m


PRIOR_COLS     = ["prev_stat", "prev_MIN", "has_prev", "team_changed"]
FORECAST_FROM  = "2022-23"     # 2021-22 nie ma poprzedniego sezonu w zbiorze


def forecast_check(data: pd.DataFrame, rng) -> dict:
    """Bez rynku: czy poprzedni sezon zmniejsza MAE prognozy w pierwszych EARLY_GAMES meczach sezonu?
    Model trenowany na meczach sprzed startu sezonu (bez danych z sezonu), base vs base + PRIOR_COLS."""
    d = data.merge(season_summary(data), on=["PLAYER_ID", "SEASON"], how="left")
    d["has_prev"] = (d["n_prev"].fillna(0) >= MIN_PREV).astype(int)
    d["team_changed"] = (d["has_prev"].astype(bool) & (d["TEAM_ID"] != d["prev_TEAM"])).astype(int)
    out = {}
    for st in STATS:
        d["prev_stat"] = d[f"prev_{st}"].where(d["has_prev"] == 1)
        d["prev_MIN"] = d["prev_MIN"].where(d["has_prev"] == 1)
        base = list(TARGETS[st][0])
        res = {}
        for season in sorted(s for s in d["SEASON"].unique() if s >= FORECAST_FROM):
            start = d.loc[d["SEASON"] == season, "GAME_DATE"].min()
            test = d[(d["SEASON"] == season) & (d["games_in_season"] < EARLY_GAMES) & (d["MIN"] > 0)
                     & (d["IS_PLAYOFFS"] == 0)]
            err = {}
            for name, cols in (("base", base), ("base+prior", base + PRIOR_COLS)):
                m = train_points_model(d, cutoff=start, features=cols, target=st)
                err[name] = np.abs(test[st].to_numpy() - m.mu(test))
            diff = err["base+prior"] - err["base"]
            res[season] = {"n": int(len(test)), "mae_base": round(float(err["base"].mean()), 4),
                           "mae_prior": round(float(err["base+prior"].mean()), 4),
                           "delta": round(float(diff.mean()), 4),
                           "ci95": _cluster_ci(diff, test["GAME_ID"].astype(str).to_numpy(), rng)}
        out[st] = res
        logger.info("forecast %s: %s", st, {k: (v["delta"], v["ci95"]) for k, v in res.items()})
    return out


def run(final: bool = False) -> dict:
    rng = np.random.default_rng(0)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    pred = pred[pred["stat"].isin(STATS)]
    data = load_dataset()
    m = add_features(pred, data, load_ages(sorted(data["SEASON"].unique())[-2:]))
    dev = m[m["game_date"] < FINAL_FROM]
    report = {"generated": date.today().isoformat(), "sample": "final" if final else "dev",
              "final_from": str(FINAL_FROM.date()), "predictions": mm.latest_predictions().name,
              "coverage": m.attrs["coverage"],
              "dev_share": {"team_changed": round(float(dev["team_changed"].mean()), 3),
                            "early": round(float((dev["prev_gap_early"] != 0).mean()), 3)},
              **evaluate(m, final, FEATURES, rng),
              "forecast_early_season": forecast_check(data, rng)}
    out = mm.REPORT_DIR / f"prior_role_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    run(ap.parse_args().final)
