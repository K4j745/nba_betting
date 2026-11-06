# nba_betting/market_tests.py
"""
Pakiet tanich testów: czy prosta cecha przesuwa cenę Kalshi 1 h przed meczem?

    logit p = α + β·logit p_Kalshi + δ·x          (walk-forward, market_model)

Każdą cechę oceniamy osobno, względem samej rekalibracji rynku (bez x):
  - last_dev   odchylenie ostatniego meczu od średniej z 10, w odchyleniach std (przereakcja: δ < 0),
  - east_home  gospodarz po podróży na wschód, liczba stref (jet lag),
  - early_season  ≤ EARLY_GAMES meczów zawodnika w sezonie,
  - early_tip  start przed EARLY_TIP_ET czasu wschodniego.

Progi ustalone przed spojrzeniem na wyniki. Wiele testów: przechodzi tylko cecha, której CI Δ log-loss
na poziomie 1 − 0,05/liczba testów jest całe poniżej zera. Linie od FINAL_FROM zostają na
potwierdzenie (--final), tylko dla cech, które przeszły.

Użycie:
  python market_tests.py            # → reports/market_tests_<data>.json
  python market_tests.py --final
"""
import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from evaluate_stats import _logloss
from feature_lab import FINAL_FROM
from history import TARGETS, load_dataset

logger = logging.getLogger(__name__)

FEATURES     = ("last_dev", "east_home", "early_season", "early_tip")
EARLY_GAMES  = 20
EARLY_TIP_ET = 17
N_BOOT       = 5000


def add_features(pred: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """pred: predykcje eval_stats (jedna statystyka lub wiele), data: zbiór z history.py."""
    d = data.sort_values("GAME_DATE")
    ctx = d[["GAME_ID", "PLAYER_ID", "IS_HOME", "tz_shift", "games_in_season"]].copy()
    for st in TARGETS:
        prev = d.groupby("PLAYER_ID")[st].shift(1)
        roll = prev.groupby(d["PLAYER_ID"]).rolling(10, min_periods=5)
        mean, std = (getattr(roll, f)().reset_index(level=0, drop=True) for f in ("mean", "std"))
        ctx[f"dev_{st}"] = ((prev - mean) / std.where(std > 0)).clip(-3, 3)
    ctx["GAME_ID"] = ctx["GAME_ID"].astype(str)
    m = pred.assign(GAME_ID=pred["GAME_ID"].astype(str)).merge(ctx, on=["GAME_ID", "PLAYER_ID"], how="left")
    m["last_dev"] = np.select([m["stat"] == st for st in TARGETS], [m[f"dev_{st}"] for st in TARGETS], np.nan)
    m["last_dev"] = m["last_dev"].fillna(0.0)
    m["east_home"] = m["tz_shift"].clip(lower=0).fillna(0) * m["IS_HOME"].fillna(0)
    m["early_season"] = (m["games_in_season"] <= EARLY_GAMES).astype(int)
    tip = pd.to_datetime(m["tip"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    m["early_tip"] = (tip.dt.hour < EARLY_TIP_ET).astype(int)
    m["lm"] = mm._logit(m["p_mid"])
    return m


def game_ci(values: np.ndarray, games: np.ndarray, rng, level: float) -> list[float]:
    """CI średniej z bootstrapu po meczach, na dowolnym poziomie."""
    ug, inv = np.unique(games, return_inverse=True)
    sums, counts = np.bincount(inv, values), np.bincount(inv)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=N_BOOT)
    means = (w @ sums) / (w @ counts)
    a = (1 - level) / 2 * 100
    return [round(float(x), 5) for x in np.percentile(means, (a, 100 - a))]


def evaluate(m: pd.DataFrame, final: bool, names: tuple, rng) -> dict:
    level = 1 - 0.05 / (len(names) * m["stat"].nunique())
    out = {"ci_level": round(level, 4)}
    for stat, d in m.groupby("stat"):
        d = d.dropna(subset=["p_mid", "y_over"]).sort_values("game_date").reset_index(drop=True)
        keep = ((d["game_date"] >= FINAL_FROM) if final else (d["game_date"] < FINAL_FROM)).to_numpy()
        ref, _ = mm.walk_forward(d, ["lm"])
        y, games = d["y_over"].to_numpy(), d["GAME_ID"].to_numpy()
        res = {}
        for x in names:
            p, _ = mm.walk_forward(d, ["lm", x])
            ok = keep & ~np.isnan(p) & ~np.isnan(ref)
            diff = _logloss(p[ok], y[ok]) - _logloss(ref[ok], y[ok])
            ci = game_ci(diff, games[ok], rng, level)
            res[x] = {"n": int(ok.sum()), "share_nonzero": round(float((d.loc[ok, x] != 0).mean()), 3),
                      "delta_logloss": round(float(diff.mean()), 5), "ci": ci, "passed": ci[1] < 0,
                      "delta_coef": mm.full_fit_ci(d[keep], ["lm", x], rng, n_boot=200)[x]}
        out[stat] = res
        logger.info("%s: %s", stat, {k: (v["delta_logloss"], v["ci"], v["delta_coef"]["coef"]) for k, v in res.items()})
    return out


def run(final: bool = False, names: tuple = FEATURES) -> dict:
    rng = np.random.default_rng(0)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    m = add_features(pred, load_dataset())
    report = {"generated": date.today().isoformat(), "sample": "final" if final else "dev",
              "final_from": str(FINAL_FROM.date()), **evaluate(m, final, names, rng)}
    out = mm.REPORT_DIR / f"market_tests_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    run(ap.parse_args().final)
