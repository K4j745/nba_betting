# nba_betting/count_dist.py
"""
Test rozkładu zliczeń dla zbiórek i asyst: czy inny rozkład wokół tego samego μ
daje lepsze P(over) niż obecny ujemny dwumianowy z σ(μ) = a + b·μ.

μ = mu_base z evaluate_stats (walk-forward tygodniowo). Warianty różnią się tylko
rozkładem; ich parametry dopasowujemy na liniach Kalshi z wcześniejszych tygodni
(rozszerzające się okno, min. MIN_TRAIN linii), więc nic nie widzi przyszłości:

  current   — p_base z modelu (NB, σ(μ) z walidacji modelu): punkt odniesienia
  normal    — N(μ, σ(μ)), σ(μ) = a + b·μ z reszt wcześniejszych linii
  poisson   — Poisson(μ)
  nb_sigma  — NB o wariancji σ(μ)² dopasowanej jak w normal (obecna metoda, inne dane)
  nb_const  — NB o stałym r: Var = μ + μ²/r, r z największej wiarygodności
  nb_player — NB o wariancji D·μ, D = dyspersja zawodnika z 30 poprzednich meczów,
              ściągnięta do średniej ligi (n0 = 20) i przeskalowana do poziomu z wcześniejszych linii

Ocena na liniach sprzed FINAL_FROM (dev):
  - log-loss i kalibracja względem wyniku (lepsza prognoza?),
  - Δ log-loss vs current, CI z bootstrapu po meczach z poprawką Bonferroniego,
  - korekta rynku: logit p = α + β·lm + γ·(logit p_wariant − lm) vs to samo z p_base
    (czy lepszy rozkład daje przewagę nad ceną Kalshi?).
Holdout (--final) tylko dla wariantów, które przejdą dev.

Użycie:
  python count_dist.py            # → reports/count_dist_<data>.json
  python count_dist.py --final
"""
import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom, norm, poisson

import market_model as mm
from evaluate_stats import _logloss
from feature_lab import FINAL_FROM
from history import load_dataset
from market_tests import game_ci
from points_model import _fit_sigma, p_over_count

logger = logging.getLogger(__name__)

STATS     = ("REB", "AST")
VARIANTS  = ("normal", "poisson", "nb_sigma", "nb_const", "nb_player")
MIN_TRAIN = 400
PLAYER_N  = 30      # mecze do dyspersji zawodnika
SHRINK_N0 = 20      # siła ściągania do średniej ligi
MU_FLOOR  = 0.05


def _clip_p(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def nb_sf(k, mu, r):
    """P(Y > k) dla NB o średniej μ i parametrze r (Var = μ + μ²/r)."""
    return nbinom.sf(k, r, r / (r + mu))


def fit_nb_r(y: np.ndarray, mu: np.ndarray) -> float:
    """r z największej wiarygodności przy znanym μ (szukanie po log r)."""
    mu = np.clip(mu, MU_FLOOR, None)

    def nll(log_r):
        r = np.exp(log_r)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
        return -ll.sum()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 8), method="bounded").x))


def player_dispersion(pred: pd.DataFrame, data: pd.DataFrame, stat: str) -> pd.Series:
    """Wariancja / średnia statystyki zawodnika z PLAYER_N meczów sprzed dnia linii (NaN, gdy < 5 meczów)."""
    g = data[["PLAYER_ID", "GAME_DATE", stat]].dropna().sort_values("GAME_DATE")
    roll = g.groupby("PLAYER_ID")[stat].rolling(PLAYER_N, min_periods=5)
    g = g.assign(_var=roll.var().reset_index(level=0, drop=True),
                 _mean=roll.mean().reset_index(level=0, drop=True),
                 _n=roll.count().reset_index(level=0, drop=True))
    g = g[g["_mean"] > 0]
    left = pred[["PLAYER_ID", "game_date"]].astype({"PLAYER_ID": "int64"}).reset_index().sort_values("game_date")
    g = g.astype({"PLAYER_ID": "int64"})
    m = pd.merge_asof(left, g.rename(columns={"GAME_DATE": "game_date"})[["PLAYER_ID", "game_date", "_var", "_mean", "_n"]],
                      on="game_date", by="PLAYER_ID", allow_exact_matches=False)
    m = m.set_index("index").reindex(pred.index)
    return pd.DataFrame({"D": m["_var"] / m["_mean"], "n": m["_n"]}, index=pred.index)


def variant_probs(d: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Kolumny p_<wariant>; parametry z linii sprzed tygodnia (NaN, dopóki < MIN_TRAIN)."""
    d = d.sort_values("game_date").reset_index(drop=True)
    week = d["game_date"] - pd.to_timedelta(d["game_date"].dt.weekday, unit="D")
    mu = np.clip(d["mu_base"].to_numpy(dtype=float), MU_FLOOR, None)
    y = d[stat].to_numpy(dtype=float)
    k = np.floor(d["line"].to_numpy(dtype=float))
    out = {f"p_{v}": np.full(len(d), np.nan) for v in VARIANTS}
    params = []
    for w in week.unique():
        cur, past = (week == w).to_numpy(), (d["game_date"] < w).to_numpy()
        if past.sum() < MIN_TRAIN:
            continue
        a, b = _fit_sigma(mu[past], y[past] - mu[past])
        sig = np.clip(a + b * mu[cur], 0.3, None)
        r = fit_nb_r(y[past], mu[past])
        # nb_player: D ściągnięte do średniej ligi, potem skala dopasowana na wcześniejszych liniach
        Dp, n = d["D"].to_numpy(), d["n"].fillna(0).to_numpy()
        d_league = np.nanmean(Dp[past])
        Ds = np.where(np.isnan(Dp), d_league, (n * np.nan_to_num(Dp) + SHRINK_N0 * d_league) / (n + SHRINK_N0))
        scale = np.sum((y[past] - mu[past]) ** 2) / np.sum(Ds[past] * mu[past])
        var_p = np.maximum(scale * Ds[cur] * mu[cur], mu[cur] * 1.001)

        out["p_normal"][cur] = 1 - norm.cdf((d["line"].to_numpy()[cur] - mu[cur]) / sig)
        out["p_poisson"][cur] = poisson.sf(k[cur], mu[cur])
        out["p_nb_sigma"][cur] = p_over_count(mu[cur], sig, k[cur] + 0.5)
        out["p_nb_const"][cur] = nb_sf(k[cur], mu[cur], r)
        out["p_nb_player"][cur] = p_over_count(mu[cur], np.sqrt(var_p), k[cur] + 0.5)
        params.append({"week": str(pd.Timestamp(w).date()), "sigma_a": round(a, 3), "sigma_b": round(b, 3),
                       "nb_r": round(r, 2), "player_scale": round(float(scale), 3)})
    for c, v in out.items():
        d[c] = _clip_p(v)
    d.attrs["params"] = params
    return d


def calibration_error(p: np.ndarray, y: np.ndarray) -> float:
    """Średni ważony |średnie p − odsetek over| w 10 kwantylach p."""
    bins = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())


def evaluate(d: pd.DataFrame, final: bool, rng, level: float, names=VARIANTS) -> dict:
    cols = ["p_mid", "y_over", "p_base"] + [f"p_{v}" for v in names]     # bez tygodni rozgrzewki
    d = mm.add_terms(d.dropna(subset=cols).sort_values("game_date").reset_index(drop=True))
    keep = ((d["game_date"] >= FINAL_FROM) if final else (d["game_date"] < FINAL_FROM)).to_numpy()
    y, games = d["y_over"].to_numpy(), d["GAME_ID"].to_numpy()
    ref_mkt, _ = mm.walk_forward(d, ["lm", "d_base"])
    ok_all = keep & ~np.isnan(ref_mkt)
    res = {"n": int(ok_all.sum()), "n_games": int(pd.Series(games[ok_all]).nunique()),
           "current": {"logloss": round(float(_logloss(d["p_base"].to_numpy()[ok_all], y[ok_all]).mean()), 5),
                       "calib_err": round(calibration_error(d["p_base"].to_numpy()[ok_all], y[ok_all]), 4),
                       "mean_p": round(float(d["p_base"].to_numpy()[ok_all].mean()), 3)},
           "market": {"logloss": round(float(_logloss(d["p_mid"].to_numpy()[ok_all], y[ok_all]).mean()), 5)},
           "over_rate": round(float(y[ok_all].mean()), 3), "variants": {}}
    for v in names:
        p = d[f"p_{v}"].to_numpy()
        diff = _logloss(p[ok_all], y[ok_all]) - _logloss(d["p_base"].to_numpy()[ok_all], y[ok_all])
        ci = game_ci(diff, games[ok_all], rng, level)
        d[f"d_{v}"] = mm._logit(p) - d["lm"]
        pm, _ = mm.walk_forward(d, ["lm", f"d_{v}"])
        ok = ok_all & ~np.isnan(pm)
        dm = _logloss(pm[ok], y[ok]) - _logloss(ref_mkt[ok], y[ok])
        ci_m = game_ci(dm, games[ok], rng, level)
        res["variants"][v] = {
            "logloss": round(float(_logloss(p[ok_all], y[ok_all]).mean()), 5),
            "calib_err": round(calibration_error(p[ok_all], y[ok_all]), 4),
            "mean_p": round(float(p[ok_all].mean()), 3),
            "vs_current": {"delta_logloss": round(float(diff.mean()), 5), "ci": ci, "better": ci[1] < 0},
            "vs_market_corrected": {"delta_logloss": round(float(dm.mean()), 5), "ci": ci_m, "passed": ci_m[1] < 0},
        }
    return res


def run(final: bool = False) -> dict:
    rng = np.random.default_rng(0)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    data = load_dataset()
    level = 1 - 0.05 / (len(VARIANTS) * len(STATS))
    report = {"generated": date.today().isoformat(), "sample": "final" if final else "dev",
              "final_from": str(FINAL_FROM.date()), "ci_level": round(level, 4),
              "predictions": mm.latest_predictions().name, "stats": {}}
    for stat in STATS:
        d = pred[pred["stat"] == stat].dropna(subset=[stat, "mu_base"]).copy()
        d = d.join(player_dispersion(d, data, stat))
        d = variant_probs(d, stat)
        report["stats"][stat] = {**evaluate(d, final, rng, level), "params_last": d.attrs["params"][-1:]}
        logger.info("%s: %s", stat, {v: (r["vs_current"]["delta_logloss"], r["vs_market_corrected"]["delta_logloss"])
                                     for v, r in report["stats"][stat]["variants"].items()})
    out = mm.REPORT_DIR / f"count_dist_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    run(ap.parse_args().final)
