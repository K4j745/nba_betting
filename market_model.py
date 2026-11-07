# nba_betting/market_model.py
"""
Model „od rynku”: zaczynamy od ceny rynku i uczymy się tylko KOREKTY.

    logit p = α + β · logit(p_rynku) + γ · (logit p_modelu − logit p_rynku) [+ δ · x]

  - p_rynku  = środek bid/ask Kalshi ≥ 1 h przed meczem (evaluate_stats → *_predictions.csv)
  - p_modelu = PointsModel walk-forward BEZ absencji (p_base). p_absence liczy absencje z tego,
               kto faktycznie nie zagrał — rynek 1 h przed meczem tego nie wie (późne skreślenia),
               więc korekta na p_absence pokazałaby fałszywą przewagę. Absencje wejdą dopiero
               z historycznych raportów kontuzji (to, co było wiadomo przed meczem).
  - β ≈ 1, γ ≈ 0 → rynek wie wszystko, co model; γ > 0 istotnie → model dodaje informację.

Absencje (kolumny vac_*):
  - *_rep: koledzy z rotacji Out/Doubtful w raporcie kontuzji z chwili start − 1 h — to samo,
           co rynek wiedział w momencie ceny (uczciwy test),
  - *_act: koledzy z rotacji, którzy faktycznie nie zagrali — górna granica (zawiera późne
           skreślenia, których rynek nie znał); tylko dla porównania, nie do kuponu.
  Rotacja = jak history.py: mecz dla drużyny w ostatnich ABSENCE_DAYS dniach i ≥ ROTATION_MIN min
  średnio w ostatnich 10 meczach. vac_min = suma ich minut, vac_stat = suma ich średniej statystyki.

Walk-forward: regresja logistyczna (L2) trenowana na wszystkich liniach sprzed tygodnia,
predykcja na tydzień; pierwsze MIN_TRAIN linii tylko do treningu.

Użycie:
  python market_model.py                      # najnowszy reports/eval_stats_*_predictions.csv
  python market_model.py --pred ścieżka.csv   # → reports/market_model_<data>.json
"""
import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate_stats import _cluster_ci, _logloss, calibration, simulate
from history import ABSENCE_DAYS, ROTATION_MIN, load_logs, norm_name

logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports")
MIN_TRAIN  = 400       # linii przed pierwszą predykcją
C_L2       = 1.0       # siła regularyzacji (mniejsze C = mocniejsza)
EPS        = 1e-4
VARIANTS   = {
    "market_only": ["lm"],                                    # rekalibracja samego rynku (czy ma bias?)
    "market+model": ["lm", "d_base"],                         # korekta o różnicę model − rynek
    "market+model+report": ["lm", "d_base", "vac_min_rep", "vac_stat_rep"],   # uczciwe absencje
    "market+model+actual": ["lm", "d_base", "vac_min_act", "vac_stat_act"],   # górna granica (wyciek)
}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def add_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lm"] = _logit(df["p_mid"])
    df["d_base"] = _logit(df["p_base"]) - df["lm"]
    return df


def absence_features(pred: pd.DataFrame, reports: pd.DataFrame | None = None,
                     hours_before: int = 1) -> pd.DataFrame:
    """Dokleja vac_{min,stat}_{rep,act} (skala: minuty/48, statystyka/10) do każdej linii."""
    import injuries

    logs, _ = load_logs(offline=True)
    logs = logs.sort_values("GAME_DATE").copy()
    logs["name_key"] = logs["PLAYER_NAME"].map(norm_name)
    g = logs.groupby("PLAYER_ID")
    for c in ("MIN", "PTS", "REB", "AST"):             # średnia z 10 meczów DO i z tym meczem
        logs[f"{c}_10"] = g[c].transform(lambda s: s.rolling(10, min_periods=1).mean())
    team_of = logs.set_index(["PLAYER_ID", "GAME_ID"])[["TEAM_ID", "TEAM_ABBREVIATION"]]
    played = logs.groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"].apply(set).to_dict()

    reports = injuries.load_report_history() if reports is None else reports
    rep = reports[reports["status"].isin(injuries.OUT_STATUSES)]
    out_by = rep.groupby(["snapshot", "game_date", "team"])["name_key"].apply(set).to_dict()
    have_snap = set(reports["snapshot"].astype(str))

    pred = pred.copy()
    pred["PLAYER_ID"] = pred["PLAYER_ID"].astype("Int64")
    pred = pred.join(team_of, on=["PLAYER_ID", "GAME_ID"])
    rows = []
    cache = {}
    for r in pred.itertuples():
        snap = injuries._snap_key((pd.Timestamp(r.tip) - pd.Timedelta(hours=hours_before)).to_pydatetime()) \
            if pd.notna(r.tip) else None
        key = (r.TEAM_ID, r.game_date)
        if key not in cache:
            win = logs[(logs["TEAM_ID"] == r.TEAM_ID) & (logs["GAME_DATE"] < r.game_date)
                       & (logs["GAME_DATE"] >= r.game_date - pd.Timedelta(days=ABSENCE_DAYS))]
            last = win.groupby("PLAYER_ID").tail(1)
            cache[key] = last[last["MIN_10"] >= ROTATION_MIN]
        rot = cache[key][cache[key]["PLAYER_ID"] != r.PLAYER_ID]
        out_rep = out_by.get((snap, r.game_date, r.TEAM_ABBREVIATION), set())
        rep_mask = rot["name_key"].isin(out_rep)
        act_mask = ~rot["PLAYER_ID"].isin(played.get((r.GAME_ID, r.TEAM_ID), set()))
        st = f"{r.stat}_10"
        rows.append({
            "has_report": snap in have_snap,
            "vac_min_rep": rot.loc[rep_mask, "MIN_10"].sum() / 48, "vac_stat_rep": rot.loc[rep_mask, st].sum() / 10,
            "vac_min_act": rot.loc[act_mask, "MIN_10"].sum() / 48, "vac_stat_act": rot.loc[act_mask, st].sum() / 10,
        })
    return pd.concat([pred.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def walk_forward(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[dict]]:
    """p_korekty dla każdej linii (NaN w okresie rozgrzewki) + współczynniki z każdego tygodnia."""
    week = df["game_date"] - pd.to_timedelta(df["game_date"].dt.weekday, unit="D")
    p = pd.Series(np.nan, index=df.index)
    coefs = []
    for w in sorted(week.unique()):
        tr, te = df[week < w], df[week == w]
        if len(tr) < MIN_TRAIN:
            continue
        lr = LogisticRegression(C=C_L2).fit(tr[cols], tr["y_over"])
        p[te.index] = lr.predict_proba(te[cols])[:, 1]
        coefs.append({"week": str(pd.Timestamp(w).date()), "n_train": len(tr),
                      "intercept": round(float(lr.intercept_[0]), 3),
                      **{c: round(float(v), 3) for c, v in zip(cols, lr.coef_[0])}})
    return p.to_numpy(), coefs          # w kolejności wierszy df (bez sortowania)


def full_fit_ci(df: pd.DataFrame, cols: list[str], rng, n_boot: int = 500) -> dict:
    """Współczynniki na całej próbie + CI z bootstrapu po meczach (czy γ ≠ 0?)."""
    games = df["GAME_ID"].astype(str).to_numpy()
    ug, inv = np.unique(games, return_inverse=True)
    idx_by_game = [np.flatnonzero(inv == g) for g in range(len(ug))]
    fit = lambda d: LogisticRegression(C=C_L2).fit(d[cols], d["y_over"]).coef_[0]
    boot = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_game[g] for g in rng.integers(0, len(ug), len(ug))])
        boot.append(fit(df.iloc[pick]))
    boot = np.array(boot)
    est = fit(df)
    return {c: {"coef": round(float(est[i]), 3),
                "ci95": [round(float(np.percentile(boot[:, i], q)), 3) for q in (2.5, 97.5)]}
            for i, c in enumerate(cols)}


def evaluate(pred: pd.DataFrame, rng) -> dict:
    out = {}
    for stat, df in pred.groupby("stat"):
        df = add_terms(df.dropna(subset=["p_mid", "p_base", "y_over"]))
        res = {"n_lines": int(len(df)), "variants": {}}
        for name, cols in VARIANTS.items():
            p, coefs = walk_forward(df, cols)
            ok = ~np.isnan(p)
            sub = df[ok].assign(p_corr=p[ok])
            y = sub["y_over"].to_numpy()
            diff = _logloss(sub["p_corr"].to_numpy(), y) - _logloss(sub["p_mid"].to_numpy(), y)
            res["variants"][name] = {
                "n": int(ok.sum()), "n_games": int(sub["GAME_ID"].nunique()),
                "logloss_corr": round(float(_logloss(sub["p_corr"].to_numpy(), y).mean()), 4),
                "logloss_market": round(float(_logloss(sub["p_mid"].to_numpy(), y).mean()), 4),
                "logloss_diff": round(float(diff.mean()), 4),
                "logloss_diff_ci95": _cluster_ci(diff, sub["GAME_ID"].astype(str).to_numpy(), rng),
                "bets": simulate(sub, "p_corr", rng),
                "calibration": calibration(sub, "p_corr"),
                "coefs_last": coefs[-1] if coefs else None,
                "coefs_full_sample": full_fit_ci(df, cols, rng),
            }
        out[stat] = res
        v = res["variants"]["market+model"]
        logger.info("%s: n=%d Δlog-loss %+.4f CI %s | γ %s", stat, v["n"], v["logloss_diff"],
                    v["logloss_diff_ci95"], v["coefs_full_sample"]["d_base"])
    return out


def latest_predictions(report_dir: Path = REPORT_DIR) -> Path:
    files = sorted(report_dir.glob("eval_stats_*_predictions.csv"))
    if not files:
        raise FileNotFoundError("Brak reports/eval_stats_*_predictions.csv — uruchom: python evaluate_stats.py")
    return files[-1]


def run(pred_path: Path | None = None) -> dict:
    pred_path = pred_path or latest_predictions()
    pred = pd.read_csv(pred_path, parse_dates=["game_date"], dtype={"GAME_ID": str})
    pred["tip"] = pd.to_datetime(pred["tip"], utc=True, format="ISO8601")
    pred = absence_features(pred)
    n_rep = int(pred["has_report"].sum())
    logger.info("Linii z raportem kontuzji: %d / %d", n_rep, len(pred))
    pred = pred[pred["has_report"]]                     # wszystkie warianty na tej samej próbie
    report = {"generated": date.today().isoformat(), "predictions": str(pred_path),
              "formula": "logit p = a + b·logit(p_market) + g·(logit p_base − logit p_market)",
              "stats": evaluate(pred, np.random.default_rng(0))}
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f"market_model_{report['generated']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path)
    args = ap.parse_args()
    r = run(args.pred)
    for st, v in r["stats"].items():
        for name, x in v["variants"].items():
            b5 = next(b for b in x["bets"] if b["ev_threshold"] == 0.05)
            print(f"{st:3} {name:13} n={x['n']:5} Δlog-loss {x['logloss_diff']:+.4f} {x['logloss_diff_ci95']} "
                  f"| EV>5%: {b5['bets']} zakł., ROI {b5.get('roi')} {b5.get('roi_ci95')} "
                  f"| {x['coefs_full_sample']}")
