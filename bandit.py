# nba_betting/bandit.py
"""
Test bandyty offline: czy polityka ucząca się z nagrody (zysku) stawia lepiej niż „nie stawiaj”?

Każda linia Kalshi (z eval_stats, cena ≥ 1 h przed meczem) to jedna decyzja: YES (over po ask),
NO (under po 1 − bid) albo pas. Nagroda = zwrot na zainwestowaną kwotę po prowizji Kalshi,
pas = 0. Zakład nie zmienia przyszłych stanów, więc to bandyta kontekstowy, a nie pełne RL.

Polityki (decyzje dzień po dniu, uczenie tylko na dniach wcześniejszych):
  - never        nigdy nie stawia (nagroda 0),
  - random       losowa strona w 10% linii,
  - ev5          obecna reguła: p modelu (bez absencji) i EV > 5%,
  - linucb       LinUCB: osobna regresja grzbietowa nagrody dla YES i NO, wybór najwyższej
                 górnej granicy; uczy się WYŁĄCZNIE z nagród zakładów, które sam postawił,
  - full_info    to samo bez eksploracji, ale uczy się z wyników obu stron każdej linii
                 (wynik meczu znamy niezależnie od decyzji, więc to dozwolone i silniejsze).
Cechy: logit ceny, różnica model − rynek, spread, typ statystyki. Parametry ustalone z góry.
Linie od FINAL_FROM nietknięte; --final tylko, jeśli jakaś polityka przejdzie na dev.

Użycie:
  python bandit.py            # → reports/bandit_<data>.json
  python bandit.py --final
"""
import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from evaluate_stats import _cluster_ci
from feature_lab import FINAL_FROM
from kalshi import kalshi_fee

logger = logging.getLogger(__name__)

ALPHA      = 0.25      # siła eksploracji LinUCB
RIDGE      = 1.0
RANDOM_P   = 0.10
EV_MIN     = 0.05
WARMUP     = 400       # linii, zanim polityki uczące się zaczną stawiać


def prepare(pred: pd.DataFrame) -> pd.DataFrame:
    d = pred.dropna(subset=["p_mid", "p_base", "y_over", "bid", "ask"]).copy()
    d = d.sort_values(["game_date", "GAME_ID"]).reset_index(drop=True)
    d["cost_yes"] = d["ask"] + d["ask"].map(kalshi_fee)
    d["cost_no"]  = (1 - d["bid"]) + (1 - d["bid"]).map(kalshi_fee)
    y = d["y_over"].to_numpy()
    d["r_yes"] = (y - d["cost_yes"]) / d["cost_yes"]
    d["r_no"]  = ((1 - y) - d["cost_no"]) / d["cost_no"]
    lm = mm._logit(d["p_mid"])
    d["x_lm"], d["x_gap"] = lm, mm._logit(d["p_base"]) - lm
    d["x_spread"] = (d["ask"] - d["bid"]) * 10
    for st in ("REB", "AST"):
        d[f"x_{st}"] = (d["stat"] == st).astype(float)
    return d


def features(d: pd.DataFrame) -> np.ndarray:
    cols = ["x_lm", "x_gap", "x_spread", "x_REB", "x_AST"]
    return np.column_stack([np.ones(len(d)), d[cols].to_numpy()])


class Ridge:
    """Regresja grzbietowa aktualizowana przyrostowo (A = λI + ΣxxT, b = Σrx)."""
    def __init__(self, k: int):
        self.A, self.b = RIDGE * np.eye(k), np.zeros(k)

    def update(self, X: np.ndarray, r: np.ndarray):
        self.A += X.T @ X
        self.b += X.T @ r

    def predict(self, X: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        inv = np.linalg.inv(self.A)
        mean = X @ (inv @ self.b)
        return mean + alpha * np.sqrt(np.einsum("ij,jk,ik->i", X, inv, X)) if alpha else mean


def run_policies(d: pd.DataFrame, rng) -> dict[str, np.ndarray]:
    """Akcja dla każdej linii: +1 YES, −1 NO, 0 pas."""
    X, r_yes, r_no = features(d), d["r_yes"].to_numpy(), d["r_no"].to_numpy()
    ev_yes = d["p_base"].to_numpy() / d["cost_yes"].to_numpy() - 1
    ev_no = (1 - d["p_base"].to_numpy()) / d["cost_no"].to_numpy() - 1
    acts = {
        "never": np.zeros(len(d), int),
        "random": np.where(rng.random(len(d)) < RANDOM_P, rng.choice([-1, 1], len(d)), 0),
        "ev5": np.where(np.maximum(ev_yes, ev_no) > EV_MIN, np.where(ev_yes >= ev_no, 1, -1), 0),
        "linucb": np.zeros(len(d), int), "full_info": np.zeros(len(d), int),
    }
    ucb = {1: Ridge(X.shape[1]), -1: Ridge(X.shape[1])}
    full = {1: Ridge(X.shape[1]), -1: Ridge(X.shape[1])}
    seen = 0
    for _, idx in d.groupby("game_date", sort=True).indices.items():
        Xd = X[idx]
        if seen >= WARMUP:
            for name, models, alpha in (("linucb", ucb, ALPHA), ("full_info", full, 0.0)):
                s_yes, s_no = models[1].predict(Xd, alpha), models[-1].predict(Xd, alpha)
                best = np.maximum(s_yes, s_no)
                acts[name][idx] = np.where(best > 0, np.where(s_yes >= s_no, 1, -1), 0)
        # uczenie po dniu: LinUCB tylko z własnych zakładów, full_info z obu stron każdej linii
        for side, r in ((1, r_yes), (-1, r_no)):
            mine = idx[acts["linucb"][idx] == side] if seen >= WARMUP else idx   # rozgrzewka: obie strony
            ucb[side].update(X[mine], r[mine])
            full[side].update(Xd, r[idx])
        seen += len(idx)
    return acts


def score(d: pd.DataFrame, act: np.ndarray, keep: np.ndarray, rng) -> dict:
    reward = np.where(act == 1, d["r_yes"], np.where(act == -1, d["r_no"], 0.0))
    sel = keep & (act != 0)
    out = {"bets": int(sel.sum()), "share_of_lines": round(float(sel.sum() / keep.sum()), 3)}
    if sel.sum() >= 2:
        games = d["GAME_ID"].astype(str).to_numpy()
        out.update(roi=round(float(reward[sel].mean()), 4), roi_ci95=_cluster_ci(reward[sel], games[sel], rng),
                   units=round(float(reward[sel].sum()), 2),
                   hit_rate=round(float((reward[sel] > 0).mean()), 3),
                   over_share=round(float((act[sel] == 1).mean()), 3))
    return out


def run(final: bool = False) -> dict:
    rng = np.random.default_rng(0)
    path = sorted(mm.REPORT_DIR.glob("eval_stats_*_predictions.csv"))[-1]
    d = prepare(pd.read_csv(path, parse_dates=["game_date"], dtype={"GAME_ID": str}))
    acts = run_policies(d, rng)
    keep = ((d["game_date"] >= FINAL_FROM) if final else (d["game_date"] < FINAL_FROM)).to_numpy()
    report = {"generated": date.today().isoformat(), "source": path.name,
              "sample": "final" if final else f"dev (< {FINAL_FROM.date()})", "lines": int(keep.sum()),
              "params": {"alpha": ALPHA, "ridge": RIDGE, "warmup": WARMUP, "ev_min": EV_MIN},
              "policies": {n: score(d, a, keep, rng) for n, a in acts.items()},
              "by_stat": {st: {n: score(d, a, keep & (d["stat"] == st).to_numpy(), rng)
                               for n, a in acts.items() if n != "never"} for st in ("PTS", "REB", "AST")}}
    out = mm.REPORT_DIR / f"bandit_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for n, s in report["policies"].items():
        logger.info("%-9s %s", n, s)
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    run(ap.parse_args().final)
