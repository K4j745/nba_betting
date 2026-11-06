# nba_betting/points_model.py
"""
Model rozkładu punktów zawodnika (bez cech rynkowych).

  μ  — XGBRegressor na cechach z history.FEATURE_COLS
  σ  — odchylenie reszt jako funkcja liniowa μ (dopasowane na zbiorze walidacyjnym)
  P(PTS > linia) = 1 − Φ((linia − μ) / σ)     (linie są x.5, więc bez korekty ciągłości)
  REB/AST: małe, prawoskośne liczby całkowite — normalny zawyżał P(over) o ~6–8 pp
  (kalibracja na sezonie 2025-26), więc rozkład ujemny dwumianowy o tej samej średniej μ
  i wariancji σ² (Poisson, gdy σ² ≤ μ): P(Y > k + 0,5) = P(Y ≥ k + 1).

Model nie widzi kursów, więc porównanie z rynkiem w evaluate.py jest uczciwe:
mierzymy, czy statystyka sama z siebie wie coś, czego linia nie wie.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson
from xgboost import XGBRegressor

from history import FEATURE_COLS

logger = logging.getLogger(__name__)

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=100,
    random_state=42,
    verbosity=0,
)
VALID_ROWS = 6000    # ostatnie ~6000 meczów-zawodnika przed odcięciem = walidacja (early stopping + σ)
                     # liczone w wierszach, nie dniach — przed startem sezonu dni to przerwa letnia


@dataclass
class PointsModel:
    reg: XGBRegressor
    sigma_a: float
    sigma_b: float
    cutoff: pd.Timestamp
    features: list[str]
    dist: str = "normal"          # "normal" (PTS) albo "negbin" (REB, AST)

    def mu(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.reg.predict(X[self.features]), 0, None)

    def sigma(self, mu: np.ndarray) -> np.ndarray:
        floor = 1.5 if self.dist == "normal" else 0.3
        return np.clip(self.sigma_a + self.sigma_b * mu, floor, None)

    def p_over(self, X: pd.DataFrame, line) -> np.ndarray:
        m = self.mu(X)
        line = np.broadcast_to(np.asarray(line, dtype=float), m.shape)
        if self.dist == "normal":
            return 1 - norm.cdf((line - m) / self.sigma(m))
        return p_over_count(m, self.sigma(m), line)


def p_over_count(mu, sigma, line) -> np.ndarray:
    """P(Y > line) dla liczby całkowitej Y: ujemny dwumianowy (μ, σ²), Poisson gdy σ² ≤ μ."""
    mu = np.clip(np.asarray(mu, dtype=float), 0.05, None)
    var = np.asarray(sigma, dtype=float) ** 2
    k = np.floor(np.asarray(line, dtype=float))          # Y > k + 0,5  ⇔  Y > k
    over = var > mu * 1.0001
    v = np.where(over, var, mu * 2)                      # placeholder, żeby nie dzielić przez 0
    n, p = mu ** 2 / (v - mu), mu / v
    return np.where(over, nbinom.sf(k, n, p), poisson.sf(k, mu))


def _fit_sigma(mu: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """σ(μ) = a + b·μ z odchyleń reszt w decylach μ."""
    bins = pd.qcut(mu, 10, duplicates="drop")
    stats = pd.DataFrame({"mu": mu, "r": resid}).groupby(bins, observed=True).agg(
        mu=("mu", "mean"), sd=("r", "std"), n=("r", "size"))
    b, a = np.polyfit(stats["mu"], stats["sd"], 1, w=np.sqrt(stats["n"]))
    return float(a), float(b)


def train_points_model(df: pd.DataFrame, cutoff, features: list[str] = FEATURE_COLS,
                       target: str = "PTS") -> PointsModel:
    """
    Trenuje na meczach przed `cutoff` (walidacja = ostatnie VALID_ROWS meczów przed cutoff).
    target: PTS (domyślnie), REB albo AST — ten sam model μ + σ(μ) dla każdej statystyki.
    """
    cutoff = pd.Timestamp(cutoff)
    features = list(features)
    hist = df[(df["GAME_DATE"] < cutoff) & df[target].notna()].sort_values("GAME_DATE")
    valid_start = hist["GAME_DATE"].iloc[-VALID_ROWS]
    tr, va = hist[hist["GAME_DATE"] < valid_start], hist[hist["GAME_DATE"] >= valid_start]

    reg = XGBRegressor(**XGB_PARAMS)
    reg.fit(tr[features], tr[target], eval_set=[(va[features], va[target])], verbose=False)

    mu_va = np.clip(reg.predict(va[features]), 0, None)
    a, b = _fit_sigma(mu_va, va[target].to_numpy() - mu_va)
    mae = float(np.mean(np.abs(va[target] - mu_va)))
    logger.info(
        "PointsModel[%s] cutoff=%s cech=%d: train=%d valid=%d trees=%d valid_MAE=%.3f σ=%.2f+%.3f·μ",
        target, cutoff.date(), len(features), len(tr), len(va), reg.best_iteration + 1, mae, a, b,
    )
    return PointsModel(reg=reg, sigma_a=a, sigma_b=b, cutoff=cutoff, features=features,
                       dist="normal" if target == "PTS" else "negbin")
