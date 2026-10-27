# nba_betting/points_model.py
"""
Model rozkładu punktów zawodnika (bez cech rynkowych).

  μ  — XGBRegressor na cechach z history.FEATURE_COLS
  σ  — odchylenie reszt jako funkcja liniowa μ (dopasowane na zbiorze walidacyjnym)
  P(PTS > linia) = 1 − Φ((linia − μ) / σ)     (linie są x.5, więc bez korekty ciągłości)

Model nie widzi kursów, więc porównanie z rynkiem w evaluate.py jest uczciwe:
mierzymy, czy statystyka sama z siebie wie coś, czego linia nie wie.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
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

    def mu(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.reg.predict(X[self.features]), 0, None)

    def sigma(self, mu: np.ndarray) -> np.ndarray:
        return np.clip(self.sigma_a + self.sigma_b * mu, 1.5, None)

    def p_over(self, X: pd.DataFrame, line) -> np.ndarray:
        m = self.mu(X)
        return 1 - norm.cdf((np.asarray(line, dtype=float) - m) / self.sigma(m))


def _fit_sigma(mu: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """σ(μ) = a + b·μ z odchyleń reszt w decylach μ."""
    bins = pd.qcut(mu, 10, duplicates="drop")
    stats = pd.DataFrame({"mu": mu, "r": resid}).groupby(bins, observed=True).agg(
        mu=("mu", "mean"), sd=("r", "std"), n=("r", "size"))
    b, a = np.polyfit(stats["mu"], stats["sd"], 1, w=np.sqrt(stats["n"]))
    return float(a), float(b)


def train_points_model(df: pd.DataFrame, cutoff, features: list[str] = FEATURE_COLS) -> PointsModel:
    """Trenuje na meczach przed `cutoff` (walidacja = ostatnie VALID_ROWS meczów przed cutoff)."""
    cutoff = pd.Timestamp(cutoff)
    features = list(features)
    hist = df[(df["GAME_DATE"] < cutoff) & df["PTS"].notna()].sort_values("GAME_DATE")
    valid_start = hist["GAME_DATE"].iloc[-VALID_ROWS]
    tr, va = hist[hist["GAME_DATE"] < valid_start], hist[hist["GAME_DATE"] >= valid_start]

    reg = XGBRegressor(**XGB_PARAMS)
    reg.fit(tr[features], tr["PTS"], eval_set=[(va[features], va["PTS"])], verbose=False)

    mu_va = np.clip(reg.predict(va[features]), 0, None)
    a, b = _fit_sigma(mu_va, va["PTS"].to_numpy() - mu_va)
    mae = float(np.mean(np.abs(va["PTS"] - mu_va)))
    logger.info(
        "PointsModel cutoff=%s cech=%d: train=%d valid=%d trees=%d valid_MAE=%.3f σ=%.2f+%.3f·μ",
        cutoff.date(), len(features), len(tr), len(va), reg.best_iteration + 1, mae, a, b,
    )
    return PointsModel(reg=reg, sigma_a=a, sigma_b=b, cutoff=cutoff, features=features)
