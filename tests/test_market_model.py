# tests/test_market_model.py
"""Model od rynku: predykcje walk-forward w kolejności wierszy wejścia (bez sortowania)."""
import numpy as np
import pandas as pd

import market_model as mm


def test_walk_forward_keeps_row_order(monkeypatch):
    monkeypatch.setattr(mm, "MIN_TRAIN", 50)
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.Timestamp("2026-01-05") + pd.to_timedelta(rng.integers(0, 70, n), unit="D")
    p_mid = rng.uniform(0.2, 0.8, n)
    df = pd.DataFrame({"game_date": dates, "p_mid": p_mid, "p_base": p_mid,
                       "y_over": (rng.uniform(size=n) < p_mid).astype(int)})
    df = mm.add_terms(df)                                  # wiersze NIE są posortowane po dacie
    p, coefs = mm.walk_forward(df, ["lm"])
    ok = ~np.isnan(p)
    assert coefs and ok.sum() > 100
    # predykcja rośnie z ceną rynku → korelacja z p_mid tego samego wiersza bliska 1
    assert np.corrcoef(p[ok], df["p_mid"].to_numpy()[ok])[0, 1] > 0.95
    # rozgrzewka = najwcześniejsze tygodnie, nie pierwsze wiersze
    assert df.loc[~ok, "game_date"].max() < df.loc[ok, "game_date"].min()
