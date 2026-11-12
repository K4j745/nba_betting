import numpy as np
import pandas as pd
from scipy.stats import nbinom

import count_dist as cd


def test_fit_nb_r_recovers_dispersion():
    rng = np.random.default_rng(0)
    mu, r = np.full(20000, 6.0), 5.0
    y = nbinom.rvs(r, r / (r + mu), random_state=rng)
    assert abs(cd.fit_nb_r(y, mu) - r) < 0.6


def test_nb_sf_matches_mean_and_half_line():
    # P(Y > 6.5) == P(Y >= 7)
    p = cd.nb_sf(np.array([6.0]), np.array([6.0]), 10.0)[0]
    assert abs(p - nbinom.sf(6, 10.0, 10.0 / 16.0)) < 1e-12


def test_player_dispersion_uses_only_earlier_games():
    data = pd.DataFrame({"PLAYER_ID": [1] * 7,
                         "GAME_DATE": pd.date_range("2026-01-01", periods=7),
                         "REB": [2, 8, 2, 8, 2, 8, 100]})
    pred = pd.DataFrame({"PLAYER_ID": [1.0], "game_date": [pd.Timestamp("2026-01-07")]})
    out = cd.player_dispersion(pred, data, "REB")
    # mecz z 7.01 (100 zbiórek) nie może wejść do dyspersji linii z tego dnia
    vals = pd.Series([2, 8, 2, 8, 2, 8])
    assert abs(out["D"].iloc[0] - vals.var() / vals.mean()) < 1e-9
    assert out["n"].iloc[0] == 6
