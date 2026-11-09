import numpy as np
import pandas as pd

import bandit


def _pred(n=40, y=1):
    return pd.DataFrame({"stat": "PTS", "game_date": pd.date_range("2026-01-01", periods=n), "GAME_ID": [f"g{i}" for i in range(n)],
                         "p_mid": 0.5, "p_base": 0.5, "bid": 0.49, "ask": 0.51, "y_over": y})


def test_reward_includes_fee_and_pass_is_zero():
    d = bandit.prepare(_pred(2))
    cost = 0.51 + 0.02                               # prowizja przy ~0,5 to 2 ¢
    assert np.isclose(d["r_yes"].iloc[0], (1 - cost) / cost) and np.isclose(d["r_no"].iloc[0], -1)


def test_full_info_learns_that_over_always_wins():
    bandit.WARMUP = 5
    d = bandit.prepare(_pred(40, y=1))
    acts = bandit.run_policies(d, np.random.default_rng(0))
    assert (acts["full_info"][10:] == 1).all() and (acts["full_info"][:5] == 0).all()


def test_policies_never_bet_before_seeing_any_day():
    bandit.WARMUP = 5
    d = bandit.prepare(_pred(40, y=0))
    acts = bandit.run_policies(d, np.random.default_rng(0))
    assert acts["linucb"][0] == 0 and (acts["full_info"][10:] != 1).all()
