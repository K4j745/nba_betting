import numpy as np
import pandas as pd

import market_gap as mg

DAY = pd.Timestamp("2026-11-01")
SNAP = pd.Timestamp("2026-11-01 23:00", tz="UTC")


def _reads(*rows):
    return pd.DataFrame([{"game_date": DAY, "name_key": "a b", "pin_line": l, "p_pin": p,
                          "fetched_at": pd.Timestamp(t, tz="UTC")} for l, p, t in rows])


def test_read_after_price_is_ignored():
    lines = pd.DataFrame([{"game_date": DAY, "name_key": "a b", "snap_ts": SNAP}])
    out = mg.last_read_before(_reads((20.5, 0.5, "2026-11-01 20:00"), (22.5, 0.5, "2026-11-01 23:30")), lines)
    assert out["pin_line"].tolist() == [20.5]


def test_last_read_picks_line_nearest_even():
    lines = pd.DataFrame([{"game_date": DAY, "name_key": "a b", "snap_ts": SNAP}])
    out = mg.last_read_before(_reads((19.5, 0.6, "2026-11-01 20:00"), (20.5, 0.52, "2026-11-01 20:00")), lines)
    assert out["pin_line"].tolist() == [20.5]


def test_kalshi_interpolation_in_logit_and_range():
    ladder = pd.DataFrame({"game_date": DAY, "name_key": "a b", "line": [14.5, 19.5, 24.5], "p_mid": [0.8, 0.5, 0.2]})
    pairs = pd.DataFrame({"game_date": DAY, "name_key": "a b", "pin_line": [19.5, 17.0, 29.5]})
    p = mg.kalshi_at(ladder, pairs)
    assert np.isclose(p[0], 0.5) and np.isnan(p[2])
    assert np.isclose(p[1], 1 / (1 + np.exp(-np.log(4) / 2)))
