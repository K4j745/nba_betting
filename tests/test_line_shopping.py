# tests/test_line_shopping.py
"""Line shopping (krok 6): przeliczanie linii, dopasowanie do Pinnacle, najlepsza cena."""
import numpy as np
import pandas as pd
import pytest

from line_shopping import attach_reference, best_price, p_over_at, value_vs_pinnacle

T0 = pd.Timestamp("2026-10-30 23:00", tz="UTC")


def test_p_over_at_identity_and_monotone():
    assert p_over_at(0.55, 24.5, 24.5) == 0.55
    hi, lo = p_over_at(0.5, 24.5, 23.5), p_over_at(0.5, 24.5, 25.5)
    assert hi > 0.5 > lo                         # niższa linia → over bardziej prawdopodobny
    assert hi - 0.5 == pytest.approx(0.5 - lo, abs=0.01)
    assert 0.03 < 0.5 - lo < 0.07                 # 1 pkt przy σ≈8,5 → φ(0)/σ ≈ 4,7 pp


def _snap(book, line, over, under, minutes_before, p_fair=None, player="A"):
    p = (1 / over) / (1 / over + 1 / under)
    return dict(player_name=player, game_time=T0, fetched_at=T0 - pd.Timedelta(minutes=minutes_before),
                bookmaker=book, line=line, over_odds=over, under_odds=under,
                p_fair=p_fair if p_fair is not None else p)


def test_reference_from_same_run_and_closing():
    snaps = pd.DataFrame([
        _snap("pinnacle", 24.5, 1.95, 1.95, 300),
        _snap("unibet", 24.5, 2.10, 1.75, 299),     # ten sam run → ref z −300 min
        _snap("pinnacle", 24.5, 1.80, 2.10, 60),    # zamknięcie: over podrożał
    ])
    m = attach_reference(snaps)
    assert len(m) == 1
    assert m.loc[0, "ref_p"] == pytest.approx(0.5)
    assert m.loc[0, "close_p"] > 0.5
    value, bets = value_vs_pinnacle(m, np.random.default_rng(0))
    b = bets.iloc[0]
    assert b["side"] == "over" and b["ev"] == pytest.approx(0.5 * 2.10 - 1)
    assert b["clv"] > b["ev"]                     # linia poszła w naszą stronę


def test_reference_too_far_in_time_is_ignored():
    snaps = pd.DataFrame([_snap("pinnacle", 24.5, 1.95, 1.95, 600), _snap("unibet", 24.5, 2.1, 1.75, 100)])
    assert attach_reference(snaps)["ref_p"].isna().all()


def test_best_price_and_arbitrage():
    snaps = pd.DataFrame([
        _snap("pinnacle", 20.5, 1.95, 1.90, 60),
        _snap("unibet", 20.5, 2.15, 1.70, 60),      # best over 2.15 + best under 1.90 → suma < 1
        _snap("unibet", 30.5, 1.9, 1.9, 60, player="B"),   # tylko 1 bukmacher → pominięty
    ])
    r = best_price(snaps)
    assert r["n_groups"] == 1
    assert r["arbitrage_count"] == 1
    assert r["margin_best_of"] == pytest.approx(1 / 2.15 + 1 / 1.90 - 1, abs=1e-4)
