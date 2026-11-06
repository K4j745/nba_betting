# tests/test_kalshi.py
"""Kalshi: parsowanie rynków, linia główna, propsy z prowizją; rozkład dla REB/AST; wyniki kuponów."""
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import kalshi
from live import STAT_COUPONS_DDL, stat_results
from points_model import p_over_count


def _market(**kw):
    m = {"ticker": "KXNBAREB-26OCT29CLEATL-ATLJJOHNSON1-8", "event_ticker": "KXNBAREB-26OCT29CLEATL",
         "title": "Jalen Johnson: 8+ rebounds", "yes_sub_title": "Jalen Johnson: 8+",
         "floor_strike": 7.5, "occurrence_datetime": "2026-10-30T03:00:00Z", "result": "yes",
         "expiration_value": "9"}
    m.update(kw)
    return m


def test_parse_event_ticker():
    day, away, home = kalshi.parse_event_ticker("KXNBAREB-26OCT29CLEATL")
    assert day == pd.Timestamp("2026-10-29") and (away, home) == ("CLE", "ATL")


def test_parse_market_tip_and_line():
    r = kalshi.parse_market(_market(), "REB")
    assert r["player"] == "Jalen Johnson" and r["line"] == 7.5 and r["actual"] == 9
    assert r["tip"] == pd.Timestamp("2026-10-30T00:00:00Z")


def test_parse_market_old_format_falls_back_to_expected_expiration():
    m = _market(occurrence_datetime=None, expected_expiration_time="2026-04-18T02:30:00Z")
    assert kalshi.parse_market(m, "REB")["tip"] == pd.Timestamp("2026-04-17T23:30:00Z")


def test_mid_rejects_wide_spread():
    assert np.isnan(kalshi._mid(0.05, 0.99)[2])
    assert kalshi._mid("0.48", "0.52")[2] == 0.5


def test_main_line_is_closest_to_half():
    df = pd.DataFrame({"stat": "REB", "event_ticker": "E", "name_key": "a",
                       "line": [5.5, 7.5, 9.5], "p_mid": [0.8, 0.45, 0.2]})
    assert kalshi.main_lines(df)["line"].tolist() == [7.5]


def test_to_props_includes_fee():
    lines = pd.DataFrame([{**kalshi.parse_market(_market(), "REB"), "bid": 0.48, "ask": 0.50, "p_mid": 0.49}])
    p = kalshi.to_props(lines)[0]
    assert p["over_odds"] == round(1 / (0.50 + 0.02), 3)      # prowizja 0,07·0,5·0,5 → 2 ¢
    assert p["under_odds"] == round(1 / (0.52 + 0.02), 3)
    assert p["bookmaker"] == "kalshi" and p["stat"] == "REB"


def test_hist_candle_reads_archive_format(monkeypatch):
    candles = {"candlesticks": [
        {"end_period_ts": 100, "yes_bid": {"close": "0.40"}, "yes_ask": {"close": "0.44"}},
        {"end_period_ts": 200, "yes_bid": {"close": "0.90"}, "yes_ask": {"close": "0.95"}}]}
    monkeypatch.setattr(kalshi, "_get", lambda *a, **k: candles)
    r = kalshi._hist_candle("T", target=150)                  # świeca po starcie − 1 h odrzucona
    assert r["bid"] == 0.40 and np.isclose(r["p_mid"], 0.42)


def test_count_distribution_matches_negbin_and_poisson():
    p = p_over_count([4.0], [3.0], [5.5])                     # var 9 > μ 4 → NB(n=16/5, p=4/9)
    assert np.isclose(p[0], nbinom.sf(5, 16 / 5, 4 / 9))
    p_po = p_over_count([4.0], [1.5], [3.5])                  # var 2,25 < μ → Poisson
    assert 0.5 < p_po[0] < 0.6


def test_stat_results_scores_from_logs(tmp_path):
    db = tmp_path / "t.db"
    coupon = {"picks": [{"player": "A", "bet": "over", "line": 7.5, "odds": 1.9, "game": "X @ Y"},
                        {"player": "B", "bet": "under", "line": 4.5, "odds": 2.0, "game": "X @ Y"},
                        {"player": "C", "bet": "over", "line": 3.5, "odds": 1.8, "game": "X @ Y"}]}
    with sqlite3.connect(db) as c:
        c.execute(STAT_COUPONS_DDL)
        c.execute("INSERT INTO stat_coupons (date, stat, coupon_json) VALUES ('2026-10-29', 'REB', ?)",
                  (json.dumps(coupon),))
    actual = {"A": 9.0, "B": 6.0}                              # C: brak logu → czeka
    res = {r["player"]: r for r in stat_results("REB", db_path=str(db), actual_fn=lambda n, d: actual.get(n))}
    assert res["A"]["result"] == "WIN" and res["A"]["profit"] == 0.9
    assert res["B"]["result"] == "LOSS" and res["B"]["profit"] == -1.0
    assert res["C"]["result"] is None
