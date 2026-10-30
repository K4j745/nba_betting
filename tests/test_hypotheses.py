# tests/test_hypotheses.py
"""Flagi revenge / nemesis muszą korzystać wyłącznie z meczów sprzed danego dnia."""
import numpy as np
import pandas as pd
import pytest

from hypotheses import flag_effect, nemesis_scores, revenge_flags, slope_effect

A, B, C = 1, 2, 3


def _games(rows):
    """rows: (data, drużyna, rywal[, reszta])"""
    return pd.DataFrame([
        dict(PLAYER_ID=7, GAME_ID=f"G{i}", GAME_DATE=pd.Timestamp(d), TEAM_ID=t, OPP_ID=o,
             resid=(r[0] if r else 0.0))
        for i, (d, t, o, *r) in enumerate(rows)
    ])


def test_revenge_only_after_leaving_team():
    f = revenge_flags(_games([
        ("2024-01-01", A, B),   # gra dla A
        ("2024-01-05", A, C),
        ("2024-02-01", B, A),   # przeszedł do B → pierwszy mecz z A = revenge, first
        ("2024-03-01", B, C),
        ("2024-04-01", B, A),   # drugi mecz z A = revenge, nie first
    ]))
    assert list(f["is_revenge"]) == [False, False, True, False, True]
    assert list(f["revenge_first"]) == [False, False, True, False, False]
    assert f.loc[2, "revenge_days"] == 27        # 5.01 → 1.02


def test_playing_against_future_team_is_not_revenge():
    # mecz z B ZANIM zawodnik do B przeszedł — nie może być revenge (brak wycieku)
    f = revenge_flags(_games([("2024-01-01", A, B), ("2024-02-01", B, C)]))
    assert not f["is_revenge"].any()


def test_return_to_former_team_resets_first_meeting():
    f = revenge_flags(_games([
        ("2024-01-01", A, B), ("2024-02-01", B, A),   # A → B, revenge first
        ("2024-03-01", A, B),                          # wraca do A: B to teraz była drużyna
        ("2024-04-01", B, A),                          # znów w B: mecz z A = nowy „pierwszy”
    ]))
    assert list(f["revenge_first"]) == [False, True, True, True]


def test_nemesis_uses_previous_meetings_only():
    res = nemesis_scores(_games([
        ("2024-01-01", A, B, -6.0), ("2024-01-10", A, B, -6.0), ("2024-01-20", A, B, -6.0),
        ("2024-01-25", A, C, 2.0), ("2024-01-26", A, C, 2.0), ("2024-01-27", A, C, 2.0),
        ("2024-02-01", A, B, 10.0),
    ]), k=5)
    assert res.loc[0, "nemesis_score"] == 0                      # brak poprzednich meczów
    assert res.loc[2, "nemesis_score"] == pytest.approx(-12 / 7)  # 2 mecze, k=5
    last = res.iloc[-1]
    assert last["nemesis_n"] == 3
    assert last["nemesis_score"] == pytest.approx(-18 / 8)       # bez dzisiejszego +10
    assert last["is_nemesis"]                                    # B: najgorszy rywal z ≥3 meczami
    assert not res.loc[:5, "is_nemesis"].any()


def test_flag_effect_and_slope():
    rng = np.random.default_rng(1)
    n = 4000
    res = pd.DataFrame({"PLAYER_ID": rng.integers(0, 200, n), "flag": rng.random(n) < 0.2})
    res["x"] = rng.normal(size=n)
    res["resid"] = rng.normal(size=n) + 2.0 * res["flag"] + 0.5 * res["x"]
    e = flag_effect(res, "flag")
    assert e["diff"] == pytest.approx(2.0, abs=0.15)
    assert 1.5 < e["diff_ci95"][0] < e["diff"] < e["diff_ci95"][1] < 2.5
    s = slope_effect(res, "x")
    assert s["slope"] == pytest.approx(0.5, abs=0.1)
