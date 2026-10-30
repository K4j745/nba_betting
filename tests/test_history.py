# tests/test_history.py
"""Cechy z history.py muszą korzystać wyłącznie z meczów sprzed danego dnia."""
import numpy as np
import pandas as pd
import pytest

from history import STAT_COLS, TEAM_ABBR_TO_ID, team_game_table, player_game_table, make_features

LAL, BOS = TEAM_ABBR_TO_ID["LAL"], TEAM_ABBR_TO_ID["BOS"]


def _logs():
    """LAL vs BOS, 4 mecze co 2 dni, 1 zawodnik LAL z PTS 10, 20, 30, 40."""
    dates = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-07"])
    team_rows, player_rows = [], []
    for i, d in enumerate(dates):
        gid = f"00225{i:05d}"
        for tid, opp, home in ((LAL, BOS, True), (BOS, LAL, False)):
            abbr = {LAL: "LAL", BOS: "BOS"}
            team_rows.append(dict(
                SEASON="2024-25", IS_PLAYOFFS=0, GAME_ID=gid, GAME_DATE=d, TEAM_ID=tid,
                TEAM_ABBREVIATION=abbr[tid],
                MATCHUP=f"{abbr[tid]} {'vs.' if home else '@'} {abbr[opp]}",
                MIN=240, FGA=80, FTA=20, OREB=10, TOV=12,
                PTS=100 + 10 * i if tid == BOS else 100,
            ))
        player_rows.append(dict(
            SEASON="2024-25", IS_PLAYOFFS=0, GAME_ID=gid, GAME_DATE=d, PLAYER_ID=7,
            PLAYER_NAME="Test Player", TEAM_ID=LAL, TEAM_ABBREVIATION="LAL",
            MATCHUP="LAL vs. BOS", MIN=30, PTS=10 * (i + 1), FGA=15, FTA=4, FG3A=5, TOV=2,
            FGM=6, FTM=3, OREB=1, DREB=4, AST=3, STL=1, BLK=0, PF=2,
        ))
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)


@pytest.fixture
def feats():
    p, t = _logs()
    tg = team_game_table(t)
    return make_features(player_game_table(p, tg), tg).sort_values("GAME_DATE").reset_index(drop=True)


def test_first_game_has_no_history(feats):
    assert np.isnan(feats.loc[0, "pts_avg_3"])
    assert feats.loc[0, "days_rest"] == 10


def test_rolling_uses_only_previous_games(feats):
    # mecz 3 (PTS=30): poprzednie 10, 20 → średnia 15, nie 20
    assert feats.loc[2, "pts_avg_3"] == pytest.approx(15.0)
    assert feats.loc[3, "pts_avg_3"] == pytest.approx(20.0)
    assert feats.loc[3, "min_last"] == 30
    assert feats.loc[1, "days_rest"] == 2


def test_h2h_excludes_current_game(feats):
    assert np.isnan(feats.loc[0, "pts_avg_vs_opp"])
    assert feats.loc[1, "pts_avg_vs_opp"] == pytest.approx(10.0)


def test_opponent_form_excludes_same_day(feats):
    # rolling drużyny wymaga min. 3 meczów → dopiero 4. mecz zawodnika ma wartość,
    # i to z meczów 1–3 (rywal BBB zdobył 100, 110, 120 przy 100 punktach drużyny AAA)
    assert feats["opp_def_rtg"].isna().sum() == 3
    assert not np.isnan(feats.loc[3, "opp_def_rtg"])


def test_future_rows_get_features():
    p, t = _logs()
    future = p.iloc[[-1]].copy()
    future["GAME_ID"] = "0022599999"
    future["GAME_DATE"] = pd.Timestamp("2025-01-09")
    for c in STAT_COLS:
        if c in future:
            future[c] = np.nan
    tg = team_game_table(t)
    games = player_game_table(pd.concat([p, future]), tg)
    out = make_features(games, tg).set_index("GAME_ID")
    assert out.loc["0022599999", "pts_avg_3"] == pytest.approx(30.0)
    assert out.loc["0022599999", "days_rest"] == 2


def test_absent_rotation_teammate_is_counted():
    """Kolega z rotacji (30 min) gra w meczach 1–3, w 4. go nie ma → vac_min > 0 tylko w meczu 4."""
    p, t = _logs()
    mate = p.iloc[:3].copy()
    mate["PLAYER_ID"], mate["PLAYER_NAME"], mate["PTS"] = 8, "Teammate", 20
    tg = team_game_table(t)
    f = make_features(player_game_table(pd.concat([p, mate]), tg), tg)
    me = f[f["PLAYER_ID"] == 7].sort_values("GAME_DATE").reset_index(drop=True)
    assert list(me["n_out"]) == [0, 0, 0, 1]
    assert me.loc[3, "vac_min"] == pytest.approx(30.0)
    assert me.loc[3, "vac_pts"] == pytest.approx(20.0)
    assert me.loc[3, "vac_pts_delta"] == pytest.approx(20.0)


def test_true_shooting_and_game_score_use_previous_games(feats):
    # TSA na mecz = 15 + 0.44·4 = 16.76; mecz 3 widzi tylko mecze 1–2 (PTS 10 + 20)
    assert np.isnan(feats.loc[0, "ts_10"])
    assert feats.loc[2, "ts_10"] == pytest.approx(30 / (2 * 2 * 16.76))
    assert feats.loc[2, "ts_season"] == pytest.approx(feats.loc[2, "ts_10"])
    # GmSc meczu 1: 10 + 2.4 − 10.5 − 0.4 + 0.7 + 1.2 + 1 + 2.1 + 0 − 0.8 − 2 = 3.7
    assert feats.loc[1, "gmsc_avg_10"] == pytest.approx(3.7)


def test_fatigue_and_travel_use_schedule_before_game():
    # mecze co 2 dni: 1.01 dom, 3.01 i 5.01 w Bostonie (wyjazd), 7.01 znów dom
    p, t = _logs()
    p.loc[[1, 2], "MATCHUP"] = "LAL @ BOS"
    tg = team_game_table(t)
    f = make_features(player_game_table(p, tg), tg).sort_values("GAME_DATE").reset_index(drop=True)
    assert list(f["games_7d"]) == [0, 1, 2, 3]
    assert list(f["min_7d"]) == [0, 30, 60, 90]
    assert list(f["is_3in4"]) == [0, 0, 0, 0]       # co 2 dni: nigdy 3 mecze w 4 dni
    assert list(f["road_trip"]) == [0, 1, 2, 0]
    # LA → Boston ≈ 4170 km, na wschód o 3 h; drugi mecz w Bostonie bez podróży
    assert f.loc[1, "travel_km"] == pytest.approx(4170, rel=0.02)
    assert f.loc[1, "tz_shift"] == 3
    assert f.loc[2, "travel_km"] == pytest.approx(0)
    assert f.loc[3, "tz_shift"] == -3
    assert f.loc[0, "travel_km"] == 0                # pierwszy mecz — brak poprzedniego


def test_three_in_four_nights():
    p, t = _logs()
    for df in (p, t):   # trzeci mecz dzień wcześniej: 1.01, 3.01, 4.01 → 3 mecze w 4 dni
        df.loc[df["GAME_DATE"] == "2025-01-05", "GAME_DATE"] = pd.Timestamp("2025-01-04")
    tg = team_game_table(t)
    f = make_features(player_game_table(p, tg), tg).sort_values("GAME_DATE").reset_index(drop=True)
    assert list(f["is_3in4"]) == [0, 0, 1, 0]


def test_game_context_from_team_ratings(feats):
    # mecze 1–3: LAL zdobywa 100, BOS 100/110/120, posiadania równe → LAL oczekuje 100,
    # BOS 110; LAL u siebie (+2) → oczekiwana różnica −8, suma 210
    row = feats.loc[3]
    assert row["team_exp_pts"] == pytest.approx(100.0)
    assert row["exp_total"] == pytest.approx(210.0)
    assert row["exp_margin"] == pytest.approx(-8.0)
    assert row["abs_margin"] == pytest.approx(8.0)
    assert feats["exp_margin"].isna().sum() == 3     # ratingi dopiero od 3 meczów drużyny
