import sqlite3

import pandas as pd

import daily


def _preds():
    return pd.DataFrame([
        # mecz A: pewny under zbiórek (p 0,70) z EV > 0, ryzykowny over punktów z wyższym EV
        dict(game="A", player_name="X", stat="REB", line=8.5, p_over=0.30, over_odds=2.4, under_odds=1.55,
             bookmaker="kalshi", mu=7.0, player_id=1),
        dict(game="A", player_name="Y", stat="PTS", line=20.5, p_over=0.45, over_odds=2.9, under_odds=1.4,
             bookmaker="pinnacle", mu=21.0, player_id=2),
        # ta sama linia Y u drugiego bukmachera z lepszym kursem over
        dict(game="A", player_name="Y", stat="PTS", line=20.5, p_over=0.45, over_odds=2.95, under_odds=1.35,
             bookmaker="unibet", mu=21.0, player_id=2),
        # mecz B: nic z EV ≥ 0
        dict(game="B", player_name="Z", stat="AST", line=5.5, p_over=0.5, over_odds=1.9, under_odds=1.9,
             bookmaker="kalshi", mu=5.5, player_id=3),
    ])


def test_candidates_keep_best_odds_per_line_and_side():
    c = daily.candidates(_preds())
    y_over = c[(c["player_name"] == "Y") & (c["bet"] == "over")]
    assert len(y_over) == 1 and y_over["bookmaker"].iloc[0] == "unibet"


def test_pick_confidence_vs_ev_rule():
    c = daily.candidates(_preds())
    conf = daily.pick(c, "confidence")
    ev = daily.pick(c, "ev")
    assert list(conf["game"]) == ["A"]                      # mecz B bez kandydata z EV ≥ 0
    assert conf.iloc[0]["player_name"] == "X" and conf.iloc[0]["bet"] == "under"   # p 0,70
    assert ev.iloc[0]["player_name"] == "Y" and ev.iloc[0]["bet"] == "over"        # EV 0,45·2,95−1 = 0,33


def test_build_and_history_from_live_predictions(tmp_path):
    db = str(tmp_path / "t.db")
    p = _preds().assign(game_date="2026-11-10", created_at="2026-11-10 21:00:00", best_side="", best_ev=0.0,
                        model_cutoff="2026-11-10")
    with sqlite3.connect(db) as conn:
        p.to_sql("live_predictions", conn, index=False)
    d = daily.build("2026-11-10", db)
    assert d["n_games"] == 2 and len(d["picks"]) == 1 and d["picks"][0]["confidence"] == 0.7
    h = daily.history(db, actual_fns={"REB": lambda name, day: 6.0}, before="2026-11-11")
    assert h[0]["result"] == "WIN" and h[0]["profit"] == 0.55
