from datetime import datetime, timezone

import pandas as pd

import upcoming

DAY = pd.Timestamp("2026-11-10")


def _games():
    rows = []
    for i in range(8):
        d = DAY - pd.Timedelta(days=8 - i)
        for pid, name, mins in ((1, "Star Player", 36), (2, "Bench Guy", 12), (3, "Old Player", 30)):
            if pid == 3 and i > 0:          # grał dawno temu, nie należy do obecnej rotacji
                continue
            rows.append({"GAME_ID": f"g{i}", "GAME_DATE": d - pd.Timedelta(days=30 if pid == 3 else 0),
                         "PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ID": 10, "TEAM_ABBREVIATION": "BOS",
                         "OPP_ID": 20, "IS_HOME": i % 2, "SEASON": "2026-27",
                         "MIN": mins, "PTS": 20 + i, "REB": 5, "AST": 3})
    rows.append({**rows[0], "TEAM_ID": 20, "TEAM_ABBREVIATION": "NYK", "PLAYER_ID": 9, "PLAYER_NAME": "Rival"})
    return pd.DataFrame(rows)


def test_rotation_sorted_by_minutes_last_five_newest_first():
    p = upcoming.team_players(_games(), "BOS", DAY, set())
    assert [x["player"] for x in p] == ["Star Player", "Bench Guy"]
    assert len(p[0]["games"]) == 5 and p[0]["games"][0]["pts"] == 27 and p[0]["games"][0]["opp"] == "NYK"
    assert p[0]["avg5"]["PTS"] == 25.0


def test_injury_report_marks_player_out():
    p = upcoming.team_players(_games(), "BOS", DAY, {("BOS", "star player")})
    assert p[0]["status"] == "out" and p[1]["status"] is None


def test_build_keeps_only_games_in_horizon():
    ev = [{"home_team": "Boston Celtics", "away_team": "New York Knicks", "commence_time": "2026-11-11T00:30:00Z"},
          {"home_team": "Boston Celtics", "away_team": "New York Knicks", "commence_time": "2026-11-14T00:30:00Z"}]
    d = upcoming.build(ev, _games(), now=datetime(2026, 11, 10, 12, tzinfo=timezone.utc))
    assert len(d["games"]) == 1 and d["games"][0]["game_date"] == "2026-11-10"
    assert d["games"][0]["home"] == "BOS" and list(d["games"][0]["teams"]) == ["NYK", "BOS"]
