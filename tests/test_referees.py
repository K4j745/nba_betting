import numpy as np
import pandas as pd

import referees as rf


def _data(n_days=80):
    """Po jednym meczu dziennie, 2 drużyny × 1 zawodnik; zbiórki rosną z sędzią 7."""
    rows = []
    for i in range(n_days):
        gid = f"g{i:03d}"
        big = i % 2 == 0                                  # sędzia 7 prowadzi mecze parzyste
        for team in (1, 2):
            rows.append({"GAME_ID": gid, "GAME_DATE": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                         "PLAYER_ID": team, "REB": 30 if big else 20, "AST": 10, "PF": 10,
                         "FGA": 40, "FTA": 10, "OREB": 5, "TOV": 7})
    data = pd.DataFrame(rows)
    off = pd.DataFrame([{"GAME_ID": f"g{i:03d}", "OFFICIAL_ID": 7 if i % 2 == 0 else 8, "OFFICIAL": "x"}
                        for i in range(n_days)])
    return data, off


def test_game_totals_baseline_uses_only_earlier_days():
    data, _ = _data()
    t = rf.game_totals(data)
    # rozbieg: < 50 wcześniejszych meczów → brak bazy
    assert t["res_ref_reb"].iloc[:50].isna().all()
    # baza = średnia wcześniejszych meczów (naprzemiennie 60 i 40 zbiórek) = 50
    row = t.iloc[60]
    assert abs((row["REB"] - row["res_ref_reb"]) - 50) < 1e-9


def test_referee_effect_point_in_time_and_shrunk():
    data, off = _data()
    f = rf.referee_features(rf.game_totals(data), off).set_index("GAME_ID")
    # sędzia 7 (mecze z 60 zbiórkami) ma dodatni efekt, sędzia 8 ujemny
    assert f.loc["g078", "ref_reb"] > 0 > f.loc["g079", "ref_reb"]
    # pierwszy mecz z bazą: sędzia nie ma jeszcze wcześniejszych reszt → 0
    assert f.loc["g050", "ref_reb"] == 0
    # efekt ściągnięty: |efekt| < surowa reszta (10)
    assert abs(f.loc["g078", "ref_reb"]) < 10
    # bez wycieku: efekt meczu nie zależy od jego własnego wyniku
    data2 = data.copy()
    data2.loc[data2["GAME_ID"] == "g078", "REB"] = 500
    f2 = rf.referee_features(rf.game_totals(data2), off).set_index("GAME_ID")
    assert np.isclose(f.loc["g078", "ref_reb"], f2.loc["g078", "ref_reb"])
