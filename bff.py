# nba_betting/bff.py
"""
bff / no_bff: jak zawodnik gra bez KONKRETNEGO kolegi, zamiast ogólnych vac_*.

Efekt pary (zawodnik P, kolega T) tylko z meczów sprzed dnia linii, w oknie WINDOW_DAYS,
w okresie, gdy T był w drużynie:

    efekt = (średnia P bez T − średnia P z T) · n_bez / (n_bez + SHRINK)

Ściąganie do zera, bo meczów bez T bywa kilka. Cecha `bff` to suma efektów kolegów z rotacji
oznaczonych Out/Doubtful w raporcie z chwili ceny (start − 1 h), na 10 jednostek statystyki.
Test: market_tests.evaluate, względem samej rekalibracji rynku.

Użycie:
  python bff.py            # → reports/bff_<data>.json
  python bff.py --final
"""
import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

import injuries
import market_model as mm
import market_tests as mt
from feature_lab import FINAL_FROM
from history import ABSENCE_DAYS, ROTATION_MIN, load_logs, norm_name

logger = logging.getLogger(__name__)

SHRINK      = 5
WINDOW_DAYS = 400


def pair_effect(games: pd.DataFrame, player: int, mate: int, stat: str, day: pd.Timestamp) -> float:
    """games: logi jednej drużyny. Efekt nieobecności `mate` dla `player` przed dniem `day`."""
    past = games[(games["GAME_DATE"] < day) & (games["GAME_DATE"] >= day - pd.Timedelta(days=WINDOW_DAYS))]
    m_games = past.loc[past["PLAYER_ID"] == mate, "GAME_DATE"]
    if m_games.empty:
        return 0.0
    mine = past[(past["PLAYER_ID"] == player) & past["GAME_DATE"].between(m_games.min(), m_games.max())]
    without = ~mine["GAME_ID"].isin(past.loc[past["PLAYER_ID"] == mate, "GAME_ID"])
    n = int(without.sum())
    if n == 0 or n == len(mine):
        return 0.0
    return float((mine.loc[without, stat].mean() - mine.loc[~without, stat].mean()) * n / (n + SHRINK))


def add_bff(pred: pd.DataFrame) -> pd.DataFrame:
    logs, _ = load_logs(offline=True)
    logs = logs.sort_values("GAME_DATE").copy()
    logs["name_key"] = logs["PLAYER_NAME"].map(norm_name)
    logs["MIN_10"] = logs.groupby("PLAYER_ID")["MIN"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    by_team = dict(tuple(logs.groupby("TEAM_ID")))
    team_of = logs.set_index(["PLAYER_ID", "GAME_ID"])[["TEAM_ID", "TEAM_ABBREVIATION"]]

    reports = injuries.load_report_history()
    rep = reports[reports["status"].isin(injuries.OUT_STATUSES)]
    out_by = rep.groupby(["snapshot", "game_date", "team"])["name_key"].apply(set).to_dict()
    have_snap = set(reports["snapshot"].astype(str))

    pred = pred.copy()
    pred["PLAYER_ID"] = pred["PLAYER_ID"].astype("Int64")
    pred = pred.join(team_of, on=["PLAYER_ID", "GAME_ID"])
    cache, rows = {}, []
    for r in pred.itertuples():
        snap = injuries._snap_key((pd.Timestamp(r.tip) - pd.Timedelta(hours=1)).to_pydatetime()) if pd.notna(r.tip) else None
        total = 0.0
        out = out_by.get((snap, r.game_date, r.TEAM_ABBREVIATION), set()) if pd.notna(r.TEAM_ID) else set()
        if out:
            games = by_team[r.TEAM_ID]
            key = (r.TEAM_ID, r.game_date)
            if key not in cache:
                win = games[(games["GAME_DATE"] < r.game_date)
                            & (games["GAME_DATE"] >= r.game_date - pd.Timedelta(days=ABSENCE_DAYS))]
                last = win.groupby("PLAYER_ID").tail(1)
                cache[key] = last[last["MIN_10"] >= ROTATION_MIN]
            rot = cache[key]
            mates = rot.loc[rot["name_key"].isin(out) & (rot["PLAYER_ID"] != r.PLAYER_ID), "PLAYER_ID"]
            total = sum(pair_effect(games, r.PLAYER_ID, t, r.stat, r.game_date) for t in mates)
        rows.append({"has_report": snap in have_snap, "bff": total / 10})
    return pd.concat([pred.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def run(final: bool = False) -> dict:
    rng = np.random.default_rng(0)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    m = add_bff(pred)
    m = m[m["has_report"]].assign(lm=lambda d: mm._logit(d["p_mid"]))
    report = {"generated": date.today().isoformat(), "sample": "final" if final else "dev",
              "final_from": str(FINAL_FROM.date()), **mt.evaluate(m, final, ("bff",), rng)}
    out = mm.REPORT_DIR / f"bff_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    run(ap.parse_args().final)
