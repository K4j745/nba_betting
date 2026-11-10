# nba_betting/upcoming.py
"""
Zakładka „Zawodnicy”: najbliższe mecze i ostatnie mecze zawodników z rotacji.

Terminarz z Odds API /events (bez kosztu kredytów), a gdy go brak, z otwartych rynków Kalshi; zawodnicy i statystyki z logów w data/raw.
Dla każdej drużyny TOP_N zawodników z największą średnią minut z ostatnich 10 meczów (zakres
projektu), którzy grali dla niej w ostatnich ABSENCE_DAYS dniach; dla każdego LAST_N ostatnich meczów.
Status Out/Doubtful z raportu kontuzji, jeśli był dostępny; linie główne Kalshi, jeśli są otwarte.

Wynik → upcoming.json, który main.export_dashboard_data dołącza do dashboard_data.json.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from history import ABSENCE_DAYS, norm_name

logger = logging.getLogger(__name__)

UPCOMING_PATH = Path("upcoming.json")
TOP_N         = 7
LAST_N        = 5
HORIZON_H     = 36


def _et_date(ts: pd.Timestamp) -> str:
    return ts.tz_convert("America/New_York").strftime("%Y-%m-%d")


def kalshi_lines() -> tuple[list[dict], dict]:
    """(mecze w formacie /events, {(name_key, stat): linia główna}) z otwartych rynków Kalshi."""
    import kalshi
    from live import TEAM_NAME_TO_ABBR

    abbr_to_name = {v: k for k, v in TEAM_NAME_TO_ABBR.items()}
    events, lines = {}, {}
    for stat in ("PTS", "REB", "AST"):
        df = kalshi.live_lines(stat)
        if df.empty:
            continue
        for r in df.dropna(subset=["tip"]).drop_duplicates("event_ticker").itertuples():
            events[(r.away, r.home)] = {"home_team": abbr_to_name.get(r.home, r.home),
                                        "away_team": abbr_to_name.get(r.away, r.away),
                                        "commence_time": r.tip.isoformat()}
        for r in kalshi.main_lines(df).itertuples():
            lines[(r.name_key, stat)] = r.line
    return list(events.values()), lines


def team_players(games: pd.DataFrame, team: str, day: pd.Timestamp, out_keys: set,
                 lines: dict | None = None) -> list[dict]:
    """games: tabela mecz-zawodnik (live.load_live_games). Rotacja drużyny przed dniem `day`."""
    g = games[games["GAME_DATE"] < day].sort_values("GAME_DATE")
    recent = g[(g["TEAM_ABBREVIATION"] == team) & (g["GAME_DATE"] >= day - pd.Timedelta(days=ABSENCE_DAYS))]
    if recent.empty:
        return []
    players = []
    for pid in recent["PLAYER_ID"].unique():
        own = g[(g["PLAYER_ID"] == pid) & (g["MIN"] > 0)]
        if own.empty:
            continue
        players.append((own["MIN"].tail(10).mean(), pid, own))
    players.sort(key=lambda x: x[0], reverse=True)
    id_to_abbr = games.drop_duplicates("TEAM_ID").set_index("TEAM_ID")["TEAM_ABBREVIATION"].to_dict()
    out = []
    for _, pid, own in players[:TOP_N]:
        last = own.tail(LAST_N).iloc[::-1]
        name = own["PLAYER_NAME"].iloc[-1]
        season = own[own["SEASON"] == own["SEASON"].iloc[-1]]
        out.append({
            "player": name,
            "id": int(pid),
            "status": "out" if (team, norm_name(name)) in out_keys else None,
            "lines": {st: (lines or {}).get((norm_name(name), st)) for st in ("PTS", "REB", "AST")},
            "avg5": {c: round(float(last[c].mean()), 1) for c in ("MIN", "PTS", "REB", "AST")},
            "season": {c: round(float(season[c].mean()), 1) for c in ("MIN", "PTS", "REB", "AST")},
            "games": [{"date": r.GAME_DATE.strftime("%Y-%m-%d"),
                       "opp": id_to_abbr.get(r.OPP_ID, "?"), "home": bool(r.IS_HOME),
                       "min": round(float(r.MIN)), "pts": int(r.PTS), "reb": int(r.REB), "ast": int(r.AST)}
                      for r in last.itertuples()],
        })
    return out


def build(events: list[dict], games: pd.DataFrame, out: pd.DataFrame | None = None,
          now: datetime | None = None, lines: dict | None = None) -> dict:
    """events: Odds API /events (home_team, away_team, commence_time)."""
    from live import TEAM_NAME_TO_ABBR

    now = pd.Timestamp(now or datetime.now(timezone.utc))
    out_keys = set() if out is None else set(zip(out["team"], out["name_key"]))
    result = []
    for e in sorted(events, key=lambda e: e["commence_time"]):
        tip = pd.Timestamp(e["commence_time"])
        if not now <= tip <= now + timedelta(hours=HORIZON_H):
            continue
        home, away = TEAM_NAME_TO_ABBR.get(e["home_team"]), TEAM_NAME_TO_ABBR.get(e["away_team"])
        if not home or not away:
            continue
        day = pd.Timestamp(_et_date(tip))
        result.append({
            "game_date": _et_date(tip), "tip": tip.isoformat(),
            "home": home, "away": away, "home_name": e["home_team"], "away_name": e["away_team"],
            "teams": {t: team_players(games, t, day, out_keys, lines) for t in (away, home)},
        })
    return {"generated": now.isoformat(), "injury_report": out is not None, "games": result}


def save(data: dict, path: Path = UPCOMING_PATH) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path = UPCOMING_PATH) -> dict | None:
    """Najbliższe mecze, tylko te, które jeszcze się nie zaczęły (plik bywa z poprzedniego runu)."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    data["games"] = [g for g in data.get("games", []) if pd.Timestamp(g["tip"]) > now - timedelta(hours=3)]
    return data
