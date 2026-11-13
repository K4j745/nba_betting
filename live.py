# nba_betting/live.py
"""
Pipeline „na żywo” na danych z repo (branch `data` → data/raw), bez stats.nba.com.

  build_live_dataset()  — zbiór z data/raw (tryb offline history.py)
  run_coupon()          — propsy z Odds API → cechy → PointsModel (trening w każdym runie,
                          ~5 s) → wszystkie predykcje do live_predictions, picki z EV ≥ EV_MIN
  actual_points_fn()    — punkty z logów do weryfikacji kuponów

Model live: cechy bazowe + absencje (history.TARGETS[stat]), gdy jest oficjalny raport kontuzji
NBA (injuries.py — Out/Doubtful przed meczem); bez raportu: tylko bazowe.
Statystyki: PTS (Odds API → tabela coupons, tracker) oraz REB/AST (Kalshi → stat_coupons,
weryfikacja wprost z logów w stat_results()).

⚠️ Ewaluacja (reports/eval_*) nie pokazała przewagi nad rynkiem — kupon to symulacja
(paper trading), która ma zbierać wyniki poza próbą, a nie rekomendacja zakładów.
"""
import json
import logging
import sqlite3

import numpy as np
import pandas as pd

from config import DB_PATH, ODDS_MAX, ODDS_MIN, STAKE_PER_PICK
from history import (
    ABSENCE_DAYS, STAT_COLS, TARGETS, TEAM_ABBR_TO_ID,
    load_logs, make_features, norm_name, player_game_table, season_str, team_game_table,
)
from points_model import train_points_model

logger = logging.getLogger(__name__)

EV_MIN        = 0.05   # minimalna wartość oczekiwana zakładu
STAT_ODDS     = (1.40, 2.60)   # REB/AST z Kalshi: linie główne ~50% → kurs ~1,9–2,1 po prowizji
STAT_LABEL    = {"PTS": "pkt", "REB": "zb.", "AST": "as."}
OPP_FEATURE   = {"PTS": "opp_def_rtg", "REB": "opp_reb_allowed", "AST": "opp_ast_allowed"}
MAX_STALE_DAYS = 2     # dane starsze niż tyle dni → brak kuponu (laptop nie zsynchronizował)

try:
    from nba_api.stats.static import teams as _nba_teams
    TEAM_NAME_TO_ABBR = {t["full_name"]: t["abbreviation"] for t in _nba_teams.get_teams()}
except ImportError:  # pragma: no cover
    TEAM_NAME_TO_ABBR = {}


# ─────────────────────────────────────────────────────────────────────────────
# Dane
# ─────────────────────────────────────────────────────────────────────────────
def load_live_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tabela mecz-zawodnik, tabela meczów drużyn) z data/raw — bez cech."""
    p, t = load_logs(offline=True)
    if p.empty:
        raise FileNotFoundError("Brak logów w data/raw — czy branch `data` został pobrany?")
    team_games = team_game_table(t)
    return player_game_table(p, team_games), team_games


def data_age_days(games: pd.DataFrame, today: str) -> int:
    return (pd.Timestamp(today) - games["GAME_DATE"].max()).days


# ─────────────────────────────────────────────────────────────────────────────
# Propsy → wiersze przyszłych meczów
# ─────────────────────────────────────────────────────────────────────────────
def season_of(game_date) -> str:
    """Sezon z daty meczu (od sierpnia nowy sezon) — nie z ostatniego meczu zawodnika, który na
    początku sezonu jest jeszcze z poprzedniego (źle liczyłoby games_in_season i poprzedni sezon)."""
    d = pd.Timestamp(game_date)
    return season_str(d.year if d.month >= 8 else d.year - 1)


def props_to_rows(props: list[dict], games: pd.DataFrame, game_date: str) -> pd.DataFrame:
    """
    Jeden wiersz na zawodnika z propsem (cechy liczone tak jak w treningu).
    Drużyna = drużyna z ostatniego meczu zawodnika; musi grać w tym evencie.
    """
    last = games.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    by_name = {norm_name(n): r for n, r in zip(last["PLAYER_NAME"], last.to_dict("records"))}

    rows, skipped = [], 0
    seen = set()
    for prop in props:
        r = by_name.get(norm_name(prop["player_name"]))
        home = TEAM_NAME_TO_ABBR.get(prop.get("home_team", ""))
        away = TEAM_NAME_TO_ABBR.get(prop.get("away_team", ""))
        if r is None or r["TEAM_ABBREVIATION"] not in (home, away):
            skipped += 1
            continue
        if r["PLAYER_ID"] in seen:
            continue
        seen.add(r["PLAYER_ID"])
        is_home = r["TEAM_ABBREVIATION"] == home
        opp = away if is_home else home
        rows.append({
            "SEASON": season_of(game_date), "IS_PLAYOFFS": r["IS_PLAYOFFS"],
            "GAME_ID": f"LIVE-{game_date}-{home}-{away}",
            "GAME_DATE": pd.Timestamp(game_date),
            "PLAYER_ID": r["PLAYER_ID"], "PLAYER_NAME": r["PLAYER_NAME"],
            "TEAM_ID": r["TEAM_ID"], "TEAM_ABBREVIATION": r["TEAM_ABBREVIATION"],
            "OPP_ID": TEAM_ABBR_TO_ID[opp], "IS_HOME": int(is_home),
            **{c: np.nan for c in STAT_COLS},
        })
    if skipped:
        logger.info("props_to_rows: pominięto %d propsów (brak zawodnika w logach lub inna drużyna)", skipped)
    return pd.DataFrame(rows)


def roster_fillers(games: pd.DataFrame, future: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """
    Wiersze zastępcze dla meczów na żywo: każdy, kto grał dla drużyny w ostatnich ABSENCE_DAYS
    dniach i NIE jest Out/Doubtful w raporcie kontuzji, „gra” dziś. Dzięki temu team_absences
    (to samo co w treningu) widzi jako nieobecnych tylko zawodników z raportu — bez tego każdy
    zawodnik bez propsa wyglądałby na nieobecnego.
    out: kolumny team (skrót), name_key.
    """
    if future.empty:
        return future
    day = future["GAME_DATE"].iloc[0]
    recent = games[(games["GAME_DATE"] < day) & (games["GAME_DATE"] >= day - pd.Timedelta(days=ABSENCE_DAYS))]
    last = recent.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    out_keys = set(zip(out["team"], out["name_key"]))
    rows = []
    for (gid, tid), g in future.groupby(["GAME_ID", "TEAM_ID"]):
        ref = g.iloc[0]
        cand = last[(last["TEAM_ID"] == tid) & ~last["PLAYER_ID"].isin(g["PLAYER_ID"])]
        for r in cand.itertuples():
            if (r.TEAM_ABBREVIATION, norm_name(r.PLAYER_NAME)) in out_keys:
                continue
            row = ref.to_dict()
            row.update(PLAYER_ID=r.PLAYER_ID, PLAYER_NAME=r.PLAYER_NAME, SEASON=r.SEASON, _FILLER=True)
            rows.append(row)
    return pd.DataFrame(rows)


def live_features(props: list[dict], games: pd.DataFrame, team_games: pd.DataFrame,
                  game_date: str, out: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (historia z cechami do treningu, dzisiejsze wiersze z cechami) — jedno liczenie cech.
    out = Out/Doubtful z raportu kontuzji (None → cechy absencji dla dzisiejszych meczów
    są bez znaczenia i model ich nie używa).
    """
    future = props_to_rows(props, games, game_date)
    parts = [games, future]
    if out is not None and not future.empty:
        parts.append(roster_fillers(games, future, out))
    feats = make_features(pd.concat(parts, ignore_index=True), team_games)
    is_live = feats["GAME_ID"].str.startswith("LIVE-")
    if "_FILLER" in feats:
        keep = feats["_FILLER"].fillna(False).astype(bool)
        feats = feats[~keep].drop(columns="_FILLER")
        is_live = is_live[~keep]
    return feats[~is_live].reset_index(drop=True), feats[is_live].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Kupon
# ─────────────────────────────────────────────────────────────────────────────
def _save_predictions(rows: list[dict], db_path: str):
    if not rows:
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_predictions (
                game_date TEXT, player_name TEXT, player_id INTEGER, game TEXT,
                bookmaker TEXT, line REAL, over_odds REAL, under_odds REAL,
                mu REAL, p_over REAL, best_side TEXT, best_ev REAL,
                model_cutoff TEXT, created_at TEXT DEFAULT (datetime('now')),
                stat TEXT DEFAULT 'PTS'
            )""")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(live_predictions)")}
        if "stat" not in cols:   # baza sprzed REB/AST
            conn.execute("ALTER TABLE live_predictions ADD COLUMN stat TEXT DEFAULT 'PTS'")
        pd.DataFrame(rows).to_sql("live_predictions", conn, if_exists="append", index=False)


def run_coupon(props: list[dict], game_date: str, db_path: str = DB_PATH,
               stake: float = STAKE_PER_PICK, out: pd.DataFrame | None = None,
               stat: str = "PTS") -> dict:
    """
    Zwraca dict kuponu (format zgodny z tracker/dashboard) albo
    {"error": "<powód>"} gdy dane są nieaktualne lub żaden prop nie pasuje.
    stat = PTS → tabela coupons (tracker); REB/AST → stat_coupons (kursy z Kalshi).
    """
    games, team_games = load_live_games()
    age = data_age_days(games, game_date)
    if age > MAX_STALE_DAYS:
        logger.error("Dane NBA sprzed %d dni (ostatni mecz %s)", age, games["GAME_DATE"].max().date())
        return {"error": "stale", "data_age_days": age}

    dataset, feats = live_features(props, games, team_games, game_date, out=out)
    if feats.empty:
        return {"error": "no_features"}

    base, absence = TARGETS[stat]
    features = base + absence if out is not None else base
    model = train_points_model(dataset, cutoff=game_date, features=features, target=stat)
    odds_min, odds_max = (ODDS_MIN, ODDS_MAX) if stat == "PTS" else STAT_ODDS
    lo, opp, unit = stat.lower(), OPP_FEATURE[stat], STAT_LABEL[stat]
    feats["mu"] = model.mu(feats)
    by_name = {norm_name(n): i for i, n in enumerate(feats["PLAYER_NAME"])}

    preds, candidates = [], []
    for prop in props:
        i = by_name.get(norm_name(prop["player_name"]))
        if i is None:
            continue
        row = feats.iloc[[i]]
        p_over = float(model.p_over(row, prop["line"])[0])
        ev_over  = p_over * prop["over_odds"] - 1
        ev_under = (1 - p_over) * prop["under_odds"] - 1
        side = "over" if ev_over >= ev_under else "under"
        ev, odds = (ev_over, prop["over_odds"]) if side == "over" else (ev_under, prop["under_odds"])
        game = f"{prop.get('away_team', '?')} @ {prop.get('home_team', '?')}"
        mu = float(row["mu"].iloc[0])
        preds.append({
            "game_date": game_date, "player_name": row["PLAYER_NAME"].iloc[0],
            "player_id": int(row["PLAYER_ID"].iloc[0]), "game": game,
            "bookmaker": prop.get("bookmaker"), "line": prop["line"],
            "over_odds": prop["over_odds"], "under_odds": prop["under_odds"],
            "mu": round(mu, 2), "p_over": round(p_over, 4), "best_side": side,
            "best_ev": round(ev, 4), "model_cutoff": game_date, "stat": stat,
        })
        if ev >= EV_MIN and odds_min <= odds <= odds_max:
            p_side = p_over if side == "over" else 1 - p_over
            r = row.iloc[0]
            candidates.append({
                "game": game, "player": r["PLAYER_NAME"], "bet": side, "stat": stat,
                "line": prop["line"], "odds": odds, "bookmaker": prop.get("bookmaker"),
                "pts_predicted": round(mu, 1), "p_over": round(p_over, 3),
                "confidence": int(round(100 * p_side)), "ev": round(ev, 4),
                "p_market": prop.get("p_market"),
                "top_features": [f"{lo}_ewm", "min_avg_10", opp],
                "reasoning": (
                    f"model {mu:.1f} {unit} vs linia {prop['line']} ({prop.get('bookmaker')}), "
                    f"ewm {r[f'{lo}_ewm']:.1f}, min {r['min_avg_10']:.1f}, "
                    f"{opp} {r[opp]:.1f}, EV {ev:+.1%}"
                ),
            })
    _save_predictions(preds, db_path)

    best: dict[str, dict] = {}
    for c in candidates:   # max 1 pick na mecz — najwyższe EV
        if c["game"] not in best or c["ev"] > best[c["game"]]["ev"]:
            best[c["game"]] = c
    picks = sorted(best.values(), key=lambda c: c["ev"], reverse=True)

    coupon = {
        "date": game_date,
        "model_version": f"{stat.lower()}-{game_date}",
        "stat": stat,
        "paper_trading": True,
        "injury_report": out is not None,
        "n_out": None if out is None else int(len(out)),
        "n_predictions": len(preds),
        "picks": picks,
        "total_picks": len(picks),
        "simulated_stake_per_pick": stake,
        "simulated_total_stake": round(stake * len(picks), 2),
    }
    with sqlite3.connect(db_path) as conn:
        if stat == "PTS":
            conn.execute("""CREATE TABLE IF NOT EXISTS coupons (
                id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, coupon_json TEXT,
                created_at TEXT DEFAULT (datetime('now')))""")
            conn.execute("INSERT INTO coupons (date, coupon_json) VALUES (?, ?)",
                         (game_date, json.dumps(coupon, ensure_ascii=False)))
        else:
            conn.execute(STAT_COUPONS_DDL)
            conn.execute("INSERT INTO stat_coupons (date, stat, coupon_json) VALUES (?, ?, ?)",
                         (game_date, stat, json.dumps(coupon, ensure_ascii=False)))
    logger.info("Kupon %s %s: %d predykcji, %d picków (EV ≥ %.0f%%)",
                stat, game_date, len(preds), len(picks), EV_MIN * 100)
    return coupon


# ─────────────────────────────────────────────────────────────────────────────
# Weryfikacja
# ─────────────────────────────────────────────────────────────────────────────
STAT_COUPONS_DDL = """CREATE TABLE IF NOT EXISTS stat_coupons (
    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, stat TEXT, coupon_json TEXT,
    created_at TEXT DEFAULT (datetime('now')))"""


def actual_points_fn():
    """Funkcja (player_name, 'YYYY-MM-DD') → punkty albo None, z logów w data/raw."""
    p, _ = load_logs(offline=True)
    p = p.assign(_k=p["PLAYER_NAME"].map(norm_name), _d=p["GAME_DATE"].dt.date.astype(str))
    lookup = {(k, d): float(v) for k, d, v in zip(p["_k"], p["_d"], p["PTS"])}
    return lambda name, d: lookup.get((norm_name(name), d))


def actual_stat_fn(stat: str):
    """Funkcja (player_name, 'YYYY-MM-DD') → wartość statystyki albo None, z logów w data/raw."""
    p, _ = load_logs(offline=True)
    p = p.assign(_k=p["PLAYER_NAME"].map(norm_name), _d=p["GAME_DATE"].dt.date.astype(str))
    lookup = {(k, d): float(v) for k, d, v in zip(p["_k"], p["_d"], p[stat])}
    return lambda name, d: lookup.get((norm_name(name), d))


def stat_results(stat: str, db_path: str = DB_PATH, actual_fn=None) -> list[dict]:
    """
    Picki z ostatniego kuponu REB/AST każdego dnia + wynik z logów (paper trading, stawka 1).
    Bez osobnej tabeli wyników: liczone przy eksporcie, więc późne logi same się „dopisują”.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(STAT_COUPONS_DDL)
        rows = conn.execute("""SELECT date, coupon_json FROM stat_coupons WHERE stat = ?
                               AND id IN (SELECT MAX(id) FROM stat_coupons GROUP BY date, stat)
                               ORDER BY date""", (stat,)).fetchall()
    if not rows:
        return []
    actual_fn = actual_fn or actual_stat_fn(stat)
    out = []
    for d, cj in rows:
        for pick in json.loads(cj).get("picks", []):
            a = actual_fn(pick["player"], d)
            res = None if a is None else ("WIN" if (a > pick["line"]) == (pick["bet"] == "over") else "LOSS")
            out.append({**pick, "date": d, "actual": a, "result": res,
                        "profit": None if res is None else round(pick["odds"] - 1 if res == "WIN" else -1.0, 3)})
    return out
