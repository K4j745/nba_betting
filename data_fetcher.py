# nba_betting/data_fetcher.py
import time
import random
import sqlite3
import logging
import pickle
from datetime import datetime, date
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import (
    scoreboardv2,
    playergamelog,
    leaguedashteamstats,
    playerdashboardbygeneralsplits,
    leaguedashptdefend,
)
from nba_api.stats.static import players as nba_players, teams as nba_teams

from config import (
    DB_PATH, CURRENT_SEASON, SEASON_TYPE,
    TTL_GAME_LOG, TTL_SEASON_AVG, TTL_TEAM_DEF, TTL_BOX_SCORE,
    REQUEST_DELAY_MIN, REQUEST_DELAY_MAX,
)

logger = logging.getLogger(__name__)

# ── Team lookup tables ────────────────────────────────────────────────────────
_TEAM_ABB_TO_ID = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
_TEAM_ID_TO_ABB = {v: k for k, v in _TEAM_ABB_TO_ID.items()}


def _rand_sleep():
    """Rate-limiting: losowe opóźnienie 0.6–1.0s między requestami."""
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    logger.debug("Rate-limit sleep: %.2fs", delay)
    time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# SQLiteCache — key-value store z TTL
# ─────────────────────────────────────────────────────────────────────────────
class SQLiteCache:
    """
    Cache oparty na SQLite.  Dane serializowane przez pickle.
    Każdy wpis ma timestamp creation — get() sprawdza TTL automatycznie.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cache (
                    key        TEXT PRIMARY KEY,
                    data       BLOB    NOT NULL,
                    created_at REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS odds_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT,
                    stat_type   TEXT,
                    line        REAL,
                    over_odds   REAL,
                    under_odds  REAL,
                    home_team   TEXT,
                    away_team   TEXT,
                    game_time   TEXT,
                    fetched_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS coupons (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        TEXT,
                    coupon_json TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                );

            """)
        logger.info("SQLite initialized → %s", self.db_path)

    def get(self, key: str, ttl: int) -> Optional[pd.DataFrame]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data, created_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            logger.debug("Cache MISS (no entry): %s", key)
            return None
        data, created_at = row
        if (time.time() - created_at) > ttl:
            logger.debug("Cache MISS (expired): %s", key)
            return None
        logger.debug("Cache HIT: %s", key)
        return pickle.loads(data)

    def set(self, key: str, df: pd.DataFrame):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, data, created_at) "
                "VALUES (?, ?, ?)",
                (key, pickle.dumps(df), time.time()),
            )
            conn.commit()
        logger.debug("Cache SET: %s (%d rows)", key, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# NBADataFetcher — główna klasa Modułu 1
# ─────────────────────────────────────────────────────────────────────────────
class NBADataFetcher:
    """
    Centralny klient nba_api z lokalnym cache SQLite.

    Przykład użycia:
        fetcher = NBADataFetcher()
        games   = fetcher.get_todays_games()
        stats   = fetcher.get_player_recent_stats(player_id=1629029, last_n=20)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.cache = SQLiteCache(db_path)
        logger.info("NBADataFetcher ready (db=%s)", db_path)

    # ── 1. Dzisiejsze mecze ──────────────────────────────────────────────────
    def get_todays_games(self) -> list[dict]:
        """
        Pobiera mecze na dziś z ScoreboardV2.
        Zwraca: [{game_id, home_team, away_team, home_team_id,
                  away_team_id, game_time_et, status}]
        TTL: 1h (mecze nie zmieniają się w ciągu dnia, ale status tak)
        """
        today_str = date.today().isoformat()
        cache_key = f"todays_games_{today_str}"
        cached = self.cache.get(cache_key, ttl=3600)
        if cached is not None:
            return cached.to_dict("records")

        _rand_sleep()
        board = scoreboardv2.ScoreboardV2(
            game_date=today_str,
            league_id="00",
            day_offset=0,
        )
        games_df   = board.game_header.get_data_frame()
        line_score = board.line_score.get_data_frame()

        if games_df.empty:
            logger.warning("Brak meczów na %s", today_str)
            return []

        results = []
        for _, row in games_df.iterrows():
            gid = row["GAME_ID"]
            home_ls = line_score[
                (line_score["GAME_ID"] == gid) &
                (line_score["TEAM_ID"] == row["HOME_TEAM_ID"])
            ]
            away_ls = line_score[
                (line_score["GAME_ID"] == gid) &
                (line_score["TEAM_ID"] == row["VISITOR_TEAM_ID"])
            ]
            results.append({
                "game_id":      gid,
                "home_team":    home_ls["TEAM_ABBREVIATION"].values[0]
                                if not home_ls.empty else "?",
                "away_team":    away_ls["TEAM_ABBREVIATION"].values[0]
                                if not away_ls.empty else "?",
                "home_team_id": int(row["HOME_TEAM_ID"]),
                "away_team_id": int(row["VISITOR_TEAM_ID"]),
                "game_time_et": str(row.get("GAME_STATUS_TEXT", "")),
                "status":       str(row.get("GAME_STATUS_TEXT", "")),
            })

        self.cache.set(cache_key, pd.DataFrame(results))
        logger.info("Pobrano %d meczów na %s", len(results), today_str)
        return results

    # ── 2. Ostatnie mecze zawodnika ──────────────────────────────────────────
    def get_player_recent_stats(
        self, player_id: int, last_n: int = 20
    ) -> pd.DataFrame:
        """
        Zwraca DataFrame z ostatnimi `last_n` meczami zawodnika.

        Kolumny wyjściowe:
            GAME_DATE, GAME_ID, PTS, REB, AST, MIN, FGA, FGM, FTA, FTM,
            USG_PCT, opponent_team_id, is_home, days_rest, is_back_to_back

        Uwagi:
        - USG_PCT pochodzi z sezonu bieżącego (brak per-game w API).
        - days_rest = 0 to B2B, clipped do 7.
        """
        cache_key = f"recent_stats_{player_id}_{last_n}_{CURRENT_SEASON}"
        cached = self.cache.get(cache_key, ttl=TTL_GAME_LOG)
        if cached is not None:
            return cached

        # ── Pobierz game log ──────────────────────────────
        _rand_sleep()
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=CURRENT_SEASON,
            season_type_all_star=SEASON_TYPE,
        )
        df = gl.get_data_frames()[0]

        if df.empty:
            logger.warning("Brak game log dla player_id=%d", player_id)
            return pd.DataFrame()

        df = df.sort_values("GAME_DATE", ascending=False).head(last_n).copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # ── is_home + opponent ────────────────────────────
        # MATCHUP: "LAL vs. GSW" = domowy | "LAL @ GSW" = wyjazdowy
        df["is_home"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
        df["opp_abb"] = df["MATCHUP"].apply(
            lambda x: x.split(" vs. ")[-1] if "vs." in x
                      else x.split(" @ ")[-1]
        )
        df["opponent_team_id"] = df["opp_abb"].map(
            lambda a: _TEAM_ABB_TO_ID.get(a, 0)
        )

        # ── days_rest + back-to-back ──────────────────────
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        df["days_rest"] = (
            df["GAME_DATE"].diff()
            .dt.days
            .fillna(3)
            .clip(upper=7)
            .astype(int)
        )
        df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
        df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

        # ── USG_PCT (sezonowa) ────────────────────────────
        _rand_sleep()
        try:
            adv = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                player_id=player_id,
                season=CURRENT_SEASON,
                measure_type_detailed="Usage",
                per_mode_detailed="PerGame",
            )
            adv_df = adv.get_data_frames()[0]
            usg = float(adv_df["USG_PCT"].iloc[0]) if not adv_df.empty else 0.0
        except Exception as e:
            logger.warning("USG_PCT fetch failed: %s", e)
            usg = 0.0
        df["USG_PCT"] = usg

        keep = [
            "GAME_DATE", "GAME_ID", "PTS", "REB", "AST", "MIN",
            "FGA", "FGM", "FTA", "FTM", "USG_PCT",
            "opponent_team_id", "is_home", "days_rest", "is_back_to_back",
        ]
        df = df[[c for c in keep if c in df.columns]]

        self.cache.set(cache_key, df)
        logger.info(
            "player_id=%d: %d meczów pobrano (USG%%=%.1f%%)",
            player_id, len(df), usg * 100,
        )
        return df

    # ── 3. Historia vs konkretny rywal ───────────────────────────────────────
    def get_player_vs_opponent(
        self, player_id: int, opponent_team_id: int, last_n: int = 10
    ) -> pd.DataFrame:
        """
        Historyczne mecze zawodnika vs konkretny rywal (ostatnie 3 sezony).
        Cache permanentny (TTL_BOX_SCORE=24h), dane archiwalne się nie zmieniają.
        """
        cache_key = f"h2h_{player_id}_{opponent_team_id}"
        cached = self.cache.get(cache_key, ttl=TTL_BOX_SCORE)
        if cached is not None:
            return cached.head(last_n)

        all_frames = []
        current_year = int(CURRENT_SEASON[:4])

        for offset in range(3):
            year = current_year - offset
            season_str = f"{year}-{str(year + 1)[-2:]}"
            _rand_sleep()
            try:
                gl = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season_str,
                    season_type_all_star=SEASON_TYPE,
                )
                df_s = gl.get_data_frames()[0]
                if not df_s.empty:
                    all_frames.append(df_s)
            except Exception as exc:
                logger.warning("Sezon %s niedostępny: %s", season_str, exc)

        if not all_frames:
            logger.warning(
                "Brak danych H2H: player=%d vs team=%d",
                player_id, opponent_team_id,
            )
            return pd.DataFrame()

        full_df = pd.concat(all_frames, ignore_index=True)
        full_df["GAME_DATE"] = pd.to_datetime(full_df["GAME_DATE"])

        full_df["opp_abb"] = full_df["MATCHUP"].apply(
            lambda x: x.split(" vs. ")[-1] if "vs." in x
                      else x.split(" @ ")[-1]
        )
        target_abb = _TEAM_ID_TO_ABB.get(opponent_team_id, "")
        vs_df = (
            full_df[full_df["opp_abb"] == target_abb]
            .sort_values("GAME_DATE", ascending=False)
            .reset_index(drop=True)
        )

        self.cache.set(cache_key, vs_df)
        logger.info(
            "H2H player=%d vs team=%d: %d meczów (zwracam %d)",
            player_id, opponent_team_id, len(vs_df), min(len(vs_df), last_n),
        )
        return vs_df.head(last_n)

    # ── 4. Średnie sezonowe ──────────────────────────────────────────────────
    def get_player_season_avg(self, player_id: int) -> dict:
        """
        Zwraca słownik ze średnimi sezonowymi.
        Łączy Base + Advanced (USG_PCT) z PlayerDashboardByGeneralSplits.
        """
        cache_key = f"season_avg_{player_id}_{CURRENT_SEASON}"
        cached = self.cache.get(cache_key, ttl=TTL_SEASON_AVG)
        if cached is not None:
            return cached.iloc[0].to_dict()

        _rand_sleep()
        base = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=CURRENT_SEASON,
            measure_type_detailed="Base",
            per_mode_detailed="PerGame",
        )
        base_df = base.get_data_frames()[0]
        if base_df.empty:
            logger.warning("Brak season avg dla player_id=%d", player_id)
            return {}

        _rand_sleep()
        adv = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=CURRENT_SEASON,
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame",
        )
        adv_df = adv.get_data_frames()[0]

        cols = ["PTS", "REB", "AST", "MIN", "FGA", "FGM", "FTA", "FTM", "GP"]
        result_df = base_df[[c for c in cols if c in base_df.columns]].copy()
        if not adv_df.empty and "USG_PCT" in adv_df.columns:
            result_df["USG_PCT"] = float(adv_df["USG_PCT"].iloc[0])

        self.cache.set(cache_key, result_df)
        logger.info("Sezon avg: player_id=%d → PTS=%.1f",
                    player_id, float(result_df["PTS"].iloc[0]))
        return result_df.iloc[0].to_dict()

    # ── 5. Statystyki defensywne drużyny ────────────────────────────────────
    def get_team_defensive_stats(self, team_id: int) -> dict:
        """
        Zwraca: def_rating, pace, pts_allowed_PG/SG/SF/PF/C.

        Źródła:
        - DEF_RATING, PACE: LeagueDashTeamStats (Advanced)
        - pts_allowed_to_position: LeagueDashPtDefend per kategoria obrony
          (Guards / Forwards / Center), zagregowane po drużynie
        """
        cache_key = f"team_def_{team_id}_{CURRENT_SEASON}"
        cached = self.cache.get(cache_key, ttl=TTL_TEAM_DEF)
        if cached is not None:
            return cached.iloc[0].to_dict()

        # ── DEF_RTG + PACE ────────────────────────────────
        _rand_sleep()
        adv_teams = leaguedashteamstats.LeagueDashTeamStats(
            season=CURRENT_SEASON,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star=SEASON_TYPE,
        )
        adv_df = adv_teams.get_data_frames()[0]
        team_row = adv_df[adv_df["TEAM_ID"] == team_id]

        if team_row.empty:
            logger.warning("Brak advanced stats dla team_id=%d", team_id)
            return {}

        def_rtg = float(team_row["DEF_RATING"].iloc[0])
        pace    = float(team_row["PACE"].iloc[0])

        # ── pts_allowed_to_position ───────────────────────
        # LeagueDashPtDefend zwraca stats per obrońca;
        # filtrujemy po TEAM_ID i bierzemy średnią pts per pozycja
        pos_map = {
            "PG": "Guards",
            "SG": "Guards",
            "SF": "Forwards",
            "PF": "Forwards",
            "C":  "Center",
        }
        pos_pts: dict[str, float] = {}

        # Pobieramy każdą kategorię raz (Guards / Forwards / Center)
        fetched_cats: dict[str, pd.DataFrame] = {}
        for pos, cat in pos_map.items():
            if cat not in fetched_cats:
                _rand_sleep()
                try:
                    defend_ep = leaguedashptdefend.LeagueDashPtDefend(
                        season=CURRENT_SEASON,
                        defense_category=cat,
                        per_mode_simple="PerGame",
                        season_type_all_star=SEASON_TYPE,
                    )
                    fetched_cats[cat] = defend_ep.get_data_frames()[0]
                except Exception as exc:
                    logger.warning("LeagueDashPtDefend [%s] error: %s", cat, exc)
                    fetched_cats[cat] = pd.DataFrame()

            cat_df = fetched_cats[cat]
            if not cat_df.empty and "TEAM_ID" in cat_df.columns:
                team_def = cat_df[cat_df["TEAM_ID"] == team_id]
                pts_col = "PLAYER_PTS" if "PLAYER_PTS" in cat_df.columns else "PTS"
                if not team_def.empty and pts_col in team_def.columns:
                    pos_pts[f"pts_allowed_{pos}"] = float(team_def[pts_col].mean())
                else:
                    # fallback: śr. ligowa
                    pos_pts[f"pts_allowed_{pos}"] = (
                        float(cat_df[pts_col].mean()) if pts_col in cat_df.columns
                        else 0.0
                    )
            else:
                pos_pts[f"pts_allowed_{pos}"] = 0.0

        result = {
            "team_id":   team_id,
            "def_rating": def_rtg,
            "pace":       pace,
            **pos_pts,
        }
        result_df = pd.DataFrame([result])
        self.cache.set(cache_key, result_df)
        logger.info(
            "Team def stats [id=%d]: DEF_RTG=%.1f PACE=%.1f",
            team_id, def_rtg, pace,
        )
        return result
