# nba_betting/history.py
"""
Historyczny zbiór mecz-zawodnik z cechami liczonymi WYŁĄCZNIE z danych sprzed meczu.

Źródło: nba_api LeagueGameLog — jeden request zwraca cały sezon (zawodnicy albo
drużyny), więc 5 sezonów to kilkanaście zapytań zamiast tysięcy.

Ten sam kod liczy cechy do treningu i do predykcji: nadchodzące mecze dokleja się
jako wiersze bez statystyk (PTS = NaN) i przechodzą one przez te same funkcje.

Użycie:
  python history.py                       # pobierz brakujące sezony + zbuduj zbiór
  python history.py --refresh-current     # odśwież bieżący sezon (nowe mecze)
  python history.py --offline             # tylko pliki z data/raw (GitHub Actions — bez stats.nba.com)
"""
import argparse
import logging
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import CURRENT_SEASON

logger = logging.getLogger(__name__)

try:
    from nba_api.stats.static import teams as _nba_teams
    TEAM_ABBR_TO_ID = {t["abbreviation"]: t["id"] for t in _nba_teams.get_teams()}
except ImportError:  # pragma: no cover
    TEAM_ABBR_TO_ID = {}

DATA_DIR      = Path("data")
RAW_DIR       = DATA_DIR / "raw"
DATASET_PATH  = DATA_DIR / "player_games.csv.gz"
SEASON_TYPES  = ("Regular Season", "Playoffs")
FIRST_SEASON  = 2021          # 2021-22: pierwszy pełny sezon po skróconych 2019-20/2020-21

# Okna rolling
TEAM_WINDOW   = 15            # mecze drużyny do DEF_RTG / pace
REST_CAP      = 10            # dni odpoczynku obcinamy (przerwa między sezonami ≠ informacja)
ABSENCE_DAYS  = 14            # nieobecny = grał dla drużyny w ostatnich 14 dniach, a dziś go nie ma
ROTATION_MIN  = 15.0          # … i średnio ≥ 15 min w ostatnich 10 meczach (zawodnik rotacji)


def norm_name(name: str) -> str:
    """Nazwisko do dopasowań między źródłami: bez akcentów, kropek, sufiksów Jr/III."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return " ".join(s.split())


def season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def all_seasons() -> list[str]:
    last = int(CURRENT_SEASON[:4])
    return [season_str(y) for y in range(FIRST_SEASON, last + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Pobieranie surowych logów (cache w data/raw/)
# ─────────────────────────────────────────────────────────────────────────────
def _raw_path(kind: str, season: str, season_type: str) -> Path:
    tag = "rs" if season_type == "Regular Season" else "po"
    return RAW_DIR / f"{kind}_{season}_{tag}.csv.gz"


def fetch_league_log(kind: str, season: str, season_type: str, refresh: bool = False,
                     offline: bool = False) -> pd.DataFrame:
    """kind: 'P' (zawodnicy) albo 'T' (drużyny). Zakończone sezony pobieramy raz.
    offline=True: tylko plik z data/raw, brak pliku → pusty DataFrame."""
    path = _raw_path(kind, season, season_type)
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"GAME_ID": str})
    if offline:
        logger.warning("Brak %s (tryb offline) — pomijam", path)
        return pd.DataFrame()

    from nba_api.stats.endpoints import leaguegamelog
    logger.info("Pobieram LeagueGameLog %s %s %s", kind, season, season_type)
    time.sleep(1.0)
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation=kind,
        timeout=60,
    ).get_data_frames()[0]
    df["SEASON"] = season
    df["IS_PLAYOFFS"] = int(season_type == "Playoffs")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return df


def load_logs(refresh_current: bool = False, offline: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    players, teams = [], []
    for season in all_seasons():
        refresh = refresh_current and season == CURRENT_SEASON and not offline
        for st in SEASON_TYPES:
            players.append(fetch_league_log("P", season, st, refresh, offline))
            teams.append(fetch_league_log("T", season, st, refresh, offline))
    p = pd.concat([d for d in players if not d.empty], ignore_index=True)
    t = pd.concat([d for d in teams if not d.empty], ignore_index=True)
    for d in (p, t):
        d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
        d["GAME_ID"] = d["GAME_ID"].astype(str).str.zfill(10)
    return p, t


# ─────────────────────────────────────────────────────────────────────────────
# Cechy drużynowe (rywal): DEF_RTG i pace z ostatnich TEAM_WINDOW meczów
# ─────────────────────────────────────────────────────────────────────────────
def team_game_table(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Jeden wiersz na drużynę-mecz z posiadaniami i punktami straconymi."""
    t = team_logs.copy()
    t["POSS"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    opp = t[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(
        columns={"TEAM_ID": "OPP_ID", "PTS": "PTS_ALLOWED", "POSS": "OPP_POSS"}
    )
    t = t.merge(opp, on="GAME_ID")
    t = t[t["TEAM_ID"] != t["OPP_ID"]]
    t["GAME_POSS"] = (t["POSS"] + t["OPP_POSS"]) / 2
    return t[[
        "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "OPP_ID",
        "MIN", "FGA", "FTA", "TOV", "PTS", "PTS_ALLOWED", "GAME_POSS",
    ]].rename(columns={"MIN": "TEAM_MIN", "FGA": "TEAM_FGA", "FTA": "TEAM_FTA", "TOV": "TEAM_TOV"})


def team_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Stan drużyny PO każdym meczu (rolling włącznie z tym meczem).
    Łączony z meczami zawodników przez merge_asof z allow_exact_matches=False,
    więc dla meczu w dniu D używa tylko meczów sprzed D.
    """
    g = team_games.sort_values("GAME_DATE").groupby("TEAM_ID", group_keys=False)
    out = team_games[["TEAM_ID", "GAME_DATE"]].copy()
    roll = lambda col: g[col].transform(lambda s: s.rolling(TEAM_WINDOW, min_periods=3).sum())
    out["def_rtg"] = 100 * roll("PTS_ALLOWED") / roll("GAME_POSS")
    out["pace"] = g["GAME_POSS"].transform(lambda s: s.rolling(TEAM_WINDOW, min_periods=3).mean())
    return out.sort_values("GAME_DATE")


# ─────────────────────────────────────────────────────────────────────────────
# Tabela mecz-zawodnik
# ─────────────────────────────────────────────────────────────────────────────
# Statystyki meczu (dla przyszłych meczów = NaN)
STAT_COLS = [
    "MIN", "PTS", "FGM", "FGA", "FTM", "FTA", "FG3A", "OREB", "DREB",
    "AST", "STL", "BLK", "TOV", "PF", "USG", "GMSC",
]


def player_game_table(player_logs: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    p = player_logs.copy()
    p["IS_HOME"] = p["MATCHUP"].str.contains(" vs. ").astype(int)
    # Rywal z MATCHUP ("LAL vs. GSW" / "LAL @ GSW") — działa też dla przyszłych meczów,
    # które nie mają jeszcze wiersza w logu drużyn.
    opp_abbr = p["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[-1]
    p["OPP_ID"] = opp_abbr.map(TEAM_ABBR_TO_ID).astype("int64")
    p = p.merge(
        team_games[["GAME_ID", "TEAM_ID", "TEAM_MIN", "TEAM_FGA", "TEAM_FTA", "TEAM_TOV"]],
        on=["GAME_ID", "TEAM_ID"], how="left",
    )
    # Usage% per mecz (wzór Basketball-Reference na danych z box score)
    plays = p["FGA"] + 0.44 * p["FTA"] + p["TOV"]
    team_plays = p["TEAM_FGA"] + 0.44 * p["TEAM_FTA"] + p["TEAM_TOV"]
    p["USG"] = np.where(
        p["MIN"] > 0,
        plays * (p["TEAM_MIN"] / 5) / (p["MIN"] * team_plays),
        np.nan,
    )
    # Game Score (Hollinger) per mecz — jedna liczba „jak dobry był mecz”
    p["GMSC"] = (
        p["PTS"] + 0.4 * p["FGM"] - 0.7 * p["FGA"] - 0.4 * (p["FTA"] - p["FTM"])
        + 0.7 * p["OREB"] + 0.3 * p["DREB"] + p["STL"] + 0.7 * p["AST"]
        + 0.7 * p["BLK"] - 0.4 * p["PF"] - p["TOV"]
    )
    keep = [
        "SEASON", "IS_PLAYOFFS", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
        "TEAM_ID", "TEAM_ABBREVIATION", "OPP_ID", "IS_HOME", *STAT_COLS,
    ]
    return p[keep]


def _roll_mean(g, col, n, min_periods=1):
    # shift(1): tylko mecze PRZED bieżącym
    return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=min_periods).mean())


def make_features(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """
    games: tabela z player_game_table (+ ewentualnie przyszłe mecze z NaN w statystykach).
    Zwraca games + kolumny cech. Każda cecha w wierszu meczu korzysta tylko
    z wcześniejszych meczów (shift / merge_asof bez dokładnych dopasowań).
    """
    df = games.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID", group_keys=False)

    # ── Forma (przez granice sezonów — na początku sezonu to końcówka poprzedniego)
    for n in (3, 5, 10, 20):
        df[f"pts_avg_{n}"] = _roll_mean(g, "PTS", n)
    df["pts_ewm"]    = g["PTS"].transform(lambda s: s.shift(1).ewm(halflife=5, min_periods=1).mean())
    df["pts_std_10"] = g["PTS"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    df["pts_trend"]  = df["pts_avg_3"] - df["pts_avg_10"]

    # ── Minuty i rola w ataku
    df["min_avg_5"]  = _roll_mean(g, "MIN", 5)
    df["min_avg_10"] = _roll_mean(g, "MIN", 10)
    df["min_last"]   = g["MIN"].shift(1)
    pts10 = g["PTS"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    min10 = g["MIN"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    df["pts_per36_10"] = np.where(min10 > 0, 36 * pts10 / min10, np.nan)
    for col in ("FGA", "FTA", "FG3A"):
        df[f"{col.lower()}_avg_10"] = _roll_mean(g, col, 10)
    df["usg_avg_10"] = _roll_mean(g, "USG", 10)

    # ── Skuteczność (krok 4b): TS% liczone z sum (stabilniejsze niż średnia ilorazów)
    df["_TSA"] = df["FGA"] + 0.44 * df["FTA"]
    g = df.groupby("PLAYER_ID", group_keys=False)
    pts_s = lambda n: g["PTS"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    tsa_s = lambda n: g["_TSA"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    p10, t10 = pts_s(10), tsa_s(10)
    df["ts_10"] = np.where(t10 > 0, p10 / (2 * t10), np.nan)
    gs_ = df.groupby(["PLAYER_ID", "SEASON"], group_keys=False)
    p_season = gs_["PTS"].transform(lambda s: s.shift(1).expanding().sum())
    t_season = gs_["_TSA"].transform(lambda s: s.shift(1).expanding().sum())
    df["ts_season"] = np.where(t_season > 0, p_season / (2 * t_season), np.nan)
    df["gmsc_avg_10"] = _roll_mean(g, "GMSC", 10)
    df = df.drop(columns="_TSA")

    # ── Sezon do tej pory
    gs = df.groupby(["PLAYER_ID", "SEASON"], group_keys=False)
    df["season_pts_avg"]   = gs["PTS"].transform(lambda s: s.shift(1).expanding().mean())
    df["games_in_season"]  = gs.cumcount()

    # ── Kontekst: odpoczynek, back-to-back
    prev_date = g["GAME_DATE"].shift(1)
    df["days_rest"] = (df["GAME_DATE"] - prev_date).dt.days.clip(upper=REST_CAP).fillna(REST_CAP)
    df["is_b2b"]    = (df["days_rest"] == 1).astype(int)

    # ── H2H: ostatnie 5 meczów przeciwko temu rywalowi
    gh = df.groupby(["PLAYER_ID", "OPP_ID"], group_keys=False)
    df["pts_avg_vs_opp"] = gh["PTS"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["h2h_games"]      = gh.cumcount().clip(upper=5)

    # ── Absencje kolegów z rotacji (bff / no_bff — wersja drużynowa)
    for col in ("MIN", "PTS", "FGA"):
        df[f"_{col}_post"] = g[col].transform(lambda s: s.rolling(10, min_periods=3).mean())
    df = df.merge(team_absences(df), on=["GAME_ID", "TEAM_ID"], how="left")
    for col in ABSENCE_COLS:
        df[col] = df[col].fillna(0.0)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID", group_keys=False)
    # różnica względem ostatnich 5 meczów zawodnika: >0 = dziś brakuje więcej niż ostatnio,
    # <0 = ktoś wraca, więc średnie zawodnika z ostatnich meczów mogą być zawyżone
    df["vac_pts_delta"] = df["vac_pts"] - _roll_mean(g, "vac_pts", 5)
    df["vac_min_delta"] = df["vac_min"] - _roll_mean(g, "vac_min", 5)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_post")])

    # ── Rywal i własna drużyna: stan sprzed dnia meczu
    form = team_form(team_games)
    df = df.sort_values("GAME_DATE")
    opp = form.rename(columns={"TEAM_ID": "OPP_ID", "def_rtg": "opp_def_rtg", "pace": "opp_pace"})
    df = pd.merge_asof(df, opp, on="GAME_DATE", by="OPP_ID", allow_exact_matches=False)
    own = form.rename(columns={"def_rtg": "_own_def", "pace": "team_pace"}).drop(columns="_own_def")
    df = pd.merge_asof(df, own, on="GAME_DATE", by="TEAM_ID", allow_exact_matches=False)

    return df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


ABSENCE_COLS = ["n_out", "vac_min", "vac_pts", "vac_fga"]


def team_absences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dla każdego meczu drużyny: kto z rotacji nie zagrał i ile minut/punktów/rzutów po nim zostaje.

    Kandydaci = zawodnicy, którzy grali dla tej drużyny w ostatnich ABSENCE_DAYS dniach
    (wartości _*_post z ich ostatniego meczu przed dniem meczu — tylko przeszłość).
    Nieobecny = kandydat z _MIN_post ≥ ROTATION_MIN, którego nie ma w tym meczu.
    Zawodnik wymieniony do innej drużyny przez 14 dni liczy się jako brak w starej —
    to zgodne z rzeczywistością (jego minuty trzeba rozdzielić).
    """
    apps = df[["TEAM_ID", "SEASON", "PLAYER_ID", "GAME_DATE", "_MIN_post", "_PTS_post", "_FGA_post"]] \
        .dropna(subset=["_MIN_post"])
    games = df[["TEAM_ID", "SEASON", "GAME_ID", "GAME_DATE"]].drop_duplicates()
    present = set(zip(df["GAME_ID"], df["PLAYER_ID"]))
    window = pd.Timedelta(days=ABSENCE_DAYS)

    out = []
    for (team, season), tg in games.groupby(["TEAM_ID", "SEASON"]):
        ta = apps[(apps["TEAM_ID"] == team) & (apps["SEASON"] == season)]
        m = tg.merge(ta.drop(columns=["TEAM_ID", "SEASON"]), how="cross", suffixes=("", "_app"))
        m = m[(m["GAME_DATE_app"] < m["GAME_DATE"]) & (m["GAME_DATE_app"] >= m["GAME_DATE"] - window)]
        if m.empty:
            continue
        m = m.sort_values("GAME_DATE_app").drop_duplicates(["GAME_ID", "PLAYER_ID"], keep="last")
        m = m[m["_MIN_post"] >= ROTATION_MIN]
        m = m[[(gid, pid) not in present for gid, pid in zip(m["GAME_ID"], m["PLAYER_ID"])]]
        if m.empty:
            continue
        out.append(m.groupby(["GAME_ID", "TEAM_ID"]).agg(
            n_out=("PLAYER_ID", "size"), vac_min=("_MIN_post", "sum"),
            vac_pts=("_PTS_post", "sum"), vac_fga=("_FGA_post", "sum"),
        ).reset_index())
    if not out:
        return pd.DataFrame(columns=["GAME_ID", "TEAM_ID", *ABSENCE_COLS])
    return pd.concat(out, ignore_index=True)


BASE_FEATURE_COLS = [
    "pts_avg_3", "pts_avg_5", "pts_avg_10", "pts_avg_20", "pts_ewm", "pts_std_10", "pts_trend",
    "min_avg_5", "min_avg_10", "min_last", "pts_per36_10",
    "fga_avg_10", "fta_avg_10", "fg3a_avg_10", "usg_avg_10",
    "season_pts_avg", "games_in_season",
    "IS_HOME", "IS_PLAYOFFS", "days_rest", "is_b2b",
    "pts_avg_vs_opp", "h2h_games",
    "opp_def_rtg", "opp_pace", "team_pace",
]
ABSENCE_FEATURE_COLS = [*ABSENCE_COLS, "vac_pts_delta", "vac_min_delta"]
EFFICIENCY_FEATURE_COLS = ["ts_10", "ts_season", "gmsc_avg_10"]
FEATURE_COLS = BASE_FEATURE_COLS + ABSENCE_FEATURE_COLS + EFFICIENCY_FEATURE_COLS


def build_dataset(refresh_current: bool = False, offline: bool = False) -> pd.DataFrame:
    p, t = load_logs(refresh_current=refresh_current, offline=offline)
    team_games = team_game_table(t)
    games = player_game_table(p, team_games)
    df = make_features(games, team_games)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False, compression="gzip")
    logger.info(
        "Zbiór: %d wierszy, %d zawodników, %s → %s",
        len(df), df["PLAYER_ID"].nunique(),
        df["GAME_DATE"].min().date(), df["GAME_DATE"].max().date(),
    )
    return df


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-current", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    build_dataset(refresh_current=args.refresh_current, offline=args.offline)
