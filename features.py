# nba_betting/features.py
import logging
from typing import Optional

import numpy as np
import pandas as pd

from data_fetcher import NBADataFetcher
from config import DB_PATH

logger = logging.getLogger(__name__)

# Mapowanie pozycji gracza → klucz w team_def_stats
POSITION_TO_DEF_KEY = {
    "G":  "pts_allowed_PG",
    "PG": "pts_allowed_PG",
    "SG": "pts_allowed_SG",
    "SF": "pts_allowed_SF",
    "PF": "pts_allowed_PF",
    "F":  "pts_allowed_SF",
    "C":  "pts_allowed_C",
    "FC": "pts_allowed_PF",
    "GF": "pts_allowed_SG",
}


# ─────────────────────────────────────────────────────────────────────────────
# Implied probability — usuwanie overround (margin)
# ─────────────────────────────────────────────────────────────────────────────
def remove_overround(over_odds: float, under_odds: float) -> tuple[float, float]:
    """
    Usuwa margin bukmachera metodą multiplikatywną.
    Surowa implied prob: p_over_raw = 1/over_odds, p_under_raw = 1/under_odds
    Suma > 1.0 to właśnie overround (vig bukmachera).

    Normalizacja: p_over_fair = p_over_raw / (p_over_raw + p_under_raw)

    Przykład:
        over=1.83, under=1.87
        p_raw = 0.5464 + 0.5348 = 1.0812  → margin ≈ 7.5%
        p_fair_over = 0.5464 / 1.0812 = 0.505
    """
    p_over_raw  = 1.0 / over_odds
    p_under_raw = 1.0 / under_odds
    total       = p_over_raw + p_under_raw

    p_over_fair  = p_over_raw  / total
    p_under_fair = p_under_raw / total

    return round(p_over_fair, 4), round(p_under_fair, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling features z game logu
# ─────────────────────────────────────────────────────────────────────────────
def compute_rolling_features(
    recent_df: pd.DataFrame,
    line: float,
) -> dict:
    """
    Wylicza wszystkie rolling features z DataFrame ostatnich meczów.

    Wymaga kolumn: PTS, MIN, USG_PCT (posortowane od najnowszego).
    Zwraca dict gotowy do wstawienia do feature vector.
    """
    if recent_df.empty:
        logger.warning("compute_rolling_features: pusty DataFrame")
        return _empty_rolling_features()

    # Sortuj chronologicznie (najstarsze pierwsze) dla rolling
    df = recent_df.sort_values("GAME_DATE").reset_index(drop=True)
    pts = df["PTS"].astype(float)

    def safe_mean(series: pd.Series, n: int) -> float:
        """Średnia z ostatnich n obserwacji (tail)."""
        tail = series.tail(n)
        return float(tail.mean()) if len(tail) > 0 else float(pts.mean())

    def safe_std(series: pd.Series, n: int) -> float:
        tail = series.tail(n)
        return float(tail.std(ddof=1)) if len(tail) > 1 else 0.0

    # ── Forma zawodnika ───────────────────────────────────────────────────────
    pts_avg_3  = safe_mean(pts, 3)
    pts_avg_5  = safe_mean(pts, 5)
    pts_avg_10 = safe_mean(pts, 10)
    pts_avg_20 = safe_mean(pts, 20)
    pts_std_10 = safe_std(pts, 10)

    # Kierunek formy: pozytywny = gracz w dobrej formie
    pts_trend = pts_avg_3 - pts_avg_10

    # % meczów historycznie powyżej tej linii (ze wszystkich dostępnych)
    pct_over_line_historical = float((pts > line).mean()) if len(pts) > 0 else 0.5

    # Minuty i usage (ostatnie 5)
    min_avg_5 = (
        df["MIN"].tail(5).astype(str)
        .apply(_parse_minutes)
        .mean()
        if "MIN" in df.columns else 0.0
    )
    usg_pct_avg_5 = (
        float(df["USG_PCT"].tail(5).mean())
        if "USG_PCT" in df.columns else 0.0
    )

    return {
        "pts_avg_3":               round(pts_avg_3,  2),
        "pts_avg_5":               round(pts_avg_5,  2),
        "pts_avg_10":              round(pts_avg_10, 2),
        "pts_avg_20":              round(pts_avg_20, 2),
        "pts_std_10":              round(pts_std_10, 2),
        "pts_trend":               round(pts_trend,  2),
        "pct_over_line_historical": round(pct_over_line_historical, 4),
        "min_avg_5":               round(min_avg_5,    2),
        "usg_pct_avg_5":           round(usg_pct_avg_5, 4),
    }


def _parse_minutes(min_str: str) -> float:
    """
    Parsuje minuty w formatach: '33:24', '33.4', '33'.
    Zwraca float minut.
    """
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(min_str)
    except (ValueError, IndexError):
        return 0.0


def _empty_rolling_features() -> dict:
    return {
        "pts_avg_3": 0.0, "pts_avg_5": 0.0, "pts_avg_10": 0.0,
        "pts_avg_20": 0.0, "pts_std_10": 0.0, "pts_trend": 0.0,
        "pct_over_line_historical": 0.5, "min_avg_5": 0.0, "usg_pct_avg_5": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H2H features vs konkretny rywal
# ─────────────────────────────────────────────────────────────────────────────
def compute_h2h_features(
    vs_df: pd.DataFrame,
    line: float,
    last_n: int = 5,
) -> dict:
    """
    Wylicza historyczne features vs konkretny rywal.
    """
    if vs_df.empty:
        return {
            "pts_avg_vs_opponent":  0.0,
            "pct_over_vs_opponent": 0.5,
            "h2h_games_available":  0,
        }

    df = vs_df.head(last_n)
    pts = df["PTS"].astype(float)

    return {
        "pts_avg_vs_opponent":  round(float(pts.mean()), 2),
        "pct_over_vs_opponent": round(float((pts > line).mean()), 4),
        "h2h_games_available":  len(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kontekst meczu — rest days, is_home
# ─────────────────────────────────────────────────────────────────────────────
def compute_game_context(
    recent_df: pd.DataFrame,
    is_home: int,
) -> dict:
    """
    Wyciąga is_home, days_rest, is_back_to_back z ostatniego meczu
    (najnowszy wpis w recent_df = nadchodzący kontekst).
    """
    if recent_df.empty:
        return {
            "is_home":         is_home,
            "days_rest":       2,
            "is_back_to_back": 0,
        }

    # Najbliższy mecz = pierwszy wiersz (df posortowany desc)
    last = recent_df.sort_values("GAME_DATE", ascending=False).iloc[0]
    return {
        "is_home":         int(is_home),
        "days_rest":       int(last.get("days_rest", 2)),
        "is_back_to_back": int(last.get("is_back_to_back", 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Defensywne features rywala
# ─────────────────────────────────────────────────────────────────────────────
def compute_defensive_features(
    team_def: dict,
    player_position: str = "PG",
) -> dict:
    """
    Wyciąga DEF_RTG, PACE i pts_allowed_to_position ze słownika
    zwróconego przez NBADataFetcher.get_team_defensive_stats().

    player_position: pozycja gracza (PG/SG/SF/PF/C/G/F)
    """
    if not team_def:
        return {
            "opp_def_rating":            110.0,  # średnia ligowa
            "opp_pace":                   99.0,
            "opp_pts_allowed_to_position": 18.0,
        }

    pos_key = POSITION_TO_DEF_KEY.get(player_position.upper(), "pts_allowed_PG")
    pts_allowed = team_def.get(pos_key, team_def.get("pts_allowed_PG", 18.0))

    return {
        "opp_def_rating":             round(float(team_def.get("def_rating", 110.0)), 2),
        "opp_pace":                   round(float(team_def.get("pace", 99.0)),        2),
        "opp_pts_allowed_to_position": round(float(pts_allowed),                       2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sygnał rynkowy
# ─────────────────────────────────────────────────────────────────────────────
def compute_market_features(
    line: float,
    over_odds: float,
    under_odds: float,
) -> dict:
    """
    Wylicza implied probabilities z usunięciem overround (vig).

    over_implied_prob  > 0.5 → rynek faworyzuje over
    under_implied_prob > 0.5 → rynek faworyzuje under
    market_margin = overround bukmachera (im wyższy tym mniej zaufania)
    """
    p_over, p_under = remove_overround(over_odds, under_odds)
    market_margin   = round((1/over_odds + 1/under_odds) - 1.0, 4)

    return {
        "line":               float(line),
        "over_implied_prob":  p_over,
        "under_implied_prob": p_under,
        "market_margin":      market_margin,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Główna funkcja — build_feature_vector
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(
    player_id:        int,
    opponent_team_id: int,
    line:             float,
    over_odds:        float,
    under_odds:       float,
    is_home:          int,
    player_position:  str = "PG",
    fetcher:          Optional[NBADataFetcher] = None,
    db_path:          str = DB_PATH,
) -> pd.DataFrame:
    """
    Buduje kompletny wektor cech dla jednego prop betu.

    Argumenty:
        player_id        : ID zawodnika (nba_api)
        opponent_team_id : ID drużyny rywala (nba_api)
        line             : linia bukmachera (np. 25.5)
        over_odds        : kurs na over (decimal, np. 1.83)
        under_odds       : kurs na under (decimal, np. 1.87)
        is_home          : 1 jeśli zawodnik gra u siebie
        player_position  : pozycja gracza (PG/SG/SF/PF/C)
        fetcher          : opcjonalnie wstrzyknięty NBADataFetcher (DI dla testów)

    Zwraca:
        pd.DataFrame z 1 wierszem i 24 kolumnami (features)

    Wszystkie sub-featury są pogrupowane w 4 kategorie:
        1. FORMA        — rolling pts, trend, consistency
        2. KONTEKST     — home/away, rest, b2b
        3. H2H          — historia vs ten rywal
        4. RYNEK        — linia, implied prob, margin
    """
    if fetcher is None:
        fetcher = NBADataFetcher(db_path=db_path)

    logger.info(
        "build_feature_vector: player=%d vs team=%d line=%.1f",
        player_id, opponent_team_id, line,
    )

    # ── Pobierz dane ──────────────────────────────────────────────────────────
    recent_df  = fetcher.get_player_recent_stats(player_id, last_n=20)
    vs_df      = fetcher.get_player_vs_opponent(player_id, opponent_team_id, last_n=10)
    team_def   = fetcher.get_team_defensive_stats(opponent_team_id)

    # ── Wylicz grupy features ─────────────────────────────────────────────────
    rolling   = compute_rolling_features(recent_df, line)
    context   = compute_game_context(recent_df, is_home)
    h2h       = compute_h2h_features(vs_df, line)
    defensive = compute_defensive_features(team_def, player_position)
    market    = compute_market_features(line, over_odds, under_odds)

    # ── Złóż w jeden słownik ──────────────────────────────────────────────────
    features = {
        # Meta (nie wchodzi do modelu — do debugowania)
        "_player_id":        player_id,
        "_opponent_team_id": opponent_team_id,

        # 1. FORMA
        **rolling,

        # 2. KONTEKST
        **context,

        # 3. DEFENSYWA RYWALA
        **defensive,

        # 4. H2H
        **h2h,

        # 5. RYNEK
        **market,
    }

    df = pd.DataFrame([features])

    logger.info(
        "Feature vector gotowy: %d features | "
        "pts_avg_10=%.1f trend=%.1f p_over=%.3f",
        len(df.columns),
        features["pts_avg_10"],
        features["pts_trend"],
        features["over_implied_prob"],
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Kolumny wchodzące DO modelu (bez meta _*)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURE_COLS = [
    # Forma
    "pts_avg_3", "pts_avg_5", "pts_avg_10", "pts_avg_20",
    "pts_std_10", "pts_trend", "pct_over_line_historical",
    "min_avg_5", "usg_pct_avg_5",
    # Kontekst
    "is_home", "days_rest", "is_back_to_back",
    # Defensywa rywala
    "opp_def_rating", "opp_pace", "opp_pts_allowed_to_position",
    # H2H
    "pts_avg_vs_opponent", "pct_over_vs_opponent", "h2h_games_available",
    # Rynek
    "line", "over_implied_prob", "under_implied_prob", "market_margin",
]
