# nba_betting/config.py
import asyncio
import sys
import os
from datetime import date

# Windows fix dla asyncio + Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ── API Keys ──────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # ustaw jako GitHub Secret: ODDS_API_KEY

# ── Betting Parameters ────────────────────────────────────
ODDS_MIN            = 1.40
ODDS_MAX            = 1.90
STAKE_PER_PICK      = 10.0
CONFIDENCE_THRESHOLD = 60

# ── Model ─────────────────────────────────────────────────
MODEL_DIR    = "models"
DB_PATH      = "nba_betting.db"

# ── Cache TTL (sekundy) ───────────────────────────────────
TTL_GAME_LOG   = 8  * 3600   # 8h
TTL_SEASON_AVG = 8  * 3600
TTL_TEAM_DEF   = 24 * 3600
TTL_BOX_SCORE  = 24 * 3600   # permanentne po pobraniu

# ── nba_api ───────────────────────────────────────────────
def season_for_date(d) -> str:
    """Sezon NBA dla daty, np. 2026-10-27 → "2026-27", 2026-03-13 → "2025-26".
    Nowy sezon liczymy od 1 października (preseason/start regularnego ~ koniec paź.)."""
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


# Sezon wyliczany z daty — wcześniej był wpisany na sztywno i się zestarzał.
# Można nadpisać zmienną środowiskową NBA_SEASON (np. do testów historycznych).
CURRENT_SEASON    = os.getenv("NBA_SEASON") or season_for_date(date.today())
SEASON_TYPE       = "Regular Season"
REQUEST_DELAY_MIN = 0.6
REQUEST_DELAY_MAX = 1.0
