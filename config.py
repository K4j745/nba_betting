# nba_betting/config.py
import asyncio
import sys
import os

# Windows fix dla asyncio + Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ── API Keys ──────────────────────────────────────────────
ODDS_API_KEY = os.getenv("932f5bb78fb5c5f5f2a410e8c77a898c", "6bd33386a07cadc44863240776b4ec1f") # klucze api do oodds api

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
CURRENT_SEASON    = "2025-26"
SEASON_TYPE       = "Regular Season"
REQUEST_DELAY_MIN = 0.6
REQUEST_DELAY_MAX = 1.0
