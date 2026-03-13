# nba_betting/odds_fetcher.py
import asyncio
import json
import logging
import random
import re
import sqlite3
import time
from datetime import datetime
from typing import Optional

import requests
import pandas as pd
from rapidfuzz import process, fuzz   # pip install rapidfuzz

from config import (
    DB_PATH, ODDS_API_KEY, ODDS_MIN, ODDS_MAX,
)

logger = logging.getLogger(__name__)

# ── Stałe ─────────────────────────────────────────────────────────────────────
ODDS_API_BASE   = "https://api.the-odds-api.com/v4"
SPORT           = "basketball_nba"
REGIONS         = "eu"          # kursy w formacie dziesiętnym (1.83 etc.)
MARKETS         = "player_points"
BOOKMAKERS      = "bet365,unibet,pinnacle"  # kolejność priorytetu
FUZZY_THRESHOLD = 80            # min. score dla dopasowania nazwy gracza

# Znane strony z propsami — scraping jako opcja pierwsza
SCRAPE_TARGETS = [
    "https://www.pinnacle.com/en/basketball/nba/matchups",
    "https://www.bet365.com/#/AC/B18/C20604387/",
]

# DEFINICJA _over_under
def _pair_over_under(raw_props: list[dict]) -> list[dict]:
    """
    Odds API zwraca over i under jako osobne rekordy.
    Łączymy je w jeden: {player, line, over_odds, under_odds}.
    """
    paired: dict[tuple, dict] = {}
    for prop in raw_props:
        key = (
            prop["player_name"],
            prop["line"],
            prop.get("home_team", ""),
            prop.get("bookmaker", ""),
        )
        if key not in paired:
            paired[key] = {
                "player_name": prop["player_name"],
                "stat_type":   prop["stat_type"],
                "line":        prop["line"],
                "over_odds":   None,
                "under_odds":  None,
                "home_team":   prop.get("home_team", ""),
                "away_team":   prop.get("away_team", ""),
                "game_time":   prop.get("game_time", ""),
                "bookmaker":   prop.get("bookmaker", ""),
                "source":      prop.get("source", ""),
            }
        side = prop.get("side", "")
        if side == "over":
            paired[key]["over_odds"] = prop["odds"]
        elif side == "under":
            paired[key]["under_odds"] = prop["odds"]

    result = [
        p for p in paired.values()
        if p["over_odds"] is not None and p["under_odds"] is not None
    ]
    logger.debug("_pair_over_under: %d raw → %d paired", len(raw_props), len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Konwersja kursów: American ↔ Decimal
# ─────────────────────────────────────────────────────────────────────────────
def american_to_decimal(american: int) -> float:
    """Konwertuje kurs amerykański na dziesiętny."""
    if american > 0:
        return round(american / 100 + 1, 3)
    return round(100 / abs(american) + 1, 3)


def decimal_in_range(odds: float) -> bool:
    return ODDS_MIN <= odds <= ODDS_MAX


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy matching nazw graczy
# ─────────────────────────────────────────────────────────────────────────────
class PlayerNameMatcher:
    """
    Dopasowuje nazwy graczy między różnymi źródłami danych.
    Przykład: "Shai Gilgeous Alexander" ↔ "Shai Gilgeous-Alexander"

    Użycie:
        matcher = PlayerNameMatcher(nba_api_names)
        matched = matcher.match("Gilgeous-Alexander S.")
    """

    def __init__(self, canonical_names: list[str]):
        """
        canonical_names: lista officjalnych nazw z nba_api
        (np. ['Victor Wembanyama', 'Jayson Tatum', ...])
        """
        self.canonical = canonical_names
        # Pre-normalizacja: lowercase, usuń myślniki i kropki
        self._normalized = [self._normalize(n) for n in canonical_names]

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[-.'']", " ", name).lower().strip()

    def match(self, raw_name: str, threshold: int = FUZZY_THRESHOLD) -> Optional[str]:
        """
        Zwraca canonical name lub None jeśli score < threshold.
        Używa token_sort_ratio — odporny na różną kolejność słów.
        """
        normalized_raw = self._normalize(raw_name)
        result = process.extractOne(
            normalized_raw,
            self._normalized,
            scorer=fuzz.token_sort_ratio,
        )
        if result is None or result[1] < threshold:
            logger.debug(
                "Fuzzy MISS: '%s' → best='%s' score=%s",
                raw_name, result[0] if result else "?", result[1] if result else 0,
            )
            return None
        idx = result[2]
        matched = self.canonical[idx]
        logger.debug(
            "Fuzzy MATCH: '%s' → '%s' (score=%d)", raw_name, matched, result[1]
        )
        return matched


# ─────────────────────────────────────────────────────────────────────────────
# Playwright scraper (opcja 1)
# ─────────────────────────────────────────────────────────────────────────────
async def _scrape_props_playwright(url: str) -> list[dict]:  # pragma: no cover
    """
    Próbuje pobrać propsy przez Playwright + stealth.
    Zwraca listę raw dict lub [] jeśli Cloudflare blokuje / timeout.

    Instalacja:
        pip install playwright playwright-stealth
        playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import stealth_async   # pip install playwright-stealth
    except ImportError:
        logger.warning("playwright lub playwright-stealth niezainstalowany — skip scraping")
        return []

    props = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = await context.new_page()
            await stealth_async(page)   # patch: webdriver, plugins, headless flags

            logger.info("Playwright: otwieranie %s", url)
            await page.goto(url, wait_until="networkidle", timeout=30_000)

            # Wykryj Cloudflare challenge
            title = await page.title()
            if "just a moment" in title.lower() or "cloudflare" in title.lower():
                logger.warning("Cloudflare challenge wykryty na %s — czekam 15s", url)
                await page.wait_for_function(
                    "() => !document.title.toLowerCase().includes('just a moment')",
                    timeout=15_000,
                )
                await asyncio.sleep(random.uniform(2.0, 4.0))

            # Ludzkie zachowanie: scroll + random wait
            await page.mouse.wheel(0, random.randint(300, 700))
            await asyncio.sleep(random.uniform(1.0, 2.5))

            # ── Parsowanie — dostosuj selektor do konkretnej strony ──
            # To jest szablon; każda strona ma inną strukturę HTML.
            # Pinnacle przykład (uproszczony):
            rows = await page.query_selector_all("[data-test-id='player-prop-row']")
            for row in rows:
                try:
                    player = await row.query_selector(".player-name")
                    line   = await row.query_selector(".line-value")
                    over_o = await row.query_selector(".over-odds")
                    under_o = await row.query_selector(".under-odds")

                    if not all([player, line, over_o, under_o]):
                        continue

                    props.append({
                        "player_name": (await player.inner_text()).strip(),
                        "stat_type":   "player_points",
                        "line":        float((await line.inner_text()).strip()),
                        "over_odds":   float((await over_o.inner_text()).strip()),
                        "under_odds":  float((await under_o.inner_text()).strip()),
                        "home_team":   "",
                        "away_team":   "",
                        "game_time":   "",
                        "source":      "playwright",
                    })
                except Exception as e:
                    logger.debug("Parse error na wierszu: %s", e)

            await browser.close()
            logger.info("Playwright: pobrano %d propsów z %s", len(props), url)

    except Exception as exc:
        logger.warning("Playwright scraping FAILED (%s): %s", url, exc)
        return []

    return props


# ─────────────────────────────────────────────────────────────────────────────
# The Odds API (fallback)
# ─────────────────────────────────────────────────────────────────────────────
class OddsAPIClient:
    """
    Klient The Odds API v4.
    Darmowy tier: 500 req/miesiąc — każde wywołanie loguje zużycie.
    """

    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._requests_remaining: Optional[int] = None

    def _get(self, path: str, params: dict) -> Optional[dict | list]:
        params["apiKey"] = self.api_key
        url = f"{ODDS_API_BASE}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            # Quota tracking z headerów
            remaining = resp.headers.get("x-requests-remaining")
            used      = resp.headers.get("x-requests-used")
            if remaining:
                self._requests_remaining = int(remaining)
                logger.info(
                    "Odds API quota: %s użytych, %s pozostałych",
                    used, remaining,
                )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error("Odds API HTTP error %s: %s", resp.status_code, e)
            if resp.status_code == 401:
                logger.error("Nieprawidłowy ODDS_API_KEY!")
            elif resp.status_code == 422:
                logger.error("Zły parametr zapytania do Odds API")
            return None
        except requests.RequestException as e:
            logger.error("Odds API request failed: %s", e)
            return None

    def get_player_props_raw(self, sport: str = SPORT) -> list[dict]:
        """
        Pobiera player_points propsy dla wszystkich dzisiejszych meczów NBA.
        Zwraca surowe dane z API (lista eventów z oddsami).
        """
        # Krok 1: lista eventów (game_id + drużyny)
        events = self._get(
            f"/sports/{sport}/events",
            {"regions": REGIONS},
        )
        if not events:
            logger.warning("Brak eventów z Odds API")
            return []

        all_props = []
        for event in events:
            event_id   = event["id"]
            home_team  = event["home_team"]
            away_team  = event["away_team"]
            game_time  = event["commence_time"]

            # Krok 2: propsy per event
            odds_data = self._get(
                f"/sports/{sport}/events/{event_id}/odds",
                {
                    "regions":  REGIONS,
                    "markets":  MARKETS,
                    "bookmakers": BOOKMAKERS,
                    "oddsFormat": "decimal",
                },
            )
            if not odds_data:
                continue

            for bookmaker in odds_data.get("bookmakers", []):
                bk_name = bookmaker["key"]
                for market in bookmaker.get("markets", []):
                    if market["key"] != "player_points":
                        continue
                    for outcome in market.get("outcomes", []):
                        all_props.append({
                            "player_name": outcome["description"],
                            "stat_type":   "player_points",
                            "line":        float(outcome.get("point", 0)),
                            "side":        outcome["name"].lower(),  # "over"/"under"
                            "odds":        float(outcome["price"]),
                            "home_team":   home_team,
                            "away_team":   away_team,
                            "game_time":   game_time,
                            "bookmaker":   bk_name,
                            "source":      "odds_api",
                        })
            time.sleep(0.3)  # grzeczny klient

        logger.info("Odds API: pobrano %d raw outcomes", len(all_props))
        return all_props


# ── Normalizacja: łączenie over/under w jeden rekord ──────────────────────
@staticmethod
def _pair_over_under(raw_props: list[dict]) -> list[dict]:
    """
    Odds API zwraca over i under jako osobne rekordy.
    Łączymy je w jeden: {player, line, over_odds, under_odds}.
    """
    # Klucz: (player_name, line, home_team, bookmaker)
    paired: dict[tuple, dict] = {}
    for prop in raw_props:
        key = (
            prop["player_name"],
            prop["line"],
            prop.get("home_team", ""),
            prop.get("bookmaker", ""),
        )
        if key not in paired:
            paired[key] = {
                "player_name": prop["player_name"],
                "stat_type":   prop["stat_type"],
                "line":        prop["line"],
                "over_odds":   None,
                "under_odds":  None,
                "home_team":   prop.get("home_team", ""),
                "away_team":   prop.get("away_team", ""),
                "game_time":   prop.get("game_time", ""),
                "bookmaker":   prop.get("bookmaker", ""),
                "source":      prop.get("source", ""),
            }
        side = prop.get("side", "")
        if side == "over":
            paired[key]["over_odds"] = prop["odds"]
        elif side == "under":
            paired[key]["under_odds"] = prop["odds"]

    # Odrzuć rekordy bez obu stron
    result = [
        p for p in paired.values()
        if p["over_odds"] is not None and p["under_odds"] is not None
    ]
    logger.debug("_pair_over_under: %d raw → %d paired", len(raw_props), len(result))
    return result
# ─────────────────────────────────────────────────────────────────────────────
# OddsFetcher — główna klasa Modułu 2
# ─────────────────────────────────────────────────────────────────────────────
class OddsFetcher:
    """
    Pobiera propsy NBA z dwóch źródeł:
      1. Playwright scraping (próba pierwsza)
      2. The Odds API (fallback)

    Wynik jest filtrowany, normalizowany i zapisywany do SQLite.
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        nba_player_names: Optional[list[str]] = None,
        odds_min: float = ODDS_MIN,
        odds_max: float = ODDS_MAX,
    ):
        self.db_path    = db_path
        self.odds_min   = odds_min
        self.odds_max   = odds_max
        self.api_client = OddsAPIClient()
        self.matcher    = PlayerNameMatcher(nba_player_names or [])
        self._init_db()
        logger.info(
            "OddsFetcher ready (odds_range=%.2f–%.2f)",
            odds_min, odds_max,
        )

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS odds_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT    NOT NULL,
                    stat_type   TEXT    NOT NULL,
                    line        REAL    NOT NULL,
                    over_odds   REAL,
                    under_odds  REAL,
                    home_team   TEXT,
                    away_team   TEXT,
                    game_time   TEXT,
                    bookmaker   TEXT,
                    source      TEXT,
                    fetched_at  TEXT DEFAULT (datetime('now'))
                )
            """)
             # ── Migracja: dodaj brakujące kolumny jeśli DB jest stara ──────────
            existing_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(odds_history)")
            }
            migrations = {
                "bookmaker": "TEXT",
                "source":    "TEXT",
            }
            for col, col_type in migrations.items():
                if col not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE odds_history ADD COLUMN {col} {col_type}"
                    )
                    logger.info("Migracja DB: dodano kolumnę odds_history.%s", col)
            conn.commit()

    

    # ── Filtrowanie kursów ────────────────────────────────────────────────────
    def _filter_odds(self, props: list[dict]) -> list[dict]:
        """
        Przepuszcza tylko rekordy gdzie over_odds LUB under_odds
        mieści się w konfigurowanym przedziale.
        """
        filtered = []
        for p in props:
            over_ok  = p["over_odds"]  and decimal_in_range(p["over_odds"])
            under_ok = p["under_odds"] and decimal_in_range(p["under_odds"])
            if over_ok or under_ok:
                filtered.append(p)
        logger.debug(
            "Filtr kursów [%.2f–%.2f]: %d → %d propsów",
            self.odds_min, self.odds_max, len(props), len(filtered),
        )
        return filtered

    # ── Fuzzy matching nazw ───────────────────────────────────────────────────
    def _resolve_player_names(self, props: list[dict]) -> list[dict]:
        """
        Dopasowuje player_name do canonical NBA names.
        Jeśli matcher jest pusty (brak listy), zwraca props bez zmian.
        """
        if not self.matcher.canonical:
            return props
        resolved = []
        for p in props:
            canonical = self.matcher.match(p["player_name"])
            if canonical is None:
                logger.debug("Odrzucono gracza (fuzzy miss): '%s'", p["player_name"])
                continue
            p = {**p, "player_name": canonical, "raw_name": p["player_name"]}
            resolved.append(p)
        logger.info(
            "Fuzzy matching: %d → %d propsów (odrzucono %d nieznanych graczy)",
            len(props), len(resolved), len(props) - len(resolved),
        )
        return resolved

    # ── Zapis do SQLite ───────────────────────────────────────────────────────
    def _save_to_db(self, props: list[dict]):
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO odds_history
                    (player_name, stat_type, line, over_odds, under_odds,
                     home_team, away_team, game_time, bookmaker, source)
                VALUES
                    (:player_name, :stat_type, :line, :over_odds, :under_odds,
                     :home_team, :away_team, :game_time, :bookmaker, :source)
                """,
                props,
            )
            conn.commit()
        logger.info("Zapisano %d propsów do odds_history", len(props))

    # ── Główna metoda ─────────────────────────────────────────────────────────
    def get_player_props(
        self,
        sport: str = SPORT,
        use_scraping: bool = True,
    ) -> list[dict]:
        """
        Pobiera propsy NBA:
          1. Playwright scraping (jeśli use_scraping=True)
          2. Fallback: The Odds API

        Zwraca listę dict:
          {player_name, stat_type, line, over_odds, under_odds,
           home_team, away_team, game_time, bookmaker, source}

        Filtruje: tylko player_points, kursy w przedziale ODDS_MIN–ODDS_MAX.
        Zapisuje wszystkie pobrane kursy do SQLite (analytics line movement).
        """
        raw_props: list[dict] = []
        source_used = "none"

        # ── Próba 1: Playwright ───────────────────────────────────────────────
        if use_scraping:
            for url in SCRAPE_TARGETS:
                scraped = asyncio.run(_scrape_props_playwright(url))
                if scraped:
                    # Playwright zwraca już spaired format
                    raw_props = scraped
                    source_used = "playwright"
                    logger.info("Playwright OK: %d propsów z %s", len(scraped), url)
                    break
                logger.warning("Playwright FAIL dla %s — próbuję kolejny", url)

        # ── Fallback: The Odds API ────────────────────────────────────────────
        if not raw_props:
            logger.info("Fallback → The Odds API")
            raw_api = self.api_client.get_player_props_raw(sport)
            if raw_api:
                raw_props = _pair_over_under(raw_api)
                source_used = "odds_api"
            else:
                logger.error("Oba źródła danych zawiodły — brak propsów!")
                return []

        logger.info(
            "Źródło: %s | Propsów przed filtrem: %d",
            source_used, len(raw_props),
        )

        # ── Pipeline: filter → fuzzy match → save ────────────────────────────
        props = self._filter_odds(raw_props)
        props = self._resolve_player_names(props)

        # Zapisz DO BAZY wszystkie (przed filtrem confidence) — line movement
        if props:
            self._save_to_db(props)

        return props

    # ── Analiza line movement ─────────────────────────────────────────────────
    def get_line_movement(
        self,
        player_name: str,
        hours_back: int = 24,
    ) -> pd.DataFrame:
        """
        Zwraca historię kursów dla gracza z ostatnich `hours_back` godzin.
        Przydatne do wykrywania "sharp movement" (nagły ruch linii).
        """
        query = """
            SELECT player_name, line, over_odds, under_odds,
                   bookmaker, fetched_at
            FROM odds_history
            WHERE player_name = ?
              AND fetched_at >= datetime('now', ?)
            ORDER BY fetched_at ASC
        """
        with self._conn() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(player_name, f"-{hours_back} hours"),
            )
        logger.info(
            "Line movement dla '%s': %d wpisów z ostatnich %dh",
            player_name, len(df), hours_back,
        )
        return df
