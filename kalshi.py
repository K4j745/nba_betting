# nba_betting/kalshi.py
"""
Kalshi — darmowe, publiczne (bez klucza) ceny rynków na statystyki zawodników NBA.

Serie: KXNBAPTS (punkty), KXNBAREB (zbiórki), KXNBAAST (asysty). Rynek = „Gracz: N+”
(floor_strike = N − 0,5, czyli over N − 0,5). Cena YES ∈ [0, 1] = prawdopodobieństwo rynku;
bierzemy środek bid/ask, tylko gdy spread ≤ MAX_SPREAD (rzadkie rynki z bid 0,05 / ask 0,99
nic nie mówią). Kalshi jest dostępne tylko w USA — tu służy jako ODNIESIENIE rynkowe.

Czas: occurrence_datetime = start meczu + 3 h (sprawdzone: CLE@ATL 29.10.2026, start 00:00 UTC,
occurrence 03:00); w starszych rynkach brak tego pola — wtedy expected_expiration_time (też start + 3 h,
sprawdzone: CHA@ORL 17.04.2026, GSW@ORL 18.11.2025). Cena historyczna = ostatnia świeca godzinowa kończąca się ≥ 1 h przed startem.
Ticker wydarzenia: KXNBAREB-26OCT29CLEATL → data 2026-10-29, gość CLE, gospodarz ATL.

Rynki rozliczone przed /historical/cutoff są w archiwum (/historical/markets); ich świece
pobiera się pojedynczo, więc dla archiwum ceny tylko wokół linii głównej (bisekcja).
Historia (od grudnia 2025) → data/raw/kalshi_lines.csv.gz, uzupełniana przyrostowo przez
laptop (scripts/sync_nba_data.py) i trafia na branch `data` razem z logami NBA.

Użycie:
  python kalshi.py --backfill            # cała historia (kilka tysięcy zapytań, ~15 min)
  python kalshi.py --update              # tylko nowe rozliczone wydarzenia
  python kalshi.py --live REB            # otwarte rynki teraz (główne linie)
"""
import argparse
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from history import RAW_DIR, norm_name

logger = logging.getLogger(__name__)

API          = "https://api.elections.kalshi.com/trade-api/v2"
SERIES       = {"PTS": "KXNBAPTS", "REB": "KXNBAREB", "AST": "KXNBAAST"}
HISTORY_PATH = RAW_DIR / "kalshi_lines.csv.gz"
MAX_SPREAD   = 0.10
TIP_OFFSET   = timedelta(hours=3)       # occurrence_datetime − start meczu
SNAP_BEFORE  = timedelta(hours=1)       # cena historyczna: ≥ 1 h przed startem
BATCH        = 100                      # limit tickerów w /markets/candlesticks
SLEEP        = 0.08                     # ~12 zapytań/s — poniżej limitu publicznego API
SAVE_EVERY   = 100                      # zapis historii co tyle wydarzeń
FEE_RATE     = 0.07                     # prowizja: ceil(0,07 · C · P · (1 − P)) za C kontraktów

_session = requests.Session()


def _get(path: str, **params) -> dict:
    for attempt in range(5):
        try:
            r = _session.get(f"{API}{path}", params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 4:
                raise
            logger.warning("Kalshi %s: %s — ponawiam", path, e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        time.sleep(SLEEP)
        return r.json()
    r.raise_for_status()
    return {}


def _paginate(path: str, key: str, **params) -> list[dict]:
    out, cursor = [], None
    while True:
        d = _get(path, limit=1000 if key == "markets" else 200, **params, **({"cursor": cursor} if cursor else {}))
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return out


# ─────────────────────────────────────────────────────────────────────────────
# Parsowanie
# ─────────────────────────────────────────────────────────────────────────────
def parse_event_ticker(event_ticker: str) -> tuple[pd.Timestamp, str, str]:
    """'KXNBAREB-26OCT29CLEATL' → (2026-10-29, 'CLE', 'ATL')."""
    code = event_ticker.split("-")[1]
    day = pd.to_datetime(code[:7], format="%y%b%d")
    return day, code[7:10], code[10:13]


def parse_market(m: dict, stat: str) -> dict:
    day, away, home = parse_event_ticker(m["event_ticker"])
    player = (m.get("yes_sub_title") or m["title"]).split(":")[0].strip()
    # starsze rynki (do ~04.2026) nie mają occurrence_datetime; expected_expiration_time = start + 3 h
    occ = m.get("occurrence_datetime") or m.get("expected_expiration_time")
    occ = pd.Timestamp(occ) if occ else pd.NaT
    actual = pd.to_numeric(m.get("expiration_value"), errors="coerce")
    return {
        "stat": stat, "ticker": m["ticker"], "event_ticker": m["event_ticker"],
        "player": player, "name_key": norm_name(player),
        "game_date": day, "away": away, "home": home,
        "tip": occ - TIP_OFFSET if pd.notna(occ) else pd.NaT,
        "line": float(m["floor_strike"]),
        "result": m.get("result") or None, "actual": actual,
    }


def _mid(bid, ask) -> tuple[float, float, float]:
    bid, ask = float(bid or 0), float(ask or 0)
    if bid <= 0 or ask <= 0 or ask - bid > MAX_SPREAD:
        return bid, ask, float("nan")
    return bid, ask, (bid + ask) / 2


def kalshi_fee(price: float) -> float:
    """Prowizja Kalshi (taker) za 1 kontrakt kupiony po cenie `price`, zaokrąglona w górę do centa."""
    return math.ceil(round(FEE_RATE * price * (1 - price) * 100, 6)) / 100


def main_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Jedna linia na (statystyka, zawodnik, mecz): próg z ceną najbliższą 0,5 („linia główna”)."""
    ok = lines.dropna(subset=["p_mid"]).copy()
    ok["_d"] = (ok["p_mid"] - 0.5).abs()
    return ok.sort_values("_d").drop_duplicates(["stat", "event_ticker", "name_key"]).drop(columns="_d")


# ─────────────────────────────────────────────────────────────────────────────
# Historia: rozliczone rynki + cena ze świec 1 h przed meczem
# ─────────────────────────────────────────────────────────────────────────────
def _snapshot_prices(markets: pd.DataFrame) -> pd.DataFrame:
    """Dla rynków jednego wydarzenia (ten sam start): bid/ask/mid z ostatniej świecy ≤ start − 1 h."""
    tip = markets["tip"].iloc[0]
    target = int((tip - SNAP_BEFORE).timestamp())
    rows = []
    tickers = markets["ticker"].tolist()
    for i in range(0, len(tickers), BATCH):
        d = _get("/markets/candlesticks", market_tickers=",".join(tickers[i:i + BATCH]),
                 start_ts=target - 24 * 3600, end_ts=target, period_interval=60)
        for mk in d.get("markets", []):
            cs = [c for c in mk.get("candlesticks", []) if c["end_period_ts"] <= target]
            if not cs:
                continue
            rows.append(_candle_row(mk["market_ticker"], cs[-1]))
    return pd.DataFrame(rows, columns=["ticker", "bid", "ask", "p_mid", "snap_ts"])


def _candle_row(ticker: str, c: dict) -> dict:
    """Świeca → bid/ask/mid (API bieżące: close_dollars, archiwum: close)."""
    px = lambda side: c[side].get("close_dollars", c[side].get("close"))
    bid, ask, mid = _mid(px("yes_bid"), px("yes_ask"))
    return {"ticker": ticker, "bid": bid, "ask": ask, "p_mid": mid,
            "snap_ts": pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC")}


def _hist_candle(ticker: str, target: int) -> dict | None:
    """Archiwum (rynki rozliczone przed /historical/cutoff): świece tylko pojedynczo."""
    d = _get(f"/historical/markets/{ticker}/candlesticks",
             start_ts=target - 24 * 3600, end_ts=target, period_interval=60)
    cs = [c for c in d.get("candlesticks", []) if c["end_period_ts"] <= target]
    return _candle_row(ticker, cs[-1]) if cs else None


def _bisect_prices(markets: pd.DataFrame) -> pd.DataFrame:
    """
    Archiwum: zamiast świec dla każdego progu (~45 rynków na mecz) bisekcja po drabince progów
    każdego zawodnika — szukamy progu z ceną najbliższą 0,5 (linia główna), ~2–3 zapytania.
    """
    target = int((markets["tip"].iloc[0] - SNAP_BEFORE).timestamp())
    rows = []
    for _, pm in markets.sort_values("line").groupby("name_key"):
        tickers = pm["ticker"].tolist()
        lo, hi = 0, len(tickers) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            r = _hist_candle(tickers[mid], target)
            if r is None:
                break
            rows.append(r)
            p = r["p_mid"] if pd.notna(r["p_mid"]) else (r["bid"] + r["ask"]) / 2
            if p > 0.5:      # over wyżej niż 50% → linia główna na wyższym progu
                lo = mid + 1
            else:
                hi = mid - 1
    return pd.DataFrame(rows, columns=["ticker", "bid", "ask", "p_mid", "snap_ts"])


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    h = pd.read_csv(HISTORY_PATH, parse_dates=["game_date"])
    for c in ("tip", "snap_ts"):
        h[c] = pd.to_datetime(h[c], utc=True)
    return h


def update_history(stats=tuple(SERIES), full: bool = False) -> pd.DataFrame:
    """
    Dopisuje rozliczone wydarzenia, których nie ma w HISTORY_PATH (full=True: od zera).
    Zapytania: ~1 na 1000 rynków (lista) + ~1 na wydarzenie (świece, archiwum: ~2–3 na zawodnika).
    Zapis co SAVE_EVERY wydarzeń — przerwany backfill wznawia się przez --update.
    """
    hist = pd.DataFrame() if full else load_history()
    done = set(hist["event_ticker"]) if not hist.empty else set()
    new = []
    for stat in stats:
        raw = [(m, False) for m in _paginate("/markets", "markets", series_ticker=SERIES[stat], status="settled")]
        raw += [(m, True) for m in _paginate("/historical/markets", "markets", series_ticker=SERIES[stat])]
        ms = pd.DataFrame([{**parse_market(m, stat), "_archived": a} for m, a in raw])
        ms = ms[~ms["event_ticker"].isin(done) & ms["tip"].notna()].drop_duplicates("ticker")
        if ms.empty:
            continue
        logger.info("Kalshi %s: %d nowych rynków w %d wydarzeniach", stat, len(ms), ms["event_ticker"].nunique())
        for k, (_, ev) in enumerate(ms.groupby("event_ticker")):
            px = _bisect_prices(ev) if ev["_archived"].iloc[0] else _snapshot_prices(ev)
            ev = ev.drop(columns="_archived")
            new.append(ev.merge(px, on="ticker", how="left"))
            if (k + 1) % SAVE_EVERY == 0:
                hist, new = _save(hist, new)
                logger.info("  %s: %d/%d wydarzeń", stat, k + 1, ms["event_ticker"].nunique())
    hist, _ = _save(hist, new)
    return hist


def _save(hist: pd.DataFrame, new: list) -> tuple[pd.DataFrame, list]:
    if new:
        hist = pd.concat([hist, *new], ignore_index=True)
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(HISTORY_PATH, index=False)
        logger.info("Kalshi: %d rynków w historii (%s)", len(hist), HISTORY_PATH)
    return hist, []


# ─────────────────────────────────────────────────────────────────────────────
# Na żywo: otwarte rynki teraz
# ─────────────────────────────────────────────────────────────────────────────
def live_lines(stat: str, game_date=None) -> pd.DataFrame:
    """Otwarte rynki statystyki (cena = środek bid/ask teraz), opcjonalnie tylko mecze z game_date."""
    rows = []
    for m in _paginate("/markets", "markets", series_ticker=SERIES[stat], status="open"):
        r = parse_market(m, stat)
        r["bid"], r["ask"], r["p_mid"] = _mid(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
        r["snap_ts"] = pd.Timestamp.now(tz="UTC")
        rows.append(r)
    df = pd.DataFrame(rows)
    if game_date is not None and not df.empty:
        df = df[df["game_date"] == pd.Timestamp(game_date)]
    return df


def to_props(lines: pd.DataFrame) -> list[dict]:
    """
    Główne linie Kalshi → format propsów jak z Odds API. Kurs = 1 / (cena + prowizja):
    over = kupno YES po ask, under = kupno NO po (1 − bid).
    """
    from live import TEAM_NAME_TO_ABBR
    abbr_to_name = {v: k for k, v in TEAM_NAME_TO_ABBR.items()}
    props = []
    for r in main_lines(lines).itertuples():
        no = 1 - r.bid
        props.append({
            "player_name": r.player, "line": r.line,
            "over_odds": round(1 / (r.ask + kalshi_fee(r.ask)), 3),
            "under_odds": round(1 / (no + kalshi_fee(no)), 3),
            "p_market": r.p_mid,
            "home_team": abbr_to_name.get(r.home, r.home), "away_team": abbr_to_name.get(r.away, r.away),
            "game_time": r.tip.isoformat() if pd.notna(r.tip) else None,
            "bookmaker": "kalshi", "stat": r.stat,
        })
    return props


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backfill", action="store_true")
    g.add_argument("--update", action="store_true")
    g.add_argument("--live", choices=list(SERIES))
    args = ap.parse_args()
    if args.live:
        print(main_lines(live_lines(args.live)).head(30).to_string())
    else:
        h = update_history(full=args.backfill)
        print(h.groupby("stat").agg(rynki=("ticker", "size"), z_cena=("p_mid", "count"),
                                    mecze=("event_ticker", "nunique")))
