# nba_betting/injuries.py
"""
Oficjalny raport kontuzji NBA (PDF) — kto nie zagra, ZANIM mecz się zacznie.

Źródło: https://ak-static.cms.nba.com/referee/injury/Injury-Report_<data>_<godzina>.pdf
  - od sezonu 2025-26 co 15 min:  Injury-Report_2026-10-29_05_30PM.pdf
  - wcześniej co godzinę:          Injury-Report_2025-12-01_05PM.pdf
Godziny w czasie wschodnim (ET). Brak pliku → S3 zwraca 403 AccessDenied (to nie blokada).
Zwykłe zapytanie HTTP, bez udawania przeglądarki.

Statusy: Out, Doubtful, Questionable, Probable, Available. Do cech absencji liczymy
OUT_STATUSES (Out, Doubtful) — Doubtful prawie zawsze nie gra.

Użycie:
  python injuries.py                   # najnowszy raport → tabela
  python injuries.py --validate 2026-10-22 2026-10-29
      # zgodność raportu z tym, kto faktycznie nie zagrał (logi z data/raw)
"""
import argparse
import io
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from history import norm_name

logger = logging.getLogger(__name__)

BASE_URL      = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{}.pdf"
ET            = ZoneInfo("America/New_York")
STATUSES      = ("Out", "Doubtful", "Questionable", "Probable", "Available")
OUT_STATUSES  = ("Out", "Doubtful")
LOOKBACK_H    = 12
# nagłówki kolumn w PDF → nazwy pól (granice kolumn liczone z pozycji nagłówków)
HEADER = [("Game", "game_date"), ("Game", "game_time"), ("Matchup", "matchup"), ("Team", "team"),
          ("Player", "player"), ("Current", "status"), ("Reason", "reason")]

try:
    from nba_api.stats.static import teams as _nba_teams
    TEAM_FULL_TO_ABBR = {re.sub(r"\s", "", t["full_name"]).lower(): t["abbreviation"]
                         for t in _nba_teams.get_teams()}
except ImportError:  # pragma: no cover
    TEAM_FULL_TO_ABBR = {}


# ─────────────────────────────────────────────────────────────────────────────
# Pobieranie
# ─────────────────────────────────────────────────────────────────────────────
def report_names(t: datetime) -> list[str]:
    """Możliwe nazwy pliku dla chwili t (ET): nowy format co 15 min, stary co godzinę."""
    t = t.astimezone(ET)
    names = [t.strftime("%Y-%m-%d_%I_%M%p")]
    if t.minute == 0:
        names.append(t.strftime("%Y-%m-%d_%I%p"))
    return names


def latest_report(before: datetime, session: requests.Session | None = None) -> tuple[datetime, bytes] | None:
    """Najnowszy raport opublikowany nie później niż `before` (szuka do LOOKBACK_H wstecz)."""
    s = session or requests.Session()
    t = before.astimezone(ET).replace(second=0, microsecond=0)
    t -= timedelta(minutes=t.minute % 15)
    for _ in range(LOOKBACK_H * 4):
        for name in report_names(t):
            try:
                r = s.get(BASE_URL.format(name), timeout=20)
            except requests.RequestException as e:
                logger.warning("Raport kontuzji %s: %s", name, e)
                return None
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return t, r.content
        t -= timedelta(minutes=15)
    logger.warning("Brak raportu kontuzji w %d h przed %s", LOOKBACK_H, before)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Parsowanie
# ─────────────────────────────────────────────────────────────────────────────
def _player_key(raw: str) -> str:
    """'Porter Jr., Michael' → norm_name('Michael Porter Jr.')."""
    last, _, first = raw.partition(",")
    return norm_name(f"{first.strip()} {last.strip()}" if first else raw)


def parse_report(pdf_bytes: bytes) -> pd.DataFrame:
    """Wiersze zawodników: game_date, matchup, team (skrót), player, name_key, status."""
    import pdfplumber

    rows = []
    ctx = {"game_date": None, "matchup": None, "team": None}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        bounds = None
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5)
            # granice kolumn z nagłówka (powtarza się na każdej stronie)
            hdr = [w for w in words if w["text"] in {h for h, _ in HEADER}]
            if len(hdr) >= len(HEADER):
                xs = sorted(w["x0"] for w in hdr)[:len(HEADER)]
                bounds = [x - 3 for x in xs]
                hdr_top = max(w["top"] for w in hdr)
            if bounds is None:
                continue
            lines: dict[int, list] = {}
            for w in words:
                if w["top"] <= hdr_top + 2:
                    continue
                lines.setdefault(round(w["top"]), []).append(w)
            for top in sorted(lines):
                cols = {f: [] for _, f in HEADER}
                for w in sorted(lines[top], key=lambda w: w["x0"]):
                    i = max(k for k, b in enumerate(bounds) if w["x0"] >= b) if w["x0"] >= bounds[0] else 0
                    cols[HEADER[i][1]].append(w["text"])
                txt = {k: " ".join(v) for k, v in cols.items()}
                if re.match(r"\d{2}/\d{2}/\d{4}", txt["game_date"]):
                    ctx["game_date"] = txt["game_date"]
                if "@" in txt["matchup"]:
                    ctx["matchup"] = txt["matchup"].replace(" ", "")
                if txt["team"]:
                    ctx["team"] = TEAM_FULL_TO_ABBR.get(re.sub(r"\s", "", txt["team"]).lower(), ctx["team"])
                status = txt["status"].split(" ")[0] if txt["status"] else ""
                if status in STATUSES and "," in txt["player"]:
                    rows.append({**ctx, "player": txt["player"], "name_key": _player_key(txt["player"]),
                                 "status": status, "reason": txt["reason"]})
    df = pd.DataFrame(rows, columns=["game_date", "matchup", "team", "player", "name_key", "status", "reason"])
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"], format="%m/%d/%Y")
    return df


def out_players(report: pd.DataFrame, game_date) -> pd.DataFrame:
    """Zawodnicy Out/Doubtful na dany dzień (team, name_key)."""
    d = pd.Timestamp(game_date).normalize()
    r = report[(report["game_date"] == d) & report["status"].isin(OUT_STATUSES)]
    return r[["team", "name_key", "status"]].drop_duplicates()


def fetch_out_players(game_date, before: datetime | None = None) -> pd.DataFrame | None:
    """Najnowszy raport przed `before` (domyślnie teraz) → Out/Doubtful na game_date. None = brak raportu."""
    rep = latest_report(before or datetime.now(ET))
    if rep is None:
        return None
    t, pdf = rep
    df = parse_report(pdf)
    logger.info("Raport kontuzji %s ET: %d wierszy, %d Out/Doubtful na %s",
                t.strftime("%Y-%m-%d %H:%M"), len(df), len(out_players(df, game_date)), game_date)
    return out_players(df, game_date)


# ─────────────────────────────────────────────────────────────────────────────
# Walidacja: czy „Out” z raportu = nie zagrał?
# ─────────────────────────────────────────────────────────────────────────────
def validate(dates: list[str], hour_et: int = 13, minute: int = 30) -> dict:
    """
    Dla każdego dnia: raport z godziny hour_et:minute ET (≈ wieczorny run pipeline'u)
    vs logi. Liczy dla zawodników rotacji (śr. ≥ ROTATION_MIN min w ostatnich 10 meczach):
      precision = ilu z Out/Doubtful faktycznie nie zagrało,
      recall    = ilu nieobecnych z rotacji raport oznaczył jako Out/Doubtful.
    """
    from history import ROTATION_MIN, load_logs

    p, _ = load_logs(offline=True)
    p = p.sort_values("GAME_DATE")
    p["name_key"] = p["PLAYER_NAME"].map(norm_name)
    stats = {"days": 0, "out_listed": 0, "out_listed_played": 0, "rotation_absent": 0, "rotation_absent_listed": 0}
    misses = []
    for d in dates:
        day = pd.Timestamp(d)
        t = datetime(day.year, day.month, day.day, hour_et, minute, tzinfo=ET)
        rep = latest_report(t)
        if rep is None:
            continue
        r = parse_report(rep[1])
        r = r[r["game_date"] == day]
        played = p[p["GAME_DATE"] == day]
        if played.empty or r.empty:
            continue
        stats["days"] += 1
        teams_today = set(played["TEAM_ABBREVIATION"])
        played_keys = set(played["name_key"])
        out = r[r["status"].isin(OUT_STATUSES)]
        stats["out_listed"] += len(out)
        stats["out_listed_played"] += int(out["name_key"].isin(played_keys).sum())
        # rotacja: ostatnie 10 meczów przed dniem, ta sama drużyna, mecz w ciągu 14 dni
        hist = p[(p["GAME_DATE"] < day) & (p["GAME_DATE"] >= day - pd.Timedelta(days=14))]
        last = hist.groupby("PLAYER_ID").tail(1)
        avg_min = p[p["GAME_DATE"] < day].groupby("PLAYER_ID")["MIN"].apply(lambda s: s.tail(10).mean())
        rot = last[last["TEAM_ABBREVIATION"].isin(teams_today) & last["PLAYER_ID"].map(avg_min).ge(ROTATION_MIN)]
        absent = rot[~rot["PLAYER_ID"].isin(played["PLAYER_ID"])]
        listed = absent["name_key"].isin(set(out["name_key"]))
        stats["rotation_absent"] += len(absent)
        stats["rotation_absent_listed"] += int(listed.sum())
        misses += [f"{d} {n}" for n in absent.loc[~listed, "PLAYER_NAME"]]
    stats["precision"] = round(1 - stats["out_listed_played"] / max(stats["out_listed"], 1), 3)
    stats["recall"] = round(stats["rotation_absent_listed"] / max(stats["rotation_absent"], 1), 3)
    stats["not_listed_examples"] = misses[:15]
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", nargs=2, metavar=("OD", "DO"))
    ap.add_argument("--hour", type=int, default=13)
    args = ap.parse_args()
    if args.validate:
        days = pd.date_range(*args.validate).strftime("%Y-%m-%d").tolist()
        print(validate(days, hour_et=args.hour))
    else:
        rep = latest_report(datetime.now(ET))
        if rep:
            df = parse_report(rep[1])
            print(rep[0], len(df))
            print(df.head(30).to_string())
