# nba_betting/referees.py
"""
Sędziowie meczu jako cecha dla zbiórek i asyst.

Obsada sędziowska jest publikowana przez NBA ok. 9:00 ET w dniu meczu, więc przed
ceną Kalshi (≥ 1 h przed meczem) jest znana — cecha jest uczciwa, ale rynek też ją zna.

Dane: boxscoresummaryv3 (stats.nba.com, przez NBAStatsHTTP — v2 nie zwraca sędziów
od sezonu 2025-26), jeden mecz na zapytanie → data/raw/officials.csv.gz (cache, dociągany).
stats.nba.com nie odpowiada z runnera CI, więc pobieranie tylko lokalnie.

Cechy (point-in-time): dla każdego meczu sumy obu drużyn — zbiórki, asysty, faule,
posiadania (FGA + 0,44·FTA − OREB + TOV) — minus średnia z PRIOR_GAMES poprzednich
meczów ligi. Efekt sędziego = średnia tych reszt z jego meczów sprzed dnia meczu,
ściągnięta do zera: n / (n + SHRINK_N0). Cecha meczu = średnia efektów trzech sędziów:
  ref_reb, ref_ast, ref_pf, ref_pace

Ocena jak market_tests.py: logit p = α + β·logit p_Kalshi + γ·x, walk-forward tygodniowo,
Δ log-loss z CI z bootstrapu po meczach (Bonferroni), selekcja na liniach sprzed FINAL_FROM.

Użycie:
  python referees.py fetch [--max-games N]
  python referees.py              # → reports/referees_<data>.json
  python referees.py --final      # holdout, tylko gdy coś przeszło dev
"""
import argparse
import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from feature_lab import FINAL_FROM
from history import RAW_DIR, load_dataset
from market_tests import evaluate

logger = logging.getLogger(__name__)

OFFICIALS_PATH = RAW_DIR / "officials.csv.gz"
FIRST_SEASON   = "2024-25"     # sezon przed pierwszą linią Kalshi: rozbieg efektów sędziów
SLEEP_S        = 2.5
BACKOFF_S      = (0, 60, 300, 900)   # przerwa przed kolejną próbą: timeout/odmowa → dłuższe czekanie
SAVE_EVERY     = 25
PRIOR_GAMES    = 300           # średnia ligi z tylu poprzednich meczów
SHRINK_N0      = 20            # mecze sędziego, przy których efekt jest ściągnięty w połowie
FEATURES       = ("ref_reb", "ref_ast", "ref_pf", "ref_pace")
STATS          = ("REB", "AST")


# ─────────────────────────────────────────────────────────────────────────────
# Pobieranie
# ─────────────────────────────────────────────────────────────────────────────
def fetch_game(game_id: str) -> list[dict]:
    from nba_api.stats.library.http import NBAStatsHTTP
    for attempt, wait in enumerate(BACKOFF_S):
        time.sleep(SLEEP_S + wait)
        try:
            j = NBAStatsHTTP().send_api_request(endpoint="boxscoresummaryv3",
                                                parameters={"GameID": game_id, "LeagueID": "00"},
                                                timeout=60).get_dict()
            offs = j.get("boxScoreSummary", {}).get("officials", []) or []
            return [{"GAME_ID": game_id, "OFFICIAL_ID": int(o["personId"]),
                     "OFFICIAL": f"{o.get('firstName', '')} {o.get('familyName', '')}".strip()} for o in offs]
        except Exception as e:                           # timeout/odmowa: dłuższa przerwa, potem rezygnacja
            logger.warning("%s próba %d: %s", game_id, attempt + 1, e)
    raise RuntimeError(f"stats.nba.com nie odpowiada ({game_id}) — przerywam")


def load_officials() -> pd.DataFrame:
    if not OFFICIALS_PATH.exists():
        return pd.DataFrame(columns=["GAME_ID", "OFFICIAL_ID", "OFFICIAL"])
    return pd.read_csv(OFFICIALS_PATH, dtype={"GAME_ID": str})


def fetch(max_games: int | None = None) -> int:
    """Dociąga sędziów meczów od FIRST_SEASON, których brakuje w cache. Mecz bez sędziów → wiersz z NaN."""
    data = load_dataset()
    games = (data[data["SEASON"] >= FIRST_SEASON].drop_duplicates("GAME_ID")
             .sort_values("GAME_DATE")["GAME_ID"].astype(str).tolist())
    have = load_officials()
    todo = [g for g in games if g not in set(have["GAME_ID"])]
    logger.info("Sędziowie: %d meczów w cache, %d do pobrania", have["GAME_ID"].nunique(), len(todo))
    rows, done = [], 0
    for g in todo[:max_games]:
        rows += fetch_game(g) or [{"GAME_ID": g, "OFFICIAL_ID": np.nan, "OFFICIAL": None}]
        done += 1
        if done % SAVE_EVERY == 0 or done == len(todo[:max_games]):
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            have.to_csv(OFFICIALS_PATH, index=False, compression="gzip")
            rows = []
            logger.info("pobrano %d/%d meczów", done, len(todo[:max_games]))
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Cechy
# ─────────────────────────────────────────────────────────────────────────────
def game_totals(data: pd.DataFrame) -> pd.DataFrame:
    """Sumy obu drużyn na mecz i reszty względem średniej z PRIOR_GAMES poprzednich meczów (bez tego samego dnia)."""
    g = data.groupby("GAME_ID").agg(GAME_DATE=("GAME_DATE", "first"), REB=("REB", "sum"), AST=("AST", "sum"),
                                    PF=("PF", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum"),
                                    OREB=("OREB", "sum"), TOV=("TOV", "sum")).reset_index()
    g["PACE"] = g["FGA"] + 0.44 * g["FTA"] - g["OREB"] + g["TOV"]
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    for c, f in (("REB", "ref_reb"), ("AST", "ref_ast"), ("PF", "ref_pf"), ("PACE", "ref_pace")):
        # średnia z PRIOR_GAMES meczów zakończonych przed dniem meczu
        day_end = g[c].rolling(PRIOR_GAMES, min_periods=50).mean().groupby(g["GAME_DATE"]).last()
        base = g["GAME_DATE"].map(day_end.shift(1))
        g[f"res_{f}"] = g[c] - base
    g["GAME_ID"] = g["GAME_ID"].astype(str)
    return g


def referee_features(totals: pd.DataFrame, officials: pd.DataFrame) -> pd.DataFrame:
    """GAME_ID → ref_* : średnia efektów sędziów z ich meczów sprzed dnia meczu (ściągnięta do 0)."""
    off = officials.dropna(subset=["OFFICIAL_ID"]).merge(totals, on="GAME_ID", how="inner")
    off = off.sort_values(["OFFICIAL_ID", "GAME_DATE"]).reset_index(drop=True)
    eff = off[["GAME_ID", "OFFICIAL_ID", "GAME_DATE"]].copy()
    for f in FEATURES:
        r = off[f"res_{f}"]
        grp = off["OFFICIAL_ID"]
        csum = r.fillna(0).groupby(grp).cumsum() - r.fillna(0)
        cnt = r.notna().astype(int).groupby(grp).cumsum() - r.notna().astype(int)
        # mecze tego samego dnia nie są „wcześniej”: cofnij do stanu sprzed dnia
        key = [grp, off["GAME_DATE"]]
        csum, cnt = csum.groupby(key).transform("first"), cnt.groupby(key).transform("first")
        mean = csum / cnt.where(cnt > 0)
        eff[f] = (mean * cnt / (cnt + SHRINK_N0)).fillna(0.0)
    return eff.groupby("GAME_ID")[list(FEATURES)].mean().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Ocena
# ─────────────────────────────────────────────────────────────────────────────
def run(final: bool = False) -> dict:
    rng = np.random.default_rng(0)
    officials = load_officials()
    if officials.empty:
        raise FileNotFoundError("Brak data/raw/officials.csv.gz — uruchom: python referees.py fetch")
    data = load_dataset()
    feats = referee_features(game_totals(data), officials)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    m = pred[pred["stat"].isin(STATS)].merge(feats, on="GAME_ID", how="left")
    covered = m[list(FEATURES)].notna().all(axis=1)
    m[list(FEATURES)] = m[list(FEATURES)].fillna(0.0)
    m["lm"] = mm._logit(m["p_mid"])
    report = {"generated": date.today().isoformat(), "sample": "final" if final else "dev",
              "final_from": str(FINAL_FROM.date()), "predictions": mm.latest_predictions().name,
              "coverage": {st: round(float(covered[m["stat"] == st].mean()), 3) for st in STATS},
              "n_officials": int(officials["OFFICIAL_ID"].nunique()),
              "feature_sd": {f: round(float(feats[f].std()), 3) for f in FEATURES},
              **evaluate(m, final, FEATURES, rng)}
    out = mm.REPORT_DIR / f"referees_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="eval", choices=["fetch", "eval"])
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.max_games)
    else:
        run(a.final)
