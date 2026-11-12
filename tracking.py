# nba_betting/tracking.py
"""
Statystyki trackingowe NBA (Second Spectrum) per zawodnik-mecz i test: czy poprawiają cenę Kalshi
dla zbiórek i asyst?

Źródło: nba_api LeagueDashPtStats z date_from = date_to = dzień meczu. Zawodnik gra najwyżej jeden mecz
dziennie, więc sumy z jednego dnia = statystyki z jednego meczu. Cztery typy pomiarów na dzień:
  Rebounding  szanse na zbiórkę (REB_CHANCES), zbiórki contested / uncontested,
  Passing     podania, potencjalne asysty (POTENTIAL_AST), asysty wtórne,
  Possessions kontakty z piłką (TOUCHES), czas posiadania,
  Drives      wejścia pod kosz (DRIVES) i podania po nich.
Cache: data/raw/TR_<sezon>_<rs|po>.csv.gz, dopisywany dzień po dniu (przerwane pobieranie można
wznowić). stats.nba.com nie odpowiada runnerom Actions — pobiera laptop, jak logi w history.py.

Cechy (point-in-time: shift(1) przed oknami, czyli wyłącznie mecze sprzed dnia linii):
  x{S}_10     oczekiwana statystyka z trackingu = średnia wolumenu z 10 meczów × skuteczność zawodnika
              z 40 meczów ściągnięta do średniej ligi (SHRINK_K szans); wolumen: REB_CHANCES / POTENTIAL_AST,
  gap         x{S}_10 − średnia statystyki z 10 meczów   (pech/szczęście: rynek patrzy na wynik, nie na szanse),
  line        x{S}_10 − linia Kalshi                     (czy tracking widzi inną średnią niż rynek),
  trend       log(wolumen z 5 meczów / wolumen z 20)      (zmiana roli, zanim widać ją w wyniku).
Brak historii trackingu → 0 (jak w market_tests). Cechy ustalone przed spojrzeniem na wyniki.

Ocena jak market_tests.py: logit p = α + β·logit p_Kalshi + γ·x, walk-forward tygodniowo, Δ log-loss
względem samej rekalibracji rynku, CI z bootstrapu po meczach na poziomie 1 − 0,05 / liczba testów,
selekcja tylko na liniach < FINAL_FROM; --final wyłącznie dla cech, które przeszły dev.

Użycie:
  python tracking.py fetch               # dociągnij brakujące dni (sezony od FIRST_TRACK_SEASON)
  python tracking.py                     # → reports/tracking_<data>.json (dev)
  python tracking.py --final gap_REB …   # potwierdzenie na liniach od FINAL_FROM
"""
import argparse
import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

import market_model as mm
from evaluate_stats import _logloss
from feature_lab import FINAL_FROM
from history import RAW_DIR, SEASON_TYPES, load_logs
from market_tests import game_ci

logger = logging.getLogger(__name__)

FIRST_TRACK_SEASON = "2024-25"   # sezon przed pierwszą linią Kalshi (11.2025): rozbieg okien i średnia ligi
MEASURES     = ("Rebounding", "Passing", "Possessions", "Drives")
SLEEP_S      = 1.5               # przerwa między zapytaniami do stats.nba.com
RETRIES      = 3
VOLUME       = {"REB": "REB_CHANCES", "AST": "POTENTIAL_AST"}
STATS        = tuple(VOLUME)
FEATURES     = ("gap", "line", "trend")
SHRINK_K     = 30.0              # szans: siła ściągania skuteczności zawodnika do średniej ligi
CONV_WINDOW  = 40
ID_COLS      = ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "W", "L", "MIN")
STABILITY_COLS = ("REB", "REB_CHANCES", "REB_CONTEST", "REB_UNCONTEST", "AST", "POTENTIAL_AST",
                  "PASSES_MADE", "TOUCHES", "DRIVES")
N_BOOT       = 5000
REB_COUNT_COLS = ("OREB_CONTEST", "OREB_UNCONTEST", "OREB_CHANCES", "DREB_CONTEST", "DREB_UNCONTEST",
                  "DREB_CHANCES", "REB_CONTEST", "REB_UNCONTEST", "REB_CHANCES")


# ─────────────────────────────────────────────────────────────────────────────
# Pobieranie
# ─────────────────────────────────────────────────────────────────────────────
def _path(season: str, season_type: str):
    return RAW_DIR / f"TR_{season}_{'rs' if season_type == 'Regular Season' else 'po'}.csv.gz"


def merge_measures(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Łączy tabele czterech pomiarów jednego dnia po PLAYER_ID; kolumny powtórzone bierze z pierwszej."""
    out = None
    for df in frames.values():
        if df.empty:
            continue
        if out is None:
            out = df.copy()
            continue
        out = out.merge(df, on="PLAYER_ID", how="outer", suffixes=("", "_dup"))
        for c in [c for c in out.columns if c.endswith("_dup")]:
            out[c[:-4]] = out[c[:-4]].fillna(out[c])
            out = out.drop(columns=c)
    return out if out is not None else pd.DataFrame()


def fetch_day(day: pd.Timestamp, season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashptstats
    frames = {}
    for m in MEASURES:
        for attempt in range(RETRIES):
            time.sleep(SLEEP_S * (1 + 2 * attempt))
            try:
                frames[m] = leaguedashptstats.LeagueDashPtStats(
                    season=season, season_type_all_star=season_type, per_mode_simple="Totals",
                    player_or_team="Player", pt_measure_type=m,
                    date_from_nullable=day.strftime("%m/%d/%Y"), date_to_nullable=day.strftime("%m/%d/%Y"),
                    timeout=60).get_data_frames()[0]
                break
            except Exception as e:                       # timeout/odmowa: dłuższa przerwa, potem rezygnacja
                logger.warning("%s %s próba %d: %s", day.date(), m, attempt + 1, e)
        else:
            raise RuntimeError(f"stats.nba.com nie odpowiada ({day.date()} {m}) — przerywam")
    df = merge_measures(frames)
    return df.assign(GAME_DATE=day.normalize(), SEASON=season)


def fetch(seasons: list[str] | None = None, max_days: int | None = None) -> int:
    """Dociąga dni meczowe z logów, których brakuje w cache. Zwraca liczbę pobranych dni."""
    logs, _ = load_logs(offline=True)
    seasons = seasons or sorted(s for s in logs["SEASON"].unique() if s >= FIRST_TRACK_SEASON)
    done = 0
    for season in seasons:
        for st in SEASON_TYPES:
            days = logs[(logs["SEASON"] == season) & (logs["IS_PLAYOFFS"] == int(st == "Playoffs"))]["GAME_DATE"]
            path = _path(season, st)
            have = pd.read_csv(path, parse_dates=["GAME_DATE"]) if path.exists() else pd.DataFrame()
            got = set(have["GAME_DATE"]) if not have.empty else set()
            todo = sorted(set(days.dt.normalize()) - got)
            logger.info("%s %s: %d dni w cache, %d do pobrania", season, st, len(got), len(todo))
            for day in todo:
                if max_days is not None and done >= max_days:
                    return done
                have = pd.concat([have, fetch_day(day, season, st)], ignore_index=True)
                have.to_csv(path, index=False, compression="gzip")
                done += 1
                if done % 20 == 0:
                    logger.info("pobrano %d dni (ostatni %s)", done, day.date())
    return done


def load_tracking() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("TR_*.csv.gz"))
    if not files:
        raise FileNotFoundError("Brak data/raw/TR_*.csv.gz — uruchom: python tracking.py fetch")
    tr = pd.concat([pd.read_csv(f, parse_dates=["GAME_DATE"]) for f in files], ignore_index=True)
    return tr.drop_duplicates(["PLAYER_ID", "GAME_DATE"], keep="last")


# ─────────────────────────────────────────────────────────────────────────────
# Cechy
# ─────────────────────────────────────────────────────────────────────────────
def player_games(logs: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Mecze zawodników (z logów) z dołączonym trackingiem z tego samego dnia."""
    g = logs[logs["MIN"] > 0][["PLAYER_ID", "GAME_ID", "GAME_DATE", "MIN", "REB", "AST"]].copy()
    g["GAME_DATE"] = g["GAME_DATE"].dt.normalize()
    cols = [c for c in tr.columns if c not in ID_COLS and c not in ("GAME_DATE", "SEASON", "REB", "AST")]
    g = g.merge(tr[["PLAYER_ID", "GAME_DATE", *cols]], on=["PLAYER_ID", "GAME_DATE"], how="left")
    # brak w tabeli Rebounding przy obecności w Passing = zero szans (krótkie występy, 0 zbiórek w logach)
    present = g["POTENTIAL_AST"].notna()
    for c in REB_COUNT_COLS:
        if c in g:
            g.loc[present, c] = g.loc[present, c].fillna(0.0)
    g["GAME_ID"] = g["GAME_ID"].astype(str).str.zfill(10)
    return g.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def league_conversion(g: pd.DataFrame, before: pd.Timestamp) -> dict[str, float]:
    """Skuteczność ligi (statystyka / wolumen) z meczów przed `before` — stała bez wycieku."""
    p = g[(g["GAME_DATE"] < before) & g["REB_CHANCES"].notna()]
    return {s: float(p[s].sum() / p[v].sum()) for s, v in VOLUME.items()}


def rolling_features(g: pd.DataFrame, conv: dict[str, float]) -> pd.DataFrame:
    """Cechy na wiersz mecz-zawodnik z meczów WCZEŚNIEJSZYCH (shift(1))."""
    g = g.copy()
    def roll(col, w, how="mean", minp=None):
        prev = g.groupby("PLAYER_ID")[col].shift(1)
        r = prev.groupby(g["PLAYER_ID"]).rolling(w, min_periods=minp or max(1, w // 2))
        return getattr(r, how)().reset_index(level=0, drop=True)
    for s, v in VOLUME.items():
        has = g[v].notna()
        g[f"_{s}_t"] = g[s].where(has)                    # statystyka tylko z meczów z trackingiem
        stat_sum, vol_sum = roll(f"_{s}_t", CONV_WINDOW, "sum", 1), roll(v, CONV_WINDOW, "sum", 1)
        conv_p = (stat_sum.fillna(0) + SHRINK_K * conv[s]) / (vol_sum.fillna(0) + SHRINK_K)
        g[f"x{s}_10"] = roll(v, 10) * conv_p
        g[f"{s}_10"] = roll(s, 10)
        g[f"gap_{s}"] = g[f"x{s}_10"] - g[f"{s}_10"]
        g[f"trend_{s}"] = np.log((roll(v, 5, minp=3) + 0.5) / (roll(v, 20, minp=10) + 0.5))
        g = g.drop(columns=f"_{s}_t")
    return g


def attach(pred: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Dokleja cechy do linii Kalshi (pred z evaluate_stats) i liczy `line`. Brak historii → 0."""
    keep = ["GAME_ID", "PLAYER_ID", *[f"{c}{s}{t}" for s in STATS for c, t in (("x", "_10"), ("gap_", ""),
                                                                                ("trend_", ""))]]
    f = feats[keep].copy()
    m = pred.assign(GAME_ID=pred["GAME_ID"].astype(str).str.zfill(10)).merge(f, on=["GAME_ID", "PLAYER_ID"],
                                                                             how="left")
    m["has_tracking"] = 0
    for s in STATS:
        is_s = m["stat"] == s
        m.loc[is_s, "has_tracking"] = m.loc[is_s, f"x{s}_10"].notna().astype(int)
        m[f"line_{s}"] = np.where(is_s, m[f"x{s}_10"] - m["line"], np.nan)
        for p in FEATURES:
            m.loc[~is_s, f"{p}_{s}"] = np.nan
            m.loc[is_s, f"{p}_{s}"] = m.loc[is_s, f"{p}_{s}"].fillna(0.0)
    m["lm"] = mm._logit(m["p_mid"])
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Ocena
# ─────────────────────────────────────────────────────────────────────────────
def stability(g: pd.DataFrame, since: pd.Timestamp) -> dict:
    """Hipoteza „tracking stabilniejszy”: korelacja średniej z 10 poprzednich meczów z następnym meczem
    i z kolejnymi 10 (bez nakładania), wiersze od `since` z ≥ 10 meczami historii z trackingiem."""
    out = {}
    by = g.groupby("PLAYER_ID")
    for c in STABILITY_COLS:
        if c not in g:
            continue
        prev = by[c].shift(1)
        past = prev.groupby(g["PLAYER_ID"]).rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
        nxt = by[c].transform(lambda s: s[::-1].rolling(10, min_periods=10).mean()[::-1])
        ok = (g["GAME_DATE"] >= since) & past.notna() & g[c].notna()
        ok10 = ok & nxt.notna()
        out[c] = {"n": int(ok.sum()), "r_next_game": round(float(np.corrcoef(past[ok], g.loc[ok, c])[0, 1]), 3),
                  "r_next_10": round(float(np.corrcoef(past[ok10], nxt[ok10])[0, 1]), 3)}
    # czy oczekiwana statystyka z trackingu przewiduje następny mecz lepiej niż sama średnia z 10?
    for s in STATS:
        ok = (g["GAME_DATE"] >= since) & g[f"x{s}_10"].notna() & g[f"{s}_10"].notna() & g[f"{s}"].notna()
        y = g.loc[ok, s]
        out[f"mae_next_{s}"] = {"n": int(ok.sum()),
                                "avg10": round(float((g.loc[ok, f"{s}_10"] - y).abs().mean()), 4),
                                "tracking_x10": round(float((g.loc[ok, f"x{s}_10"] - y).abs().mean()), 4)}
    return out


def evaluate(m: pd.DataFrame, final: bool, names: dict[str, list[str]], rng) -> dict:
    n_tests = sum(len(v) for v in names.values())
    level = 1 - 0.05 / n_tests
    out = {"ci_level": round(level, 4), "n_tests": n_tests}
    for stat, feats in names.items():
        d = m[m["stat"] == stat].dropna(subset=["p_mid", "y_over"]).sort_values("game_date").reset_index(drop=True)
        keep = ((d["game_date"] >= FINAL_FROM) if final else (d["game_date"] < FINAL_FROM)).to_numpy()
        ref, _ = mm.walk_forward(d, ["lm"])
        y, games = d["y_over"].to_numpy(), d["GAME_ID"].to_numpy()
        ll_mid = _logloss(d["p_mid"].to_numpy(), y)
        res = {"n_lines": int(keep.sum()), "tracking_coverage": round(float(d.loc[keep, "has_tracking"].mean()), 3)}
        for f in feats:
            col = f"{f}_{stat}"
            p, _ = mm.walk_forward(d, ["lm", col])
            ok = keep & ~np.isnan(p) & ~np.isnan(ref)
            diff = _logloss(p[ok], y[ok]) - _logloss(ref[ok], y[ok])
            vs_mid = _logloss(p[ok], y[ok]) - ll_mid[ok]
            ci = game_ci(diff, games[ok], rng, level)
            res[f] = {"n": int(ok.sum()), "n_games": int(len(np.unique(games[ok]))),
                      "delta_logloss_vs_recal": round(float(diff.mean()), 5), "ci": ci,
                      "ci95": game_ci(diff, games[ok], rng, 0.95),
                      "delta_logloss_vs_kalshi": round(float(vs_mid.mean()), 5),
                      "ci95_vs_kalshi": game_ci(vs_mid, games[ok], rng, 0.95),
                      "passed": ci[1] < 0,
                      "gamma": mm.full_fit_ci(d[keep], ["lm", col], rng, n_boot=200)[col]}
        out[stat] = res
        logger.info("%s: %s", stat, {k: (v["delta_logloss_vs_recal"], v["ci"], v["gamma"]["coef"])
                                    for k, v in res.items() if isinstance(v, dict)})
    return out


def run(final: bool = False, only: list[str] | None = None) -> dict:
    rng = np.random.default_rng(0)
    logs, _ = load_logs(offline=True)
    tr = load_tracking()
    g = player_games(logs, tr)
    first_line = pd.Timestamp("2025-10-01")
    conv = league_conversion(g, first_line)
    g = rolling_features(g, conv)
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    m = attach(pred[pred["stat"].isin(STATS)], g)
    if only:                                           # --final: tylko cechy, które przeszły dev
        names = {s: [f for f in FEATURES if f"{f}_{s}" in only] for s in STATS}
        names = {s: v for s, v in names.items() if v}
    else:
        names = {s: list(FEATURES) for s in STATS}
    report = {"generated": date.today().isoformat(), "sample": "final" if final else f"dev (< {FINAL_FROM.date()})",
              "final_from": str(FINAL_FROM.date()), "predictions": str(mm.latest_predictions()),
              "formula": "logit p = a + b·logit(p_kalshi) + g·x; Δ vs a + b·logit(p_kalshi)",
              "tracking_days": int(tr["GAME_DATE"].nunique()),
              "tracking_range": [str(tr["GAME_DATE"].min().date()), str(tr["GAME_DATE"].max().date())],
              "league_conversion": {s: round(v, 4) for s, v in conv.items()},
              "stability": stability(g, first_line),
              **evaluate(m, final, names, rng)}
    passed = [f"{f}_{s}" for s in names for f in names[s] if report[s][f]["passed"]]
    report["passed"] = passed
    report["verdict"] = (f"przechodzi: {', '.join(passed)} — potwierdź: python tracking.py --final {' '.join(passed)}"
                         if passed else "brak przewagi: żadna cecha nie poprawia ceny Kalshi; holdout nieużyty")
    out = mm.REPORT_DIR / f"tracking_{report['generated']}{'_final' if final else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="eval", choices=("eval", "fetch"))
    ap.add_argument("--final", nargs="*", metavar="CECHA", help="np. gap_REB line_AST")
    ap.add_argument("--max-days", type=int)
    a = ap.parse_args()
    if a.cmd == "fetch":
        logger.info("Pobrano %d dni", fetch(max_days=a.max_days))
    else:
        run(final=a.final is not None, only=a.final)
