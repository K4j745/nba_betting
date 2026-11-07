# nba_betting/early_entry.py
"""
Test wejścia wcześniej: czy ceny Kalshi na OFFSETS_H godzin przed meczem da się pobić?

Cena z godziny przed meczem zawiera już raport kontuzji (market_model.py: brak poprawy).
Wcześniej rynek wie mniej, ale i my wiemy mniej — dlatego absencje liczymy z raportu
z TEJ SAMEJ chwili co cena. Dla każdej statystyki i każdego wyprzedzenia:

  1. rynek wcześnie vs rynek 1 h przed meczem: ile informacji dochodzi po drodze,
  2. korekta wczesnej ceny (market_model) walk-forward: Δ log-loss vs wczesna cena, zakłady,
  3. CLV: kupno po wczesnym ask/bid przy EV > EV_MIN vs środek ceny 1 h przed meczem.
     Mniej szumu niż wynik meczu: jeśli sygnał jest prawdziwy, rynek przesuwa się w naszą stronę.

Użycie:
  python early_entry.py --fetch     # ceny → data/raw/kalshi_early.csv.gz, raporty → injury_reports
  python early_entry.py             # → reports/early_entry_<data>.json
"""
import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import injuries
import kalshi
import market_model as mm
from evaluate_stats import _cluster_ci, _logloss, simulate
from history import RAW_DIR, norm_name

logger = logging.getLogger(__name__)

OFFSETS_H  = (10, 4)
EARLY_PATH = RAW_DIR / "kalshi_early.csv.gz"
EV_MIN     = 0.05
SAVE_EVERY = 1000
COLS       = ["ticker", "offset_h", "bid", "ask", "p_mid", "snap_ts"]


# ─────────────────────────────────────────────────────────────────────────────
# Pobieranie: jedno zapytanie na rynek (świece z całego dnia przed meczem)
# ─────────────────────────────────────────────────────────────────────────────
def _pick(ticker: str, candles: list[dict], tip: pd.Timestamp) -> list[dict]:
    """Dla każdego wyprzedzenia ostatnia świeca kończąca się ≤ start − h."""
    rows = []
    for h in OFFSETS_H:
        target = int((tip - pd.Timedelta(hours=h)).timestamp())
        cs = [c for c in candles if c["end_period_ts"] <= target]
        if cs:
            rows.append({**kalshi._candle_row(ticker, cs[-1]), "offset_h": h})
    return rows


def fetch_prices(lines: pd.DataFrame) -> pd.DataFrame:
    """lines: ticker, tip. Wznawialne: pomija tickery już obecne w EARLY_PATH."""
    old = load_early()
    done = set(old["ticker"]) | set(_empty_tickers())
    todo = lines[~lines["ticker"].isin(done)].drop_duplicates("ticker")
    cutoff = pd.Timestamp(kalshi._get("/historical/cutoff")["market_settled_ts"])
    logger.info("Rynków do pobrania: %d (gotowe: %d), archiwum do %s", len(todo), len(done), cutoff.date())
    rows, empty = [], []

    def save(hist):
        nonlocal rows, empty
        hist = pd.concat([hist, pd.DataFrame(rows, columns=COLS)], ignore_index=True) if rows else hist
        EARLY_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(EARLY_PATH, index=False)
        with open(_empty_path(), "a") as f:
            f.writelines(t + "\n" for t in empty)
        rows, empty = [], []
        return hist

    hist = old
    archived = todo[todo["tip"] < cutoff]
    for i, r in enumerate(archived.itertuples(), 1):
        start = int((r.tip - pd.Timedelta(hours=36)).timestamp())
        d = kalshi._get(f"/historical/markets/{r.ticker}/candlesticks", start_ts=start,
                        end_ts=int(r.tip.timestamp()), period_interval=60)
        got = _pick(r.ticker, d.get("candlesticks", []), r.tip)
        rows += got
        if not got:
            empty.append(r.ticker)
        if i % SAVE_EVERY == 0:
            hist = save(hist)
            logger.info("  archiwum: %d/%d", i, len(archived))
    for tip, g in todo[todo["tip"] >= cutoff].groupby("tip"):
        tickers = g["ticker"].tolist()
        for i in range(0, len(tickers), kalshi.BATCH):
            d = kalshi._get("/markets/candlesticks", market_tickers=",".join(tickers[i:i + kalshi.BATCH]),
                            start_ts=int((tip - pd.Timedelta(hours=36)).timestamp()),
                            end_ts=int(tip.timestamp()), period_interval=60)
            for mk in d.get("markets", []):
                rows += _pick(mk["market_ticker"], mk.get("candlesticks", []), tip)
    return save(hist)


def _empty_path() -> Path:
    return RAW_DIR / "kalshi_early_empty.txt"      # rynki bez świec (otwarte później) — nie pytać ponownie


def _empty_tickers() -> list[str]:
    return _empty_path().read_text().split() if _empty_path().exists() else []


def load_early() -> pd.DataFrame:
    if not EARLY_PATH.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(EARLY_PATH)


def load_lines() -> pd.DataFrame:
    """Predykcje z najnowszego eval_stats + ticker linii głównej (stat, dzień, zawodnik, próg)."""
    pred = pd.read_csv(mm.latest_predictions(), parse_dates=["game_date"], dtype={"GAME_ID": str})
    pred["tip"] = pd.to_datetime(pred["tip"], utc=True, format="ISO8601")
    h = kalshi.load_history()[["stat", "game_date", "name_key", "line", "ticker"]]
    pred["name_key"] = pred["player_name"].map(norm_name)
    return pred.merge(h.drop_duplicates(["stat", "game_date", "name_key", "line"]),
                      on=["stat", "game_date", "name_key", "line"], how="left")


def fetch() -> None:
    lines = load_lines()
    logger.info("Linii: %d, z tickerem: %d", len(lines), lines["ticker"].notna().sum())
    fetch_prices(lines.dropna(subset=["ticker", "tip"]))
    tips = lines.dropna(subset=["tip"])["tip"].drop_duplicates()
    injuries.report_history([(t - pd.Timedelta(hours=h)).to_pydatetime() for t in tips for h in OFFSETS_H])


# ─────────────────────────────────────────────────────────────────────────────
# Ewaluacja
# ─────────────────────────────────────────────────────────────────────────────
def clv(df: pd.DataFrame, p_col: str, rng) -> dict:
    """Kupno po wczesnej cenie przy EV > EV_MIN; zysk w centach względem środka ceny 1 h przed meczem."""
    p, ask, bid, late = (df[c].to_numpy() for c in (p_col, "ask", "bid", "p_late"))
    ev_y, ev_n = p / ask - 1, (1 - p) / (1 - bid) - 1
    yes = ev_y >= ev_n
    sel = np.where(yes, ev_y, ev_n) > EV_MIN
    gain = np.where(yes, late - ask, bid - late)
    out = {"bets": int(sel.sum())}
    if sel.sum() >= 2:
        out.update(mean_cents=round(float(gain[sel].mean() * 100), 2),
                   ci95=[round(x * 100, 2) for x in _cluster_ci(gain[sel], df["GAME_ID"].astype(str).to_numpy()[sel], rng)],
                   share_positive=round(float((gain[sel] > 0).mean()), 3))
    return out


def evaluate_offset(lines: pd.DataFrame, early: pd.DataFrame, h: int, rng) -> dict:
    e = early[early["offset_h"] == h].dropna(subset=["p_mid"])
    df = lines.rename(columns={"p_mid": "p_late"}).drop(columns=["bid", "ask"]).merge(
        e[["ticker", "bid", "ask", "p_mid"]], on="ticker")
    df = mm.absence_features(df, hours_before=h)
    df = df[df["has_report"]]
    res = {}
    for stat, d in df.groupby("stat"):
        d = mm.add_terms(d.dropna(subset=["p_mid", "p_base", "y_over"])).reset_index(drop=True)
        y = d["y_over"].to_numpy()
        games = d["GAME_ID"].astype(str).to_numpy()
        info = _logloss(d["p_mid"].to_numpy(), y) - _logloss(d["p_late"].to_numpy(), y)
        r = {"n": int(len(d)), "spread_mean": round(float((d["ask"] - d["bid"]).mean()), 3),
             "logloss_early": round(float(_logloss(d["p_mid"].to_numpy(), y).mean()), 4),
             "logloss_late": round(float(_logloss(d["p_late"].to_numpy(), y).mean()), 4),
             "early_minus_late": round(float(info.mean()), 4), "early_minus_late_ci95": _cluster_ci(info, games, rng),
             "price_move_abs": round(float((d["p_late"] - d["p_mid"]).abs().mean()), 4),
             "variants": {}}
        for name in ("market+model", "market+model+report"):
            p, _ = mm.walk_forward(d, mm.VARIANTS[name])
            ok = ~np.isnan(p)
            s = d[ok].assign(p_corr=p[ok])
            diff = _logloss(s["p_corr"].to_numpy(), s["y_over"].to_numpy()) - _logloss(s["p_mid"].to_numpy(), s["y_over"].to_numpy())
            r["variants"][name] = {
                "n": int(ok.sum()), "logloss_diff": round(float(diff.mean()), 4),
                "logloss_diff_ci95": _cluster_ci(diff, s["GAME_ID"].astype(str).to_numpy(), rng),
                "bets": simulate(s, "p_corr", rng), "clv": clv(s, "p_corr", rng),
                "coefs_full_sample": mm.full_fit_ci(d, mm.VARIANTS[name], rng, n_boot=200),
            }
        r["clv_raw_model"] = clv(d, "p_base", rng)         # sam model, bez ściągania do rynku
        res[stat] = r
        v = r["variants"]["market+model+report"]
        logger.info("%s −%d h: n=%d Δ %+.4f %s CLV %s", stat, h, r["n"], v["logloss_diff"],
                    v["logloss_diff_ci95"], v["clv"])
    return res


def run() -> dict:
    rng = np.random.default_rng(0)
    lines, early = load_lines().dropna(subset=["ticker", "tip"]), load_early()
    report = {"generated": date.today().isoformat(), "offsets_h": list(OFFSETS_H),
              "results": {f"{h}h": evaluate_offset(lines, early, h, rng) for h in OFFSETS_H}}
    mm.REPORT_DIR.mkdir(exist_ok=True)
    out = mm.REPORT_DIR / f"early_entry_{report['generated']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", out)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    fetch() if args.fetch else run()
