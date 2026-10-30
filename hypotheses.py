# nba_betting/hypotheses.py
"""
Krok 5: revenge i nemesis jako TESTY HIPOTEZ, zanim staną się cechami.

Pytanie: czy w takich meczach zawodnik systematycznie odbiega od prognozy modelu?
  reszta = PTS − μ, gdzie μ to predykcja POZA PRÓBĄ (PointsModel na BASE_FEATURE_COLS,
  trenowany przed startem każdego sezonu testowego).
Jeśli średnia reszta w meczach z flagą ≠ 0 (CI z bootstrapu po zawodnikach), model
nie wychwytuje efektu i cecha ma sens. Jeśli ≈ 0 — model już to wie albo efektu nie ma.

Hipotezy:
  revenge        — mecz przeciwko drużynie, w której zawodnik wcześniej grał (od 2021-22;
                   wcześniejszych klubów nie znamy). Warianty: pierwszy mecz po odejściu,
                   mecz w ciągu 365 dni od ostatniego meczu dla tej drużyny.
  nemesis (drużynowy) — reszty z poprzednich meczów z tym rywalem (ściągnięte do 0, K_SHRINK)
                   vs reszta dziś. Nachylenie > 0 = „zawodnik ma rywali, na których mu nie idzie”
                   i model tego nie widzi (base ma już pts_avg_vs_opp — średnie punkty vs rywal).
                   Flaga is_nemesis: rywal z najniższym takim wynikiem (≥ NEMESIS_MIN_GAMES meczów).
  Nemesis jako konkretny OBROŃCA wymaga danych o pojedynkach (BoxScoreMatchups, tysiące zapytań) —
  poza zakresem tego kroku.

Użycie:
  python hypotheses.py            # → reports/hypotheses_<data>.json
"""
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from history import BASE_FEATURE_COLS, load_dataset
from points_model import train_points_model

logger = logging.getLogger(__name__)

TEST_SEASONS      = ["2023-24", "2024-25", "2025-26", "2026-27"]
REVENGE_DAYS      = 365
K_SHRINK          = 5        # „pseudo-mecze” z resztą 0 przy ściąganiu wyniku vs rywal
NEMESIS_MIN_GAMES = 3
N_BOOT            = 2000
REPORT_DIR        = Path("reports")


# ─────────────────────────────────────────────────────────────────────────────
# Predykcje poza próbą
# ─────────────────────────────────────────────────────────────────────────────
def season_start(season: str) -> pd.Timestamp:
    return pd.Timestamp(f"{season[:4]}-10-01")


def out_of_sample_residuals(df: pd.DataFrame, seasons=TEST_SEASONS) -> pd.DataFrame:
    """Wiersze sezonów testowych (zagrane, MIN > 0) z kolumnami mu i resid = PTS − mu."""
    parts = []
    for s in seasons:
        test = df[(df["SEASON"] == s) & df["PTS"].notna() & (df["MIN"] > 0)]
        if test.empty:
            continue
        model = train_points_model(df, cutoff=season_start(s), features=BASE_FEATURE_COLS)
        parts.append(test.assign(mu=model.mu(test)))
    out = pd.concat(parts, ignore_index=True)
    out["resid"] = out["PTS"] - out["mu"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Flagi (tylko przeszłość)
# ─────────────────────────────────────────────────────────────────────────────
def revenge_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dla każdego meczu: czy rywal to była drużyna zawodnika (grał dla niej PRZED tym meczem
    i dziś gra w innej). Zwraca df z kolumnami is_revenge, revenge_first, revenge_days.
    """
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    is_rev = np.zeros(len(df), dtype=bool)
    first = np.zeros(len(df), dtype=bool)
    days_since = np.full(len(df), np.nan)
    team = df["TEAM_ID"].to_numpy()
    opp = df["OPP_ID"].to_numpy()
    d = df["GAME_DATE"].to_numpy()
    for idx in df.groupby("PLAYER_ID").indices.values():
        last_for: dict = {}        # drużyna → data ostatniego meczu zawodnika w jej barwach
        met_since: set = set()     # byłe drużyny, z którymi już grał po odejściu
        for i in idx:
            o, t = opp[i], team[i]
            if o in last_for and o != t:
                is_rev[i] = True
                days_since[i] = (d[i] - last_for[o]) / np.timedelta64(1, "D")
                first[i] = o not in met_since
                met_since.add(o)
            if t in met_since:      # wrócił do byłej drużyny — kolejne odejście liczy się od nowa
                met_since.discard(t)
            last_for[t] = d[i]
    return df.assign(is_revenge=is_rev, revenge_first=first, revenge_days=days_since)


def nemesis_scores(res: pd.DataFrame, k: float = K_SHRINK) -> pd.DataFrame:
    """
    nemesis_score = Σ reszt z POPRZEDNICH meczów vs ten rywal / (n + k)  (ujemny = zwykle gra
    poniżej prognozy z tym rywalem), nemesis_n = liczba tych meczów.
    is_nemesis = rywal z najniższym wynikiem spośród rywali z ≥ NEMESIS_MIN_GAMES meczami,
    wyznaczany na podstawie stanu sprzed meczu.
    """
    res = res.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = res.groupby(["PLAYER_ID", "OPP_ID"], group_keys=False)
    prev_sum = g["resid"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prev_n = g.cumcount()
    res["nemesis_score"] = prev_sum / (prev_n + k)
    res["nemesis_n"] = prev_n

    is_nem = np.zeros(len(res), dtype=bool)
    pid, opp = res["PLAYER_ID"].to_numpy(), res["OPP_ID"].to_numpy()
    r = res["resid"].to_numpy()
    for idx in res.groupby("PLAYER_ID").indices.values():
        sums: dict = {}
        cnts: dict = {}
        for i in idx:
            o = opp[i]
            eligible = {t: sums[t] / (cnts[t] + k) for t in cnts if cnts[t] >= NEMESIS_MIN_GAMES}
            if o in eligible and eligible[o] == min(eligible.values()) and eligible[o] < 0:
                is_nem[i] = True
            sums[o] = sums.get(o, 0.0) + r[i]
            cnts[o] = cnts.get(o, 0) + 1
    res["is_nemesis"] = is_nem
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Statystyka: bootstrap po zawodnikach (mecze tego samego zawodnika nie są niezależne)
# ─────────────────────────────────────────────────────────────────────────────
def flag_effect(res: pd.DataFrame, flag: str, value: str = "resid", rng=None) -> dict:
    """Średnia `value` w meczach z flagą minus bez flagi, CI95 z bootstrapu po zawodnikach."""
    rng = rng or np.random.default_rng(0)
    f = res[flag].to_numpy(bool)
    y = res[value].to_numpy(float)
    players = res["PLAYER_ID"].to_numpy()
    up, inv = np.unique(players, return_inverse=True)
    sf = np.bincount(inv, y * f, len(up)); cf = np.bincount(inv, f, len(up))
    su = np.bincount(inv, y * ~f, len(up)); cu = np.bincount(inv, ~f, len(up))
    diffs = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))
        if (w * cf).sum() == 0:
            continue
        diffs.append((w @ sf) / (w @ cf) - (w @ su) / (w @ cu))
    n = int(f.sum())
    return {
        "n_games": n, "n_players": int((cf > 0).sum()),
        "mean_flag": round(float(y[f].mean()), 3) if n else None,
        "mean_other": round(float(y[~f].mean()), 3),
        "diff": round(float(y[f].mean() - y[~f].mean()), 3) if n else None,
        "diff_ci95": [round(float(np.percentile(diffs, q)), 3) for q in (2.5, 97.5)] if diffs else None,
    }


def slope_effect(res: pd.DataFrame, x: str, y: str = "resid", rng=None) -> dict:
    """Nachylenie y ~ x (MNK), CI95 z bootstrapu po zawodnikach."""
    rng = rng or np.random.default_rng(0)
    xv, yv = res[x].to_numpy(float), res[y].to_numpy(float)
    up, inv = np.unique(res["PLAYER_ID"].to_numpy(), return_inverse=True)
    stats = [np.bincount(inv, v, len(up)) for v in (np.ones_like(xv), xv, yv, xv * yv, xv * xv)]

    def slope(w):
        n, sx, sy, sxy, sxx = (w @ s for s in stats)
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)

    boots = [slope(np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))) for _ in range(N_BOOT)]
    return {
        "n_games": len(res), "slope": round(float(slope(np.ones(len(up)))), 4),
        "slope_ci95": [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rynek: czy na liniach bukmacherów over wpada częściej, niż zakłada kurs?
# ─────────────────────────────────────────────────────────────────────────────
def market_check(res: pd.DataFrame, flag: str) -> dict | None:
    preds = sorted(REPORT_DIR.glob("eval_*_predictions.csv"))
    if not preds:
        return None
    lines = pd.read_csv(preds[-1], usecols=["GAME_ID", "PLAYER_ID", "line", "p_market", "PTS", "bookmaker"])
    lines = lines.dropna(subset=["GAME_ID", "PLAYER_ID"])
    lines["GAME_ID"] = lines["GAME_ID"].astype(str).str.zfill(10)
    lines["PLAYER_ID"] = lines["PLAYER_ID"].astype("int64")
    m = lines.merge(res[["GAME_ID", "PLAYER_ID", flag]], on=["GAME_ID", "PLAYER_ID"])
    m = m[m[flag]]
    if m.empty:
        return {"n_lines": 0}
    over = (m["PTS"] > m["line"]).astype(float)
    return {"n_lines": len(m), "over_rate": round(float(over.mean()), 3),
            "p_market_mean": round(float(m["p_market"].mean()), 3)}


def run() -> dict:
    df = load_dataset()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    flags = revenge_flags(df)[["GAME_ID", "PLAYER_ID", "is_revenge", "revenge_first", "revenge_days"]]
    res = out_of_sample_residuals(df).merge(flags, on=["GAME_ID", "PLAYER_ID"], how="left")
    res["revenge_recent"] = res["is_revenge"] & (res["revenge_days"] <= REVENGE_DAYS)
    res = nemesis_scores(res)
    with_hist = res[res["nemesis_n"] >= NEMESIS_MIN_GAMES]

    report = {
        "generated": date.today().isoformat(),
        "test_seasons": TEST_SEASONS,
        "n_games": len(res),
        "resid_mean": round(float(res["resid"].mean()), 3),
        "revenge": {k: flag_effect(res, k) for k in ("is_revenge", "revenge_first", "revenge_recent")},
        "revenge_market": {k: market_check(res, k) for k in ("is_revenge", "revenge_first")},
        "nemesis": {
            "slope_all": slope_effect(res[res["nemesis_n"] > 0], "nemesis_score"),
            f"slope_n>={NEMESIS_MIN_GAMES}": slope_effect(with_hist, "nemesis_score"),
            "is_nemesis": flag_effect(res, "is_nemesis"),
        },
        "nemesis_market": market_check(res, "is_nemesis"),
        "note": ("reszta = PTS − μ poza próbą; diff = średnia reszta z flagą − bez; "
                 "CI95 z bootstrapu po zawodnikach. Wiele porównań: pojedyncze CI bez zera "
                 "traktować jako wskazówkę, nie dowód."),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"hypotheses_{report['generated']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Raport: %s", path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(run(), indent=2, ensure_ascii=False))
