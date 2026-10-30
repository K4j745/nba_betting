# nba_betting/main.py
"""
NBA Betting Pipeline — Orchestrator

Harmonogram (domyślny):
  09:00  → pipeline dzienny (propsy → features → model → kupon)
  10:00  → weryfikacja wczorajszych wyników + raport + retrain jeśli należy
  09:30  → backup bazy danych

Uruchomienie:
  python main.py                    # tryb scheduler (działa non-stop)
  python main.py --run-now          # odpal pipeline od razu (debug)
  python main.py --verify-only      # tylko weryfikacja i raport
  python main.py --date 2026-03-10  # pipeline dla konkretnej daty
"""

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import schedule

from config import (
    DB_PATH, MODEL_DIR, ODDS_API_KEY,
    STAKE_PER_PICK,
    ODDS_MIN, ODDS_MAX,
)
from odds_fetcher import OddsFetcher
from tracker import ResultTracker
import live

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "nba_betting.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Powody pustego kuponu (trafiają do dashboard_data.json jako `no_games_reason`)
# ─────────────────────────────────────────────────────────────────────────────

REASON_OK            = "ok"
REASON_NO_PLAYERS    = "player_lookup_failed"
REASON_NO_MODEL      = "model_missing"
REASON_NO_PROPS      = "no_props_today"
REASON_NO_FEATURES   = "no_features_built"
REASON_NO_PICKS      = "no_picks_passed_filters"
REASON_NO_DATA       = "nba_data_missing"
REASON_DATA_STALE    = "nba_data_stale"

# Czytelne komunikaty pokazywane na dashboardzie (pole `status_message`).
_REASON_MESSAGES: dict[str, str] = {
    REASON_OK:          "",
    REASON_NO_PLAYERS:  "Nie udało się pobrać listy zawodników NBA (problem z nba_api).",
    REASON_NO_MODEL:    "Brak wytrenowanego modelu — uruchom `python train.py`.",
    REASON_NO_PROPS:    "Brak meczów / propsów na dziś (API nie zwróciło zakładów na punkty zawodników).",
    REASON_NO_FEATURES: "Są propsy, ale nie udało się zbudować feature vectorów (prawdopodobnie nba_api zablokowane na runnerze CI).",
    REASON_NO_PICKS:    "Są mecze, ale żaden pick nie przeszedł filtrów (kurs 1.40–1.90 + pewność ≥ 60%).",
    REASON_NO_DATA:     "Brak logów NBA (branch `data`) — uruchom synchronizację na laptopie.",
    REASON_DATA_STALE:  "Logi NBA są nieaktualne (laptop nie zsynchronizował danych) — kursy zapisane, kupon pominięty.",
}


def _nba_today() -> str:
    """
    Zwraca "dzień meczowy" NBA jako ISO date w strefie US/Eastern.

    BUG (przed poprawką): używaliśmy `date.today()`, które na runnerze GitHub
    Actions (UTC) zwraca już JUTRZEJSZĄ datę dla wieczornych meczów NBA. Cron
    odpalał się 18:00 UTC → kupon zapisywany pod datą ET, a eksport czytał kupon
    pod datą UTC. Te dwie daty się rozjeżdżały po północy UTC i `today_coupon`
    zawsze wychodził null. FIX: liczymy datę meczową w strefie America/New_York,
    tak samo przy zapisie i przy odczycie kuponu.
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        # Fallback gdy brak bazy tz (np. Windows bez tzdata): ET ≈ UTC-4/-5.
        return (datetime.now(timezone.utc) - timedelta(hours=4)).date().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Helpery — mapowanie zawodników i drużyn
# ─────────────────────────────────────────────────────────────────────────────

def _build_player_lookup() -> dict[str, int]:
    """Zwraca {full_name: player_id} dla wszystkich graczy NBA."""
    try:
        from nba_api.stats.static import players as nba_players
        return {p["full_name"]: p["id"] for p in nba_players.get_players()}
    except Exception as e:
        logger.error("Błąd przy budowaniu player_lookup: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Kroki pipeline
# ─────────────────────────────────────────────────────────────────────────────

def step_fetch_props(
    odds_fetcher: OddsFetcher,
    player_lookup: dict[str, int],
) -> list[dict]:
    """
    Krok 1: Pobierz propsy z bukmacherów.
    Filtruje tylko graczy których znamy (mamy player_id).
    Zwraca listę propsów z player_id.
    """
    logger.info("── Krok 1: Pobieranie propsów ──────────────────────────────")
    try:
        raw_props = odds_fetcher.get_player_props(use_scraping=False)
    except Exception as e:
        logger.error("Błąd przy pobieraniu propsów: %s", e)
        return []

    resolved = []
    skipped  = 0
    for prop in raw_props:
        pid = player_lookup.get(prop["player_name"])
        if pid is None:
            # Spróbuj przez matcher (fuzzy)
            try:
                matched = odds_fetcher.name_matcher.match(prop["player_name"])
                pid     = player_lookup.get(matched) if matched else None
            except Exception:
                pass

        if pid is None:
            skipped += 1
            continue

        prop["player_id"] = pid
        resolved.append(prop)

    logger.info(
        "Propsy: %d pobrane, %d dopasowane, %d pominięte (brak player_id)",
        len(raw_props), len(resolved), skipped,
    )
    return resolved

def step_backup_db(db_path: str = DB_PATH):
    """Krok opcjonalny: backup bazy danych."""
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    dst = backup_dir / f"nba_betting_{date.today().isoformat()}.db"
    try:
        shutil.copy2(db_path, dst)
        logger.info("Backup DB: %s", dst)
    except Exception as e:
        logger.error("Błąd backupu DB: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Główny pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_pipeline(game_date: str | None = None):
    """
    Główny pipeline dzienny:
      1. Pobierz propsy (Odds API) i zapisz do odds_history
      2. live.run_coupon: cechy z logów w data/raw → PointsModel → kupon (paper trading)
    """
    # Dzień meczowy NBA liczymy w strefie ET (patrz _nba_today) — spójnie
    # przy zapisie kuponu i przy odczycie w export_dashboard_data.
    if game_date is None:
        game_date = _nba_today()

    logger.info("=" * 60)
    logger.info("START PIPELINE DZIENNEGO — %s", game_date)
    logger.info("=" * 60)

    player_lookup = _build_player_lookup()

    if not player_lookup:
        logger.error("Brak player_lookup — przerywam pipeline")
        export_dashboard_data(db_path=DB_PATH, reason=REASON_NO_PLAYERS)
        return

    # Pobierz listę graczy do OddsFetcher
    player_names = list(player_lookup.keys())
    odds_fetcher  = OddsFetcher(
        #api_key=ODDS_API_KEY,
        nba_player_names=player_names,
        db_path=DB_PATH,
    )

    # Kroki pipeline — z diagnostyką ilościową (ile propsów → features → picków).
    # Propsy pobieramy PRZED sprawdzeniem modelu: zapis do odds_history to
    # zbiór do ewaluacji, który ma rosnąć także wtedy, gdy kuponu nie ma.
    props = step_fetch_props(odds_fetcher, player_lookup)
    logger.info("DIAG: propsów z API = %d", len(props))
    # Spread/total meczów — na razie tylko zbieramy historię (krok 4c), model ich nie używa
    try:
        logger.info("DIAG: linii meczowych (spread/total) = %d", odds_fetcher.fetch_game_lines())
    except Exception as e:
        logger.error("Błąd przy pobieraniu linii meczowych: %s", e)
    if not props:
        logger.warning("Brak propsów — pipeline zakończony bez kuponu")
        export_dashboard_data(db_path=DB_PATH, reason=REASON_NO_PROPS)
        return

    # Kupon z PointsModel na logach z data/raw (branch `data`, synchronizowany z laptopa) —
    # stats.nba.com nie odpowiada runnerom CI, więc nie pytamy go wcale.
    try:
        coupon = live.run_coupon(props, game_date, db_path=DB_PATH)
    except FileNotFoundError as e:
        logger.error("%s", e)
        export_dashboard_data(db_path=DB_PATH, reason=REASON_NO_DATA)
        return

    if coupon.get("error") == "stale":
        export_dashboard_data(db_path=DB_PATH, reason=REASON_DATA_STALE)
        return
    if coupon.get("error") == "no_features":
        export_dashboard_data(db_path=DB_PATH, reason=REASON_NO_FEATURES)
        return

    if coupon["picks"]:
        _print_coupon(coupon)
    logger.info(
        "DIAG: predykcji = %d, picków (EV ≥ %.0f%%, kurs %.2f–%.2f) = %d",
        coupon["n_predictions"], live.EV_MIN * 100, ODDS_MIN, ODDS_MAX, coupon["total_picks"],
    )
    reason = REASON_OK if coupon["total_picks"] > 0 else REASON_NO_PICKS

    logger.info(
        "KONIEC PIPELINE — %d picków wygenerowanych dla %s",
        coupon["total_picks"], game_date,
    )

    export_dashboard_data(db_path=DB_PATH, reason=reason)


def run_verify_and_report():
    """
    Pipeline weryfikacyjny (następnego dnia rano):
      4. Weryfikuj wczoraj (punkty z logów w data/raw)
      5. Raport 30-dniowy
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    logger.info("=" * 60)
    logger.info("START WERYFIKACJI — weryfikuję %s", yesterday)
    logger.info("=" * 60)

    tracker = ResultTracker(db_path=DB_PATH)
    try:
        pts_fn = live.actual_points_fn()
    except Exception as e:
        logger.error("Brak logów do weryfikacji (data/raw): %s", e)
        pts_fn = None

    if pts_fn is not None:
        results = tracker.verify_yesterday(yesterday, actual_pts_fn=pts_fn)
        if not results.empty:
            logger.info(
                "Wczoraj: %dW / %dL / %dP | P&L=%+.2f",
                (results["result"] == "WIN").sum(), (results["result"] == "LOSS").sum(),
                (results["result"] == "PUSH").sum(), results["simulated_profit"].sum(),
            )
        tracker.generate_report(last_n_days=30)
    # Retrain nie jest potrzebny: PointsModel trenuje się w każdym runie pipeline'u.
    step_backup_db()
    export_dashboard_data(db_path=DB_PATH)

    logger.info("KONIEC WERYFIKACJI")


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler():
    """
    Uruchamia scheduler — działa non-stop.

    Harmonogram:
      09:00  → pipeline dzienny (propsy → kupon)
      10:00  → weryfikacja + raport + retrain
      09:30  → backup DB
    """
    logger.info("Scheduler uruchomiony")
    logger.info("  09:00 → pipeline dzienny")
    logger.info("  10:00 → weryfikacja i raport")
    logger.info("  09:30 → backup DB")
    logger.info("Ctrl+C aby zatrzymać")

    schedule.every().day.at("09:00").do(run_daily_pipeline)
    schedule.every().day.at("10:00").do(run_verify_and_report)
    schedule.every().day.at("09:30").do(step_backup_db)

    while True:
        schedule.run_pending()
        time.sleep(60)  # sprawdzaj co minutę


# ─────────────────────────────────────────────────────────────────────────────
# Print helper
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard export
# ─────────────────────────────────────────────────────────────────────────────

def _latest_research_summary(report_dir: str = "reports") -> dict | None:
    """Skrót najnowszego raportu evaluate.py (model vs rynek) dla dashboardu."""
    import json as _json
    reports = sorted(p for p in Path(report_dir).glob("eval_*.json"))
    if not reports:
        return None
    r = _json.loads(reports[-1].read_text(encoding="utf-8"))
    rows = []
    for book, sets in r.get("books", {}).items():
        for name, v in sets.items():
            sc = v["scores"]
            bet5 = next((b for b in v["bets"] if b["ev_threshold"] == 0.05), {})
            rows.append({
                "book": book, "features": name, "n": sc["n"],
                "logloss_model": sc["logloss_model"], "logloss_market": sc["logloss_market"],
                "logloss_diff_ci95": sc["logloss_diff_ci95"],
                "mae_model": sc["mae_points"], "mae_line": sc["mae_line"],
                "bets_ev5": bet5.get("bets", 0), "roi_ev5": bet5.get("roi"),
                "roi_ev5_ci95": bet5.get("roi_ci95"),
            })
    return {
        "report": reports[-1].name,
        "generated": r.get("generated"),
        "game_dates": r.get("game_dates", []),
        "rows": rows,
    }


def export_dashboard_data(
    db_path: str = DB_PATH,
    output_path: str = "dashboard_data.json",
    reason: str = REASON_OK,
) -> dict:
    """
    Buduje dashboard_data.json z bazy SQLite.
    Wywoływane automatycznie przy KAŻDYM wyjściu z pipeline (również gdy nie ma
    meczów/propsów) oraz po weryfikacji. `reason` mówi dashboardowi, dlaczego
    `today_coupon` jest pusty (mapowane na czytelny `status_message`).
    """
    import json as _json
    import sqlite3 as _sqlite3
    from collections import defaultdict
    from datetime import datetime as _dt, date as _date, timedelta as _td
    import math as _math

    # Dzień meczowy w ET — ten sam co przy zapisie kuponu (patrz _nba_today).
    today     = _nba_today()
    since_30d = (_date.fromisoformat(today) - _td(days=30)).isoformat()
    since_90d = (_date.fromisoformat(today) - _td(days=90)).isoformat()

    try:
        import pandas as _pd
    except ImportError:
        logger.error("pandas niedostępne — pomijam eksport dashboardu")
        return {}

    def _safe(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and _math.isnan(v):
                return None
        except Exception:
            pass
        if hasattr(v, "item"):
            return v.item()
        return v

    conn = _sqlite3.connect(db_path)

    # ── 1. Dzisiejszy kupon ─────────────────────────────────────────
    row = conn.execute(
        "SELECT coupon_json FROM coupons WHERE date = ? ORDER BY created_at DESC LIMIT 1",
        (today,),
    ).fetchone()
    today_coupon = _json.loads(row[0]) if row else None

    has_picks = bool(today_coupon and today_coupon.get("picks"))
    if has_picks:
        for pick in today_coupon["picks"]:
            p = pick["p_over"] if pick["bet"] == "over" else (1.0 - pick["p_over"])
            pick["ev"] = round(p * pick["odds"] - 1.0, 4)
        today_coupon["picks"].sort(key=lambda x: x.get("ev", 0), reverse=True)
        reason = REASON_OK
    elif reason == REASON_OK:
        # Wywołano bez jawnego powodu (np. z weryfikacji), a kuponu brak —
        # rozróżnij "kupon istnieje ale pusty" od "w ogóle brak meczów".
        reason = REASON_NO_PICKS if today_coupon is not None else REASON_NO_PROPS

    # today_coupon eksportujemy tylko gdy ma realne picki — inaczej null + reason.
    if not has_picks:
        today_coupon = None

    # ── 2. KPI ──────────────────────────────────────────────────────
    df_all = _pd.read_sql_query(
        "SELECT result, simulated_profit FROM results WHERE result IN ('WIN','LOSS')",
        conn,
    )
    df_30d = _pd.read_sql_query(
        "SELECT result, simulated_profit FROM results "
        "WHERE result IN ('WIN','LOSS') AND game_date >= ?",
        conn, params=(since_30d,),
    )

    total     = len(df_all)
    wins      = int((df_all["result"] == "WIN").sum()) if not df_all.empty else 0
    win_rate  = round(wins / total, 4) if total > 0 else 0.0
    pnl_total = round(float(df_all["simulated_profit"].sum()), 2) if not df_all.empty else 0.0
    stake_30d = len(df_30d) * STAKE_PER_PICK
    roi_30d   = (
        round(float(df_30d["simulated_profit"].sum()) / stake_30d, 4)
        if stake_30d > 0 else 0.0
    )

    # ── 3. Bankroll curve (90d) ─────────────────────────────────────
    df_pnl = _pd.read_sql_query(
        "SELECT game_date, simulated_profit FROM results "
        "WHERE result IN ('WIN','LOSS') AND game_date >= ? ORDER BY game_date, id",
        conn, params=(since_90d,),
    )
    cum = 0.0
    bankroll_curve = []
    for _, r in df_pnl.iterrows():
        cum += float(r["simulated_profit"])
        bankroll_curve.append({"date": r["game_date"], "cumulative_pnl": round(cum, 2)})

    # ── 4. Win rate per drużyna ─────────────────────────────────────
    df_games = _pd.read_sql_query(
        "SELECT game, result, simulated_profit FROM results WHERE result IN ('WIN','LOSS')",
        conn,
    )
    team_agg: dict = defaultdict(lambda: {"wins": 0, "total": 0, "profit": 0.0})
    for _, r in df_games.iterrows():
        game   = str(r.get("game", ""))
        result = str(r.get("result", ""))
        profit = float(r.get("simulated_profit", 0))
        sep = " vs " if " vs " in game else (" VS " if " VS " in game else None)
        if sep is None:
            continue
        parts = [p.strip() for p in game.split(sep, 1) if p.strip()]
        share = profit / max(len(parts), 1)
        for team in parts:
            team_agg[team]["total"] += 1
            team_agg[team]["wins"]  += 1 if result == "WIN" else 0
            team_agg[team]["profit"] += share

    win_rate_by_team = sorted(
        [
            {
                "team":     team,
                "n":        s["total"],
                "win_rate": round(s["wins"] / s["total"], 4),
                "profit":   round(s["profit"], 2),
            }
            for team, s in team_agg.items()
            if s["total"] >= 3
        ],
        key=lambda x: x["win_rate"],
        reverse=True,
    )[:15]

    # ── 5. Kalibracja modelu ────────────────────────────────────────
    df_cal = _pd.read_sql_query(
        "SELECT p_over, bet, result FROM results "
        "WHERE result IN ('WIN','LOSS') AND p_over IS NOT NULL",
        conn,
    )
    calibration: dict = {"prob_pred": [], "prob_true": [], "n_samples": 0}
    if len(df_cal) >= 10:
        try:
            from sklearn.calibration import calibration_curve as _cc
            df_cal["y_true"] = (
                ((df_cal["bet"] == "over")  & (df_cal["result"] == "WIN")) |
                ((df_cal["bet"] == "under") & (df_cal["result"] == "LOSS"))
            ).astype(int)
            prob_true, prob_pred = _cc(
                df_cal["y_true"], df_cal["p_over"], n_bins=5, strategy="quantile",
            )
            calibration = {
                "prob_pred": [round(float(p), 3) for p in prob_pred],
                "prob_true": [round(float(p), 3) for p in prob_true],
                "n_samples": len(df_cal),
            }
        except Exception as exc:
            logger.warning("Kalibracja w eksporcie dashboardu nie powiodła się: %s", exc)

    # ── 6. Historia zakładów ────────────────────────────────────────
    df_hist = _pd.read_sql_query(
        "SELECT game_date, player_name, game, bet, line, odds, p_over, "
        "confidence, actual_pts, result, simulated_profit "
        "FROM results ORDER BY game_date DESC, id DESC LIMIT 200",
        conn,
    )
    bet_history = [
        {
            "date":       r["game_date"],
            "player":     r["player_name"],
            "game":       r["game"],
            "bet":        r["bet"],
            "line":       _safe(r["line"]),
            "odds":       _safe(r["odds"]),
            "p_over":     _safe(r["p_over"]),
            "confidence": int(r["confidence"]) if _pd.notna(r.get("confidence")) else None,
            "actual_pts": _safe(r["actual_pts"]),
            "result":     r["result"],
            "profit":     round(float(r["simulated_profit"]), 2),
        }
        for _, r in df_hist.iterrows()
    ]

    # ── 7. Statystyki modelu ────────────────────────────────────────
    mrow = conn.execute(
        "SELECT version, trained_at, regressor_mae, classifier_auc "
        "FROM model_history ORDER BY version DESC LIMIT 1"
    ).fetchone()
    # Pokazuj tylko model, którego pliki istnieją — w model_history zostaje
    # wpis o usuniętym modelu v1 trenowanym na danych syntetycznych.
    if mrow and not (Path(MODEL_DIR) / f"regressor_v{mrow[0]}.pkl").exists():
        mrow = None
    model_stats = {
        "version":    mrow[0] if mrow else None,
        "trained_at": mrow[1] if mrow else None,
        "cv_mae":     mrow[2] if mrow else None,
        "cv_auc":     mrow[3] if mrow else None,
    }

    conn.close()

    data = {
        "generated_at":    _dt.now().isoformat(),
        "game_date":       today,
        "today_coupon":    today_coupon,
        "no_games_reason": None if reason == REASON_OK else reason,
        "status_message":  _REASON_MESSAGES.get(reason, ""),
        "kpi": {
            "win_rate":    win_rate,
            "roi_30d":     roi_30d,
            "pnl_total":   pnl_total,
            "total_picks": total,
            "total_wins":  wins,
        },
        "bankroll_curve":   bankroll_curve,
        "win_rate_by_team": win_rate_by_team,
        "calibration":      calibration,
        "bet_history":      bet_history,
        "model_stats":      model_stats,
        "research":         _latest_research_summary(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Dashboard data -> %s", output_path)
    return data


def _print_coupon(coupon: dict):
    print("\n" + "=" * 60)
    print(f"KUPON NA DZIŚ: {coupon['date']}")
    print("=" * 60)
    for pick in coupon["picks"]:
        print(
            f"  [{pick['confidence']:3d}] {pick['player']:<22} "
            f"{pick['bet'].upper()} {pick['line']} @ {pick['odds']} | "
            f"pred={pick['pts_predicted']}pts | p_over={pick['p_over']}"
        )
        print(f"       → {pick['reasoning']}")
    print("-" * 60)
    print(
        f"  Łącznie: {coupon['total_picks']} picks | "
        f"Stake: {coupon['simulated_total_stake']} PLN"
    )
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="NBA Betting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python main.py                      # scheduler (non-stop)
  python main.py --run-now            # pipeline od razu (debug)
  python main.py --verify-only        # tylko weryfikacja
  python main.py --date 2026-03-10    # pipeline dla konkretnej daty
        """,
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Odpal pipeline natychmiast (bez schedulera)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Tylko weryfikacja i raport",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Data w formacie YYYY-MM-DD (domyślnie: dziś)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verify_only:
        run_verify_and_report()
    elif args.run_now:
        run_daily_pipeline(game_date=args.date)
    elif args.date:
        run_daily_pipeline(game_date=args.date)
    else:
        run_scheduler()
