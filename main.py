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
from datetime import date, timedelta
from pathlib import Path

import schedule

from config import (
    DB_PATH, MODEL_DIR, ODDS_API_KEY,
    CONFIDENCE_THRESHOLD, STAKE_PER_PICK,
    ODDS_MIN, ODDS_MAX,
)
from data_fetcher import NBADataFetcher
from features import build_feature_vector, MODEL_FEATURE_COLS
from model import PropPredictor
from odds_fetcher import OddsFetcher
from tracker import ResultTracker

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


def _build_team_lookup() -> dict[str, int]:
    """
    Zwraca mapowanie → team_id dla:
    - skrótów:      {"DAL": 1610612742}
    - pełnych nazw: {"Dallas Mavericks": 1610612742}
    - nicków:       {"Mavericks": 1610612742}
    - miast:        {"Dallas": 1610612742}
    """
    try:
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        lookup = {}
        for t in all_teams:
            tid = t["id"]
            for key in [
                t["abbreviation"],
                t["abbreviation"].upper(),
                t["full_name"],
                t["full_name"].upper(),
                t["nickname"],
                t["nickname"].upper(),
                t["city"],
                t["city"].upper(),
            ]:
                lookup[key] = tid
        logger.info(
            "team_lookup: %d wpisów dla %d drużyn", len(lookup), len(all_teams)
        )
        return lookup
    except Exception as e:
        logger.error("Błąd przy budowaniu team_lookup: %s", e)
        return {}


def _resolve_opponent_team_id(
    game: str,
    home_team: str,
    is_home: int,
    team_lookup: dict[str, int],
) -> int | None:
    """
    Obsługuje formaty Odds API:
      game      = "San Antonio Spurs vs Philadelphia 76ers"
      home_team = "San Antonio Spurs"
    """
    try:
        sep = " VS " if " VS " in game.upper() else " @ "
        parts = game.upper().split(sep) if sep in game.upper() else game.split(" vs ")
        if len(parts) != 2:
            return None

        part_home = parts[0].strip()
        part_away = parts[1].strip()

        # Dopasuj home_team do właściwej strony
        if home_team.upper() in part_home.upper():
            opp_name = parts[1].strip()   # zachowaj oryginalne wielkości
        else:
            opp_name = parts[0].strip()

        # Szukaj w lookup (pełna nazwa, skrót, miasto)
        result = (
            team_lookup.get(opp_name) or
            team_lookup.get(opp_name.upper()) or
            team_lookup.get(opp_name.split()[-1])    # ostatni token = nickname
        )

        if result is None:
            logger.warning("Nie znaleziono team_id dla: '%s' (game='%s')", opp_name, game)

        return result

    except Exception as e:
        logger.warning("_resolve_opponent_team_id error: %s | game='%s'", e, game)
        return None


def _guess_position(player_id: int, fetcher: NBADataFetcher) -> str:
    """
    Próbuje zgadnąć pozycję gracza z ostatnich statystyk.
    Fallback: 'PG'
    """
    try:
        recent = fetcher.get_player_recent_stats(player_id, last_n=1)
        if not recent.empty and "POSITION" in recent.columns:
            return str(recent.iloc[0]["POSITION"])
    except Exception:
        pass
    return "PG"


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

def _get_player_team(player_id: int) -> str | None:
    """Zwraca nazwę drużyny gracza z nba_api."""
    try:
        from nba_api.stats.static import players as nba_players
        from nba_api.stats.endpoints import commonplayerinfo
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df   = info.get_data_frames()[0]
        return str(df["TEAM_NAME"].iloc[0]) if not df.empty else None
    except Exception:
        return None

def step_build_features(
    props: list[dict],
    fetcher: NBADataFetcher,
    team_lookup: dict[str, int],
) -> list[dict]:
    """
    Krok 2: Buduj feature vectory dla każdego propa.
    Zwraca listę propsów wzbogaconych o 'features_df'.
    """
    logger.info("── Krok 2: Budowanie feature vectorów ──────────────────────")
    enriched = []
    errors   = 0

    for prop in props:
        try:
            player_id  = prop["player_id"]
            home_team  = prop.get("home_team", "")
            away_team  = prop.get("away_team", "")
            is_home    = 0  # TODO: poprawić gdy będzie cache drużyn

            # Rywal = away_team (bo is_home=0, gracz jest gościem)
            # Dla połowy graczy będzie odwrotnie, ale jeden feature z 22 — akceptowalne
            opp_name    = home_team  # gracz away → rywal to home
            opp_team_id = (
                team_lookup.get(opp_name) or
                team_lookup.get(opp_name.upper()) or
                team_lookup.get(opp_name.split()[-1])  # ostatni token = nickname
            )

            if opp_team_id is None:
                logger.warning(
                    "Pominięto %s — brak team_id dla '%s' (home='%s' away='%s')",
                    prop["player_name"], opp_name, home_team, away_team,
                )
                errors += 1
                continue

            position = _guess_position(player_id, fetcher)

            fv = build_feature_vector(
                player_id=player_id,
                opponent_team_id=opp_team_id,
                line=float(prop["line"]),
                over_odds=float(prop["over_odds"]),
                under_odds=float(prop["under_odds"]),
                is_home=is_home,
                player_position=position,
                fetcher=fetcher,
            )

            enriched.append({**prop, "features_df": fv, "player_position": position})

        except Exception as e:
            logger.warning(
                "Błąd budowania features dla %s: %s",
                prop.get("player_name", "?"), e,
            )
            errors += 1

    logger.info("Features: %d zbudowane, %d błędów", len(enriched), errors)
    return enriched


def step_generate_coupon(
    enriched_props: list[dict],
    predictor: PropPredictor,
    game_date: str,
) -> dict:
    """
    Krok 3: Predykcja + generowanie kuponu.
    """
    logger.info("── Krok 3: Generowanie kuponu ──────────────────────────────")

    coupon = predictor.generate_coupon(
        enriched_props,
        game_date=game_date,
        stake=STAKE_PER_PICK,
        min_confidence=CONFIDENCE_THRESHOLD,
    )

    if not coupon["picks"]:
        logger.warning("Kupon pusty — żaden prop nie spełnił kryteriów filtrowania")
    else:
        _print_coupon(coupon)

    return coupon


def step_verify_and_report(
    tracker: ResultTracker,
    predictor: PropPredictor,
    yesterday: str,
) -> dict:
    """
    Krok 4: Weryfikacja wczorajszych wyników + raport + retrain.
    """
    logger.info("── Krok 4: Weryfikacja i raport ────────────────────────────")

    # Weryfikacja
    results = tracker.verify_yesterday(yesterday)
    if results.empty:
        logger.warning("Brak wyników do weryfikacji dla %s", yesterday)
    else:
        wins   = (results["result"] == "WIN").sum()
        losses = (results["result"] == "LOSS").sum()
        pushes = (results["result"] == "PUSH").sum()
        profit = results["simulated_profit"].sum()
        logger.info(
            "Wczoraj: %dW / %dL / %dP | P&L=%+.2f PLN",
            wins, losses, pushes, profit,
        )

    # Raport 30-dniowy
    report = tracker.generate_report(last_n_days=30)

    # Retrain jeśli potrzeba
    was_retrained = tracker.retrain_if_due(predictor, min_new_samples=30)
    if was_retrained:
        logger.info(
            "Model przetrenowany — nowa wersja: v%d", predictor.version
        )

    return report


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
      1. Pobierz propsy
      2. Zbuduj feature vectory
      3. Wygeneruj kupon
    """
    if game_date is None:
        game_date = date.today().isoformat()

    logger.info("=" * 60)
    logger.info("START PIPELINE DZIENNEGO — %s", game_date)
    logger.info("=" * 60)

    # Inicjalizacja komponentów
    fetcher      = NBADataFetcher(db_path=DB_PATH)
    player_lookup = _build_player_lookup()
    team_lookup   = _build_team_lookup()

    if not player_lookup:
        logger.error("Brak player_lookup — przerywam pipeline")
        return

    # Pobierz listę graczy do OddsFetcher
    player_names = list(player_lookup.keys())
    odds_fetcher  = OddsFetcher(
        #api_key=ODDS_API_KEY,
        nba_player_names=player_names,
        db_path=DB_PATH,
    )

    predictor = PropPredictor(db_path=DB_PATH, model_dir=MODEL_DIR)

    # Załaduj model (jeśli istnieje) lub poinformuj o braku
    try:
        predictor.load()
        logger.info("Załadowano model v%d", predictor.version)
    except FileNotFoundError:
        logger.warning(
            "Brak wytrenowanego modelu — kupon nie zostanie wygenerowany. "
            "Uruchom najpierw trening: python train.py"
        )
        return

    # Kroki pipeline
    props    = step_fetch_props(odds_fetcher, player_lookup)
    if not props:
        logger.warning("Brak propsów — pipeline zakończony bez kuponu")
        return

    enriched = step_build_features(props, fetcher, team_lookup)
    if not enriched:
        logger.warning("Brak feature vectorów — pipeline zakończony bez kuponu")
        return

    coupon = step_generate_coupon(enriched, predictor, game_date)

    logger.info(
        "KONIEC PIPELINE — %d picków wygenerowanych dla %s",
        coupon["total_picks"], game_date,
    )

    export_dashboard_data(db_path=DB_PATH)


def run_verify_and_report():
    """
    Pipeline weryfikacyjny (następnego dnia rano):
      4. Weryfikuj wczoraj
      5. Raport 30-dniowy
      6. Retrain jeśli należy
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    logger.info("=" * 60)
    logger.info("START WERYFIKACJI — weryfikuję %s", yesterday)
    logger.info("=" * 60)

    fetcher   = NBADataFetcher(db_path=DB_PATH)
    tracker   = ResultTracker(db_path=DB_PATH, fetcher=fetcher)
    predictor = PropPredictor(db_path=DB_PATH, model_dir=MODEL_DIR)

    try:
        predictor.load()
    except FileNotFoundError:
        logger.error("Brak modelu — nie można wykonać retrain")

    step_verify_and_report(tracker, predictor, yesterday)
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

def export_dashboard_data(
    db_path: str = DB_PATH,
    output_path: str = "dashboard_data.json",
) -> dict:
    """
    Buduje dashboard_data.json z bazy SQLite.
    Wywoływane automatycznie po każdym uruchomieniu pipeline i weryfikacji.
    """
    import json as _json
    import sqlite3 as _sqlite3
    from collections import defaultdict
    from datetime import datetime as _dt, date as _date, timedelta as _td
    import math as _math

    today     = _date.today().isoformat()
    since_30d = (_date.today() - _td(days=30)).isoformat()
    since_90d = (_date.today() - _td(days=90)).isoformat()

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

    if today_coupon and today_coupon.get("picks"):
        for pick in today_coupon["picks"]:
            p = pick["p_over"] if pick["bet"] == "over" else (1.0 - pick["p_over"])
            pick["ev"] = round(p * pick["odds"] - 1.0, 4)
        today_coupon["picks"].sort(key=lambda x: x.get("ev", 0), reverse=True)

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
    model_stats = {
        "version":    mrow[0] if mrow else None,
        "trained_at": mrow[1] if mrow else None,
        "cv_mae":     mrow[2] if mrow else None,
        "cv_auc":     mrow[3] if mrow else None,
    }

    conn.close()

    data = {
        "generated_at": _dt.now().isoformat(),
        "today_coupon": today_coupon,
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
