# nba_betting/scripts/probe_nba_api.py
"""
Diagnostyka: czy z danej maszyny (np. runner GitHub Actions) da się pobrać dane z stats.nba.com.

Sprawdza stats.nba.com (nba_api.stats — game logi, statystyki zaawansowane),
z którego korzysta cały pipeline.

Użycie:
  python scripts/probe_nba_api.py
Kod wyjścia 0 zawsze — wynik czytamy z logu.
"""
import time


def _probe(name, fn):
    t0 = time.time()
    try:
        info = fn()
        print(f"OK    {name:<34} {time.time() - t0:5.1f}s  {info}")
    except Exception as e:
        print(f"FAIL  {name:<34} {time.time() - t0:5.1f}s  {type(e).__name__}: {str(e)[:120]}")


def main():
    from config import CURRENT_SEASON
    from nba_api.stats.endpoints import leaguegamelog

    _probe(
        f"stats.nba.com LeagueGameLog {CURRENT_SEASON}",
        lambda: f"{len(leaguegamelog.LeagueGameLog(season=CURRENT_SEASON, player_or_team_abbreviation='P', timeout=20).get_data_frames()[0])} wierszy",
    )


if __name__ == "__main__":
    main()
