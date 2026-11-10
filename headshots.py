"""Zdjęcia zawodników: dopasowanie nazwiska do ID NBA i link do oficjalnego CDN.

Bazą zawodników jest statyczna lista z nba_api (wszyscy gracze w historii, z ID
NBA, bez zapytań do sieci). Zdjęć nie kopiujemy do repo — dashboard ładuje je
bezpośrednio z cdn.nba.com, a gdy brak dopasowania, pokazuje inicjały.
"""
from functools import lru_cache

from history import norm_name

HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/260x190/{id}.png"


@lru_cache(maxsize=1)
def player_db() -> dict[str, int]:
    """{norm_name: player_id}; przy powtórzonym nazwisku wygrywa aktywny gracz."""
    try:
        from nba_api.stats.static import players
    except ImportError:
        return {}
    db = {}
    for p in sorted(players.get_players(), key=lambda p: p["is_active"]):
        db[norm_name(p["full_name"])] = p["id"]
    return db


def player_id(name: str, db: dict[str, int] | None = None) -> int | None:
    return (player_db() if db is None else db).get(norm_name(name))


def _names(data: dict):
    for p in (data.get("today_coupon") or {}).get("picks", []):
        yield p.get("player")
    for r in data.get("bet_history") or []:
        yield r.get("player")
    for tab in (data.get("stats") or {}).values():
        for p in ((tab or {}).get("coupon") or {}).get("picks", []):
            yield p.get("player")
        for r in (tab or {}).get("history") or []:
            yield r.get("player")
    for g in (data.get("upcoming") or {}).get("games", []):
        for players in (g.get("teams") or {}).values():
            for p in players:
                yield p.get("player")


def attach(data: dict, db: dict[str, int] | None = None) -> dict:
    """Dodaje data["player_ids"] = {nazwisko z dashboardu: ID NBA}."""
    ids = {}
    for name in _names(data):
        if name and name not in ids:
            pid = player_id(name, db)
            if pid is not None:
                ids[name] = pid
    data["player_ids"] = ids
    return data
