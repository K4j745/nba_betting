import headshots

DB = {"luka doncic": 1629029, "jaren jackson": 1628991}


def test_player_id_normalizes_accents_and_suffixes():
    assert headshots.player_id("Luka Dončić", DB) == 1629029
    assert headshots.player_id("Jaren Jackson Jr.", DB) == 1628991
    assert headshots.player_id("Nobody Known", DB) is None


def test_attach_collects_names_from_all_sections():
    data = {
        "today_coupon": {"picks": [{"player": "Luka Doncic"}]},
        "stats": {"REB": {"coupon": {"picks": [{"player": "Jaren Jackson Jr."}]}, "history": []}},
        "upcoming": {"games": [{"teams": {"DAL": [{"player": "Unknown Guy"}]}}]},
    }
    ids = headshots.attach(data, DB)["player_ids"]
    assert ids == {"Luka Doncic": 1629029, "Jaren Jackson Jr.": 1628991}


def test_real_db_has_active_players():
    db = headshots.player_db()
    assert db.get("luka doncic") == 1629029
