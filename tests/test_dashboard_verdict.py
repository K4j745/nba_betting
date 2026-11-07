import json

import main


def _tab(diff, ci):
    return {"kalshi_eval": {"rows": [{"features": "base", "logloss_diff": diff, "logloss_diff_ci95": ci, "n": 100}]}}


def test_verdict_has_no_edge_when_every_ci_reaches_zero():
    v = main._verdict({"PTS": _tab(0.02, [0.01, 0.03]), "REB": _tab(-0.001, [-0.01, 0.002])})
    assert v["has_edge"] is False and set(v["stats"]) == {"PTS", "REB"}


def test_verdict_has_edge_only_when_whole_ci_is_below_zero():
    assert main._verdict({"PTS": _tab(-0.02, [-0.03, -0.01])})["has_edge"] is True


def test_verdict_is_empty_without_reports():
    assert main._verdict({"PTS": {"kalshi_eval": None}}) is None


def test_research_log_reads_file_and_tolerates_missing_one(tmp_path):
    assert main._research_log(str(tmp_path)) == []
    (tmp_path / "research_log.json").write_text(json.dumps([{"name": "x"}]), encoding="utf-8")
    assert main._research_log(str(tmp_path)) == [{"name": "x"}]
