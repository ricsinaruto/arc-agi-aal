import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import score_marks as sm  # noqa: E402


def _log(rows):
    return "\n".join(json.dumps(r) for r in rows)


def _model(n):
    return {"kind": "model", "purpose": "gameplay", "usage": {"completion_tokens": n}}


def _req(actions, levels):
    return {"kind": "model_request", "runtime": {"actions_executed": actions, "levels_completed": levels}}


def test_official_formula():
    # level 1 at baseline (100), level 2 at 2x baseline (25), level 3 uncleared; weights 1, 2, 3
    assert abs(sm.try_score([10, 50], [10, 20, 30]) - (100 * 1 + 25 * 2) / 6) < 1e-9
    assert sm.try_score([5], [10, 10]) == min(115 * 1 / 3, 1 / 3 * 100)  # capped at 115, then by the cleared share
    assert sm.try_score([], [10]) == 0.0


def test_level_ups_are_stamped_with_tokens_before_them(tmp_path):
    t = sm.parse_log(_log([_req(0, 0), _model(60000), _req(12, 1), _model(50000), _req(30, 2), _model(5000)]))
    assert t["ups"] == [(60000, 12), (110000, 30)] and t["tokens"] == 115000
    covered, ups = sm.at_mark(t, 100000)
    assert covered and ups == [12]
    g = tmp_path / "ab12" / "v1"
    g.mkdir(parents=True)
    (g / "metadata.json").write_text(json.dumps({"baseline_actions": [12, 18]}))
    short = sm.parse_log(_log([_req(0, 0), _model(40000)]))  # stopped short of the mark: not covered
    res = sm.score({"k000": t, "k001": short}, [{"game_id": "k000", "source_game_id": "ab12-v1"},
                                                 {"game_id": "k001", "source_game_id": "ab12-v1"}], [100000], tmp_path)
    r = res[100000]
    assert r["coverage"] == 0.5 and not r["reportable"]
    assert abs(r["games"]["ab12"]["score"] - (100 / 3) / 2) < 1e-3 and r["games"]["ab12"]["levels"] == 0.5


def test_finished_try_is_covered_and_non_gameplay_tokens_ignored():
    t = sm.parse_log(_log([_model(1000), {"kind": "model", "purpose": "compaction", "usage": {"completion_tokens": 99999}},
                           {"kind": "finished", "actions": 7, "levels": 1}]))
    assert t["tokens"] == 1000 and t["finished_at"] == 1000 and t["ups"] == [(1000, 7)]
    assert sm.at_mark(t, 100000) == (True, [7])
