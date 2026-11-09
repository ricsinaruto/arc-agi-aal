"""Branch groups (arena.branch): cut points, collect gates and checks, pick once, whole-level tokens, state and the
resume from the round archive."""
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

from arena import branch

GAME = "sc25"


def recorded_try(folder, label, replies=9, cleared=True, scripted=0, tokens=100):
    """A level try on disk: ``replies`` gameplay replies (one action each, ``tokens`` completion tokens each), the
    first ``scripted`` of them a restored prefix, a clear after the last one when ``cleared``."""
    att = Path(folder) / "attempts" / label
    att.mkdir(parents=True)
    steps = replies
    obs = [{"kind": "observation", "step": s, "levels": 1 if cleared and s == steps else 0,
            "frames": [[("%x" % (s % 16)) * 64] * 64], "action": {"action": "ACTION1"} if s else None}
           for s in range(steps + 1)]
    log = list(obs)
    if cleared:
        log.append({"kind": "level_bench_clear", "step": steps})
    (att / "log.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in log))
    rows = [{"n": i, "step": i, "gameplay": True, "usage": {"prompt_tokens": 50 + i, "completion_tokens": tokens},
             **({"scripted": True} if i < scripted else {})} for i in range(replies)]
    (att / "rollout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"label": label, "game": GAME, "outcome": "cleared" if cleared else "budget",
            "completion_tokens": tokens * replies, "level_rows": [{"tokens": tokens * replies, "actions": replies}] if cleared else []}


def fake_loop(tmp_path, **b):
    feed = []
    sets = {"L1": {"name": "L1", "role": "train", "game": GAME, "level": 1, "fresh": True},
            "L2": {"name": "L2", "role": "train", "game": GAME, "level": 2},
            "T": {"name": "T", "role": "test", "game": GAME, "level": 1}}
    loop = SimpleNamespace(cfg={"mg": {"branch": {"levels": [1], "parents": 2, **b}}, "mg_sets": sets},
                           output=tmp_path / "loop", feed=SimpleNamespace(emit=lambda kind, **k: feed.append((kind, k))))
    return loop, sets


def test_cut_point_counts_live_replies_before_the_clear(tmp_path):
    recorded_try(tmp_path / "f", "a", replies=9)
    assert branch.cut_point(tmp_path / "f/attempts/a", 0.67) == (6, 600, 9)
    recorded_try(tmp_path / "f", "b", replies=9, scripted=3)  # a restored parent: its replayed prefix is not counted
    assert branch.cut_point(tmp_path / "f/attempts/b", 0.67) == (7, 400, 6)
    recorded_try(tmp_path / "f", "c", replies=2)
    assert branch.cut_point(tmp_path / "f/attempts/c", 0.67) is None


def test_collect_gate_checks_snapshot_and_pick_once(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    loop, sets = fake_loop(tmp_path, clear_gate=0.9)
    f1 = tmp_path / "r01-L1"
    rows = [recorded_try(f1, "a"), recorded_try(f1, "b", cleared=False, replies=12),
            {**recorded_try(f1, "c"), "restore": {"ok": False}}, recorded_try(f1, "d", replies=2)]
    got = branch.collect(loop, 1, {"L1": (f1, rows, {}), "L2": (f1, rows, {}), "T": (f1, rows, {})}, 0)
    # clear rate 3/4 < 0.9: every try offered; L2 is not a branch level, T not a training set
    assert got["by_set"] == {"L1": {"clear_rate": 0.75, "offered": 4, "cuts": 2}}
    assert set(got["dropped"]) <= {"parent_restore_not_exact", "too_short"}
    assert len(loop.mg_branch) == 2
    for rec in loop.mg_branch:
        snap = Path(rec["attempt"])
        assert snap.parent == tmp_path / "vol" / "branch" and (snap / "rollout.jsonl").is_file()
        last = json.loads((snap / "log.jsonl").read_text().splitlines()[-1])
        assert last["kind"] == "observation" and last["step"] == rec["request"]  # cut at the request's board
    # gate: a set clearing at or above clear_gate offers only its longer-than-median tries
    loop2, _ = fake_loop(tmp_path / "x", clear_gate=0.5, parents=5)
    f2 = tmp_path / "x" / "r01-L1"
    rows2 = [recorded_try(f2, "a", replies=6), recorded_try(f2, "b", replies=9), recorded_try(f2, "c", replies=12)]
    got2 = branch.collect(loop2, 1, {"L1": (f2, rows2, {})}, 0)
    assert got2["by_set"]["L1"]["offered"] == 1 and [r["live"] for r in loop2.mg_branch] == [12]
    # pick: next round only, each cut once
    assert branch.pick(loop, 1, "L1") == []
    picked = branch.pick(loop, 2, "L1")
    assert len(picked) == 2 and branch.pick(loop, 3, "L1") == []


def test_continuation_rows_count_the_whole_level(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    loop, sets = fake_loop(tmp_path, parents=1, clear_gate=1.01)
    f1 = tmp_path / "r01-L1"
    branch.collect(loop, 1, {"L1": (f1, [recorded_try(f1, "a")], {})}, 0)
    (rec,) = loop.mg_branch
    child = {"label": "x", "restore": {"source": rec["attempt"], "ok": True}, "level_rows": [{"tokens": 250}]}
    other = {"label": "y", "level_rows": [{"tokens": 250}]}
    out = branch.with_prefix(loop, [child, other])
    assert out[0]["level_rows"][0]["tokens"] == 250 + rec["prefix_tokens"] == 850
    assert out[1] is other and child["level_rows"][0]["tokens"] == 250  # input untouched


def test_remainder_scoring_uses_actions_and_tokens_after_the_cut_against_the_group_medians(tmp_path, monkeypatch):
    import math

    from arena.rl_single import branch_rem_terms

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    loop, sets = fake_loop(tmp_path, parents=1, clear_gate=1.01, score="remainder")
    f1 = tmp_path / "r01-L1"
    branch.collect(loop, 1, {"L1": (f1, [recorded_try(f1, "a")], {})}, 0)
    (rec,) = loop.mg_branch
    assert rec["prefix_actions"] == 6  # cut at request 6 (step 6) of a level that began at step 0
    src = {"source": rec["attempt"], "ok": True}

    def child(label, actions, tokens, cleared=True):
        return {"label": label, "restore": src, "outcome": "cleared" if cleared else "budget",
                "level_rows": [{"tokens": tokens, "actions": actions}] if cleared else [{"tokens": tokens, "actions": actions}]}

    rows = [child("x", 9, 300), child("y", 7, 100), child("z", 13, 500), child("w", 20, 900, cleared=False),
            {"label": "fresh", "outcome": "cleared", "level_rows": [{"tokens": 250, "actions": 4}]}]
    out = branch.score_rows(loop, rows)
    assert [r["label"] for r in out] == ["x", "y", "z", "w", "fresh"]
    assert [r["branch_rem"]["a"] for r in out[:4]] == [3, 1, 7, 14]
    assert out[0]["branch_rem"] == {"pre": 6, "a": 3, "a_med": 3, "t_med": 300}
    assert "branch_rem" not in out[4] and out[4]["level_rows"][0]["tokens"] == 250  # not a continuation: whole level
    assert out[0]["level_rows"][0]["tokens"] == 300  # remainder tokens: no prefix added
    ra, rt = branch_rem_terms(out[1]["branch_rem"], 100, None, 1.15)
    assert math.isclose(ra, 2 * math.log(3)) and math.isclose(rt, math.log(3))
    # RHAE's cap binds on the whole level: baseline 10 -> cap at 10/sqrt(1.15) = 9.3 actions, 3.3 after a 6 prefix
    ra, _ = branch_rem_terms(out[1]["branch_rem"], 100, 10, 1.15)
    assert math.isclose(ra, 0.0)
    # score "whole" (default) keeps the prefix tokens; old cuts without prefix_actions stay whole under remainder
    loop.cfg["mg"]["branch"]["score"] = "whole"
    assert branch.score_rows(loop, rows[:1])[0]["level_rows"][0]["tokens"] == 300 + rec["prefix_tokens"]
    loop.cfg["mg"]["branch"]["score"] = "remainder"
    del rec["prefix_actions"]
    assert "branch_rem" not in branch.score_rows(loop, rows[:1])[0]


def test_unseen_read_plays_its_test_set_every_nth_round():
    import pytest

    from arena.rl_single import MultiMixin

    class Loop(MultiMixin):
        def __init__(self, u):
            self.output = Path("/nonexistent")
            self.cfg = {"mg": {"unseen_read": u},
                        "mg_sets": {"L4": {"name": "L4", "role": "test", "tries": 0, "round_tries": 0},
                                    "cn": {"name": "cn", "role": "test", "tries": 8, "round_tries": 0},
                                    "L1": {"name": "L1", "role": "train", "tries": 6}}}

    loop = Loop({"set": "L4", "tries": 32, "every": 3})
    assert [loop.mg_test_tries(n, "L4") for n in range(1, 8)] == [0, 0, 32, 0, 0, 32, 0]
    assert loop.mg_test_tries(3, "cn") == 0
    assert [Loop({"set": "L4", "tries": 32, "every": 3, "offset": 8}).mg_test_tries(n, "L4") for n in range(8, 15)] == [32, 0, 0, 32, 0, 0, 32]
    with pytest.raises(ValueError):
        loop.mg_unseen_read({"sets": {}})  # no screen entry: no base comparison
    for bad in ({"set": "L1", "tries": 32}, {"set": "L4", "tries": 0}, {"set": "L4", "tries": 4, "each": 2}):
        with pytest.raises(ValueError):
            Loop(bad).mg_unseen_read()
    assert Loop(None).mg_test_tries(3, "L4") == 0
    # the matched base read: a bench of its own (role base_read), held to the tail
    lb = Loop({"set": "L4", "tries": 32, "every": 3, "base_first": True})
    assert lb.mg_unseen_read()["base_first"] and lb.cfg["mg_sets"]["L4-base"]["role"] == "base_read"
    assert lb.mg_tail("r12-L4-base", lb.cfg["mg_sets"]["L4-base"])["last"] is True


def test_resume_finds_cuts_and_rebuilds_a_lost_one_from_the_round_archive(tmp_path, monkeypatch):
    run = tmp_path / "vol"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(run))
    loop, sets = fake_loop(tmp_path, parents=1, clear_gate=1.01)
    f1 = tmp_path / "play" / "r01-L1"
    branch.collect(loop, 1, {"L1": (f1, [recorded_try(f1, "a")], {})}, 0)
    data = branch.state(loop)
    (rec,) = data["branch"]
    (run / "rounds").mkdir(parents=True)
    with tarfile.open(run / "rounds" / "r01.tar", "w:gz", compresslevel=1) as tar:
        tar.add(f1, arcname=f1.name)  # what mg_pack_round writes at the round's end
    from arena.restore import load

    want = load(rec["attempt"], rec["request"])["stop"]
    fresh, _ = fake_loop(tmp_path)
    assert branch.restore_state(fresh, json.loads(json.dumps(data)), run) == {"pool": 1, "missing": 0, "from_tar": 0}
    import shutil

    shutil.rmtree(rec["attempt"])  # the snapshot is lost: rebuilt from the archive
    shutil.rmtree(f1)
    again, _ = fake_loop(tmp_path)
    got = branch.restore_state(again, json.loads(json.dumps(data)), run)
    assert got == {"pool": 1, "missing": 0, "from_tar": 1}
    assert load(again.mg_branch[0]["attempt"], rec["request"])["stop"] == want


def test_anchor_credit_skips_the_cut_board_and_credits_later_shared_boards(tmp_path):
    """v6 keeps GiGPO anchors with branch groups: the cut board (every continuation's first live reply) is the group's
    own advantage, so no anchor there; later boards the continuations share get step credit."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_mg_control import make_loop

    loop = make_loop(tmp_path, mg={"reward_shape": "logeq", "turn_credit": {"anchor": 1.0, "token": 0.0, "clip": 1.0}})
    f = tmp_path / "r07-L1-br"
    grp = [{**recorded_try(f, lab, replies=9, scripted=3, cleared=c), "point": "sc25-game-L1-x"}
           for lab, c in (("a", True), ("b", True), ("c", False))]
    out = loop.mg_turn_credit(f, grp, [1.0, 1.0, -2.77], None)
    assert all(3 not in o["extra"] for o in out) and all(set(o["tokens"]) == set(range(3, 9)) for o in out)
    assert out[0]["extra"][4] > 0 and out[2]["extra"][4] < 0


def test_heldout_alternate_switches_after_two_long_play_rounds_and_digest_gate():
    from arena.rl_single import MultiMixin

    class Loop(MultiMixin):
        def __init__(self, mg):
            self.cfg, self.events = {"mg": mg}, []
            self.feed = SimpleNamespace(emit=lambda kind, **k: self.events.append((kind, k)))

    loop = Loop({"heldout_alternate": {"margin_s": 600, "rounds": 2}})
    assert loop.mg_heldout_alternate(10, 5000, 5500)["streak"] == 0      # trainer the longer lane
    assert loop.mg_heldout_alternate(11, 6200, 5500)["streak"] == 1      # play longer by 700 s
    assert loop.mg_heldout_alternate(12, 5900, None)["streak"] == 1      # no update: holds
    assert loop.mg_heldout_alternate(13, 5600, 5500)["streak"] == 0      # by 100 s: resets
    loop.mg_heldout_alternate(14, 6200, 5500)
    got = loop.mg_heldout_alternate(15, 6300, 5500)
    assert got["on"] and got["since"] == 15 and [k for k, _ in loop.events] == ["heldout_alternate"]
    assert loop.mg_heldout_alternate(16, 1000, 5500)["on"]  # for good

    gate = Loop({"digest_gate": True})
    same = {"source": "/runs/validation/x/harvest/r09-L2-a", "prompt_digest": {"same": True}}
    assert gate.mg_digest_ok({"restore": same})
    assert not gate.mg_digest_ok({"restore": {**same, "prompt_digest": {"same": False}}})
    assert not gate.mg_digest_ok({"restore": {"source": "/tmp/branch-from-tar/r09-x/cut", "prompt_digest": None}})
    assert gate.mg_digest_ok({"restore": {"source": "/runs/validation/rec/attempts/a01", "prompt_digest": None}})  # bank start
    assert gate.mg_digest_ok({"label": "fresh"})
    assert Loop({}).mg_digest_ok({"restore": {**same, "prompt_digest": {"same": False}}})
