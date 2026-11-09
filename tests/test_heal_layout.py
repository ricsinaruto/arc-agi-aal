"""Streaming heal round card layout (modal_rl1.stream_layout / stream_teacher_env, ARC_HEAL_TRANKS): pure functions read
from the launcher's source (no modal import), as heal_preflight runs them."""
import ast
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "modal_rl1.py"


def fn(name):
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (name, "stream_layout")]
    ns = {"os": os}
    exec(compile(ast.Module(keep, []), "modal_rl1_" + name, "exec"), ns)
    return ns[name]


def test_default_layout_is_unchanged():
    layout = fn("stream_layout")
    assert layout(4) == (2, "0,1", "2,3", 2)  # the 2 + 2 rounds: the pinned devices before the knob
    assert layout(4, 0) == layout(4, 2)
    assert fn("trainer_ranks")(4, "play") == 2 and fn("trainer_ranks")(4) == 4 and fn("trainer_ranks")(2) == 2


def test_three_plus_one_and_others():
    layout = fn("stream_layout")
    assert layout(4, 3) == (3, "0,1,2", "3", 1)
    assert layout(5, 3) == (3, "0,1,2", "3,4", 2)
    assert layout(8, 6) == (6, "0,1,2,3,4,5", "6,7", 2)
    assert fn("trainer_ranks")(4, "play", 3) == 3
    for cards, t in ((4, 4), (4, 5), (1, 0), (2, 0)):
        with pytest.raises(ValueError):
            layout(cards, t)


def test_teacher_env():
    env = fn("stream_teacher_env")
    assert env(2, "") == {}
    assert env(1, "RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1") == {"RL1_EXPERTS_FP8": "1", "RL1_STREAM_CHAINS": "1"}
    with pytest.raises(ValueError, match="one-card"):
        env(1, "")  # BF16 Flash does not fit one card
    with pytest.raises(ValueError, match="not a teacher flag"):
        env(2, "HEAL_CAPTURE_CACHE=1")
    c7k = "RL1_EXPERTS_FP8=1,RL1_QSA_TIES=index,HEAL_VERIFY_SLICING=pieces,HEAL_VERIFY_KL=0.052"  # the c7k teacher env
    assert env(1, c7k)["HEAL_VERIFY_SLICING"] == "pieces"


def test_confirm_reads_the_first_play_only():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("stream_trainer_env", "follow_path")]
    ns = {"NEBIUS_RCLONE": {}, "NEBIUS_BUCKET": "b"}
    exec(compile(ast.Module(keep, []), "modal_rl1_env", "exec"), ns)
    env = ns["stream_trainer_env"]({}, chunks="c", k=1, max_age=8, deltas="d", run_id="r", play="live,old1,old2")
    assert env["HEAL_DELTA_CONFIRM"] == "arc:b/runs/validation/live/arena/working/trials/duck-transfer/weights-follow.jsonl"
    assert "HEAL_DELTA_CONFIRM" not in ns["stream_trainer_env"]({}, chunks="", k=0, max_age=0, deltas="", run_id="r")
    assert env["HEAL_DELTA_REMOTE"] == "arc:b/runs/validation/r/deltas"
    moved = ns["stream_trainer_env"]({"HEAL_DELTA_RUN": "first"}, chunks="", k=0, max_age=0, deltas="", run_id="r")
    assert moved["HEAL_DELTA_REMOTE"] == "arc:b/runs/validation/first/deltas"  # a renamed restart keeps its play


def test_env_pairs_allows_heal_flags_and_nccl_debug_only():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "env_pairs"]
    ns = {}
    exec(compile(ast.Module(keep, []), "modal_rl1_env", "exec"), ns)
    assert ns["env_pairs"]("HEAL_A=1,NCCL_DEBUG=INFO") == {"HEAL_A": "1", "NCCL_DEBUG": "INFO"}
    with pytest.raises(ValueError):
        ns["env_pairs"]("NCCL_P2P_LEVEL=NVL")


def test_restart_guard_halts_a_second_attempt_without_a_checkpoint(tmp_path, monkeypatch):
    """a re-run input (same call id) alerts; it resumes from a complete checkpoint, otherwise halts."""
    import json
    import time

    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("restart_guard", "preempt_retry")]
    ns = {"os": os, "json": json, "time": time, "modal": None, "runs": None}
    exec(compile(ast.Module(keep, []), "modal_rl1_guard", "exec"), ns)
    guard, retry = ns["restart_guard"], ns["preempt_retry"]
    monkeypatch.delenv("ARC_HEAL_RESTART_GO", raising=False)
    out = tmp_path / "run"
    assert guard(out, retry(out), call_id="fc-1") is None  # first attempt
    (out / "train").mkdir()
    time.sleep(1.1)
    halt = guard(out, retry(out), call_id="fc-1")  # Modal re-ran it: no checkpoint -> halt
    assert halt["halted"] and halt["attempt"] == 2 and json.loads((out / "restart-alert.json").read_text())["action"] == "halt"
    time.sleep(1.1)
    assert guard(out, retry(out), call_id="fc-2") is None  # a new launch (new call) is attempt 1
    ck = out / "train/ckpt-6"
    ck.mkdir(parents=True)
    (ck / "progress.json").write_text(json.dumps({"student": "s"}))
    time.sleep(1.1)
    r = retry(out)
    assert r["step0"] == 6 and guard(out, r, call_id="fc-2") is None  # attempt 2 with a checkpoint: resume + alert
    assert json.loads((out / "restart-alert.json").read_text())["action"] == "resume"


def test_live_play_knobs():
    """24 Oct: <round>/stream/knobs.json swaps the student plays; confirm defaults to every listed play; anything else
    is refused (the round keeps its list); the trainer env names the live confirm list only when the round writes one."""
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("stream_knobs", "stream_trainer_env", "follow_path")]
    ns = {"NEBIUS_RCLONE": {}, "NEBIUS_BUCKET": "b"}
    exec(compile(ast.Module(keep, []), "modal_rl1_knobs", "exec"), ns)
    knobs = ns["stream_knobs"]
    assert knobs({"plays": ["v23", "v22", "v23"]}) == {"plays": ["v23", "v22"], "confirm": ["v23", "v22"], "lr": None, "ckpt": None}
    assert knobs({"plays": ["v23", "v22"], "confirm": ["v23"]}) == {"plays": ["v23", "v22"], "confirm": ["v23"], "lr": None, "ckpt": None}
    assert knobs({"lr": 5e-6}) == {"plays": None, "confirm": None, "lr": 5e-6, "ckpt": None}
    assert knobs({"ckpt": "s7"})["ckpt"] == "s7" and knobs({"ckpt": 3})["ckpt"] == "3"  # c7n: live checkpoint tag
    for bad in ({"plays": "v23"}, {"plays": ["v23"], "confirm": ["v21"]}, {"plays": ["../x"]}, {"plays": [], "x": 1}, [],
                {}, {"lr": 0.01}, {"lr": True}, {"confirm": ["v23"]}, {"ckpt": True}, {"ckpt": "a b"}, {"ckpt": ["x"]}):
        with pytest.raises(ValueError):
            knobs(bad)
    env = ns["stream_trainer_env"]({}, chunks="c", k=1, max_age=8, deltas="d", run_id="r", play="v21,v22", confirm_live="/c.json",
                                   live_knobs="/t.json")
    assert env["HEAL_DELTA_CONFIRM_LIVE"] == "/c.json" and env["HEAL_LIVE_KNOBS"] == "/t.json"
    assert "HEAL_DELTA_CONFIRM_LIVE" not in ns["stream_trainer_env"]({}, chunks="c", k=1, max_age=8, deltas="d", run_id="r", play="v21")
