"""rl_rollback.py on a v7 run (step ids sNNNN): from the step's snapshot or the current state cut back; the restart
resumes on the restored version, makes new group ids only and never trains an abandoned-branch group."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from test_rl_async_loop import events, v7_loop

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rl_rollback", ROOT / "scripts/posttrain/rl_rollback.py")
RB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RB)


def run_first(tmp_path, monkeypatch, steps=4):
    loop = v7_loop(tmp_path, monkeypatch, [], steps=steps, play_delay=0.05)
    asyncio.run(loop.mg_loop())
    out = tmp_path / "out"
    rows = [{"kind": k, **d} for k, d in loop.feed.events]
    for r in rows:  # the stub trainer reports no grad norm; the real and fake ones do
        if r["kind"] == "step" and r.get("grad_norm") is None:
            r["grad_norm"] = 0.01
    feed = "\n".join(json.dumps(r) for r in rows) + "\n"
    return loop, out, json.loads((out / "v7_state.json").read_text()), feed


def test_snapshot_rollback_and_resume(tmp_path, monkeypatch):
    first, out, state, feed = run_first(tmp_path, monkeypatch)
    snap_path = Path(state["checkpoint"]).parent / "s0002" / "v7_state.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    if not snap_path.exists():  # the stub trainer's checkpoint makes no folder: the loop wrote none
        pytest.skip("no snapshot folder from the stub trainer")
    snap = json.loads(snap_path.read_text())
    new, problems = RB.rollback_v7(state, feed, 2, snap, ["state.json", "v7_state.json"])
    assert not problems and new["step"] == 2 and new["book"]["current"] == 2 and new["checkpoint"].endswith("/s0002")
    assert new["made"]["n"] == state["made"]["n"]
    keep = {g["gid"] for g in new["buffer"]["ready"]} | {g["gid"] for g in new["running"]}
    later = {f"g{n:05d}" for n in range(1, state["made"]["n"] + 1)} - keep
    assert all(new["buffer"]["closed"].get(g) in ("rolled_back", "consumed", "dropped") for g in later)
    assert not keep & set(new["buffer"]["closed"])


def test_cut_back_rollback_checks_and_resume(tmp_path, monkeypatch):
    first, out, state, feed = run_first(tmp_path, monkeypatch)
    # refusals: a later step, a step without a feed row, a checkpoint folder without files
    assert RB.rollback_v7(state, feed, state["step"], None, ["state.json"])[1]
    assert RB.rollback_v7(state, feed.replace('"kind": "step"', '"kind": "x"'), 2, None, ["state.json"])[1]
    assert RB.rollback_v7(state, feed, 2, None, ["other"])[1]
    new, problems = RB.rollback_v7(state, feed, 2, None, ["state.json"])
    assert not problems and new["book"]["lineage"][-1] == 2 and new["book"]["served"] == 2
    assert all(int(g["version"]) <= 2 for g in new["buffer"]["ready"] + new["running"])
    (out / "v7_state.json").write_text(json.dumps(new))
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=4, play_delay=0.05)
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    assert asyncio.run(second.mg_loop()) == "max_steps"
    start = events(second, "v7_start")[0]
    assert start["resume"] and start["step"] == 2 and start["version"] == 2
    assert [s["version"] for s in events(second, "step")] == [3, 4]  # the new branch
    made = int(state["made"]["n"])
    new_gids = [d["gid"] for d in events(second, "group_start")]
    replayed = {g["gid"] for g in new["running"]} | {g["gid"] for g in new["buffer"]["ready"]}
    assert all(int(g[1:]) > made for g in new_gids if g not in replayed)  # fresh ids only
    trained = {gid for gid, why in second.v7["buffer"].closed.items() if why == "consumed"}
    rolled = {gid for gid, why in new["buffer"]["closed"].items() if why == "rolled_back"}
    assert not trained & rolled


def test_cli_offline_v7(tmp_path, monkeypatch, capsys):
    first, out, state, feed = run_first(tmp_path, monkeypatch)
    capsys.readouterr()  # the run's own stdout (a stall dump under load) is not the CLI's
    (tmp_path / "feed.jsonl").write_text(feed)
    monkeypatch.setattr("sys.argv", ["rl_rollback.py", "validation/x", "s0002", "--state", str(out / "v7_state.json"),
                                     "--feed", str(tmp_path / "feed.jsonl"), "--files", "state.json", "--apply",
                                     "--out", str(tmp_path / "new.json")])
    RB.main()
    got = json.loads(capsys.readouterr().out.split("written:")[0])
    assert got["ok"] and got["after"]["step"] == 2 and json.loads((tmp_path / "new.json").read_text())["step"] == 2
    monkeypatch.setattr("sys.argv", ["rl_rollback.py", "validation/x", "s0002", "--state", str(out / "v7_state.json"),
                                     "--feed", str(tmp_path / "feed.jsonl"), "--lr", "1e-6"])
    with pytest.raises(SystemExit):
        RB.main()
