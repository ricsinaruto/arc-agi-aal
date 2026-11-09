"""Checkpoint and resume (arena.checkpoint): a trial killed with SIGKILL and resumed in a new process ends in the state
of an uninterrupted run; the pieces (state overlay, environment replay, interpreter replay); off by default."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena import checkpoint as ck

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests/checkpoint_driver.py"
CHECKPOINTS = {"ARC_CHECKPOINT_SECONDS": "1", "ARC_CHECKPOINT_PATIENCE": "0.3", "ARC_CHECKPOINT_HOLD": "3"}
# The v7.1p harness (dist v71p-excl5 study), on the plain gate.
V71 = {
    "harness_fixes": {"fixes": ["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox",
                                "timeout_interrupt", "prompt_text", "summary_headroom", "text_count"],
                      "summary_room_tokens": 4096, "inventory_max_tokens": 1536, "inventory_receipts": 12,
                      "stuck_turns": 16, "stuck_notice_turns": 8, "stuck_no_change_actions": 5,
                      "interrupt_grace_seconds": 3.0},
    "drain_half": {"drain_tokens": 59392, "turn_cap": 150, "turn_drain": 30, "coalesce": True,
                   "tool_output_tokens": 3072, "middle_truncation": True, "drop_malformed": True, "note": True},
    "hx_harness": {"toolbox": True, "reflect": True, "economy": True, "memo": True, "reflect_requests": 8,
                   "reflect_every": 8, "reflect_no_change": 5, "reflect_revisits": 3, "reflect_deaths": 2,
                   "reflect_min_gap": 3, "reflect_style": 2, "reflect_stall": 5, "memo_once": 1},
    "dfr_harness": {"vision": True, "guards": True, "actions": True, "level_prompt": True, "frame_diff": True,
                    "animation": True},
    "dfr_loop": {"prompt_fix": True, "yield_tokens": 0, "no_world_model": False, "action_feedback": False,
                 "inventory_animation": True},
    "loop_rule": {"action": "fresh_context", "streak": 20, "mean_completion": 300, "similarity": 0.9},
    "reasoning_history": "verbatim",
    "priority": {"refresh_queue": True, "tail_fade": True, "tail_fade_fraction": 0.2, "tail_lookup": "remaining",
                 "normalize_score": True, "human_actions": 25.0, "action_scale": 115.0, "token_scale": 62000.0,
                 "action_weight": 0.25, "tail_base": 2.0, "tail_cap": 4.0},
}


def drive(output, features, **env):
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, str(DRIVER), str(output), json.dumps(features)], start_new_session=True,
        stdout=(output / "driver.log").open("ab"), stderr=subprocess.STDOUT,
        env={**{k: v for k, v in os.environ.items() if not k.startswith("ARC_CHECKPOINT")},
             "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}", **env})


def finished(output, features, **env):
    process = drive(output, features, **env)
    assert process.wait(timeout=180) == 0, (output / "driver.log").read_text()[-3000:]
    return json.loads((output / "digest.json").read_text())


def wait_for(condition, seconds=60):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


@pytest.mark.parametrize("features,forced", [({}, True), (V71, False), ({**V71, "cache_admission": "hybrid_blocks"}, True)],
                         ids=["plain-forced", "v71-periodic", "v71-blocks-forced"])
def test_a_killed_trial_resumes_into_the_uninterrupted_state(tmp_path, features, forced):
    reference = finished(tmp_path / "ref", features)
    assert reference["checkpoint"] is None and reference["status"] == "complete"
    assert all(g["steps"] > 0 and g["error"] is None for g in reference["games"].values())
    assert {g["status"] for g in reference["games"].values()} == {"won", "action_limit"}

    cut = tmp_path / "cut"
    folder = cut / "t/checkpoint"
    process = drive(cut, features, **CHECKPOINTS)
    wait_for(lambda: len(ck.generations(folder)) >= 2 or process.poll() is not None)
    if forced:  # the runner's preemption notice: a checkpoint now, whatever the interval
        newest = ck.generations(folder)[-1].name
        (folder / "request").touch()
        wait_for(lambda: not (folder / "request").exists() and ck.generations(folder)[-1].name != newest)
    assert process.poll() is None, (cut / "driver.log").read_text()[-3000:]
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    assert not (cut / "digest.json").exists()
    _, manifest = ck.latest(folder)
    assert manifest["capture"]["unresolved"] in ({}, {"_asyncio:Future": 1}, {"_asyncio:Future": 2})
    assert any(not g["finished"] and g["steps"] for g in manifest["games"].values())  # cut in mid-game

    resumed = finished(cut, features, **CHECKPOINTS, ARC_CHECKPOINT_RESUME="1", ARC_CHECKPOINT_ATTEMPT="2")
    histories = [json.loads((run / "histories.json").read_text()) for run in (tmp_path / "ref", cut)]
    for game_id, row in reference["games"].items():
        other = resumed["games"][game_id]
        assert histories[0][game_id] == histories[1][game_id], game_id
        assert {k: (v, other[k]) for k, v in row.items() if v != other[k]} == {}, game_id
    for key in ("status", "end_reason", "task_errors", "requests", "scheduler"):
        assert resumed[key] == reference[key], key
    record = resumed["checkpoint"]
    assert record["kind"] == "resumed" and record["attempt"] == 2
    assert [s["kind"] for s in record["segments"]] == ["fresh", "resumed"]
    segment = record["segments"][-1]
    assert segment["from_checkpoint"] == manifest["seq"]
    assert segment["resumed_at_elapsed_seconds"] == manifest["elapsed_seconds"]
    assert segment["sandbox"]["reset"] == [] and segment["sandbox"]["inexact_calls"] == 0
    assert segment["sandbox"]["replayed"] == sum(g["sandbox_calls"] > 0 and not g["finished"]
                                                 for g in manifest["games"].values())
    assert segment["discarded_inflight"] == manifest["inflight_at_cut"]
    assert set(segment["restore"]["unresolved"]) <= {"_asyncio:Future"}
    # One clock for the whole run: the resumed log continues the checkpoint's elapsed time.
    log = [json.loads(line) for line in (cut / "t/admission.jsonl").read_text().splitlines()]
    mark = next(i for i, e in enumerate(log) if e.get("event") == "resumed")
    assert abs(log[mark]["wall_seconds"] - manifest["elapsed_seconds"]) < 0.5
    assert all(e["wall_seconds"] >= manifest["elapsed_seconds"] - 0.5 for e in log[mark:] if e.get("wall_seconds"))


def test_checkpointing_is_off_without_the_environment_and_refuses_what_it_cannot_cut(tmp_path):
    study = {}
    assert ck.from_env(study, "duck-transfer", tmp_path, environ={}) is None
    assert ck.from_env(study, "duck-transfer", tmp_path, environ={"ARC_CHECKPOINT_RESUME": "1"}) is None
    events = []
    assert ck.from_env({"explorer": {"at_start": True}}, "duck-transfer", tmp_path, log=events.append,
                       environ={"ARC_CHECKPOINT_SECONDS": "600"}) is None
    assert ck.from_env(study, "duck-public", tmp_path, environ={"ARC_CHECKPOINT_SECONDS": "600"}) is None
    assert events == [{"event": "checkpoint_off", "unsupported": ["explorer"]}]
    made = ck.from_env(study, "duck-transfer", tmp_path, environ={"ARC_CHECKPOINT_SECONDS": "600"})
    assert made.cfg["interval_seconds"] == 600 and not made.resuming and made.summary()["kind"] == "fresh"
    # A resume without a checkpoint is a fresh attempt (flagged by its number), never one in a used folder.
    again = ck.from_env(study, "duck-transfer", tmp_path, environ={"ARC_CHECKPOINT_SECONDS": "600",
                                                                  "ARC_CHECKPOINT_RESUME": "1",
                                                                  "ARC_CHECKPOINT_ATTEMPT": "3"})
    assert not again.resuming and again.summary()["attempt"] == 3 and again.summary()["kind"] == "fresh"
    (tmp_path / "progress.jsonl").write_text("{}\n")
    with pytest.raises(ck.ResumeError):
        ck.from_env(study, "duck-transfer", tmp_path, environ={"ARC_CHECKPOINT_SECONDS": "600",
                                                              "ARC_CHECKPOINT_RESUME": "1"})


def test_a_torn_or_altered_generation_is_skipped(tmp_path):
    trial, tmp_path = tmp_path, tmp_path / "checkpoint"
    for seq, body in ((1, b"first"), (2, b"second")):
        folder = tmp_path / f"gen-{seq:06d}"
        folder.mkdir(parents=True)
        (folder / "state.pkl.z").write_bytes(body)
        (folder / "actions.json").write_text("{}")
        (folder / "manifest.json").write_text(json.dumps({
            "seq": seq, "elapsed_seconds": 300.0 * seq, "bytes": len(body),
            "sha256": __import__("hashlib").sha256(body).hexdigest()}))
    (tmp_path / "gen-000003.tmp").mkdir()  # being written when the machine went
    assert ck.latest(tmp_path)[1]["seq"] == 2
    # What a resumed trial has played is what its wall no longer needs (arena.run); nothing without the resume.
    resume = {"ARC_CHECKPOINT_SECONDS": "600", "ARC_CHECKPOINT_RESUME": "1"}
    assert ck.played_seconds(trial, resume) == 600.0 and ck.played_seconds(trial, {"ARC_CHECKPOINT_SECONDS": "600"}) == 0
    (tmp_path / "gen-000002/state.pkl.z").write_bytes(b"sec0nd")
    assert ck.latest(tmp_path)[1]["seq"] == 1 and ck.played_seconds(trial, resume) == 300.0
    (tmp_path / "gen-000001/manifest.json").unlink()
    assert ck.latest(tmp_path) is None


def test_the_overlay_keeps_process_objects_and_restores_state_in_place():
    from arena.clock import ActiveClock, GlobalClock
    from arena.scheduler import GameRecord

    def trial():
        wall = GlobalClock(900)
        games = [GameRecord(f"g{i}", i, ActiveClock(wall)) for i in range(2)]
        for game in games:
            game.ready = asyncio.Event()
            game.listener = lambda: None
        note = None

        def carry(value=None):
            nonlocal note
            note = note if value is None else value
            return note

        return {"wall": wall, "games": games, "carry": carry, "lock": threading.Lock()}

    old = trial()
    old["wall"].release()
    for game in old["games"]:
        game.clock.release()
    old["games"][0].clock.pause()
    old["games"][0].ready.set()
    old["games"][1].clock.charge(1000, 250)
    old["games"][1].levels = 2
    old["games"][1].extra = SimpleNamespace(clock=old["games"][0].clock)
    old["carry"]("kept")
    data, report = ck.capture(old)
    assert report["unresolved"] == {} and report["anchors"] == 5

    new = trial()
    held = (new["wall"], new["games"][1], new["games"][0].ready, new["lock"], new["games"][0].listener, new["carry"])
    loaded, report = ck.restore(new, data)
    assert report["unresolved"] == {} and report["matched"] == 5
    assert (new["wall"], new["games"][1], new["games"][0].ready, new["lock"], new["games"][0].listener,
            new["carry"]) == held  # every object the new process holds is the same object
    assert new["wall"].start == old["wall"].start and new["games"][1].levels == 2
    assert new["games"][1].clock.completion_tokens() == 250 and new["games"][1].clock.global_clock is new["wall"]
    assert new["games"][0].clock.parked and not new["games"][0].clock._event.is_set()  # the event follows the state
    assert new["games"][0].ready.is_set() and not new["games"][1].ready.is_set()
    assert new["games"][1].extra.clock is new["games"][0].clock and new["carry"]() == "kept"
    assert loaded["games"][0] is new["games"][0]


def test_an_environment_replays_its_recorded_calls_and_refuses_another_board():
    from arcengine import GameAction
    from checkpoint_driver import ToyEnv

    logs = {}
    env = ck.RecordingArcade(SimpleNamespace(make=lambda game_id, **kw: ToyEnv(1)), logs).make("g")
    for name, data in (("ACTION1", {}), ("ACTION6", {"x": 3, "y": 4}), ("ACTION4", {}), ("ACTION4", {})):
        env.step(GameAction[name], data)
    env.reset()
    board = env.observation_space.frame[-1]
    recorded = json.loads(json.dumps(logs["g"]))
    assert [call[0] for call in recorded] == ["ACTION1", "ACTION6", "ACTION4", "ACTION4", "RESET"]
    again = ck.RecordingArcade(SimpleNamespace(make=lambda game_id, **kw: ToyEnv(1)), {}, {"g": recorded}).make("g")
    assert (again.observation_space.frame[-1] == board).all() and again._log == recorded
    with pytest.raises(ck.ResumeError, match="another board"):
        ck.RecordingArcade(SimpleNamespace(make=lambda game_id, **kw: ToyEnv(2)), {}, {"g": recorded}).make("g")


def sandbox():
    from duck_transfer import loader
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap

    run = loader.load().run_sandboxed_python
    made = StatefulSandbox(stateful_bootstrap(preserve_transition_results(run.__globals__["_SANDBOX_BOOTSTRAP"])),
                           color_chars=run.__globals__["ARC_COLOR_CHARS"])
    made.recorder = ck.SandboxRecorder()
    return made


def test_an_interpreter_is_rebuilt_from_its_recorded_inputs_or_reset_when_it_diverges():
    state = {"current_frame": None, "history": []}

    def run(box, code):
        return box.run(code=code, timeout_seconds=20, initial_state=state,
                       action_handler=lambda actions: {"action_result": {"executed": True}, "state": state})

    first = sandbox()
    try:
        assert run(first, "seen = [1]\ndef grow():\n    seen.append(len(seen) * 2)\n    return seen")["error"] == ""
        assert run(first, "grow()\naction([{'action': 'UP'}])\nprint(grow())")["stdout"] == "[1, 2, 4]\n"
        second = sandbox()
        second.recorder.__dict__.update({k: list(v) if isinstance(v, list) else v
                                         for k, v in vars(first.recorder).items()})
        try:
            report = ck.replay_sandbox(second)
            assert report["requests"] == 2 and report["inexact"] == 0 and "reset" not in report
            assert run(second, "print(grow())")["stdout"] == "[1, 2, 4, 6]\n" and second.restarts == 0
        finally:
            second.close()
        third = sandbox()
        third.recorder.__dict__.update(vars(first.recorder))
        third.recorder.seen = [("final", None), ("final", None)]  # the recorded call had asked for an action
        try:
            assert ck.replay_sandbox(third)["reset"] == "action_diverged" and third.restarts == 1
            assert "NameError" in run(third, "print(grow())")["error"]
        finally:
            third.close()
    finally:
        first.close()
