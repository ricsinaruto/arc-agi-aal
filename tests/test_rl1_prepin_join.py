"""U2 (27 Oct): a GPU op waits for the prepin thread's pinned allocations before it starts (U1d: 2,696 cudaHostAlloc
calls, 9.8 s, inside the replay leg's first accumulate); the wait is reported beside the op's seconds."""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))


def fake(rt, monkeypatch, alive_s):
    monkeypatch.setattr(rt, "op_paths", lambda model, cmd: None)
    monkeypatch.setattr(rt, "gpu_used_gib", lambda: (0.0, 0.0, None))
    done = {}
    thread = threading.Thread(target=lambda: (time.sleep(alive_s), done.setdefault("at", time.time())))
    thread.start()
    events = []
    me = types.SimpleNamespace(_prepin=(thread, {}), parked=False, model=None, save_join=lambda: None,
                               rec=types.SimpleNamespace(event=lambda *a, **k: events.append(k)))
    me.op_accumulate = lambda cmd: {"started": time.time()}
    return me, done, thread


def test_gpu_op_starts_after_the_prepin(monkeypatch):
    import rl1_trainer as rt

    me, done, _ = fake(rt, monkeypatch, 0.3)
    out = rt.Trainer.run_op(me, {"op": "accumulate"}, 0)
    assert out["ok"] and out["started"] >= done["at"] and out["prepin_join_s"] >= 0.2


def test_off_does_not_wait(monkeypatch):
    import rl1_trainer as rt

    monkeypatch.setenv("RL1_PREPIN_JOIN", "0")
    me, done, thread = fake(rt, monkeypatch, 0.3)
    out = rt.Trainer.run_op(me, {"op": "accumulate"}, 0)
    assert "prepin_join_s" not in out and "at" not in done
    thread.join()


def test_op_reports_its_pre_and_wall_seconds(monkeypatch):
    """U4 (27 Oct): ~3-4 s of each accumulate sat outside train_seconds; run_op splits pre-op from the op's wall."""
    import rl1_trainer as rt

    me, _, thread = fake(rt, monkeypatch, 0.0)
    thread.join()
    me.save_join = lambda: time.sleep(0.2)
    me.op_accumulate = lambda cmd: (time.sleep(0.3), {})[1]
    out = rt.Trainer.run_op(me, {"op": "accumulate"}, 0)
    assert 0.15 <= out["pre_seconds"] <= out["seconds"] and 0.25 <= out["op_wall_seconds"] <= out["seconds"]


def test_a_failed_op_logs_its_traceback_and_position(monkeypatch):
    """v17 a3/a5 (30 Oct): a failed accumulate's traceback lived only in the container's done/ file; the trainer
    log row keeps it, with the windows reached and the adaptive-ckpt state."""
    import rl1_trainer as rt

    events = []
    me, _, thread = fake(rt, monkeypatch, 0.0)
    thread.join()
    me.rec = types.SimpleNamespace(event=lambda *a, **k: events.append(k))
    me.optimizer = types.SimpleNamespace(zero_grad=lambda set_to_none=True: None)

    def boom(cmd):
        raise RuntimeError("CUDA out of memory (stand-in)")

    me.op_accumulate = boom
    monkeypatch.setitem(rt.PROGRESS, "windows", 41)
    out = rt.Trainer.run_op(me, {"op": "accumulate"}, 0)
    assert not out["ok"] and "boom" in out["traceback"]
    assert out["failed_at"]["progress"]["windows"] == 41 and "budget_gib" in out["failed_at"]["ckpt"]
    assert "boom" in events[-1]["traceback"] and events[-1]["failed_at"]["progress"]["windows"] == 41
