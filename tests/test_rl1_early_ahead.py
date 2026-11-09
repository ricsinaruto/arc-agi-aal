"""trainer_fast.early_ahead (rl1_trainer.Trainer.early_ahead): the loop's ahead-early.json is expanded and every
window submitted to the encoders, keyed as the next accumulate takes them (D2 27 Oct: 22.8 s first-window wait)."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import rl1_encode  # noqa: E402
import rl1_trainer  # noqa: E402


class Pool:
    def __init__(self):
        self.sent = []

    def submit(self, fn, *args):
        self.sent.append((fn, args))
        return ("future", args[0]["name"])


def trainer(tmp_path):
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.dir, t.early, t.ahead_plan = tmp_path, {"token": None}, None
    t.pool = Pool()
    t.encoder_pool = lambda procs: t.pool
    return t


def write(tmp_path, token, specs, procs=2):
    (tmp_path / "ahead-early.json").write_text(json.dumps({"token": token, "ahead": specs, "encode_procs": procs,
                                                           "signed": False, "share_prefix": False}))


def test_early_ahead_encodes_the_files_windows_once_per_token(tmp_path):
    t = trainer(tmp_path)
    t.early_ahead()  # no file yet: nothing
    assert t.pool.sent == [] and t.early["token"] is None
    roll = tmp_path / "rollout.jsonl"
    roll.write_text("{}\n")
    specs = [{"name": n, "w": w, "rollout": str(roll)} for n, w in (("a", 1.0), ("b", 0.0), ("c", 0.5))]  # b: untrained
    write(tmp_path, "s0001-1", specs)
    t.early_ahead()
    assert [args[0]["name"] for fn, args in t.pool.sent] == ["a", "c"]
    assert all(fn is rl1_encode.window and args[1] == 0 for fn, args in t.pool.sent)
    assert set(t.ahead_windows) == {rl1_encode.spec_key(specs[0]), rl1_encode.spec_key(specs[2])}
    assert t.early["state"] == "done" and t.early["windows"] == 2 and t.ahead_plan is None
    t.early_ahead()  # the same token: not again
    assert len(t.pool.sent) == 2
    d = {"name": "d", "w": 1.0, "rollout": str(roll)}
    write(tmp_path, "s0002-1", [d])
    t.early_ahead()
    assert len(t.pool.sent) == 3 and rl1_encode.spec_key(d) in t.ahead_windows
    e = {"name": "e", "w": 1.0, "rollout": str(roll)}
    write(tmp_path, "s0002-a2", [d, e])  # rl_seq one_call: a cumulative file; only the new window is encoded
    t.early_ahead()
    assert [args[0]["name"] for fn, args in t.pool.sent[3:]] == ["e"] and t.ahead_windows[rl1_encode.spec_key(d)]


def test_early_ahead_without_encoders_or_specs_does_nothing(tmp_path):
    t = trainer(tmp_path)
    write(tmp_path, "s0001-1", [{"name": "a", "w": 1.0, "rollout": "x"}], procs=0)
    t.early_ahead()
    assert t.pool.sent == [] and t.early["state"] == "done" and "why" in t.early
    rl1_trainer.EARLY["hook"] = lambda: 1 / 0
    rl1_trainer.early_poll()  # a failing hook never raises
    rl1_trainer.EARLY["hook"] = None


def test_profiled_picks_only_the_named_calls(monkeypatch, tmp_path):
    """RL1_PROFILE_OPS: "step,accumulate:2" profiles every step and only the second accumulate; others run bare."""
    monkeypatch.setenv("RL1_PROFILE_OPS", "step,accumulate:2")
    monkeypatch.setattr(rl1_trainer, "PROFILE_SEEN", {})
    hits = []

    class Prof:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            hits.append(True)
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self):
            return type("T", (), {"table": lambda self, **kw: "table"})()

    import torch.profiler

    monkeypatch.setattr(torch.profiler, "profile", Prof)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    for op in ("accumulate", "accumulate", "accumulate", "step", "park"):
        with rl1_trainer.profiled(op) as out:
            pass
        assert bool(out) == (op == "step" or (op == "accumulate" and rl1_trainer.PROFILE_SEEN[op] == 2)), (op, out)
    assert len(hits) == 2


def test_profile_tables_never_fail_the_op(monkeypatch, tmp_path):
    """U1d v1: with_stack's key_averages raised UnicodeDecodeError after the call and failed the accumulate; the
    tables and stacks are best-effort (errors recorded), and the sync stacks come from raw events, a bad one skipped."""
    monkeypatch.setenv("RL1_PROFILE_OPS", "accumulate")
    monkeypatch.setenv("RL1_PROFILE_STACK", "1")
    monkeypatch.setattr(rl1_trainer, "PROFILE_SEEN", {})

    class Ev:
        def __init__(self, name, stack):
            self.name, self.stack, self.self_cpu_time_total = name, stack, 2000.0

    class Bad:
        name = "aten::item"

        @property
        def stack(self):
            raise UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")

    class Prof:
        def __init__(self, **kw):
            assert kw.get("with_stack") is True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self, **kw):
            raise UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")

        def events(self):
            return [Ev("aten::item", ["a.py(1): f", "b.py(2): g"]), Ev("aten::item", ["a.py(1): f", "b.py(2): g"]),
                    Ev("aten::mm", []), Bad()]

    import torch.profiler

    monkeypatch.setattr(torch.profiler, "profile", Prof)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    with rl1_trainer.profiled("accumulate") as prof:
        pass
    text = open(prof["path"]).read()
    assert "aten::item calls=2 self_cpu_ms=4.0 :: a.py(1): f < b.py(2): g" in text
    assert any("tables" in e for e in prof["errors"]) and "stack events skipped: 1" in prof["errors"]


def test_profile_trace_is_dumped_raw_and_gzipped(monkeypatch, tmp_path):
    """U3: RL1_PROFILE_TRACE writes the profiled call's raw timeline (chrome trace) as .trace.json.gz beside the tables."""
    import gzip
    import json

    monkeypatch.setenv("RL1_PROFILE_OPS", "accumulate:2")
    monkeypatch.setenv("RL1_PROFILE_TRACE", "1")
    monkeypatch.setattr(rl1_trainer, "PROFILE_SEEN", {})

    class Prof:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self):
            return type("T", (), {"table": lambda self, **kw: "table"})()

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps({"traceEvents": [{"name": "k", "ph": "X"}]}))

    import torch.profiler

    monkeypatch.setattr(torch.profiler, "profile", Prof)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    for _ in range(2):
        with rl1_trainer.profiled("accumulate") as out:
            pass
    assert out["trace"].endswith("accumulate-2-rank0.trace.json.gz") and "errors" not in out
    assert json.loads(gzip.open(out["trace"]).read())["traceEvents"][0]["name"] == "k"
    assert not (tmp_path / "profiles" / "accumulate-2-rank0.trace.json").exists()


class Done:
    def __init__(self, value):
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value


def test_early_ahead_keys_match_the_accumulate_under_whole_segment(tmp_path):
    """gate-v1/v2 (28 Oct): ahead_hits 0 for 192 windows. The call expands whole_segment(attempt) (expand_cmd); the
    ahead plan expanded the bare attempt, so every key differed. Both now plan the same windows."""
    from types import SimpleNamespace

    t = trainer(tmp_path)
    t.args = SimpleNamespace(window=4096)

    class Expanding(Pool):
        def submit(self, fn, *args):
            if fn is rl1_encode.expand_one:
                spec = args[0]
                return Done([{**spec, "name": spec["name"] + "#0", "whole": False}])
            return super().submit(fn, *args)
    t.pool = Expanding()
    roll = tmp_path / "rollout.jsonl"
    roll.write_text("{}\n")
    attempts = [{"name": n, "w": 1.0, "rollout": str(roll), "whole": True} for n in ("a", "b")]
    (tmp_path / "ahead-early.json").write_text(json.dumps({"token": "s0002-e2", "ahead": attempts, "encode_procs": 2,
                                                           "signed": False, "share_prefix": False,
                                                           "whole_segment": True}))
    t.early_ahead()
    want = {rl1_encode.spec_key(s) for s in t.expand_cmd({"attempts": attempts, "whole_segment": True,
                                                         "encode_procs": 2})}
    assert want and set(t.ahead_windows) == want
