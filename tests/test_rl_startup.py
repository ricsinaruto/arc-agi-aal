"""S2 resume/startup paths of the mg loop: early play on the checkpoint's adapter, the resume skips (park check, own
state copy) and the trainer load beside server 2's boot. The mg loop runs on stand-ins (trainer, server, play)."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from arena import level_bench, rl_loop
from arena.fake_trainer import adapter

REAL_TIME = {"test_overlap_starts_the_trainer_with_the_boot_and_gates_its_card_use",  # a real trainer subprocess
             "test_loop_lag_catches_a_blocking_call"}  # measures a real blocking call


@pytest.fixture(autouse=True)
def virtual_time(monkeypatch, request):
    """Every loop here runs on virtual time (tests/vclock.py): event order follows the fakes' delays, never CPU speed
    (25 Oct: [True] variants flaked under CI load). Tests that need the real clock are listed in REAL_TIME."""
    if request.node.originalname in REAL_TIME:
        yield
        return
    import vclock

    old = vclock.vclock(monkeypatch, vclock.fixture_modules())
    yield
    asyncio.set_event_loop_policy(old)



class Feed:
    def __init__(self, log):
        self.events, self.log = [], log

    def emit(self, kind, **data):
        self.events.append((kind, data))
        self.log.append(("feed", kind))

    def kinds(self):
        return [k for k, _ in self.events]


class Trainer:
    def __init__(self, log, kl=0.02, delay=0.01):
        self.log, self.kl, self.delay, self.ops = log, kl, delay, []

    async def call(self, op, timeout=3600, **payload):
        await asyncio.sleep(self.delay)
        self.ops.append((op, payload))
        self.log.append(("trainer", op))
        if op == "save":
            adapter(payload["path"])
            return {"ok": True, "path": payload["path"]}
        if op == "score":  # as the real trainer: kl_base only once a base reference was scored or imported
            if payload.get("label") == "base":
                self.based = True
            return {"ok": True, "windows": {"p1": {"kl_base": self.kl} if getattr(self, "based", False) else {}}}
        if op == "base_import":
            self.based = True
        if op == "checkpoint":
            Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            (Path(payload["path"]) / "state.json").write_text("{}")
            return {"ok": True, "path": payload["path"]}
        if op == "park_check":
            return {"ok": True, "identical": True}
        return {"ok": True, "updates": 5}

    async def must(self, op, timeout=3600, **payload):
        return await self.call(op, timeout=timeout, **payload)


class Server:
    model = "base"

    def __init__(self, log):
        self.log = log

    async def load(self, name, path):
        self.log.append(("server.load", str(path)))
        return 0.1


def source_run(root, kl=0.02, with_adapter=True, park=True):
    """An earlier run's dir: mg_state.json at round 5, its state r05 (with the served adapter) and feed."""
    ckpt = root / "state" / "r05"
    ckpt.mkdir(parents=True)
    (ckpt / "state.json").write_text("{}")
    (ckpt / "mg_state.json").write_text("{}")
    if with_adapter:
        adapter(ckpt / "adapter")
    (root / "mg_state.json").write_text(json.dumps({"round": 5, "played_by": 5, "checkpoint": str(ckpt),
                                                    "policy_uid": "uid5", "lr": 6e-5}))
    feed = root / "arena" / "trials" / "rl-loop" / "rl"
    feed.mkdir(parents=True)
    rows = [{"kind": "round", "round": 5, "accepted_updates": 1, "update": {"probes": {"p1": {"kl_base": kl}}}}]
    if park:
        rows.insert(0, {"kind": "park_check", "identical": True, "epoch": 1.0})
    (feed / "feed.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root


def make_loop(tmp_path, monkeypatch, log, *, mg=None, kl=0.02, load_delay=0.2, play_delay=0.3, source=None):
    from arena import rl_single

    real = asyncio.sleep  # the time share's 30 s queue poll and the 120 s ticker, shortened
    monkeypatch.setattr(rl_single, "asyncio", type("A", (), {**{k: getattr(asyncio, k) for k in dir(asyncio)
                                                                 if not k.startswith("__")},
                                                              "sleep": staticmethod(lambda s: real(min(s, 0.05)))}))
    out = tmp_path / "out"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    monkeypatch.delenv("ARENA_KNOBS", raising=False)
    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    monkeypatch.delenv("ARC_PREFLIGHT_STUB", raising=False)
    monkeypatch.delenv("ARC_RL1_MAX_ROUNDS", raising=False)
    monkeypatch.setattr(level_bench, "TRY_KEEP", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    src = source or source_run(tmp_path / "src")
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = rl_loop.settings({"mode": "loop"})
    loop.cfg["mg"] = {"sets": [{"name": "tr", "role": "train", "tries": 2}, {"name": "te", "role": "test", "tries": 2}],
                      "resume": str(src), "checkpoint": True, "rollback": True, "max_rounds": 6, "pack_rounds": False,
                      "try_keep": True, **(mg or {})}
    loop.study, loop.trial = {}, {}
    loop.feed, loop.server, loop.trainer = Feed(log), Server(log), None
    loop.work = tmp_path / "work"
    loop.adapters = loop.work / "adapters"
    loop.output = out / "arena" / "trials" / "rl-loop"
    (loop.output / "rl").mkdir(parents=True)
    loop.history, loop.deadline, loop.started = [], time.time() + 10 ** 6, time.time()
    loop.serving_adapter = None

    async def start_trainer(gate=None):
        log.append(("trainer_start",))
        await asyncio.sleep(load_delay)
        loop.trainer = Trainer(log, kl=kl)
        log.append(("trainer_ready",))
        return {"seconds": load_delay}

    async def play(name, spec, starts, tries, *, model, seed, **kw):
        log.append(("play_start", name, model))
        if level_bench.TRY_KEEP is not None:  # a finished try kept for reuse (mg.try_keep)
            keep = Path(level_bench.TRY_KEEP) / name
            (keep / "attempts" / "a1").mkdir(parents=True, exist_ok=True)
            (keep / "attempts" / "a1" / "rollout.jsonl").write_text("{}\n")
            (keep / "tries.jsonl").write_text(json.dumps({"label": "a1", "outcome": "cleared"}) + "\n")
        folder = loop.output / "rl" / name
        folder.mkdir(parents=True, exist_ok=True)
        await asyncio.sleep(play_delay)
        log.append(("play_end", name))
        return folder, [], {"seconds": play_delay, "attempts": 0, "usable": 0, "tokens": 0}

    loop.start_trainer = start_trainer
    loop.mg_play = play
    loop.mg_screen_file = lambda: {"sets": {n: {"folder": str(tmp_path / "scr" / n), "starts": []} for n in ("tr", "te")}}
    loop.mg_rows = lambda folder: []
    loop.mg_matched = lambda spec, entry, points: [("s1",), ("s2",), ("s3",)]
    loop.mg_pick = lambda number, name, s, pool: (["s1"], 2)
    loop.mg_ok_starts = lambda name, screen: ["t1"]
    loop.mg_tests = lambda results, screen: {}
    loop.mg_select = lambda folder, rows, name=None: ([], {})
    for name in ("mg_guard_stop", "mg_rules", "mg_rule_f", "mg_success"):
        setattr(loop, name, lambda *a, **k: None)
    return loop


def trainer_ops(log):
    return [e[1] for e in log if e[0] == "trainer"]


def first(log, entry):
    return next(i for i, e in enumerate(log) if e[:len(entry)] == entry)


# --- keys off: today's sequence -----------------------------------------------------------------------------------

def test_keys_off_resume_runs_todays_sequence(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log)
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    resume = str(loop.work / "adapters" / "resume-r05")
    steps = [e for e in log if e[0] in ("trainer_start", "trainer_ready", "trainer", "server.load", "play_start")]
    assert steps == [("trainer_start",), ("trainer_ready",), ("trainer", "score"), ("trainer", "load_checkpoint"),
                     ("trainer", "save"), ("server.load", resume), ("trainer", "score"), ("trainer", "checkpoint"),
                     ("play_start", "r06-tr", "policy"), ("play_start", "r06-te", "policy")]
    kinds = loop.feed.kinds()
    assert kinds.index("state_check") < kinds.index("checkpoint") < kinds.index("resume") < kinds.index("round")
    assert not {"early_play", "early_play_verified", "checkpoint_copy", "early_play_discard"} & set(kinds)
    own = tmp_path / "out" / "state" / "r05"
    assert (own / "state.json").is_file() and not (own / "adapter").exists()  # no adapter save with the key off
    state = json.loads((tmp_path / "out" / "mg_state.json").read_text())
    assert state["round"] == 6 and state["checkpoint"] == str(own)


def test_keys_off_park_check_runs_in_start_before_the_state_load(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"timeshare": {"min_queued": 1}})
    monkeypatch.setattr(loop, "ts_boot", lambda: _noop(log, "ts_boot"), raising=False)
    monkeypatch.setattr(loop, "ts_on", lambda *a: _noop(log, "ts_on"), raising=False)
    monkeypatch.setattr(loop, "ts_off", lambda *a: _noop(log, "ts_off"), raising=False)
    level_bench.SHARED_SLOTS2 = None
    asyncio.run(loop.mg_loop())
    ops = trainer_ops(log)
    assert ops[:4] == ["score", "park_check", "load_checkpoint", "save"] and "checkpoint" in ops
    assert first(log, ("ts_boot",)) < first(log, ("trainer_start",))  # server 2 boots, then the trainer loads


async def _noop(log, name):
    log.append((name,))
    return {}


# --- early play ------------------------------------------------------------------------------------------------------

def test_early_play_starts_before_the_trainer_and_trains_only_after_the_gate(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"early_play": True})
    asyncio.run(loop.mg_loop())
    early = str(loop.work / "adapters" / "resume-r05")
    assert [e for e in log if e[0] == "server.load"] == [("server.load", early)]  # served once, before play
    assert first(log, ("server.load",)) < first(log, ("play_start", "r06-tr")) < first(log, ("trainer_start",))
    assert first(log, ("trainer", "load_checkpoint")) < first(log, ("play_end", "r06-tr"))  # overlapped
    assert trainer_ops(log) == ["score", "load_checkpoint", "save", "score", "checkpoint", "save"]
    assert loop.trainer.ops[2][1]["path"].endswith("resume-r05-check")  # the identity check's own save
    kinds = loop.feed.kinds()
    assert kinds.index("early_play") < kinds.index("state_check") < kinds.index("early_play_check") \
        < kinds.index("early_play_verified") < kinds.index("round")
    assert dict(loop.feed.events)["early_play_check"]["ok"] and loop.serving_adapter == early
    assert loop.policy_uid == "uid5"  # the restored state's uid: finished-try reuse keys match the source's
    own = tmp_path / "out" / "state" / "r05"
    assert (own / "adapter" / "adapter_model.safetensors").is_file()  # the next resume can play early too


def test_early_play_falls_back_to_todays_order_without_a_saved_adapter(tmp_path, monkeypatch):
    log = []
    src = source_run(tmp_path / "src", with_adapter=False)
    loop = make_loop(tmp_path, monkeypatch, log, mg={"early_play": True}, source=src)
    asyncio.run(loop.mg_loop())
    assert dict(loop.feed.events)["early_play"]["on"] is False
    assert first(log, ("trainer", "checkpoint")) < first(log, ("play_start", "r06-tr"))


@pytest.mark.parametrize("why, mg, knobs", [
    ("single_card", {"single_card": True}, None),
    ("carry", {"carry": True}, None),
    ("rollback knob pending", {}, {"rollback": {"to": "r04"}}),
])
def test_early_play_refuses_paths_that_need_the_trainer_first(tmp_path, monkeypatch, why, mg, knobs):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"early_play": True, **mg})
    state = json.loads((tmp_path / "src" / "mg_state.json").read_text())
    if why == "carry":
        state["played_by"] = 4
    if knobs:
        path = tmp_path / "knobs.json"
        path.write_text(json.dumps(knobs))
        monkeypatch.setenv("ARENA_KNOBS", str(path))
    assert asyncio.run(loop.mg_early_serve(state, state["checkpoint"])) is None
    row = dict(loop.feed.events)["early_play"]
    assert row["on"] is False and row["why"].startswith(why) and log == [("feed", "early_play")]


def test_rollback_knob_after_a_later_round_does_not_block_early_play(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"early_play": True})
    path = tmp_path / "knobs.json"
    path.write_text(json.dumps({"rollback": {"to": "r04", "after_round": 6}}))
    monkeypatch.setenv("ARENA_KNOBS", str(path))
    state = json.loads((tmp_path / "src" / "mg_state.json").read_text())
    assert loop.mg_rollback_pending(6, state) is False and loop.mg_rollback_pending(7, state) is True
    state["knobs_seen"] = {"rollback": {"to": "r04", "after_round": 6}}
    assert loop.mg_rollback_pending(7, state) is False  # applied before


def failed_gate(tmp_path, monkeypatch, **kw):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"early_play": True}, play_delay=5.0, **kw)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(loop.mg_loop())
    return loop, log, error


def test_failed_state_check_discards_the_early_tries(tmp_path, monkeypatch):
    loop, log, error = failed_gate(tmp_path, monkeypatch, kl=0.05)
    assert "state check" in str(error.value)
    kinds = loop.feed.kinds()
    assert "early_play_verified" not in kinds and "round" not in kinds and "early_play_discard" in kinds
    assert not any(e[:2] == ("play_end", "r06-tr") for e in log)  # play stopped, not waited out
    row = dict(loop.feed.events)["early_play_discard"]
    partial = tmp_path / "out" / "partial"
    keys = [p.name for p in partial.iterdir()]
    assert len(keys) == 1 and ".unusable-" in keys[0] and row["keep"] == str(partial / keys[0])
    assert level_bench.kept_tries("r06-tr", [partial / "p05-uid5"]) == {}  # a restart reuses none of them
    assert sorted(row["benches"]) == ["r06-te", "r06-tr"]
    assert all((loop.output / "rl" / b / "UNUSABLE.json").is_file() for b in row["benches"])
    assert not (tmp_path / "out" / "mg_state.json").exists()  # no resume point from the unverified round
    assert "checkpoint" not in trainer_ops(log)


def test_adapter_mismatch_fails_the_gate_and_discards(tmp_path, monkeypatch):
    src = source_run(tmp_path / "src")
    path = src / "state" / "r05" / "adapter" / "adapter_model.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-1] = 7  # one tensor byte differs
    path.write_bytes(bytes(raw))
    loop, log, error = failed_gate(tmp_path, monkeypatch, source=src)
    assert "differs" in str(error.value)
    assert dict(loop.feed.events)["early_play_check"]["tensors_same"] is False
    assert "early_play_discard" in loop.feed.kinds() and "round" not in loop.feed.kinds()


def test_early_play_with_time_share_turns_card_2_on_after_the_gate(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"early_play": True, "timeshare": {"min_queued": 1, "park_check": False}})
    monkeypatch.setattr(loop, "ts_boot", lambda: _noop(log, "ts_boot"), raising=False)
    monkeypatch.setattr(loop, "ts_on", lambda *a: _noop(log, "ts_on"), raising=False)
    monkeypatch.setattr(loop, "ts_off", lambda *a: _noop(log, "ts_off"), raising=False)
    asyncio.run(loop.mg_loop())
    assert first(log, ("play_start", "r06-tr")) < first(log, ("ts_boot",)) < first(log, ("trainer_start",))
    assert first(log, ("feed", "early_play_verified")) < first(log, ("ts_on",)) < first(log, ("play_end", "r06-tr"))


# --- resume skips ------------------------------------------------------------------------------------------------------

def test_resume_skip_copies_the_verified_state_and_skips_the_park_check(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"resume_skip": True, "timeshare": {"min_queued": 1}})
    for name in ("ts_boot", "ts_on", "ts_off"):
        monkeypatch.setattr(loop, name, (lambda n: lambda *a: _noop(log, n))(name), raising=False)
    asyncio.run(loop.mg_loop())
    ops = trainer_ops(log)
    assert "park_check" not in ops and "checkpoint" not in ops
    rows = dict(loop.feed.events)
    assert rows["park_check"]["skipped"] and rows["park_check"]["why"] == "resume_skip"
    own = tmp_path / "out" / "state" / "r05"
    assert rows["checkpoint_copy"]["gib"] >= 0 and (own / "state.json").is_file() and (own / "adapter").is_dir()
    snap = json.loads((own / "mg_state.json").read_text())  # this run's snapshot, not the source's copy
    assert snap["checkpoint"] == str(own) and snap["policy_uid"] == "uid5"
    assert json.loads((tmp_path / "out" / "mg_state.json").read_text())["checkpoint"] == str(own)


def test_resume_skip_runs_the_checks_when_the_state_is_not_verified(tmp_path, monkeypatch):
    log = []
    src = source_run(tmp_path / "src")
    feed = src / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
    feed.write_text(json.dumps({"kind": "park_check", "identical": True}) + "\n")  # no recorded probes: check skipped
    loop = make_loop(tmp_path, monkeypatch, log, mg={"resume_skip": True, "timeshare": {"min_queued": 1}}, source=src)
    for name in ("ts_boot", "ts_on", "ts_off"):
        monkeypatch.setattr(loop, name, (lambda n: lambda *a: _noop(log, n))(name), raising=False)
    asyncio.run(loop.mg_loop())
    ops = trainer_ops(log)
    assert ops.index("load_checkpoint") < ops.index("park_check") and "checkpoint" in ops
    assert "checkpoint_copy" not in loop.feed.kinds()


def test_resume_skip_needs_a_passed_park_check_in_the_resumed_run(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, monkeypatch, [])
    src = tmp_path / "src"
    feed = src / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
    assert loop.mg_source_park(src)["identical"] is True
    feed.write_text(json.dumps({"kind": "park_check", "identical": True}) + "\n"
                    + json.dumps({"kind": "park_check", "identical": False}) + "\n")
    assert loop.mg_source_park(src) is None  # the last check failed
    feed.write_text(json.dumps({"kind": "park_check", "skipped": True, "why": "resume_skip"}) + "\n")
    assert loop.mg_source_park(src)["skipped"] is True  # a chain of resumes
    assert loop.mg_source_park(tmp_path / "nowhere") is None

    loop.trainer = Trainer([])
    loop.mg_park_deferred = None
    feed.write_text("")
    assert asyncio.run(loop.mg_resume_park([], verified=True))["identical"]  # no source pass: runs
    loop.mg_park_deferred = None
    feed.write_text(json.dumps({"kind": "park_check", "identical": True}) + "\n")
    assert asyncio.run(loop.mg_resume_park([], verified=False))["identical"]  # unverified state: runs
    assert asyncio.run(loop.mg_resume_park([], verified=True)) is None  # nothing deferred: no-op
    assert [op for op, _ in loop.trainer.ops] == ["park_check", "park_check"]


def test_a_failed_state_copy_falls_back_to_the_trainer_checkpoint(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, monkeypatch, [])
    loop.trainer, loop.policy_uid = Trainer([]), "u"
    own = tmp_path / "out" / "state" / "r05"
    row = asyncio.run(loop.mg_copy_state(tmp_path / "missing", own, {"round": 5}, 5))
    assert "error" in row and [op for op, _ in loop.trainer.ops] == ["checkpoint"]
    assert json.loads((own / "mg_state.json").read_text())["checkpoint"] == str(own)


# --- trainer load beside server 2's boot ------------------------------------------------------------------------------

def test_overlap_starts_the_trainer_with_the_boot_and_gates_its_card_use(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_PREFLIGHT_FAKE_TRAINER", "1")
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = rl_loop.settings({"mode": "loop"})
    loop.cfg.update(trainer_python=sys.executable, trainer_timeout=60)
    loop.source, loop.work, loop.output = Path(rl_loop.__file__).resolve().parents[2], tmp_path / "w", tmp_path / "o"
    (loop.output / "rl").mkdir(parents=True)
    loop.gpu = type("G", (), {"mark": lambda self, phase: None})()
    alive = loop.work / "trainer" / "alive.json"
    seen = {}

    async def boot():
        await asyncio.sleep(2.0)  # the trainer process is up (its CPU part) while server 2 boots
        seen["process_up"] = loop.trainer.process.poll() is None
        seen["ready_during_boot"] = alive.exists()

    trainer = asyncio.run(loop.mg_overlap_boot(boot))
    try:
        assert seen == {"process_up": True, "ready_during_boot": False}  # its card part waited for the gate
        assert trainer["loaded"]["gate_waited_s"] >= 1.0 and trainer["overlap"]["boot_s"] >= 2.0
        assert (loop.work / "trainer-gpu-gate").is_file()
    finally:
        loop.trainer.close()


def test_overlap_gate_opens_when_server_2_fails_to_boot(tmp_path, monkeypatch):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.work = tmp_path / "w"
    calls = []

    async def start_trainer(gate=None):
        calls.append(("start", str(gate)))
        while not Path(gate).exists():
            await asyncio.sleep(0.05)
        return {"seconds": 0.1}

    async def boot():
        await asyncio.sleep(0.2)
        loop.ts_dead = True  # boot2's handler: server 2 stopped, the run goes on with one play server

    loop.start_trainer = start_trainer
    out = asyncio.run(loop.mg_overlap_boot(boot))
    assert calls and json.loads((loop.work / "trainer-gpu-gate").read_text())["server2_dead"] is True
    assert out["overlap"]["boot_s"] >= 0.2


def test_trainer_gpu_gate_holds_the_first_cuda_init(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(rl_loop.__file__).resolve().parents[2] / "scripts" / "posttrain"))
    import rl1_trainer
    import threading
    import types

    gate = tmp_path / "gate"
    inits = []
    cuda = types.SimpleNamespace(_lazy_init=lambda: inits.append(time.time()), is_initialized=lambda: bool(inits))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr("builtins.__import__", _imports_without(("transformers", "peft", "axolotl.cli.config",
                                                                  "axolotl.train")))
    monkeypatch.setenv("RL1_GPU_GATE", str(gate))
    state = rl1_trainer.gpu_gate()
    worker = threading.Thread(target=cuda._lazy_init)
    worker.start()
    time.sleep(1.5)
    assert worker.is_alive() and inits == []  # waiting: no card use before the gate
    gate.write_text("{}")
    worker.join(5)
    assert not worker.is_alive() and len(inits) == 1 and state["waited_s"] >= 1.0
    cuda._lazy_init()  # once open, straight through
    monkeypatch.delenv("RL1_GPU_GATE")
    assert rl1_trainer.gpu_gate() is None


def _imports_without(names):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name in names:
            raise ImportError(name)
        return real(name, *a, **k)

    return fake


def test_start_trainer_passes_the_gate_only_when_asked(tmp_path, monkeypatch):
    captured = []

    class FakeTrainerProc:
        def __init__(self, folder, command, env, log_path):
            captured.append(env)

        async def ready(self, timeout):
            return {"loaded": {}}

    monkeypatch.setattr(rl_loop, "Trainer", FakeTrainerProc)
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = rl_loop.settings({"mode": "loop"})
    loop.source, loop.work, loop.output = tmp_path, tmp_path / "w", tmp_path / "o"
    loop.gpu = type("G", (), {"mark": lambda self, phase: None})()
    monkeypatch.delenv("RL1_GPU_GATE", raising=False)
    asyncio.run(loop.start_trainer())
    asyncio.run(loop.start_trainer(gate=tmp_path / "g"))
    assert "RL1_GPU_GATE" not in captured[0] and captured[1]["RL1_GPU_GATE"] == str(tmp_path / "g")


def test_new_keys_are_known_and_default_off():
    assert {"early_play", "resume_skip", "trainer_overlap"} <= rl_loop.MG_KEYS
    src = (Path(rl_loop.__file__).with_name("rl_single.py")).read_text()
    for key in ("early_play", "resume_skip", "trainer_overlap"):
        assert f'mg.get("{key}")' in src and f'"{key}": True' not in src


# --- mg.resume_train_pending ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("early", [False, True])
def test_pending_round_trains_first_while_the_next_round_plays(tmp_path, monkeypatch, early):
    log = []
    # virtual time (tests/vclock.py): the early-play trainer load + first update fit inside the default 0.3 s play
    loop = make_loop(tmp_path, monkeypatch, log, mg={"resume_train_pending": True, "max_rounds": 8,
                                                     **({"early_play": True} if early else {})})
    data = [("f", {"label": "a"}, 1.0, "k", "tr")]
    loop.mg_pending_round = lambda state, sets: {"round": 6, "played_by": 5, "data": data, "info": {}}
    seen = []

    async def update(number, new, probes):
        log.append(("update", number))
        seen.append((number, new))
        await asyncio.sleep(0.1)
        return {"ok": True}, str(loop.work / "adapters" / f"u{number}")

    async def after(number, row, update_row, serve, load=True):
        log.append(("after", number))
        return None, True

    loop.mg_update, loop.mg_after_update = update, after
    asyncio.run(loop.mg_loop())
    assert seen[0] == (7, data)  # round 6 (the pending one) trains in round 7's iteration
    plays = [e for e in log if e[0] == "play_start"]
    assert plays[0] == ("play_start", "r07-tr", "policy")  # no replay of round 6
    assert first(log, ("trainer_ready",)) < first(log, ("update", 7)) < first(log, ("play_end", "r07-tr"))
    assert first(log, ("after", 7)) < first(log, ("play_start", "r08-tr"))
    rows = [d for k, d in loop.feed.events if k == "round"]
    assert rows[0]["round"] == 7 and rows[0]["played_by"] == 5 and rows[1]["played_by"] == 7


def test_pending_round_fallback_keeps_todays_first_round(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"resume_train_pending": True})
    loop.mg_pending_round = lambda state, sets: None
    asyncio.run(loop.mg_loop())
    assert [e for e in log if e[0] == "play_start"][0] == ("play_start", "r06-tr", "policy")


# --- mg.timeshare.lazy_boot (launch -> first update) ---------------------------------------------------

def test_lazy_boot_loads_the_trainer_at_the_loop_top_and_never_boots_server_2_at_start(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log,
                     mg={"early_play": True, "trainer_overlap": True,
                         "timeshare": {"min_queued": 1, "park_check": False, "lazy_boot": True}})
    monkeypatch.setattr(loop, "ts_boot", lambda **k: _noop(log, "ts_boot"), raising=False)
    monkeypatch.setattr(loop, "ts_on", lambda *a: _noop(log, "ts_on"), raising=False)
    monkeypatch.setattr(loop, "ts_off", lambda *a: _noop(log, "ts_off"), raising=False)
    screen = loop.mg_screen_file

    def slow_screen():  # the loop's synchronous setup (r4: ~150 s before its first await)
        log.append(("screen",))
        time.sleep(0.2)
        return screen()

    loop.mg_screen_file = slow_screen
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    assert ("ts_boot",) not in log and log.count(("trainer_start",)) == 1
    assert first(log, ("trainer_start",)) < first(log, ("screen",))  # spawned before the setup, not at its first await
    assert first(log, ("trainer_start",)) < first(log, ("server.load",)) < first(log, ("play_start", "r06-tr"))
    assert first(log, ("feed", "early_play_verified")) < first(log, ("ts_on",))


def _ts_loop(tmp_path, monkeypatch, log, free=265.0):
    from arena import rl_single

    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = rl_loop.settings({"mode": "loop"})
    loop.cfg["mg"] = {"timeshare": {"lazy_boot": True, "min_queued": 1, "gpu_util": 0.88}, "sets": []}
    loop.feed, loop.trainer = Feed(log), Trainer(log)
    monkeypatch.setattr(rl_loop, "card_memory_of", lambda card: (268.0 - free, 268.0))
    pool = level_bench.SharedSlots(8, None)
    pool.pending = {"b": (0, 5)}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)

    class S2:
        card, memory = 1, {}

        async def wake(self):
            log.append(("wake",))
            return 0.1

        async def serve(self, name, path):
            log.append(("serve", name))
            return 0.2

        async def sleep(self):
            log.append(("sleep",))
            return 0.3

    async def ts_boot(sleep=True):
        log.append(("ts_boot", sleep))
        loop.server2 = S2()
        level_bench.SHARED_SLOTS2 = level_bench.SharedSlots(8, None)
        return {"boot_seconds": 5.0}

    loop.ts_boot = ts_boot
    assert rl_single  # the mixin is part of Loop
    return loop


def _on(loop, number):
    async def go():
        busy = asyncio.get_running_loop().create_future()  # the round's play still running
        try:
            return await loop.ts_on(number, [busy], "/a")
        finally:
            busy.cancel()
    return asyncio.run(go())


def test_lazy_boot_boots_server_2_awake_at_the_first_time_share(tmp_path, monkeypatch):
    log = []
    loop = _ts_loop(tmp_path, monkeypatch, log)
    assert asyncio.run(loop.ts_off()) == {"served": 0, "booted": False}  # before any boot: nothing to sleep
    info = _on(loop, 7)
    assert [e for e in log if e[0] != "feed"][:4] == [("trainer", "park"), ("ts_boot", False), ("wake",), ("serve", "policy")]
    assert info["boot_seconds"] == 5.0 and info["need_gib"] == round(0.88 * 268 + 1.0, 2)
    assert level_bench.SHARED_SLOTS2.active
    _on(loop, 8)  # booted: later time shares wake only
    assert log.count(("ts_boot", False)) == 1


def test_lazy_boot_skips_when_the_parked_card_lacks_room(tmp_path, monkeypatch):
    log = []
    loop = _ts_loop(tmp_path, monkeypatch, log, free=200.0)
    info = _on(loop, 7)
    assert ("ts_boot", False) not in log and info["free_gib"] == 200.0
    assert ("feed", "timeshare_skip") in log


def test_fake_smi_follows_the_fake_trainers_park(tmp_path, monkeypatch):
    """CPU preflight realism for lazy boot: a park frees the trainer card in the stand-in nvidia-smi, an update fills it."""
    import importlib.util
    import subprocess

    from arena import fake_trainer

    spec = importlib.util.spec_from_file_location(
        "modal_rl1_src", Path(rl_loop.__file__).resolve().parents[2] / "scripts" / "posttrain" / "modal_rl1.py")
    text = Path(spec.origin).read_text()
    fake_smi = text.split('FAKE_SMI = r"""', 1)[1].split('"""', 1)[0]
    smi = tmp_path / "nvidia-smi"
    smi.write_text(fake_smi)
    monkeypatch.setenv("ARC_FAKE_SMI_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    def used():
        out = subprocess.run([sys.executable, str(smi), "-i", "1", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, env=os.environ).stdout
        return [float(v) for v in out.split(",")]

    assert used() == [250560.0, 275040.0]  # no file: the resident trainer
    fake = fake_trainer.Fake()
    fake.run("park", {})
    assert used() == [3072.0, 275040.0]
    fake.run("update", {"attempts": []})
    assert used() == [250560.0, 275040.0]


def test_kept_try_reuse_copies_off_the_event_loop():
    """r4 (22 Oct): ~170 kept-try copies from the runs volume ran inside the bench's launch loop and held all play."""
    src = (Path(level_bench.__file__)).read_text()
    body = src.split("async def reuse(i):", 1)[1].split("key = object()", 1)[0]
    assert "await asyncio.to_thread(shutil.copytree" in body and "\n            shutil.copytree" not in body
    assert "if kept and await reuse(order[next_index]):" in src


def test_loop_lag_catches_a_blocking_call():
    import asyncio, time
    from types import SimpleNamespace
    from arena import rl_single

    rows = []
    fake = SimpleNamespace(loop_lag={"tick": 0.0, "round": 0.0}, feed=SimpleNamespace(emit=lambda k, **f: rows.append((k, f))))

    async def main():
        task = asyncio.ensure_future(rl_single.MultiMixin.mg_loop_lag(fake, every=0.1, alert=0.5))
        await asyncio.sleep(0.25)
        time.sleep(0.8)  # a synchronous copy on the event loop
        await asyncio.sleep(0.25)
        task.cancel()

    asyncio.run(main())
    assert fake.loop_lag["round"] > 0.6 and rows and rows[0][0] == "loop_lag"


def test_no_instance_attribute_shadows_a_method():
    """pfK3 (22 Oct): ``self.mg_lag = {...}`` shadowed the method mg_lag and the first update crashed. Checks the
    Loop class and its mixins (one instance)."""
    import ast
    from pathlib import Path
    from arena import rl_loop

    src = Path(rl_loop.__file__).parent
    names = {"Loop", "LadderMixin", "SingleLevelMixin", "OpdMixin", "TransferMixin", "MultiMixin"}
    methods, assigned = set(), {}
    for f in ("rl_loop.py", "rl_single.py", "rl_ladder.py"):
        for cls in ast.parse((src / f).read_text()).body:
            if not (isinstance(cls, ast.ClassDef) and cls.name in names):
                continue
            for node in ast.walk(cls):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node in cls.body
                        and not any("property" in ast.dump(d) or "setter" in ast.dump(d) for d in node.decorator_list)):
                    methods.add(node.name)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
                for t in targets:
                    for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
                            assigned.setdefault(e.attr, f"{f}:{e.lineno}")
    # mg_screen: the screen-mode entry method; loop mode (never a screen) stores the screen file under the same name
    clash = {k: v for k, v in assigned.items() if k in methods and k != "mg_screen"}
    assert not clash, clash


def _reload_case(tmp_path, monkeypatch, noise, jump, merged_off=0.0):
    import asyncio, itertools
    from types import SimpleNamespace
    from arena import rl_single

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    rows, calls = [], itertools.count()
    loaded = {"done": False}

    async def must(op, **kw):
        if op == "load_checkpoint":
            loaded["done"] = True
            return {"updates": 3}
        i = next(calls)
        base = 0.05 + (jump if loaded["done"] else 0.0) + (noise if i % 2 else 0.0)
        return {"windows": {"p1": {"kl_base": base}, "p2": {"kl_base": base + 0.01}}}

    fake = SimpleNamespace(trainer=SimpleNamespace(must=must), feed=SimpleNamespace(emit=lambda k, **f: rows.append((k, f))))
    merged = {"probes": {"p1": {"kl_base": 0.05 + merged_off}, "p2": {"kl_base": 0.06 + merged_off}}}
    try:
        asyncio.run(rl_single.MultiMixin.mg_reload_check(fake, 13, merged, ["p1", "p2"]))
        return True, rows
    except RuntimeError:
        return False, rows


def test_reload_check_calibrates_on_the_measured_repeat_floor(tmp_path, monkeypatch):
    ok, rows = _reload_case(tmp_path, monkeypatch, noise=6e-4, jump=1.5e-3)  # r4-like: noisy stack, reload within 3x
    assert ok and rows[0][1]["repeat_floor"] > 5e-4
    ok, _ = _reload_case(tmp_path, monkeypatch, noise=0.0, jump=1.5e-3)  # exact stack: the same jump is a real bug
    assert not ok
    ok, _ = _reload_case(tmp_path, monkeypatch, noise=3e-3, jump=6e-3)  # a noisy floor cannot pass past the ceiling
    assert not ok
    ok, _ = _reload_case(tmp_path, monkeypatch, noise=0.0, jump=0.0, merged_off=1e-2)  # update probes off the saved state
    assert not ok
