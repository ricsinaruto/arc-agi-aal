"""The redesign loop (arena.rl_seq) on a fake loop: phases, carried tries, switch order, harvest, resume."""
import asyncio
import json
import random
from pathlib import Path

import httpx
import pytest

from arena import level_bench, phase, rl_seq

FIX = Path(__file__).parent / "fixtures" / "rl_backlog"
GAMES = ["sb26", "s5i5", "ka59", "su15"]


class Feed:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **data):
        self.rows.append({"kind": kind, **data})

    def of(self, kind):
        return [r for r in self.rows if r["kind"] == kind]


class Upstream(httpx.AsyncBaseTransport):
    """The play servers: a reply after a short delay; counts server-side aborts."""

    def __init__(self):
        self.cancelled = 0

    async def handle_async_request(self, request):
        try:
            await asyncio.sleep(0.002)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return httpx.Response(200, json={"ok": True})


class HoldOne(Upstream):
    """Upstream that holds its ``nth`` request until the gate aborts it (a phase stop), so every run has an abort to
    resend (CI 28 Oct: the timing-only version saw 0 aborts on 3 runs)."""

    def __init__(self, nth=3):
        super().__init__()
        self.n, self.nth = 0, nth

    async def handle_async_request(self, request):
        self.n += 1
        if self.n == self.nth:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return await super().handle_async_request(request)


class FakeServer:
    def __init__(self, log, root="http://s0"):
        self.log, self.root, self.model = log, root, "base"

    async def sleep(self):
        self.log.append("sleep")

    async def wake(self):
        self.log.append("wake")

    async def load(self, name, path):
        self.log.append(f"load:{name}:{Path(path).name}")

    async def unload(self, name):
        self.log.append(f"unload:{name}")


class FakeTrainer:
    def __init__(self, log):
        self.log, self.steps = log, 0

    async def must(self, op, **kw):
        self.log.append(op)
        if op == "save":
            Path(kw["path"]).mkdir(parents=True, exist_ok=True)
        if op == "checkpoint":
            Path(kw["path"]).mkdir(parents=True, exist_ok=True)
        return {"ok": True}

    async def call(self, op, **kw):
        self.log.append(op)
        if op == "step":
            self.steps += 1
            Path(kw["save"]).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "trained_tokens": 10}


class FakeLoop:
    def __init__(self, tmp, max_steps=3, seed=0, clear_p=0.3):
        self.log = []
        self.feed, self.source, self.output, self.work = Feed(), FIX, tmp / "out", tmp / "work"
        self.cfg = {"lr": 1e-4, "mg": {"redesign": {"history": "v85p_rows.json", "streams": 50, "max_steps": max_steps,
                                                   "board_check": False, "games": GAMES,
                                                   "backlog": {"phase_groups": 4}}}}
        self.server = FakeServer(self.log)
        self.base_url = "http://s0/v1"
        self.trainer = None
        self.rng, self.clear_p = random.Random(seed), clear_p
        self.accumulated = []

    async def start_trainer(self):
        self.log.append("start_trainer")
        self.trainer = FakeTrainer(self.log)

    def v7_trainer_args(self):
        return {}

    def v7_found(self, path):
        return path

    def v7_prepare(self, g):
        cache = self.v7.setdefault("prepared", {})  # as Loop.v7_prepare: a group prepared early is taken from the cache
        if g.gid in cache:
            return cache.pop(g.gid)
        self.accumulated.append(g.gid)
        if g.gid.startswith("r"):  # a replayed group: Loop.v7_specs reads its set's cap (U1 v2 N1)
            assert self.cfg["mg_sets"][g.unit]["cap"] > 0
        rows = self.v7["rows"].pop(g.gid)
        return {"gid": g.gid, "tries": len(rows), "trained": len(rows)}, [{"name": r["label"]} for r in rows]

    def mg_attempt_start(self, spec, dest):
        return 7

    async def mg_play(self, name, spec, starts, tries, *, model, seed, harvest, base_url=None):
        assert (model, harvest) in {("policy", True), ("base", False)}  # base: the control group (no harvest)
        self.__dict__.setdefault("played", []).append((name, model, len(starts), spec["cap"]))
        assert level_bench.ENGINE_POLL is False  # G-long v5 N3: benches never poll /metrics beside the guard
        self.__dict__.setdefault("urls", []).append(base_url)
        client = httpx.AsyncClient(base_url=base_url or "http://s0/v1", transport=level_bench.PHASE_GATE)
        rows = []

        async def one(i):
            for _ in range(self.rng.randint(1, 4)):  # model requests through the gate
                await client.post("chat/completions", json={"messages": [{"role": "user", "content": name}]})
            cleared = self.rng.random() < self.clear_p
            row = {"label": f"a{i:03d}", "outcome": "cleared" if cleared else "budget", "completion_tokens": 900,
                   "level_rows": [{"tokens": 800, "actions": 20, "score": 50.0}] if cleared else []}
            if cleared:
                row["harvest"] = {"request": 7, "step": 3, "state": "NOT_FINISHED"}
            rows.append(row)
            level_bench.PROGRESS_HOOK(name, row)

        await asyncio.gather(*(one(i) for i in range(tries)))
        await client.aclose()
        return self.output / "rl" / name, rows, {}


@pytest.fixture
def harvest_ok(monkeypatch):
    import arena.restore as restore

    monkeypatch.setattr(rl_seq.harvest_mod, "snapshot", lambda a, d: Path(d).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(restore, "unsafe_from", lambda d: (None, None))
    monkeypatch.setattr(restore, "load", lambda d, r: {"level": int(Path(d).name.split("-L")[1].split("-")[0]) + 1})


def test_phases_train_full_groups_and_carry_the_rest(tmp_path, harvest_ok, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path)
    upstream = HoldOne()
    gate = phase.PhaseGate(inner=upstream)
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=gate), 30))
    assert why == "max_steps" and loop.trainer.steps == 3 and level_bench.ENGINE_POLL is True
    done = loop.feed.of("group_done")  # the exporter's train.rhae_per_10k rows: every try with its fields
    assert done and all(len(r["attempts"]) == r["tries"] for r in done)
    assert all("completion_tokens" in a for r in done for a in r["attempts"])
    # start-up: servers sleep, trainer loads, initial LoRA saved and served before any play
    assert loop.log[:6] == ["sleep", "start_trainer", "save", "park", "wake", "load:policy:v0000"]
    phases = loop.feed.of("seq_phase")
    assert len(phases) == 3 and all(len(r["top"]) > 0 for r in phases)
    # groups in flight at each stop carried over; their in-flight replies were aborted and resent
    # (HoldOne's held request keeps its group in flight at the first stop; later stops depend on timing: CI 28 Oct)
    assert phases[0]["carried"] > 0 and sum(r["stop"]["aborted"] for r in phases) > 0
    assert 0 < gate.stats["resent"] <= gate.stats["aborted"]  # every resend follows an abort; teardown cuts the last
    # every switch: drain -> sleep -> unpark -> accumulate x groups -> step -> checkpoint -> park -> wake -> load
    i = loop.log.index("unpark")
    assert loop.log[i - 1] == "sleep" and "step" in loop.log[i:] and loop.log[loop.log.index("step", i) + 1] == "checkpoint"
    assert [r["step"] for r in loop.feed.of("step")] == [1, 2, 3]
    assert [r["step"] for r in loop.feed.of("seq_train")] == [0, 1, 2]  # the watcher's train-phase start (infra)
    assert loop.feed.of("step")[0]["groups"] == 4  # phase_groups full groups per step
    assert not phase.TIME.paused and level_bench.PHASE_GATE is None
    assert loop.mg_screen == {"sets": {}}  # v7_specs / mg_level_ref read it (pf R3 died without it)
    saved = [r for r in loop.feed.of("seq_phase")][-1]["stats"]["saved"]
    assert saved > 0  # cleared tries saved deeper starts
    assert (tmp_path / "vo" / "seq_state.json").is_file() and (tmp_path / "vo" / "adapters" / "v0003").is_dir()


def test_resume_drops_in_flight_groups_and_keeps_step(tmp_path, harvest_ok, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=1)
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    state = json.loads((tmp_path / "vo" / "seq_state.json").read_text())
    assert state["step"] == 1
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "vo"))
    again = FakeLoop(tmp_path / "b", max_steps=2)
    asyncio.run(asyncio.wait_for(rl_seq.run(again, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert "load_checkpoint" in again.log and [r["step"] for r in again.feed.of("step")] == [2]
    assert again.feed.of("seq_resume")[0]["step"] == 1


def test_settings_require_streams_and_steps():
    with pytest.raises(ValueError):
        rl_seq.settings({"redesign": {"streams": 8}})
    with pytest.raises(ValueError):
        rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1, "nope": 1}})
    assert rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1}})["warmup"] == 10
    with pytest.raises(ValueError):
        rl_seq.settings({"redesign": {"streams": 8, "max_steps": -1}})
    assert rl_seq.settings({"redesign": {"streams": 8, "max_steps": 0}})["max_steps"] == 0  # a warm-up leg


def test_warmup_leg_boots_serves_v0_and_ends_without_play(tmp_path, harvest_ok, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=0)
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "max_steps" and loop.trainer.steps == 0 and loop.feed.of("seq_start")
    assert "load:policy:v0000" in loop.log and not loop.feed.of("seq_group")


def test_one_server_per_card_booted_and_stopped(tmp_path, harvest_ok, monkeypatch):
    import arena.rl_loop as rl_loop

    booted, stopped = [], []

    class Card(FakeServer):
        def __init__(self, server_dir, model, *, card, port, gpu_util, name):
            super().__init__(self_log, f"http://127.0.0.1:{port}")
            self.card = card

        async def boot(self):
            booted.append(self.card)
            return {"card": self.card}

        def stop(self):
            stopped.append(self.card)

        async def post(self, path, payload=None, timeout=900):
            assert path == "/tokenize" and any(isinstance(m["content"], list) for m in payload["messages"])
            self_log.append(f"warm:{self.card}")
            return {"count": 1}

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.setattr(rl_loop, "SecondServer", Card)
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.server_dir = tmp_path / "srv"
    loop.cfg["mg"]["redesign"]["cards"] = 3
    self_log = loop.log
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert booted == [1, 2] and stopped == [1, 2]
    assert loop.cfg["mg"]["trainer_cards"] == 3  # one trainer rank per card
    # groups pinned across the three servers
    assert set(loop.urls) == {"http://s0/v1", "http://127.0.0.1:1236/v1", "http://127.0.0.1:1237/v1"}
    assert {r["server"] for r in loop.feed.of("seq_group")} == {0, 1, 2}  # H2 (26 Oct): which server took each group
    # H2 attempt 1: every server's image processor warmed (no generation) before play, before the sleep
    assert sorted(loop.log[:2]) == ["warm:1", "warm:2"] and loop.log[2:5] == ["sleep"] * 3
    assert loop.log.count("load:policy:v0001") == 3
    warmed = loop.feed.of("seq_start")[0]["phases"]["warm_render_s"]  # every server (cache probe off); a failure is a row
    assert len(warmed) == 3 and "AttributeError" in warmed[0] and warmed[1:] == [0.0, 0.0]


def test_preflight_judge_nk_restore_checks():
    """NK: with restore_check, every restored attempt with a scripted prefix wrote its reply-id
    seeds (seeded=1), its first live request never timed out waiting for them, and the digest gate dropped no try."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    base, rl = "pf", "pf/arena/trials/rl-loop/rl"
    feed = [{"kind": "start"}, {"kind": "seq_resume", "step": 1}, {"kind": "seq_start"},
            {"kind": "step", "step": 2, "version": 2, "grad_norm": 0.1},
            {"kind": "seq_phase", "step": 2, "stop": {"switch_s": 1.0}, "start": {"load_s": 2.0}, "stats": {"saved": 3}},
            {"kind": "seq_end"}]
    good = {"source": "/runs/validation/preflight/pf/harvest/s0001-x", "scripted": 5, "ok": True,
            "prompt_digest": {"same": True}, "state": {"ok": True}, "seeded": 1, "seed_wait_s": 0.2, "seed_wait_timeout": 0}
    files = {f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in feed),
             f"{base}/arena/bootstrap-results.json": json.dumps({"rl-loop": "complete"}),
             f"{base}/arena/serving/vllm-openai-server.log": 'x STUB {"errors": 0, "exact": 0, "seeds": 1}',
             f"{base}/bootstrap.log": "ok"}
    dirs = {f"{base}/state": ["s0001", "s0002"], rl: ["p2g1-sc25-L2"]}
    study = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 2}}}}

    def judged(rows, extra=(), check=True):
        files[f"{rl}/p2g1-sc25-L2/level-bench.json"] = json.dumps({"attempts": [{"label": f"t{i}", "restore": r}
                                                                                for i, r in enumerate(rows)]})
        files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed[:3] + list(extra) + feed[3:])
        return modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", study,
                               {"exit_code": 0, "restore_check": check})

    out = judged([good, dict(good)])
    assert out["verdict"] == "pass", out["checks"]
    # v17: the resumed attempt's route restore read indexed tars only, its time reported
    idx = {"kind": "served_restore", "seconds": 2.5, "files": 40, "indexed": 3, "scanned": 0}
    ix = judged([good], [idx])
    assert ix["checks"]["restore_indexed"] is True and ix["restore"]["served"]["seconds"] == 2.5
    assert judged([good], [{**idx, "scanned": 1}])["checks"]["restore_indexed"] is False
    assert judged([good], [{**idx, "indexed": 0}])["verdict"] == "fail"
    # the cancelled attempt's feed lagged its checkpoint (v15 NK: feed to step 1, resumed at 2): steps from the resume on
    lag = [{"kind": "step", "step": 1, "version": 1, "grad_norm": 0.1}, {"kind": "seq_resume", "step": 2},
           {"kind": "seq_start"}, {"kind": "step", "step": 3, "version": 3, "grad_norm": 0.1},
           {"kind": "seq_phase", "step": 3, "stop": {"switch_s": 1.0}, "start": {"load_s": 2.0}, "stats": {"saved": 3}},
           {"kind": "seq_end"}]
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in [{"kind": "start"}] + lag)
    three = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 3}}}}
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", three,
                           {"exit_code": 0, "restore_check": True})["checks"]["steps_ok"] is True
    gap = [e for e in lag if e.get("step") != 3 or e["kind"] != "step"] + [{"kind": "step", "step": 4, "grad_norm": 0.1}]
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in [{"kind": "start"}] + gap)
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", three,
                           {"exit_code": 0})["checks"]["steps_ok"] is False  # a gap after the resume still fails
    assert out["restore"]["seed"] == {"scripted": 2, "live": 2, "seeded": 2, "timeouts": 0, "gate_drops": 0}
    # a restore cut before its first live request: seeded, no wait yet; at least one must reach it
    cut = {k: v for k, v in good.items() if k not in ("seed_wait_s", "seed_wait_timeout")}
    assert judged([good, cut])["verdict"] == "pass"
    assert judged([cut])["checks"]["restore_played"] is False
    # the leg played no restored start: a failure (the self-kill and the resume make one certain)
    none = judged([])
    assert none["verdict"] == "fail" and none["checks"]["restore_played"] is False
    assert judged([], [{"kind": "digest_gate", "bench": "p2g1", "dropped": 1}])["verdict"] == "fail"  # others still fail
    assert judged([good, {**good, "seeded": 0}])["checks"]["restore_seeded"] is False
    assert judged([good, {**good, "seed_wait_s": None, "seed_wait_timeout": 1}])["checks"]["seed_wait_no_timeout"] is False
    out = judged([good], [{"kind": "digest_gate", "bench": "p2g1", "dropped": 2}])
    assert out["verdict"] == "fail" and out["checks"]["digest_gate_no_drops"] is False
    # NKc (29 Oct): a fresh attempt 1 on an NK leg fails by name
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in [f for f in feed if f.get("kind") != "seq_resume"])
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study,
                           {"exit_code": 0, "restore_check": True})["checks"]["resumed_from_checkpoint"] is False
    # other legs: no restore checks, no rows in the report
    out = judged([], check=False)
    assert "restore_seeded" not in out["checks"] and "rows" not in out["restore"] and "seed" not in out["restore"]


def test_nk_attempt0_self_kill_dies_at_its_first_checkpoint_past_step_0(tmp_path, monkeypatch):
    """attempt 0 dies as a preemption would (feed mirrored, volume committed, os._exit), not
    through a Modal cancel that can run the finally (identity + pack: a1 then started fresh)."""
    import sys
    import threading

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    live, mirror = tmp_path / "live", tmp_path / "out/rl"
    live.mkdir()
    (live / "feed.jsonl").write_text(json.dumps({"kind": "checkpoint_state", "step": 0}) + "\n")
    commits, exits = [], []
    monkeypatch.setattr(modal_rl1, "commit_volumes", lambda: commits.append(1))

    class Stop:
        n = 0

        def wait(self, s):
            Stop.n += 1
            if Stop.n == 2:  # the first checkpoint past step 0 lands
                with (live / "feed.jsonl").open("a") as f:
                    f.write(json.dumps({"kind": "checkpoint_state", "step": 1}) + "\n")
            if Stop.n == 4:  # a group finishes after it (v17NKr: played tries for a1 to restore)
                with (live / "feed.jsonl").open("a") as f:
                    f.write(json.dumps({"kind": "group_done", "gid": "p2g1"}) + "\n")
            if Stop.n == 6:  # a harvested start saved (v17NKt: a1 drew none without one)
                with (live / "feed.jsonl").open("a") as f:
                    f.write(json.dumps({"kind": "seq_phase", "stats": {"saved": 2}}) + "\n")
                    # an L1 group launched: not a harvested start, no kill yet
                    f.write(json.dumps({"kind": "seq_group", "gid": "p2g3", "set": "sb26-L1", "start": "sb26-L1"}) + "\n")
            if Stop.n == 7:  # a group on a harvested start in flight
                assert exits == []
                with (live / "feed.jsonl").open("a") as f:
                    f.write(json.dumps({"kind": "seq_group", "gid": "p2g4", "set": "sb26-L3",
                                        "start": "sb26-L3-p2g1-t0"}) + "\n")
            return Stop.n > 8

    modal_rl1.self_kill(Stop(), live, mirror, poll=0, exit=exits.append)
    assert exits == [137] and commits == [1] and '"saved": 2' in (mirror / "feed.jsonl").read_text()
    exits.clear()
    (live / "feed.jsonl").write_text(json.dumps({"kind": "checkpoint_state", "step": 1}) + "\n")
    Stop.n = 0  # no group after the checkpoint: killed once after_s passes
    modal_rl1.self_kill(Stop(), live, mirror, poll=0, exit=exits.append, after_s=0.0)
    assert exits == [137]
    exits.clear()
    (live / "feed.jsonl").write_text(json.dumps({"kind": "checkpoint_state", "step": 0}) + "\n")
    Stop.n = 10  # stop set before any checkpoint past 0: never kills
    modal_rl1.self_kill(type("S", (), {"wait": lambda self, s: True})(), live, mirror, poll=0, exit=exits.append)
    assert exits == []


def test_redesign_configs_bundle_their_history():
    """pf R2 (26 Oct) died on a history file the bundle lacked: every mg.redesign config ships it under ``files``."""
    root = Path(__file__).resolve().parents[1]
    seen = 0
    for path in sorted((root / "launch/rl/runs").glob("*.json")):
        cfg = json.loads(path.read_text())
        mg = ((cfg.get("study") or {}).get("rl_loop") or {}).get("mg") or {}
        if not mg.get("redesign"):
            continue
        seen += 1
        history = rl_seq.settings(mg)["history"]  # the whole mg: freeze_policy reads mg.draft_weight
        assert history in cfg["files"], f"{path.name}: {history} not in files"
        assert (root / cfg["files"][history]).is_file()
    assert seen


def test_preflight_judge_reads_a_redesign_feed():
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    base, rl = "pf", "pf/arena/trials/rl-loop/rl"
    feed = [{"kind": "start"}, {"kind": "seq_start"},
            {"kind": "step", "step": 1, "version": 1, "grad_norm": 0.1}, {"kind": "seq_phase", "step": 1, "stop": {"switch_s": 1.0},
                                                                 "start": {"load_s": 2.0}, "stats": {"saved": 3}},
            {"kind": "step", "step": 2, "version": 2, "grad_norm": 0.1}, {"kind": "seq_phase", "step": 2, "stop": {"switch_s": 1.0},
                                                                 "start": {"load_s": 2.0}, "stats": {"saved": 5}},
            {"kind": "seq_end"}]
    files = {f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in feed),
             f"{base}/arena/bootstrap-results.json": json.dumps({"rl-loop": "complete"}),
             f"{base}/arena/serving/vllm-openai-server.log": 'x STUB {"errors": 0, "exact": 4}',
             f"{base}/bootstrap.log": "ok"}
    dirs = {f"{base}/state": ["s0001", "s0002"], rl: []}
    study = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 2}}}}
    out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})
    assert out["verdict"] == "pass", out["checks"]
    # F1's early prepare (gate-v1 28 Oct): an alert-only prepare failure, or a pending row short of its groups, fails
    for bad in ([{"kind": "alert", "why": "seq: ahead prepare failed (alert only)", "error": "KeyError('mg_sets')"}],
                [{"kind": "seq_ahead_pending", "groups": 16, "prepared": 0}]):
        files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed[:2] + bad + feed[2:])
        out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})
        assert out["verdict"] == "fail" and out["checks"]["ahead_prepared"] is False
    # ...and the trainer must take them (gate legs 28 Oct: 0 of 192 hit): >= 90% of an ahead update's windows
    for share, verdict in ((0.0, "fail"), (None, "fail"), (0.95, "pass")):
        files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed[:2] + [
            {"kind": "seq_accumulate", "one_call": True, "ahead_expected": True, "ahead_hit_share": share}] + feed[2:])
        assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})["verdict"] == verdict
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed[:2] + [
        {"kind": "seq_ahead_pending", "groups": 16, "prepared": 16}] + feed[2:])
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})["verdict"] == "pass"
    # a resumed attempt (NK2 28 Oct): exact stub hits are impossible, so that check is N/A with its reason, and the
    # stub must still answer without errors; a fresh attempt keeps the exact-hit rule
    resumed = feed[:2] + [{"kind": "seq_resume", "step": 1}] + feed[2:]
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in resumed)
    for stub, verdict in (('{"errors": 0, "exact": 0, "any": 18}', "pass"), ('{"errors": 2, "exact": 0}', "fail")):
        files[f"{base}/arena/serving/vllm-openai-server.log"] = f"x STUB {stub}"
        out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", study, {"exit_code": 0})
        assert out["verdict"] == verdict, out["checks"]
        assert "stub_clean_on_recorded_paths" not in out["checks"] and "resumed attempt" in out["na"][
            "stub_clean_on_recorded_paths"] and out["checks"]["resumed_from_checkpoint"]
    # a resumed update leg (N11m3 28 Oct: KeyError, the stub check popped twice): replayed groups, nothing played
    replayed = feed[:2] + [{"kind": "seq_resume", "step": 1}, {"kind": "seq_replay", "groups": 3},
                           {"kind": "seq_train", "groups": 3}, {"kind": "step", "step": 1, "version": 1, "groups": 3}]
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in replayed)
    out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", study, {"exit_code": 0})
    assert "stub_no_errors" not in out["checks"] and "stub_clean_on_recorded_paths" not in out["checks"]
    assert out["checks"]["replay_trained"] and out["checks"]["resumed_from_checkpoint"]
    # a preempted leg resumed at step 0 (N1f 29 Oct): the mid-play restore, no checkpoint, no seq_resume row
    early = feed[:2] + [{"kind": "seq_play_resume", "step": 0, "groups": 1}] + feed[2:]
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in early)
    out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", study, {"exit_code": 0})
    assert out["checks"]["resumed_from_checkpoint"] is True
    out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "pf-lost-1", study,
                          {"exit_code": 0, "restore_check": True})
    assert out["checks"]["resumed_from_checkpoint"] is False  # an NK leg still needs a checkpoint resume
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed)
    out = modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})
    assert out["verdict"] == "fail" and out["checks"]["stub_clean_on_recorded_paths"] is False and "na" not in out
    files[f"{base}/arena/serving/vllm-openai-server.log"] = 'x STUB {"errors": 0, "exact": 4}'
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed)
    # an all-ties phase switches without a step (rd resume leg 26 Oct): counted, not a failure
    feed[4:4] = [{"kind": "seq_step_skip", "step": 2, "groups": 0},
                 {"kind": "seq_phase", "step": 1, "stop": {"switch_s": 0.4}, "start": {"load_s": 0.0}, "stats": {"saved": 4}}]
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed)
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})["verdict"] == "pass"
    # the CPU leg's shrink caps max_steps at 2 (dl2-v2 N1, 26 Oct: config 3, leg ran 2): judged on the leg's own cap
    deep = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 3}}}}
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", deep, {"exit_code": 0})["verdict"] == "fail"
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in [{"kind": "preflight_shrink", "max_steps": 2}] + feed)
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", deep, {"exit_code": 0})["verdict"] == "pass"
    # back-off (rl_seq.backoff): a preempting phase followed by another needs a cut; the back-off leg needs one
    j = lambda ident=None: modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study,
                                           {"exit_code": 0, **(ident or {})})
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed)
    assert j({"preempt_after": 5000})["checks"]["backoff_on_pressure"] is False  # the leg asked for a cut: none
    hot = [dict(e, guard={"kv_peak": 0.3, "preemptions": 1}) if e["kind"] == "seq_phase" and e["step"] == 1 else e
           for e in feed]
    hot.insert(hot.index(next(e for e in hot if e.get("guard"))) + 1, {"kind": "seq_train"})
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in hot)
    assert j()["checks"]["backoff_on_pressure"] is False  # pressure, a later phase, no cut
    hot.insert(hot.index({"kind": "seq_train"}), {"kind": "seq_knobs", "backoff": {"cut": 20}})
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in hot)
    assert j({"preempt_after": 5000})["checks"]["backoff_on_pressure"] is True
    # a 1-step leg (D2 v2 N1, 27 Oct): no next phase to cut; the guard's read of the stub's preemption is the check
    one = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 1}}}}
    j1 = lambda f: modal_rl1.judge(base, {**files, f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in f)}.get,
                                   lambda d: dirs.get(d, []), "", one, {"exit_code": 0, "preempt_after": 1000})
    seen = [{"kind": "preflight_shrink", "max_steps": 1}, *feed[:3],
            {"kind": "seq_phase", "step": 1, "guard": {"kv_peak": 0.03, "preemptions": 2}}]
    assert j1(seen)["checks"]["backoff_on_pressure"] is True
    assert j1([*seen, {"kind": "seq_group"}])["checks"]["backoff_on_pressure"] is True  # the last phase's refill
    assert j1([*seen[:-1], {"kind": "seq_phase", "step": 1, "guard": {"kv_peak": 0.03, "preemptions": 0}}]
              )["checks"]["backoff_on_pressure"] is False
    # pressure only in the last played phase of a 2-step leg (real-v6 N9 27 Oct: the resumed update, then one phase):
    # no later phase to cut; the guard's read is the check
    two = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 2}}}}
    j2 = lambda f: modal_rl1.judge(base, {**files, f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in f)}.get,
                                   lambda d: dirs.get(d, []), "", two, {"exit_code": 0, "preempt_after": 1000})
    assert j2([*feed[:3], seen[-1]])["checks"]["backoff_on_pressure"] is True
    # an update leg (replay_groups, U1 27 Oct): nothing played, so no harvest / stub request / pressure; the step trains
    # the saved groups
    leg = [{"kind": "seq_start"}, {"kind": "seq_replay", "groups": 16}, {"kind": "seq_train", "groups": 16},
           {"kind": "step", "step": 1, "grad_norm": 0.1, "groups": 16},
           {"kind": "seq_phase", "step": 1, "stop": {"switch_s": 1.0}, "start": {"load_s": 2.0}, "stats": {"saved": 0}},
           {"kind": "seq_end"}]
    out = j1(leg)
    assert out["verdict"] == "pass" and out["checks"]["replay_trained"] and "harvest_saved" not in out["checks"], out
    assert j1([dict(e, groups=15) if e["kind"] == "step" else e for e in leg])["verdict"] == "fail"
    assert "harvest_saved" in j1([*leg[:-1], {"kind": "seq_group"}, leg[-1]])["checks"]  # it played: full checks
    # a play_only leg (G-long 27 Oct): play phases, admission rows, no switch or step
    po = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 1, "play_only": True}}}}
    jp = lambda f: modal_rl1.judge(base, {**files, f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in f)}.get,
                                   lambda d: dirs.get(d, []), "", po, {"exit_code": 0, "preempt_after": 1000})
    play = [{"kind": "seq_start"}, {"kind": "seq_group"}, {"kind": "seq_admit", "servers": [{"admitted": 12}]},
            {"kind": "seq_play_phase", "step": 0, "stats": {"saved": 3}, "guard": {"kv_peak": 0.1, "preemptions": 1}},
            {"kind": "seq_end", "why": "deadline"}]
    out = jp(play)
    assert out["verdict"] == "pass" and out["checks"]["play_phases"] and out["checks"]["admit_logged"], out
    assert "steps_ok" not in out["checks"]
    assert jp([e for e in play if e["kind"] != "seq_admit"])["verdict"] == "fail"
    assert jp([*play[:-1], {"kind": "seq_train"}, play[-1]])["verdict"] == "fail"
    # N3 (G-long v5): a scripted /metrics outage must raise the blind hold, then the guard must see again
    jo = lambda f: modal_rl1.judge(base, {**files, f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in f)}.get,
                                   lambda d: dirs.get(d, []), "", po,
                                   {"exit_code": 0, "preempt_after": 1000, "metrics_outage": [160, 90]})
    blind = {"kind": "alert", "why": "guard blind: http://s /metrics unread 3 polls; admission held at 12 (was 14)",
             "url": "http://s"}
    held = [*play[:3], blind, {"kind": "seq_guard_sees", "url": "http://s"}, *play[3:]]
    out = jo(held)
    assert out["verdict"] == "pass" and out["checks"]["guard_blind_hold"] and out["checks"]["feed_final_no_error"], out
    out = jo([e for e in held if e["kind"] != "seq_guard_sees"])
    assert out["checks"]["guard_blind_hold"] is False and out["verdict"] == "fail"
    assert jo(play)["checks"]["guard_blind_hold"] is False  # no hold during the outage
    assert jp(held)["checks"]["feed_final_no_error"] is True  # unscripted, a blind the guard saw past is the hold working
    assert jp([e for e in held if e["kind"] != "seq_guard_sees"])["checks"]["feed_final_no_error"] is False  # never seen
    other = [dict(e, url="http://t") if e["kind"] == "seq_guard_sees" else e for e in held]
    assert jp(other)["checks"]["feed_final_no_error"] is False  # another server seen again does not clear this one
    feed[-1] = {"kind": "error"}
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in feed)
    assert modal_rl1.judge(base, files.get, lambda d: dirs.get(d, []), "", study, {"exit_code": 0})["verdict"] == "fail"



def test_preflight_judge_reads_a_p1_lr_probe_feed():
    """P1 (the lr probe before the run, rl_seq.lr_probe): judged on its probe rows."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    base, rl = "pf", "pf/arena/trials/rl-loop/rl"
    rates = [{"kind": "lr_probe_rate", "lr": lr, "ok": True, "steps": [{"kl_base": 0.01}],
              "reload": {"ok": True, "update_after_reload": True}} for lr in (7.5e-5, 1.5e-4)]
    feed = [{"kind": "start"}, *rates, {"kind": "lr_probe", "rates": 2}]
    files = {f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in feed),
             f"{base}/arena/bootstrap-results.json": json.dumps({"rl-loop": "complete"}),
             f"{base}/bootstrap.log": "ok"}
    study = {"rl_loop": {"mg": {"lr_probe": {"rates": [7.5e-5, 1.5e-4]}, "reload_check": True}}}
    judge = lambda: modal_rl1.judge(base, files.get, lambda d: [], "", study, {"exit_code": 0})
    out = judge()
    assert out["verdict"] == "pass", out["checks"]
    assert set(out["checks"]) == {"exit_clean", "trial_complete", "no_error", "server_stopped", "every_arm_ok",
                                  "steps_probed", "first_arm_smoke"}
    rates[1]["ok"] = False
    files[f"{rl}/feed.jsonl"] = "\n".join(json.dumps(e) for e in [{"kind": "start"}, *rates, {"kind": "lr_probe"}])
    assert judge()["verdict"] == "fail"

def test_inexact_restored_tries_never_train_and_are_counted(tmp_path):
    """a start failing the game-state or context check never trains; the drops are counted."""
    from arena import rl_seq

    loop = FakeLoop(tmp_path)
    loop.trainer = FakeTrainer(loop.log)

    class G:
        gid = "g00001"

    rows = [{"label": "a000"},
            {"label": "a001", "restore": {"ok": True, "state": {"ok": True}}},
            {"label": "a002", "restore": {"ok": False, "state": {"ok": True}, "diff_count": 1}},  # context differs
            {"label": "a003", "restore": {"ok": False, "state": {"ok": False}}},  # board differs
            {"label": "a004", "resumed": {"ok": False, "diff_count": 2}}]
    out = asyncio.run(rl_seq.train_step(loop, [(G(), rows)], 1e-4, tmp_path / "s1"))
    assert out["restore_dropped"] == 3 and out["tries"] == 2
    drop = [e for e in loop.feed.rows if e["kind"] == "seq_restore_drop"][0]
    assert drop["labels"] == ["a002", "a003", "a004"] and drop["why"] == ["context", "state", "state"]
    assert rl_seq.exact_start({"restore": {"ok": True}, "resumed": {"ok": True}})


def test_bench_deadline_ends_the_run_without_training_empty_groups(tmp_path, harvest_ok, monkeypatch):
    """T3 v3 (26 Oct): past the level bench's deadline every new group failed at once and an empty group closed
    the phase; now the first such failure stops new groups and the run ends with why "deadline", no step."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=3)
    real, calls = loop.mg_play, []

    async def late(name, spec, starts, tries, **kw):
        calls.append(name)
        if len(calls) > 2:
            raise TimeoutError("Only 239 s left for the level bench")
        return await real(name, spec, starts, tries, **kw)

    loop.mg_play = late
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "deadline" and loop.trainer.steps == 0
    assert loop.feed.of("seq_deadline") and not loop.feed.of("alert") and not loop.feed.of("seq_train")
    assert loop.feed.of("seq_end")[-1]["why"] == "deadline"



def test_play_only_plays_phases_without_a_step_and_logs_admission(tmp_path, harvest_ok, monkeypatch):
    """G-long (27 Oct): play_only closes each phase's groups and plays on (carried tries keep their streams), never
    switches or steps; seq_admit rows give per-server admitted tries, the slot pool and the shared exact KV gate."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.cfg["mg"]["redesign"].update(play_only=True, admit_log_s=0.01)
    loop.cfg["mg"]["active_tries"] = 50
    real, calls = loop.mg_play, []

    async def late(name, spec, starts, tries, **kw):
        calls.append(name)
        if len(calls) > 14:
            raise TimeoutError("Only 239 s left for the level bench")
        await asyncio.sleep(0.02)
        return await real(name, spec, starts, tries, **kw)

    loop.mg_play = late
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "deadline" and (loop.trainer is None or loop.trainer.steps == 0)
    assert len(loop.feed.of("seq_play_phase")) >= 2 and not loop.feed.of("seq_train")
    rows = [r for r in loop.feed.of("seq_admit") if "servers" in r]
    assert rows and all(r["streams"] == 50 and len(r["servers"]) == 1 for r in rows)
    assert any(r["servers"][0]["admitted"] > 0 for r in rows)
    assert all(r["servers"][0]["admitted"] + r["parked"] <= 50 for r in rows)



def test_play_only_preflight_ends_after_max_steps_phases(tmp_path, harvest_ok, monkeypatch):
    """A preflight (ARC_PREFLIGHT_STUB) of a play_only leg ends after max_steps play phases, not at its deadline."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", "{}")
    loop = FakeLoop(tmp_path, max_steps=2, clear_p=0.5)
    loop.cfg["mg"]["redesign"].update(play_only=True, admit_log_s=0.01)
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "max_steps" and len(loop.feed.of("seq_play_phase")) == 2 and not loop.feed.of("seq_train")


def test_admit_row_reads_the_pool_and_the_shared_gate():
    from types import SimpleNamespace

    pool = level_bench.SharedSlots(200, shared_gate=True)
    assert pool.shared_gate and pool.kv_fill is None
    pool.gate = SimpleNamespace(snapshot=lambda: {"inflight": 7, "waiting": 2, "reserved": 900, "capacity": 1913})
    backlog = SimpleNamespace(streams=400, parked=5, groups={"g1": {"group": {"tries": 12}, "results": {0: {}, 1: {}}},
                                                             "g2": {"group": {"tries": 12}, "results": {}}})
    seq = SimpleNamespace(urls=["http://a/v1", "http://b/v1"], on_server={"g1": 0, "g2": 1}, backlog=backlog)
    old = level_bench.SHARED_SLOTS_BY_URL
    level_bench.SHARED_SLOTS_BY_URL = {"http://a/v1": pool}
    try:
        row = rl_seq.admit_row(seq)
    finally:
        level_bench.SHARED_SLOTS_BY_URL = old
    a, b = row["servers"]
    assert a["admitted"] == 10 and a["slots"] == 200 and a["gate_inflight"] == 7 and a["gate_waiting"] == 2
    assert b == {"admitted": 12} and row["parked"] == 5 and a["contexts"] == {"n": 0}
    live = [SimpleNamespace(next_estimate=n * 1024, floor_tokens=f * 1024) for n, f in
            ((10, 0), (20, 0), (0, 70), (50, 0), (110, 0))]
    assert rl_seq.context_spread(live) == {"n": 5, "p10": 10240, "p50": 51200, "p90": 112640, "max": 112640,
                                           "ge40k": 3, "ge64k": 2, "ge100k": 1}


def test_l2_per_side_12_trains_every_try_with_a_nonzero_advantage():
    """every group trains all 12 tries (l2_per_side 12 in the redesign configs); exact-zero advantages
    still drop."""
    from types import SimpleNamespace

    from arena.rl_loop import Loop

    rewards = [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, -1.0, -1.0, -2.0, (3 + 2 + 2 + 4 + 0.5 - 4) / 11]
    rows = [{"label": f"a{i:03d}", "rw": r, "level_rows": [], "completion_tokens": 10} for i, r in enumerate(rewards)]
    fake = SimpleNamespace(cfg={"mg": {"group_size": 12}, "l2_per_side": 12}, mg_turn={})
    fake.mg_groups = lambda rs, n: {"k": rs}
    fake.mg_reward = lambda r, name: r["rw"]
    fake.mg_preflight_spread = lambda rw, info: rw
    fake.mg_turn_credit = lambda folder, grp, rw, name: [0.0] * len(grp)
    out, info = Loop.mg_select_rows(fake, "/f", rows, "set")
    total = sum(rewards)
    adv = [r - (total - r) / 11 for r in rewards]
    assert {row["label"] for _, row, _, _ in out} == {rows[i]["label"] for i in range(12) if adv[i] != 0}
    fake.cfg["l2_per_side"] = 3
    assert len(Loop.mg_select_rows(fake, "/f", rows, "set")[0]) == 6  # the v7 recipe: 3 + 3


@pytest.mark.parametrize("adapter", [None, "adapters/s0017"])
def test_kv_dump_runs_under_the_play_adapter_before_the_gate_opens(tmp_path, harvest_ok, monkeypatch, adapter):
    """mg.redesign.kv_golden (E4 A/B, E1): every server dumps after the adapter load, before any play request; the
    seq_boot kv_dump row carries the reference card's floor and each server's |dlogp|; off by default. With
    kv_golden.adapter (lora-l1) the dump runs under that trained adapter (bundle path), loaded and unloaded around it."""
    import arena.rl_loop as rl_loop

    probe = {"label": "p0", "depth": 10, "start": 3, "tokens": [11, 12],
             "body": {"messages": [{"role": "user", "content": "p0"}], "chat_template_kwargs": {}}}
    (tmp_path / "probes.jsonl").write_text(json.dumps(probe) + "\n")

    class Card(FakeServer):
        def __init__(self, server_dir=None, model="base", *, card=0, port=1234, gpu_util=None, name=None):
            super().__init__(self_log, f"http://127.0.0.1:{port}")
            self.card = card

        async def boot(self):
            return {"card": self.card}

        def stop(self):
            pass

        async def post(self, path, payload=None, timeout=900):
            if path == "/tokenize":
                return {"count": 1}
            assert path == "/v1/chat/completions" and payload["model"] == ("kv-golden" if adapter else "policy")
            self_log.append(f"dump:{self.card}")
            return {"prompt_logprobs": [None, None, None, {"11": {"logprob": -0.5}}, {"12": {"logprob": -0.25}}]}

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.setattr(rl_loop, "SecondServer", Card)
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.server_dir = tmp_path / "srv"
    self_log = loop.log
    loop.server = Card(card=0)
    spec = {"probes": str(tmp_path / "probes.jsonl"), **({"adapter": adapter} if adapter else {})}
    loop.cfg["mg"]["redesign"].update(cards=2, kv_golden=spec)
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    first_load = loop.log.index("load:policy:v0000")
    dumps = [i for i, x in enumerate(loop.log) if x.startswith("dump:")]
    assert len(dumps) == 3 and min(dumps) > first_load  # one pass per server + the reference's all-at-once pass
    after = loop.log[max(dumps) + 1:]
    if adapter:
        assert loop.log[first_load + 2:first_load + 4] == ["load:kv-golden:s0017"] * 2
        assert after[:2] == ["unload:kv-golden"] * 2 and after[2] == "sleep"
    else:
        assert loop.log[first_load + 1] == "load:policy:v0000" and after[0] == "sleep"  # then play, train
    row = [r for r in loop.feed.of("seq_boot") if r.get("stage") == "kv_dump"][0]["kv_dump"]
    assert row["ref"] == "server2" and row["servers"]["server1"]["max_abs"] == 0 and row["floor"]["tokens"] == 2
    assert row["adapter"] == (str(Path(loop.source) / adapter) if adapter else None)
    assert loop.feed.of("seq_start")[0]["phases"]["kv_golden_s"] is not None
    assert (loop.output / "kv-golden" / "server1.json").is_file()
    off = FakeLoop(tmp_path / "off", max_steps=1)  # off by default: no dump row
    asyncio.run(asyncio.wait_for(rl_seq.run(off, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert not [r for r in off.feed.of("seq_boot") if r.get("stage") == "kv_dump"]
    assert off.feed.of("seq_start")[0]["phases"]["kv_golden_s"] is None


@pytest.mark.parametrize("stale", [False, True])
def test_r3_check_boot_stage_rows_and_alert(tmp_path, harvest_ok, monkeypatch, stale):
    """mg.redesign.r3_check (H5): the A/S/B replay on every server before the gate opens; a seq_boot r3_check row;
    restored rows that differ from the first request's are a feed alert (the verdict fails)."""
    import functools

    import arena.rl_loop as rl_loop
    from arena import r3_offload_check
    from test_routed_offload import fake_server, probes_file

    class Card(FakeServer):
        def __init__(self, server_dir=None, model="base", *, card=0, port=1234, gpu_util=None, name=None):
            super().__init__(self_log, f"http://127.0.0.1:{port}")

        async def boot(self):
            return {}

        def stop(self):
            pass

    route = tmp_path / "route"
    route.mkdir()
    monkeypatch.setattr(r3_offload_check, "boot_check", functools.partial(
        r3_offload_check.boot_check, transport=fake_server(route, stale=stale)))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.setattr(rl_loop, "SecondServer", Card)
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.server_dir = tmp_path / "srv"
    loop.server_dir.mkdir()
    (loop.server_dir / "r3-offload-11.json").write_text(json.dumps({"bound": 1}))
    loop.study = {"server": {"served_route": str(route)}}
    self_log = loop.log
    loop.server = Card()
    loop.cfg["mg"]["redesign"].update(cards=2, r3_check={"probes": str(probes_file(tmp_path)), "min_hits": 6,
                                                         "settle_s": 0})
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    row = [r for r in loop.feed.of("seq_boot") if r.get("stage") == "r3_check"][0]["r3_check"]
    assert row["ok"] is (not stale) and set(row["servers"]) == {"server1", "server2"} and row["bound"] == 1
    alerts = [r["why"] for r in loop.feed.of("alert") if r["why"].startswith("r3_check")]
    assert len(alerts) == stale and loop.feed.of("r3_offload")[0]["metrics"]["infra.r3_offload_bound"] == 1
    assert loop.feed.of("seq_start")[0]["phases"]["kv_golden_s"] is not None
    assert (loop.output / "r3-check" / "verdict.json").is_file() and not list(route.glob("*.rt"))


class ProbeTrainer(FakeTrainer):
    """Records each rate's accumulate specs; probe KL grows with the steps since the last reset."""

    def __init__(self, log):
        super().__init__(log)
        self.since, self.specs = 0, []

    async def must(self, op, **kw):
        if op == "reset":
            self.since = 0
            self.specs.append([])
        if op == "score":
            self.log.append(f"score:{kw['label']}")
            return {"windows": {w["name"]: {"kl_base": 0.001 * self.since, "logratio_base": 0.0} for w in kw["windows"]}}
        return await super().must(op, **kw)

    async def call(self, op, **kw):
        if op == "accumulate":
            self.specs[-1].append([a["name"] for a in kw["attempts"]])
        if op == "step":
            self.since += 1
        return await super().call(op, **kw)


class ProbeLoop(FakeLoop):
    def __init__(self, tmp, **kw):
        super().__init__(tmp, max_steps=0, **kw)
        self.cfg["mg"]["redesign"]["lr_probe"] = {"rates": [5e-5, 1e-4], "steps": 2, "warmup": 0,
                                                  "probes": [{"folder": "/runs/x/heldout", "labels": ["h0", "h1"]}]}
        self.prepares = {}

    async def start_trainer(self):
        self.log.append("start_trainer")
        self.trainer = ProbeTrainer(self.log)

    def v7_prepare(self, g):
        cache = self.v7.setdefault("prepared", {})
        if g.gid in cache:  # as Loop.v7_prepare: a cached preparation is used as is
            return cache.pop(g.gid)
        self.prepares[g.gid] = self.prepares.get(g.gid, 0) + 1
        return super().v7_prepare(g)

    def windows(self, folder, rows, weights=None, prefix=""):
        return [{"rollout": f"{folder}/attempts/{r['label']}/rollout.jsonl", "name": prefix + r["label"]} for r in rows]


def test_lr_probe_trains_every_rate_on_the_same_groups_through_the_run_step(tmp_path, harvest_ok, monkeypatch):
    """P1 on rl_seq: the probe's step is the run's (train_step); every rate starts from the reset
    trainer on the same groups, in the same order, with the same prepared specs; nothing is served after play."""
    import sys

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = ProbeLoop(tmp_path)
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "lr_probe"
    plays = loop.feed.of("seq_probe_play")
    assert [r["phase"] for r in plays] == [1, 2] and all(r["groups"] == 4 for r in plays)
    assert all(n == 1 for n in loop.prepares.values()) and len(loop.prepares) == 8  # prepared once each
    first, second = loop.trainer.specs
    assert first == second and len(first) == 8  # same specs, same order, every group
    rates = loop.feed.of("lr_probe_rate")
    assert [r["lr"] for r in rates] == [5e-5, 1e-4] and all(r["ok"] for r in rates)
    assert [[s["kl_base"] for s in r["steps"]] for r in rates] == [[0.001, 0.002], [0.001, 0.002]]  # reset per rate
    assert loop.trainer.steps == 4 and not loop.feed.of("step") and loop.feed.of("lr_probe")
    i = loop.log.index("reset")
    assert loop.log[i + 1] == "score:base"  # the base reference after every reset
    # no adapter is loaded after play (only v0000 at the start); no checkpoint written
    assert [x for x in loop.log if x.startswith("load:")] == ["load:policy:v0000"] and "checkpoint" not in loop.log
    # modal_rl1's judge reads it as a P1 feed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    base, rl = "pf", "pf/arena/trials/rl-loop/rl"
    files = {f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in [{"kind": "start"}] + loop.feed.rows),
             f"{base}/arena/bootstrap-results.json": json.dumps({"rl-loop": "complete"}), f"{base}/bootstrap.log": "ok"}
    out = modal_rl1.judge(base, files.get, lambda d: [], "", {"rl_loop": {"mg": {}}}, {"exit_code": 0})
    assert out["verdict"] == "pass", out["checks"]


def test_lr_probe_settings_are_checked():
    ok = {"rates": [1e-4], "steps": 2, "probes": [{"folder": "f", "labels": ["a"]}]}
    assert rl_seq.settings({"redesign": {"streams": 8, "max_steps": 0, "lr_probe": ok}})["lr_probe"] == ok
    for bad in ({**ok, "rates": []}, {**ok, "steps": 0}, {**ok, "probes": [{"folder": "f"}]}, {**ok, "nope": 1},
                {**ok, "warmup": -1}, {**ok, "rates": [True]}):
        with pytest.raises(ValueError):
            rl_seq.settings({"redesign": {"streams": 8, "max_steps": 0, "lr_probe": bad}})


def test_lr_warms_up_over_the_run_and_a_resume_takes_its_steps_lr(tmp_path, harvest_ok, monkeypatch):
    """10-step linear warm-up on a fresh start, none on resume, then constant: a resumed run's
    steps take the fresh run's lr at those steps."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=2)
    loop.cfg["lr"] = 3e-5
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert [r["lr"] for r in loop.feed.of("step")] == pytest.approx([3e-6, 6e-6])
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "vo"))
    again = FakeLoop(tmp_path / "b", max_steps=4)
    again.cfg["lr"] = 3e-5
    asyncio.run(asyncio.wait_for(rl_seq.run(again, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert [r["step"] for r in again.feed.of("step")] == [3, 4]
    assert [r["lr"] for r in again.feed.of("step")] == pytest.approx([9e-6, 1.2e-5])  # no re-warm from 3e-6


def test_schedule_check_passes_the_owner_schedule_and_flags_a_rewarm(monkeypatch):
    rl_loop = {"lr": 3e-5, "mg": {"redesign": {"streams": 8, "max_steps": 100, "warmup": 10}}}
    assert rl_seq.schedule_check(rl_loop) == []
    assert [rl_seq.lr_at(3e-5, 10, k) for k in (0, 4, 9, 50)] == pytest.approx([3e-6, 1.5e-5, 3e-5, 3e-5])
    assert rl_seq.schedule_check({"lr": 1e-4, "mg": {}}) == []  # not a redesign run
    real = rl_seq.Seq.lr

    def rewarm(self):  # the old per-container counter: a resume restarts the ramp
        return rl_seq.lr_at(float(self.loop.cfg["lr"]), int(self.cfg["warmup"]), getattr(self, "since_resume", 0))

    monkeypatch.setattr(rl_seq.Seq, "lr", rewarm)
    assert any("resume at step" in p for p in rl_seq.schedule_check(rl_loop))
    monkeypatch.setattr(rl_seq.Seq, "lr", real)


def test_launcher_check_runs_the_schedule_check():
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launch/rl"))
    config = importlib.import_module("config")
    src = (Path(__file__).resolve().parents[1] / "launch/rl/config.py").read_text()
    assert "schedule_check" in src and callable(config.checks)


def test_hread_exports_land_in_step_order_when_an_earlier_copy_is_slower(tmp_path, harvest_ok, monkeypatch):
    """CI 5d326b889: exports ran side by side and hread_due rows came in completion order ([0, 4, 2])."""
    import time as _time

    from arena import hread

    def export(src, dest, meta=None):
        _time.sleep(0.3 if meta["step"] == 1 else 0.0)  # step 1's copy is the slow one
        return {"digest": f"d{meta['step']}", "total_bytes": 1, "files": [], "draft": None}

    monkeypatch.setattr(hread, "export_adapter", export)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=3)
    loop.cfg["mg"]["redesign"]["hread"] = {"every": 1}
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert [r["step"] for r in loop.feed.of("hread_due")] == [0, 1, 2, 3]


def test_live_knobs_change_streams_slots_and_phase_groups_at_a_phase_start(tmp_path):
    """Checklist item 5: streams, per-server slots and phase_groups are read live at each play
    phase's start, slots capped by max_num_seqs and streams by servers x slots; a bad file changes nothing."""
    from types import SimpleNamespace
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"seq": {"active_tries": 200, "streams": 400, "phase_groups": 8}, "other": 1}))
    pools = {u: level_bench.SharedSlots(120) for u in ("http://a/v1", "http://b/v1")}
    loop = SimpleNamespace(study={"server": {"max_num_seqs": 152}})
    seq = SimpleNamespace(backlog=SimpleNamespace(streams=240, cfg={"phase_groups": 16}))
    got = rl_seq.apply_knobs(loop, seq, rl_seq.live_knobs(str(knobs)), pools)
    assert got == {"active_tries": [120, 152], "streams": [240, 304], "phase_groups": [16, 8]}
    assert [p.slots for p in pools.values()] == [152, 152] and seq.backlog.streams == 304
    assert rl_seq.apply_knobs(loop, seq, rl_seq.live_knobs(str(knobs)), pools) == {}  # unchanged: no row
    knobs.write_text("{not json")
    assert rl_seq.live_knobs(str(knobs)) == {} and rl_seq.live_knobs(str(tmp_path / "none.json")) == {}
    knobs.write_text(json.dumps({"seq": {"streams": 280, "phase_groups": True, "active_tries": -1}}))
    assert rl_seq.live_knobs(str(knobs)) == {"streams": 280}
    assert rl_seq.apply_knobs(loop, seq, {"streams": 280}, pools) == {"streams": [304, 280]}
    knobs.write_text(json.dumps({"seq": {"level_inflight": 2}}))  # live, over a resumed state's value
    seq.backlog.cfg["level_inflight"] = None
    assert rl_seq.apply_knobs(loop, seq, rl_seq.live_knobs(str(knobs)), pools) == {"level_inflight": [None, 2]}
    assert seq.backlog.cfg["level_inflight"] == 2



def test_live_knobs_keep_a_shrink_preflight_small(tmp_path, monkeypatch):
    """real-v12 N1/NK (28 Oct): the launch's phase_groups knob (16) undid the shrink's 2 and ran phase 1 to the
    45-min ceiling; under a shrink stub the knob is capped at 2, other knobs pass."""
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"seq": {"phase_groups": 16, "streams": 172}}))
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"shrink": True}))
    assert rl_seq.live_knobs(str(knobs)) == {"phase_groups": 2, "streams": 172}
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"shrink": False}))
    assert rl_seq.live_knobs(str(knobs))["phase_groups"] == 16
    monkeypatch.delenv("ARC_PREFLIGHT_STUB")
    assert rl_seq.live_knobs(str(knobs))["phase_groups"] == 16

def test_the_loop_reads_live_knobs_before_each_phase():
    """run() applies the knobs right before backlog.start_phase (source order), so a change lands at the next phase."""
    import inspect
    src = inspect.getsource(rl_seq.run)
    assert src.index("apply_knobs(loop, seq, knobs)") < src.index("seq.backlog.start_phase()")


def test_damage_probe_base_actions_from_the_history(tmp_path):
    """H3/H4 (26 Oct): seq_accumulate's base_actions was null (rl_seq has no mg screen), so the damage probe (2x base
    actions per clear) had no reference. The base is the history's median actions to clear the level; the step row
    carries the median ratio over its groups."""
    from arena import rl_backlog

    hist = [{"game": "sb26", "level": 1, "cleared": True, "tokens": 900, "actions": a} for a in (10, 12, 14)]
    cur = rl_backlog.Curriculum(hist)
    assert rl_seq.base_actions(cur, {"game": "sb26", "level": 1}) == 12
    assert rl_seq.base_actions(cur, {"game": "sb26", "level": 2}) is None  # under 2 clears: no base
    loop = FakeLoop(tmp_path)
    loop.trainer = FakeTrainer(loop.log)
    loop.cfg["mg_sets"] = {"sb26-L1": {"game": "sb26", "level": 1}}
    loop.seq_base_actions = lambda unit: rl_seq.base_actions(cur, loop.cfg["mg_sets"].get(unit) or {})
    head = loop.v7_prepare

    def prepare(g):
        row, specs = head(g)
        return {**row, "clear_actions": 18.0, "base_actions": None}, specs
    loop.v7_prepare = prepare

    class G:
        gid, unit = "g00001", "sb26-L1"

    out = asyncio.run(rl_seq.train_step(loop, [(G(), [{"label": "a000"}])], 1e-4, tmp_path / "s1"))
    acc = [e for e in loop.feed.rows if e["kind"] == "seq_accumulate"][0]
    assert acc["base_actions"] == 12 and out["actions_per_clear_vs_base"] == 1.5


def test_preflight_shrink_caps_the_redesign_steps_and_groups():
    """26 Oct: the real-run config (64 steps x 16 groups) ran its CPU preflight leg past the 45-min ceiling; the shrink
    caps mg.redesign at 2 steps x 2 groups (H4's own size), never raising a smaller setting."""
    from arena.rl_loop import Loop
    mg = {"redesign": {"max_steps": 64, "streams": 240}}
    Loop.shrink_mg(mg)
    assert mg["redesign"]["max_steps"] == 2 and mg["redesign"]["backlog"] == {"phase_groups": 2}
    mg = {"redesign": {"max_steps": 1, "backlog": {"phase_groups": 1, "budget_ceiling": 4000}}}
    Loop.shrink_mg(mg)
    assert mg["redesign"] == {"max_steps": 1, "backlog": {"phase_groups": 1, "budget_ceiling": 4000}}
    mg = {"redesign": {"max_steps": 64}}
    Loop.shrink_mg(mg, 4)  # the NK leg's steps (ARC_RL1_PREFLIGHT_MAX_STEPS): play after its resume
    assert mg["redesign"]["max_steps"] == 4


# ---- stage D in the redesign loop (lora-l2: the MTP draft's LoRA trained each step, served after each switch) ----

def write_draft(folder, modules=("a", "b")):
    """rl1_draft.export's two files: the json and 2 tensors per module (header only: draft_files reads no data)."""
    import struct

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "draft_lora.json").write_text(json.dumps({"modules": {m: {} for m in modules}}))
    head = json.dumps({f"{m}.{p}": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
                       for m in modules for p in ("lora_A", "lora_B")}).encode()
    (folder / "draft_lora.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0")


class DraftTrainer(FakeTrainer):
    """The trainer under mg.draft_weight: every saved adapter carries the draft's files (``drop``: a step whose save
    lost them); every call's arguments kept."""

    def __init__(self, log, drop=None):
        super().__init__(log)
        self.drop, self.calls = drop, []

    async def must(self, op, **kw):
        self.calls.append((op, kw))
        out = await super().must(op, **kw)
        if op == "save":
            write_draft(kw["path"])
        return out

    async def call(self, op, **kw):
        self.calls.append((op, kw))
        out = await super().call(op, **kw)
        if op == "step":
            if self.steps != self.drop:
                write_draft(kw["save"])
            out.update(draft_kl=0.4, draft_paccept=0.8, draft_rows=10)
        return out


class DraftLoop(FakeLoop):
    def __init__(self, tmp, drop=None, weight=1.0, **kw):
        super().__init__(tmp, **kw)
        self.cfg["mg"].update(draft_weight=weight, draft_lr_scale=2.0)
        self.drop = drop

    async def start_trainer(self):
        self.log.append("start_trainer")
        self.trainer = DraftTrainer(self.log, self.drop)


@pytest.fixture
def merging(monkeypatch):
    """Every play server merges the draft LoRA (ARC_DRAFT_LORA=1)."""
    envs = {"server1": {"port": 1234, "card": "0", "env": {"ARC_DRAFT_LORA": "1"}}}
    monkeypatch.setattr(rl_seq, "server_envs", lambda loop, extra: envs)
    return envs


def run_seq(loop):
    return asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))


def test_draft_files_ride_every_served_adapter_and_a_lost_one_ends_the_run(tmp_path, harvest_ok, merging, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = DraftLoop(tmp_path, max_steps=2)
    loaded = []
    load = FakeServer.load

    async def spy(self, name, path):
        loaded.append(Path(path))
        await load(self, name, path)

    monkeypatch.setattr(FakeServer, "load", spy)
    assert run_seq(loop) == "max_steps"
    assert [p.name for p in loaded] == ["v0000", "v0001", "v0002"]
    assert all((p / "draft_lora.json").is_file() for p in loaded)  # exported and loaded per step
    assert [r["draft_kl"] for r in loop.feed.of("step")] == [0.4, 0.4]  # the draft's step metrics in the step rows
    lost = DraftLoop(tmp_path / "b", drop=2, max_steps=3)
    with pytest.raises(ValueError, match="draft_lora.json"):
        run_seq(lost)
    assert lost.trainer.steps == 2 and [r["step"] for r in lost.feed.of("seq_phase")] == [1]


def test_a_trained_draft_needs_every_play_server_to_merge_it(tmp_path, harvest_ok, monkeypatch):
    envs = {"server1": {"env": {"ARC_DRAFT_LORA": "1"}}, "server2": {"env": {"ARC_X": "1"}}}
    monkeypatch.setattr(rl_seq, "server_envs", lambda loop, extra: envs)
    with pytest.raises(ValueError, match="server2.*ARC_DRAFT_LORA"):
        run_seq(DraftLoop(tmp_path))
    assert rl_seq.draft_serving(envs) == ["server2: ARC_DRAFT_LORA None (not 1): the trained draft would not be served"]
    assert rl_seq.draft_serving({"server1": {"error": "x"}}) == []  # unreadable: an alert, not a refusal
    assert run_seq(FakeLoop(tmp_path / "off", max_steps=1)) == "max_steps"  # no draft: no requirement


def test_resume_carries_the_draft_state(tmp_path, harvest_ok, merging, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    run_seq(DraftLoop(tmp_path, max_steps=1))
    state = json.loads((tmp_path / "vo" / "seq_state.json").read_text())
    assert state["draft"] == {"weight": 1.0, "lr_scale": 2.0}
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "vo"))
    again = DraftLoop(tmp_path / "b", max_steps=2)
    run_seq(again)
    ops = [op for op, _ in again.trainer.calls]
    assert ops.index("load_checkpoint") < ops.index("save")  # the checkpoint (policy + draft) before v0001 is served
    assert (again.work / "adapters" / "v0001" / "draft_lora.json").is_file()
    row = again.feed.of("seq_resume")[0]
    assert row["step"] == 1 and row["draft"] == {"weight": 1.0, "lr_scale": 2.0} and row["draft_now"] == row["draft"]
    off = FakeLoop(tmp_path / "c", max_steps=2)  # its checkpoint holds the draft's LoRA: a draft-off trainer can't load it
    with pytest.raises(ValueError, match="draft"):
        run_seq(off)


def test_phase_rows_carry_mtp_acceptance_and_gen_rate(tmp_path, harvest_ok, merging, monkeypatch):
    calls = []

    async def counters(url, get=None):
        n = len(calls)
        calls.append(url)
        return {"vllm:generation_tokens_total": 1000.0 * n, "vllm:spec_decode_num_drafts_total": 100.0 * n,
                "vllm:spec_decode_num_draft_tokens_total": 200.0 * n,
                "vllm:spec_decode_num_accepted_tokens_total": 150.0 * n,
                "vllm:spec_decode_num_accepted_tokens_per_pos_total:0": 80.0 * n}

    monkeypatch.setattr(rl_seq, "spec_counters", counters)
    loop = DraftLoop(tmp_path, max_steps=2)
    run_seq(loop)
    rows = [r["mtp"] for r in loop.feed.of("seq_phase")]
    assert len(rows) == 2 and [r["version"] for r in rows] == [0, 1]  # the adapter version each phase played
    for r in rows:
        assert r["accept"] == 0.75 and r["pos1"] == 0.8 and r["mean_len"] == 2.5 and r["gen_tokens"] == 1000
        assert r["drafts"] == 100 and r["gen_tok_s"] > 0
    m = [json.loads(x)["m"] for x in (loop.output / "train-metrics.jsonl").read_text().splitlines()]
    assert [x["infra.draft_acceptance"] for x in m] == [0.75, 0.75] and all("infra.gen_tok_s" in x for x in m)
    assert rl_seq.mtp_row([None], [None], 10.0) is None  # no metrics: no row
    assert rl_seq.mtp_row([{}], [{"vllm:generation_tokens_total": 5.0}], 10.0)["accept"] is None  # MTP off


def test_an_mtp_off_server_alerts_when_the_draft_trains(tmp_path, harvest_ok, merging, monkeypatch):
    seen = []

    async def counters(url, get=None):
        seen.append(url)
        return {"vllm:generation_tokens_total": 1000.0 * len(seen)}

    monkeypatch.setattr(rl_seq, "spec_counters", counters)
    loop = DraftLoop(tmp_path, max_steps=1)
    run_seq(loop)
    assert any("drafted nothing" in r["why"] for r in loop.feed.of("alert"))


def test_freeze_policy_and_init_checkpoint(tmp_path, harvest_ok, merging, monkeypatch):
    loop = DraftLoop(tmp_path, max_steps=2)
    loop.cfg["mg"]["redesign"].update(freeze_policy=True, init_checkpoint="/runs/r15/state/s0017", warmup=0)
    run_seq(loop)
    calls = loop.trainer.calls
    ops = [op for op, _ in calls]
    assert ops.index("load_checkpoint") < ops.index("save")  # v0000 serves the checkpoint's policy + a fresh draft
    assert dict(calls)["load_checkpoint"]["path"] == "/runs/r15/state/s0017"
    assert all(kw.get("policy_frozen") is True for op, kw in calls if op in ("accumulate", "step"))
    assert loop.feed.of("seq_init")[0]["path"] == "/runs/r15/state/s0017"
    plain = DraftLoop(tmp_path / "b", max_steps=1)
    run_seq(plain)
    assert not any("policy_frozen" in kw for _, kw in plain.trainer.calls) and "load_checkpoint" not in plain.log


def test_the_preflight_stand_in_trainer_writes_and_refuses_as_the_real_one(tmp_path, monkeypatch):
    """CPU preflight (arena.fake_trainer) under mg.draft_weight: every adapter it writes passes draft_files(need=True)
    and its step rows carry the draft keys; policy_frozen without a draft is refused, as rl1_trainer does."""
    from arena import fake_trainer
    from arena.rl_loop import DRAFT_ROW, draft_files

    monkeypatch.delenv("RL1_DRAFT_WEIGHT", raising=False)
    assert draft_files(fake_trainer.adapter(tmp_path / "off")) == 0
    with pytest.raises(ValueError, match="policy_frozen"):
        fake_trainer.Fake().run("update", {"attempts": [], "lr": 1e-5, "policy_frozen": True})
    monkeypatch.setenv("RL1_DRAFT_WEIGHT", "1.0")
    assert draft_files(fake_trainer.adapter(tmp_path / "on"), need=True) == 2
    out = fake_trainer.Fake().run("update", {"attempts": [], "lr": 1e-5, "policy_frozen": True, "save": str(tmp_path / "s")})
    assert set(DRAFT_ROW) <= set(out) and out["update_norm"] == 0.0 and draft_files(tmp_path / "s", need=True) == 2


def test_draft_settings_are_checked():
    base = {"streams": 8, "max_steps": 1}
    with pytest.raises(ValueError, match="freeze_policy"):
        rl_seq.settings({"redesign": {**base, "freeze_policy": True}})  # nothing would train
    with pytest.raises(ValueError, match="freeze_policy"):
        rl_seq.settings({"draft_weight": 1.0, "redesign": {**base, "freeze_policy": 1}})
    with pytest.raises(ValueError, match="freeze_policy"):
        rl_seq.settings({"draft_weight": 1.0, "redesign": {**base, "freeze_policy": True, "lr_probe": {
            "rates": [1e-5], "steps": 1, "probes": [{"folder": "/x", "labels": ["a"]}]}}})
    with pytest.raises(ValueError, match="init_checkpoint"):
        rl_seq.settings({"redesign": {**base, "init_checkpoint": "state/s0017"}})
    assert rl_seq.settings({"draft_weight": 1.0, "redesign": {**base, "freeze_policy": True,
                                                              "init_checkpoint": "/runs/x/state/s0017"}})


def test_phase_start_backoff_cuts_slots_after_a_preemption_or_high_kv():
    """an overnight run backs off on its own: a phase with any preemption or KV peak > 0.85 starts
    the next at active_tries - 20 per server (floor 120), logged; a fresh knob put resets the cut."""
    st = {}
    k, row = rl_seq.backoff({}, None, st, 200)
    assert k == {} and row is None  # first phase, no guard: as configured
    k, row = rl_seq.backoff({}, {"kv_peak": 0.6, "preemptions": 0}, st, 200)
    assert k == {} and row is None
    k, row = rl_seq.backoff({}, {"kv_peak": 0.9, "preemptions": 0}, st, 200)
    assert k == {"active_tries": 180} and row["cut"] == 20
    k, row = rl_seq.backoff({}, {"kv_peak": 0.5, "preemptions": 3}, st, 180)
    assert k == {"active_tries": 160} and row["cut"] == 40
    for slots in (160, 140, 120):
        k, row = rl_seq.backoff({}, {"kv_peak": 0.95, "preemptions": 1}, st, slots)
    assert k.get("active_tries", 120) == 120 and row is None  # at the floor: no further cut
    k, row = rl_seq.backoff({"active_tries": 220}, {"kv_peak": 0.5, "preemptions": 0}, st, 120)
    assert k == {"active_tries": 220} and row is None and st["cut"] == 0  # a put is a new base
    k, row = rl_seq.backoff({"active_tries": 220}, {"kv_peak": 0.86}, st, 220)
    assert k == {"active_tries": 200}  # the same put again keeps the cut
    st = {}
    k, row = rl_seq.backoff({}, {"preemptions": 1}, st, 100)
    assert row is None and k == {}  # under the floor already: unchanged


def test_server_guard_reads_kv_peak_and_preemptions_over_a_phase():
    """One poll at the phase start, one at its stop (the wait between ends at stop): KV max over servers and polls,
    preemptions as each server's counter delta."""
    reads = iter([0.2, 0.7, 0.9, 0.3])
    pre = {"a": iter([5, 7]), "b": iter([0, 1])}

    async def get(url):
        name = "a" if "//a" in url else "b"
        return f"vllm:kv_cache_usage_perc {next(reads)}\nvllm:num_preemptions_total {next(pre[name])}\n"

    async def go():
        stop = asyncio.Event()
        task = asyncio.ensure_future(rl_seq.server_guard(["http://a/v1", "http://b/v1"], stop, every=30, get=get))
        await asyncio.sleep(0.05)
        stop.set()
        return await asyncio.wait_for(task, 2)

    out = asyncio.run(go())
    assert out == {"kv_peak": 0.9, "preemptions": 3, "polls": 4, "blind_polls": 0, "read_s_max": 0.0}


def test_the_loop_backs_off_before_the_knobs_apply():
    import inspect
    src = inspect.getsource(rl_seq.run)
    assert src.index("backoff(live_knobs(), guard") < src.index("apply_knobs(loop, seq, knobs)") < src.index(
        "seq.backlog.start_phase()")
    assert src.index("guard = await guard_task") < src.index("ctl.to_train()")


def test_each_accumulate_carries_the_next_groups_specs_ahead(tmp_path):
    """Encode-ahead (D1 26 Oct: host gaps inside the update): every accumulate call but the last names the next group
    with specs as ``ahead`` (rl1_trainer.ahead_start encodes it during this call); a group with no specs is skipped."""
    loop = FakeLoop(tmp_path)
    calls = []

    class T(FakeTrainer):
        async def call(self, op, **kw):
            calls.append((op, kw))
            return await super().call(op, **kw)
    loop.trainer = T(loop.log)
    specs = {"g1": [{"name": "a"}], "g2": [], "g3": [{"name": "c"}]}
    loop.v7_prepare = lambda g: ({"gid": g.gid, "set": "x", "tries": 1, "trained": len(specs[g.gid])}, specs[g.gid])

    def grp(gid):
        return type("G", (), {"gid": gid, "unit": "x"})()

    asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s1"))
    acc = [kw for op, kw in calls if op == "accumulate"]
    assert [a["attempts"] for a in acc] == [[{"name": "a"}], [{"name": "c"}]]
    assert acc[0]["ahead"] == [{"name": "c"}] and "ahead" not in acc[1]
    assert [op for op, _ in calls][-1] == "step"


def test_one_call_sends_every_groups_windows_in_one_accumulate(tmp_path):
    """mg.redesign.one_call: one accumulate per update with every group's specs
    (first=True, no ahead), then the step; each group still gets its row."""
    loop = FakeLoop(tmp_path)
    loop.cfg["mg"]["redesign"] = {**(loop.cfg["mg"].get("redesign") or {}), "one_call": True}
    calls = []

    class T(FakeTrainer):
        async def call(self, op, **kw):
            calls.append((op, kw))
            return await super().call(op, **kw)
    loop.trainer = T(loop.log)
    specs = {"g1": [{"name": "a"}, {"name": "b"}], "g2": [], "g3": [{"name": "c"}]}
    loop.v7_prepare = lambda g: ({"gid": g.gid, "set": "x", "tries": 1, "trained": len(specs[g.gid])}, specs[g.gid])

    def grp(gid):
        return type("G", (), {"gid": gid, "unit": "x"})()

    out = asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s1"))
    acc = [kw for op, kw in calls if op == "accumulate"]
    assert len(acc) == 1 and acc[0]["attempts"] == [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    assert acc[0]["first"] is True and "ahead" not in acc[0] and [op for op, _ in calls][-1] == "step"
    assert out["groups"] == 3 and out["trained"] == 3
    one = [r for r in loop.feed.of("seq_accumulate") if r.get("one_call")]
    assert len(one) == 1 and one[0]["groups"] == 2 and one[0]["windows"] == 3


def test_fake_trainer_answers_save_join():  # mg.async_adapter_save: the loop joins before the servers load (D2 N1)
    from arena import fake_trainer

    assert fake_trainer.Fake(16384).run("save_join", {}) == {"pending": False}


def test_early_ahead_hands_the_first_groups_specs_to_the_trainer_folder(tmp_path):
    """trainer_fast.early_ahead (D2 27 Oct: the first window waited 22.8 s on its encode): when a play phase ends the
    first group with exact-start rows is prepared once and its specs written to <work>/trainer/ahead-early.json;
    train_step then accumulates the same (cached) specs and prepares no group twice."""
    import json

    loop = ProbeLoop(tmp_path)
    loop.v7 = {}
    rows = {"g1": [{"label": "a000", "restore": {"state": {"ok": False}}}],  # not an exact start: skipped
            "g2": [{"label": "a001"}, {"label": "a002"}], "g3": [{"label": "a003"}]}
    rl_seq_exact = rl_seq.exact_start
    rl_seq.exact_start = lambda r: "restore" not in r
    try:
        def grp(gid):
            return type("G", (), {"gid": gid, "unit": "x"})()

        groups = [(grp(k), v) for k, v in rows.items()]
        assert rl_seq.early_ahead(loop, groups, "s0001-1") == 2
        got = json.loads((tmp_path / "work" / "trainer" / "ahead-early.json").read_text())
        assert got["token"] == "s0001-1" and got["gid"] == "g2" and got["ahead"] == [{"name": "a001"}, {"name": "a002"}]
        calls = []

        class T(FakeTrainer):
            async def call(self, op, **kw):
                calls.append((op, kw))
                return await super().call(op, **kw)
        loop.trainer = T(loop.log)
        asyncio.run(rl_seq.train_step(loop, groups[1:], 1e-4, tmp_path / "s1"))
        acc = [kw for op, kw in calls if op == "accumulate"]
        assert acc[0]["attempts"] == got["ahead"] and loop.prepares == {"g2": 1, "g3": 1}
        assert not loop.v7["rows"] and not loop.v7["prepared"]
        assert rl_seq.early_ahead(loop, groups[:1], "s0002-1") == 0  # no exact-start rows: nothing written
    finally:
        rl_seq.exact_start = rl_seq_exact


def test_replay_groups_load_saved_groups_in_folder_order(tmp_path):
    """mg.redesign.replay_groups (update legs, 27 Oct): saved group folders become groups without play; a group with
    too few finished tries (or a try without its rollout) is skipped; settings validate the spec."""
    import json

    root = tmp_path / "rl"

    def saved(name, n, missing=0):
        folder = root / name
        rows = []
        for i in range(n):
            label = f"a{i:03d}"
            game, level = name.split("-", 1)[1].rsplit("-L", 1)
            rows.append({"label": label, "game": f"{game}-7f", "point": f"{game}-game-L{level}-{game}-L{level}",
                         "outcome": "cleared", "restore": {"ok": True}})
            if i >= missing:
                (folder / "attempts" / label).mkdir(parents=True)
                (folder / "attempts" / label / "rollout.jsonl").write_text("{}\n")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "level-bench.json").write_text(json.dumps({"attempts": rows}))

    saved("p1g1-sb26-L1", 12)
    saved("p1g2-sb26-L2", 12, missing=8)  # 4 rollouts on disk: under min_tries
    saved("p1g3-ls20-L3", 6)
    got = rl_seq.replay_groups({"roots": [str(root)], "groups": 5, "min_tries": 6})
    assert [(g.gid, g.unit, len(r)) for g, r in got] == [("r01-p1g1-sb26-L1", "sb26-L1", 12),
                                                          ("r02-p1g3-ls20-L3", "ls20-L3", 6)]
    assert len(rl_seq.replay_groups({"roots": [str(root)], "groups": 1})) == 1
    rep = rl_seq.replay_groups({"roots": [str(root)], "groups": 5, "min_tries": 6, "repeat": True})
    assert [g.gid for g, _ in rep] == ["r01-p1g1-sb26-L1", "r02-p1g3-ls20-L3", "r03-p1g1-sb26-L1", "r04-p1g3-ls20-L3",
                                       "r05-p1g1-sb26-L1"]
    two = {"roots": [str(root)], "groups": 3, "min_tries": 6, "repeat": True, "updates": 2}  # U3: update u's slice
    assert [g.gid for g, _ in rl_seq.replay_groups(two, 0)] == ["r01-p1g1-sb26-L1", "r02-p1g3-ls20-L3",
                                                                 "r03-p1g1-sb26-L1"]
    assert [g.gid for g, _ in rl_seq.replay_groups(two, 1)] == ["r04-p1g3-ls20-L3", "r05-p1g1-sb26-L1",
                                                                 "r06-p1g3-ls20-L3"]
    mg = {"redesign": {"streams": 8, "max_steps": 1, "replay_groups": {"roots": [str(root)], "groups": 16}}}
    assert rl_seq.settings(mg)["replay_groups"]["groups"] == 16
    assert rl_seq.settings({"redesign": {"streams": 8, "max_steps": 2, "replay_groups": two}})["replay_groups"] == two
    # updates k without repeat: distinct groups for every update (no early stop at one update's count)
    assert [g.gid for g, _ in rl_seq.replay_groups({"roots": [str(root)], "groups": 1, "min_tries": 6, "updates": 2}, 1)] \
        == ["r02-p1g3-ls20-L3"]
    # longest (U4): groups ordered by their tries' max prompt tokens
    bench = root / "p1g3-ls20-L3" / "level-bench.json"
    rows = json.loads(bench.read_text())["attempts"]
    rows[0]["max_prompt_tokens"] = 90000
    bench.write_text(json.dumps({"attempts": rows}))
    got = rl_seq.replay_groups({"roots": [str(root)], "groups": 2, "min_tries": 6, "longest": True})
    assert [g.unit for g, _ in got] == ["ls20-L3", "sb26-L1"]
    for bad in ({"roots": []}, {"roots": ["x"], "groups": 0}, {"roots": ["x"], "groups": 2, "extra": 1},
                {"roots": ["x"], "groups": 2, "updates": 0}):
        with pytest.raises(ValueError, match="replay_groups"):
            rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1, "replay_groups": bad}})


def test_replay_groups_leg_trains_saved_groups_through_the_runs_switch(tmp_path, harvest_ok, monkeypatch):
    """Update leg (27 Oct): with replay_groups the first step plays nothing; the saved groups go through the run's
    own switch (drain, sleep, unpark), accumulate, step, checkpoint and the switch back (adapter served); with
    early_ahead the first group's specs reach the trainer folder before the switch."""
    import json

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    root = tmp_path / "saved"
    for name in ("p1g1-sb26-L1", "p1g2-ls20-L2"):
        rows = []
        for i in range(3):
            label = f"a{i:03d}"
            game, level = name.split("-", 1)[1].rsplit("-L", 1)
            (root / name / "attempts" / label).mkdir(parents=True)
            (root / name / "attempts" / label / "rollout.jsonl").write_text("{}\n")
            rows.append({"label": label, "game": f"{game}-7f", "point": f"{game}-game-L{level}-{game}-L{level}",
                         "outcome": "budget", "token_budget": 5000 + i})
        (root / name / "level-bench.json").write_text(json.dumps({"attempts": rows}))
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.cfg["mg"]["redesign"]["replay_groups"] = {"roots": [str(root)], "groups": 16}
    loop.cfg["mg"]["trainer_fast"] = {"early_ahead": True}
    why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "max_steps" and loop.trainer.steps == 1
    assert loop.feed.of("seq_replay")[0]["groups"] == 2 and loop.feed.of("seq_train")[0]["groups"] == 2
    assert loop.feed.of("seq_early_ahead")[0]["specs"] == 3
    assert loop.accumulated == ["r01-p1g1-sb26-L1", "r02-p1g2-ls20-L2"]  # each prepared once (the first early)
    assert loop.feed.of("step")[0]["groups"] == 2 and len(loop.feed.of("seq_phase")) == 1
    i = loop.log.index("unpark")
    assert loop.log[i - 1] == "sleep" and loop.log[loop.log.index("step", i) + 1] == "checkpoint"
    assert (tmp_path / "vo" / "adapters" / "v0001").is_dir()
    assert loop.cfg["mg_sets"]["ls20-L2"] == {"name": "ls20-L2", "game": "ls20", "level": 2, "role": "train",
                                              "fresh": False, "cap": 5002}


def test_play_only_state_seeds_a_new_runs_backlog_and_step_0(tmp_path, harvest_ok, monkeypatch):
    """a play_only run writes seq_state.json (backlog, curriculum, every phase's full group
    folders) at each phase end; seed_from starts a fresh run from it: its saved starts and curriculum carry over (in-flight
    groups' starts go back, games outside the roster go), and step 0 trains the seed's first phase's first groups."""
    import json

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo1"))
    loop = FakeLoop(tmp_path / "a", max_steps=1, clear_p=0.6)
    loop.cfg["mg"]["redesign"].update(play_only=True, admit_log_s=0.01)
    real, calls = loop.mg_play, []

    async def late(name, spec, starts, tries, **kw):
        calls.append(name)
        if len(calls) > 12:
            raise TimeoutError("Only 239 s left for the level bench")
        await asyncio.sleep(0.02)
        return await real(name, spec, starts, tries, **kw)

    loop.mg_play = late
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    state = json.loads((tmp_path / "vo1" / "seq_state.json").read_text())
    assert state["play_only"] and state["version"] == 0 and len(state["phases"][0]) == 4  # phase_groups 4
    saved = [s for s in state["backlog"]["starts"] if not s["l1"]]
    assert saved and state["backlog"]["curriculum"]["fresh"]
    for folder in state["phases"][0]:  # the benches' reports (FakeLoop's mg_play writes none)
        name = Path(folder).name
        game, level = name.split("-", 1)[1].rsplit("-L", 1)
        rows = []
        for i in range(3):
            (Path(folder) / "attempts" / f"a{i:03d}").mkdir(parents=True, exist_ok=True)
            (Path(folder) / "attempts" / f"a{i:03d}" / "rollout.jsonl").write_text("{}\n")
            rows.append({"label": f"a{i:03d}", "game": f"{game}-7f", "point": f"{game}-game-L{level}-{game}-L{level}",
                         "outcome": "budget", "token_budget": 5000})
        (Path(folder) / "level-bench.json").write_text(json.dumps({"attempts": rows}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo2"))
    seeded = FakeLoop(tmp_path / "b", max_steps=1)
    seeded.cfg["mg"]["redesign"]["seed_from"] = {"root": str(tmp_path / "vo1"), "groups": 3}
    why = asyncio.run(asyncio.wait_for(rl_seq.run(seeded, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert why == "max_steps" and seeded.trainer.steps == 1
    seed = seeded.feed.of("seq_seed")[0]
    assert seed["groups"] == 3 and seed["outside_roster"] == 0
    assert seeded.feed.of("seq_replay")[0]["groups"] == 3
    assert seeded.feed.of("seq_train")[0]["groups"] == 3 and seeded.feed.of("step")[0]["groups"] == 3
    assert seeded.accumulated == [f"s{i + 1:02d}-{Path(f).name}" for i, f in enumerate(state["phases"][0][:3])]
    for bad in ({"root": "rel"}, {"root": "/x", "groups": 0}, {"root": "/x", "groups": 2, "x": 1}):
        with pytest.raises(ValueError, match="seed_from"):
            rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1, "seed_from": bad}})
    with pytest.raises(ValueError, match="seed_from"):
        rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1, "play_only": True,
                                      "seed_from": {"root": "/x", "groups": 2}}})


def test_sampler_reads_the_prefix_cache_counters():
    """G-long (27 Oct): the live sampler keeps vLLM's prefix-cache queries / hits (summed over engines and servers)."""
    modal_rl1 = pytest.importorskip("modal_rl1")
    text = ('# HELP x\nvllm:prefix_cache_queries_total{engine="0"} 1000.0\nvllm:prefix_cache_hits_total{engine="0"} 900.0\n'
            'vllm:prefix_cache_queries_total{engine="1"} 500.0\nvllm:prefix_cache_hits_total{engine="1"} 400.0\n'
            'vllm:kv_cache_usage_perc{engine="0"} 0.5\n')
    one = modal_rl1.vllm_metrics(text)
    assert one["prefix_cache_queries_total"] == 1500.0 and one["prefix_cache_hits_total"] == 1300.0
    both = modal_rl1.vllm_totals([one, one])
    assert both["prefix_cache_hits_total"] == 2600.0 and both["prefix_cache_queries_total"] == 3000.0


def test_dwell_snapshot_and_admit_log_split_live_tries_and_engine_gap(tmp_path):
    """Dwell export: live tries by state, their seconds by kind, finished tries' seconds, the engine
    gap (sending - running - waiting) and loop lag on seq_admit rows."""
    import asyncio
    from types import SimpleNamespace

    def slot(tool=None, status="closed", parked=False, reqs=(), inf=0.0, tl=0.0, act=0.0):
        clock = SimpleNamespace(_tool_started=tool, parked=parked, start=0.0, active_seconds=lambda: act,
                                inference_seconds=lambda: inf, tool_seconds=lambda: tl)
        return Slot(game=SimpleNamespace(clock=clock, admission_attempts=[{"status": status}],
                                         requests=[{"gate_wait_seconds": g} for g in reqs]),
                    next_estimate=1000, floor_tokens=0)

    class Slot(SimpleNamespace):
        __hash__ = object.__hash__

    pool = level_bench.SharedSlots(10, shared_gate=True)
    a, b, c, d, e = slot(inf=5, act=10), slot(status="waiting", reqs=(1.0, 2.0), act=4), slot(tool=1.0, tl=3, act=6), \
        slot(parked=True), slot(act=2)
    for s in (a, b, c, d, e):
        pool.enter(s)
    pool.begin(a)
    snap = rl_seq.dwell_snapshot(pool)
    assert snap["now"] == {"sending": 1, "gate_wait": 1, "tool": 1, "parked": 1, "other": 1}
    assert snap["live_s"] == {"wall": 22.0, "inference": 5.0, "tool": 3.0, "gate_wait": 3.0, "harness": 11.0, "tries": 5}
    done = rl_seq.dwell_add(dict.fromkeys(rl_seq.DWELL_KEYS, 0.0), {"wall_seconds": 100, "inference_seconds": 60,
                                                                     "tool_seconds": 10, "gate_wait_seconds": 5})
    assert done["harness"] == 25.0 and done["tries"] == 1

    rows = []
    backlog = SimpleNamespace(streams=10, parked=0, groups={})
    seq = SimpleNamespace(urls=["http://a/v1"], on_server={}, backlog=backlog, dwell_done=done)
    loop = SimpleNamespace(feed=SimpleNamespace(emit=lambda kind, **kw: rows.append((kind, kw))),
                           loop_lag={"tick": 0.0, "round": 0.0, "admit": 1.25})
    import time as _time

    rl_seq.LAST_METRICS["http://a/v1"] = (_time.time(), {"vllm:num_requests_running": 0.0,
                                                         "vllm:num_requests_waiting": 0.0})
    old = level_bench.SHARED_SLOTS_BY_URL
    level_bench.SHARED_SLOTS_BY_URL = {"http://a/v1": pool}

    async def go():
        task = asyncio.ensure_future(rl_seq.admit_log(loop, seq, 0.01))
        while not rows:
            await asyncio.sleep(0.01)
        task.cancel()

    try:
        asyncio.run(go())
    finally:
        level_bench.SHARED_SLOTS_BY_URL = old
        rl_seq.LAST_METRICS.clear()
    kind, row = rows[0]
    eng = row["servers"][0]["engine"]
    assert kind == "seq_admit" and (eng["running"], eng["waiting"], eng["gap"]) == (0, 0, 1)
    assert row["loop_lag_s"] == 1.25 and loop.loop_lag["admit"] == 0.0 and row["dwell_done"]["tries"] == 1
    assert row["servers"][0]["dwell"]["now"]["sending"] == 1


def test_guard_goes_blind_after_three_failed_polls_and_backoff_fails_closed():
    """(G-long v4 N1/N2: the guard read /metrics once in 240 s): a server unread BLIND_AFTER polls
    in a row is blind (on_blind holds admission) until a read comes back; a phase that was blind cuts like a
    preemption; the poll interval must exceed the read timeout."""
    import asyncio

    calls, events = {"n": 0}, []

    async def get(url):
        calls["n"] += 1
        if 2 <= calls["n"] <= 5:  # polls 2-5 fail: blind from poll 4, sees again at 6
            raise OSError("down")
        return "vllm:kv_cache_usage_perc 0.5\nvllm:num_preemptions_total 0\n"

    async def go():
        stop = asyncio.Event()
        task = asyncio.ensure_future(rl_seq.server_guard(["http://a/v1"], stop, every=0.02, get=get, timeout=0.01,
                                                         on_blind=lambda u, on: events.append((u, on, calls["n"]))))
        while calls["n"] < 8:
            await asyncio.sleep(0.01)
        stop.set()
        return await task

    try:
        out = asyncio.run(go())
    finally:
        rl_seq.LAST_METRICS.clear()
    assert events == [("http://a/v1", True, 4), ("http://a/v1", False, 6)] and out["blind_polls"] >= 2

    async def down(url):
        calls["n"] += 1
        raise OSError("down")

    async def ends_blind():
        stop = asyncio.Event()
        task = asyncio.ensure_future(rl_seq.server_guard(["http://a/v1"], stop, every=0.02, get=down, timeout=0.01,
                                                         on_blind=lambda u, on: events.append((u, on))))
        while calls["n"] < 12:
            await asyncio.sleep(0.01)
        stop.set()
        return await task

    events.clear()
    asyncio.run(ends_blind())
    assert events == [("http://a/v1", True), ("http://a/v1", None)]  # released at phase end, never "seen"
    with pytest.raises(ValueError, match="exceed"):
        asyncio.run(rl_seq.server_guard(["u"], asyncio.Event(), every=5, timeout=5))
    knobs, row = rl_seq.backoff({}, {"preemptions": 0, "kv_peak": 0.1, "blind_polls": 2}, {}, 280)
    assert row is not None and knobs["active_tries"] == 260


def test_stub_metrics_outage_runs_from_the_first_gameplay_request(monkeypatch):
    from arena import replay_stub

    state = replay_stub.State(corpus=None, speed=0)
    state.metrics_outage = [40.0, 65.0]
    clock = {"t": 1000.0}
    monkeypatch.setattr(replay_stub.time, "time", lambda: clock["t"])
    assert not state.outage()  # no play yet: never out
    state.play_began = 1000.0
    for t, out in ((1039.0, False), (1040.0, True), (1104.9, True), (1105.0, False)):
        clock["t"] = t
        assert state.outage() is out, t


def test_preflight_leg_guard_acks_a_resubmit(tmp_path, monkeypatch):
    """real-v5 DB / U4b N1 (27 Oct): a hand-stopped CPU preflight leg blocked its own resubmit; a new call of a preflight
    leg acks itself, a Modal auto-restart of it (same call id) and a run's relaunch stay blocked."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    import modal

    monkeypatch.setattr(modal_rl1, "BUNDLE", Path(__file__).resolve().parents[1])
    monkeypatch.setattr(modal_rl1, "runs", type("V", (), {"commit": staticmethod(lambda: None)})())
    monkeypatch.delenv("ARC_RESTART_ACK", raising=False)
    calls = iter(["c1", "c2", "c3", "c3", "r1", "r2"])
    monkeypatch.setattr(modal, "current_function_call_id", lambda: next(calls))
    g = lambda rel: modal_rl1.restart_guard(rel, root=str(tmp_path))
    g("validation/preflight/x-v1N1").start()  # c1 dies (no end)
    assert g("validation/preflight/x-v1N1").start().restart  # c2: a new call of the leg proceeds
    blocked = sys.modules["arc_runops"].restart.RestartBlocked
    g("validation/preflight/x-v1N1").start()  # c3 dies
    with pytest.raises(blocked):
        g("validation/preflight/x-v1N1").start()  # c3 again: Modal's auto-restart, no checkpoint
    g("validation/x-v1").start()  # r1 dies
    with pytest.raises(blocked):
        g("validation/x-v1").start()  # r2: a run's relaunch still needs the ack


def test_one_call_prepares_groups_during_play(tmp_path):
    """one_call + early_ahead (U4b 27 Oct: the one call's expand of 192 tries held both cards idle 54 s, 22%): each
    group is prepared as it finishes during play and the trainer folder's ahead file carries every spec prepared so
    far; at the phase end early() adds the rest; each group is prepared once and the step accumulates the same specs."""
    import json

    loop = FakeLoop(tmp_path, max_steps=2)
    loop.cfg["mg"]["redesign"]["one_call"] = True
    loop.cfg["mg"]["trainer_fast"] = {"early_ahead": True}
    writes = []
    real = rl_seq.write_ahead

    def spy(lp, specs, token, gid=None):
        writes.append((token, len(specs)))
        return real(lp, specs, token, gid)
    rl_seq.write_ahead = spy
    rl_seq.AHEAD_EVERY_S, every = 0.0, rl_seq.AHEAD_EVERY_S
    try:
        why = asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 60))
    finally:
        rl_seq.write_ahead, rl_seq.AHEAD_EVERY_S = real, every
    assert why == "max_steps"
    early = loop.feed.of("seq_early_ahead")
    assert len(early) == 2 and all(e["during_play"] >= 1 and e["groups"] >= e["during_play"] for e in early), early
    assert len(loop.accumulated) == len(set(loop.accumulated))  # no group prepared twice
    one = [r for r in loop.feed.of("seq_accumulate") if r.get("one_call")]
    assert [r["windows"] for r in one] == [e["specs"] for e in early]  # the call's windows = the ahead file's
    assert len(writes) > len(early)  # rewritten during play, not only at the phase end
    at_end = [(t, n) for t, n in writes if "-e" in t]  # early()'s writes: this update's groups only, in its order
    assert [n for _, n in at_end] == [e["specs"] for e in early] and at_end[-1][0].startswith("s0002-e")
    assert json.loads((tmp_path / "work" / "trainer" / "ahead-early.json").read_text())["ahead"]
    assert not [a for a in loop.feed.of("alert") if "ahead" in str(a.get("why"))]


def test_gpu_util_between_reads_the_gpulog_window(tmp_path):
    log = tmp_path / "gpulog.txt"
    log.write_text("0, 0 %, 70 MiB, 275040 MiB 1, 0 %, 4 MiB, 275040 MiB 100\n"
                   "0, 90 %, 70 MiB, 275040 MiB 1, 70 %, 4 MiB, 275040 MiB 101\n"
                   "0, 100 %, 70 MiB, 275040 MiB 1, 100 %, 4 MiB, 275040 MiB 102\nbroken\n")
    assert rl_seq.gpu_util_between(101, 102, log) == 90.0
    assert rl_seq.gpu_util_between(100, 102, log) == 60.0 and rl_seq.gpu_util_between(200, 300, log) is None
    assert rl_seq.gpu_util_between(0, 1, tmp_path / "none.txt") is None


def test_settings_refuse_held_out_games():
    with pytest.raises(ValueError, match="held-out"):
        rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1, "games": ["sb26", "sp80"]}})
    assert rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1}})["streams"] == 8  # default: rl_backlog.TRAIN


class SavingLoop(FakeLoop):
    """FakeLoop whose groups write their folders as level benches do (level-bench.json, attempts/<label>/rollout.jsonl)."""

    def __init__(self, tmp, max_steps=3, trainer=None):
        super().__init__(tmp, max_steps=max_steps)
        self.trainer_cls = trainer or FakeTrainer

    async def start_trainer(self):
        self.log.append("start_trainer")
        self.trainer = self.trainer_cls(self.log)

    async def mg_play(self, name, spec, starts, tries, **kw):
        folder, rows, play = await super().mg_play(name, spec, starts, tries, **kw)
        for r in rows:
            r["point"] = f"{spec['game']}-game-L{spec['level']}-{spec['game']}-L{spec['level']}"
            (folder / "attempts" / r["label"]).mkdir(parents=True, exist_ok=True)
            (folder / "attempts" / r["label"] / "rollout.jsonl").write_text("{}\n")
        (folder / "level-bench.json").write_text(json.dumps({"attempts": rows}))
        return folder, rows, play


def test_update_failure_resumes_from_saved_tries_without_replay(tmp_path, harvest_ok, monkeypatch):
    """each finished group is copied to the volume at once; before an update its groups are listed in
    seq_pending.json; when that update fails, the resumed run trains those saved tries first, with no play."""
    class Dies(FakeTrainer):
        async def call(self, op, **kw):
            if op == "step" and self.steps == 1:  # the second update fails
                self.log.append(op)
                return {"ok": False, "error": "a bug in the update"}
            return await super().call(op, **kw)

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = SavingLoop(tmp_path / "a", max_steps=3, trainer=Dies)
    with pytest.raises(Exception, match="a bug in the update"):
        asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    pend = json.loads((tmp_path / "vo" / "seq_pending.json").read_text())
    assert pend["step"] == 2 and len(pend["groups"]) == 4
    assert all((tmp_path / "vo" / g["rel"] / "level-bench.json").is_file() for g in pend["groups"])
    assert all(g["rel"].startswith("arena/trials/rl-loop/rl/") for g in pend["groups"])
    assert json.loads((tmp_path / "vo" / "seq_state.json").read_text())["step"] == 1
    assert loop.feed.of("seq_pending")[-1] == {**loop.feed.of("seq_pending")[-1], "step": 2, "groups": 4, "full": 4}

    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "vo"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo2"))
    again = SavingLoop(tmp_path / "b", max_steps=3)
    asyncio.run(asyncio.wait_for(rl_seq.run(again, gate=phase.PhaseGate(inner=Upstream())), 30))
    res, rep = again.feed.of("seq_resume")[0], again.feed.of("seq_replay")[0]
    assert res["step"] == 1 and res["pending_groups"] == 4 and rep["resumed_pending"] is True
    assert rep["gids"] == [g["gid"] for g in pend["groups"]]  # the same groups, same ids
    steps = again.feed.of("step")
    assert [s["step"] for s in steps] == [2, 3] and steps[0]["groups"] == 4
    kinds = [e["kind"] for e in again.feed.rows]
    # no play before the resumed update; step 3's groups are new plays or, when run 1's phase finished more groups
    # than one update takes before its stop, carried full groups (CI d2d776755: no seq_group at all, timing)
    assert "seq_group" not in kinds[:kinds.index("step")]
    # a resume whose pending file is for another step plays as before
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "vo2"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo3"))
    third = SavingLoop(tmp_path / "c", max_steps=4)
    asyncio.run(asyncio.wait_for(rl_seq.run(third, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert third.feed.of("seq_resume")[0]["pending_groups"] == 0 and not third.feed.of("seq_replay")


def test_first_update_failure_resumes_pending_only(tmp_path, harvest_ok, monkeypatch):
    """real-v5 27 Oct: stopped in its first update (only s0001.partial, no seq_state.json); the resume starts the
    trainer fresh at step 0 and trains the saved tries of update 1 first, with no play before it."""
    class Dies(FakeTrainer):
        async def call(self, op, **kw):
            if op == "step" and self.steps == 0:
                self.log.append(op)
                return {"ok": False, "error": "stopped in update 1"}
            return await super().call(op, **kw)

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = SavingLoop(tmp_path / "a", max_steps=2, trainer=Dies)
    with pytest.raises(Exception, match="stopped in update 1"):
        asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    pend = json.loads((tmp_path / "vo" / "seq_pending.json").read_text())
    assert pend["step"] == 1 and not (tmp_path / "vo" / "seq_state.json").exists()
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "vo"))
    assert rl_seq.pending_only(tmp_path / "vo") and rl_seq.resume_state_exists()
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo2"))
    again = SavingLoop(tmp_path / "b", max_steps=2)
    asyncio.run(asyncio.wait_for(rl_seq.run(again, gate=phase.PhaseGate(inner=Upstream())), 30))
    res, rep = again.feed.of("seq_resume")[0], again.feed.of("seq_replay")[0]
    assert res["step"] == 0 and res["pending_only"] and res["pending_groups"] == len(pend["groups"])
    assert rep["gids"] == [g["gid"] for g in pend["groups"]]
    steps = again.feed.of("step")
    assert [s["step"] for s in steps] == [1, 2]
    kinds = [e["kind"] for e in again.feed.rows]
    assert kinds.index("seq_group") > kinds.index("step")


def test_resume_takes_pending_from_a_later_root(tmp_path, harvest_ok, monkeypatch):
    """real-v7 27 Oct: resumed from real-v6 (state step 1), played update 2's groups and was stopped at the switch:
    its root holds seq_pending.json (step 2) and the group folders but no seq_state. A resume from both roots loads
    real-v6's state and trains real-v7's groups as step 2 with no play before it."""
    class Dies(FakeTrainer):
        async def call(self, op, **kw):
            if op == "step":
                self.log.append(op)
                return {"ok": False, "error": "stopped before update 2"}
            return await super().call(op, **kw)

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v6"))
    asyncio.run(asyncio.wait_for(rl_seq.run(SavingLoop(tmp_path / "a", max_steps=1),
                                            gate=phase.PhaseGate(inner=Upstream())), 30))
    (tmp_path / "v6" / "seq_play.json").unlink()  # the pending path alone (play_restore, carried: their own test)
    st = json.loads((tmp_path / "v6" / "seq_state.json").read_text())
    (tmp_path / "v6" / "seq_state.json").write_text(json.dumps({**st, "mirrored": {}}))
    assert json.loads((tmp_path / "v6" / "seq_state.json").read_text())["step"] == 1
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "v6"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v7"))
    v7 = SavingLoop(tmp_path / "b", max_steps=3, trainer=Dies)
    with pytest.raises(Exception, match="stopped before update 2"):
        asyncio.run(asyncio.wait_for(rl_seq.run(v7, gate=phase.PhaseGate(inner=Upstream())), 30))
    (tmp_path / "v7" / "seq_state.json").unlink(missing_ok=True)  # real-v7's root: no state landed
    rl = tmp_path / "v7" / "arena/trials/rl-loop/rl"  # its seq_pending.json as put_pending writes it at the switch
    done = [r for r in v7.feed.of("group_done") if r.get("tries") and (rl / f"{r['gid']}-{r['set']}").is_dir()]
    assert done
    pend = {"step": 2, "version": 1, "groups": [
        {"gid": r["gid"], "game": r["set"].rpartition("-L")[0], "level": int(r["set"].rpartition("-L")[2]),
         "version": int(r["version"]), "rel": f"arena/trials/rl-loop/rl/{r['gid']}-{r['set']}"} for r in done]}
    (tmp_path / "v7" / "seq_pending.json").write_text(json.dumps(pend))
    done = [r["gid"] for r in done]

    monkeypatch.setenv("ARC_RL1_RESUME", f"{tmp_path / 'v6'},{tmp_path / 'v7'}")
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v8"))
    v8 = SavingLoop(tmp_path / "c", max_steps=3)
    asyncio.run(asyncio.wait_for(rl_seq.run(v8, gate=phase.PhaseGate(inner=Upstream())), 30))
    res, rep = v8.feed.of("seq_resume")[0], v8.feed.of("seq_replay")[0]
    assert res["step"] == 1 and res["root"] == str(tmp_path / "v6") and res["pending_root"] == str(tmp_path / "v7")
    assert res["pending_groups"] == len(done) and rep["gids"] == done
    steps = v8.feed.of("step")
    assert steps[0]["step"] == 2 and steps[0]["groups"] == len(done)
    kinds = [e["kind"] for e in v8.feed.rows]
    assert kinds.index("seq_group") > kinds.index("step")  # no play before the resumed update


def test_judge_kill_in_update_legs():
    """leg a (die_step) saved its phase's groups before the scripted update failure; leg b resumed and
    trained exactly those saved groups as its first step, with no play before it."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    modal_rl1 = pytest.importorskip("modal_rl1")
    base, rl = "pf", "pf/arena/trials/rl-loop/rl"
    study = {"rl_loop": {"mg": {"redesign": {"streams": 24, "max_steps": 2}}}}
    files = {f"{base}/arena/bootstrap-results.json": json.dumps({"rl-loop": "complete"}),
             f"{base}/arena/serving/vllm-openai-server.log": 'x STUB {"errors": 0, "exact": 4}', f"{base}/bootstrap.log": "ok"}
    dirs = {f"{base}/state": ["s0001", "s0002"], rl: []}

    def judge(feed, ident):
        return modal_rl1.judge(base, {**files, f"{rl}/feed.jsonl": "\n".join(json.dumps(e) for e in feed)}.get,
                               lambda d: dirs.get(d, []), "", study, {"exit_code": 0, **ident})["checks"]
    a = [{"kind": "seq_start"}, {"kind": "step", "step": 1, "grad_norm": 0.1}, {"kind": "seq_group"},
         {"kind": "seq_pending", "step": 2, "groups": 4, "full": 4},
         {"kind": "error", "error": "step: preflight die_step 2: the update fails"}]
    c = judge(a, {"die_step": 2, "exit_code": 1, "error": "the scripted update failure"})
    assert c["pending_saved_before_die"] and c["feed_final_no_error"]
    assert c["exit_clean"] and c["trial_complete"] and c["steps_ok"]  # leg a's end is its scripted failure (real-v6 N8a)
    c = judge(a[:-1], {"die_step": 2, "exit_code": 1, "error": "something else"})
    assert not c["exit_clean"] and not c["trial_complete"]  # no scripted error row: a real failure
    assert not judge([e for e in a if e["kind"] != "seq_pending"], {"die_step": 2})["pending_saved_before_die"]
    assert not judge([{**e, "groups": 3} if e["kind"] == "seq_pending" else e for e in a],
                     {"die_step": 2})["pending_saved_before_die"]  # a group not saved
    b = [{"kind": "seq_start"}, {"kind": "seq_resume", "step": 1, "pending_groups": 4},
         {"kind": "seq_replay", "groups": 4, "resumed_pending": True}, {"kind": "step", "step": 2, "groups": 4, "grad_norm": 0.1},
         {"kind": "seq_group"}, {"kind": "seq_end"}]
    assert judge(b, {"expect_resume_pending": True})["resume_trains_saved_tries"]
    played = b[:2] + [{"kind": "seq_group"}] + b[2:]
    assert not judge(played, {"expect_resume_pending": True})["resume_trains_saved_tries"]
    assert not judge([e for e in b if e["kind"] != "seq_replay"], {"expect_resume_pending": True})["resume_trains_saved_tries"]


def test_host_close_trains_on_the_groups_full_so_far(tmp_path, harvest_ok, monkeypatch):
    """real-v7 27 Oct: redesign.host_close_gib (default 190); free host under it during play closes each phase early
    and the update trains on the groups already full (wired into the run loop)."""
    monkeypatch.setattr(rl_seq, "host_free_gib", lambda: 10.0)
    monkeypatch.setattr(rl_seq, "HOST_EVERY_S", 0.0)
    loop = SavingLoop(tmp_path, max_steps=2)
    loop.cfg["mg"]["redesign"]["host_close_gib"] = 100.0
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    closes, trains = loop.feed.of("seq_host_close"), loop.feed.of("seq_train")
    assert closes and [t["step"] for t in trains] == [0, 1]
    for c in closes:
        t = next(t for t in trains if t["step"] == c["step"])
        assert 1 <= c["groups"] <= t["groups"] <= 4  # full groups only grow between the close and the update


def test_pending_groups_are_prepared_before_the_trainer_loads(tmp_path, harvest_ok, monkeypatch):
    """F1 (real-v6 update 1: setup 28.7 s, cards idle, the 16 saved groups prepared at the call):
    with one_call + early_ahead, a resume's saved groups are read from the roots and their prepare starts before the
    trainer loads; the ahead file carries them all before the update, and early() prepares none again."""
    class Dies(FakeTrainer):
        async def call(self, op, **kw):
            if op == "step":
                self.log.append(op)
                return {"ok": False, "error": "stopped before update 2"}
            return await super().call(op, **kw)

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v6"))
    asyncio.run(asyncio.wait_for(rl_seq.run(SavingLoop(tmp_path / "a", max_steps=1),
                                            gate=phase.PhaseGate(inner=Upstream())), 30))
    (tmp_path / "v6" / "seq_play.json").unlink()  # the pending path alone (play_restore, carried: their own test)
    st = json.loads((tmp_path / "v6" / "seq_state.json").read_text())
    (tmp_path / "v6" / "seq_state.json").write_text(json.dumps({**st, "mirrored": {}}))
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "v6"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v7"))
    v7 = SavingLoop(tmp_path / "b", max_steps=3, trainer=Dies)
    with pytest.raises(Exception, match="stopped before update 2"):
        asyncio.run(asyncio.wait_for(rl_seq.run(v7, gate=phase.PhaseGate(inner=Upstream())), 30))
    (tmp_path / "v7" / "seq_state.json").unlink(missing_ok=True)
    rl = tmp_path / "v7" / "arena/trials/rl-loop/rl"
    done = [r for r in v7.feed.of("group_done") if r.get("tries") and (rl / f"{r['gid']}-{r['set']}").is_dir()]
    pend = {"step": 2, "version": 1, "groups": [
        {"gid": r["gid"], "game": r["set"].rpartition("-L")[0], "level": int(r["set"].rpartition("-L")[2]),
         "version": int(r["version"]), "rel": f"arena/trials/rl-loop/rl/{r['gid']}-{r['set']}"} for r in done]}
    (tmp_path / "v7" / "seq_pending.json").write_text(json.dumps(pend))

    monkeypatch.setenv("ARC_RL1_RESUME", f"{tmp_path / 'v6'},{tmp_path / 'v7'}")
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v8"))
    v8 = SavingLoop(tmp_path / "c", max_steps=3)
    v8.cfg["mg"]["redesign"]["one_call"] = True
    v8.cfg["mg"]["trainer_fast"] = {"early_ahead": True}
    seen, prepared, real_prepare = {}, [], rl_seq.prepare_group

    def spy(lp, g, rows):
        prepared.append(str(g.done["folder"]))
        assert lp.cfg["mg_sets"][g.unit]["name"] == g.unit  # Loop.v7_specs (level_norm) reads the set's cap
        return real_prepare(lp, g, rows)
    monkeypatch.setattr(rl_seq, "prepare_group", spy)
    start = v8.start_trainer

    async def start_trainer():
        seq_tasks = [t for t in asyncio.all_tasks() if "all_then_write" in repr(t.get_coro())]
        seen["prepare_running"] = bool(seq_tasks)
        await start()
    v8.start_trainer = start_trainer
    asyncio.run(asyncio.wait_for(rl_seq.run(v8, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert seen["prepare_running"]  # the saved groups' prepare started before the trainer loaded
    row = v8.feed.of("seq_ahead_pending")[0]
    assert row["groups"] == row["prepared"] == len(pend["groups"])
    early = v8.feed.of("seq_early_ahead")[0]
    done = [g["gid"] for g in pend["groups"]]
    assert early["step"] == 2 and early["groups"] == early["during_play"] == len(done)  # none prepared at the call
    saved = [str(tmp_path / "v7" / g["rel"]) for g in pend["groups"]]
    assert all(prepared.count(f) == 1 for f in saved)  # each saved group prepared once (by folder: gids recur, the
    # resumed backlog reissues them; the trainer keys windows by the whole spec, rollout path included)
    assert v8.feed.of("step")[0]["step"] == 2 and v8.feed.of("step")[0]["groups"] == len(done)


def test_served_route_is_on_from_the_trainer_start_and_mirrored(tmp_path, monkeypatch):
    """R3: mg.served_route sets the capture
    directory on the loop (window specs carry it), the trainer's router is confirmed "served", a resume's mirrored
    route records are restored before the trainer starts, and finished records are mirrored to <run>/served-route."""
    from arena import rl_loop, served_route

    route, old = tmp_path / "route", tmp_path / "v6" / "served-route"
    route.mkdir()
    (tmp_path / "srcroute").mkdir()
    (tmp_path / "srcroute" / "r1-0.rt").write_bytes(b"x" * 10)
    served_route.mirror(tmp_path / "srcroute", old, {}, settle=0.0)
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "v6"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v7"))
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.cfg["mg"]["served_route"] = True
    loop.study = {"server": {"served_route": str(route)}}
    routed = {}
    start = loop.start_trainer

    async def start_trainer():
        routed["restored"] = (route / "r1-0.rt").is_file()
        await start()
    loop.start_trainer = start_trainer
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert loop.served_dir == str(route) and routed["restored"]
    assert loop.feed.of("served_restore")[0]["files"] == 1
    assert loop.feed.of("served_route")[0]["dir"] == str(route) and "route" in loop.log
    assert rl_loop.Loop.windows(loop, tmp_path, [{"label": "a1"}])[0]["served_route"] == str(route)
    (route / "r2-0.rt").write_bytes(b"y")
    seen = {}
    served_route.mirror(route, tmp_path / "v7" / "served-route", seen, settle=0.0)
    assert {"r1-0.rt", "r2-0.rt"} <= set(seen)
    cfg = {"trainer_script": "rl1/rl1_trainer.py", "trainer_python": "python", "model_dir": "m", "rank": 8, "lr": 1e-5,
           "window": 1024, "optimizer": "adamw", "loop_lora": False, "tis_cap": None, "path_chunk": 16384,
           "mg": {"served_route": True}}
    assert rl_loop.trainer_command(cfg, tmp_path, tmp_path)[1]["RL1_ROUTER"] == "served"
    cfg["mg"]["redesign"] = {"play_only": True}
    assert "RL1_ROUTER" not in rl_loop.trainer_command(cfg, tmp_path, tmp_path)[1]


def test_served_route_off_leaves_the_loop_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v7"))
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.study = {"server": {"served_route": str(tmp_path / "route")}}
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert not getattr(loop, "served_dir", None) and not loop.feed.of("served_route") and "route" not in loop.log


class RecyclingServer(FakeServer):
    async def recycle(self):
        self.log.append("recycle")
        return {"pid": 2}


def test_recycle_replaces_the_servers_at_the_switch_back_and_resume_follows(tmp_path, harvest_ok, monkeypatch):
    """mg.redesign.recycle: every ``every`` switches the servers are recycled (not woken) after
    the park, the adapter is loaded on the fresh servers, and a resume of that run trains on."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "a"))
    loop = SavingLoop(tmp_path / "x", max_steps=3)
    loop.server = RecyclingServer(loop.log)
    loop.cfg["mg"]["redesign"]["recycle"] = {"every": 2}
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    rows = loop.feed.of("seq_recycle")
    assert [r["step"] for r in rows] == [2] and rows[0]["why"] == "every"
    i = loop.log.index("recycle")
    assert "wake" not in loop.log[i - 2:i + 1] and loop.log[i + 1].startswith("load:policy")
    phases = loop.feed.of("seq_phase")
    assert [bool(p["start"].get("recycled")) for p in phases] == [False, True, False][:len(phases)]
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "a"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "b"))
    again = SavingLoop(tmp_path / "y", max_steps=4)
    again.server = RecyclingServer(again.log)
    again.cfg["mg"]["redesign"]["recycle"] = {"every": 2}
    asyncio.run(asyncio.wait_for(rl_seq.run(again, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert again.feed.of("seq_resume")[0]["step"] == 3 and again.feed.of("step")[-1]["step"] == 4


def test_recycle_setting_is_checked():
    base = {"history": "h", "streams": 4, "max_steps": 1}
    with pytest.raises(ValueError, match="recycle"):
        rl_seq.settings({"redesign": {**base, "recycle": {"every_n": 2}}})
    assert rl_seq.settings({"redesign": {**base, "recycle": {"host_gib": 300}}})["recycle"] == {"host_gib": 300}


def test_served_prune_drops_trained_tries_records_only(tmp_path):
    """R3 housekeeping (gate-v1 28 Oct: the capture dir grew 9.3 GB in one phase on host memory): after a step, the
    trained tries' route records go; other tries' records and files younger than the mirror's settle stay."""
    import os
    import time as _time

    route = tmp_path / "route"
    route.mkdir()
    folder = tmp_path / "g1"
    (folder / "attempts" / "a1").mkdir(parents=True)
    (folder / "attempts" / "a1" / "rollout.jsonl").write_text(
        json.dumps({"rid": "r1"}) + "\n" + json.dumps({"rid": "r2"}) + "\n")
    old = _time.time() - 600
    for name in ("r1-0.rt", "r1-0.mm.json", "r2-5.rt", "r9-0.rt"):
        (route / name).write_bytes(b"x" * 4)
        os.utime(route / name, (old, old))
    (route / "r2-9.rt").write_bytes(b"y")  # young: not mirrored yet
    loop = FakeLoop(tmp_path)
    loop.served_dir = str(route)
    seq = rl_seq.Seq(loop, rl_seq.settings(loop.cfg["mg"]))
    asyncio.run(seq.served_prune([(rl_seq.Group("p1g1", "sb26-L1", 0, folder), [{"label": "a1"}])]))
    row = loop.feed.of("seq_served_prune")[0]
    assert row["files"] == 3 and row["young"] == 1 and row["rids"] == 2
    assert sorted(p.name for p in route.iterdir()) == ["r2-9.rt", "r9-0.rt"]


def _try_folder(root, rel, rids):
    folder = root / rel
    (folder / "attempts" / "a1").mkdir(parents=True)
    (folder / "attempts" / "a1" / "rollout.jsonl").write_text("".join(json.dumps({"rid": r}) + "\n" for r in rids))
    return folder


def test_restore_takes_only_the_saved_groups_routes(tmp_path):
    """real-v13 29 Oct: the trainer waited 890 s while the resume copied v12's whole 205 GB mirror (65K files) for 0
    saved groups. Now: the state's carried groups and the next update's pending / mid-play groups, by request id."""
    from arena import served_route

    run = tmp_path / "v12"
    _try_folder(run, "g/carried", ["chatcmpl-c1"])
    _try_folder(run, "g/pending", ["chatcmpl-p1"])
    _try_folder(run, "g/play", ["chatcmpl-m1"])
    _try_folder(run, "g/stale", ["chatcmpl-s1"])
    (run / "seq_state.json").write_text(json.dumps({"step": 10, "root": "/runs/validation/v12",
                                                    "mirrored": {"c": {"abs": "/runs/validation/v12/g/carried"}}}))
    (run / "seq_pending.json").write_text(json.dumps({"step": 11, "groups": [{"gid": "p", "rel": "g/pending"}]}))
    (run / "seq_play.json").write_text(json.dumps({"step": 11, "root": "/runs/validation/v12",
                                                   "groups": {"m": {"rel": "g/play"}}}))
    assert rl_seq.saved_rids([run]) == {"chatcmpl-c1", "chatcmpl-p1", "chatcmpl-m1"}
    (run / "seq_pending.json").write_text(json.dumps({"step": 10, "groups": [{"gid": "s", "rel": "g/stale"}]}))
    assert rl_seq.saved_rids([run]) == {"chatcmpl-c1", "chatcmpl-m1"}  # update 10 trained: its routes stay behind
    assert rl_seq.saved_rids([tmp_path / "nothing"]) is None  # no rl_seq state: restore all (as before)

    src = tmp_path / "capture"
    src.mkdir()
    for name in ("chatcmpl-c1-aa.rt", "chatcmpl-c1-aa.mm.json", "chatcmpl-s1-bb.rt", "chatcmpl-x-cc.rt", "layers.json"):
        (src / name).write_bytes(b"x")
    served_route.mirror(src, run / "served-route", {}, settle=0.0)
    dst = tmp_path / "live"
    got = served_route.restore([run / "served-route"], dst, {"chatcmpl-c1"})
    assert sorted(p.name for p in dst.iterdir()) == ["chatcmpl-c1-aa.mm.json", "chatcmpl-c1-aa.rt", "layers.json"]
    assert got["files"] == 3 and got["skipped"] == 2  # layers.json counts
    assert served_route.rid_of("chatcmpl-80636e01-89cbbaa6.mm.json") == "chatcmpl-80636e01"


def test_mirror_prune_keeps_passes_of_untrained_groups(tmp_path):
    """The volume mirror (real-v12: ~19 GB a step, never cut) loses the passes older than the oldest untrained group's
    launch; nothing while an untrained group came from an earlier container."""
    import os
    import time as _time

    mirror = tmp_path / "v13" / "served-route"
    mirror.mkdir(parents=True)
    now = _time.time()
    for name, age in (("pass-1.tar", 900), ("pass-2.tar", 300), ("pass-3.tar", 10)):
        (mirror / name).write_bytes(b"x" * 5)
        os.utime(mirror / name, (now - age, now - age))
    (mirror / "layers.json").write_text("{}")
    os.utime(mirror / "layers.json", (now - 900, now - 900))
    loop = FakeLoop(tmp_path)
    loop.served_dir = str(tmp_path / "route")
    (tmp_path / "route").mkdir()
    loop.served_mirror_dir = mirror
    seq = rl_seq.Seq(loop, rl_seq.settings(loop.cfg["mg"]))
    seq.backlog.groups = {"g9": {}}
    seq.backlog.full = [{"group": {"id": "g8"}}]  # restored from the earlier leg: launch time unknown
    seq.launched_t = {"g9": now - 600}
    asyncio.run(seq.served_prune([]))
    assert loop.feed.of("seq_served_prune")[0]["mirror"]["tars"] == 0
    seq.backlog.full = []  # g8 trained
    asyncio.run(seq.served_prune([]))
    assert loop.feed.of("seq_served_prune")[1]["mirror"] == {"tars": 1, "bytes": 5}
    assert sorted(p.name for p in mirror.iterdir()) == ["layers.json", "pass-2.tar", "pass-3.tar"]
    seq.backlog.groups = {}
    asyncio.run(seq.served_prune([]))
    assert sorted(p.name for p in mirror.iterdir()) == ["layers.json"]


def test_whole_segment_is_a_redesign_setting_over_fast():
    """tries longer than the window split into windows again."""
    from arena import rl_loop
    assert rl_loop.fast({})["whole_segment"] is True
    assert rl_loop.fast({"redesign": {"whole_segment": False}})["whole_segment"] is False
    assert rl_loop.fast({"redesign": {"whole_segment": False}})["fit_tokens"] == rl_loop.FAST["fit_tokens"]


def test_guard_auto_resumes_a_step0_preemption_with_saved_pending(tmp_path):
    """real-v11 (28 Oct): a preemption before the first checkpoint auto-resumes when step 0's groups are saved
    (seq_pending.json of step 1), and stays blocked without them."""
    import json as _json

    from arc_infra.runops import restart as R

    def die_preempted(run):
        g = R.RestartGuard(tmp_path, run, policy="resume", call_id="c1")
        g.start()
        st = _json.loads(g.path.read_text())
        st["attempts"][-1]["signal"] = {"name": "SIGINT", "t": 1.0}  # what watch()'s handler records
        g.path.write_text(_json.dumps(st))

    die_preempted("a")
    assert R.RestartGuard(tmp_path, "a", policy="resume", call_id="c1").start(0, pending=True).restart
    die_preempted("b")
    with pytest.raises(R.RestartBlocked):
        R.RestartGuard(tmp_path, "b", policy="resume", call_id="c1").start(0)

    sys_path = Path(__file__).resolve().parents[1] / "scripts/posttrain"
    import sys

    sys.path.insert(0, str(sys_path))
    modal_rl1 = pytest.importorskip("modal_rl1")
    (tmp_path / "seq_pending.json").write_text(_json.dumps({"step": 1, "groups": [{"gid": "g1"}]}))
    assert modal_rl1.pending_step1(tmp_path)
    (tmp_path / "seq_pending.json").unlink()
    (tmp_path / "seq_play.json").write_text(_json.dumps({"step": 1, "groups": {"g1": {"rel": "x"}}}))
    assert modal_rl1.pending_step1(tmp_path)  # mid-play saved groups count too
    (tmp_path / "seq_state.json").write_text("{}")
    assert not modal_rl1.pending_step1(tmp_path)


class StallingLoop(SavingLoop):
    """Groups after the first ``free`` stall until cancelled: the phase never fills (a container killed mid-play)."""

    def __init__(self, tmp, free=2, **kw):
        super().__init__(tmp, **kw)
        self.free = free

    async def mg_play(self, name, spec, starts, tries, **kw):
        if int(name.split("-")[0].split("g")[1]) > self.free:
            await asyncio.Event().wait()
        return await super().mg_play(name, spec, starts, tries, **kw)


async def _kill_when_played(loop, root, groups, timeout=20.0):
    task = asyncio.ensure_future(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())))
    began = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - began < timeout:
        await asyncio.sleep(0.01)
        try:
            if len(json.loads((root / "seq_play.json").read_text())["groups"]) >= groups:
                break
        except (OSError, ValueError):
            continue
    task.cancel()  # the container dies: no end-of-run collection, no state write
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("at_step", [0, 1])
def test_a_mid_play_kill_resumes_its_played_groups_without_replay(tmp_path, harvest_ok, monkeypatch, at_step):
    """Checklist item 6 (real-v11 attempt 2 lost ~28 min of play to a preemption before any phase
    closed): seq_play.json holds the backlog and the phase's full groups' saved folders as each lands; a restart
    (root renamed -lost-, as modal_rl1 does) trains them with the phase's new groups, none played again."""
    vo = tmp_path / "vo"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(vo))
    if at_step:  # step 1 checkpointed, then the next phase is killed mid-play
        asyncio.run(asyncio.wait_for(rl_seq.run(SavingLoop(tmp_path / "a0", max_steps=1),
                                                gate=phase.PhaseGate(inner=Upstream())), 30))
        (vo / "seq_play.json").unlink()  # the update's own write; the killed leg writes its own
        st = json.loads((vo / "seq_state.json").read_text())  # no carried groups: the killed leg's phase plays
        (vo / "seq_state.json").write_text(json.dumps({**st, "mirrored": {}}))
        monkeypatch.setenv("ARC_RL1_RESUME", str(vo))
        monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo1"))
        vo = tmp_path / "vo1"
    killed = StallingLoop(tmp_path / "a", free=2 + 100 * at_step, max_steps=2)
    if at_step:  # the resumed phase's groups (ids go on from the saved counter): all but the first two stall
        began = []

        async def mg_play(name, spec, starts, tries, _base=SavingLoop.mg_play, **kw):
            began.append(name)
            if len(began) > 2:
                await asyncio.Event().wait()
            return await _base(killed, name, spec, starts, tries, **kw)
        killed.mg_play = mg_play
    asyncio.run(_kill_when_played(killed, vo, 2))
    play = json.loads((vo / "seq_play.json").read_text())
    assert play["step"] == at_step + 1 and len(play["groups"]) >= 2 and not (vo / "seq_pending.json").exists()
    saved = sorted(play["groups"])
    # a harvested start may already sit in an in-flight group (unknown levels rank first, so pump takes it at once;
    # CI 29 Oct under load): saved with its group, returned to the starts by abandon_inflight on restore
    pool = play["backlog"]["starts"] + [g["group"]["start"] for g in play["backlog"]["groups"]]
    harvested = [s for s in pool if not s["l1"] and s["state"]["attempt"].startswith(f"{vo}/")]
    assert harvested  # cleared tries saved deeper starts under the dead leg's root
    lost = vo.with_name(vo.name + "-lost-1")
    vo.rename(lost)  # a preemption restart: the dead attempt's root renamed, its paths remapped on load

    roots = ([str(tmp_path / "vo")] if at_step else []) + [str(lost)]
    monkeypatch.setenv("ARC_RL1_RESUME", ",".join(roots))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(vo))
    again = SavingLoop(tmp_path / "b", max_steps=at_step + 1)
    ids, restored, real = {h["id"] for h in harvested}, [], rl_seq.Seq.play_restore

    async def spy(self, roots):  # the dead leg's saved starts as loaded: under its -lost- root
        n = await real(self, roots)
        restored.extend(s for s in self.backlog.starts.values() if s["id"] in ids)
        return n
    monkeypatch.setattr(rl_seq.Seq, "play_restore", spy)
    asyncio.run(asyncio.wait_for(rl_seq.run(again, gate=phase.PhaseGate(inner=Upstream())), 30))
    assert len(restored) == len(ids) and all(s["state"]["attempt"].startswith(f"{lost}/") for s in restored)
    res = again.feed.of("seq_play_resume")[0]
    assert res["gids"] == [g for g in saved if g in res["gids"]] and sorted(res["gids"]) == saved and not res["lost"]
    assert res["root"] == str(lost) and res["dropped_groups"]  # the stalled groups died with the container
    launched = {r["gid"] for r in again.feed.of("seq_group")}
    assert not launched & set(saved)  # none played again
    step = again.feed.of("step")[0]
    assert step["step"] == at_step + 1 and step["groups"] == 4
    assert set(saved) <= set(again.accumulated)  # trained, from the -lost- folders
    assert not again.feed.of("seq_replay")
    assert rl_seq.remap_root({"a": f"{vo}/harvest/x", "b": f"{vo}-other/y"}, str(vo), str(lost)) == {
        "a": f"{lost}/harvest/x", "b": f"{vo}-other/y"}


def test_written_root_of_states_without_a_root():
    """real-v11's states (no "root" field) read from a -lost- rename or a per-run container's /in/<volume>/ mount."""
    assert rl_seq.written_root("/runs/validation/r-v11-lost-1793197210") == "/runs/validation/r-v11"
    assert rl_seq.written_root("/in/arc-prolong-runs/validation/r-v11-attempt3") == "/runs/validation/r-v11-attempt3"
    assert rl_seq.written_root("/runs/validation/r-v11") == "/runs/validation/r-v11"
    state = {"backlog": {"starts": [{"state": {"attempt": "/runs/validation/r-v11-attempt3/harvest/s0001-x"}}]}}
    got = rl_seq.remap_root(state, rl_seq.written_root("/in/arc-prolong-runs/validation/r-v11-attempt3"),
                            "/in/arc-prolong-runs/validation/r-v11-attempt3")
    assert got["backlog"]["starts"][0]["state"]["attempt"] == "/in/arc-prolong-runs/validation/r-v11-attempt3/harvest/s0001-x"


def test_served_mirror_skips_a_file_pruned_mid_pass(tmp_path, monkeypatch):
    """real-v12 (28 Oct): the post-step prune removed a .rt between the scan and tar.add; the whole pass failed
    (served_mirror_failed). Now that file is skipped and the rest are mirrored."""
    import os
    import tarfile
    import time

    from arena import served_route

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    old = time.time() - 120
    for n in ("a.rt", "b.rt"):
        (src / n).write_bytes(b"x" * 10)
        os.utime(src / n, (old, old))
    real = tarfile.TarFile.add

    def add(self, name, arcname=None, **kw):
        if arcname == "a.rt":
            os.unlink(name)
        return real(self, name, arcname=arcname, **kw)
    monkeypatch.setattr(tarfile.TarFile, "add", add)
    seen = {}
    got = served_route.mirror(src, dst, seen)
    assert got["files"] == 1 and list(seen) == ["b.rt"]
    tars = list(dst.glob("pass-*.tar"))
    assert len(tars) == 1 and not list(dst.glob("*.part"))
    with tarfile.open(tars[0]) as t:
        assert t.getnames() == ["b.rt"]


def test_budget_floor_is_a_live_switch(tmp_path):
    from types import SimpleNamespace
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"seq": {"budget_floor": True}}))
    assert rl_seq.live_knobs(str(knobs)) == {"budget_floor": True}
    knobs.write_text(json.dumps({"seq": {"budget_floor": 0}}))
    assert rl_seq.live_knobs(str(knobs)) == {"budget_floor": False}
    knobs.write_text(json.dumps({"seq": {"budget_floor": "yes"}}))
    assert rl_seq.live_knobs(str(knobs)) == {}
    loop = SimpleNamespace(study={"server": {"max_num_seqs": 152}})
    cur = SimpleNamespace(cfg={"budget_floor": False})
    seq = SimpleNamespace(backlog=SimpleNamespace(streams=240, cfg={"phase_groups": 16}, cur=cur))
    assert rl_seq.apply_knobs(loop, seq, {"budget_floor": True}) == {"budget_floor": [False, True]}
    assert cur.cfg["budget_floor"] is True and seq.backlog.cfg["budget_floor"] is True
    assert rl_seq.apply_knobs(loop, seq, {"budget_floor": True}) == {}
    knobs.write_text(json.dumps({"seq": {"budget_guard": True, "budget_floor": False}}))  # the guard
    assert rl_seq.live_knobs(str(knobs)) == {"budget_floor": False, "budget_guard": True}
    knobs.write_text(json.dumps({"seq": {"budget_censored": 1}}))
    assert rl_seq.live_knobs(str(knobs)) == {"budget_censored": True}
    assert rl_seq.apply_knobs(loop, seq, {"budget_guard": True}) == {"budget_guard": [False, True]}
    assert cur.cfg["budget_guard"] is True and seq.backlog.cfg["budget_guard"] is True


def test_served_prune_drops_warmups_once_their_groups_trained(tmp_path):
    """mg.leader_first warm-ups (chatcmpl-warm-*) are R3 donors of the carried groups they warmed: kept while a group
    launched before them is untrained, then deleted; a resume's restore takes them all."""
    import os
    import time as _time

    from arena import served_route

    route = tmp_path / "route"
    route.mkdir()
    now = _time.time()
    for name, age in (("chatcmpl-warm-a-0.rt", 900), ("chatcmpl-warm-b-0.rt", 300), ("chatcmpl-warm-c-0.rt", 10)):
        (route / name).write_bytes(b"x" * 4)
        os.utime(route / name, (now - age, now - age))
    loop = FakeLoop(tmp_path)
    loop.served_dir = str(route)
    seq = rl_seq.Seq(loop, rl_seq.settings(loop.cfg["mg"]))
    seq.backlog.groups = {"g9": {}}
    seq.launched_t = {"g9": now - 1000}  # launched before every warm-up: all kept
    asyncio.run(seq.served_prune([]))
    assert len(list(route.iterdir())) == 3
    seq.launched_t = {"g9": now - 600}  # the oldest untrained group came after warm-up a
    asyncio.run(seq.served_prune([]))
    assert loop.feed.of("seq_served_prune")[1]["warm"] == 1
    assert sorted(p.name for p in route.iterdir()) == ["chatcmpl-warm-b-0.rt", "chatcmpl-warm-c-0.rt"]
    seq.backlog.groups = {}
    asyncio.run(seq.served_prune([]))  # nothing untrained: all but the unsettled one
    assert sorted(p.name for p in route.iterdir()) == ["chatcmpl-warm-c-0.rt"]
    served_route.mirror(route, tmp_path / "mirror", {}, settle=0.0)
    got = served_route.restore([tmp_path / "mirror"], tmp_path / "live", {"chatcmpl-other"})
    assert got["files"] == 1 and (tmp_path / "live" / "chatcmpl-warm-c-0.rt").exists()


def test_stub_names_a_reply_by_its_request_id():
    """As vLLM (chatcmpl-<X-Request-Id or request_id>): leader_first warm-ups keep their warm- id in a preflight."""
    from arena import replay_stub

    state = replay_stub.State(corpus=None, speed=0)
    code, body = replay_stub.chat_reply(state, {"messages": [{"role": "user", "content": "x"}], "request_id": "warm-1"})
    assert code == 200 and body["id"] == "chatcmpl-warm-1"
    code, body = replay_stub.chat_reply(state, {"messages": [{"role": "user", "content": "x"}], "_request_id": "warm-2",
                                                "request_id": "warm-1"})
    assert body["id"] == "chatcmpl-warm-2"
    assert replay_stub.chat_reply(state, {"messages": [{"role": "user", "content": "x"}]})[1]["id"].startswith(
        "chatcmpl-stub-")


def test_kv_cap_is_a_live_knob_with_hysteresis(tmp_path):
    """cap the engine KV live (v15 at 0.77-0.79 evicted between-turn contexts). The guard's every poll
    reads seq.kv_cap; at kv >= cap the server's gates hold, below cap - 0.02 they admit again; a phase end releases."""
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"seq": {"kv_cap": 0.73, "streams": 172}}))
    assert rl_seq.live_knobs(str(knobs)) == {"kv_cap": 0.73, "streams": 172}
    knobs.write_text(json.dumps({"seq": {"kv_cap": 1.5}}))
    assert rl_seq.live_knobs(str(knobs)) == {}
    want = [rl_seq.kv_want(0.73, kv, was) for kv, was in ((0.70, False), (0.73, False), (0.72, True), (0.70, True))]
    assert want == [False, True, True, False]
    assert not rl_seq.kv_want(None, 0.9, True) and not rl_seq.kv_want(0.73, None, True)
    seen = []

    async def get(url):
        return "vllm:kv_cache_usage_perc 0.75\nvllm:num_preemptions_total 0\n"

    async def go():
        stop = asyncio.Event()
        task = asyncio.ensure_future(rl_seq.server_guard(["http://a/v1"], stop, every=30, get=get,
                                                         on_kv=lambda u, kv: seen.append((u, kv))))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, 2)
    asyncio.run(go())
    assert seen == [("http://a/v1", 0.75), ("http://a/v1", 0.75), ("http://a/v1", None)]


def test_minibatches_step_each_chunk_of_whole_groups(tmp_path):
    """mg.redesign.minibatches 4 = 4 disjoint chunks of whole groups, one optimizer step each
    (the last one saves the adapter), 1 pass; the step row carries each step's safety fields."""
    assert rl_seq.split_chunks(range(16), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    assert [len(c) for c in rl_seq.split_chunks(range(15), 4)] == [4, 4, 4, 3]
    assert rl_seq.split_chunks(range(2), 4) == [[0], [1]] and rl_seq.split_chunks([], 4) == []
    with pytest.raises(ValueError, match="minibatches"):
        rl_seq.settings({"redesign": {"streams": 4, "max_steps": 1, "minibatches": 4}})
    assert rl_seq.settings({"redesign": {"streams": 4, "max_steps": 1, "minibatches": 4, "one_call": True}})
    loop = FakeLoop(tmp_path)
    loop.cfg["mg"]["redesign"] = {**(loop.cfg["mg"].get("redesign") or {}), "one_call": True, "minibatches": 2}
    calls = []

    class T(FakeTrainer):
        async def call(self, op, **kw):
            calls.append((op, kw))
            if op == "step" and "save" not in kw:
                self.log.append(op)
                return {"ok": True, "grad_norm": 0.01, "update_norm": 0.1, "k3": {"approx_kl": 1e-4, "clip_frac": 0.0},
                        "trained_tokens": 7}
            out = await super().call(op, **kw)
            return {**out, "grad_norm": 0.02, "update_norm": 0.2, "k3": {"approx_kl": 2e-4, "clip_frac": 0.01},
                    "trained_tokens": 5} if op == "step" else out
    loop.trainer = T(loop.log)
    specs = {"g1": [{"name": "a"}], "g2": [{"name": "b"}, {"name": "c"}], "g3": [], "g4": [{"name": "d"}]}
    loop.v7_prepare = lambda g: ({"gid": g.gid, "set": "x", "tries": 1, "trained": len(specs[g.gid])}, specs[g.gid])

    def grp(gid):
        return type("G", (), {"gid": gid, "unit": "x"})()

    out = asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s1"))
    ops = [(op, [a["name"] for a in kw.get("attempts") or []], "save" in kw) for op, kw in calls]
    # chunk 2's θ0 log-probabilities by a forward-only pass before the first step; its step clipped
    assert ops == [("accumulate", ["a", "b", "c"], False), ("old_pass", ["d"], False), ("step", [], False),
                   ("accumulate", ["d"], False), ("step", [], True)]
    assert calls[0][1]["ahead"] == [{"name": "d"}] and calls[0][1]["first"] and calls[3][1]["first"]
    assert calls[0][1]["old_reset"] and "ratio_clip" not in calls[0][1] and calls[3][1]["ratio_clip"] == "gspo"
    assert calls[1][1]["ahead"] == [{"name": "d"}] and out["ratio_clip"] == "gspo" and out["epochs"] == 1
    assert [m["groups"] for m in out["minibatches"]] == [2, 1] and out["update_norm_sum"] == 0.3
    assert out["trained_tokens"] == 12  # the update's, not its last step's (v17 U24 read 1.01M of 1.82M)
    assert out["minibatches"][1]["approx_kl"] == 2e-4 and out["grad_norm_max"] == 0.02 and out["groups"] == 4
    assert [r["minibatch"] for r in loop.feed.of("seq_accumulate") if r.get("minibatch")] == [1, 2]
    # the knob fallback: minibatches 1 from a live knob runs one accumulate + one saved step
    calls.clear()
    out = asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s2",
                                        minibatches=1))
    assert [op for op, _ in calls] == ["accumulate", "step"] and "minibatches" not in out
    assert "ratio_clip" not in calls[0][1] and "keep_old" not in calls[0][1]  # one step: no θ0 work at all
    # ratio_clip "off" by knob: a2's unclipped steps, no old pass
    calls.clear()
    out = asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s3",
                                        ratio_clip="off"))
    assert [op for op, _ in calls] == ["accumulate", "step", "accumulate", "step"]
    assert not any("ratio_clip" in kw or "keep_old" in kw for _, kw in calls) and out["ratio_clip"] is None
    # design B: epochs 2 over one chunk: epoch 1 keeps θ0 as it trains, epoch 2 is clipped against it
    calls.clear()
    out = asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s4",
                                        minibatches=1, epochs=2, ratio_clip="ppo"))
    ops = [(op, [a["name"] for a in kw.get("attempts") or []], "save" in kw) for op, kw in calls]
    assert ops == [("accumulate", ["a", "b", "c", "d"], False), ("step", [], False),
                   ("accumulate", ["a", "b", "c", "d"], False), ("step", [], True)]
    assert calls[0][1]["keep_old"] and "ratio_clip" not in calls[0][1] and calls[2][1]["ratio_clip"] == "ppo"
    assert "keep_old" not in calls[2][1] and [m["epoch"] for m in out["minibatches"]] == [1, 2]
    # epochs 2 x minibatches 2: θ0 of chunk 2 by old_pass, then every later step clipped, none keeps θ0 again
    calls.clear()
    out = asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in specs], 1e-4, tmp_path / "s5",
                                        epochs=2))
    ops = [(op, [a["name"] for a in kw.get("attempts") or []]) for op, kw in calls]
    assert ops == [("accumulate", ["a", "b", "c"]), ("old_pass", ["d"]), ("step", []), ("accumulate", ["d"]),
                   ("step", []), ("accumulate", ["a", "b", "c"]), ("step", []), ("accumulate", ["d"]), ("step", [])]
    accs = [kw for op, kw in calls if op == "accumulate"]
    assert [bool(kw.get("ratio_clip")) for kw in accs] == [False, True, True, True]
    assert [bool(kw.get("keep_old")) for kw in accs] == [True, False, False, False]
    assert [(m["epoch"], m["groups"]) for m in out["minibatches"]] == [(1, 2), (1, 1), (2, 2), (2, 1)]


def test_minibatches_through_the_fake_trainer_tally_theta0(tmp_path):
    """The preflight stand-in (arena.fake_trainer) takes old_pass and reports the clipped call's θ0 tally as the
    trainer does: every window of chunk 2 attached."""
    from arena import fake_trainer

    f = fake_trainer.Fake.__new__(fake_trainer.Fake)
    f.served = {"windows": 0, "no_record": 0}
    f.served_count = lambda spec: None
    fake_trainer.reply_tokens, saved = (lambda rollout: (5, 1)), fake_trainer.reply_tokens
    try:
        got = f.run("old_pass", {"attempts": [{"name": "d", "rollout": "x"}], "old_reset": True})
        assert got["ok"] and got["old_windows"] == 1 and got["old_kept"] == 1
        assert f.old_names == {"d"}
    finally:
        fake_trainer.reply_tokens = saved


def test_live_knobs_lr_scale_and_minibatches(tmp_path):
    k = tmp_path / "knobs.json"
    k.write_text(json.dumps({"seq": {"lr_scale": 0.5, "minibatches": 1, "kv_cap": 0.73}}))
    assert rl_seq.live_knobs(str(k)) == {"lr_scale": 0.5, "minibatches": 1, "kv_cap": 0.73}
    k.write_text(json.dumps({"seq": {"lr_scale": 1.5, "minibatches": 0}}))
    assert rl_seq.live_knobs(str(k)) == {}
    k.write_text(json.dumps({"seq": {"lr_scale": True, "minibatches": 2.5}}))
    assert rl_seq.live_knobs(str(k)) == {}
    k.write_text(json.dumps({"seq": {"epochs": 2, "ratio_clip": "ppo"}}))
    assert rl_seq.live_knobs(str(k)) == {"epochs": 2, "ratio_clip": "ppo"}
    for bad in ({"epochs": 5}, {"epochs": 0}, {"epochs": True}, {"ratio_clip": "grpo"}, {"ratio_clip": 0.2}):
        k.write_text(json.dumps({"seq": bad}))
        assert rl_seq.live_knobs(str(k)) == {}
    k.write_text(json.dumps({"seq": {"ckpt_budget_gib": 235}}))
    assert rl_seq.live_knobs(str(k)) == {"ckpt_budget_gib": 235.0}
    for bad in (100, 270, True):  # out of range: the settings' budget stays
        k.write_text(json.dumps({"seq": {"ckpt_budget_gib": bad}}))
        assert rl_seq.live_knobs(str(k)) == {}


def test_ckpt_budget_knob_overrides_the_trainer_args():
    from arena import rl_loop

    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"cap": 4}, "path_chunk": 16384, "l2_norm": True, "clip": 1.0}
    assert loop.v7_trainer_args()["ckpt_budget_gib"] == 240
    loop.ckpt_budget_gib = 230.0
    assert loop.v7_trainer_args()["ckpt_budget_gib"] == 230.0


def test_control_twin_plays_base_untrained_every_n_updates(tmp_path, harvest_ok, monkeypatch):
    """every ``every``-th update the first group launched on a level with 2 finished
    groups gets an untrained twin on the base model, same start and budget; its tries reach only control_done (never
    the backlog, group_done, harvest or a step) and its route records go at the next prune."""
    import os
    import time as _time

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=4, clear_p=0.0)  # no clears: every level repeats from L1
    loop.cfg["mg"]["redesign"]["control"] = {"every": 2, "tries": 3}
    assert run_seq(loop) == "max_steps"
    ctl, done = loop.feed.of("seq_control"), loop.feed.of("control_done")
    assert ctl and len({r["step"] for r in ctl}) == len(ctl) and all(r["step"] % 2 == 0 for r in ctl)
    groups = {r["gid"]: r for r in loop.feed.of("seq_group")}
    for r in ctl:
        pair = groups[r["pair"]]
        assert r["gid"] == r["pair"] + "c" and r["model"] == "base" and r["tries"] == 3
        assert (r["set"], r["budget"], r["start"]) == (pair["set"], pair["budget"], pair["start"])
        assert r["gid"] not in groups
    assert {r["gid"] for r in done} <= {r["gid"] for r in ctl} and all(r["version"] == 0 for r in done)
    assert all(len(r["attempts"]) == r["tries"] == 3 for r in done)
    assert not any(g["gid"].endswith("c") for g in loop.feed.of("group_done"))
    assert sum(r["groups"] for r in loop.feed.of("step")) == 4 * 4  # phase_groups 4 per step, controls never trained
    base = [x for x in loop.played if x[1] == "base"]
    assert len(base) == len(ctl)
    # route records: a finished control's go at the next prune, a young one stays for the one after
    route = tmp_path / "route"
    route.mkdir()
    old = _time.time() - 600
    for name in ("c1-0.rt", "c2-0.rt"):
        (route / name).write_bytes(b"x")
    os.utime(route / "c1-0.rt", (old, old))
    seq = rl_seq.Seq(loop, rl_seq.settings(loop.cfg["mg"]))
    loop.served_dir = str(route)
    seq.control_rids = {"c1", "c2"}
    asyncio.run(seq.served_prune([]))
    assert sorted(p.name for p in route.iterdir()) == ["c2-0.rt"] and seq.control_rids == {"c2"}


def test_control_skipped_for_want_of_a_driver_slot_leaves_a_row(tmp_path, monkeypatch):
    """a control the drivers cannot place was skipped silently; now it leaves seq_control_skipped."""
    from types import SimpleNamespace

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=1)
    loop.cfg["mg"]["redesign"]["control"] = {"every": 2, "tries": 3}
    seq = rl_seq.Seq(loop, rl_seq.settings(loop.cfg["mg"]))
    seq.step, seq.drivers = 4, SimpleNamespace(assign=lambda *a: None)
    grp = {"id": "p5g9", "game": "sb26", "level": 3, "start": {"id": "s"}}
    seq.backlog.cur.fresh[("sb26", 3)] = [1, 2]
    seq.control(grp, {"name": "sb26-L3", "cap": 100}, [], "http://a/v1")
    rows = loop.feed.of("seq_control_skipped")
    assert len(rows) == 1 and rows[0]["pair"] == "p5g9" and rows[0]["step"] == 4 and not loop.feed.of("seq_control")
    assert not seq.controls


def test_update_mix_knobs_are_live_and_0_turns_one_off(tmp_path, monkeypatch):
    from types import SimpleNamespace
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"seq": {"game_cap": 4, "min_update_tokens": 3000000, "max_update_groups": True}}))
    got = rl_seq.live_knobs(str(knobs))
    assert got == {"game_cap": 4, "min_update_tokens": 3000000}
    seq = SimpleNamespace(backlog=SimpleNamespace(streams=240, cfg={"phase_groups": 16, "game_cap": None}))
    loop = SimpleNamespace(study={})
    assert rl_seq.apply_knobs(loop, seq, got, {}) == {"game_cap": [None, 4], "min_update_tokens": [None, 3000000]}
    assert rl_seq.apply_knobs(loop, seq, {"game_cap": 0}, {}) == {"game_cap": [4, None]}
    assert seq.backlog.cfg["game_cap"] is None and seq.backlog.cfg["min_update_tokens"] == 3000000
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"shrink": True}))
    assert rl_seq.live_knobs(str(knobs)) == {"game_cap": 4}  # a shrink's 2 tiny groups never reach a token floor


def test_loop_runs_with_the_update_mix_on(tmp_path, monkeypatch):
    """game_cap 1 + a token floor through the whole fake loop: every step trains at most one group per game, holds
    the floor (or the group ceiling), and the run reaches max_steps."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = FakeLoop(tmp_path, max_steps=3)
    loop.cfg["mg"]["redesign"]["backlog"].update(game_cap=1, min_update_tokens=1, max_update_groups=6)
    assert run_seq(loop) == "max_steps"
    sets = {r["gid"]: r["set"].split("-")[0] for r in loop.feed.of("group_done")}
    per_step, cur = [], []
    for r in loop.feed.rows:
        if r.get("kind") == "seq_accumulate" and r.get("gid"):
            cur.append(sets.get(r["gid"]))
        if r.get("kind") == "step":
            per_step.append(cur)
            cur = []
    assert len(per_step) == 3 and all(s and len(s) == len(set(s)) for s in per_step), per_step


def test_lvl_window_knob_only_lowers_the_launchs_value(tmp_path):
    from types import SimpleNamespace
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"seq": {"lvl_window": 34816}}))
    assert rl_seq.live_knobs(str(knobs)) == {"lvl_window": 34816}
    knobs.write_text(json.dumps({"seq": {"lvl_window": 1024}}))
    assert rl_seq.live_knobs(str(knobs)) == {}
    loop = SimpleNamespace(study={}, cfg={"lvl_window": 36864})
    seq = SimpleNamespace(backlog=SimpleNamespace(streams=240, cfg={"phase_groups": 16}))
    assert rl_seq.apply_knobs(loop, seq, {"lvl_window": 34816}, {}) == {"lvl_window": [36864, 34816]}
    assert rl_seq.apply_knobs(loop, seq, {"lvl_window": 34816}, {}) == {}
    assert rl_seq.apply_knobs(loop, seq, {"lvl_window": 40960}, {}) == {"lvl_window": [34816, 36864]}  # back, not past
    assert loop.cfg["lvl_window"] == 36864


def test_control_settings():
    base = {"streams": 4, "max_steps": 1}
    assert rl_seq.settings({"redesign": {**base, "control": {"every": 4, "tries": 8}}})["control"]["every"] == 4
    for bad in ({"every": 0}, {"tries": 1.5}, {"every": 4, "x": 1}, 4):
        with pytest.raises(ValueError):
            rl_seq.settings({"redesign": {**base, "control": bad}})


def test_restore_reads_indexed_tars_by_offset_and_scans_old_ones(tmp_path):
    """(v16: 223 s reading every tar's headers for 1 file): the mirror writes pass-*.idx.json beside each
    tar; restore opens a tar only for wanted members and reads them by offset (same bytes and mtime as the tar's);
    a tar without an index (mirrored before) is scanned as before; tar_index + write_index index an old one; the
    prune removes a tar's index with it; an index never restores as a loose file."""
    import os
    import tarfile

    from arena import served_route

    src, mirror = tmp_path / "capture", tmp_path / "run" / "served-route"
    src.mkdir()
    for name, data in (("chatcmpl-a-1.rt", b"A" * 1000), ("chatcmpl-a-1.mm.json", b"{}"), ("chatcmpl-b-2.rt", b"B" * 77),
                       ("chatcmpl-warm-w-3.rt", b"W")):
        (src / name).write_bytes(data)
        os.utime(src / name, (1000.0, 1000.0))
    served_route.mirror(src, mirror, {}, settle=0.0)
    tars = sorted(mirror.glob("pass-*.tar"))
    assert len(tars) == 1 and served_route.index_path(tars[0]).is_file()
    assert json.loads(served_route.index_path(tars[0]).read_text()) == served_route.tar_index(tars[0])
    opened = []
    real = tarfile.open
    monkey = lambda *a, **k: (opened.append(a[0]), real(*a, **k))[1]  # noqa: E731
    tarfile.open = monkey
    try:
        prog = {}
        got = served_route.restore([mirror], tmp_path / "live", {"chatcmpl-a"}, progress=prog)
        assert opened == []  # indexed: no tarfile scan
        assert got["indexed"] == 1 and got["scanned"] == 0 and got["files"] == 3  # a's two files + the warm-up's
        live = tmp_path / "live"
        assert (live / "chatcmpl-a-1.rt").read_bytes() == b"A" * 1000 and (live / "chatcmpl-a-1.mm.json").read_bytes() == b"{}"
        assert (live / "chatcmpl-a-1.rt").stat().st_mtime == 1000.0 and not (live / "chatcmpl-b-2.rt").exists()
        assert prog["tars_total"] == 1 and prog["tars_done"] == 1
        assert not list(live.glob("*.idx.json")) and not list(live.glob("*.part"))
        served_route.index_path(tars[0]).unlink()  # a tar mirrored before the index: scanned, same result
        got = served_route.restore([mirror], tmp_path / "live2", {"chatcmpl-a"})
        assert got["scanned"] == 1 and got["files"] == 3 and len(opened) == 1
        assert (tmp_path / "live2" / "chatcmpl-a-1.rt").read_bytes() == b"A" * 1000
    finally:
        tarfile.open = real
    served_route.write_index(tars[0], served_route.tar_index(tars[0]))
    assert served_route.restore([mirror], tmp_path / "live3", None)["files"] == 4  # all, no index file restored
    assert sorted(p.name for p in (tmp_path / "live3").iterdir()) == sorted(
        ["chatcmpl-a-1.rt", "chatcmpl-a-1.mm.json", "chatcmpl-b-2.rt", "chatcmpl-warm-w-3.rt"])
    os.utime(tars[0], (5.0, 5.0))
    assert served_route.prune_mirror(mirror, 10.0)["tars"] == 1 and not list(mirror.glob("pass-*"))


def test_index_mirror_indexes_only_unindexed_tars(tmp_path):
    from arena import served_route

    src, mirror = tmp_path / "c", tmp_path / "m"
    src.mkdir()
    (src / "chatcmpl-a-1.rt").write_bytes(b"x" * 600)
    served_route.mirror(src, mirror, {}, settle=0.0)
    tar = next(mirror.glob("pass-*.tar"))
    assert served_route.index_mirror(mirror) == {"tars": 1, "indexed": 0, "members": 0}
    want = json.loads(served_route.index_path(tar).read_text())
    served_route.index_path(tar).unlink()
    assert served_route.index_mirror(mirror) == {"tars": 1, "indexed": 1, "members": 1}
    assert json.loads(served_route.index_path(tar).read_text()) == want  # the write-time offsets match a header read


def test_update_takes_the_lr_scale_knob(tmp_path, harvest_ok, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    k = tmp_path / "knobs.json"
    k.write_text(json.dumps({"seq": {"lr_scale": 0.5}}))
    monkeypatch.setenv("ARENA_KNOBS", str(k))
    loop = FakeLoop(tmp_path, max_steps=2)
    assert run_seq(loop) == "max_steps"
    lrs = [r["lr"] for r in loop.feed.of("step")]
    base = [rl_seq.lr_at(float(loop.cfg["lr"]), int(rl_seq.DEFAULTS["warmup"]), s) for s in (0, 1)]
    assert lrs == pytest.approx([0.5 * b for b in base])
    assert [r["lr_scale"] for r in loop.feed.of("seq_train_knobs")] == [0.5, 0.5]


def test_restore_compare_masks_object_addresses():
    """v17 30 Oct: a tool result's '<function <lambda> at 0x2b39...>' differs on every replay; masked, the start is
    exact; any other difference still shows."""
    from arena import restore

    rec = [{"role": "tool", "content": "box_lock=<function <lambda> at 0x2b7a2f485f80>, mv = <itertools.cycle object at 0x2ae93edae>"}]
    live = [{"role": "tool", "content": "box_lock=<function <lambda> at 0x2b3910761f80>, mv = <itertools.cycle object at 0x2b3910a89d00>"}]
    row = {"keep": 3, "messages": rec, "template": {}, "tools_now": None}
    assert restore.compare(row, {}, 3, live) == []
    other = [{"role": "tool", "content": live[0]["content"].replace("box_lock", "box_lok")}]
    assert restore.compare(row, {}, 3, other)


def _route_rec(d, name, start, tokens, lora="p"):
    import numpy as np

    t = np.asarray(tokens, dtype=np.int32)
    head = np.array([start, len(t), 2, 1], dtype=np.int32).tobytes()
    (d / f"{name}.rt").write_bytes(head + t.tobytes() + np.zeros((len(t), 2, 1), dtype=np.int16).tobytes())
    (d / f"{name}.mm.json").write_text(json.dumps({"lora": lora}))


def test_served_store_follows_prune_and_new_requests(tmp_path):
    """v17 a3 U20: a cached store read a pruned donor (FileNotFoundError) and never saw later requests."""
    import os

    import numpy as np

    from arena import served_route as sr

    toks = np.arange(100, 200)
    _route_rec(tmp_path, "chatcmpl-old-abcdefgh", 0, toks[:60])
    store = sr.Store(tmp_path, scratch=True)
    own = {"meta": {"lora": "p"}, "lo": 0}
    assert store.head("chatcmpl-old") is not None  # cached head, as a donor ranking leaves it
    _route_rec(tmp_path, "chatcmpl-new-abcdefgh", 0, toks[:80])
    for f in tmp_path.glob("chatcmpl-old-*"):
        f.unlink()
    assert store.record("chatcmpl-old") is None  # pruned: no record, no raise
    st = os.stat(tmp_path)
    os.utime(tmp_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))  # coarse-mtime filesystems
    store.refresh()
    assert store.record("chatcmpl-new") is not None
    assert "chatcmpl-old" not in store.files and store.index[toks[:sr.KEY].astype(np.int32).tobytes()] == ["chatcmpl-new"]
    assert [r["hi"] for r in store.scratch_for(own, toks[:90])] == [80]
    # a head still being written is indexed once complete
    (tmp_path / "chatcmpl-part-abcdefgh.rt").write_bytes(np.array([0, 80, 2, 1], dtype=np.int32).tobytes())
    store.refresh()
    assert not store.known["chatcmpl-part-abcdefgh"][2]
    _route_rec(tmp_path, "chatcmpl-part-abcdefgh", 0, toks[:80])
    store.refresh()
    assert store.known["chatcmpl-part-abcdefgh"][2] and store.files["chatcmpl-part"] == ["chatcmpl-part-abcdefgh"]


def test_call_health_reports_r3_coverage():
    row = rl_seq.call_health({"served": {"windows": 6, "no_record": 2}, "encode": {"errors": ["e"] * 3}})
    assert row == {"encode_errors": 3, "served_windows": 8, "served_no_record": 2, "served_share": 0.75}
    assert rl_seq.call_health({}) == {"encode_errors": 0}
