"""Stage H (26 Oct): N-card launch pieces: the boot timeline, the launcher's card count and host plan, the prebuilt
agent venv."""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arena import boot_timeline

ROOT = Path(__file__).resolve().parents[1]
LOG = """(APIServer pid=1) INFO 10-26 10:14:21 [api_utils.py:333] start
(EngineCore pid=2) INFO 10-26 10:15:14 [core.py:122] Initializing a V1 LLM engine (v0) with config
(Worker pid=3) INFO 10-26 10:15:28 [model_runner.py:385] Loading model from scratch...
(Worker pid=3) INFO 10-26 10:17:40 [default_loader.py:430] Loading weights took 90.81 seconds
(Worker pid=3) INFO 10-26 10:18:52 [default_loader.py:430] Loading weights took 47.72 seconds
(Worker pid=3) INFO 10-26 10:18:53 [model_runner.py:407] Model loading took 57.22 GiB memory and 204.885438 seconds
(Worker pid=3) INFO 10-26 10:19:46 [backends.py:1155] Dynamo bytecode transform time: 25.65 s
(Worker pid=3) INFO 10-26 10:20:39 [monitor.py:53] torch.compile took 78.51 s in total
(Worker pid=3) 2026-10-26 10:21:19,901 - INFO - autotuner.py:829 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker pid=3) 2026-10-26 10:22:39,202 - INFO - autotuner.py:852 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker pid=3) INFO 10-26 10:24:08 [model_runner.py:953] Graph capturing finished in 89 secs, took 2.41 GiB
(EngineCore pid=2) INFO 10-26 10:24:20 [core.py:347] init engine (profile, create kv cache, warmup model) took 326.63 s (compilation: 82.68 s)
(APIServer pid=1) INFO 10-26 10:24:30 [api_server.py:657] Starting vLLM server on http://127.0.0.1:1234
"""
ZERO = 1793009585.7  # T3 attempt 2's container start (10:13:05.7 UTC)


def test_server_phases_from_the_t3_log_shape():
    out = boot_timeline.server_phases(LOG, ZERO)
    assert out["weights_s"] == [90.81, 47.72] and out["graphs_s"] == 89.0 and out["autotune_s"] == 80.0
    assert out["engine_init_s"] == 326.63 and out["compile_s"] == [78.51]
    assert out["start_t"] == 75.3 and out["ready_t"] == 684.3 and out["api_ready_at"] == 609.0
    assert out["aot_loaded"] is False


def test_timeline_reads_launcher_rows_and_provenance(tmp_path):
    serving, output = tmp_path / "serving", tmp_path / "out"
    serving.mkdir()
    output.mkdir()
    (serving / "vllm-openai-server.log").write_text(LOG)
    (serving / "vllm-setup-provenance.json").write_text(json.dumps({"setup_started_epoch": ZERO + 31.8,
                                                                   "completed_epoch": ZERO + 685.4}))
    (output / "bootstrap.log").write_text(
        '{"event": "weights_prefetched", "threads": 16, "gib": 125.91, "seconds": 25.6}\n'
        'ARC_RUNTIME_CACHE_RESTORED {"key": "87ae", "parts": 9, "seconds": 22.9}\n'
        '{"event": "boot_cache_restore", "found": false, "path": "/runs/boot-cache/hero/x.tar"}\nnoise {not json\n')
    line = boot_timeline.timeline(serving, {"ARC_RL1_STARTED_EPOCH": str(ZERO), "ARC_VALIDATION_OUTPUT": str(output)})
    assert line["setup_t"] == 31.8 and line["setup_done_t"] == 685.4 and line["runtime_cache_s"] == 22.9
    assert line["boot_cache_restore"]["found"] is False and line["weights_prefetched"]["seconds"] == 25.6
    assert line["server1"]["ready_t"] == 684.3
    assert boot_timeline.timeline(tmp_path / "none", {}) == {"started_epoch": None}  # nothing there: no failure


def test_first_token_waits_for_tokens_past_the_baseline():
    seen = iter([5.0, 5.0, 9.0])

    async def get(url):
        return f"vllm:generation_tokens_total {next(seen)}\n"

    waited = asyncio.run(boot_timeline.first_token(["http://a/v1"], {"http://a/v1": 5.0}, timeout=5, poll=0.01, get=get))
    assert waited is not None

    async def none(url):
        raise OSError("down")

    assert asyncio.run(boot_timeline.first_token(["http://a/v1"], {}, timeout=0.05, poll=0.01, get=none)) is None


def test_launcher_cards_host_plan_and_trigger():
    m = pytest.importorskip("modal_rl1")
    assert m.study_cards({"rl_loop": {"mg": {"redesign": {"cards": 4}}}}) == 4
    assert m.study_cards(None) == 1
    plan = m.host_plan(8, {"MemTotal": 2000.0, "MemAvailable": 1900.0})
    assert plan["server_gib"] == 122.5 and plan["need_gib"] == 8 * 122.5 + 8 * m.TRAINER_HOST_GIB and plan["ok"]
    assert not m.host_plan(8, {"MemTotal": 1000.0})["ok"]
    shared = m.host_plan(8, {"MemTotal": 1000.0}, {"shared_ple": True})
    assert shared["need_gib"] == pytest.approx(8 * 122.5 + 8 * 40.0 - 7 * 47.7)
    assert m.host_plan(2, {})["cards"] == 2 and "ok" not in m.host_plan(2, {})
    info = m.meminfo("/nonexistent")
    assert info == {}


def test_trainer_cache_restore_and_save(tmp_path, monkeypatch):
    m = pytest.importorskip("modal_rl1")
    import threading

    root, archives, feed = tmp_path / "cache", tmp_path / "arch", tmp_path / "feed.jsonl"
    identity = {}
    assert not m.trainer_cache_restore("k1", identity, root, archives)["restored"]  # nothing on the volume yet
    (root / "triton" / "abc").mkdir(parents=True)
    (root / "triton" / "abc" / "kernel.cubin").write_bytes(b"x")
    (root / "inductor" / "fx").mkdir(parents=True)
    (root / "inductor" / "fx" / "graph").write_bytes(b"y")
    stop = threading.Event()
    feed.write_text('{"kind": "step", "step": 1}\n')
    row = m.trainer_cache_save(stop, "k1", feed, root, archives, poll=0.01, commit=lambda: None)
    assert row["files"] == 2 and (archives / "k1.tar").is_file()
    assert m.trainer_cache_save(stop, "k1", feed, root, archives, poll=0.01, commit=lambda: None) is None  # no growth
    fresh = tmp_path / "fresh"
    out = m.trainer_cache_restore("k1", identity, fresh, archives)
    assert out["restored"] and out["files"] == 2 and (fresh / "triton/abc/kernel.cubin").read_bytes() == b"x"
    carried = m.trainer_cache_restore("k9", identity, tmp_path / "other", archives)  # a new trainer key (26 Oct)
    assert carried["restored"] and carried["carried"] == "k1" and (tmp_path / "other/triton/abc/kernel.cubin").is_file()
    stop.set()
    assert m.trainer_cache_save(stop, "k2", tmp_path / "none.jsonl", root, archives, poll=0.01) is None  # no step


def test_trainer_command_points_the_caches_at_the_restored_folder(tmp_path, monkeypatch):
    from arena.rl_loop import trainer_command

    cfg = {"trainer_script": "rl1/rl1_trainer.py", "trainer_python": "py", "model_dir": "/m", "rank": 16, "lr": 1e-5,
           "window": 1024, "optimizer": "adamw32", "loop_lora": False, "tis_cap": 0, "path_chunk": 16384, "mg": {}}
    monkeypatch.delenv("ARC_PREFLIGHT_FAKE_TRAINER", raising=False)
    monkeypatch.delenv("ARC_TRAINER_CACHE", raising=False)
    _, env = trainer_command(cfg, tmp_path, tmp_path / "w")
    assert "TRITON_CACHE_AUTOTUNING" not in env
    monkeypatch.setenv("ARC_TRAINER_CACHE", "/tmp/rl1-compile-cache")
    _, env = trainer_command(cfg, tmp_path, tmp_path / "w")
    assert env["TRITON_CACHE_DIR"] == "/tmp/rl1-compile-cache/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/rl1-compile-cache/inductor" and env["TRITON_CACHE_AUTOTUNING"] == "1"


def test_launcher_n_card_function_settings():
    """The launcher's N-card plumbing as source: GPU B300:N for any N > 1, the GPU-0 nvidia-smi shim for N >= 2, the
    16-core reservation kept with a burst ceiling above two cards."""
    text = (ROOT / "scripts/posttrain/modal_rl1.py").read_text()
    assert 'GPU, GPU_RATE = f"{GPU}:{CARDS}", CARDS * GPU_RATE' in text
    assert "if CARDS >= 2 and shutil.which(\"nvidia-smi\")" in text
    assert "RUN_CPU = RL_CPU if CARDS <= 2 else (RL_CPU, 8.0 * CARDS)" in text


def test_prebuilt_venv_used_only_on_an_equal_stamp(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/arena"))
    try:
        import bundle_bootstrap as BB
    finally:
        sys.path.pop(0)
    bundle, pre = tmp_path / "bundle", tmp_path / "pre"
    bundle.mkdir()
    (bundle / "requirements.txt").write_text("numpy==2.5.3 --hash=sha256:00\n")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(pre)], check=True)
    env = {BB.PREBUILT_ENV: str(pre)}
    assert BB.prebuilt_environment(bundle, tmp_path / "v0", env) is None  # no stamp
    (pre / BB.STAMP).write_text("other\n")
    assert BB.prebuilt_environment(bundle, tmp_path / "v1", env) is None
    (pre / BB.STAMP).write_text(BB.requirements_stamp(bundle) + "\n")
    python = BB.prebuilt_environment(bundle, tmp_path / "v2", env)
    assert python == str(tmp_path / "v2/bin/python")
    out = subprocess.run([python, "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()).resolve() == (tmp_path / "v2").resolve()
    assert BB.prebuilt_environment(bundle, tmp_path / "v3", {}) is None


def test_zero_adapter_keeps_the_layout_with_zero_bytes(tmp_path):
    from arena import fake_trainer, rl_seq
    from arena.rl_loop import adapter_tensors

    template = Path(fake_trainer.adapter(tmp_path / "t"))
    raw = bytearray((template / "adapter_model.safetensors").read_bytes())
    raw[-1] = 7  # a trained adapter's nonzero byte
    (template / "adapter_model.safetensors").write_bytes(bytes(raw))
    n = rl_seq.zero_adapter(template, tmp_path / "z")
    out = (tmp_path / "z" / "adapter_model.safetensors").read_bytes()
    assert n == 4 and len(out) == len(raw) and out[:len(raw) - 8] == bytes(raw[:len(raw) - 8])
    assert set(out[-8:]) == {0}
    assert adapter_tensors(tmp_path / "z") == 4  # the loop's manifest check passes
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(bytes(raw[8:8 + size]))
    header[next(iter(header))]["dtype"] = "I8"
    head = json.dumps(header).encode()
    (template / "adapter_model.safetensors").write_bytes(len(head).to_bytes(8, "little") + head + b"\0" * 8)
    with pytest.raises(ValueError, match="dtypes"):
        rl_seq.zero_adapter(template, tmp_path / "z2")


def test_defer_trainer_plays_first_and_joins_at_the_first_switch(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_rl_seq import FakeLoop, FakeTrainer, Upstream
    finally:
        sys.path.pop(0)
    import arena.restore as restore
    from arena import phase, rl_seq

    monkeypatch.setattr(rl_seq.harvest_mod, "snapshot", lambda a, d: Path(d).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(restore, "unsafe_from", lambda d: (None, None))
    monkeypatch.setattr(restore, "load", lambda d, r: {"level": int(Path(d).name.split("-L")[1].split("-")[0]) + 1})
    monkeypatch.setenv("ARC_PREFLIGHT_FAKE_TRAINER", "1")  # no template: the stand-in adapter shape
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    loop = FakeLoop(tmp_path, max_steps=2)
    loop.cfg["mg"]["redesign"]["defer_trainer"] = {}
    gate_seen = {}

    async def start_trainer(gate=None):
        loop.log.append("start_trainer")
        while not Path(gate).exists():  # the card part waits for the gate, as rl1_trainer.gpu_gate
            await asyncio.sleep(0.01)
        gate_seen["open"] = loop.log.count("sleep")
        loop.trainer = FakeTrainer(loop.log)
        return {"seconds": 0.0}

    loop.start_trainer = start_trainer
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    log = loop.log
    assert log.index("load:policy:v0000") < log.index("sleep")  # played on the zero adapter before any sleep
    assert "save" not in log[:log.index("sleep")]  # no trainer save before the first switch
    assert gate_seen["open"] == 1  # the gate opened only once the server slept
    start = loop.feed.of("seq_start")[0]
    assert start["phases"]["deferred_trainer"] and start["phases"]["zero_adapter_tensors"] == 4
    assert loop.feed.of("seq_trainer_joined") and loop.feed.of("seq_end")[0]["step"] == 2
    zero = (tmp_path / "work/adapters/v0000/adapter_model.safetensors").read_bytes()
    assert set(zero[-8:]) == {0}


def test_defer_trainer_skipped_on_a_resume(tmp_path, monkeypatch):
    from arena import rl_seq

    root = tmp_path / "r"
    root.mkdir()
    (root / "seq_state.json").write_text("{}")
    monkeypatch.setenv("ARC_RL1_RESUME", str(root))
    assert rl_seq.resume_state_exists()
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "none"))
    assert not rl_seq.resume_state_exists()


def test_async_checkpoint_writes_state_only_after_the_folder_lands(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_rl_seq import FakeLoop, FakeTrainer, Upstream
    finally:
        sys.path.pop(0)
    import arena.restore as restore
    from arena import phase, rl_seq

    monkeypatch.setattr(rl_seq.harvest_mod, "snapshot", lambda a, d: Path(d).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(restore, "unsafe_from", lambda d: (None, None))
    monkeypatch.setattr(restore, "load", lambda d, r: {"level": int(Path(d).name.split("-L")[1].split("-")[0]) + 1})
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    pending, written = [], []

    class BgTrainer(FakeTrainer):
        async def must(self, op, **kw):
            if op == "checkpoint" and kw.get("background"):
                self.log.append(op)
                for p in pending:  # a new checkpoint joins the earlier write
                    Path(p).mkdir(parents=True, exist_ok=True)
                pending[:] = [kw["path"]]
                return {"ok": True, "background": True}
            if op == "checkpoint_wait":
                for p in pending:
                    Path(p).mkdir(parents=True, exist_ok=True)
                pending.clear()
                return {"ok": True, "pending": True}
            return await super().must(op, **kw)

    loop = FakeLoop(tmp_path, max_steps=2)
    loop.cfg["mg"]["async_checkpoint"] = True

    async def start_trainer(gate=None):
        loop.trainer = BgTrainer(loop.log)

    loop.start_trainer = start_trainer
    put = rl_seq.Seq.put_state

    def spy(self, ckpt=None, adapter=None, state=None):
        written.append((state or {}).get("step"))
        assert Path(state["checkpoint"]).is_dir()  # never a state naming a folder still being written
        return put(self, ckpt, adapter, state)

    monkeypatch.setattr(rl_seq.Seq, "put_state", spy)
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=Upstream())), 30))
    steps = loop.feed.of("step")
    assert len(steps) == 2 and all(r["checkpoint_background"] for r in steps)
    assert written == [1, 2] and [r["step"] for r in loop.feed.of("checkpoint_state")] == [1, 2]


