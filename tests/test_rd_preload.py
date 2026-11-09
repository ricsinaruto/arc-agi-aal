"""defer_trainer host preload (H attempt 2, 26 Oct: 305 s of the trainer load ran after the GPU gate)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/posttrain"))


def test_preload_needs_the_gate(monkeypatch):
    fq = pytest.importorskip("flash_qlora_job")
    assert fq.preload_wanted({"RL1_PRELOAD_HOST": "1", "RL1_GPU_GATE": "/tmp/g"})
    assert not fq.preload_wanted({"RL1_PRELOAD_HOST": "1"}) and not fq.preload_wanted({"RL1_GPU_GATE": "/tmp/g"})


def test_to_card_moves_all_but_the_table():
    torch = pytest.importorskip("torch")
    fq = pytest.importorskip("flash_qlora_job")

    class Ple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ngram_embedding = torch.nn.Embedding(10, 4)

    model = torch.nn.Sequential(torch.nn.Linear(4, 3), Ple())
    model.register_buffer("scale", torch.ones(2))
    lora = model[0].weight
    moved = fq.to_card(model, "meta")
    assert model[0].weight.device.type == "meta" and model[0].bias.device.type == "meta"
    assert model.scale.device.type == "meta" and model[1].ngram_embedding.weight.device.type == "cpu"
    assert moved == (12 + 3 + 2) * 4 and isinstance(model[0].weight, torch.nn.Parameter) and lora is not None
    assert fq.to_card(model, "meta") == 0  # already there


def test_preload_shims_answer_without_cuda_then_restore():
    torch = pytest.importorskip("torch")
    fq = pytest.importorskip("flash_qlora_job")
    import torch._dynamo.device_interface as di

    real = torch.cuda.get_device_capability
    info = fq.preload_shims(fq.card_info({"RL1_PRELOAD_CAPABILITY": "10.0"}))
    try:
        assert info["major"] == 10 and fq.preload_shims() is None  # once
        assert torch.cuda.get_device_capability() == (10, 0) and torch.cuda.current_device() == 0
        assert torch.cuda.get_device_properties(0).major == 10 and not torch.cuda.is_initialized()
        import os

        os.environ["RL1_PRELOAD_CAPABILITY"] = "10.0"  # 2-rank leg: Axolotl's p2p check, no context
        try:
            assert torch.cuda.can_device_access_peer(0, 1) is True and not torch.cuda.is_initialized()
        finally:
            os.environ.pop("RL1_PRELOAD_CAPABILITY")
        di.caching_worker_device_properties["cuda"] = ["filled meanwhile"]
    finally:
        assert fq.drop_preload_shims()["major"] == 10
    assert torch.cuda.get_device_capability is real and "cuda" not in di.caching_worker_device_properties
    assert fq.drop_preload_shims() is None


def test_stored_bytes_counts_inner_tensors():
    torch = pytest.importorskip("torch")
    fq = pytest.importorskip("flash_qlora_job")

    class Packed:  # an NVFP4Tensor reports BF16 numel; its storage is the packed data plus scales
        qdata, scale = torch.zeros(4, 8, dtype=torch.uint8), torch.zeros(4, 1, dtype=torch.float32)

        def __tensor_flatten__(self):
            return ["qdata", "scale"], {}

    assert fq.stored_bytes(Packed()) == 32 + 16 and fq.stored_bytes(torch.zeros(3, dtype=torch.bfloat16)) == 6


def test_load_digest_is_placement_free():
    torch = pytest.importorskip("torch")
    fq = pytest.importorskip("flash_qlora_job")

    a, b = torch.nn.Module(), torch.nn.Module()
    for m in (a, b):
        torch.manual_seed(0)
        m.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
        m.layers[3].register_buffer("scale16", torch.ones(3, dtype=torch.bfloat16))
    m = torch.nn.Module()
    m.model = a
    n = torch.nn.Module()
    n.model = b
    one, two = fq.load_digest(m), fq.load_digest(n)
    assert one == two and one["tensors"] == 3
    with torch.no_grad():
        b.layers[3].weight[0, 0] += 1
    assert fq.load_digest(n)["sha256"] != one["sha256"]


def test_wait_gate(tmp_path):
    fq = pytest.importorskip("flash_qlora_job")
    gate = tmp_path / "gate"
    gate.write_text("")
    assert fq.wait_gate(str(gate), poll=0.01) == 0.0


def test_defer_trainer_preload_host_reaches_the_trainer(tmp_path, monkeypatch):
    from arena import trainer_prespawn

    cfg = {"trainer_script": "rl1/rl1_trainer.py", "trainer_python": "py", "model_dir": "/m", "rank": 16, "lr": 1e-5,
           "window": 1024, "optimizer": "adamw32", "loop_lora": False, "tis_cap": 0, "path_chunk": 16384,
           "mg": {"redesign": {"defer_trainer": {"template": "/x", "preload_host": True}}}}
    monkeypatch.delenv("ARC_PREFLIGHT_FAKE_TRAINER", raising=False)
    _, env = trainer_prespawn.command_for(cfg, tmp_path, tmp_path / "w", gate=tmp_path / "gate")
    assert env["RL1_PRELOAD_HOST"] == "1" and env["RL1_GPU_GATE"] == str(tmp_path / "gate")
    _, env = trainer_prespawn.command_for(cfg, tmp_path, tmp_path / "w")  # no gate: a plain load
    assert "RL1_PRELOAD_HOST" not in env
    cfg["mg"]["redesign"]["defer_trainer"] = {"template": "/x", "preload_host": True, "preload_need_gib": 300}
    assert trainer_prespawn.command_for(cfg, tmp_path, tmp_path / "w", gate=tmp_path / "g")[1]["RL1_PRELOAD_NEED_GIB"] == "300.0"
    cfg["mg"]["redesign"]["defer_trainer"] = {"template": "/x"}
    assert "RL1_PRELOAD_HOST" not in trainer_prespawn.command_for(cfg, tmp_path, tmp_path / "w", gate=tmp_path / "g")[1]


def _study(**defer):
    return {"rl_loop": {"trainer_script": "rl1/rl1_trainer.py", "trainer_python": "py", "model_dir": "/m", "rank": 16,
                        "lr": 1e-5, "window": 1024, "optimizer": "adamw32", "loop_lora": False, "tis_cap": 0,
                        "path_chunk": 16384,
                        "mg": {"redesign": {"cards": 2, "defer_trainer": {"template": "/x", **defer}}}}}


def test_preloading_trainer_prespawns_with_the_loops_gate(tmp_path, monkeypatch):
    from arena import rl_loop, trainer_prespawn

    monkeypatch.delenv("ARC_PREFLIGHT_FAKE_TRAINER", raising=False)
    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    assert not trainer_prespawn.wanted(_study())  # no preload: the loop starts it (as H2)
    study = _study(preload_host=True)
    assert trainer_prespawn.wanted(study)
    resumed = tmp_path / "resume"
    resumed.mkdir()
    (resumed / "seq_state.json").write_text("{}")
    # a resume pre-spawns too (real-v9 28 Oct: +330 s loading after both boots); rl_seq opens its gate once asleep
    assert trainer_prespawn.preload_prespawn(study, {"ARC_RL1_RESUME": str(resumed)})
    study["rl_loop"]["mg"]["draft_weight"] = 0.5
    assert not trainer_prespawn.preload_prespawn(study, {})  # drafting: no defer
    study = _study(preload_host=True)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    monkeypatch.setattr(rl_loop, "settings", lambda raw, check=True: raw)
    spawned = {}

    class Popen:
        pid = 4242

        def __init__(self, command, env, **kw):
            spawned.update(command=command, env=env)

    monkeypatch.setattr(trainer_prespawn.subprocess, "Popen", Popen)
    work = tmp_path / "w"
    work.mkdir()
    (work / trainer_prespawn.GATE).write_text("")  # a stale gate from an earlier start: removed
    assert trainer_prespawn.spawn(path, tmp_path, tmp_path / "log", work=work) == 4242
    env = spawned["env"]
    assert env["RL1_GPU_GATE"] == str(work / "trainer-gpu-gate") and not (work / "trainer-gpu-gate").exists()
    assert env["RL1_PRELOAD_HOST"] == "1" and env["RL1_PRELOAD_RESERVE_GIB"] == "184.8"
    # the loop's own start (rl_seq: start_trainer(gate=work/trainer-gpu-gate)) adopts it: equal but the reserve
    assert spawned["env"]["RL1_DIST"] == "1"  # one rank per card, as rl_seq sets mg.trainer_cards (pf v2N2)
    study["rl_loop"]["mg"]["trainer_cards"] = 2
    command, loop_env = trainer_prespawn.command_for(study["rl_loop"], tmp_path, work, work / "trainer-gpu-gate")
    record = json.loads((work / "trainer" / trainer_prespawn.RECORD).read_text())
    assert record["command"] == command
    assert {k for k in set(record["env"]) | set(loop_env) if record["env"].get(k) != loop_env.get(k)} \
        <= trainer_prespawn.ALLOWED


def test_preload_room_counts_the_reserve():
    fq = pytest.importorskip("flash_qlora_job")
    room = fq.preload_room({"RL1_PRELOAD_NEED_GIB": "100", "RL1_PRELOAD_RESERVE_GIB": "100"}, {"mem_available_gib": 205.0})
    assert room["need_gib"] == 210.0 and not room["ok"]


def test_preload_room_against_available_and_cgroup(tmp_path):
    fq = pytest.importorskip("flash_qlora_job")
    (tmp_path / "meminfo").write_text(f"MemTotal: {1000 * 2 ** 20} kB\nMemAvailable: {900 * 2 ** 20} kB\n")
    (tmp_path / "memory.max").write_text(str(1200 * 2 ** 30))
    (tmp_path / "memory.current").write_text(str(700 * 2 ** 30))
    memory = fq.host_memory(tmp_path / "meminfo", tmp_path)
    assert memory == {"mem_total_gib": 1000.0, "mem_available_gib": 900.0, "cgroup_max_gib": 1200.0,
                      "cgroup_current_gib": 700.0}
    one = fq.preload_room({}, memory)  # the cgroup's 500 GiB headroom is the smaller room
    assert one["room_gib"] == 500.0 and one["need_gib"] == 420.0 and one["ok"]
    assert not fq.preload_room({"LOCAL_WORLD_SIZE": "2"}, memory)["ok"]  # two trainers loading at once: 840
    assert fq.preload_room({"LOCAL_WORLD_SIZE": "2", "RL1_PRELOAD_NEED_GIB": "200"}, memory)["ok"]
    assert not fq.preload_room({}, {"mem_available_gib": None})["ok"]  # unknown room: the direct path


def test_preload_skipped_without_room(monkeypatch, capsys):
    fq = pytest.importorskip("flash_qlora_job")
    monkeypatch.setattr(fq, "PRELOAD", {})
    monkeypatch.setenv("RL1_PRELOAD_HOST", "1")
    monkeypatch.setenv("RL1_GPU_GATE", "/tmp/never")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("RL1_PRELOAD_NEED_GIB", raising=False)
    monkeypatch.setattr(fq, "host_memory", lambda: {"mem_available_gib": 300.0})
    assert fq.preload_wanted() is False and fq.PRELOAD["asked"] and not fq.PRELOAD["room"]["ok"]
    assert '"event": "preload_skipped"' in capsys.readouterr().out
    monkeypatch.setattr(fq, "host_memory", lambda: {"mem_available_gib": 3000.0})
    assert fq.preload_wanted() is False  # decided once per process
    monkeypatch.setattr(fq, "PRELOAD", {})
    assert fq.preload_wanted() is True
    monkeypatch.delenv("RL1_PRELOAD_HOST")
    monkeypatch.setattr(fq, "PRELOAD", {})
    assert fq.preload_wanted() is False and "room" not in fq.PRELOAD


def test_dist_defers_the_card_under_preload(monkeypatch):
    # H3 (26 Oct): rank setup's torch.cuda.set_device opened the context, which waited ~450 s for the gate before the
    # host load began (cuda_before_gate true); preloading ranks set the card after the gate (fq.load)
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    import rl1_dist

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(("set_device", d)))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda backend, **kw: calls.append(("init", backend)))
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "new_group", lambda **kw: "ctl")
    monkeypatch.setenv("RL1_DIST", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    d = rl1_dist.init(defer_device=True)
    assert calls == [("init", "nccl")] and d.device == "cuda:1" and d.rank == 1
    calls.clear()
    rl1_dist.init()
    assert calls == [("set_device", 1), ("init", "nccl")]  # the direct path as before


def test_resume_opens_a_prespawned_trainers_gate(tmp_path):
    from types import SimpleNamespace

    from arena import rl_seq, trainer_prespawn

    loop = SimpleNamespace(work=tmp_path)
    assert rl_seq.prespawned_gate(loop) == {}  # no pre-spawn: a plain load, no gate file
    assert not (tmp_path / trainer_prespawn.GATE).exists()
    (tmp_path / "trainer").mkdir()
    (tmp_path / "trainer" / trainer_prespawn.RECORD).write_text("{}")
    got = rl_seq.prespawned_gate(loop)
    assert got == {"gate": tmp_path / trainer_prespawn.GATE} and got["gate"].is_file()


def test_release_host_returns_freed_arena_memory():
    """real-v10 28 Oct: ~93 GiB per trainer rank sat in ~1,700 glibc arena regions of 1-64 MB after the threaded host
    load; release_host (gc + malloc_trim) after the move returns freed arena chunks to the OS."""
    import sys
    import threading
    from pathlib import Path

    if not Path("/proc/self/status").exists():
        pytest.skip("needs /proc and glibc (Linux)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
    fq = pytest.importorskip("flash_qlora_job")
    keep = []

    def churn():  # a load-like pattern: many 1 MB blocks per thread, most freed, a few kept
        blocks = [bytearray(1 << 20) for _ in range(256)]
        keep.append(blocks[::64])
        del blocks

    threads = [threading.Thread(target=churn) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = fq.release_host()
    if not out.get("trimmed"):
        pytest.skip("no glibc malloc_trim")
    before, after = out["before"].get("VmRSS", 0.0), out["after"].get("VmRSS", 0.0)
    assert after <= before  # never grows; on glibc the freed ~4 GiB of arena chunks go back
