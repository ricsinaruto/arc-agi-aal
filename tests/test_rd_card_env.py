"""Per-card server env (RL 26 Oct): one N-card run as a same-host A/B (mg.redesign.card_env)."""
import importlib.util
import json
from pathlib import Path

import pytest

from arena import rl_loop, rl_seq

ROOT = Path(__file__).resolve().parents[1]
BASE = {"ARC_LORA_SINGLE": "1", "ARC_MOE_LORA_BM16": "1", "ARC_EXPERT_MASK_COMPACT": "1", "PATH": "/usr/bin",
        "XDG_CACHE_HOME": "/tmp/qwen-cache", "VLLM_CACHE_ROOT": "/tmp/qwen-cache/vllm-env/aaaa",
        "ARC_SERVER2_ENV": json.dumps({"ARC_SLEEP_KEEP_KV": "1"})}


def test_card_env_settings():
    mg = {"redesign": {"streams": 8, "max_steps": 1, "cards": 2,
                       "card_env": {"0": {"ARC_LORA_SINGLE": "1"}, "1": {"ARC_LORA_SINGLE": None, "X": 3}}}}
    cfg = rl_seq.settings(mg)
    assert rl_seq.card_env(cfg) == {0: {"ARC_LORA_SINGLE": "1"}, 1: {"ARC_LORA_SINGLE": None, "X": "3"}}
    for bad in ({"2": {}}, {"a": {}}, {"1": ["x"]}, {"1": {"K": [1]}}):
        with pytest.raises(ValueError):
            rl_seq.settings({"redesign": {**mg["redesign"], "card_env": bad}})


def test_card_environment_removes_overrides_and_moves_the_aot_root():
    off = rl_loop.card_environment(BASE, {"ARC_LORA_SINGLE": None, "ARC_MOE_LORA_BM16": None}, "1")
    assert "ARC_LORA_SINGLE" not in off and "ARC_MOE_LORA_BM16" not in off and off["PATH"] == "/usr/bin"
    # a traced flag differs: server 1's AOT graph (traced with BM16) is not this server's
    assert off["VLLM_CACHE_ROOT"].startswith("/tmp/qwen-cache/vllm-env/") and off["VLLM_CACHE_ROOT"] != BASE["VLLM_CACHE_ROOT"]
    same = rl_loop.card_environment(BASE, {}, "1")
    assert same == BASE  # no overrides: server 1's env and graph root, as before
    keep_kv = rl_loop.card_environment(BASE, {"ARC_SLEEP_KEEP_KV": "1"}, "1")  # per-process (ARC_SERVER2_ENV)
    assert keep_kv["VLLM_CACHE_ROOT"] == BASE["VLLM_CACHE_ROOT"] and keep_kv["ARC_SLEEP_KEEP_KV"] == "1"
    with pytest.raises(ValueError, match="ARC_NEW_PATCH"):  # its patched files were never installed
        rl_loop.card_environment(BASE, {"ARC_NEW_PATCH": "1"}, "1")
    rec = rl_loop.server_env_record({**off, "CUDA_VISIBLE_DEVICES": "1", "HOME": "/root"})
    assert "HOME" not in rec and rec["CUDA_VISIBLE_DEVICES"] == "1" and "PATH" not in rec


def _server_dir(tmp_path, monkeypatch, env):
    """A fake first server: its identity names this process; /proc/<pid>/environ and cmdline are faked."""
    sd = tmp_path / "serving"
    sd.mkdir()
    (sd / "vllm-server-identity.json").write_text(json.dumps({"pid": 4242, "argv": ["vllm", "serve", "--port", "1234"]}))
    real = Path.read_bytes

    def read_bytes(self):
        if str(self) == "/proc/4242/environ":
            return "\0".join(f"{k}={v}" for k, v in env.items()).encode()
        if str(self) == "/proc/4242/cmdline":
            return b"vllm\0serve\0--port\x001234\0"
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.delenv("ARC_REAL_NVIDIA_SMI", raising=False)
    return sd


def test_second_servers_take_their_card_env_and_the_record_shows_each(tmp_path, monkeypatch):
    sd = _server_dir(tmp_path, monkeypatch, BASE)
    envs = {1: {"ARC_LORA_SINGLE": None, "ARC_MOE_LORA_BM16": None}}
    two, three = [rl_loop.SecondServer(sd, "m", card=i, port=1235 + i, gpu_util=None, name=f"server{i + 1}",
                                       env=envs.get(i)) for i in (1, 2)]
    _, e2 = two.command()
    _, e3 = three.command()
    assert "ARC_LORA_SINGLE" not in e2 and e2["CUDA_VISIBLE_DEVICES"] == "1"
    assert e3["ARC_LORA_SINGLE"] == "1" and e3["CUDA_VISIBLE_DEVICES"] == "2"

    class Loop:
        server_dir = sd

    two.effective_env = rl_loop.server_env_record(e2)
    rows = rl_seq.server_envs(Loop, [two, three])
    assert rows["server1"]["env"]["ARC_MOE_LORA_BM16"] == "1" and "ARC_MOE_LORA_BM16" not in rows["server2"]["env"]
    assert rows["server3"]["env"]["ARC_LORA_SINGLE"] == "1" and rows["server2"]["port"] == 1236
    assert json.loads((sd / "server-env.json").read_text()) == rows


def test_bootstrap_merges_card_zero_into_the_first_server(monkeypatch):
    import sys
    import types

    # in a bundle this file is bootstrap_arena.py beside bundle_bootstrap's bootstrap.py
    fake = types.ModuleType("bootstrap")
    fake.collect_evidence = fake.install_agent_environment = fake.verify = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "bootstrap", fake)
    spec = importlib.util.spec_from_file_location("arena_bootstrap", ROOT / "scripts/arena/bootstrap.py")
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)
    study = {"rl_loop": {"mg": {"redesign": {"card_env": {"0": {"ARC_LORA_SINGLE": "1", "ARC_OLD": None}}}}}}
    assert boot.card0_env(study) == {"ARC_LORA_SINGLE": "1", "ARC_OLD": None}
    assert boot.card0_env({"rl_loop": {"mg": {}}}) == {}


def test_parallel_boot_spawns_each_card_once_and_boot_waits_on_it(tmp_path, monkeypatch):
    """the boot thread starts server 2 beside server 1 (arena.server_start); SecondServer.boot then
    waits on that live process (no second spawn); whichever caller claims the card first spawns it."""
    import asyncio
    import subprocess
    import sys

    sd = _server_dir(tmp_path, monkeypatch, BASE)
    spawned = []
    real_popen = subprocess.Popen

    def popen(argv, **kw):  # a live stand-in process per spawn (a vLLM server would be argv)
        spawned.append(argv)
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
                          stdout=kw.get("stdout"), stderr=kw.get("stderr"))

    monkeypatch.setattr(rl_loop.subprocess, "Popen", popen)
    first = rl_loop.SecondServer(sd, "m", card=1, port=1236, gpu_util=None, name="server2")
    argv, env = first.start()  # the boot thread
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and len(spawned) == 1
    loop_side = rl_loop.SecondServer(sd, "m", card=1, port=1236, gpu_util=None, name="server2")
    assert loop_side.start() is None and len(spawned) == 1  # claimed: no second spawn
    got = loop_side.started()
    assert got is not None and got[0] == argv and loop_side.process.pid == first.process.pid

    async def ready_at_once(self, *a, **k):  # the HTTP readiness loop and mask check are not this test
        return {}

    monkeypatch.setattr(rl_loop.SecondServer, "mask_check", ready_at_once)

    class Resp:
        status_code = 200

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(rl_loop, "card_memory_of", lambda card: (1.0,))
    mem = asyncio.run(loop_side.boot())
    assert len(spawned) == 1 and mem["parallel_boot_s"] >= 0  # waited on the early server, spawned nothing
    first.process.kill()
    fresh = rl_loop.SecondServer(sd, "m", card=2, port=1237, gpu_util=None, name="server3")
    asyncio.run(fresh.boot())  # no early server for card 2: boot spawns it itself
    assert len(spawned) == 2 and "parallel_boot_s" not in fresh.memory
    fresh.process.kill()


def test_server_start_builds_the_servers_rl_seq_boots(tmp_path):
    from arena import server_start

    study = json.loads((ROOT / "launch/rl/runs/rl1-loop-seq-real-v10.json").read_text())["study"]
    got = server_start.servers(study, tmp_path)
    assert [(s.card, s.port, s.folder.name) for s in got] == [("1", 1236, "server2")]
    sys_path = ROOT / "scripts/posttrain"
    import sys

    sys.path.insert(0, str(sys_path))
    import pytest

    modal_rl1 = pytest.importorskip("modal_rl1")
    assert modal_rl1.parallel_ready('{"event": "boot_cache_restore", "found": true}', "") == "cache_restored"
    assert modal_rl1.parallel_ready('{"event": "boot_cache_carry", "found": true}', "torch.compile took 3 s") is None
    assert modal_rl1.parallel_ready("", "torch.compile took 3 s\ntorch.compile took 2 s") == "compiled"
