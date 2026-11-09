"""Kernel-profile pilot (docs/validation-20260930/kernel-profile.md): the process / cgroup readings, the pure-decode
test, the prompt subsets and the whole point loop against a fake engine. CPU only."""

import asyncio
import gzip
import json
import math
from pathlib import Path

import httpx
from test_arena_cache import layout
from test_hero_v2_precheck import FakeEngine

from arena import kernel_profile as kp
from arena import throughput_bench as tb


def test_process_roles_and_cpu_readings(tmp_path):
    for pid, comm, cmd in ((11, "VLLM::EngineCor", "VLLM::EngineCore"), (12, "VLLM::Worker_TP", "VLLM::Worker_TP0"),
                           (13, "python", "python -m vllm.entrypoints.cli.main serve /m --port 1"),
                           (14, "python", "python bench.py")):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "comm").write_text(comm + "\n")
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
        # utime 300 ticks, stime 100 ticks (fields 14 and 15 of /proc/pid/stat)
        (d / "stat").write_text(f"{pid} ({comm}) S 1 1 1 0 -1 0 0 0 0 0 300 100 0 0 20 0 1 0 0 0 0\n")
    (tmp_path / "stat").write_text("cpu  100 0 50 800 0 0 0 50 0 0\ncpu0 1 1 1 1\n")
    roles = kp.serving_processes(tmp_path)
    assert roles == {"engine_core": [11], "worker": [12], "api_server": [13]}
    assert kp.process_cpu_seconds(11, tmp_path) == 400 / kp.TICK
    assert kp.process_cpu_seconds(99, tmp_path) is None
    assert kp.host_cpu(tmp_path)["steal"] == 50
    first = {"epoch": 0.0, "processes": {"engine_core": 1.0}, "cgroup": {"usage_usec": 0, "nr_throttled": 0},
             "host": {"user": 0, "idle": 0, "steal": 0}}
    last = {"epoch": 2.0, "processes": {"engine_core": 1.5}, "cgroup": {"usage_usec": 4_000_000, "nr_throttled": 3},
            "host": {"user": 60, "idle": 30, "steal": 10}}
    delta = kp.cpu_delta(first, last, steps=50)
    assert delta["process_cpu_ms_per_step"] == {"engine_core": 10.0}
    assert delta["process_cpu_share"] == {"engine_core": 0.25}
    assert delta["cgroup"] == {"usage_usec": 4_000_000, "nr_throttled": 3, "cores_used": 2.0}
    assert delta["host_share"]["steal"] == 0.1


def test_pure_decode_and_subsets():
    c = kp.COUNTERS
    a = (0.0, {c["steps"]: 10, c["scheduled_tokens"]: 400, c["generated"]: 400})
    b = (1.0, {c["steps"]: 40, c["scheduled_tokens"]: 1600, c["generated"]: 1600})
    mixed = (1.0, {c["steps"]: 40, c["scheduled_tokens"]: 5700, c["generated"]: 1600})
    assert kp.pure_decode(a, b) and not kp.pure_decode(a, mixed) and not kp.pure_decode(a, a)
    counts = [5, 9, 1, 7]
    assert kp.chosen_indices(counts, 2, "first") == [0, 1]
    assert kp.chosen_indices(counts, 2, "longest") == [1, 3]
    assert kp.gate_blocks([3000, 100], 136, 3136, 9) == (1 + 9) + (1 + 9)
    body = kp.forced({"seed": 1, "max_tokens": 1, "temperature": 0.7}, 64, {"temperature": 0.0}, ("seed",))
    assert body == {"max_tokens": 64, "min_tokens": 64, "ignore_eos": True, "temperature": 0.0}


class ProfilingEngine(FakeEngine):
    """The fake engine plus /start_profile and /stop_profile (a small chrome trace per profile)."""

    def __init__(self, folder, **kwargs):
        super().__init__(256, 16, **kwargs)
        self.folder, self.profiles = Path(folder), 0

    async def handler(self, request):
        path = request.url.path
        if path.endswith("/start_profile"):
            return httpx.Response(200, json={})
        if path.endswith("/stop_profile"):
            self.profiles += 1
            self.folder.mkdir(parents=True, exist_ok=True)
            events = [{"ph": "X", "cat": "kernel", "name": f"marlin_moe_{i % 3}", "ts": 10 * i, "dur": 8}
                      for i in range(30)]
            with gzip.open(self.folder / f"rank0.{self.profiles}.pt.trace.json.gz", "wt") as f:
                json.dump({"traceEvents": events}, f)
            return httpx.Response(200, json={})
        return await super().handler(request)


async def test_points_against_a_fake_engine(tmp_path, monkeypatch):
    engine = ProfilingEngine(tmp_path / tb.PROFILE_DIR)
    transport = httpx.MockTransport(engine.handler)
    real_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    monkeypatch.setattr(tb.GpuMonitor, "run", lambda self: asyncio.sleep(3600))

    async def no_smi(*args, **kwargs):
        return None

    monkeypatch.setattr(kp, "smi", no_smi)
    (tmp_path / "server.log").write_text("worker ARC_CACHE_LAYOUT " + json.dumps(layout()) + "\n")
    tools = [{"type": "function", "function": {"name": "python", "parameters": {"type": "object"}}}]
    contexts = [{"messages": [{"role": "system", "content": "s" * (600 + 60 * i)}, {"role": "user", "content": "go"}],
                 "tools": tools} for i in range(6)]
    (tmp_path / "src/bench").mkdir(parents=True)
    (tmp_path / "src/bench/ctx.json").write_text(json.dumps({"contexts": contexts}))
    points = [{"label": "R4", "content": "real", "streams": 4, "profile": True, "profile_seconds": 0.1},
              {"label": "R3-greedy", "content": "real", "streams": 3, "subset": "longest",
               "overrides": {"temperature": 0.0}},
              {"label": "R4-cpu1", "content": "real", "streams": 4, "load": {"cpu_burners": 1}},
              {"label": "R4-tokenize", "content": "real", "streams": 4, "load": {"tokenize_rate": 20}},
              {"label": "R4-mixed", "content": "real", "streams": 4,
               "inject": {"prefix_tokens": 300, "tail_tokens": 80, "rate": 5}},
              {"label": "S3", "content": "synthetic", "streams": 3, "drop": ["seed"]}]
    spec = {"contexts_file": "bench/ctx.json", "output_tokens": 256, "synthetic_streams": 3, "calibrate_tokens": 400,
            "window_seconds": 0.6, "settle_seconds": 10, "quiet_seconds": 0.2, "sample_seconds": 0.1,
            "inject_lead_seconds": 0.2, "load_lead_seconds": 0.1, "points": points}
    study = {"kv_margin": 0.9, "generation_slots": 32, "completion_cap": 16384, "reasoning_effort": "xhigh",
             "temperature": 0.7, "server": {"mtp_tokens": 0},
             "throughput_bench": {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None,
                                  "kernel_profile": spec}}
    report = await tb.run_bench(study, {"name": "kprof", "kind": "serving-bench"}, output=tmp_path / "out",
                                server_dir=tmp_path, base_url="http://model/v1", transport=transport,
                                log=lambda *a, **k: None, source=tmp_path / "src")
    assert report["status"] == "complete", report["task_errors"]
    result = report["kernel_profile"]
    assert not result["errors"], result["errors"]
    assert [p["label"] for p in result["points"]] == [p["label"] for p in points]
    assert len(result["real_prompt_tokens"]) == 6 and len(result["synthetic_prompt_tokens"]) == 3
    for point in result["points"]:
        assert point["valid"] and not point["ended_early"] and point["steps"] > 0
        assert math.isclose(point["generated_per_step"], point["streams"], rel_tol=0.05)
        assert point["pure_decode"] and "process_cpu_ms_per_step" in point["cpu"]
    by = {p["label"]: p for p in result["points"]}
    assert by["R3-greedy"]["mean_prompt_tokens"] > by["R4"]["mean_prompt_tokens"]  # the longest three
    assert by["R4-cpu1"]["cpu_burners"] == 1 and by["R4-tokenize"]["tokenize_load"]["completed"] >= 1
    assert by["R4-mixed"]["inject"]["sent"] >= 1
    (trace,) = by["R4"]["profile"]["traces"]
    assert trace["summary"]["kernels"] == 30 and (tmp_path / "out" / trace["file"]).is_file()
    assert engine.running == 0
