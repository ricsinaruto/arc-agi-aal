"""Pinned serving recipe guard, drain check, GPU sampling and profile validation."""

import asyncio
import json
import os
import sys
import time

import httpx
import pytest

from arc_infra.config import Config
from arc_infra.metrics import GpuMonitor, Monitor, parse_gpu_rows
from arc_infra.serving import BASE_URL, RUNTIME_DATASET, SOURCE_DATASET, load_recipe
from arc_infra.serving_checks import validate_config, wait_idle


def test_changed_external_setup_is_rejected_before_import(tmp_path):
    for name in (SOURCE_DATASET, RUNTIME_DATASET):
        (tmp_path / "datasets" / name).mkdir(parents=True)
    marker = tmp_path / "should-not-execute"
    (tmp_path / "datasets" / SOURCE_DATASET / "serving_setup.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    with pytest.raises(RuntimeError, match="differs from reviewed"):
        load_recipe(tmp_path / "output", tmp_path)
    assert not marker.exists()


def test_duck_profile_config_is_accepted_and_deviations_rejected():
    from pathlib import Path

    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/duck_controls.yaml")
    validate_config(cfg)
    cfg.temperature = 0.6
    with pytest.raises(ValueError, match="temperature 0.7"):
        validate_config(cfg)
    cfg.temperature = 0.7
    cfg.gameplay_seconds = 700
    with pytest.raises(ValueError, match="600-, 900-"):
        validate_config(cfg)
    cfg.gameplay_seconds = 900
    assert cfg.max_model_len == 65536 and cfg.reasoning_effort == "low"
    assert cfg.max_output_tokens == 8192
    validate_config(cfg)
    cfg.max_output_tokens = 1024
    with pytest.raises(ValueError, match="output cap"):
        validate_config(cfg)
    cfg.max_output_tokens = 0
    validate_config(cfg)
    cfg.max_model_len = 32768
    validate_config(cfg)
    cfg.max_model_len = 49152
    with pytest.raises(ValueError, match="pinned serving profile"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="reasoning_effort"):
        Config(reasoning_effort="high")


def test_setup_arguments_and_context_override(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from arc_infra import setup

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "kaggle"))
    from flash_bootstrap import setup_command

    args = setup.parse(["--output", "/o"])
    assert (args.max_model_len, args.kv_cache_gib, args.mtp_tokens) == (32768, 5, 3)
    args = setup.parse(["--output", "/o", "--max-model-len", "65536", "--kv-cache-gib", "8"])
    assert (args.max_model_len, args.kv_cache_gib) == (65536, 8)
    assert args.prefix_caching is False
    assert setup.parse(["--output", "/o", "--prefix-caching"]).prefix_caching is True
    with pytest.raises(SystemExit):
        setup.parse(["--output", "/o", "--max-model-len", "49152"])
    recipe = setup.configure(SimpleNamespace(ANALYZER_CONTEXT=32768), 65536)
    assert recipe.ANALYZER_CONTEXT == 65536
    with pytest.raises(ValueError, match="Unsupported"):
        setup.configure(recipe, 49152)
    command = setup_command("/src", "/out", {"max_model_len": 65536, "kv_cache_gib": 8})
    assert command[-4:] == ["--max-model-len", "65536", "--kv-cache-gib", "8"]
    assert command[command.index("--output") + 1] == "/out/server"
    assert "--max-model-len" not in setup_command("/src", "/out")
    assert "--prefix-caching" not in setup_command("/src", "/out", {"max_model_len": 65536})
    assert setup_command("/src", "/out", {"prefix_caching": True})[-1] == "--prefix-caching"
    # v5b: the corrected trial's server runs the fork with a profiled KV cache, in its own dir.
    fork = setup_command(
        "/src", "/out",
        {"max_model_len": 65536, "kv_cache_gib": 0, "prefix_caching": True, "gpu_memory_utilization": "0.96"},
        working="server-duck-transfer",
    )
    assert fork[fork.index("--output") + 1] == "/out/server-duck-transfer"
    assert fork[-5:] == ["--kv-cache-gib", "0", "--prefix-caching", "--gpu-memory-utilization", "0.96"]
    # v5c: the fork server may run under the agent venv interpreter.
    assert setup_command("/src", "/out", {"max_model_len": 262144}, python="/venv/bin/python")[0] == "/venv/bin/python"
    from flash_bootstrap import start_server

    with pytest.raises(ValueError, match="agent venv"):
        start_server("/src", "/out", {"max_model_len": 262144, "python": "agent"})
    args = setup.parse(["--output", "/o", "--kv-cache-gib", "0", "--gpu-memory-utilization", "0.96"])
    assert args.gpu_memory_utilization == "0.96" and args.kv_cache_gib == 0
    # v5c: the native 262K context is the fork's (profiled cache) only.
    native = setup.parse(["--output", "/o", "--max-model-len", "262144", "--kv-cache-gib", "0", "--gpu-memory-utilization", "0.96"])
    assert native.max_model_len == 262144
    with pytest.raises(SystemExit):
        setup.parse(["--output", "/o", "--max-model-len", "262144"])
    assert setup.configure(SimpleNamespace(ANALYZER_CONTEXT=32768), 262144).ANALYZER_CONTEXT == 262144
    assert setup.parse(["--output", "/o"]).gpu_memory_utilization is None
    with pytest.raises(SystemExit):  # utilization needs the profiled allocation
        setup.parse(["--output", "/o", "--gpu-memory-utilization", "0.96"])
    with pytest.raises(SystemExit):
        setup.parse(["--output", "/o", "--kv-cache-gib", "0", "--gpu-memory-utilization", "1.2"])


def test_setup_stages_the_fork_only_with_a_utilization(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from arc_infra import setup
    from arc_infra.serving_fork import GPU_MEMORY_UTILIZATION_ENV, fork_identity

    calls = []
    monkeypatch.setattr(setup, "load_recipe", lambda output, root, **kw: calls.append((output, kw)) or "recipe")
    monkeypatch.setattr(setup, "stage_fork_bundle", lambda root, staging: staging)
    monkeypatch.delenv(GPU_MEMORY_UTILIZATION_ENV, raising=False)
    base = dict(output=str(tmp_path), mtp_tokens=3, generation_slots=8, prefix_caching=True)
    setup.recipe_for(SimpleNamespace(**base, kv_cache_gib=5, gpu_memory_utilization=None), tmp_path)
    assert "bundle_dir" not in calls[-1][1] and calls[-1][1]["kv_cache_memory_bytes"] == 5 * 1024**3
    assert GPU_MEMORY_UTILIZATION_ENV not in os.environ
    setup.recipe_for(SimpleNamespace(**base, kv_cache_gib=0, gpu_memory_utilization="0.96"), tmp_path)
    options = calls[-1][1]
    assert options["bundle_dir"] == tmp_path / "fork-bundle" and options["kv_cache_memory_bytes"] == 0
    assert options["setup_sha256"] == fork_identity()["fork_sha256"] and options["prefix_caching"] is True
    assert os.environ[GPU_MEMORY_UTILIZATION_ENV] == "0.96"
    monkeypatch.delenv(GPU_MEMORY_UTILIZATION_ENV, raising=False)


async def test_shared_server_drains_and_metrics_use_flash_endpoint(monkeypatch, tmp_path):
    original = httpx.AsyncClient
    requests = []

    def reply(request):
        requests.append(str(request.url))
        pending = int(len(requests) == 1)
        return httpx.Response(
            200,
            text=(
                f"vllm:num_requests_running {pending}\n"
                "vllm:num_requests_waiting 0\n"
                "vllm:generation_tokens_total 120\n"
            ),
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(reply), **kw)
    )
    await wait_idle(BASE_URL)
    monitor = Monitor(tmp_path, BASE_URL)
    await monitor.sample()
    assert len(requests) == 3 and set(requests) == {"http://127.0.0.1:1234/metrics"}
    assert monitor.samples[0]["metrics"]["vllm:generation_tokens_total"] == 120


async def test_gpu_memory_sampler_records_units_and_unavailable_values(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec

    async def fake_smi(*args, **kwargs):
        assert args[0] == "nvidia-smi" and "memory.free" in args[1]
        return await create(
            sys.executable, "-c", "print('0, 97887, 91000, 6887, 97, [N/A]')", **kwargs
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_smi)
    monitor = GpuMonitor(tmp_path, BASE_URL)
    await monitor.sample_gpu()
    sample = json.loads((tmp_path / "vllm-gpu-memory.jsonl").read_text())
    assert sample["gpus"][0]["memory.free"] == 6887
    assert sample["gpus"][0]["power.draw"] is None
    assert sample["units"]["memory"] == "MiB"
    with pytest.raises(ValueError, match="field count"):
        parse_gpu_rows("0, 97887")


async def test_cancelled_gpu_sampler_reaps_its_subprocess(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    ready = asyncio.Event()
    processes = []

    async def sleeping_smi(*args, **kwargs):
        process = await create(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        processes.append(process)
        ready.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", sleeping_smi)
    task = asyncio.create_task(GpuMonitor(tmp_path, BASE_URL).sample_gpu())
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None


async def test_cpu_snapshot_is_shared_capped_and_off_the_loop(monkeypatch):
    import threading

    from arc_infra import metrics

    calls = []

    def fake_snapshot():
        calls.append(threading.current_thread() is threading.main_thread())
        return {"epoch": time.time()}

    monkeypatch.setattr(metrics, "cpu_snapshot", fake_snapshot)
    monkeypatch.setattr(metrics, "_cpu_shared", {"row": None, "task": None})
    rows = await asyncio.gather(*(metrics.shared_cpu_snapshot(60) for _ in range(5)))
    assert calls == [False] and all(r is rows[0] for r in rows)  # one walk, in a worker thread, for every monitor
    assert await metrics.shared_cpu_snapshot(60) is rows[0] and len(calls) == 1  # inside the cap: the cached row
    assert await metrics.shared_cpu_snapshot(0) is None  # off
