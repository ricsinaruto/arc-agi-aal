"""SGLang benchmark: profiles, recipe argv/env, metrics, runtime, driver and bundle (no GPU)."""

import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sglang_bench import agent_env, bench, launch, metrics, profile, report, runtime  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/sglang-metrics.txt"


# --- profiles -------------------------------------------------------------------------------


def test_profile_arithmetic_and_game_lists():
    assert (profile.for_admitted(3).mamba_slots, profile.for_admitted(3).max_total_tokens) == (18, 824_384)
    assert (profile.for_admitted(4).mamba_slots, profile.for_admitted(4).max_total_tokens) == (24, 824_384)
    assert (profile.for_admitted(6).mamba_slots, profile.for_admitted(6).max_total_tokens) == (36, 1_048_576)
    assert profile.RECIPE_DEFAULT == profile.ServerProfile(4, 24, 824_384)
    with pytest.raises(ValueError, match="align"):
        profile.ServerProfile(4, 24, 824_385).validate()
    with pytest.raises(ValueError, match="six per admitted"):
        profile.ServerProfile(4, 20, 824_384).validate()
    ids = profile.public_game_ids()
    assert profile.games_for(3, ids) == list(profile.CORE_GAMES)
    assert profile.games_for(4, ids) == [*profile.CORE_GAMES, "tn36-ef4dde99"]
    assert profile.games_for(6, ids) == [*profile.CORE_GAMES, "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb"]
    points = profile.sweep_points()
    assert [p["name"] for p in points] == ["c3", "c4", "c6"]
    assert [p["profile"]["admitted"] for p in points] == [3, 4, 6]
    assert profile.smoke_point()["games"] == ["ft09-0d8bbf25"]
    assert profile.ServerProfile.from_dict(points[2]["profile"]).max_total_tokens == 1_048_576


# --- recipe argv and env -------------------------------------------------------------------


GOLDEN = [
    "/opt/pennyroyal/.venv/bin/sglang", "serve",
    "--model-path", "/kaggle/input/models/m/1", "--load-format", "safetensors",
    "--served-model-name", "pennyroyal", "--host", "127.0.0.1", "--port", "8001", "--tp", "1",
    "--dtype", "bfloat16", "--quantization", "modelopt_fp4", "--kv-cache-dtype", "fp8_e4m3",
    "--mem-fraction-static", "0.981", "--max-total-tokens", "824384", "--warmups=structured_output",
    "--context-length", "524288", "--json-model-override-args", launch.TARGET_OVERRIDES,
    "--page-size", "64", "--max-running-requests", "4", "--sleep-on-idle",
    "--chunked-prefill-size", "4096", "--mamba-radix-cache-strategy", "extra_buffer",
    "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", "24", "--gdn-mtp-cache-mode", "none",
    "--linear-attn-decode-backend", "flashinfer", "--linear-attn-prefill-backend", "flashinfer",
    "--mamba-track-interval", "64", "--ple-offload-embedding", "--trust-remote-code",
    "--chat-template", "/opt/pennyroyal/configs/pennyroyal/templates/froggeric-v22.5.jinja",
    "--image-processor-backend", "pil", "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--enable-request-time-stats-logging", "--enable-metrics",
    "--default-chat-template-kwargs", launch.DEFAULT_CHAT_TEMPLATE_KWARGS,
    "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4", "--speculative-draft-model-quantization", "unquant",
    "--speculative-token-map", "/opt/pennyroyal/configs/pennyroyal/frspec/flash-next-64k.pt",
    "--watchdog-timeout", "1800",
]


def argv_for(spec):
    return launch.recipe_argv(spec, model_dir="/kaggle/input/models/m/1", repo_root="/opt/pennyroyal",
                              sglang_exe="/opt/pennyroyal/.venv/bin/sglang")


def test_recipe_argv_matches_the_script_minus_hicache():
    argv = argv_for(profile.RECIPE_DEFAULT)
    assert argv == GOLDEN
    assert not any(token.startswith("--hicache") or token == "--enable-hierarchical-cache" for token in argv)
    c6 = argv_for(profile.for_admitted(6))
    changed = [(a, b) for a, b in zip(argv, c6, strict=True) if a != b]
    assert changed == [("824384", "1048576"), ("4", "6"), ("24", "36")]
    native = argv_for(profile.with_overrides(profile.RECIPE_DEFAULT, context_length=262_144))
    assert "--json-model-override-args" not in native and "262144" in native


def test_recipe_env_and_diff(tmp_path):
    spec = profile.for_admitted(3, online_fp8=False)
    env = launch.recipe_env(spec, cache_base=tmp_path / "cache", cuda_home="/usr/local/cuda-13.3",
                            repo_root="/opt/pennyroyal", base={"PATH": "/usr/bin", "NIXL_STORAGE_BASE": "/nixl"})
    assert env["SGLANG_SM120_ONLINE_MXFP8"] == "false" and env["PENNY_PLE_BACKEND"] == "ram"
    assert env["PYTHONPATH"] == "/opt/pennyroyal/python" and env["CUDACXX"] == "/usr/local/cuda-13.3/bin/nvcc"
    assert env["PATH"].startswith("/opt/pennyroyal/.venv/bin:/usr/local/cuda-13.3/bin:")
    assert env["LD_LIBRARY_PATH"].split(":")[:2] == ["/usr/local/cuda-13.3/lib64", "/usr/local/nvidia/lib64"]
    for key in ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "SGLANG_MAMBA_CONV_DTYPE", "TOKENIZERS_PARALLELISM",
                "PYTORCH_CUDA_ALLOC_CONF", "SGLANG_FORWARD_UNKNOWN_TOOLS", "HF_HUB_OFFLINE", "TMPDIR"):
        assert key in env
    assert "NIXL_STORAGE_BASE" not in env and "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR" not in env
    assert (tmp_path / "cache/sglang/jit").is_dir() and (tmp_path / "cache/triton").is_dir()
    diff = launch.recipe_diff(env, argv_for(spec))
    assert len(diff["dropped_flags"]) == 9 and diff["changed"]["--host"] == ["0.0.0.0", "127.0.0.1"]
    assert diff["changed"]["CC"] == ["/usr/bin/gcc-15", "/usr/bin/gcc"]
    assert diff["kept_pins"]["json_model_override_args"] == launch.TARGET_OVERRIDES


def test_buffered_loading_changes_only_startup_arguments(tmp_path):
    spec = profile.with_overrides(profile.RECIPE_DEFAULT,
                                  weight_loader_disable_mmap=True, weight_loader_threads=2)
    argv = argv_for(spec)
    assert argv[:-3] == GOLDEN  # All gameplay and quantization options stay identical.
    assert argv[-3:-1] == ["--weight-loader-disable-mmap", "--model-loader-extra-config"]
    assert json.loads(argv[-1]) == {"enable_multithread_load": True, "num_threads": 2}
    assert profile.ServerProfile.from_dict(spec.to_dict()) == spec
    env = launch.recipe_env(spec, cache_base=tmp_path, cuda_home="/cuda", repo_root="/repo", base={})
    assert launch.recipe_diff(env, argv)["weight_loading"] == {
        "disable_mmap": True, "extra_config": {"enable_multithread_load": True, "num_threads": 2}}


@pytest.mark.parametrize("changes", [
    {"weight_loader_threads": 0}, {"weight_loader_threads": -1},
    {"weight_loader_threads": True}, {"weight_loader_threads": 2.5},
    {"weight_loader_disable_mmap": "false"},
])
def test_invalid_weight_loading_settings_rejected(changes):
    with pytest.raises(ValueError, match="weight_loader"):
        profile.with_overrides(profile.RECIPE_DEFAULT, **changes)


STARTUP_LOG = """
[2026-09-19 10:00:00] Load weight end. type=Qwen4ExpForCausalLM, dtype=torch.bfloat16
[2026-09-19 10:00:05] max_mamba_cache_size: 24
[2026-09-19 10:00:06] max_total_num_tokens=824384, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=4, context_len=524288, available_gpu_mem=7.52 GB
[2026-09-19 10:03:00] The server is fired up and ready to roll!
[2026-09-19 10:05:00] ReqTimeStats(rid=abc, input_len=20000, cached_input_len=0, output_len=800, attempts=1, type=normal): queue_duration=0.01, initial_prefill_elapsed=4.2, post_prefill_elapsed=30.5, forward_duration=34.0, entry_time=1.0
[2026-09-19 10:06:00] ReqTimeStats(rid=def, input_len=21000, cached_input_len=20000, output_len=600, attempts=1): queue_duration=0.02, initial_prefill_elapsed=0.4, post_prefill_elapsed=20.0, forward_duration=20.4, entry_time=2.0
"""


def test_parse_startup_and_request_time_stats():
    startup = launch.parse_startup(STARTUP_LOG)
    assert startup["max_total_num_tokens"] == 824_384 and startup["max_mamba_cache_size"] == 24
    assert startup["context_len"] == 524_288 and startup["available_gpu_mem_gb"] == 7.52
    assert startup["weights_loaded"] and startup["ready_banner"] and startup["jit_compile_lines"] == 0
    clamp = launch.clamp_check(profile.for_admitted(6), startup)
    assert clamp["clamped"] and clamp["fitted_tokens"] == 824_384
    rows = launch.request_time_stats(STARTUP_LOG)
    assert [r["rid"] for r in rows] == ["abc", "def"] and rows[1]["cached_input_len"] == 20000
    assert rows[0]["initial_prefill_elapsed"] == 4.2 and rows[1]["type"] is None
    cold = launch.cold_vs_warm(rows)
    assert cold["first"]["cached_share"] == 0.0 and cold["rest"]["cached_share"] == pytest.approx(20000 / 21000, abs=1e-4)


def test_loop_payload_shape():
    payload = launch.loop_payload("pennyroyal", [{"role": "user", "content": "hi"}])
    assert payload["model"] == "pennyroyal" and payload["top_k"] == 20
    assert payload["chat_template_kwargs"] == {"enable_thinking": True} and payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "python" and "max_tokens" not in payload
    capped = launch.loop_payload("pennyroyal", [], tools=False, max_tokens=5)
    assert capped["max_tokens"] == 5 and "tools" not in capped


# --- metrics --------------------------------------------------------------------------------


def test_metrics_parser_histograms_and_summary():
    parsed = metrics.parse_sglang_metrics(FIXTURE.read_text())
    assert parsed["values"]["sglang:generation_tokens_total"] == 3000.0
    assert parsed["values"]["sglang:token_usage"] == 0.42
    e2e = parsed["histograms"]["sglang:e2e_request_latency_seconds"]
    assert e2e["count"] == 10 and e2e["buckets"][-1][0] == float("inf")
    assert metrics.histogram_quantile(e2e, 0.5) == pytest.approx(1.0 + 9 * 3 / 4)
    assert metrics.histogram_quantile(e2e, 0.95) == pytest.approx(10 + 90 * 3.5 / 4)
    later = FIXTURE.read_text().replace("generation_tokens_total{model_name=\"pennyroyal\",engine_type=\"unified\"} 3000.0",
                                        "generation_tokens_total{model_name=\"pennyroyal\",engine_type=\"unified\"} 9000.0")
    samples = [
        {"elapsed_seconds": 0.0, **parsed},
        {"elapsed_seconds": 130.0, **metrics.parse_sglang_metrics(later)},
        {"elapsed_seconds": 200.0, **metrics.parse_sglang_metrics(later)},
    ]
    summary = metrics.summary(samples)
    assert summary["valid"] and summary["accepted_generation_tokens"] == 6000.0
    assert summary["accepted_generation_tokens_per_second"] == pytest.approx(30.0)
    assert summary["generation_counter_scope"] == "finished_requests"
    assert summary["finished_request_generation_tokens"] == 6000.0
    assert summary["finished_request_generation_tokens_per_second"] == pytest.approx(30.0)
    assert summary["peak_running_requests"] == 3.0 and summary["max_total_num_tokens"] == 824_384.0
    assert summary["cache_hit_rate"] == 0.333 and summary["mean_gen_throughput_busy"] == pytest.approx(410.5)
    assert summary["e2e_request_latency_seconds"] is None  # no new requests between samples
    assert "ttft_cold_window" in summary and summary["ttft_cold_window"] is None
    assert metrics.summary(samples[:1])["valid"] is False


def test_monitor_writes_telemetry(tmp_path):
    text = FIXTURE.read_text()

    def endpoint(request):
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=text)

    monitor = metrics.SglangMonitor(tmp_path, "http://127.0.0.1:8001/v1", interval=0.05)
    monitor.start(client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    import time

    time.sleep(0.2)
    summary = monitor.stop()
    assert summary["samples"] >= 2 and summary["errors"] == []
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == summary["samples"]
    assert (tmp_path / "sglang-metrics.txt").read_text() == text


# --- runtime dataset -------------------------------------------------------------------------


def make_parts(tmp_path, part_bytes=4096):
    from package_pennyroyal_runtime import write_parts

    stage = tmp_path / "stage"
    venv = stage / "opt/pennyroyal/.venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /opt/python/cpython-3.12.13/bin\nversion = 3.12.13\n")
    (venv / "bin/python").symlink_to("/opt/python/cpython-3.12.13/bin/python3.12")
    (stage / "opt/python/cpython-3.12.13/bin").mkdir(parents=True)
    (stage / "opt/python/cpython-3.12.13/bin/python3.12").write_bytes(os.urandom(20000))
    (stage / "opt/pennyroyal/python/sglang").mkdir(parents=True)
    (stage / "opt/pennyroyal/python/sglang/__init__.py").write_text("x = 1\n")
    upload = tmp_path / "upload"
    parts = write_parts(stage, upload, prefix="t", part_bytes=part_bytes, roots=("opt/pennyroyal", "opt/python"))
    manifest = {"schema": 1, "image": "img", "index_digest": "sha256:x", "roots": ["opt/pennyroyal", "opt/python"], **parts}
    (upload / runtime.MANIFEST_NAME).write_text(json.dumps(manifest))
    return stage, upload, manifest


def test_parts_split_and_reassemble(tmp_path):
    stage, upload, manifest = make_parts(tmp_path)
    assert len(manifest["parts"]) > 1 and all(p["size"] <= 4096 for p in manifest["parts"])
    stream = b"".join((upload / p["file"]).read_bytes() for p in manifest["parts"])
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(stream))) as tar:
        names = tar.getnames()
        assert "opt/pennyroyal/.venv/pyvenv.cfg" in names
        assert all(m.uid == 0 and m.uname == "root" for m in tar.getmembers())


def test_runtime_manifest_verification_and_extraction(tmp_path):
    stage, upload, manifest = make_parts(tmp_path)
    assert runtime.verify_manifest(upload)["stream_sha256"] == manifest["stream_sha256"]
    corrupt = upload / manifest["parts"][0]["file"]
    data = bytearray(corrupt.read_bytes())
    data[10] ^= 0xFF
    corrupt.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match=manifest["parts"][0]["file"]):
        runtime.verify_manifest(upload)
    # A fresh set of parts extracts into a temporary root.
    stage2, upload2, manifest2 = make_parts(tmp_path / "second")
    root = tmp_path / "root"
    record = runtime.extract(upload2, manifest2, root=root, min_free=0)
    assert (root / "opt/pennyroyal/.venv/pyvenv.cfg").is_file() and "python_version" not in record
    with pytest.raises(RuntimeError, match="already exist"):
        runtime.extract(upload2, manifest2, root=root, min_free=0)
    relocated = runtime.relocate(root)
    assert (root / "opt/pennyroyal/.venv/pyvenv.cfg").read_text().startswith(f"home = {root}/opt/python")
    assert any(path.endswith("pyvenv.cfg") for path in relocated["changed"])
    assert os.readlink(root / "opt/pennyroyal/.venv/bin/python").startswith(str(root / "opt/python"))


def test_runtime_model_dir_matches_the_serving_pin():
    from arc_infra.serving import MODEL_SOURCE

    expected = Path("/kaggle/input/models") / MODEL_SOURCE.lower()
    assert runtime.MODEL_DIR == expected


def test_runtime_check_reports_problems(tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho '{\"python\": \"3.12.13\", \"cuda_available\": false, \"nixl_imported\": true}'\n")
    fake.chmod(0o755)
    record = runtime.check({"PATH": os.environ["PATH"]}, python=fake, cuda_home=tmp_path, cc="/bin/echo")
    assert record["runtime"]["cuda_available"] is False
    assert any("CUDA unavailable" in p for p in record["problems"]) and any("nixl" in p for p in record["problems"])
    assert "rlimit_memlock" in record["host"] or "disk_root" in record["host"]


# --- agent venv ----------------------------------------------------------------------------


def test_agent_env_install_commands(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "kaggle/runtime-pins/analysis").mkdir(parents=True)
    (source / "kaggle/runtime-pins/runtime").mkdir(parents=True)
    for rel in agent_env.PIN_FILES:
        (source / "kaggle/runtime-pins" / rel).write_text(rel)
    (source / "kaggle/flash_bootstrap.py").write_text((ROOT / "kaggle/flash_bootstrap.py").read_text())
    (source / "build-manifest.json").write_text("{}")
    wheels = tmp_path / "input/arc-prolong-runtime/wheelhouse"
    (wheels / "analysis").mkdir(parents=True)
    (wheels / "runtime").mkdir(parents=True)
    for rel in agent_env.PIN_FILES:
        (wheels / rel).write_text(rel)
    monkeypatch.setattr(agent_env, "find_wheelhouse", lambda input_root: wheels)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    record = agent_env.install(source, tmp_path / "work", runner=runner)
    AGENT_REQUIREMENTS = agent_env.agent_requirements(ROOT)
    assert record["cpu_requirements"] == AGENT_REQUIREMENTS and "arc-agi==0.9.9" in AGENT_REQUIREMENTS
    assert calls[0][1:4] == ["-m", "venv", "--without-pip"]
    assert "--no-index" in calls[1] and str(wheels / "runtime") in calls[1] and calls[1][-len(AGENT_REQUIREMENTS):] == AGENT_REQUIREMENTS
    assert calls[2][-1] == "imageio==2.37.0" and str(source / "wheels") in calls[2]
    (wheels / "manifest.json").write_text("changed")
    with pytest.raises(RuntimeError, match="metadata changed"):
        agent_env.install(source, tmp_path / "work2", runner=runner)


# --- driver ----------------------------------------------------------------------------------


def test_analyzer_environment_matches_the_notebook_block():
    env = bench.analyzer_environment("http://127.0.0.1:8001/v1", "pennyroyal", 131_072)
    assert len(env) == 20 and env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "131072"
    assert env["LOCAL_ANALYZER_PROVIDER"] == "vllm" and env["LOCAL_ANALYZER_MODEL_ID"] == "pennyroyal"
    assert env["MULTIMODAL_CONTEXT"] == "current_grid" and env["LOCAL_ANALYZER_MAX_OUTPUT"] == "0"
    command = bench.public_command("/src", "/out/c3", "/games", 900, ["a", "b"], python="/py")
    assert command[:3] == ["/py", "-u", "/src/src/duck_controls/public_trial.py"]
    assert command[-6:] == ["--passes", "1", "--game", "a", "--game", "b"] and "--seconds" in command


class FakeServer:
    def __init__(self, folder):
        self.identity_path = Path(folder) / "server-identity.json"
        self.identity_path.write_text(json.dumps({"argv": ["x"], "pid": 1}))
        self.log_path = Path(folder) / "sglang-server.log"
        self.log_path.write_text(STARTUP_LOG)
        self.base_url = "http://127.0.0.1:8001/v1"
        self.profile = profile.RECIPE_DEFAULT


@pytest.mark.parametrize("harness", ["public", "duck-transfer"])
def test_bench_driver_records_points_and_survives_failures(tmp_path, monkeypatch, harness):
    stopped = []
    monkeypatch.setattr(launch, "preflight", lambda spec, **kw: {"problems": []})
    monkeypatch.setattr(launch, "recipe_env", lambda spec, **kw: {"ENV": "1"})
    monkeypatch.setattr(launch, "recipe_argv", lambda spec, **kw: ["sglang", "serve"])
    monkeypatch.setattr(launch, "start", lambda spec, working, env, argv, preflight_record=None: FakeServer(working))
    boots = iter([{"ready_seconds": 300.0, "max_total_num_tokens": 824_384}, RuntimeError("boom")])

    def wait_ready(server, **kw):
        result = next(boots)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(launch, "wait_ready", wait_ready)
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: {"gpus": []})
    monkeypatch.setattr(launch, "warmup", lambda server, probes=False: {"chat": {"status": 200}})
    monkeypatch.setattr(launch, "stop", lambda server, **kw: stopped.append(server) or {"stopped_by": "SIGTERM"})

    class FakeMonitor:
        def __init__(self, folder, base_url, interval=5):
            pass

        def start(self):
            return self

        def stop(self):
            return {"valid": True, "accepted_generation_tokens_per_second": 300.0}

    monkeypatch.setattr(bench, "SglangMonitor", FakeMonitor)

    def fake_trial(source, folder, games_dir, seconds, games, env, *, timeout=None, python=None):
        assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "131072"
        summary = {"status": "complete", "rows": [{"game_id": g, "state": "won", "final_score": 50.0,
                                                    "levels_completed": 1, "number_of_levels": 2, "actions": 3,
                                                    "generated_tokens": 100} for g in games]}
        (Path(folder) / "taaf").mkdir(parents=True, exist_ok=True)
        (Path(folder) / "taaf/summary.json").write_text(json.dumps(summary))
        return {"exit_code": 0, "summary": summary}

    monkeypatch.setattr(bench, "run_public_trial", fake_trial)
    def fake_transfer(source, folder, games_dir, seconds, games, env, **kwargs):
        assert kwargs["base_url"] == "http://127.0.0.1:8001/v1"
        assert kwargs["model"] == "pennyroyal" and kwargs["config"] == "configs/test.yaml"
        return {"exit_code": 0, "summary": {"status": "complete", "engine": {"valid": True},
                "games": [{"game_id": g, "status": "time_limit", "score": 50., "levels": 1,
                           "actions": 3, "completion_tokens": 100} for g in games]}}
    monkeypatch.setattr(bench, "run_transfer_trial", fake_transfer)
    monkeypatch.setattr(bench.Bench, "__init__", _bench_init)
    study = {"game_seconds": 10, "analyzer_context_window": 131_072, "notebook_seconds": 100_000,
             "points": profile.sweep_points()[:2]}
    if harness == "duck-transfer":
        study["agent_config"] = "configs/test.yaml"
        study["startup_timeout_seconds"] = 600
        for point in study["points"]:
            point["harness"] = harness
    driver = bench.Bench(source=tmp_path, study=study, output=tmp_path / "out", model_dir="/m",
                         notebook_start=0.0, games_dir="/games", agent_python="/py")
    import time

    driver.deadline = time.time() + 100_000
    record = driver.run()
    assert [p["status"] for p in record["points"]] == ["complete", "error"] and record["status"] == "error"
    assert record["points"][0]["public_rows"][0]["game_id"] == "ft09-0d8bbf25"
    assert record["points"][0]["metrics"]["accepted_generation_tokens_per_second"] == 300.0
    assert record["points"][0]["request_time_stats"]["rows"] == 2
    assert "boom" in record["points"][1]["error"] and len(stopped) == 2  # the failed point still stopped its server
    assert json.loads((tmp_path / "out/study.json").read_text())["points"][1]["status"] == "error"
    assert (tmp_path / "out/c4/exception.txt").is_file()
    filtered = driver.run(["c3"])
    assert [p["point"] for p in filtered["points"]][-1] == "c3"
    table = report.render(record)
    assert "| c3 | complete | 3 |" in table and "boom" in table


def _bench_init(self, *, source, study, output, model_dir, notebook_start, games_dir, agent_python):
    import time

    self.source, self.study, self.output = Path(source), study, Path(output)
    self.model_dir, self.notebook_start = Path(model_dir), notebook_start
    self.games_dir, self.agent_python = games_dir, agent_python
    self.deadline = notebook_start + study["notebook_seconds"]
    self.cache_base = self.output / "cache"
    self.output.mkdir(parents=True, exist_ok=True)
    self.record = {"study": study, "started_epoch": time.time(), "model_dir": str(model_dir), "points": [],
                   "hardware": {"gpu_name": "fake"}, "runtime_check": None}


def test_bench_skips_points_past_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.Bench, "__init__", _bench_init)
    study = {"game_seconds": 900, "analyzer_context_window": 131_072, "notebook_seconds": 10, "points": profile.sweep_points()}
    driver = bench.Bench(source=tmp_path, study=study, output=tmp_path / "out", model_dir="/m",
                         notebook_start=0.0, games_dir="/games", agent_python="/py")
    record = driver.run()
    assert all(p["status"] == "skipped" for p in record["points"])


# --- bundle ---------------------------------------------------------------------------------


def test_bundle_and_kernel_files(tmp_path):
    from build_sglang_bench import build

    output, digest = build("fixture-bench", points=["c3", "c6"], output=tmp_path / "bundle")
    study = json.loads((output / "study.json").read_text())
    assert [p["name"] for p in study["points"]] == ["c3", "c6"] and study["mode"] == "sweep"
    assert study["analyzer_context_window"] == 131_072 and study["notebook_seconds"] == 8400
    assert (output / "src/inference/agent/tool_agent.py").read_bytes() == (ROOT / "src/inference/agent/tool_agent.py").read_bytes()
    assert (output / "src/duck_controls/public_trial.py").is_file() and (output / "src/arc_infra/arcade.py").is_file()
    assert not (output / "src/arc_infra/serving_setup.py").exists() and not (output / "src/duck_transfer").exists()
    assert (output / "kaggle/flash_bootstrap.py").is_file() and list((output / "wheels").glob("imageio-*.whl"))
    assert not list(output.rglob("__pycache__"))
    notebook = json.loads((output / "experiment.ipynb").read_text())
    code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    assert digest in code and "sglang_bench.bench" in code and "runtime-check.json" in code
    kernel = json.loads((output / "kernel-metadata.json").read_text())
    assert kernel["id"] == "richardcsaky/arc-fixture-bench" and kernel["machine_shape"] == "NvidiaRtxPro6000"
    assert kernel["enable_internet"] is False
    assert kernel["dataset_sources"] == ["richardcsaky/arc-sglang-bench", "richardcsaky/arc-sglang-runtime",
                                         "richardcsaky/arc-prolong-runtime"]
    assert (output / "dataset-upload/fixture-bench.zip.blob").read_bytes() == (output / "source.zip").read_bytes()
    smoke, _ = build("fixture-smoke", smoke=True, output=tmp_path / "smoke")
    smoke_study = json.loads((smoke / "study.json").read_text())
    assert smoke_study["mode"] == "smoke" and smoke_study["points"][0]["seconds"] == 120
    assert smoke_study["points"][0]["probes"] is True and smoke_study["points"][0]["profile"]["admitted"] == 4


def test_frozen_repeat_preserves_zip_and_executes_timing_override(tmp_path):
    import ast

    from build_sglang_bench import build, prepare_repeat

    source, digest = build('frozen', duck_transfer=True, output=tmp_path / 'original')
    output, repeated_digest = prepare_repeat(source, 'repeat', output=tmp_path / 'repeat',
                                             notebook_seconds=5340, startup_seconds=2700, kernel_timeout=5400)
    assert digest == repeated_digest
    assert (source / 'source.zip').read_bytes() == (output / 'source.zip').read_bytes()
    assert (source / 'study.json').read_bytes() == (output / 'study.json').read_bytes()
    notebook = json.loads((output / 'experiment.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    # Execute the actual generated override and verify the driver's input file.
    code = notebook['cells'][1]['source']
    override = code[code.index('study = json.loads'):code.index('import threading')]
    namespace = {'SOURCE': source, 'OUTPUT': output, 'json': json}
    exec(override, namespace)
    launch_study = json.loads((output / 'launch-study.json').read_text())
    original = json.loads((source / 'study.json').read_text())
    assert {k for k in launch_study if launch_study[k] != original[k]} == {'notebook_seconds', 'startup_timeout_seconds'}
    assert launch_study['startup_timeout_seconds'] == 2700
    assert 'str(OUTPUT / "launch-study.json")' in notebook['cells'][2]['source']
    assert json.loads((output / 'kernel-metadata.json').read_text())['id'] == 'richardcsaky/arc-repeat'
    with pytest.raises(ValueError, match='reserve'):
        prepare_repeat(source, 'short', output=tmp_path / 'short', notebook_seconds=3000,
                       startup_seconds=2700, kernel_timeout=5400)
    # Change a manifest-tracked byte without changing the parsed study.
    (source / 'study.json').write_text(json.dumps(original) + ' ')
    with pytest.raises(ValueError, match='Frozen source mismatch'):
        prepare_repeat(source, 'bad', output=tmp_path / 'bad', notebook_seconds=5340,
                       startup_seconds=2700, kernel_timeout=5400)


def test_package_script_pieces(tmp_path):
    from package_pennyroyal_runtime import case_collisions, prune, verify_stage

    assert case_collisions(["a/B", "a/b", "c"]) == [("a/B", "a/b")]
    stage = tmp_path / "stage"
    repo = stage / "opt/pennyroyal"
    (repo / ".git").mkdir(parents=True)
    (repo / "test").mkdir()
    (repo / "python/__pycache__").mkdir(parents=True)
    removed = prune(stage)
    assert removed == [".git", "test"] and not (repo / "python/__pycache__").exists()
    assert (repo / "SOURCE_COMMIT.txt").is_file()
    site = repo / ".venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    for name, version in {"torch": "2.13.0+cu130", "torchvision": "0.28.0+cu130", "torchaudio": "2.11.0+cu130",
                          "flashinfer_python": "0.6.17", "flashinfer_jit_cache": "0.6.17+cu130",
                          "sglang_kernel": "0.4.6.post1+cu130", "triton": "3.7.1"}.items():
        (site / f"{name}-{version}.dist-info").mkdir()
    (site / "flashinfer_jit_cache/jit").mkdir(parents=True)
    (site / "flashinfer_jit_cache/jit/fused_moe_120.so").write_bytes(b"")
    (repo / ".venv/bin").mkdir()
    (repo / ".venv/bin/python").symlink_to("/opt/python/cpython-3.12.13/bin/python3.12")
    (repo / ".venv/pyvenv.cfg").write_text("home = /opt/python/cpython-3.12.13/bin\n")
    (site / "__editable___sglang_finder.py").write_text("MAPPING = {'sglang': '/opt/pennyroyal/python/sglang'}\n")
    (repo / "configs/pennyroyal/templates").mkdir(parents=True)
    (repo / "configs/pennyroyal/frspec").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="template"):
        verify_stage(stage, cuda=False)
    (repo / ".venv/lib/python3.12/site-packages/torch-2.13.0+cu130.dist-info").rename(site / "torch-2.12.0.dist-info")
    with pytest.raises(RuntimeError, match="torch"):
        verify_stage(stage, cuda=False)


def test_prefetch_model_reads_every_checkpoint_file(tmp_path):
    from sglang_bench import runtime

    (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "sub").mkdir()
    lines = []
    record = runtime.prefetch_model(tmp_path, threads=2, block=16, log=lines.append)
    assert record["files"] == 2 and record["bytes"] == 102 and record["seconds"] >= 0
    assert lines and lines[0].startswith("[prefetch]")


def test_link_cudart_symlinks_the_venv_runtime_into_lib64(tmp_path):
    from sglang_bench import runtime

    source = tmp_path / "opt/pennyroyal/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"so")
    record = runtime.link_cudart(tmp_path, log=lambda *_: None)
    lib64 = tmp_path / "usr/local/cuda-13.3/lib64"
    assert (lib64 / "libcudart.so").resolve() == source and (lib64 / "libcudart.so.13").is_symlink()
    assert record["links"]["libcudart.so"] == str(source)
    runtime.link_cudart(tmp_path, log=lambda *_: None)  # idempotent
    with pytest.raises(RuntimeError, match="No shared libcudart"):
        runtime.link_cudart(tmp_path / "empty", log=lambda *_: None)


def test_link_cuda_driver_stands_in_for_the_missing_stub(tmp_path):
    from sglang_bench import runtime

    driver = tmp_path / "usr/lib/x86_64-linux-gnu/libcuda.so.1"
    driver.parent.mkdir(parents=True)
    driver.write_bytes(b"so")
    record = runtime.link_cuda_driver(tmp_path, log=lambda *_: None)
    cuda = tmp_path / "usr/local/cuda-13.3"
    assert (cuda / "lib64/stubs/libcuda.so").resolve() == driver
    assert (cuda / "lib64/libcuda.so").resolve() == driver and record["source"] == str(driver)
    runtime.link_cuda_driver(tmp_path, log=lambda *_: None)  # idempotent
    assert runtime.find_cuda_driver(tmp_path / "empty") is None
    with pytest.raises(RuntimeError, match="No libcuda.so.1"):
        runtime.link_cuda_driver(tmp_path / "empty", log=lambda *_: None)


def test_request_time_stats_converts_real_server_milliseconds():
    text = ('ReqTimeStats(rid=live, input_len=10, cached_input_len=0, output_len=18, '
            'attempts=0, type=unified): queue_duration=6.21ms, initial_prefill_elapsed=12579.07ms, '
            'post_prefill_elapsed=126.27ms, forward_duration=12.70535s, entry_time=1790071574.908')
    row = launch.request_time_stats(text)[0]
    assert row['queue_duration'] == pytest.approx(0.00621)
    assert row['initial_prefill_elapsed'] == pytest.approx(12.57907)
    assert row['post_prefill_elapsed'] == pytest.approx(0.12627)
    assert row['forward_duration'] == pytest.approx(12.70535)
    assert row['entry_time'] == 1790071574.908
    assert launch.cold_vs_warm([row])['first']['mean_initial_prefill_seconds'] == 12.579


def test_transfer_bundle_settings_match_vllm_reference(tmp_path):
    import hashlib

    from build_sglang_bench import build

    from arc_infra.config import Config

    folder, _ = build("transfer-test", duck_transfer=True, output=tmp_path)
    study = json.loads((folder / "study.json").read_text())
    cfg = Config.load(str(folder / study["agent_config"]))
    assert (cfg.max_model_len, cfg.max_output_tokens, cfg.reasoning_effort, cfg.temperature) == (65536, 8192, "xhigh", .7)
    assert study["mode"] == "duck-transfer" and study["game_seconds"] == 900
    assert study["points"][0]["games"] == list(profile.CORE_GAMES)
    assert study["points"][0]["harness"] == "duck-transfer"
    assert study["startup_timeout_seconds"] + 60 + 900 + 90 + bench.DRAIN_ALLOWANCE < study["notebook_seconds"]
    for package in ("duck_transfer", "arc_infra"):
        for source in (ROOT / "src" / package).rglob("*.py"):
            target = folder / source.relative_to(ROOT)
            assert target.read_bytes() == source.read_bytes()
    manifest = json.loads((folder / "build-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest


async def test_transfer_fleet_monitor_uses_sglang_counters(tmp_path, monkeypatch):
    from sglang_bench.transfer import FleetMonitor
    original = httpx.AsyncClient
    counter = iter([100, 700])
    def handler(request):
        return httpx.Response(200, text=f'sglang:generation_tokens_total{{model_name="pennyroyal"}} {next(counter)}\n')
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    monitor = FleetMonitor(tmp_path, "http://sglang/v1")
    await monitor.sample()
    await monitor.sample()
    monitor.samples[0]["elapsed_seconds"] = 0
    monitor.samples[1]["elapsed_seconds"] = 10
    measured = monitor.summary()
    assert measured["accepted_generation_tokens"] == 600
    assert measured["accepted_generation_tokens_per_second"] == 60
    assert (tmp_path / "sglang-metrics.txt").exists()


@pytest.mark.parametrize("tokenizer_ok", [True, False])
async def test_transfer_preflight_and_fleet_contract(tmp_path, monkeypatch, tokenizer_ok):
    from types import SimpleNamespace

    from duck_transfer import loader
    from sglang_bench import transfer

    seen = []
    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/tokenize" and payload["model"] == "pennyroyal"
        assert payload["tools"][0]["function"]["name"] == "python"
        assert payload["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
        return httpx.Response(200 if tokenizer_ok else 400,
                              json={"count": 48, "max_model_len": 524288})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(loader, "configure", lambda cfg: {"profile": "test"})
    agent = object()
    monkeypatch.setitem(sys.modules, "duck_transfer.transfer", SimpleNamespace(TransferDuckAgent=agent))
    async def fake_fleet(cfg, games_dir, folder, interpreter, monitor, **kwargs):
        seen.append(cfg)
        assert cfg.gameplay_seconds == 900 and cfg.max_model_len == 65536
        assert cfg.reasoning_effort == "xhigh" and cfg.max_output_tokens == 8192
        assert cfg.base_url == "http://sglang/v1" and cfg.parallel_games == 8
        assert kwargs["games"] == list(profile.CORE_GAMES)
        assert kwargs["clock_sessions"] == 25 and kwargs["agent_class"] is agent
        assert isinstance(monitor, transfer.FleetMonitor)
        return {"failed_sessions": 0, "task_errors": []}
    monkeypatch.setattr(transfer, "run_replicas", fake_fleet)
    args = (ROOT, tmp_path, "/games", list(profile.CORE_GAMES), 900, "http://sglang/v1",
            "pennyroyal", "configs/duck_transfer_sglang.yaml")
    if tokenizer_ok:
        result = await transfer.run(*args)
        assert result["status"] == "complete" and len(seen) == 1
        assert (tmp_path / "api-preflight.json").is_file()
        assert (tmp_path / "report.json").is_file()
    else:
        with pytest.raises(httpx.HTTPStatusError):
            await transfer.run(*args)
        assert not seen
