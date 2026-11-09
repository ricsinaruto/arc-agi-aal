"""Kernel work, phase 6 (docs/validation-20260927/kernels-phase6.md): the bench's mixed decode + injected-prefill
points and mixed-step profile, the in-job microbenchmark contract, the two-arm smoke build and its Modal budget, the
phase-2 gap analysis on synthetic telemetry. CPU only (the microbenchmark's torch parts skip without torch)."""

import ast
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from test_arena_cache import layout

from arc_infra import dense_kernels_patch as dense
from arena import phase6_checks
from arena import throughput_bench as tb

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())


def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_p6", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- bench: mixed points
def test_mixed_points_and_injector_are_validated():
    spec = {"mixed_points": [{"prompt_tokens": 1000, "concurrency": 3,
                              "inject": {"prefix_tokens": 500, "tail_tokens": 100, "rate": 2}}]}
    (point,) = tb.grid_points(spec)
    assert point == {"kind": "mixed", "prompt_tokens": 1000, "concurrency": 3, "mixed_output_tokens": 2048,
                     "inject": {"prefix_tokens": 500, "tail_tokens": 100, "rate": 2.0}}
    for bad in ({"prefix_tokens": 500, "tail_tokens": 100}, {"prefix_tokens": 10, "tail_tokens": 100, "rate": 1},
                {"prefix_tokens": 500, "tail_tokens": 100, "rate": 0}, {"prefix_tokens": 500, "tail_tokens": 100,
                                                                         "rate": 1, "x": 1}):
        with pytest.raises(ValueError):
            tb.validate_inject(bad)
    # Existing grids are unchanged: no mixed points unless asked.
    assert all(p["kind"] != "mixed" for p in tb.grid_points(tb.DEFAULT_GRID))
    # The image seed is the only new request knob, and it defaults to the prefix seed.
    assert tb.request_body(5, 40) == tb.request_body(5, 40, image_seed=5)
    assert tb.request_body(5, 40, image_seed=6)["messages"][1]["content"][1] != \
        tb.request_body(5, 40)["messages"][1]["content"][1]


class SlowServer:
    """A vLLM-like server whose forced decode requests take ``decode_seconds`` (the injector needs time to fire)."""

    def __init__(self, decode_seconds=0.6):
        self.decode_seconds = decode_seconds
        self.generated = self.prompt_compute = 0.0
        self.requests = []

    @staticmethod
    def count(body):
        return max(1, len(json.dumps(body["messages"])) // 6)

    async def handler(self, request):
        path = request.url.path
        if path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": self.count(json.loads(request.content)), "max_model_len": 262144})
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.requests.append(body)
            prompt = self.count(body)
            completion = body["max_tokens"] if body.get("ignore_eos") else 1
            if body.get("ignore_eos") and completion >= 100:
                await asyncio.sleep(self.decode_seconds)
            self.generated += completion
            self.prompt_compute += prompt
            return httpx.Response(200, json={"usage": {"prompt_tokens": prompt, "completion_tokens": completion},
                                             "choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        if path.endswith("/metrics"):
            return httpx.Response(200, text=(
                f"vllm:generation_tokens_total {self.generated}\nvllm:num_preemptions_total 0.0\n"
                f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {self.prompt_compute}\n'
                'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 0.0\n'
                "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.2\n"))
        if path.endswith("/reset_prefix_cache"):
            return httpx.Response(200, json={"success": True})
        raise AssertionError(path)


async def test_mixed_point_injects_prefill_while_the_streams_decode(tmp_path, monkeypatch):
    server = SlowServer()
    transport = httpx.MockTransport(server.handler)
    real_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    monkeypatch.setattr(tb.GpuMonitor, "run", lambda self: asyncio.sleep(3600))
    (tmp_path / "server.log").write_text("worker ARC_CACHE_LAYOUT " + json.dumps(layout()) + "\n")
    study = {"kv_margin": 0.9, "generation_slots": 25, "completion_cap": 16384, "reasoning_effort": "xhigh",
             "temperature": 0.7, "server": {"mtp_tokens": 0},
             "throughput_bench": {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None,
                                  "mixed_points": [{"prompt_tokens": 400, "concurrency": 3, "output_tokens": 128,
                                                    "inject": {"prefix_tokens": 300, "tail_tokens": 80,
                                                               "rate": 5}}]}}
    report = await tb.run_bench(study, {"name": "p6", "kind": "serving-bench"}, output=tmp_path / "out",
                                server_dir=tmp_path, base_url="http://model/v1", transport=transport,
                                log=lambda *a, **k: None)
    assert report["status"] == "complete" and not report["task_errors"]
    (point,) = report["points"]
    assert [p["pass"] for p in point["passes"]] == ["prefill", "decode", "mixed"]
    mixed = point["passes"][2]
    assert mixed["completion_tokens"] == 3 * 128 and mixed["requests"] == 3  # the streams only
    injected = mixed["injected"]
    assert injected["sent"] >= 2 and injected["completed"] == injected["sent"] and injected["failed"] == 0
    assert mixed["inject"] == {"prefix_tokens": 300, "tail_tokens": 80, "rate": 5.0}
    # Injected bodies: one shared prefix (warmed once), a fresh tail and board image each, one output token.
    bodies = [b for b in server.requests if not b.get("ignore_eos") and "Latest step" in json.dumps(b)]
    assert len(bodies) == injected["sent"]
    texts = [b["messages"][1]["content"][0]["text"] for b in bodies]
    assert len({t.split("\nLatest step")[0] for t in texts}) == 1 and len(set(texts)) == len(texts)
    assert len({b["messages"][1]["content"][1]["image_url"]["url"] for b in bodies}) == len(bodies)
    assert all(b["max_tokens"] == 1 for b in bodies)


# ----------------------------------------------------------------------------- microbenchmark contract
def test_phase6_microbench_contract(tmp_path):
    assert phase6_checks.validate({}) == {"parts": ["dense", "qsa", "w4"], "budget_seconds": 400.0, "timeout": 1500}
    for bad in ({"parts": []}, {"parts": ["gpu"]}, {"parts": ["qsa", "qsa"]}, {"budget_seconds": 5},
                {"parts": ["dense", "qsa", "w4"], "budget_seconds": 600, "timeout": 1500}, {"other": 1}):
        with pytest.raises(ValueError):
            phase6_checks.validate(bad)
    (tmp_path / phase6_checks.IDENTITY).write_text(json.dumps({"argv": ["python", "-m", "vllm", "serve", "/m/ckpt",
                                                                        "--port", "1"]}))
    assert phase6_checks.model_dir(tmp_path) == "/m/ckpt"
    spec = phase6_checks.validate({"parts": ["qsa", "w4"], "budget_seconds": 100, "timeout": 600})
    command = phase6_checks.command({"python": "/py"}, tmp_path / "o.json", spec, "/m/ckpt")
    assert command[:4] == ["/py", "-u", "-m", "arc_infra.phase6_microbench"] and command[-2:] == ["--model", "/m/ckpt"]
    assert phase6_checks.microbench(tmp_path, tmp_path, spec)["status"] == "skipped"  # no kv8 runtime record
    # The module runs inside the serving runtime: nothing from the arena package, only torch/vLLM (lazily).
    tree = ast.parse((ROOT / "src/arc_infra/phase6_microbench.py").read_text())
    top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert top <= {"argparse", "json", "sys", "time", "pathlib", "torch"}


def test_w4_quantizers_and_errors():
    torch = pytest.importorskip("torch")
    from arc_infra import phase6_microbench as mb

    g = torch.Generator().manual_seed(0)
    w = torch.randn((64, 256), generator=g) * 0.02
    for group in (32, 64, 128):
        sym = mb.quantize_int4_groups(w, group, True)
        asym = mb.quantize_int4_groups(w, group, False)
        # At most 15 distinct levels per group, and asymmetric is closer (one more level, no wasted sign range).
        assert all(len(torch.unique(row)) <= 15 for row in sym.reshape(-1, group)[:8])
        assert mb.rel(asym, w) < mb.rel(sym, w) < 0.25
    assert mb.rel(mb.quantize_int8_rows(w), w) < 0.02
    result = mb.w4_errors(w.to(torch.bfloat16), rows=64)
    assert result["int8_channel"]["output_rel"] < result["int4_g32_asym"]["output_rel"] < \
        result["int4_g128_sym"]["output_rel"]
    assert result["bytes_per_weight"]["int4_g32_asym"] == pytest.approx(0.5 + 2 / 32 + 0.5 / 32)


# ----------------------------------------------------------------------------- smoke build and budget
def test_phase6_smoke_build_and_budget(tmp_path):
    from arena import serving
    from arena.variants import apply_variant

    builder = load_builder()
    reference = tmp_path / "gate.json"
    reference.write_text(json.dumps({"role": "gate", "trial": "kernel-gate", "server_env": {},
                                     "server": {"dense_kernels": dense.PHASE2}, "greedy": {}, "teacher_forced": {}}))
    output, study = builder.build("p6", output=tmp_path / "p6", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp0-wide-kv8-hero", phase6_smoke=True,
                                  kv8_reference=str(reference))
    hero = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-hero")
    arm = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-hero-p6")
    assert study["server"] == hero["server"]  # the control arm is the hero, byte for byte
    assert [t["name"] for t in study["trials"]] == ["hero", "p6"]
    assert study["trials"][0]["server_env"] == {"ARENA_TORCH_PROFILER": "1"}
    lab = serving.lab_overrides(study["trials"][1]["server_env"])
    assert lab == {"torch_profiler": True, "dense_kernels": arm["server"]["dense_kernels"],
                   "serving_kernels": arm["server"]["serving_kernels"]}
    assert lab["dense_kernels"]["fused_splitk"] is True and lab["serving_kernels"]["ple_uva"] is True
    assert study["trials"][1]["throughput_bench"] == {"phase6_microbench": builder.PHASE6_MICROBENCH}
    grid = study["throughput_bench"]
    assert {c for _, c in grid["points"]} == {1, 6, 14} and [m["concurrency"] for m in grid["mixed_points"]] == [14, 21]
    assert grid["profile"]["concurrency"] == [14, 21] and grid["profile"]["inject"]["concurrency"] == [14]
    assert (output / grid["quality"]["reference"]).is_file() and grid["quality"]["repeat_teacher"] is False
    points = tb.grid_points(grid)
    assert [p["kind"] for p in points] == ["distinct"] * 3 + ["mixed"] * 2
    # Every mixed point fits the hero's gate with its injector (860 blocks of 1,515 tokens, 774 at the gate).
    for m in grid["mixed_points"]:
        per_stream = -(-(m["prompt_tokens"] + m["output_tokens"] + 256) // 1515)
        injector = -(-(m["inject"]["prefix_tokens"] + m["inject"]["tail_tokens"] + 256) // 1515)
        assert m["concurrency"] * per_stream + injector <= 774
    with pytest.raises(ValueError, match="phase-6"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", phase6_smoke=True)
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "kernel-phase6")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 6.0  # the bound
    assert ns["arena_arguments"]("kernel-phase6") == []
    assert study["notebook_seconds"] <= timeout - 60 and study["kernel_timeout"] <= timeout
    # The p6 trial starts after ~1,250 s (container + hero setup ~500 s, its bench ~750 s) and needs wall + 480 s.
    assert study["notebook_seconds"] - 1250 >= study["wall_seconds"] + 480 + 1200  # its setup, bench, microbench


def test_phase6_part2_build_and_budget(tmp_path):
    builder = load_builder()
    output, study = builder.build("p6b", output=tmp_path / "p6b", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp0-wide-kv8-hero", phase6_smoke=True, phase6_part2=True)
    (trial,) = study["trials"]
    assert trial["name"] == "p6b" and trial["wall_seconds"] == builder.PHASE6_PART2_WALL
    from arena import serving
    from arena.variants import apply_variant

    arm = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-hero-p6")["server"]
    assert serving.lab_overrides(trial["server_env"])["dense_kernels"] == arm["dense_kernels"]
    grid = study["throughput_bench"]
    assert tb.grid_points(grid) == [] and grid["profile"]["inject"]["concurrency"] == [14, 21]
    assert grid["phase6_microbench"] == builder.PHASE6_MICROBENCH
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "kernel-phase6b")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 4.0 and ns["arena_arguments"]("kernel-phase6b") == ["--trial", "p6b"]
    # The trial's own timeout (wall + 600 s) covers the profiles and the microbenchmark's own timeout.
    assert trial["wall_seconds"] + 600 >= 300 + builder.PHASE6_MICROBENCH["timeout"]
    assert study["kernel_timeout"] <= timeout
    # arena.run starts the trial only with wall + 480 s left of the Modal notebook (timeout - 60 s); v1's container
    # setup took ~665 s (kernel-phase6b-v1 refused a 1,200 s wall by ~5 s): keep >= 200 s of margin over 700 s.
    assert (timeout - 60) - 700 - (trial["wall_seconds"] + 480) >= 200
    with pytest.raises(ValueError, match="part2"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", phase6_part2=True, hero_smoke=True,
                      kv8_reference="x")


# ----------------------------------------------------------------------------- phase-2 gap analysis
def test_phase2_gap_fit_recovers_a_synthetic_prefill_cost(tmp_path):
    spec = importlib.util.spec_from_file_location("phase2_gap", ROOT / "scripts/arena_analysis/phase2_gap.py")
    gap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gap)
    curve = [(1, 9.0), (6, 15.0), (14, 19.0)]
    assert gap.t_syn(curve, 10) == pytest.approx(17.0) and gap.t_syn(curve, 22) == pytest.approx(23.0)
    # Intervals of 5 s: decode steps at the curve plus 0.1 ms per prefilled token.
    rows, targets = [], []
    for i in range(40):
        prefill = 2000 * (i % 7)
        busy = 5.0 - prefill * 1e-4
        steps = busy / (gap.t_syn(curve, 14) / 1000)
        rows.append((prefill, i % 3, steps))
        targets.append(prefill * 1e-4)
    c_p, c_req, c_step = gap.nnls(rows, targets)
    assert c_p == pytest.approx(1e-4, rel=1e-3) and c_req == pytest.approx(0, abs=1e-6)
    assert gap.step_mean([(0.0, 2), (1.0, 4), (3.0, 0)], 0.0, 4.0) == pytest.approx((2 + 8) / 4)


def test_standalone_modal_launchers_ship_themselves():
    """include_source=False: a launcher whose module is not added to the image fails every container at import
    (qsa-prefill-bench-v1 crash-looped on ModuleNotFoundError)."""
    for name in ("modal_qsa_bench.py", "modal_kernel_bench.py", "modal_validation.py"):
        text = (ROOT / "scripts" / name).read_text()
        assert "include_source=False" in text
        assert f'.add_local_file(__file__, "/root/{name}", copy=True)' in text, name
