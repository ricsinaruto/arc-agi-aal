"""ARC expert re-quantization (28 September; docs/validation-20260927/expert-quant.md): the capture overlay, the
capture runtime and trial plumbing, the NVFP4/GPTQ arithmetic, the input split rules, the study build and the
Modal budgets. CPU only; the torch parts are skipped when torch is not installed."""

import ast
import asyncio
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

from arc_infra import expert_capture_patch as capture_patch
from arena import expert_quant as eq

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path("./runs/expert-quant/inputs-v1")


# ----------------------------------------------------------------------------- overlay
def synthetic_runner(tmp_path, monkeypatch):
    site = tmp_path / "site"
    target = site / capture_patch.RUNNER["relative"]
    target.parent.mkdir(parents=True)
    text = "def f(self, hidden_states, router_logits, input_ids):\n    if True:\n" + capture_patch.ANCHOR + \
        "            fused_out = None\n"
    target.write_text(text)
    monkeypatch.setitem(capture_patch.RUNNER, "sha256", hashlib.sha256(text.encode()).hexdigest())
    return site, target


def test_overlay_patches_once_and_accepts_its_own_restart(tmp_path, monkeypatch):
    site, target = synthetic_runner(tmp_path, monkeypatch)
    identity = capture_patch.apply(site, tmp_path / "out")
    patched = target.read_text()
    assert patched.count("_arc_on_routed(self, hidden_states, topk_weights, topk_ids)") == 1
    assert (site / capture_patch.HELPER).read_bytes() == (ROOT / "src/arc_infra/expert_capture_runtime.py").read_bytes()
    assert capture_patch.apply(site, tmp_path / "out") == identity  # restart in the same working directory
    target.write_text(patched + "# edited\n")
    with pytest.raises(RuntimeError, match="modified"):
        capture_patch.apply(site, tmp_path / "out")


def test_overlay_rejects_unknown_file(tmp_path, monkeypatch):
    site, target = synthetic_runner(tmp_path, monkeypatch)
    target.write_text("something else\n")
    with pytest.raises(RuntimeError):
        capture_patch.apply(site, tmp_path / "out")


def test_overlay_anchor_is_in_the_pinned_runner():
    import os

    site = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
    path = site / capture_patch.RUNNER["relative"]
    if not path.is_file():
        pytest.skip("pinned runtime not extracted")
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != capture_patch.RUNNER["sha256"]:
        pytest.skip("a different runner file")
    assert b"arc_expert_capture" in capture_patch.patched_source(original)


# ----------------------------------------------------------------------------- capture runtime (torch)
def load_runtime(monkeypatch):
    pytest.importorskip("torch")
    worker_base = types.ModuleType("vllm.v1.worker.worker_base")

    class WorkerWrapperBase:
        pass

    worker_base.WorkerWrapperBase = WorkerWrapperBase
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker_base)
    spec = importlib.util.spec_from_file_location("arc_expert_capture_test",
                                                  ROOT / "src/arc_infra/expert_capture_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, WorkerWrapperBase


class Runner:
    def __init__(self, layer):
        self.layer_id = layer


def test_capture_runtime_strides_formats_and_caps(tmp_path, monkeypatch):
    rt, wrapper = load_runtime(monkeypatch)
    import numpy as np
    import torch

    runners = [Runner(0), Runner(5)]
    gen = torch.Generator().manual_seed(0)

    def forward(n):
        x = torch.randn(n, 8, generator=gen).to(torch.bfloat16)
        ids = torch.randint(0, 512, (n, 10), generator=gen, dtype=torch.int32)
        w = torch.rand(n, 10, generator=gen)
        for r in runners:
            rt.on_routed(r, x, w, ids)
        return x, ids, w

    forward(3)  # before any start: nothing recorded, RPC installed
    assert wrapper.arc_expert_capture is rt.arc_expert_capture
    reply = rt.arc_expert_capture(None, json.dumps({"op": "start", "dir": str(tmp_path / "a"), "layers": [5],
                                                    "stride": 2, "flush_rows": 4}))
    assert reply["ok"] and reply["active"]
    batches = [forward(n) for n in (5, 4, 7)]
    reply = rt.arc_expert_capture(None, json.dumps({"op": "stop"}))
    assert reply["rows"] == {"5": 8} and not reply["active"]  # rows 0,2,4 | 6,8 | 10,12,14 of 16
    x_all = torch.cat([b[0] for b in batches])[::2]
    ids_all = torch.cat([b[1] for b in batches])[::2]
    got = np.fromfile(tmp_path / "a/L05.x", dtype=np.int16).reshape(-1, 8)
    assert torch.equal(torch.from_numpy(got).view(torch.bfloat16), x_all)
    got_ids = np.fromfile(tmp_path / "a/L05.ids", dtype=np.int16).reshape(-1, 10)
    assert (got_ids == ids_all.numpy()).all()
    meta = json.loads((tmp_path / "a/meta.json").read_text())
    assert meta["rows"] == {"5": 8} and meta["shapes"]["5"]["hidden"] == 8 and not (tmp_path / "a/L00.x").exists()
    # fp8 with a per-row scale, all layers, capped rows
    rt.arc_expert_capture(None, json.dumps({"op": "start", "dir": str(tmp_path / "b"), "layers": None,
                                            "format": "fp8", "max_rows": 6}))
    x, _, _ = forward(10)
    reply = rt.arc_expert_capture(None, {"op": "stop"})
    assert reply["rows"] == {"0": 6, "5": 6}
    q = np.fromfile(tmp_path / "b/L00.x", dtype=np.uint8).reshape(6, 8)
    s = np.fromfile(tmp_path / "b/L00.s", dtype=np.float32)
    back = torch.from_numpy(q).view(torch.float8_e4m3fn).float() * torch.from_numpy(s).unsqueeze(1)
    assert torch.allclose(back, x[:6].float(), rtol=0.07, atol=1e-6)
    assert rt.arc_expert_capture(None, {"op": "bogus"})["ok"] is False


# ----------------------------------------------------------------------------- trial plumbing
class CaptureServer:
    """/collective_rpc into the real runtime; /completions feeds two layers' rows per prompt token."""

    def __init__(self, rt):
        import torch

        self.rt, self.torch, self.prompts = rt, torch, []

    def handler(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith("/collective_rpc"):
            assert body["method"] == "arc_expert_capture"
            return httpx.Response(200, json={"results": [self.rt.arc_expert_capture(None, body["args"][0])]})
        if request.url.path.endswith("/completions"):
            torch = self.torch
            n = len(body["prompt"])
            self.prompts.append(n)
            x = torch.ones(n, 4, dtype=torch.bfloat16)
            ids = torch.zeros(n, 10, dtype=torch.int32)
            for layer in (1, 2):
                self.rt.on_routed(Runner(layer), x, torch.ones(n, 10), ids)
            return httpx.Response(200, json={"usage": {"prompt_tokens": n}, "choices": [{}]})
        raise AssertionError(request.url.path)


def test_capture_pass_through_http(tmp_path, monkeypatch):
    rt, _ = load_runtime(monkeypatch)
    rt.on_routed(Runner(1), *[__import__("torch").zeros(1, 4)] * 2, __import__("torch").zeros(1, 10))  # install
    server = CaptureServer(rt)
    client = eq.Client("http://x/v1", "m", transport=httpx.MockTransport(server.handler))
    rows = [{"index": i, "ids": list(range(n))} for i, n in enumerate((5, 7, 4))]

    async def go():
        try:
            return await eq.capture(client, rows, directory=tmp_path / "c", layers=[2], stride=3, fmt="bf16",
                                    concurrency=2)
        finally:
            await client.close()

    summary = asyncio.run(go())
    assert summary["requests"] == 3 and summary["prompt_tokens"] == 16 and not summary["failed"]
    assert summary["rows"] == {"2": 6}  # 16 rows, every 3rd
    assert json.loads((tmp_path / "c/meta.json").read_text())["rows"] == {"2": 6}


def test_server_env_and_model_dir(tmp_path, monkeypatch):
    identity = {"pid": 4242, "argv": ["/py", "-m", "vllm.entrypoints.cli.main", "serve", "/models/m", "--x"]}
    (tmp_path / "vllm-server-identity.json").write_text(json.dumps(identity))
    real = Path.read_bytes

    def fake_read_bytes(self):
        if str(self) == "/proc/4242/environ":
            return b"PYTHONPATH=/rt/site\0LD_LIBRARY_PATH=/rt/lib\0VLLM_PORT=9\0"
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    env, got = eq.server_process_env(tmp_path)
    assert env == {"PYTHONPATH": "/rt/site", "LD_LIBRARY_PATH": "/rt/lib"} and got["pid"] == 4242
    assert eq.model_dir_of(identity) == "/models/m"


# ----------------------------------------------------------------------------- arithmetic (torch)
def test_rtn_matches_posttrain_requant_and_gptq_identity_is_rtn():
    torch = pytest.importorskip("torch")
    from arena import expert_quant_math as qm
    from posttrain import requant

    gen = torch.Generator().manual_seed(1)
    w = (torch.randn(3, 32, 64, generator=gen) * 0.03).to(torch.bfloat16)
    s2 = qm.scale2_of(w)
    packed, scales = qm.rtn(w, s2)
    for e in range(3):
        ref_s2 = requant.nvfp4_scale2([w[e]])
        ref_p, ref_s = requant.nvfp4_quantize(w[e], ref_s2)
        assert torch.equal(ref_p, packed[e]) and torch.equal(ref_s.view(torch.uint8), scales[e].view(torch.uint8))
        assert torch.equal(requant.dequantize_nvfp4(ref_p, ref_s, ref_s2), qm.dequantize(packed, scales, s2)[e])
    # shared gate/up scale_2: the max of the two amaxes
    gate, up = w[:, :16], w[:, 16:] * 2
    assert torch.equal(qm.scale2_of(gate, up), qm.scale2_of(torch.cat([gate, up], 1)))
    eye = torch.eye(64).expand(3, 64, 64).clone()
    gp, gs = qm.gptq(w, qm.inverse_cholesky_upper(eye), s2)
    assert torch.equal(gp, packed) and torch.equal(gs.view(torch.uint8), scales.view(torch.uint8))
    rp, rs = qm.rtn_search(w, s2, (1.0,))
    assert torch.equal(rp, packed) and torch.equal(rs.view(torch.uint8), scales.view(torch.uint8))
    sp, ss = qm.rtn_search(w, s2)
    assert qm.relative_error(qm.dequantize(sp, ss, s2), w) <= qm.relative_error(qm.dequantize(packed, scales, s2), w)


def test_gptq_lowers_output_error_on_correlated_inputs():
    torch = pytest.importorskip("torch")
    from arena import expert_quant_math as qm

    gen = torch.Generator().manual_seed(2)
    experts, n, k, tokens = 2, 48, 128, 3000
    w = (torch.randn(experts, n, k, generator=gen) * 0.02).to(torch.bfloat16)
    mix = torch.randn(k, k, generator=gen) * 0.4 + torch.eye(k)
    x = (torch.randn(tokens, k, generator=gen) @ mix).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, generator=gen)[:1] for _ in range(tokens)])
    weights = torch.rand(tokens, 1, generator=gen)
    hess, counts = qm.hessians(x, ids, weights, range(experts), experts)
    assert counts.sum() == tokens
    sel = (ids[:, 0] == 0)
    expected = (x[sel].float() * weights[sel]).t() @ (x[sel].float() * weights[sel])
    assert torch.allclose(hess[0], expected, rtol=1e-4, atol=1e-3)
    s2 = qm.scale2_of(w)
    qm.damp_hessian(hess, counts)
    gp, gs = qm.gptq(w, qm.inverse_cholesky_upper(hess), s2)
    rp, rs = qm.rtn(w, s2)
    for e in range(experts):
        rows = x[ids[:, 0] == e].float()
        ref = rows @ w[e].float().t()
        err_g = qm.relative_error(rows @ qm.dequantize(gp, gs, s2)[e].t(), ref)
        err_r = qm.relative_error(rows @ qm.dequantize(rp, rs, s2)[e].t(), ref)
        assert err_g < 0.85 * err_r


def test_damp_handles_empty_and_dead_inputs():
    torch = pytest.importorskip("torch")
    from arena import expert_quant_math as qm

    hess = torch.zeros(2, 4, 4)
    hess[1] = torch.diag(torch.tensor([2.0, 0.0, 2.0, 2.0]))
    qm.damp_hessian(hess, torch.tensor([0.0, 5.0]), percdamp=0.1)
    assert torch.allclose(hess[0], torch.eye(4) * 1.1)
    assert hess[1, 1, 1] > 0
    qm.inverse_cholesky_upper(hess)


# ----------------------------------------------------------------------------- inputs
def load_inputs_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("expert_quant_inputs", ROOT / "scripts/expert_quant_inputs.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_split_rules_exclude_holdout_and_separate_eval_games():
    from posttrain.split import DEV, HOLDOUT

    mod = load_inputs_module()
    assert not set(mod.EVAL_FAMILIES) & set(HOLDOUT) and set(mod.EVAL_FAMILIES) <= set(DEV)
    assert all(not mod.eligible(g) for g in HOLDOUT)
    assert all(mod.eligible(g) for g in DEV) and not mod.eligible("ft09") and not mod.eligible("q006")
    assert {mod.split_of(g) for g in mod.EVAL_FAMILIES} == {"eval"}
    assert {mod.split_of(g) for g in DEV if g not in mod.EVAL_FAMILIES} == {"calib"}


def test_built_inputs_respect_the_split():
    from posttrain.split import HOLDOUT

    if not (INPUTS / "selection.json").is_file():
        pytest.skip("inputs not built here")
    selection = json.loads((INPUTS / "selection.json").read_text())
    for name, digest in selection["sha256"].items():
        assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == digest
    calib = selection["calib"]["families"] + selection["stage1"]["calib"]["families"]
    evals = selection["eval"]["families"] + selection["stage1"]["eval"]["families"]
    assert not set(calib) & set(evals) and not (set(calib) | set(evals)) & set(HOLDOUT)
    assert set(evals) <= set(selection["eval_families"])
    stage1 = {(r["run"], r["game"]) for r in json.loads((INPUTS / "stage1.json").read_text())}
    stage2 = {(r["run"], r["game"]) for name in ("calib.json", "eval.json")
              for r in json.loads((INPUTS / name).read_text())}
    assert not stage1 & stage2


# ----------------------------------------------------------------------------- build and Modal
def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_eq", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_inputs(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir(parents=True)
    (folder / "stage1.json").write_text(json.dumps([{"index": 0, "split": "calib", "ids": [1, 2]}]))
    (folder / "selection.json").write_text(json.dumps({"eval_families": ["cn04"], "holdout_excluded": ["sb26"]}))
    return folder


def test_bench_study_build(tmp_path):
    builder = load_builder()
    output, study = builder.build("eq", output=tmp_path / "eq", paired_30m=True, expert_quant=fake_inputs(tmp_path),
                                  expert_quant_phase="bench")
    assert study["trials"] == [{"name": "expert-quant", "kind": "expert-quant", "phase": "bench",
                                "competition": False, "wall_seconds": 60}]
    assert study["server"]["expert_capture"] is True and study["server"]["mtp_tokens"] == 0
    assert "dense_kernels" not in study["server"] and "serving_kernels" not in study["server"]
    settings = study["expert_quant"]
    assert settings["layers"][0] == 20 and len(settings["layers"]) == 6 and settings["bf16_dir"].startswith("/assets/")
    assert (output / "configs/expert-quant/stage1.json").is_file()
    assert study["notebook_seconds"] == 1200 - 60
    with pytest.raises(ValueError, match="plain Flash"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, expert_quant=fake_inputs(tmp_path / "x"),
                      arena_variant="reclaim-fixed-2x")


def test_calibrate_and_evaluate_study_builds(tmp_path):
    builder = load_builder()
    inputs = fake_inputs(tmp_path)
    for name in ("calib.json", "eval.json"):
        (inputs / name).write_text("[]")
    (inputs / "quality").mkdir()
    (inputs / "quality/prompts.json").write_text(json.dumps({"teacher_forced": [], "greedy": []}))
    _, study = builder.build("cal", output=tmp_path / "cal", paired_30m=True, expert_quant=inputs,
                             expert_quant_phase="calibrate")
    settings = study["expert_quant"]
    assert study["trials"][0]["phase"] == "calibrate" and settings["overlay_dir"].startswith("/assets/flash-input/")
    assert settings["calib_stride"] >= 1 and study["notebook_seconds"] == 3300 - 60
    candidates = {"eq-w4a16": {"ARENA_SERVING_KERNELS": json.dumps({"moe_backend": "marlin"})}}
    output, study = builder.build("ev", output=tmp_path / "ev", paired_30m=True, expert_quant=inputs,
                                  expert_quant_phase="evaluate", expert_quant_candidates=candidates)
    ref, gate = study["trials"]
    assert ref["role"] == "reference" and gate == {**gate, "role": "gate", "reference": "eq-ref"}
    assert gate["server_env"]["ARENA_KV_CACHE_MEMORY_BYTES"] == ref["server_env"]["ARENA_KV_CACHE_MEMORY_BYTES"]
    assert gate["quality"]["report_only"] and "expert_capture" not in study["server"]
    assert (output / "configs/kernel-quality/gsm8k.jsonl").is_file()
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validation_timeout")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    assert study["notebook_seconds"] <= ns["validation_timeout"]("arena", "expert-quant-eval") - 60
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    stage2 = sum((ns["validation_timeout"]("arena", t) + 60) * rate * 1.15
                 for t in ("expert-quant-calibrate", "expert-quant-eval"))
    assert stage2 <= 20.0  # the user's stage-2 bound, at the reserved (maximum) durations


def test_modal_bench_budget():
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "expert-quant-bench")
    assert ns["arena_arguments"]("expert-quant-bench") == ["--trial", "expert-quant"]
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 2.0  # the user's stage-1 bound


def test_bf16_fetch_helpers():
    spec = importlib.util.spec_from_file_location("modal_expert_bf16_test", ROOT / "scripts/modal_expert_bf16.py")
    source = (ROOT / "scripts/modal_expert_bf16.py").read_text()
    tree = ast.parse(source)
    ns = {}
    for name in ("layer_files", "file_specs", "storage_gib"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(spec.origin), "exec"), ns)
    weight_map = {f"model.language_model.layers.{i}.mlp.experts.{p}": f"s{2 * i + (p == 'down_proj')}"
                  for i in range(3) for p in ("gate_up_proj", "down_proj")}
    assert ns["layer_files"](weight_map, [2, 0]) == ["s0", "s1", "s4", "s5"]
    specs = ns["file_specs"]([{"path": "s0", "size": 2**30, "lfs": {"oid": "ab"}}, {"path": "c", "size": 5}],
                             ["s0", "c"])
    assert specs == {"s0": {"bytes": 2**30, "sha256": "ab"}, "c": {"bytes": 5, "sha256": None}}
    assert ns["storage_gib"](specs) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------------- stage-1 bench on CPU (torch)
def tiny_checkpoint(tmp_path, monkeypatch, *, layer=3, experts=4, hidden=64, inter=32, topk=2, rows=40):
    """A synthetic served checkpoint (ModelOpt RTN of random BF16 experts), its BF16 source and a capture."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.torch import save_file

    from arena import expert_quant_bench as bench
    from arena import expert_quant_math as qm

    for name, value in (("E", experts), ("H", hidden), ("INTER", inter), ("TOPK", topk)):
        monkeypatch.setattr(bench, name, value)
    gen = torch.Generator().manual_seed(3)
    gate_up = (torch.randn(experts, 2 * inter, hidden, generator=gen) * 0.05).to(torch.bfloat16)
    down = (torch.randn(experts, hidden, inter, generator=gen) * 0.05).to(torch.bfloat16)
    router = torch.randn(experts, hidden, generator=gen).to(torch.bfloat16)
    s13 = qm.scale2_of(gate_up[:, :inter], gate_up[:, inter:])
    s2 = qm.scale2_of(down)
    p13, f13 = qm.rtn(gate_up, s13)
    p2, f2 = qm.rtn(down, s2)
    served = {f"model.language_model.layers.{layer}.mlp.gate.weight": router.clone()}
    base = f"model.language_model.layers.{layer}.mlp.experts."
    for e in range(experts):
        for proj, part, pk, sf, s in (("gate_proj", slice(0, inter), p13, f13, s13),
                                      ("up_proj", slice(inter, 2 * inter), p13, f13, s13),
                                      ("down_proj", slice(None), p2, f2, s2)):
            served[f"{base}{e}.{proj}.weight"] = pk[e, part].clone()
            served[f"{base}{e}.{proj}.weight_scale"] = sf[e, part].clone()
            served[f"{base}{e}.{proj}.weight_scale_2"] = s[e].clone().reshape(())
            served[f"{base}{e}.{proj}.input_scale"] = torch.tensor(0.01 if proj == "down_proj" else 0.002)
    model = tmp_path / "model"
    model.mkdir()
    save_file(served, str(model / "layer.safetensors"))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "layer.safetensors"
                                                                                    for k in served}}))
    bf16 = tmp_path / "bf16"
    bf16.mkdir()
    prefix = f"model.language_model.layers.{layer}.mlp."
    source = {prefix + "experts.gate_up_proj": gate_up, prefix + "experts.down_proj": down,
              prefix + "gate.weight": router}
    save_file(source, str(bf16 / "a.safetensors"))
    (bf16 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "a.safetensors" for k in source}}))
    capture = tmp_path / "capture"
    mix = torch.randn(hidden, hidden, generator=gen) * 0.5 + torch.eye(hidden)  # correlated inputs
    for split, fmt in (("calib", "bf16"), ("eval", "bf16"), ("calib8", "fp8")):
        folder = capture / split
        folder.mkdir(parents=True)
        n = rows * (8 if split == "calib8" else 1)
        x = (torch.randn(n, hidden, generator=gen) @ mix).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(experts, generator=gen)[:topk] for _ in range(n)]).to(torch.int16)
        w = torch.softmax(torch.randn(n, topk, generator=gen), -1)
        if fmt == "fp8":
            scale = x.float().abs().amax(1).clamp(min=1e-12) / 448.0
            (x.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tofile(
                folder / f"L{layer:02d}.x")
            scale.numpy().astype(np.float32).tofile(folder / f"L{layer:02d}.s")
        else:
            x.view(torch.int16).numpy().tofile(folder / f"L{layer:02d}.x")
        ids.numpy().tofile(folder / f"L{layer:02d}.ids")
        w.numpy().astype(np.float32).tofile(folder / f"L{layer:02d}.w")
        (folder / "meta.json").write_text(json.dumps({"rows": {str(layer): n}}))
    return bench, model, bf16, capture


def test_gptq_program_on_cpu_writes_drop_in_shards(tmp_path, monkeypatch):
    bench, model, bf16, capture = tiny_checkpoint(tmp_path, monkeypatch, rows=64)
    from safetensors import safe_open

    from arena import expert_overlay
    from arena import expert_quant_gptq as gq

    overlay = tmp_path / "overlay"
    out = tmp_path / "gptq.json"
    assert gq.main(["--calib", str(capture / "calib8"), "--capture", str(capture), "--model-dir", str(model),
                    "--bf16-dir", str(bf16), "--overlay-out", str(overlay), "--out", str(out), "--layers", "3",
                    "--deadline-seconds", "600", "--eval-rows", "64", "--min-rows", "8", "--batch", "2",
                    "--blocksize", "16", "--device", "cpu"]) == 0
    result = json.loads(out.read_text())
    layer = result["layers"]["3"]
    assert layer["experts_gptq"] == 4 and layer["scale2_modelopt_equal"] == {"gate_up": 1.0, "down": 1.0}
    errors = layer["errors"]
    assert errors["gptq_w4a16"]["vs_bf16"] < errors["checkpoint_w4a16"]["vs_bf16"]
    assert "error" in errors["gptq_w4a4_cutlass"]  # no FlashInfer on CPU
    # drop-in: same names, dtypes, shapes and byte size as the base shard; scales and input scales unchanged
    base, new = model / "layer.safetensors", overlay / "layer.safetensors"
    assert new.stat().st_size == base.stat().st_size
    with safe_open(str(base), "pt") as a, safe_open(str(new), "pt") as b:
        assert set(a.keys()) == set(b.keys())
        changed = 0
        for name in a.keys():
            x, y = a.get_tensor(name), b.get_tensor(name)
            assert x.dtype == y.dtype and x.shape == y.shape
            if name.endswith(("input_scale", "weight_scale_2")):
                assert x.view(-1).tolist() == y.view(-1).tolist()
            elif name.endswith(".weight"):
                changed += int(not (x == y).all())
        assert changed > 0
    manifest = json.loads((overlay / "overlay-manifest.json").read_text())
    assert manifest["shards"]["layer.safetensors"]["sha256"] == expert_overlay.sha256_file(new)
    assert manifest["complete"] is False  # one layer of 48
    # an incomplete overlay is refused by the serving hook; a complete one becomes a symlink farm
    digest = hashlib.sha256((overlay / "overlay-manifest.json").read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="incomplete"):
        expert_overlay.materialise(model, overlay, tmp_path / "farm", digest)
    manifest["complete"] = True
    (overlay / "overlay-manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((overlay / "overlay-manifest.json").read_bytes()).hexdigest()
    record = expert_overlay.materialise(model, overlay, tmp_path / "farm", digest, log=lambda *_: None)
    farm = tmp_path / "farm"
    assert (farm / "layer.safetensors").resolve() == new.resolve()
    assert (farm / "model.safetensors.index.json").resolve() == (model / "model.safetensors.index.json").resolve()
    assert record["shards"] == 1
    with pytest.raises(RuntimeError, match="sha256"):
        expert_overlay.materialise(model, overlay, tmp_path / "farm2", "0" * 64)


def test_bench_layer_on_cpu(tmp_path, monkeypatch):
    bench, model, bf16, capture = tiny_checkpoint(tmp_path, monkeypatch)
    out = tmp_path / "bench.json"
    assert bench.main(["--capture", str(capture), "--model-dir", str(model), "--bf16-dir", str(bf16), "--layers", "3",
                       "--out", str(out), "--deadline-seconds", "600", "--eval-rows", "32", "--rtn-search",
                       "--device", "cpu"]) == 0
    result = json.loads(out.read_text())["layers"]["3"]
    assert result["router_identical"] is True
    assert all(v == 1.0 for v in result["rtn_reproduction"].values())
    errors = result["errors"]
    assert 0.03 < errors["dequant_bf16act"]["vs_bf16"] < 0.3 and errors["dequant_bf16act"]["vs_dequant"] == 0.0
    assert errors["rtn_scale_search_bf16act"]["vs_bf16"] <= errors["dequant_bf16act"]["vs_bf16"] + 1e-6
    assert result["activation_scales"]["checkpoint_a13"] == pytest.approx(0.002)
    # no vLLM/FlashInfer here: the kernels are recorded as errors, the rest completes
    assert any("cutlass" in e for e in json.loads(out.read_text())["errors"])


# ----------------------------------------------------------------------------- serving hooks
def test_serving_capture_and_expert_overlay_hooks(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from arena import expert_overlay, serving

    applied = []
    monkeypatch.setattr(serving, "apply", lambda *a: applied.append("layout"))
    monkeypatch.setattr(serving, "apply_expert_capture", lambda site, working: applied.append("capture"))

    def recipe():
        return NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["python", "serve", str(base), "--x"],
                  RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")

    base = tmp_path / "model"
    base.mkdir()
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a.w": "s1.safetensors"}}))
    (base / "s1.safetensors").write_bytes(b"base")
    (base / "config.json").write_text("{}")
    overlay = tmp_path / "input/datasets/owner/ov"
    overlay.mkdir(parents=True)
    (overlay / "s1.safetensors").write_bytes(b"newb")
    manifest = {"format": expert_overlay.FORMAT, "complete": True,
                "shards": {"s1.safetensors": {"bytes": 4, "sha256": hashlib.sha256(b"newb").hexdigest()}}}
    (overlay / expert_overlay.MANIFEST).write_text(json.dumps(manifest))
    digest = hashlib.sha256((overlay / expert_overlay.MANIFEST).read_bytes()).hexdigest()
    plain = recipe()
    receipt = serving.configure(plain, tmp_path / "w1")
    plain.verify_and_extract_runtime()
    assert applied == ["layout"] and plain.server_command()[2] == str(base) and "expert_overlay" not in receipt
    applied.clear()
    spec = {"dataset": "owner/ov", "manifest_sha256": digest}
    patched = recipe()
    receipt = serving.configure(patched, tmp_path / "w2", expert_capture=True, expert_overlay=spec,
                                input_root=tmp_path / "input")
    patched.verify_and_extract_runtime()
    assert applied == ["layout", "capture"] and receipt["expert_capture"] and receipt["expert_overlay"] == spec
    argv = patched.server_command()
    farm = Path(argv[2])
    assert farm == tmp_path / "w2/expert-overlay-checkpoint" and argv[3] == "--x"
    assert (farm / "s1.safetensors").read_bytes() == b"newb" and (farm / "config.json").resolve() == base / "config.json"
    assert patched.server_command()[2] == str(farm)  # materialised once per start
    assert json.loads((tmp_path / "w2/expert-overlay.json").read_text())["shards"] == 1
    with pytest.raises(ValueError):
        serving.configure(recipe(), tmp_path / "w3", expert_overlay={"dataset": "x"})
    assert serving.lab_overrides({"ARENA_EXPERT_OVERLAY": "off"}) == {"expert_overlay": None}
    assert serving.lab_overrides({"ARENA_EXPERT_OVERLAY": json.dumps(spec)}) == {"expert_overlay": spec}
    assert "expert_overlay" not in serving.lab_overrides({})
