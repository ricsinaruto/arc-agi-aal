"""marlin_batch_patch (MFU-2 B300 boot, 24 Oct): the patched permute_scales of the pinned NVFP4 MoE Marlin prep gives
bit-identical scales and global scales with ARC_MARLIN_BATCH=1 and without, on the plain file and on the file as
expert_mask_patch leaves it; idempotent; an unknown file is skipped, never raised on."""

import ast
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from arc_infra import marlin_batch_patch as patch

torch = pytest.importorskip("torch")
PINNED = Path(__import__("os").environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
pytestmark = pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
UTILS = "vllm/model_executor/layers/quantization/utils/marlin_utils.py"


def functions(source, names, nested_in=None):
    tree = ast.parse(source)
    body = tree.body
    if nested_in:
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == nested_in).body
    found = []
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            for a in node.args.args + node.args.kwonlyargs:
                a.annotation = None
            node.returns = None
            found.append(node)
    assert {n.name for n in found} == set(names)
    return found


def permute_scales(text, monkeypatch, E, N, K, is_a_8bit):
    utils = (PINNED / UTILS).read_text()
    nodes = functions(utils, ["get_scale_perms", "marlin_permute_scales"])
    nodes += functions(text, ["nvfp4_marlin_process_scales", "_nvfp4_compute_scale_factor"]
                       + (["_arc_batched_marlin_scales"] if "ARC marlin_batch" in text else []))
    nodes += functions(text, ["permute_scales"], nested_in="prepare_nvfp4_moe_layer_for_marlin")
    logger = types.SimpleNamespace(warning_once=lambda *a: None)
    ns = {"torch": torch, "logger": logger, "param_dtype": torch.bfloat16, "E": E, "K": K, "padded_N": N,
          "num_shards": 2, "GROUP_SIZE": 16, "is_a_8bit": is_a_8bit, "pad_w13": lambda x: x,
          "pad_w2": lambda x, packing: x, "nvfp4_marlin_process_global_scale": lambda g, d: g.to(d)}
    if "_arc_marlin_scale_factor" in text:  # expert_mask_patch's hook: its plain fallback
        ns["_arc_marlin_scale_factor"] = lambda layer, name, scales, dtype, compute: compute(scales, dtype)
        ns["layer"] = None
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "marlin", "exec"), ns)
    per_expert, calls = ns["marlin_permute_scales"], []
    ns["marlin_permute_scales"] = lambda **kw: calls.append(1) or per_expert(**kw)
    stub = types.ModuleType("stub_marlin_utils")
    stub.get_scale_perms = ns["get_scale_perms"]
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.quantization",
                 "vllm.model_executor.layers.quantization.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, UTILS[:-3].replace("/", "."), stub)
    return ns["permute_scales"], calls


def scales(E, rows, cols, peak, seed):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(E, rows, cols, generator=g) * peak).to(torch.float8_e4m3fn).to(torch.bfloat16)
    s[torch.rand(E, rows, cols, generator=g) < 0.05] = 0
    return s


def run_both(text, monkeypatch, is_a_8bit, peak):
    E, N, K = 3, 128, 256
    out = {}
    for on in (False, True):
        monkeypatch.setenv(patch.ENV, "1") if on else monkeypatch.delenv(patch.ENV, raising=False)
        fn, calls = permute_scales(text, monkeypatch, E, N, K, is_a_8bit)
        w13 = fn(scales(E, 2 * N, K // 16, peak, 1), torch.tensor(0.5), "w13")
        w2 = fn(scales(E, K, N // 16, peak, 2), torch.tensor(0.25), "w2")
        assert len(calls) == (0 if on else 2 * E)  # batched path taken only with the env
        out[on] = [t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t for pair in (w13, w2) for t in pair]
    for off, on in zip(out[False], out[True]):
        assert off.shape == on.shape and off.dtype == on.dtype and torch.equal(off, on)
    return out[True]


def expert_masked(text):
    old = "\n        combined_scale_factor = _nvfp4_compute_scale_factor(scales, param_dtype)\n"
    new = ("\n        combined_scale_factor = _arc_marlin_scale_factor(layer, name, scales, param_dtype,\n"
           "                                                         _nvfp4_compute_scale_factor)\n")
    assert text.count(old) == 1
    return text.replace(old, new)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("is_a_8bit", [False, True])
@pytest.mark.parametrize("peak", [448.0, 6.0])  # scale factor 1 and > 1
def test_bit_identical(monkeypatch, masked, is_a_8bit, peak):
    original = (PINNED / patch.RELATIVE).read_text()
    text = patch.patched_source((expert_masked(original) if masked else original).encode()).decode()
    assert text.count("ARC marlin_batch") == 2
    patched = run_both(text, monkeypatch, is_a_8bit, peak)
    monkeypatch.delenv(patch.ENV, raising=False)
    plain, _ = permute_scales(original, monkeypatch, 3, 128, 256, is_a_8bit)
    ref = plain(scales(3, 256, 16, peak, 1), torch.tensor(0.5), "w13")
    assert torch.equal(ref[0].view(torch.uint8), patched[0]) and torch.equal(ref[1], patched[1])


def test_apply_idempotent_and_skip(tmp_path, capsys):
    site = tmp_path / "site"
    (site / patch.RELATIVE).parent.mkdir(parents=True)
    shutil.copy(PINNED / patch.RELATIVE, site / patch.RELATIVE)
    identity = patch.apply(site, tmp_path / "out")
    assert identity["original_sha256"].startswith("a7c61d4f")
    assert json.loads((tmp_path / "out/marlin-batch.json").read_text()) == identity
    assert patch.apply(site, tmp_path / "out") == identity
    (site / patch.RELATIVE).write_text("x = 1\n")
    assert patch.apply(site, tmp_path / "out2") is None
    assert "ARC_MARLIN_BATCH_PATCH_SKIPPED" in capsys.readouterr().out
