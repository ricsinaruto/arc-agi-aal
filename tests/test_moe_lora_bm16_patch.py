"""moe_lora_bm16_patch: lands on the pinned punica_gpu.py (plain, after expert_mask compact, with lora_dense_decode
before or after), idempotent, skips unknown files; the patched add_lora_w13/add_lora_w2 hand vLLM's align and both
one-shot calls 16-row blocks up to the cap, and vLLM's sizes otherwise. Bitwise equality on GPU: moel4 (B300)."""

import ast
import os
import sys
import types
from pathlib import Path

import pytest

from arc_infra import moe_lora_bm16_patch as patch

PINNED = Path(os.environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
need_pinned = pytest.mark.skipif(not (PINNED / patch.PUNICA).exists(), reason="no local copy of the pinned vLLM")


def compact(original):
    from arc_infra import expert_mask_patch as em
    target = next(t for t in em.COMPACT_TARGETS if t["relative"] == patch.PUNICA)
    return em.compact_source(target, original)


@need_pinned
@pytest.mark.parametrize("masked,dense", [(False, None), (True, None), (True, "before"), (True, "after")])
def test_patch_applies_idempotent(tmp_path, masked, dense):
    dd = pytest.importorskip("arc_infra.lora_dense_decode_patch") if dense else None  # rl-async only
    original = (PINNED / patch.PUNICA).read_bytes()
    if masked:  # expert_mask_patch's compact edits on the same file come first at boot
        original = compact(original)
    if dense == "before":
        original = dd.patched_source(original)
    site = tmp_path / "site"
    (site / patch.PUNICA).parent.mkdir(parents=True)
    (site / patch.PUNICA).write_bytes(original)
    ident = patch.apply(site, tmp_path / "out")
    assert ident and (tmp_path / "out/moe-lora-bm16-patch.json").exists()
    if dense == "after":
        (site / patch.PUNICA).write_bytes(dd.patched_source((site / patch.PUNICA).read_bytes()))
    text = (site / patch.PUNICA).read_text()
    w13 = text.split("def add_lora_w13")[1].split("def add_lora_w2")[0]
    w2 = text.split("def add_lora_w2")[1]
    assert w13.index("if _ARC_BM16") < w13.index("self.moe_lora_align_block_size(")
    assert w2.index("if _ARC_BM16") < w2.index("self.add_lora_fused_moe(")
    assert ("_arc_moe_experts(" in text) == masked and ("ARC lora_dense_decode" in text) == bool(dense)
    if dense != "after":
        assert patch.apply(site, tmp_path / "out") == ident  # idempotent
        assert (site / patch.PUNICA).read_text() == text


def test_unknown_file_skipped(tmp_path, capsys):
    (tmp_path / patch.PUNICA).parent.mkdir(parents=True)
    (tmp_path / patch.PUNICA).write_text("x = 1\n")
    assert patch.apply(tmp_path, tmp_path / "out") is None
    assert "ARC_MOE_LORA_BM16_PATCH_SKIPPED" in capsys.readouterr().out
    assert (tmp_path / patch.PUNICA).read_text() == "x = 1\n"


def rule_bm(M):  # vLLM's bf16 default config rule
    return 16 if M <= 32 else 32 if M <= 96 else 64


def _punica_ns(monkeypatch, env):
    for k in ("ARC_MOE_LORA_BM16", "ARC_MOE_LORA_BM16_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    torch = pytest.importorskip("torch")
    tu = types.ModuleType("vllm.triton_utils")
    tu.tl = tu.triton = None
    lu = types.ModuleType("vllm.lora.layers.utils")
    lu.try_get_optimal_moe_lora_config = lambda op_type, M, **kw: {"BLOCK_SIZE_M": rule_bm(M)}
    ou = types.ModuleType("vllm.lora.ops.triton_ops.utils")
    ou._normalize_lora_config_keys = lambda c: c
    ou.get_lora_op_configs = None
    fc = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    fc._get_config_dtype_str = lambda **kw: None
    for name, mod in (("vllm", types.ModuleType("vllm")), ("vllm.triton_utils", tu), ("vllm.lora.layers.utils", lu),
                      ("vllm.lora.ops.triton_ops.utils", ou), ("vllm.model_executor.layers.fused_moe.config", fc)):
        monkeypatch.setitem(sys.modules, name, mod)
    text = patch.patched_source(compact((PINNED / patch.PUNICA).read_bytes())).decode()
    ns = {"torch": torch, "_arc_moe_shape": lambda s: s, "_arc_moe_experts": lambda e: e}
    exec(patch.HEADER, ns)  # module-level flags, read at import
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "PunicaWrapperGPU")
    for m in cls.body:
        if isinstance(m, ast.FunctionDef) and m.name in ("add_lora_w13", "add_lora_w2"):
            m.decorator_list = []
            exec(compile(ast.Module(body=[m], type_ignores=[]), "punica", "exec"), ns)
    return ns, torch


class FakePunica:
    def __init__(self, tlm):
        self.tlm = tlm
        self.lora_config = types.SimpleNamespace(specialize_active_lora=False)
        self.token_mapping_meta = types.SimpleNamespace(meta_args=self.meta_args)
        self.calls = []

    def meta_args(self, n, specialize):
        import torch
        return (self.tlm[:n], None, None, None, None, torch.tensor([bool((self.tlm < 0).all())]), None)

    def moe_lora_align_block_size(self, topk_ids, num_tokens, block_size, *a, **kw):
        self.calls.append(("align", block_size))
        return None, None, None, None

    def add_lora_fused_moe(self, *a, **kw):
        self.calls.append(("fused", a[10]["BLOCK_SIZE_M"], a[11]["BLOCK_SIZE_M"]))


@need_pinned
@pytest.mark.parametrize("env,M,bm,why", [
    ({}, 40, 32, "flag off"),
    ({}, 120, 64, "flag off, M > 96"),
    ({"ARC_MOE_LORA_BM16": "1"}, 40, 16, "BM16"),
    ({"ARC_MOE_LORA_BM16": "1"}, 120, 16, "BM16 at M > 96"),
    ({"ARC_MOE_LORA_BM16": "1"}, 12, 16, "already 16"),
    ({"ARC_MOE_LORA_BM16": "1"}, 2, 16, "naive"),
    ({"ARC_MOE_LORA_BM16": "1", "ARC_MOE_LORA_BM16_MAX_TOKENS": "32"}, 40, 32, "over the cap")])
def test_block_size(monkeypatch, env, M, bm, why):
    ns, torch = _punica_ns(monkeypatch, env)
    E, top_k, max_loras = 24, 4, 5
    tlm = torch.tensor([i % 3 for i in range(M)], dtype=torch.int32)
    topk_ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(M)]).to(torch.int32)
    punica = FakePunica(tlm)
    a13, b13 = tuple(torch.zeros(max_loras, E, 16, 8) for _ in range(2)), tuple(torch.zeros(max_loras, E, 8, 16)
                                                                                  for _ in range(2))
    w1 = torch.empty(E, 1, 1)
    enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    meta = ns["add_lora_w13"](punica, torch.zeros(M * top_k, 16), torch.zeros(M, 8), a13, b13, topk_ids,
                              torch.rand(M, top_k), None, w1, w1, M, top_k, max_loras, enabled, E, top_k, 2, False,
                              False)
    ns["add_lora_w2"](punica, torch.zeros(M, top_k, 8), torch.zeros(M * top_k, 8), a13[:1], b13[:1],
                      torch.rand(M, top_k), *meta, M, w1, w1, top_k, max_loras, enabled, top_k, False, 0, False)
    assert punica.calls == [("align", bm), ("fused", bm, bm), ("fused", bm, bm)], why
