"""lora_segment_patch (MFU-2, 25 Oct): the per-adapter cuBLAS shrink/expand match the per-token semantics of vLLM's
Triton lora_shrink/lora_expand (1, 3 and 5 active adapters under max_loras 5, no-adapter tokens, 1 and 3 slices,
chunked spans); small steps, ARC_LORA_SEGMENT=0, padded-short inputs and no-LoRA batches keep the Triton path; the
spans come from the real LoRAKernelMeta.prepare_tensors; the anchors hold on the pinned files; apply is idempotent
and skips an unknown file."""

import json
import sys
import types
from pathlib import Path

import pytest

from arc_infra import lora_segment_patch as patch

PINNED = Path(__import__("os").environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
need_pinned = pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
META = "vllm/lora/ops/triton_ops/lora_kernel_metadata.py"
MAX_LORAS, HIDDEN, RANK = 5, 96, 16


def seg_namespace(torch):
    call = "\n_arc_seg_wrap_prepare()\n"
    assert patch.SOURCE.count(call) == 1
    ns = {"torch": torch}
    exec(compile(patch.SOURCE.replace(call, "\n"), "lora_segment", "exec"), ns)
    return ns


class MirrorMeta:
    """LoRAKernelMeta.prepare_tensors of the pinned vLLM, for runs without the pinned copy."""

    def __init__(self, torch, max_loras, n):
        self.t = torch
        self.token_indices_sorted_by_lora_ids = torch.empty(n, dtype=torch.int32)
        self.active_lora_ids = torch.empty(max_loras + 1, dtype=torch.int32)
        self.num_tokens_per_lora = torch.zeros(max_loras + 1, dtype=torch.int32)
        self.no_lora_flag_cpu = torch.tensor([False])

    def prepare_tensors(self, mapping):
        t = self.t
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.no_lora_flag_cpu[0] = bool(t.all(mapping == -1))
        if self.no_lora_flag_cpu[0]:
            return
        self.token_indices_sorted_by_lora_ids[:mapping.size(0)] = t.sort(mapping, stable=True)[1]
        ids, counts = t.unique(mapping, sorted=True, return_counts=True)
        self.active_lora_ids[:ids.size(0)] = ids
        self.num_tokens_per_lora[:counts.size(0)] = counts


def mapping_for(torch, m, active, gen, none_share=0.2):
    slots = torch.tensor(active, dtype=torch.int32)[torch.randint(len(active), (m,), generator=gen)]
    slots[torch.rand(m, generator=gen) < none_share] = -1
    return slots


def shrink_ref(torch, x, weights, mapping, scale):
    out = torch.zeros(len(weights), x.size(0), RANK, dtype=torch.float32)
    for t, slot in enumerate(mapping.tolist()):
        if slot >= 0:
            for s, w in enumerate(weights):
                out[s, t] = ((x[t].double() @ w.squeeze(1)[slot].double().T) * scale).float()
    return out


def expand_ref(torch, xin, weights, out, mapping, offset, add):
    out = out.clone()
    for t, slot in enumerate(mapping.tolist()):
        if slot < 0:
            continue
        off = offset
        for s, w in enumerate(weights):
            w = w.squeeze(1)
            n = w.size(1)
            y = (xin[s, t].to(w.dtype).double() @ w[slot].double().T).to(out.dtype)
            out[t, off:off + n] = (out[t, off:off + n].double() + y.double()).to(out.dtype) if add else y
            off += n
    return out


def close_bf16(torch, got, ref):
    assert (got.float() - ref.float()).abs().le(ref.float().abs() * 2 ** -7 + 1e-6).all()


@pytest.mark.parametrize("active", [[2], [0, 3, 4], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("slices", [[64], [64, 32, 32]])
@pytest.mark.parametrize("add", [True, False])
def test_matches_triton_semantics(monkeypatch, active, slices, add):
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(len(active) * 7 + len(slices))
    monkeypatch.setenv("ARC_LORA_SEGMENT", "1")
    monkeypatch.setenv("ARC_LORA_SEGMENT_MIN", "256")
    monkeypatch.setenv("ARC_LORA_SEGMENT_CHUNK", "128")  # spans of ~100-500 tokens split into chunks
    ns = seg_namespace(torch)
    m, scale = 600, 0.5
    mapping = mapping_for(torch, m, active, gen)
    meta = MirrorMeta(torch, MAX_LORAS, m)
    meta.prepare_tensors(mapping)
    ns["_arc_seg_note"](meta, mapping)
    x = torch.randn(m, HIDDEN, generator=gen).bfloat16()
    a = [torch.randn(MAX_LORAS, 1, RANK, HIDDEN, generator=gen).bfloat16() for _ in slices]
    b = [torch.randn(MAX_LORAS, 1, n, RANK, generator=gen).bfloat16() for n in slices]

    shrunk = torch.full((len(slices), m, RANK), 7.0)
    shrunk.zero_()  # the op zeroes its output before the edit point
    assert ns["_arc_seg_shrink"](x, a, shrunk, meta.num_tokens_per_lora, meta.token_indices_sorted_by_lora_ids, scale)
    ref = shrink_ref(torch, x, a, mapping, scale)
    assert torch.allclose(shrunk, ref, rtol=1e-5, atol=1e-5)

    offset = 8
    out = torch.randn(m, offset + sum(slices) + 4, generator=gen).bfloat16()
    want = expand_ref(torch, shrunk, b, out, mapping, offset, add)
    assert ns["_arc_seg_expand"](shrunk, b, out, meta.num_tokens_per_lora, meta.token_indices_sorted_by_lora_ids,
                                 offset, add)
    none = mapping < 0
    assert torch.equal(out[none], want[none])  # no-adapter rows untouched
    close_bf16(torch, out, want)
    assert ns["_ARC_SEG_USED"]["shrink"] == 1 and ns["_ARC_SEG_USED"]["expand"] == 1


def test_keeps_triton_path(monkeypatch):
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(1)
    monkeypatch.setenv("ARC_LORA_SEGMENT_MIN", "256")
    ns = seg_namespace(torch)
    x = torch.randn(300, HIDDEN, generator=gen).bfloat16()
    a = [torch.randn(MAX_LORAS, 1, RANK, HIDDEN, generator=gen).bfloat16()]
    out = torch.zeros(1, 300, RANK)

    def run(mapping, rows=300, env="1"):
        monkeypatch.setenv("ARC_LORA_SEGMENT", env)
        meta = MirrorMeta(torch, MAX_LORAS, mapping.size(0))
        meta.prepare_tensors(mapping)
        ns["_arc_seg_note"](meta, mapping)
        return ns["_arc_seg_shrink"](x[:rows], a, out[:, :rows], meta.num_tokens_per_lora,
                                     meta.token_indices_sorted_by_lora_ids, 1.0)

    assert run(mapping_for(torch, 300, [1, 2], gen))
    assert not run(mapping_for(torch, 200, [1, 2], gen), rows=200)  # below ARC_LORA_SEGMENT_MIN
    assert not run(mapping_for(torch, 300, [1, 2], gen), env="0")
    assert not run(torch.full((300,), -1, dtype=torch.int32))  # no adapter in the batch
    assert not run(mapping_for(torch, 300, [1, 2], gen), rows=280)  # fewer rows than the prepared tokens
    assert run(mapping_for(torch, 280, [1, 2], gen))  # more rows (padding) than the prepared tokens


@need_pinned
def test_spans_from_real_prepare(monkeypatch):
    torch = pytest.importorskip("torch")
    module = types.ModuleType("vllm.lora.ops.triton_ops.lora_kernel_metadata")
    exec(compile((PINNED / META).read_text(), META, "exec"), module.__dict__)
    for name in ("vllm", "vllm.lora", "vllm.lora.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    parent = types.ModuleType("vllm.lora.ops.triton_ops")
    parent.lora_kernel_metadata = module
    monkeypatch.setitem(sys.modules, "vllm.lora.ops.triton_ops", parent)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("ARC_LORA_SEGMENT_MIN", "4")
    ns = seg_namespace(torch)
    plain = module.LoRAKernelMeta.prepare_tensors
    ns["_arc_seg_wrap_prepare"]()
    ns["_arc_seg_wrap_prepare"]()  # once only
    wrapped = module.LoRAKernelMeta.prepare_tensors
    assert wrapped is not plain and wrapped._arc_seg
    meta = module.LoRAKernelMeta.make(MAX_LORAS, 16, device="cpu", captured_lora_counts=[1, 2, 4, 6])
    mapping = torch.tensor([3, -1, 0, 3, 3, -1, 0, 4], dtype=torch.int32)
    meta.prepare_tensors(mapping)
    n, spans = ns["_ARC_SEG"][meta.num_tokens_per_lora.data_ptr()]
    assert n == 8 and spans == ((0, 2, 2), (3, 4, 3), (4, 7, 1))
    order = meta.token_indices_sorted_by_lora_ids[:8].tolist()
    assert [mapping[i].item() for i in order[2:4]] == [0, 0] and [mapping[i].item() for i in order[4:7]] == [3] * 3
    meta.prepare_tensors(torch.full((8,), -1, dtype=torch.int32))
    assert meta.num_tokens_per_lora.data_ptr() not in ns["_ARC_SEG"]
    monkeypatch.setattr(module.LoRAKernelMeta, "prepare_tensors", plain)


@need_pinned
def test_anchors_on_pinned():
    shrink, expand = patch.patched_sources((PINNED / patch.SHRINK).read_bytes(), (PINNED / patch.EXPAND).read_bytes())
    assert shrink.decode().count(patch.TAG) == 2 and expand.decode().count(patch.TAG) == 2
    assert shrink.decode().index("output_tensor.zero_()") < shrink.decode().index("if _arc_seg_shrink(")


@need_pinned
def test_apply_idempotent_and_skip(tmp_path, capsys):
    site, out = tmp_path / "site", tmp_path / "out"
    for rel in (patch.SHRINK, patch.EXPAND):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_bytes((PINNED / rel).read_bytes())
    first = patch.apply(site, out)
    assert first and json.loads((out / "lora-segment-patch.json").read_text()) == first
    assert patch.apply(site, out) == first
    (site / patch.SHRINK).write_bytes((PINNED / patch.SHRINK).read_bytes())
    (site / patch.EXPAND).write_text("def _lora_expand():\n    pass\n")
    shrink_before = (site / patch.SHRINK).read_bytes()
    assert patch.apply(site, tmp_path / "out2") is None
    assert "ARC_LORA_SEGMENT_PATCH_SKIPPED" in capsys.readouterr().out
    assert (site / patch.SHRINK).read_bytes() == shrink_before  # both files or neither
