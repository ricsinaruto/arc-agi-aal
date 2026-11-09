"""ARC_LORA_SEGMENT=1 (MFU-2 play throughput, 25 Oct): on eager (not graph-captured) steps of at least
ARC_LORA_SEGMENT_MIN tokens, vLLM's Triton LoRA shrink/expand (lora_shrink_op / lora_expand_op) is replaced by one
cuBLAS GEMM per active adapter over its tokens (gather rows, GEMM, scatter rows). Same per-token math: shrink is a
bf16 x bf16 product accumulated in fp32, times scale, written fp32; expand casts its input to the weight dtype,
accumulates in fp32, rounds to the output dtype and adds it to the output (or writes it). tput10 (B300 hero stack):
the Triton expand ran at ~130 GB/s on mixed steps; dense LoRA was 61.6 of 253 ms of a mixed step.

The per-adapter token spans come from LoRAKernelMeta.prepare_tensors (wrapped here; it already syncs through
torch.unique), so the ops need no extra sync. Graph capture, smaller steps, a missing span table or an unrecognized
file keep the Triton path (ARC_LORA_SEGMENT_PATCH_SKIPPED), as does ARC_LORA_SEGMENT=0 on a patched runtime. Edits
only the two op files, whose bodies run inside opaque custom ops; the marker is a *-patch.json so the boot-cache key
changes with it."""

import hashlib
import json
from pathlib import Path

ENV = "ARC_LORA_SEGMENT"
TAG = "ARC lora_segment"
BASE = "vllm/lora/ops/triton_ops/"
SHRINK = BASE + "lora_shrink_op.py"
EXPAND = BASE + "lora_expand_op.py"

SHRINK_EDITS = (
    ("    output_tensor.zero_()\n",
     "    output_tensor.zero_()\n"
     "    if _arc_seg_shrink(inputs, lora_a_weights, output_tensor, num_tokens_per_lora,  # ARC lora_segment\n"
     "                       token_indices_sorted_by_lora_ids, scaling):\n"
     "        return\n"),
)
EXPAND_EDITS = (
    ("    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1\n",
     "    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1\n"
     "    if _arc_seg_expand(inputs, lora_b_weights, output_tensor, num_tokens_per_lora,  # ARC lora_segment\n"
     "                       token_indices_sorted_by_lora_ids, offset_start, add_inputs):\n"
     "        return\n"),
)

# Appended to lora_shrink_op.py. Self-contained (torch, os) so tests can exec it on CPU.
SOURCE = '''

# ARC lora_segment: per-adapter cuBLAS shrink/expand on large eager steps (arc_infra.lora_segment_patch).
import os as _arc_os

_ARC_SEG = {}  # id(meta.num_tokens_per_lora storage) -> (num_tokens, ((slot, start, count), ...))
_ARC_SEG_USED = {"shrink": 0, "expand": 0, "tokens": 0}
_ARC_MM_OUT_DTYPE = [True]


def _arc_seg_capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _arc_seg_note(meta, token_lora_mapping):
    """After prepare_tensors: record the per-adapter spans of token_indices_sorted_by_lora_ids on the host."""
    key = meta.num_tokens_per_lora.data_ptr()
    _ARC_SEG.pop(key, None)
    if _arc_os.environ.get("ARC_LORA_SEGMENT", "1") != "1" or bool(meta.no_lora_flag_cpu[0]):
        return
    n = token_lora_mapping.size(0)
    if n < int(_arc_os.environ.get("ARC_LORA_SEGMENT_MIN", "256")) or _arc_seg_capturing():
        return
    ids = meta.active_lora_ids.tolist()
    counts = meta.num_tokens_per_lora.tolist()
    spans, start = [], 0
    for slot, count in zip(ids, counts):
        if count <= 0:
            break
        if slot >= 0:
            spans.append((slot, start, count))
        start += count
    if start == n:
        _ARC_SEG[key] = (n, tuple(spans))


def _arc_seg_spans(num_tokens_per_lora, m):
    if _arc_os.environ.get("ARC_LORA_SEGMENT", "1") != "1":
        return None
    state = _ARC_SEG.get(num_tokens_per_lora.data_ptr())
    if state is None or m < state[0] or _arc_seg_capturing():
        return None
    return state


def _arc_seg_chunks(spans, sorted_idx, chunk):
    for slot, start, count in spans:
        for s in range(start, start + count, chunk):
            yield slot, sorted_idx[s:min(s + chunk, start + count)].long()


def _arc_mm_f32(a, b):
    """a @ b with bf16/fp16 operands, fp32 accumulate and fp32 output."""
    if _ARC_MM_OUT_DTYPE[0] and a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            _ARC_MM_OUT_DTYPE[0] = False
    return torch.mm(a.float(), b.float())  # bf16 values are exact in tf32/fp32


def _arc_seg_shrink(inputs, lora_a_weights, output_tensor, num_tokens_per_lora, sorted_idx, scaling):
    state = _arc_seg_spans(num_tokens_per_lora, inputs.size(0))
    if state is None:
        return False
    ws = [w.squeeze(1) if w.ndim == 4 else w for w in lora_a_weights]
    rank = ws[0].size(1)
    chunk = int(_arc_os.environ.get("ARC_LORA_SEGMENT_CHUNK", "4096"))
    for slot, idx in _arc_seg_chunks(state[1], sorted_idx, chunk):
        a = ws[0][slot] if len(ws) == 1 else torch.cat([w[slot] for w in ws], 0)
        y = _arc_mm_f32(inputs.index_select(0, idx), a.t())
        if scaling != 1.0:
            y.mul_(scaling)
        for s in range(len(ws)):
            output_tensor[s].index_copy_(0, idx, y[:, s * rank:(s + 1) * rank].to(output_tensor.dtype))
    _ARC_SEG_USED["shrink"] += 1
    _ARC_SEG_USED["tokens"] += state[0]
    if _ARC_SEG_USED["shrink"] == 1:
        print("ARC_LORA_SEGMENT_USED " + str({"m": state[0], "adapters": len(state[1]),
                                              "mm_out_dtype": _ARC_MM_OUT_DTYPE[0]}), flush=True)
    return True


def _arc_seg_expand(inputs, lora_b_weights, output_tensor, num_tokens_per_lora, sorted_idx, offset_start, add_inputs):
    state = _arc_seg_spans(num_tokens_per_lora, inputs.size(1))
    if state is None:
        return False
    ws = [w.squeeze(1) if w.ndim == 4 else w for w in lora_b_weights]
    chunk = int(_arc_os.environ.get("ARC_LORA_SEGMENT_CHUNK", "4096"))
    for slot, idx in _arc_seg_chunks(state[1], sorted_idx, chunk):
        offset = offset_start
        for s, w in enumerate(ws):
            n = w.size(1)
            x = inputs[s].index_select(0, idx).to(w.dtype)
            y = torch.mm(x, w[slot].t()).to(output_tensor.dtype)
            dst = output_tensor[:, offset:offset + n]
            if add_inputs:
                dst.index_add_(0, idx, y)
            else:
                dst.index_copy_(0, idx, y)
            offset += n
    _ARC_SEG_USED["expand"] += 1
    return True


def _arc_seg_wrap_prepare():
    from vllm.lora.ops.triton_ops import lora_kernel_metadata as meta_module

    cls = meta_module.LoRAKernelMeta
    if getattr(cls.prepare_tensors, "_arc_seg", False):
        return
    plain = cls.prepare_tensors

    def prepare_tensors(self, token_lora_mapping):
        plain(self, token_lora_mapping)
        _arc_seg_note(self, token_lora_mapping)

    prepare_tensors._arc_seg = True
    cls.prepare_tensors = prepare_tensors


_arc_seg_wrap_prepare()
'''

EXPAND_SOURCE = '''

from vllm.lora.ops.triton_ops.lora_shrink_op import _arc_seg_expand  # noqa: E402  ARC lora_segment
'''


def _edited(original, edits, source, relative):
    text = original.decode()
    if TAG in text:
        raise RuntimeError(f"lora_segment patch: {relative} already carries the edit")
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"lora_segment anchor is not unique in {relative}: {old.strip()[:60]}")
        text = text.replace(old, new)
    text = text.rstrip("\n") + "\n" + source
    compile(text, relative, "exec")
    return text.encode()


def patched_sources(shrink, expand):
    return (_edited(shrink, SHRINK_EDITS, SOURCE, SHRINK), _edited(expand, EXPAND_EDITS, EXPAND_SOURCE, EXPAND))


def apply(site, output):
    """Never raises: an unrecognized file keeps the Triton path. Both files are edited or neither."""
    site = Path(site)
    originals = [(site / SHRINK).read_bytes(), (site / EXPAND).read_bytes()]
    marker = Path(output) / "lora-segment-patch.json"
    if any(TAG.encode() in o for o in originals):
        return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
    try:
        patched = patched_sources(*originals)
    except Exception as error:  # noqa: BLE001 - a speed path, never a reason to fail a boot
        print("ARC_LORA_SEGMENT_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    for relative, text in zip((SHRINK, EXPAND), patched):
        (site / relative).write_bytes(text)
    identity = {name: {"original_sha256": hashlib.sha256(o).hexdigest(),
                       "patched_sha256": hashlib.sha256(p).hexdigest()}
                for name, o, p in zip(("shrink", "expand"), originals, patched)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_LORA_SEGMENT_PATCH " + json.dumps(identity), flush=True)
    return identity
