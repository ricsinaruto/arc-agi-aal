"""ARC_STABLE_ALIGN=1 (MFU-2 24 Oct, microbench only): deterministic MoE align. vLLM's moe_align_block_size places
each (token, slot) inside its expert's segment by atomic increment, so two identical passes get the same segments in
a different order, and the Marlin MoE GEMM after it gives different values (det-v2). This hash-guarded edit of
vllm/_custom_ops.py re-sorts every segment by flat token index after the kernel (same layout and segment bounds,
padding entries last), for moe_align_block_size and moe_lora_align_block_size. Pure torch, no host sync, so CUDA
graphs capture it. Not for served paths until the microbench (tok/s, cross-pass |dlogp|) is read."""

import hashlib
import json
from pathlib import Path

ENV = "ARC_STABLE_ALIGN"
RELATIVE = "vllm/_custom_ops.py"
ORIGINAL_SHA256 = "9f11980f43c44857acd88e6f5c96cbd2d767b657b05207e26f064566200dae11"

# Self-contained (the engine process imports only vLLM): appended to _custom_ops.py, exec'd by the tests.
SOURCE = '''

_ARC_STABLE_ALIGN = __import__("os").environ.get("ARC_STABLE_ALIGN") == "1"  # ARC (arc_infra.stable_align)


def _arc_stable_align(sorted_ids, expert_ids, num_tokens_post_pad, block_size, numel):
    """Each expert segment of ``sorted_ids`` sorted by token index, in place; rows = num_tokens_post_pad.numel()."""
    rows = num_tokens_post_pad.numel()
    ids = sorted_ids.view(rows, -1)
    blocks = expert_ids.view(rows, -1).to(torch.int64)
    n = ids.shape[1]
    pos = torch.arange(n, device=ids.device, dtype=torch.int64)
    blk = torch.clamp(pos // block_size, max=blocks.shape[1] - 1)
    starts = torch.ones_like(blocks)
    starts[:, 1:] = (blocks[:, 1:] != blocks[:, :-1]).to(torch.int64)
    seg = torch.cumsum(starts, dim=1)[:, blk]  # [rows, n], nondecreasing along each row
    big = int(numel) + 2
    valid = pos.unsqueeze(0) < num_tokens_post_pad.view(rows, 1).to(torch.int64)
    tok = torch.clamp(ids.to(torch.int64), min=0, max=big - 1)
    key = torch.where(valid, seg * big + tok, (blocks.shape[1] + 2) * big + pos.unsqueeze(0))
    order = torch.sort(key, dim=1).indices
    ids.copy_(torch.gather(ids, 1, order))
'''

EDITS = (
    ("        expert_map,\n    )\n\n\ndef batched_moe_align_block_size(",
     "        expert_map,\n    )\n    if _ARC_STABLE_ALIGN:  # ARC\n"
     "        _arc_stable_align(sorted_token_ids, experts_ids, num_tokens_post_pad, block_size, topk_ids.numel())\n"
     "\n\ndef batched_moe_align_block_size("),
    ("        expert_map,\n    )\n\n\ndef moe_wna16_gemm(",
     "        expert_map,\n    )\n    if _ARC_STABLE_ALIGN:  # ARC\n"
     "        _arc_stable_align(sorted_token_ids, experts_ids, num_tokens_post_pad, block_size, topk_ids.numel())\n"
     "\n\ndef moe_wna16_gemm("),
)


def patched_source(original):
    text = original.decode()  # the pinned hash is recorded; another build passes if both anchors are unique
    if "_arc_stable_align" in text or "def moe_lora_align_block_size(" not in text:
        raise RuntimeError("stable_align patch rejected an unrecognized vLLM _custom_ops.py")
    for old, new in EDITS:
        if text.count(old) != 1:
            raise RuntimeError(f"stable_align anchor is not unique: {old.strip()[:60]}")
        text = text.replace(old, new)
    text = text.rstrip("\n") + "\n" + SOURCE
    compile(text, RELATIVE, "exec")
    return text.encode()


def apply(site, output):
    target = Path(site) / RELATIVE
    original = target.read_bytes()
    marker = Path(output) / "stable-align.json"  # not *-patch.json: no compiled code changes (the op is opaque)
    if marker.exists():
        identity = json.loads(marker.read_text())
        if hashlib.sha256(original).hexdigest() == identity["patched_sha256"]:
            return identity
        if b"_arc_stable_align" in original:
            raise RuntimeError("Previously patched stable_align runtime identity changed")
        # else: the runtime was extracted again (next trial on this VM): patch it again below
    if original.decode().endswith(SOURCE):  # patched by an earlier trial on this VM (its marker elsewhere)
        patched = original
    else:
        patched = patched_source(original)
        target.write_bytes(patched)
    seen = hashlib.sha256(original).hexdigest() if patched is not original else "already patched"
    identity = {"original_sha256": seen, "pinned": seen == ORIGINAL_SHA256,
                "patched_sha256": hashlib.sha256(patched).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_STABLE_ALIGN_PATCH " + json.dumps(identity), flush=True)
    return identity


# ARC_STABLE_QSA=1 (det-v4: with the align fixed, the first own diff is layer 3's QSA indexer: same selected block set,
# different order from persistent_topk / cooperative_topk, so the sparse attention sums in a different order). Blocks
# are sorted per row after the top-k (valid ids ascending, -1 padding last), before the token expansion.
# ARC_QSA_TIES=index (24 Oct): the selected SET also varies, since persistent_topk claims slots among exact-equal scores
# with atomics (csrc/persistent_topk.cuh, last radix pass) and relu-zero scores tie often; exact zeros get a
# lowest-column-first order before the top-k (the trainer's RL1_QSA_TIES=index rule, sft-heal 3e552442e).
QSA_RELATIVE = "vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
QSA_ORIGINAL_SHA256 = "c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f"
QSA_ANCHOR = "        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n"
QSA_SOURCE = '''

_ARC_STABLE_QSA = __import__("os").environ.get("ARC_STABLE_QSA") == "1"  # ARC (arc_infra.stable_align)


def _arc_sort_blocks(blocks):
    """Per-row ascending block ids, -1 padding last, in place."""
    big = torch.iinfo(torch.int32).max
    keyed = torch.where(blocks < 0, torch.full_like(blocks, big), blocks)
    keyed = torch.sort(keyed, dim=-1).values
    blocks.copy_(torch.where(keyed == big, torch.full_like(blocks, -1), keyed))


_ARC_QSA_TIES = __import__("os").environ.get("ARC_QSA_TIES") == "index"  # ARC (arc_infra.stable_align)


def _arc_tie_by_index(logits):
    """Exact-zero (all-heads relu) scores become -(1 + column) * 2**-100: below every positive score, above -inf,
    distinct, the lower column first, so the radix top-k (atomic claims among equal values) picks a fixed set."""
    col = torch.arange(1, logits.shape[-1] + 1, device=logits.device, dtype=torch.float32)
    logits.copy_(torch.where(logits == 0, col * -(2.0 ** -100), logits))
'''


def qsa_patched_source(original):
    text = original.decode()
    if "_arc_sort_blocks" in text or text.count(QSA_ANCHOR) != 1:
        raise RuntimeError("stable_qsa patch rejected an unrecognized qsa.py")
    text = text.replace(QSA_ANCHOR, "        if _ARC_QSA_TIES:  # ARC\n            _arc_tie_by_index(logits)\n" + QSA_ANCHOR
                        + "        if _ARC_STABLE_QSA:  # ARC\n            _arc_sort_blocks(blocks)\n")
    text = text.rstrip("\n") + "\n" + QSA_SOURCE
    compile(text, QSA_RELATIVE, "exec")
    return text.encode()


def apply_qsa(site):
    """Idempotent: the edit is re-made on every fresh runtime tree."""
    target = Path(site) / QSA_RELATIVE
    original = target.read_bytes()
    if original.decode().endswith(QSA_SOURCE):
        return {"already": True}
    patched = qsa_patched_source(original)
    target.write_bytes(patched)
    seen = hashlib.sha256(original).hexdigest()
    identity = {"original_sha256": seen, "pinned": seen == QSA_ORIGINAL_SHA256,
                "patched_sha256": hashlib.sha256(patched).hexdigest()}
    print("ARC_STABLE_QSA_PATCH " + json.dumps(identity), flush=True)
    return identity
