"""MFU stage Q (MFU-1, 22 Oct): the routed experts' NVFP4 -> BF16 dequant of the trainer's Axolotl path
(``nvfp4_lora.dequantize_expert_weight``, called twice per MoE layer per pass) as one coalesced Triton kernel with
the same arithmetic, so the same bits: BF16(q x (block scale x per-expert scale)) in fp32 (mfu-d2-p3: that order
reproduces Axolotl's output bit for bit on both projections; native_experts' (q x s) x pts differs in 0.17% of w2).
rl1-prof-m1a: Axolotl's ``_dequant_kernel`` 31.8 s of the 372 s step (2.4 ms per call, ~1.8 TB/s).

Switch: ``DEQUANT["on"]`` (trainer cmd ``fast_dequant``, env RL1_FAST_DEQUANT=1); swizzled scales or other layouts
run Axolotl's function."""
import os

import torch
import triton
import triton.language as tl

DEQUANT = {"on": os.environ.get("RL1_FAST_DEQUANT") == "1", "calls": 0, "fallback": 0}
ORIGINAL = {"fn": None}


@triton.jit
def _dequant(q_ptr, s_ptr, pts_ptr, out_ptr, N, K: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """out[e, r, c] = BF16(e2m1(q) * (e4m3(s[c // 16]) * pts[e])), packed nibbles low first."""
    pc, pr, e = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    r = (pr * BR + tl.arange(0, BR)).to(tl.int64)
    mr = r < N
    q = tl.load(q_ptr + e * N * (K // 2) + r[:, None] * (K // 2) + pc * (BC // 2) + tl.arange(0, BC // 2)[None, :],
                mask=mr[:, None], other=0).to(tl.int32)
    s = tl.load(s_ptr + e * N * (K // 16) + r[:, None] * (K // 16) + pc * (BC // 16) + tl.arange(0, BC // 16)[None, :],
                mask=mr[:, None], other=0)
    c = tl.reshape(tl.join(q & 15, q >> 4), (BR, BC))
    mag = c & 7
    # E2M1 magnitude as BF16 bits (exact): normal codes (mag >= 2) rebias by 126; code 1 is 0.5; sign from bit 3
    bits = ((mag << 6) + tl.where(mag >= 2, 0x3F00, tl.where(mag == 1, 0x3EC0, 0))) | ((c & 8) << 12)
    v = bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    scale = s.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(pts_ptr + e)
    w = v * tl.reshape(tl.broadcast_to(scale[:, :, None], (BR, BC // 16, 16)), (BR, BC))
    tl.store(out_ptr + e * N * K + r[:, None] * K + pc * BC + tl.arange(0, BC)[None, :], w.to(tl.bfloat16),
             mask=mr[:, None])


def _layout(w):
    """(qdata u8 [E, N, K/2], scale bits u8 [E, N, K/16], per-expert fp32 [E]) or None when not the plain layout."""
    if getattr(w, "is_swizzled_scales", False) or not hasattr(w, "qdata") or not hasattr(w, "scale"):
        return None
    q, s = w.qdata.view(torch.uint8), w.scale.view(torch.uint8)
    if q.dim() != 3 or s.dim() != 3 or not (q.is_contiguous() and s.is_contiguous()) or not q.is_cuda:
        return None
    E, N, K = q.shape[0], q.shape[1], q.shape[2] * 2
    if tuple(s.shape) != (E, N, K // 16) or K % 128:
        return None
    pts = w.per_tensor_scale
    pts = (torch.ones(E, device=q.device) if pts is None else pts.float().reshape(-1).expand(E).contiguous())
    return q, s, pts, E, N, K


def dequantize(w):
    """Drop-in for ``nvfp4_lora.dequantize_expert_weight``: [E, N, K] BF16."""
    layout = _layout(w) if DEQUANT["on"] else None
    if layout is None:
        DEQUANT["fallback"] += DEQUANT["on"]
        return ORIGINAL["fn"](w)
    q, s, pts, E, N, K = layout
    out = torch.empty((E, N, K), device=q.device, dtype=torch.bfloat16)
    BR, BC = 32, 128
    _dequant[(K // BC, triton.cdiv(N, BR), E)](q, s, pts, out, N, K=K, BR=BR, BC=BC, num_warps=4)
    DEQUANT["calls"] += 1
    return out


def install():
    """Route Axolotl's dequant (looked up as a module global at call time) through ``dequantize``."""
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    if ORIGINAL["fn"] is None:
        ORIGINAL["fn"] = nvfp4_lora.dequantize_expert_weight
        nvfp4_lora.dequantize_expert_weight = dequantize
