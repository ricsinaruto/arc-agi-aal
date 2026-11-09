"""MFU stage X (MFU-1, 22 Oct): the routed experts' elementwise work in the trainer's Axolotl dequant path, fused,
with the same rounding points (``nvfp4_lora.grouped_moe_reference_forward``, backend ``dequant``). rl1-prof-m1a:
outside the GEMMs and the dequant the experts spend ~55 s of the 372 s step in unfused elementwise / copy kernels
(the LoRA delta add, the SwiGLU chain, the router-weight scale, the gather + fp32 sum combine, and their backwards).

Kept as they are (same calls, so the same bits): routing and grouping (``route_and_group``), every GEMM (base
``grouped_up_gemm`` / ``grouped_down_gemm`` on the dequantized weights, ``_lora_delta_per_group``, ``_lora_backward_per_group``).
Fused (one Triton kernel each, the model path's BF16 roundings):
- up: h = BF16(base + delta), a = BF16(up x BF16(silu(gate))) (silu in fp32 as torch's kernel); backward: the
  autograd graph's dgate = BF16(fp32(BF16(da x up)) x silu'(gate)), dup = BF16(da x BF16(silu(gate)));
- down + combine: y = BF16(base + delta), out[t] = BF16(sum_k fp32(BF16(y_k x w_k))) in the order of
  ``_CombineByGather`` (rows of a token by a stable sort); backward: dy = BF16(dout[tok] x w), dw = sum_h dout y (torch's ops);
- ``_GatherRows`` backward: dh[t] = BF16(sum_k fp32(grad rows of t)).
FP32 hidden states (the trainer under autocast: 47 of 48 layers, rl1-prof-m1y) take the same path with that dtype's
rounding points: the grouped rows go to the up projection as BF16 casts (the trainer's ``grouped_expert_lora``), the
combine weights stay fp32 (out = BF16(sum_k fp32(y_k x w_k)), dy = BF16(fp32(dout) x w), dw fp32) and dh is the
unrounded fp32 sum.

Switch: ``EXPERTS["on"]`` (trainer cmd ``fused_experts``); other configurations (biases,
clamp, non-concat or non-SwiGLU, merge-aware, fp4_cute, CPU) run Axolotl's function."""

import torch
import triton
import triton.language as tl
from rl1_hyper import _mul, _tree  # torch's inner-dim fp32 reduce order; uncontracted fp32 multiply
from triton.language.extra import libdevice

EXPERTS = {"on": False, "calls": 0, "fallback": 0, "why": {}, "dtypes": {}}
ORIGINAL = {"fn": None}
BF16 = torch.bfloat16


@triton.jit
def _silu_parts(g):
    """torch's CUDA silu forward x / (1 + exp(-x)) and its sigmoid, fp32 (accurate exp, IEEE division)."""
    e = 1.0 + libdevice.exp(-g)
    return tl.math.div_rn(g, e), tl.math.div_rn(1.0, e)


@triton.jit
def _up_act_fwd(BASE, DELTA, H, A, rows, I: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    pr, pc = tl.program_id(0), tl.program_id(1)
    r = (pr * BR + tl.arange(0, BR)).to(tl.int64)
    c = pc * BC + tl.arange(0, BC)
    m = (r < rows)[:, None] & (c < I)[None, :]
    og = r[:, None] * (2 * I) + c[None, :]
    gate = (tl.load(BASE + og, mask=m, other=0.).to(tl.float32)
            + tl.load(DELTA + og, mask=m, other=0.).to(tl.float32)).to(tl.bfloat16)
    up = (tl.load(BASE + og + I, mask=m, other=0.).to(tl.float32)
          + tl.load(DELTA + og + I, mask=m, other=0.).to(tl.float32)).to(tl.bfloat16)
    tl.store(H + og, gate, mask=m)
    tl.store(H + og + I, up, mask=m)
    act, _ = _silu_parts(gate.to(tl.float32))
    a = up.to(tl.float32) * act.to(tl.bfloat16).to(tl.float32)
    tl.store(A + r[:, None] * I + c[None, :], a.to(tl.bfloat16), mask=m)


@triton.jit
def _up_act_bwd(DA, H, DH, rows, I: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    pr, pc = tl.program_id(0), tl.program_id(1)
    r = (pr * BR + tl.arange(0, BR)).to(tl.int64)
    c = pc * BC + tl.arange(0, BC)
    m = (r < rows)[:, None] & (c < I)[None, :]
    og = r[:, None] * (2 * I) + c[None, :]
    g = tl.load(H + og, mask=m, other=0.).to(tl.float32)
    up = tl.load(H + og + I, mask=m, other=0.).to(tl.float32)
    da = tl.load(DA + r[:, None] * I + c[None, :], mask=m, other=0.).to(tl.float32)
    act, s = _silu_parts(g)
    dup = (da * act.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    dact = (da * up).to(tl.bfloat16).to(tl.float32)
    dgate = ((dact * s) * (1.0 + _mul(g, 1.0 - s))).to(tl.bfloat16)
    tl.store(DH + og, dgate, mask=m)
    tl.store(DH + og + I, dup, mask=m)


@triton.jit
def _down_combine_fwd(BASE, DELTA, W, BYTOK, Y, OUT, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
                      BH: tl.constexpr, ROUND: tl.constexpr):
    pt, ph = tl.program_id(0), tl.program_id(1)
    t = (pt * BT + tl.arange(0, BT)).to(tl.int64)
    h = ph * BH + tl.arange(0, BH)
    mt = t < T
    m = mt[:, None] & (h < H)[None, :]
    a0 = tl.zeros([BT, BH], dtype=tl.float32)
    a1, a2, a3 = a0, a0, a0
    for k in tl.static_range(K):  # torch's reduce order over K: four accumulators (k mod 4), then ((a0+a1)+a2)+a3
        p = tl.load(BYTOK + t * K + k, mask=mt, other=0).to(tl.int64)
        off = p[:, None] * H + h[None, :]
        y = (tl.load(BASE + off, mask=m, other=0.).to(tl.float32)
             + tl.load(DELTA + off, mask=m, other=0.).to(tl.float32)).to(tl.bfloat16)
        tl.store(Y + off, y, mask=m)
        w = tl.load(W + p, mask=mt, other=0.).to(tl.float32)
        if ROUND:
            v = (y.to(tl.float32) * w[:, None]).to(tl.bfloat16).to(tl.float32)
        else:
            v = _mul(y.to(tl.float32), w[:, None])
        if k % 4 == 0:
            a0 += v
        elif k % 4 == 1:
            a1 += v
        elif k % 4 == 2:
            a2 += v
        else:
            a3 += v
    tl.store(OUT + t[:, None] * H + h[None, :], (((a0 + a1) + a2) + a3).to(tl.bfloat16), mask=m)


@triton.jit
def _combine_bwd_row(DOUT, GIDX, W, Y, DY, DW, H: tl.constexpr, V: tl.constexpr):
    """One grouped row: dy = BF16(dout[tok] x w); dw = sum_h fp32(dout[tok]) fp32(y) in torch's inner-reduce order
    (rl1_hyper._tree: V-wide slots per lane over 32 V-wide steps, then the halving shuffle tree)."""
    r = tl.program_id(0).to(tl.int64)
    tok = tl.load(GIDX + r).to(tl.int64)
    w = tl.load(W + r).to(tl.float32)
    off = tl.arange(0, 32)[:, None] * V + tl.arange(0, V)[None, :]
    acc = tl.zeros([32, V], dtype=tl.float32)
    for j in range(H // (32 * V)):
        h = j * 32 * V + off
        g = tl.load(DOUT + tok * H + h).to(tl.float32)
        tl.store(DY + r * H + h, (g * w).to(tl.bfloat16))
        acc += _mul(g, tl.load(Y + r * H + h).to(tl.float32))
    tl.store(DW + r, _tree(acc, V).to(DW.dtype.element_ty))


@triton.jit
def _combine_bwd(DOUT, GIDX, W, DY, rows, H: tl.constexpr, BR: tl.constexpr, BH: tl.constexpr):
    pr = tl.program_id(0)
    r = (pr * BR + tl.arange(0, BR)).to(tl.int64)
    mr = r < rows
    tok = tl.load(GIDX + r, mask=mr, other=0).to(tl.int64)
    w = tl.load(W + r, mask=mr, other=0.).to(tl.float32)
    for h0 in range(0, H, BH):
        h = h0 + tl.arange(0, BH)
        m = mr[:, None] & (h < H)[None, :]
        g = tl.load(DOUT + tok[:, None] * H + h[None, :], mask=m, other=0.).to(tl.float32)
        tl.store(DY + r[:, None] * H + h[None, :], (g * w[:, None]).to(tl.bfloat16), mask=m)


@triton.jit
def _gather_sum(SRC, BYTOK, OUT, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr):
    """out[t] = OUT dtype(sum_k fp32(src[bytok[t K + k]])) (``_GatherRows`` backward)."""
    pt, ph = tl.program_id(0), tl.program_id(1)
    t = (pt * BT + tl.arange(0, BT)).to(tl.int64)
    h = ph * BH + tl.arange(0, BH)
    mt = t < T
    m = mt[:, None] & (h < H)[None, :]
    a0 = tl.zeros([BT, BH], dtype=tl.float32)
    a1, a2, a3 = a0, a0, a0
    for k in tl.static_range(K):  # torch's order (see _down_combine_fwd)
        p = tl.load(BYTOK + t * K + k, mask=mt, other=0).to(tl.int64)
        v = tl.load(SRC + p[:, None] * H + h[None, :], mask=m, other=0.).to(tl.float32)
        if k % 4 == 0:
            a0 += v
        elif k % 4 == 1:
            a1 += v
        elif k % 4 == 2:
            a2 += v
        else:
            a3 += v
    tl.store(OUT + t[:, None] * H + h[None, :], (((a0 + a1) + a2) + a3).to(OUT.dtype.element_ty), mask=m)


def _nv():
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    return nvfp4_lora


def _b_c(x, A, B, E, dim1):
    L = _nv()
    return L._b3d_contiguous(B, E, dim1) if L._use_grouped_mm(x) and A.dtype == x.dtype else None


class GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, gidx, by_token):
        ctx.save_for_backward(by_token)
        ctx.T, ctx.dtype = hidden.shape[0], hidden.dtype
        return hidden.to(BF16).index_select(0, gidx)  # = the up projection's BF16 cast of the gathered rows

    @staticmethod
    def backward(ctx, grad):
        (by_token,) = ctx.saved_tensors
        T, H = ctx.T, grad.shape[-1]
        grad = grad.contiguous()
        out = torch.empty((T, H), dtype=ctx.dtype, device=grad.device)
        K = by_token.numel() // max(T, 1)
        if T:
            _gather_sum[(triton.cdiv(T, 16), triton.cdiv(H, 256))](grad, by_token, out, T, H=H, K=K, BT=16, BH=256,
                                                                     num_warps=4)
        return out, None, None


class UpAct(torch.autograd.Function):
    """a = SwiGLU(base_up(x) + LoRA delta) per expert group (``GroupedUpProjLoRA`` + ``gated_activation``)."""

    @staticmethod
    def forward(ctx, x, w1, offsets, A, B, scaling):
        L = _nv()
        E, dim1, dim2 = w1.shape
        B_c = _b_c(x, A, B, E, dim1)
        delta = L._lora_delta_per_group(x, offsets, A, B, scaling, E, dim1, dim2, B_c=B_c)
        base = L.grouped_up_gemm(x, w1, offsets, backend="torch", concat=True)
        rows, I = x.shape[0], dim1 // 2
        h = torch.empty_like(base)
        a = torch.empty((rows, I), dtype=base.dtype, device=base.device)
        if rows:
            _up_act_fwd[(triton.cdiv(rows, 32), triton.cdiv(I, 128))](base, delta, h, a, rows, I=I, BR=32, BC=128,
                                                                      num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, w1, offsets, A, B, B_c, h)
        ctx.scaling = scaling
        return a

    @staticmethod
    def backward(ctx, da):
        x, w1, offsets, A, B, B_c, h = ctx.saved_tensors
        rows, I = h.shape[0], h.shape[1] // 2
        dh = torch.empty_like(h)
        if rows:
            _up_act_bwd[(triton.cdiv(rows, 32), triton.cdiv(I, 128))](da.contiguous(), h, dh, rows, I=I, BR=32, BC=128,
                                                                      num_warps=4, enable_fp_fusion=False)
        dx, dA, dB = _nv()._lora_backward_per_group(dh, x, offsets, A, B, w1, ctx.scaling, B_c=B_c)
        return dx, None, None, dA, dB, None


class DownCombine(torch.autograd.Function):
    """out[t] = sum_k w_k (base_down(a) + LoRA delta)[row of (t, k)] (``GroupedDownProjLoRA`` + ``_CombineByGather``)."""

    @staticmethod
    def forward(ctx, a, w2, offsets, A, B, scaling, gidx, wg, by_token, T):
        L = _nv()
        E, dim1, dim2 = w2.shape
        B_c = _b_c(a, A, B, E, dim1)
        delta = L._lora_delta_per_group(a, offsets, A, B, scaling, E, dim1, dim2, B_c=B_c)
        base = L.grouped_down_gemm(a, w2, offsets, backend="torch")
        H, K = dim1, a.shape[0] // max(T, 1)
        y = torch.empty_like(base)
        out = torch.empty((T, H), dtype=base.dtype, device=base.device)
        if T:
            _down_combine_fwd[(triton.cdiv(T, 16), triton.cdiv(H, 256))](base, delta, wg, by_token, y, out, T, H=H,
                                                                         K=K, BT=16, BH=256,
                                                                         ROUND=wg.dtype == BF16, num_warps=4,
                                                                         enable_fp_fusion=False)
        ctx.save_for_backward(a, w2, offsets, A, B, B_c, y, gidx, wg)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, dout):
        a, w2, offsets, A, B, B_c, y, gidx, wg = ctx.saved_tensors
        rows, H = y.shape
        dout = dout.contiguous()
        dy = torch.empty_like(y)
        if rows and H % 128 == 0 and rows >= 16:  # torch's reduce layout for >= 16 rows (outputs)
            dw = torch.empty_like(wg)
            _combine_bwd_row[(rows,)](dout, gidx, wg, y, dy, dw, H=H, V=4, num_warps=1, enable_fp_fusion=False)
        else:
            if rows:
                _combine_bwd[(triton.cdiv(rows, 16),)](dout, gidx, wg, dy, rows, H=H, BR=16, BH=256, num_warps=4)
            # router-weight grad by _CombineByGather's own ops (an fp32 result when wg is)
            dw = (dout.index_select(0, gidx).float() * y.float()).sum(-1).to(wg.dtype)
        da, dA, dB = _nv()._lora_backward_per_group(dy, a, offsets, A, B, w2, ctx.scaling, B_c=B_c)
        return da, None, None, dA, dB, None, None, dw, None, None


def why_not(hidden, w1, b1, w2, b2, lora1, lora2, act, backend, concat, limit, gated):
    L = _nv()
    if backend != "dequant" or not hidden.is_cuda:
        return f"backend {backend}"
    if b1 is not None or b2 is not None or limit is not None or not concat or not gated or act not in ("silu", "swiglu"):
        return "layout"
    if lora1 is None or lora2 is None:
        return "no lora"
    if L.merge_aware_enabled():
        return "merge aware"
    if hidden.dtype == torch.float32 and not torch.is_autocast_enabled(hidden.device.type):
        return "fp32 without autocast"  # the BF16 cast of the grouped rows is the trainer patch's, under autocast
    if hidden.dtype not in (BF16, torch.float32) or any(t.dtype != BF16 for t in (*lora1, *lora2)):
        return f"dtype {str(hidden.dtype)[6:]}/{str(lora1[0].dtype)[6:]}"
    return None


# Expert compaction (MFU-1): the hero trainer serves the t37 mask (192 of 512 experts never routed).
# compact_experts cuts each layer's NVFP4 base (qdata, block scales, per-expert scales) to the kept experts and frees
# the 512-wide copies; routes stay global ids (router mask, R3 tables) and are mapped to compact ids here (a
# monotonic map: same sort, same groups); the LoRA parameters keep the 512 layout and each call slices the kept
# experts' factors (A rows e*r.., B columns j*E+e), so their grads scatter back with zeros on the masked experts.
COMPACT = {"on": False, "by": {}, "layers": 0, "kept": 0, "freed_bytes": 0, "calls": 0}


def _cut_lora(lora, kept, E):
    """(A [r*E, d2], B [d1, r*E]) -> the kept experts' factors (A [r*Ek, d2], B [d1, r*Ek]), differentiable."""
    A, B = lora
    r = A.shape[0] // E
    Ak = A.reshape(E, r, A.shape[1]).index_select(0, kept).reshape(-1, A.shape[1])
    Bk = B.reshape(B.shape[0], r, E).index_select(2, kept).reshape(B.shape[0], -1)
    return Ak, Bk


def compact_route(top_k_index, remap):
    """Global expert ids -> compact ids (-1 for a pruned expert: asserted on the card, no host sync)."""
    out = remap.index_select(0, top_k_index.reshape(-1)).reshape(top_k_index.shape)
    torch._assert_async((out >= 0).all(), "expert compaction: a pruned expert was routed")
    return out


def _cut_nvfp4(w, kept):
    """Cut an NVFP4 [E, N, K/2] weight to the kept experts in place (same object: the module's references hold);
    returns the bytes freed."""
    before = w.qdata.numel() * w.qdata.element_size() + w.scale.numel() * w.scale.element_size()
    w.qdata = w.qdata.index_select(0, kept).contiguous()
    w.scale = w.scale.index_select(0, kept).contiguous()
    pts = w.per_tensor_scale
    if pts is not None and pts.numel() > 1:  # per expert ([E, 1, 1] in torchao's layout): the kept rows, same rank
        w.per_tensor_scale = pts.index_select(0, kept).contiguous()
    return before - (w.qdata.numel() * w.qdata.element_size() + w.scale.numel() * w.scale.element_size())


def compact_experts(model, pruned):
    """``pruned`` {layer: [global expert ids]} (arc-expert-prune-v1). Cuts every masked layer's routed-expert base
    weights; returns COMPACT's summary. Requires the fused path (EXPERTS["on"]): any other expert path fails loudly."""
    import re

    if COMPACT["layers"]:
        return summary()
    done = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen4ExpTextExperts":
            continue
        found = re.search(r"layers\.(\d+)\.mlp", name)
        layer = int(found.group(1)) if found else None
        if layer not in pruned or not pruned[layer]:
            continue
        L = _nv()
        from axolotl.integrations.kernels.libs.sonicmoe import experts as ax_experts

        w1, _, w2, _, _, _ = ax_experts._resolve_weights_and_lora(module)  # the objects the forward passes on
        if not (L.is_nvfp4_param(w1) and L.is_nvfp4_param(w2)):
            raise RuntimeError(f"expert compaction: {name} base is not NVFP4 ({type(w1).__name__})")
        E = int(w1.qdata.shape[0])
        drop = set(int(e) for e in pruned[layer])
        kept_ids = [e for e in range(E) if e not in drop]
        dev = w1.qdata.device
        kept = torch.tensor(kept_ids, dtype=torch.long, device=dev)
        remap = torch.full((E,), -1, dtype=torch.long, device=dev)
        remap[kept] = torch.arange(len(kept_ids), dtype=torch.long, device=dev)
        freed = _cut_nvfp4(w1, kept) + _cut_nvfp4(w2, kept)
        info = {"E": E, "kept": kept, "remap": remap, "layer": layer}
        COMPACT["by"][id(w1)] = info
        COMPACT["by"][id(w2)] = info
        COMPACT["freed_bytes"] += freed
        COMPACT["kept"] = len(kept_ids)
        done += 1
    if not done:
        raise RuntimeError("expert compaction: no masked expert layer found")
    COMPACT["layers"], COMPACT["on"] = done, True
    torch.cuda.empty_cache()
    return summary()


def summary():
    return {"layers": COMPACT["layers"], "kept": COMPACT["kept"], "freed_gib": round(COMPACT["freed_bytes"] / 2 ** 30, 2),
            "calls": COMPACT["calls"]}


def moe_forward(hidden_states, top_k_index, top_k_weights, w1, b1, w2, b2, lora1, lora2, num_experts, *, act,
                backend, concat, scaling1, scaling2, limit=None, gated=True):
    """Drop-in for ``nvfp4_lora.grouped_moe_reference_forward`` (backend ``dequant``)."""
    if EXPERTS["on"] and lora1 is not None and lora2 is not None and torch.is_autocast_enabled(hidden_states.device.type):
        # the trainer's grouped-LoRA patch (rl1_trainer.grouped_expert_lora): BF16 casts of the fp32 factors
        lora1 = tuple(t.to(BF16) for t in lora1)
        lora2 = tuple(t.to(BF16) for t in lora2)
    cut = COMPACT["by"].get(id(w1)) if COMPACT["on"] else None
    why = why_not(hidden_states, w1, b1, w2, b2, lora1, lora2, act, backend, concat, limit, gated) \
        if EXPERTS["on"] else "off"
    if cut is not None:
        if why is not None:
            raise RuntimeError(f"expert compaction needs the fused expert path ({why})")
        if int(num_experts) != cut["E"] or COMPACT["by"].get(id(w2)) is not cut:
            raise RuntimeError("expert compaction: weights and layer mismatch")
        COMPACT["calls"] += 1
        top_k_index = compact_route(top_k_index, cut["remap"])
        lora1 = _cut_lora(lora1, cut["kept"], cut["E"])
        lora2 = _cut_lora(lora2, cut["kept"], cut["E"])
        num_experts = int(cut["kept"].numel())
    if why is not None:
        if EXPERTS["on"]:
            EXPERTS["fallback"] += 1
            EXPERTS["why"][why] = EXPERTS["why"].get(why, 0) + 1
        return ORIGINAL["fn"](hidden_states, top_k_index, top_k_weights, w1, b1, w2, b2, lora1, lora2, num_experts,
                              act=act, backend=backend, concat=concat, scaling1=scaling1, scaling2=scaling2,
                              limit=limit, gated=gated)
    L = _nv()
    EXPERTS["calls"] += 1
    key = str(hidden_states.dtype)[6:]
    EXPERTS["dtypes"][key] = EXPERTS["dtypes"].get(key, 0) + 1
    w1d, w2d = L.dequantize_expert_weight(w1), L.dequantize_expert_weight(w2)
    T, K = top_k_index.shape
    # route_and_group's arithmetic (sort, gather order, offsets) with this module's gather
    flat_expert = top_k_index.reshape(-1)
    wflat = top_k_weights.reshape(-1).to(hidden_states.dtype)
    token_ids = torch.arange(T, device=hidden_states.device).repeat_interleave(K)
    sorted_experts, order = torch.sort(flat_expert, stable=True)
    gidx = token_ids[order]
    wg = wflat[order]
    by_token = torch.sort(gidx, stable=True).indices
    x = GatherRows.apply(hidden_states, gidx, by_token)
    offsets = torch.searchsorted(sorted_experts, torch.arange(num_experts + 1, device=hidden_states.device,
                                                              dtype=sorted_experts.dtype))
    a = UpAct.apply(x, w1d, offsets, lora1[0], lora1[1], scaling1)
    return DownCombine.apply(a, w2d, offsets, lora2[0], lora2[1], scaling2, gidx, wg, by_token, T)


def install():
    """Route ``grouped_moe_reference_forward`` here (idempotent; re-applied after native_experts toggles)."""
    L = _nv()
    if L.grouped_moe_reference_forward is moe_forward:
        return
    ORIGINAL["fn"] = L.grouped_moe_reference_forward
    L.grouped_moe_reference_forward = moe_forward
