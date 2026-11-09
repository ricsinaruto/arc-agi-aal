"""Quantization-aware training for the export grid (config ``fake_quant``): the routed experts of the trunk run
forward through export.py's int4 (AutoRound sym, g128, the export's scale search), with a straight-through gradient
into the BF16 master weights.

Scales: ``refresh`` computes every layer's fp16 group scales from the current weights (the export's search; once at
start and after each optimizer step, when the weights change), ~1/64 of the expert bytes. Each forward (prefix chunks,
window, checkpoint recompute) rounds the weights against them: w + (dequant(round(w / s)) - w).detach(), cast to the
weights' dtype; dequant is bit-equal to the export's codes x scales (``export.int4_dequant``). The MTP head and the
shared experts stay BF16, as exported. Distillation losses: ``kl_topk``.

``QAT["grid"] = "int3served"`` (config ``qat_grid``): the served int3 experts instead (arena.expert_3bit_math.rtn with
expert_3bit_overlay's SEARCH, group 16): signed E4M3 group scales x a per-expert fp32 scale_2 (amax / (4 x 448); the
fused gate_up shares one, computed on the concatenation as convert_shard does), dequant bit-equal to rtn's.

``DENSE`` (config ``dense_int8_train``): the served W8A16 dense families (dense_kernels_runtime.INT8_FAMILIES: lm_head,
GDN in_proj_qkv / in_proj_z / out_proj, QSA q / k / v / o, the MTP draft's q / k / v / o too) forward through quantize_rows' per-row INT8
round trip with a straight-through gradient (``install_dense``; the packed GDN fast path applies it in gdn.py)."""
import torch

import export

QAT = {"on": False, "group": 128, "zero": 8, "levels": 16, "search": 20, "chunk": 8, "calls": 0, "fused": None,
       "grid": "int4", "zero_experts": 0}  # zero_experts: all-zero experts at the last int3served refresh
SCALES = {}  # id(experts module) -> {"gate_up_proj": fp16 [E, in/g, out] | int3served (s2, sc), "down_proj": ...}
NAMES = ("gate_up_proj", "down_proj")
GRIDS = ("int4", "int3served")
# int3served: expert_3bit_math FORMATS["int3"], FP8_MAX, BLOCK; expert_3bit_overlay.SEARCH
LEVELS3 = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
FP8_MAX = 448.0
BLOCK3 = 16
SEARCH3 = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
_LEVELS = {}


def levels3(device):
    key = str(device)
    if key not in _LEVELS:
        _LEVELS[key] = torch.tensor(LEVELS3, dtype=torch.float32, device=device)
    return _LEVELS[key]


def nearest(values, levels):
    """expert_3bit_math.nearest: the level of ``searchsorted(mids, v, right=True)`` (a tie takes the upper level)."""
    mids = (levels[1:] + levels[:-1]) * 0.5
    return levels[torch.searchsorted(mids, values.contiguous(), right=True)]


def scale2_of3(w):
    """expert_3bit_math.scale2_of(w, "int3"): per-expert fp32 amax / (4 x 448) of w [E, N, K]. An all-zero expert
    (0 there: its rtn is NaN) takes 1, so its grid is 0."""
    c = QAT["chunk"]  # fp32 temporaries a few experts at a time (each expert's amax is its own)
    s2 = torch.cat([w[a:a + c].float().abs().amax(dim=(-2, -1)) for a in range(0, w.shape[0], c)]) \
        * (1.0 / (abs(LEVELS3[0]) * FP8_MAX))
    zero = s2 == 0
    QAT["zero_experts"] += int(zero.sum())
    return torch.where(zero, torch.ones_like(s2), s2)


def group_scales3(w, s2):
    """expert_3bit_math.group_scales(w, s2, "int3", 16, SEARCH3), op for op: signed E4M3 group scales [E, N, K/16]
    (float32 values) that put each group's largest magnitude on -4, the multiplier with the least squared error."""
    levels = levels3(w.device)
    lmax = -float(levels[0])
    e, n, k = w.shape
    g = w.float().reshape(e, n, k // BLOCK3, BLOCK3)
    s2 = s2.float().reshape(e, 1, 1)
    idx = g.abs().argmax(-1, keepdim=True)
    top = g.gather(-1, idx).squeeze(-1)
    sign = torch.where(top > 0, -1.0, 1.0)
    amax = top.abs()
    best, best_err = None, None
    for m in SEARCH3:
        mag = (amax / (lmax * m) / s2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
        sc = mag * sign
        div = torch.where(mag == 0, torch.ones_like(mag), sc) * s2
        q = nearest(g / div.unsqueeze(-1), levels) * div.unsqueeze(-1)
        err = (q - g).pow(2).sum(-1)
        if best is None:
            best, best_err = sc, err
        else:
            take = err < best_err
            best = torch.where(take, sc, best)
            best_err = torch.minimum(err, best_err)
    return best


def scales3_of(w):
    """int3served scales of w [E, N, K]: (scale_2 fp32 [E], signed group scales [E, N, K/16] in BF16, exact: E4M3
    values), ``chunk`` experts at a time (scale_2 over each whole expert, gate and up together)."""
    with torch.no_grad():
        s2 = scale2_of3(w)
        c = QAT["chunk"]
        sc = torch.cat([group_scales3(w[a:a + c], s2[a:a + c]).to(torch.bfloat16) for a in range(0, w.shape[0], c)])
    return s2, sc


def fake3(w, scales):
    """The int3served dequant of w [E, N, K] (fp32), bit-equal to expert_3bit_math.rtn's: nearest(w / d) x d, d = the
    signed group scale (0 -> 1) x scale_2 (expert_3bit_math.divisor; broadcast over the group, not repeated)."""
    d = _div3(scales)
    return (codes3(w, scales, d) * d).reshape(w.shape)


def _div3(scales):
    """[E, N, K/16, 1] fp32 divisors: (signed group scale, 0 -> 1) x scale_2."""
    s2, sc = scales
    sc = sc.float()
    return (torch.where(sc == 0, torch.ones_like(sc), sc) * s2.float().reshape(-1, 1, 1)).unsqueeze(-1)


def codes3(w, scales, d=None):
    """The int3 levels (fp32) of w [E, N, K] on (scale_2, scales), grouped [E, N, K/16, 16]."""
    e, n, k = w.shape
    d = _div3(scales) if d is None else d
    return nearest(w.float().reshape(e, n, k // BLOCK3, BLOCK3) / d, levels3(w.device))


def _part(s, a, b):
    return (s[0][a:b], s[1][a:b]) if isinstance(s, tuple) else s[a:b]


def scales_of(w):
    """The grid's scales of w [E, out, in]: int4 fp16 group scales, ``chunk`` experts at a time; or int3served's."""
    if QAT["grid"] == "int3served":
        return scales3_of(w)
    parts = []
    with torch.no_grad():
        for a in range(0, w.shape[0], QAT["chunk"]):
            parts.append(export.int4_codes(w[a:a + QAT["chunk"]], QAT["group"], QAT["zero"], QAT["search"], QAT["levels"])[1])
    return torch.cat(parts)


def fake(w, scale16):
    """The export grid's dequantized w [E, out, in] (fp32) against fixed fp16 scales [E, in/g, out] (int3served:
    ``fake3`` against its (scale_2, scales))."""
    if QAT["grid"] == "int3served":
        return fake3(w, scale16)
    z, g = QAT["zero"], QAT["group"]
    wt = w.float().transpose(-1, -2)
    *lead, inp, out = wt.shape
    q = (wt.reshape(*lead, inp // g, g, out) / scale16.float().unsqueeze(-2)).round().clamp(-z, QAT["levels"] - 1 - z)
    q = q.nan_to_num(0.0)  # an all-zero group: fp16 scale 0, 0/0 (export: any code x 0 = 0)
    return export.int4_dequant(q.reshape(*lead, inp, out), scale16, g)


def grid(w, scale16):
    """The grid's weights in w's dtype (eager: fp32 temporaries a few experts at a time)."""
    c = QAT["chunk"]
    return torch.cat([fake(w[a:a + c], _part(scale16, a, a + c)).to(w.dtype) for a in range(0, w.shape[0], c)])


def _grid_one(w, scale16):
    return fake(w, scale16).to(w.dtype)


class _STE(torch.autograd.Function):
    """Forward: exactly the grid's weights (one fused pass when ``QAT["fused"]``); backward: identity into w."""

    @staticmethod
    def forward(ctx, w, scale16):
        with torch.no_grad():
            return QAT["fused"](w, scale16) if QAT["fused"] is not None else grid(w, scale16)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def ste(w, scale16):
    """Forward: the grid's weights in w's dtype, bit-equal to ``fake`` (26 Oct: was w + (q - w).detach(), which in
    BF16 can land an ulp off q, and three full-size passes); backward: identity into w."""
    return _STE.apply(w, scale16)


def _triton_grid():
    """``fake(w, s).to(w.dtype)`` as one Triton pass over [E, out, in]: IEEE division (``div_rn``; Triton's ``/`` is
    approximate, which is why the torch.compile'd grid missed the bit check on the real weights, util-v3), round half
    to even, NaN (0/0) -> 0 before the clamp as torch's clamp + nan_to_num, round-to-nearest-even cast."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def kernel(w_ptr, s_ptr, o_ptr, OUT, IN, G: tl.constexpr, Z: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)  # e * OUT + o
        e, o = row // OUT, row % OUT
        for start in range(0, IN, BLOCK):
            i = start + tl.arange(0, BLOCK)
            m = i < IN
            w = tl.load(w_ptr + row * IN + i, mask=m, other=0.0).to(tl.float32)
            sc = tl.load(s_ptr + e * (IN // G) * OUT + (i // G) * OUT + o, mask=m, other=1.0).to(tl.float32)
            q = libdevice.rint(tl.math.div_rn(w, sc))
            nan = q != q
            q = tl.minimum(tl.maximum(q, -Z * 1.0), (15 - Z) * 1.0)
            q = tl.where(nan, 0.0, q)
            tl.store(o_ptr + row * IN + i, (q * sc).to(o_ptr.dtype.element_ty), mask=m)

    def run(w, scale16):
        w = w.contiguous()
        *lead, out, inp = w.shape
        rows = w.numel() // inp
        y = torch.empty_like(w)
        kernel[(rows,)](w, scale16.contiguous(), y, out, inp, G=QAT["group"], Z=QAT["zero"], BLOCK=1024)
        return y

    return run


def _triton_grid3():
    """``fake3(w, s).to(w.dtype)`` as one Triton pass over [E, N, K]: d = (sc or 1) x scale_2 (one rounding, as
    torch), v = div_rn(w, d) (IEEE, as torch's division), the level as the number of midpoints <= v (searchsorted
    right=True; NaN -> the last level, as torch's searchsorted), times d, round-to-nearest-even cast."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(w_ptr, sc_ptr, s2_ptr, o_ptr, OUT, IN: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)  # e * OUT + o
        s2 = tl.load(s2_ptr + row // OUT)
        for start in range(0, IN, BLOCK):
            i = start + tl.arange(0, BLOCK)
            m = i < IN
            w = tl.load(w_ptr + row * IN + i, mask=m, other=0.0).to(tl.float32)
            sc = tl.load(sc_ptr + row * (IN // G) + i // G, mask=m, other=1.0).to(tl.float32)
            d = tl.where(sc == 0.0, 1.0, sc) * s2
            v = tl.math.div_rn(w, d)
            q = ((v >= -3.5).to(tl.float32) + (v >= -2.5).to(tl.float32) + (v >= -1.5).to(tl.float32)
                 + (v >= -0.5).to(tl.float32) + (v >= 0.5).to(tl.float32) + (v >= 1.5).to(tl.float32)
                 + (v >= 2.5).to(tl.float32)) - 4.0
            q = tl.where(v != v, 3.0, q)
            tl.store(o_ptr + row * IN + i, (q * d).to(o_ptr.dtype.element_ty), mask=m)

    def run(w, scales):
        s2, sc = scales
        w = w.contiguous()
        *lead, out, inp = w.shape
        y = torch.empty_like(w)
        kernel[(w.numel() // inp,)](w, sc.contiguous(), s2.float().contiguous(), y, out, IN=inp, G=BLOCK3, BLOCK=1024)
        return y

    return run


def fuse(samples=()):
    """``fake`` in one pass over a whole expert tensor (no fp32 temporaries; util-v1's profile: the eager chunks were
    ~5-7 s of a ~20 s example): the Triton grid (int4 or int3served), else torch.compile, the first bit-equal to the
    eager grid on every (w, scales) sample is kept. Returns its name, or False (eager stays)."""
    tri = _triton_grid3 if QAT["grid"] == "int3served" else _triton_grid
    makers = [("triton", tri)] if all(w.is_cuda for w, _ in samples) and samples else []
    makers.append(("compile", lambda: torch.compile(_grid_one, dynamic=False)))
    for name, make in makers:
        try:
            fused = make()
            with torch.no_grad():
                if all(torch.equal(fused(w, s16), grid(w, s16)) for w, s16 in samples):
                    QAT["fused"] = fused
                    return name
        except Exception:  # noqa: BLE001 - a kernel that does not build is not used
            continue
    return False


def routed(model):
    """The trunk's routed-expert modules (not the MTP head's)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    return [mod for name, mod in model.named_modules() if isinstance(mod, m.Qwen4ExpTextExperts)
            and ".language_model.layers." in f".{name}"]


def refresh(model, drift=False):
    """New scales for every routed expert. ``drift``: also measure what the stale scales cost (qat_refresh_every > 1):
    the share of int4 codes that differ against the stale vs the fresh scales (each flip is one code step off the
    export grid; quantization error per code is ~step^2/12, so a flip share f adds ~12f to the grid's error energy),
    the share of fp16 scales that changed, and the largest relative scale change. One sync at the end."""
    QAT["zero_experts"] = 0
    if not drift:
        for mod in routed(model):
            SCALES[id(mod)] = {n: scales_of(getattr(mod, n).detach()) for n in NAMES}
        return None
    if QAT["grid"] == "int3served":
        return _drift3(model)
    z, gs, c = QAT["zero"], QAT["group"], QAT["chunk"]
    flips = codes = moved = count = rel = None
    for mod in routed(model):
        old, new = SCALES.get(id(mod)), {}
        for n in NAMES:
            w, parts = getattr(mod, n).detach(), []
            with torch.no_grad():
                for a in range(0, w.shape[0], c):
                    q, s16 = export.int4_codes(w[a:a + c], gs, z, QAT["search"], QAT["levels"])
                    parts.append(s16)
                    if old is None:
                        continue
                    so = old[n][a:a + c]
                    wt = w[a:a + c].float().transpose(-1, -2)
                    *lead, inp, out = wt.shape
                    qo = (wt.reshape(*lead, inp // gs, gs, out) / so.float().unsqueeze(-2)).round().clamp(-z, QAT["levels"] - 1 - z)
                    f = (qo.reshape(q.shape).nan_to_num(0.0) != q.nan_to_num(0.0)).sum()
                    m = (so != s16).sum()
                    r = ((s16.float() - so.float()).abs() / so.float().abs().clamp_min(1e-10)).max()
                    flips = f if flips is None else flips + f
                    moved = m if moved is None else moved + m
                    rel = r if rel is None else torch.maximum(rel, r)
                    codes = (codes or 0) + q.numel()
                    count = (count or 0) + s16.numel()
            new[n] = torch.cat(parts)
        SCALES[id(mod)] = new
    if flips is None:
        return None
    return {"code_flip_share": flips.item() / codes, "scale_changed_share": moved.item() / count,
            "scale_rel_max": round(rel.item(), 6), "codes": codes, "scales": count}


def _drift3(model):
    """``refresh(drift=True)`` on the int3served grid: codes (levels) that differ between the stale and the fresh
    scales, group scales (and scale_2s) that changed, the largest relative group-scale change."""
    c = QAT["chunk"]
    flips = codes = moved = count = rel = None
    for mod in routed(model):
        old, new = SCALES.get(id(mod)), {}
        for n in NAMES:
            w = getattr(mod, n).detach()
            new[n] = fresh = scales3_of(w)
            if old is None:
                continue
            stale = old[n]
            with torch.no_grad():
                for a in range(0, w.shape[0], c):
                    qa = codes3(w[a:a + c], _part(stale, a, a + c))
                    f = (qa != codes3(w[a:a + c], _part(fresh, a, a + c))).sum()
                    flips = f if flips is None else flips + f
                    codes = (codes or 0) + qa.numel()
            m = (stale[1] != fresh[1]).sum() + (stale[0] != fresh[0]).sum()
            r = ((fresh[1].float() - stale[1].float()).abs() / stale[1].float().abs().clamp_min(1e-10)).max()
            moved = m if moved is None else moved + m
            rel = r if rel is None else torch.maximum(rel, r)
            count = (count or 0) + fresh[1].numel() + fresh[0].numel()
        SCALES[id(mod)] = new
    if flips is None:
        return None
    return {"code_flip_share": flips.item() / codes, "scale_changed_share": moved.item() / count,
            "scale_rel_max": round(rel.item(), 6), "codes": codes, "scales": count}


def install(model):
    """Wrap Qwen4ExpTextExperts.forward once: the module's weights read as their fake-quant view during the call."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextExperts
    if not getattr(cls, "_sft_qat", False):
        original = cls.forward

        def forward(self, *args, **kwargs):
            scales = SCALES.get(id(self)) if QAT["on"] else None
            if scales is None:
                return original(self, *args, **kwargs)
            QAT["calls"] += 1
            kept = {n: self._parameters[n] for n in NAMES}
            try:
                for n in NAMES:
                    self._parameters[n] = ste(kept[n], scales[n])
                return original(self, *args, **kwargs)
            finally:
                self._parameters.update(kept)

        cls.forward = forward
        cls._sft_qat = True
        cls._sft_qat_original = original
    QAT["on"] = True
    refresh(model)


# --- served INT8 dense families (dense_int8_train) ------------------------------------------------------------------
DENSE = {"on": False, "calls": 0}
INT8_DENSE = (".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".self_attn.q_proj", ".self_attn.k_proj",
              ".self_attn.v_proj", ".linear_attn.out_proj", ".self_attn.o_proj")  # + lm_head (fused there, per-row here)


def int8_rows(w, chunk_rows=8192):
    """dense_kernels_runtime.quantize_rows round trip in w's dtype, as its dequant kernel: scale = amax(|row|) / 127
    (>= 1e-12), q = round(w / scale) in [-127, 127], (q x scale) cast once."""
    out = torch.empty_like(w)
    with torch.no_grad():
        for a in range(0, w.shape[0], chunk_rows):
            x = w[a:a + chunk_rows].float()
            scale = (x.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
            out[a:a + chunk_rows] = (torch.round(x / scale[:, None]).clamp_(-127, 127) * scale[:, None]).to(w.dtype)
    return out


class _Int8STE(torch.autograd.Function):
    """Forward: ``int8_rows(w)``; backward: identity into w."""

    @staticmethod
    def forward(ctx, w):
        return int8_rows(w)

    @staticmethod
    def backward(ctx, grad):
        return grad


def int8_ste(w):
    DENSE["calls"] += 1
    return _Int8STE.apply(w)


def dense_modules(model):
    """The served INT8 families' trunk Linear modules (no MTP, no vision tower): [(name, module)]; lm_head is applied
    to its weight in train.example_loss."""
    return [(n, m) for n, m in model.named_modules() if not n.startswith("mtp") and n.endswith(INT8_DENSE)
            and ".language_model.layers." in f".{n}"]


def wrap_int8(mod):
    """``mod`` (a plain bias-free or biased nn.Linear) forwards through ``int8_ste(weight)`` while ``DENSE["on"]``."""
    if mod.__dict__.get("_sft_int8"):
        return
    if type(mod) is not torch.nn.Linear:
        raise TypeError(f"dense INT8: {type(mod).__name__} is not an nn.Linear")

    def forward(x):
        w = int8_ste(mod.weight) if DENSE["on"] else mod.weight
        return torch.nn.functional.linear(x, w, mod.bias)

    mod.forward = forward
    mod._sft_int8 = True


def mtp_dense_modules(head):
    """The MTP draft's attention projections: serving's dense_kernels_runtime.enable runs on the draft model too and
    matches families by name suffix and shape, so its qkv_proj / o_proj are INT8 (its lm_head is not: the draft's
    vocab is another shape)."""
    return [("mtp." + n, m) for n, m in head.named_modules()
            if n.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"))]


def install_dense(model, head=None):
    """Every trunk module of the served INT8 families, and the MTP head's q / k / v / o, forwards on its INT8 round
    trip (straight-through); returns their names. The packed GDN fast path (gdn.py) reads ``DENSE["on"]`` and the
    wrapped flag itself."""
    mods = dense_modules(model) + (mtp_dense_modules(head) if head is not None else [])
    for _, mod in mods:
        wrap_int8(mod)
    DENSE["on"] = True
    return [n for n, _ in mods]


def kl_trunc(logits, ids, lp, target, k=20):
    """The serving screen's truncated KL (throughput fp8kl.py), summed over rows: the teacher's top-``k`` (of its
    stored top ids ``ids``/``lp``, unrenormalised) against the student's log-probs where the token is in the student's
    top-``k`` or is the row's actual token ``target`` (what vLLM's prompt_logprobs returns), else the student's k-th
    log-prob. Also the rows whose argmax agrees (teacher's stored top-1 vs the student's)."""
    q = torch.log_softmax(logits.float(), -1)
    tv, ti = lp.float().topk(k, -1)
    tid = ids.long().gather(-1, ti)
    sv, sid = q.topk(k, -1)
    seen = (tid.unsqueeze(-1) == sid.unsqueeze(-2)).any(-1) | (tid == target.long().unsqueeze(-1))
    qu = torch.where(seen, q.gather(-1, tid), sv[:, -1:])
    return (tv.exp() * (tv - qu)).sum(), (tid[:, 0] == sid[:, 0]).sum()


def kl_rows(logits, ids, lp, target):
    """``kl_topk``'s terms per row (diagnostics): head KL (the teacher's stored top-k), tail-bucket KL, the teacher's
    and student's mass outside those ids, whether the realised token ``target`` is in the teacher's top-k, and whether
    the student's argmax is the teacher's top-1."""
    q = torch.log_softmax(logits.float(), -1)
    lp, ids = lp.float(), ids.long()
    qk = q.gather(-1, ids)
    head = (lp.exp() * (lp - qk)).sum(-1)
    p_tail = (1 - lp.exp().sum(-1)).clamp_min(0)
    q_tail = (1 - qk.exp().sum(-1)).clamp_min(1e-30)
    tail = torch.where(p_tail > 0, p_tail * (p_tail.clamp_min(1e-30).log() - q_tail.log()), torch.zeros_like(p_tail))
    top1 = ids.gather(-1, lp.argmax(-1, keepdim=True))[:, 0]
    return {"head": head, "tail": tail, "p_tail": p_tail, "q_tail": q_tail,
            "in_topk": (ids == target.long().unsqueeze(-1)).any(-1), "agree": q.argmax(-1) == top1}


def kl_topk(logits, ids, lp, w=None):
    """Forward KL(teacher || student) summed over rows, the teacher as its top-k log-probs ``lp`` [n, k] at token ids
    ``ids`` [n, k] plus one tail bucket (the rest of its mass); the student's tail is the rest of its own mass. With
    k = vocab it is the full KL. ``logits`` [n, vocab] (student, any float dtype). ``w`` [n]: per-row weights."""
    q = torch.log_softmax(logits.float(), -1)
    lp = lp.float()
    qk = q.gather(-1, ids.long())
    p_tail = (1 - lp.exp().sum(-1)).clamp_min(0)
    q_tail = (1 - qk.exp().sum(-1)).clamp_min(1e-30)
    tail = torch.where(p_tail > 0, p_tail * (p_tail.clamp_min(1e-30).log() - q_tail.log()), torch.zeros_like(p_tail))
    if w is not None:
        return (((lp.exp() * (lp - qk)).sum(-1) + tail) * w.float()).sum()
    head = (lp.exp() * (lp - qk)).sum()
    return head + tail.sum()
