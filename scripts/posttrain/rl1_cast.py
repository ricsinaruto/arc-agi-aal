"""MFU-1 item 2 (24 Oct): one BF16 copy per fp32 activation under autocast, shared by every Linear that reads it.

Autocast casts an fp32 input once per ``F.linear`` call, so an fp32 activation read by q/k/v (+ each LoRA A) is cast
N times (m1v2 profile: 0.58 s of autocast casts in Linears per 45 s of kernel time). With CAST["on"] the first Linear
casts it (the same ``.to(bfloat16)`` autocast makes) and later ones reuse that copy. Each reader still gets its own
autograd node whose backward is ``grad.to(fp32)``, created where autocast's ToCopyBackward was, so the backward
graph (and the fp32 accumulation order at the activation) is unchanged: forward, loss and grads are bitwise equal.
Also PEFT's cast of a BF16 input to the fp32 LoRA A dtype is skipped under BF16 autocast: lora_A's autocast cast
takes it straight back (BF16 -> fp32 -> BF16 is the identity, both ways in the backward).
"""

import weakref

import torch

CAST = {"on": False, "made": 0, "reused": 0}
ORIGINAL = {"linear": None, "peft_cast": None}
_CACHE = {}  # id(x) -> (weakref(x), x._version, bf16 copy)


class _Reuse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.dtype = x.dtype
        return y.view_as(y)

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.dtype), None


def _drop(key):
    return lambda _ref: _CACHE.pop(key, None)


def bf16(x):
    """x's autocast BF16 copy, made once per (tensor, version); a fresh autograd node per call."""
    key = id(x)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is x and hit[1] == x._version:
        CAST["reused"] += 1
        y = hit[2]
    else:
        with torch.no_grad():
            y = x.to(torch.bfloat16)
        _CACHE[key] = (weakref.ref(x, _drop(key)), x._version, y)
        CAST["made"] += 1
    return _Reuse.apply(x, y) if x.requires_grad and torch.is_grad_enabled() else y


def linear(x, weight, bias=None):
    if CAST["on"] and x.dtype is torch.float32 and _autocast_bf16(x):
        x = bf16(x)
    return ORIGINAL["linear"](x, weight, bias)


def _autocast_bf16(x):
    return torch.is_autocast_enabled(x.device.type) and torch.get_autocast_dtype(x.device.type) is torch.bfloat16


def peft_cast(self, x, dtype):
    if CAST["on"] and x is not None and x.dtype is torch.bfloat16 and dtype is torch.float32 and _autocast_bf16(x):
        CAST["peft_skipped"] = CAST.get("peft_skipped", 0) + 1
        return x
    return ORIGINAL["peft_cast"](self, x, dtype)


def bound(model):
    """The cache is emptied after each decoder layer: a copy never outlives the layer that made it (an fp32 tensor
    kept for the backward, e.g. a checkpoint boundary, would otherwise keep its BF16 copy too)."""
    if getattr(model, "_rl1_cast_bound", False):
        return
    for module in model.modules():
        if type(module).__name__.endswith("DecoderLayer"):
            module.register_forward_hook(lambda *_: _CACHE.clear())
    model._rl1_cast_bound = True


def install(on=True, model=None):
    if model is not None:
        bound(model)
    if ORIGINAL["linear"] is None:
        ORIGINAL["linear"] = torch.nn.functional.linear
        torch.nn.functional.linear = linear
    if ORIGINAL["peft_cast"] is None:
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
        except ImportError:
            BaseTunerLayer = None
        if BaseTunerLayer is not None and hasattr(BaseTunerLayer, "_cast_input_dtype"):
            ORIGINAL["peft_cast"] = BaseTunerLayer._cast_input_dtype
            BaseTunerLayer._cast_input_dtype = peft_cast
    CAST["on"] = bool(on)
    if not on:
        _CACHE.clear()
