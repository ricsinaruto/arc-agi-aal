"""GDN core-input dump (throughput arm, 23 Oct; GDN-state precision test). Installed as ``vllm/arc_gdn_dump.py``.

Wraps each QwenGatedDeltaNetAttention's chunk_gated_delta_rule op: for the layers in ARC_GDN_DUMP_LAYERS (model layer
indices), single-sequence prefill calls are saved from a zero initial state on, chunk by chunk, until
ARC_GDN_DUMP_TOKENS tokens per sequence; ARC_GDN_DUMP_SEQS sequences per layer. Saved per call (torch.save):
q, k (all k heads), v, g, beta and the kernel output o for v heads < ARC_GDN_DUMP_VHEADS, the call's initial state
being zero, and the token offset. Only while the file ARC_GDN_DUMP_ARMED exists (the driver creates it after start-up),
and each driven sequence must start from a zero state (the driver resets the prefix cache first).
"""
import os

import torch

DIR = os.environ.get("ARC_GDN_DUMP_DIR", "/tmp/arc_gdn_dump")
LAYERS = {int(x) for x in os.environ.get("ARC_GDN_DUMP_LAYERS", "").split(",") if x.strip()}
TOKENS = int(os.environ.get("ARC_GDN_DUMP_TOKENS", "49152"))
SEQS = int(os.environ.get("ARC_GDN_DUMP_SEQS", "2"))
VHEADS = int(os.environ.get("ARC_GDN_DUMP_VHEADS", "12"))
ARMED = os.environ.get("ARC_GDN_DUMP_ARMED", "/tmp/arc_gdn_dump_armed")  # the driver creates it (no warmup/probe dumps)
_STATE = {}  # layer -> {"seq": n, "offset": tokens so far or None, "call": n}


def _layer_index(prefix):
    parts = str(prefix).split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


class _Dump(torch.nn.Module):
    def __init__(self, inner, layer):
        super().__init__()
        self.inner = inner
        self.layer = layer

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(self, *args, **kw):
        out = self.inner(*args, **kw)
        try:
            self._maybe_dump(kw, out)
        except Exception as exc:  # noqa: BLE001 - a dump failure never breaks serving
            print(f"ARC_GDN_DUMP error layer {self.layer}: {exc}", flush=True)
        return out

    def _maybe_dump(self, kw, out):
        cu = kw.get("cu_seqlens")
        if cu is None or cu.numel() != 2 or "q" not in kw or not os.path.exists(ARMED):
            return
        st = _STATE.setdefault(self.layer, {"seq": 0, "offset": None, "call": 0})
        if st["seq"] >= SEQS:
            return
        init = kw["initial_state"]
        zero = bool((init == 0).all().item())
        n = int(cu[1].item() - cu[0].item())
        if st["offset"] is None:
            if not zero or n < 256:
                return
            st["offset"] = 0
        elif zero:  # a new sequence started before the last one reached TOKENS: close the last one
            st["seq"] += 1
            st["offset"] = 0
            if st["seq"] >= SEQS:
                return
        o = out[0] if isinstance(out, (tuple, list)) else out
        vh = VHEADS
        rec = {"layer": self.layer, "seq": st["seq"], "offset": st["offset"], "zero_init": zero, "tokens": n,
               "q": kw["q"][0].detach().to(torch.bfloat16).cpu(), "k": kw["k"][0].detach().to(torch.bfloat16).cpu(),
               "v": kw["v"][0, :, :vh].detach().to(torch.bfloat16).cpu(),
               "g": kw["g"][0, :, :vh].detach().float().cpu(), "beta": kw["beta"][0, :, :vh].detach().float().cpu(),
               "o": o[0, :, :vh].detach().to(torch.bfloat16).cpu(),
               "init": init[0, :vh].detach().float().cpu() if not zero else None,
               "scale": kw.get("scale")}
        d = os.path.join(DIR, f"L{self.layer:02d}")
        os.makedirs(d, exist_ok=True)
        torch.save(rec, os.path.join(d, f"s{st['seq']}_c{st['call']:04d}.pt"))
        st["call"] += 1
        st["offset"] += n
        if st["offset"] >= TOKENS:
            st["seq"] += 1
            st["offset"] = None


def install_layer(cls):
    if not LAYERS or getattr(cls, "_arc_gdn_dump", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        layer = _layer_index(getattr(self, "prefix", "") or kw.get("prefix", ""))
        if layer in LAYERS and hasattr(self, "chunk_gated_delta_rule"):
            self.chunk_gated_delta_rule = _Dump(self.chunk_gated_delta_rule, layer)
            print(f"ARC_GDN_DUMP layer {layer} -> {DIR}", flush=True)

    cls.__init__ = __init__
    cls._arc_gdn_dump = True
