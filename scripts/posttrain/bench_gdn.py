"""MFU stage G bench (``modal_mfu_g.py``): one real GDN layer of Flash with the trainer's rank-16 LoRA (fp32 master)
on the five projections, a prefix state from a no-grad chunked prefix pass, then a window forward (under the
trainer's non-reentrant checkpoint) and backward (recompute + backward). Modes:

  src | src2        source dumps (CPU)
  prof [L]          the model path: per-kernel profile of one prefix chunk and one window fwd+bwd
  test              rl1_gdn kernels against references on small shapes (CPU interpreter or GPU)
  bench [L ...]     the model path vs rl1_gdn: speed, memory, numerics (and the model path's own run-to-run spread)
"""
import json
import os
import sys
import time
from pathlib import Path

MODEL = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
CAPTURE = "/runs/validation/b2-tp-quant-capture-all25v71b-48l-v1-nebius-20261010/expert-capture/eval"
LAYER = int(os.environ.get("MFU_G_LAYER", "12"))
PREFIX, CHUNK = int(os.environ.get("MFU_G_PREFIX", "65536")), 16384
OUT = Path(os.environ.get("MFU_G_OUT", "/tmp"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------------------------------------- the layer
def build(layer=LAYER, seed=0):
    """The layer as the trainer runs it: BF16 frozen base, peft LoRA r16/alpha32 fp32 on the five projections."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    conf = AutoConfig.from_pretrained(MODEL)
    text = getattr(conf, "text_config", conf)
    with torch.device("cuda"):
        gdn = m.Qwen4ExpTextGatedDeltaNet(text, layer).to(torch.bfloat16)
    wmap = json.loads((Path(MODEL) / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}.linear_attn."
    state = {}
    for key in (k for k in wmap if k.startswith(pre)):
        with safe_open(str(Path(MODEL) / wmap[key]), framework="pt", device="cpu") as f:
            state[key[len(pre):]] = f.get_tensor(key)
    missing = gdn.load_state_dict(state, strict=True)
    for p in gdn.parameters():
        p.requires_grad_(False)
    gdn = inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=[
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]), gdn)
    g = torch.Generator(device="cuda").manual_seed(seed)
    for name, p in gdn.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            if "lora_B" in name:  # trained-like: non-zero B (scaled per projection in calibrate())
                p.data.normal_(0, 1, generator=g)
            p.requires_grad_(True)
    return gdn, text, missing


def calibrate(gdn, x, share=0.05):
    """Scale each lora_B so the LoRA term is ``share`` of its base output's RMS."""
    import torch

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            lin = getattr(gdn, name)
            xin = x[..., :lin.base_layer.in_features] if lin.base_layer.in_features <= x.shape[-1] else \
                x.repeat(1, 1, 3)[..., :lin.base_layer.in_features]
            base = lin.base_layer(xin).float().pow(2).mean().sqrt()
            a, b = lin.lora_A["default"], lin.lora_B["default"]
            delta = (b(a(xin)) * lin.scaling["default"]).float().pow(2).mean().sqrt()
            b.weight.data.mul_(float(share * base / delta))


_ROWS = {}


def hidden(n, H, layer=LAYER, offset=0):
    """Real normed hidden states: the captured MoE inputs of 9 layers around ``layer`` (stride-16 token samples of
    real play), shuffled once with a fixed seed; bf16 on the card."""
    import numpy as np
    import torch

    if "all" not in _ROWS:
        meta = json.loads((Path(CAPTURE) / "meta.json").read_text())
        parts = []
        for l in range(max(0, layer - 4), min(47, layer + 4) + 1):
            rows = int(meta["rows"][str(l)])
            x = np.fromfile(Path(CAPTURE) / f"L{l:02d}.x", dtype=np.int16, count=rows * H).reshape(rows, H)
            parts.append(torch.from_numpy(x.copy()).view(torch.bfloat16))
        t = torch.cat(parts)
        _ROWS["all"] = t[torch.randperm(t.shape[0], generator=torch.Generator().manual_seed(0))].cuda()
    t = _ROWS["all"]
    idx = (torch.arange(n, device=t.device) + offset) % t.shape[0]
    return t[idx][None]


class Cache:
    """The trainer's cache for one layer: stock LinearAttentionLayer for the prefix, ``read_only()`` (as
    flash_qlora_job.read_only) for the window."""

    def __init__(self, idx):
        from transformers.cache_utils import LinearAttentionLayer

        self.idx = idx
        self.layers = {idx: LinearAttentionLayer()}

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return bool(self.layers[layer_idx].has_previous_state[state_idx or 0])

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
        return self.layers[layer_idx].update_conv_state(conv_states, state_idx, **kwargs)

    def update_recurrent_state(self, states, layer_idx, state_idx=0, **kwargs):
        return self.layers[layer_idx].update_recurrent_state(states, state_idx, **kwargs)

    def read_only(self):
        import copy

        import torch

        view = copy.copy(self)
        layers = self.layers

        def update_conv_state(conv_states, layer_idx, state_idx=0, **kwargs):
            layer = layers[layer_idx]
            if not layer.has_previous_state[state_idx]:
                raise RuntimeError("no prefix state")
            return torch.cat([layer.conv_states[state_idx], conv_states], dim=-1)

        view.update_conv_state = update_conv_state
        view.update_recurrent_state = lambda states, layer_idx, state_idx=0, **kw: states
        return view


def prefix_pass(gdn, xs, idx):
    import torch

    cache = Cache(idx)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, xs.shape[1], CHUNK):
            gdn(xs[:, start:start + CHUNK], cache_params=cache)
    return cache


def window(gdn, xw, cache, gy, *, backward=True):
    """The trainer's window: checkpointed forward (non-reentrant), then backward through the recompute."""
    import torch
    from torch.utils.checkpoint import checkpoint

    ro = cache.read_only() if cache is not None else None
    x = xw.detach().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = checkpoint(lambda h: gdn(h, cache_params=ro), x, use_reentrant=False)
        loss = (out.float() * gy).sum()
    if backward:
        loss.backward()
    return out.detach(), x.grad


def lora_grads(gdn):
    return {n: p.grad.detach().clone() for n, p in gdn.named_parameters() if p.grad is not None}


def zero(gdn):
    for p in gdn.parameters():
        p.grad = None


def timed(fn, reps=5, warm=2):
    import torch

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return round(times[len(times) // 2], 3)


def env_info():
    import inspect

    import fla
    import torch
    import triton

    info = {"torch": torch.__version__, "triton": triton.__version__, "fla": fla.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    import fla.ops.backends as be
    info["fla_backends"] = sorted(os.listdir(os.path.dirname(be.__file__)))
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    src = inspect.getsource(m.Qwen4ExpTextDecoderLayer.forward)
    info["decoder_linear_attn"] = [l.strip() for l in src.splitlines() if "linear_attn" in l or "cache_params" in l]
    return info


def src3():
    import inspect

    import fla.modules.l2norm as l2
    import fla.ops.common.chunk_delta_h as h
    import fla.ops.gated_delta_rule.gate as gate
    import fla.utils as fu

    print("=== gate.py\n" + inspect.getsource(gate))
    print("=== l2norm fwd/bwd\n" + inspect.getsource(l2.l2norm_fwd) + inspect.getsource(l2.l2norm_bwd))
    print("=== input_guard\n" + inspect.getsource(fu.input_guard))
    src = inspect.getsource(h)
    print("=== chunk_delta_h head mapping")
    for i, l in enumerate(src.splitlines()):
        if "HV" in l or "i_h" in l and ("//" in l or "%" in l):
            print(i, l)
    import fla.ops.gated_delta_rule.chunk_fwd as cf
    print("=== chunk_fwd intra\n" + inspect.getsource(cf.chunk_gated_delta_rule_fwd_intra))


def src4():
    """What the trainer's Axolotl plugins do to the GDN layer and its LoRA (kernels plugin, LoRA layer classes)."""
    import subprocess

    import axolotl
    root = os.path.dirname(axolotl.__file__)
    print(subprocess.run(["grep", "-rn", "-E", "GatedDelta|linear_attn|causal_conv1d|kernelize|use_kernel_forward|"
                          "chunk_gated|in_proj_qkv", root, "--include=*.py"], capture_output=True, text=True).stdout[:12000])
    print(subprocess.run(["grep", "-rln", "-E", "class .*\\(.*(LoraLayer|lora\\.Linear|LoraLinear)", root,
                          "--include=*.py"], capture_output=True, text=True).stdout[:3000])


# ---------------------------------------------------------------------------------------------- profile
def prof(lengths):
    import torch
    from torch.profiler import ProfilerActivity, profile

    log(json.dumps(env_info()))
    gdn, text, _ = build()
    if os.environ.get("MFU_G_FAST") == "1":  # profile rl1_gdn's path instead of the model's
        import rl1_gdn

        rl1_gdn.install()
        rl1_gdn.GDN["fast"] = True
    H = text.hidden_size
    for L in lengths:
        xs = hidden(PREFIX, H)
        xw = hidden(L, H, offset=PREFIX)
        calibrate(gdn, xw)
        gy = torch.randn(1, L, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
        cache = prefix_pass(gdn, xs, LAYER)
        t_chunk = timed(lambda: prefix_pass(gdn, xs[:, :CHUNK], LAYER), reps=3)
        t_prefix = timed(lambda: prefix_pass(gdn, xs, LAYER), reps=3)

        def fwd():
            window(gdn, xw, cache, gy, backward=False)

        def fwdbwd():
            zero(gdn)
            window(gdn, xw, cache, gy)

        t_fwd = timed(fwd)
        t_all = timed(fwdbwd)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fwdbwd()
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        log(json.dumps({"L": L, "prefix_chunk_ms": t_chunk, "prefix_64k_ms": t_prefix, "window_fwd_ms": t_fwd,
                        "window_fwd_bwd_ms": t_all, "window_bwd_ms": round(t_all - t_fwd, 3), "peak_gib": round(peak, 2)}))
        for name, fn in (("prefix_chunk", lambda: prefix_pass(gdn, xs[:, :CHUNK], LAYER)), ("window", fwdbwd)):
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
                fn()
                torch.cuda.synchronize()
            rows = {}
            for e in p.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    k = e.name[:110]
                    r = rows.setdefault(k, [0.0, 0])
                    r[0] += e.device_time / 1e3
                    r[1] += 1
            total = sum(v[0] for v in rows.values())
            log(f"== {name} L={L}: kernel total {total:.2f} ms, {sum(v[1] for v in rows.values())} launches")
            for k, (ms, n) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:40]:
                print(f"  {ms:8.3f} ms {n:5d}x  {k}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    rest = [int(a) for a in sys.argv[2:]]
    if mode in ("src", "src2", "src3", "src4"):
        globals()[mode]()
    elif mode == "prof":
        prof(rest or [16384, 32768])
    elif mode == "test":
        import rl1_gdn_test
        rl1_gdn_test.main()
    elif mode == "bench":
        import gdn_compare
        gdn_compare.main(rest or [16384, 32768])
    elif mode == "keep":
        import gdn_compare
        gdn_compare.keep(rest or [16384, 32768])
