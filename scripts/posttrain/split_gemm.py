"""Per-expert base GEMM without per-expert slice gradients (arc2 speed hunt, 18 Oct).

Axolotl's sonicmoe ``nvfp4._grouped_gemm`` falls back to a Python loop over experts when the grouped tokens are not
BF16 (the trainer's are FP32), slicing ``x_grouped[start:end]`` once per expert. Each slice's backward allocates a
zero tensor the size of all grouped tokens and adds its piece into it: 512 experts x 48 layers x [288,710 x 2,560]
FP32 per 29K-token window, 78% of the backward (arc2rl-speed-v1/v2). ``x.split(sizes)`` gives the same views; its
backward is one concatenation. Forward values and gradients are unchanged (the same ``F.linear`` calls; the
gradient pieces are placed instead of added into zeros)."""
import torch
import torch.nn.functional as F

STATE = {"original": None, "on": False}


def _grouped_gemm(x_grouped, weight, expert_offsets, *, backend):
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import _use_grouped_mm

    if backend not in ("dequant", "torch"):
        return STATE["original"](x_grouped, weight, expert_offsets, backend=backend)
    if backend == "dequant":
        weight = nvfp4.dequantize_expert_weight(weight)
    if _use_grouped_mm(x_grouped) and weight.dtype == x_grouped.dtype:
        return torch._grouped_mm(x_grouped, weight.transpose(-2, -1), offs=expert_offsets.to(torch.int32))
    offsets = expert_offsets.tolist()
    sizes = [b - a for a, b in zip([0] + offsets[:-1], offsets)]
    rest = x_grouped.shape[0] - offsets[-1]
    parts = x_grouped.split(sizes + ([rest] if rest else []), dim=0)
    return torch.cat([F.linear(parts[e], weight[e]) for e in range(weight.shape[0])], dim=0)


def install(on=True):
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4

    if STATE["original"] is None:
        STATE["original"] = nvfp4._grouped_gemm
    nvfp4._grouped_gemm = _grouped_gemm if on else STATE["original"]
    STATE["on"] = bool(on)
    env_knobs()
    return STATE["on"]


class _NowExecutor:
    """ThreadPoolExecutor(1) stand-in: ``submit`` runs the call at once (no encoder thread beside the GPU loop)."""

    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def shutdown(self, *a, **k):
        pass


def knobs(switch_interval=None, prefetch=None):
    """The per-expert loop is launch bound: an encoder thread holding the GIL stalls it (speed-v3: 12.9 s windows
    alone, 33-50 s beside the next window's encoding). ``switch_interval``: sys.setswitchinterval; ``prefetch``
    False: the update encodes each window inline (op_update imports ThreadPoolExecutor at call time)."""
    import concurrent.futures as cf
    import sys

    STATE.setdefault("executor", cf.ThreadPoolExecutor)
    if switch_interval is not None:
        sys.setswitchinterval(float(switch_interval))
    if prefetch is not None:
        cf.ThreadPoolExecutor = STATE["executor"] if prefetch else _NowExecutor
    return {"switch_interval": sys.getswitchinterval(), "prefetch": cf.ThreadPoolExecutor is STATE["executor"]}


def env_knobs():
    import os

    value = os.environ.get("RL1_SWITCH_INTERVAL")
    prefetch = os.environ.get("RL1_PREFETCH")
    return knobs(float(value) if value else None, None if prefetch is None else prefetch == "1")
