"""RL1_FIRST_CALLS=1: where a trainer process's first calls spend their time (D1 26 Oct: the first accumulate 41.6 s vs
9-15 s warm; its first window's prefix pass 11.9 s vs 0.4-0.6 s). Cumulative seconds and calls since ``install`` of
Triton JIT compiles (cache hits included: a carried TRITON_CACHE_DIR still loads each kernel), Triton autotune
benchmarks (a cache miss), CuTe DSL compiles (quack / sonicmoe), and torch.compile's own compile times and cache
counters. The accumulate result carries ``first_calls`` (cumulative: the first call's row minus the second's is the
first-call cost)."""

import os
import sys
import time

STATS = {}
INSTALLED = []


def timed(name, fn):
    def wrapper(*args, **kwargs):
        began = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            row = STATS.setdefault(name, [0, 0.0])
            row[0] += 1
            row[1] += time.time() - began

    wrapper.first_call_timed = True
    wrapper.__wrapped__ = fn
    return wrapper


def install(environ=None):
    """The wrappers, once (no-op without RL1_FIRST_CALLS=1). Returns the names installed."""
    environ = os.environ if environ is None else environ
    if environ.get("RL1_FIRST_CALLS") != "1" or INSTALLED:
        return list(INSTALLED)
    try:
        import triton.runtime.autotuner as autotuner
        import triton.runtime.jit as jit

        if not getattr(jit.JITFunction._do_compile, "first_call_timed", False):
            jit.JITFunction._do_compile = timed("triton_compile", jit.JITFunction._do_compile)
            INSTALLED.append("triton_compile")
        if not getattr(autotuner.Autotuner._bench, "first_call_timed", False):
            autotuner.Autotuner._bench = timed("autotune_bench", autotuner.Autotuner._bench)
            INSTALLED.append("autotune_bench")
    except Exception:  # noqa: BLE001 - no triton here
        pass
    wrap_hub()
    try:
        import cutlass.cute as cute

        if not getattr(cute.compile, "first_call_timed", False):
            cute.compile = timed("cute_compile", cute.compile)
            INSTALLED.append("cute_compile")
    except Exception:  # noqa: BLE001 - no CuTe DSL here
        pass
    return list(INSTALLED)


def wrap_hub():
    """``hub_fetch``: huggingface_hub.snapshot_download timed where it is bound now (the package and the ``kernels``
    modules that imported it by name: the hub kernels' lazy fetch, D2 27 Oct). Idempotent."""
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "huggingface_hub" or name == "kernels" or name.startswith("kernels.")):
            continue
        fn = getattr(mod, "snapshot_download", None)
        if callable(fn) and not getattr(fn, "first_call_timed", False):
            try:
                setattr(mod, "snapshot_download", timed("hub_fetch", fn))
            except (AttributeError, TypeError):
                continue
            if "hub_fetch" not in INSTALLED:
                INSTALLED.append("hub_fetch")


def report():
    """Cumulative {name: {"calls", "seconds"}}, plus torch.compile's compile seconds and cache counters."""
    if not INSTALLED:
        return None
    out = {k: {"calls": v[0], "seconds": round(v[1], 2)} for k, v in sorted(STATS.items())}
    try:
        import torch._dynamo.utils as dynamo

        raw = dynamo.compile_times(repr="raw") or {}
        out["dynamo_seconds"] = round(sum(sum(v) for v in raw.values()), 2) if isinstance(raw, dict) else None
        counters = {group: dict(dynamo.counters[group]) for group in ("inductor", "aot_autograd", "stats")
                    if dynamo.counters.get(group)}
        keep = ("fxgraph_cache_hit", "fxgraph_cache_miss", "autograd_cache_hit", "autograd_cache_miss",
                "unique_graphs", "calls_captured")
        out["dynamo_counters"] = {f"{g}.{k}": v for g, row in counters.items() for k, v in row.items() if k in keep}
    except Exception:  # noqa: BLE001
        pass
    return out
