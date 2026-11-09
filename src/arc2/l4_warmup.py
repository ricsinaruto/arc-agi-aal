"""vLLM general plugin: the warm-up after CUDA-graph capture (``Worker.compile_or_warm_up_model``: one eager forward
of ``min(max_num_seqs, max_num_batched_tokens)`` rows, then a sampler run) capped at ``ARC2_L4_WARMUP_ROWS`` rows,
each step timed to the log, and SIGUSR1 printing every thread's stack. Six fresh first starts of the 64-seat stack
on driver 610 hung in this step (no log for 300 s after capture; the same start served again in the container);
v13 (32 seats) and the 48-seat stacks never did.

13 October (start test 3's stacks): the V2 runner is in use, so the eager forward above is not its path; the four
workers sat in the 4-bit cache's rotation (``int4_per_token_head.single_rht``) inside the first CUDA-graph capture,
after JIT-compiling Triton kernels there. ``ARC2_L4_WARM_RHT``: the rotation run eagerly, forward and inverse, at
the guard's tile sizes, before capture starts."""

import faulthandler
import json
import os
import signal
import sys
import time

ENV = "ARC2_L4_WARMUP_ROWS"
WARM_ENV = "ARC2_L4_WARM_RHT"
WARM_ROWS = (16, 64, 256)  # the guard pads to 16-row tiles; prefill-sized calls too


def note(event, **fields):
    print("ARC2_WARMUP " + json.dumps({"event": event, "pid": os.getpid(), "t": round(time.time(), 2), **fields}),
          flush=True)


def warm_rht():
    """The 4-bit cache's rotation (as the attention calls it) once per tile size and direction, eagerly."""
    import torch
    from vllm import arc_kv4

    rotate, begun = arc_kv4._rotation(), time.time()
    for rows in WARM_ROWS:
        for inverse in (False, True):
            x = torch.randn(rows, arc_kv4.HEAD_DIM, device="cuda", dtype=torch.float32)
            rotate(x, inverse=inverse)
    torch.cuda.synchronize()
    note("rht-warm", seconds=round(time.time() - begun, 2), rows=list(WARM_ROWS))


def install(cap):
    from vllm.v1.worker import gpu_worker

    if cap > 0:
        install_cap(cap)
    worker, warm = gpu_worker.Worker, gpu_worker.Worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *args, **kwargs):
        note("warm-up", v2=bool(getattr(self, "use_v2_model_runner", False)))
        if os.environ.get(WARM_ENV) == "1":
            try:
                warm_rht()
            except Exception as exc:  # noqa: BLE001 - the start goes on as pinned
                note("rht-warm-failed", error=f"{type(exc).__name__}: {exc}"[:300])
        out = warm(self, *args, **kwargs)
        note("warm-up-done")
        return out

    worker.compile_or_warm_up_model = compile_or_warm_up_model
    return True


def install_cap(cap):
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker import gpu_model_runner

    cls = gpu_model_runner.GPUModelRunner
    if getattr(cls, "_arc2_warmup", None):
        return False
    dummy, sampler = cls._dummy_run, cls._dummy_sampler_run

    def dummy_run(self, num_tokens, *args, **kwargs):
        warmup = (kwargs.get("skip_eplb") and kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.NONE
                  and not kwargs.get("is_profile"))
        if not warmup:
            return dummy(self, num_tokens, *args, **kwargs)
        rows = min(int(num_tokens), cap)
        note("forward", rows=rows, asked=int(num_tokens))
        out = dummy(self, rows, *args, **kwargs)
        note("forward-done", rows=rows)
        return out

    def dummy_sampler_run(self, *args, **kwargs):
        note("sampler")
        out = sampler(self, *args, **kwargs)
        note("sampler-done")
        return out

    cls._dummy_run, cls._dummy_sampler_run, cls._arc2_warmup = dummy_run, dummy_sampler_run, cap


def watch(every, until):
    """Every thread's stack to the log every ``every`` s for ``until`` s (C-level: prints from a stuck process)."""
    import threading

    faulthandler.dump_traceback_later(every, repeat=True, file=sys.stderr)
    timer = threading.Timer(until, faulthandler.cancel_dump_traceback_later)
    timer.daemon = True
    timer.start()


def register():
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    cap = int(os.environ.get(ENV) or 0)
    dump = os.environ.get(ENV + "_DUMP")  # "every,until" seconds: a test start's stacks through the warm-up
    if dump:  # every process that loads the plugin: the server, the engine, the workers
        every, until = (float(x) for x in dump.split(","))
        watch(every, until)
    if cap > 0 or os.environ.get(WARM_ENV) == "1":
        try:
            note("installed", cap=cap, warm_rht=os.environ.get(WARM_ENV) == "1", done=install(cap))
        except Exception as exc:  # a missing hook must not end the start: the warm-up then runs as pinned
            note("not-installed", cap=cap, error=f"{type(exc).__name__}: {exc}"[:300])
