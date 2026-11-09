"""Launcher of the proposer's GPU jobs (docs/rsi.md round 1: weights may change without training). Root-owned; runs
``python -I gpu_guard.py <script> [args...]`` in the job's sandbox (``RSI_GPU_SITE``: the serving runtime's packages,
put on the path here since -I ignores PYTHONPATH). Before the proposer's script runs, gradients are switched off in
the main thread and in every thread started later, and every PyTorch entry to training refuses: the autograd engine's
backward (``run_backward``, under Tensor.backward, autograd.backward/grad and torch.func), optimizers, re-enabling
gradients, and starting other programs (os spawn/exec/fork, subprocess, ``_posixsubprocess``), which would run
without this guard. Quantization, pruning and calibration (forward passes on practice data) work as usual.

Not proof against deliberate circumvention (raw ctypes calls into libtorch or libc); every job's script and log are
archived for the audit (``<checkpoint>/gpu-jobs/``)."""
import os
import runpy
import subprocess
import sys


class TrainingRefused(RuntimeError):
    pass


def refuse(what, why="no gradient training in round 1 (docs/rsi.md)"):
    def stop(*args, **kwargs):
        raise TrainingRefused(f"{what}: {why}")
    return stop


def guard():
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_grad_enabled(False)
        engine = torch.autograd.variable.Variable._execution_engine

        class Engine:  # the one door to backward passes (autograd.backward/grad, Tensor.backward, torch.func)
            def run_backward(self, *args, **kwargs):
                raise TrainingRefused("autograd engine: no gradient training in round 1 (docs/rsi.md)")

            def __getattr__(self, name):
                return getattr(engine, name)

        torch.autograd.variable.Variable._execution_engine = Engine()
        graph = getattr(torch.autograd, "graph", None)
        if graph is not None and hasattr(graph, "_engine_run_backward"):
            graph._engine_run_backward = refuse("autograd engine")
        torch.Tensor.backward = refuse("Tensor.backward")
        torch.autograd.backward = refuse("autograd.backward")
        torch.autograd.grad = refuse("autograd.grad")
        torch.optim.Optimizer.__init__ = refuse("optimizer")
        torch.optim.Optimizer.step = refuse("optimizer step")
        original = torch.set_grad_enabled

        class NoGrad(original):  # set_grad_enabled(True) and enable_grad() refuse; False is allowed
            def __init__(self, mode):
                if mode:
                    raise TrainingRefused("set_grad_enabled(True): no gradient training in round 1")
                super().__init__(mode)

        torch.set_grad_enabled = NoGrad
        torch.enable_grad = refuse("enable_grad")
        torch.autograd.set_grad_enabled = NoGrad
        torch.autograd.enable_grad = torch.enable_grad
    for name in ("system", "fork", "forkpty", "posix_spawn", "posix_spawnp", "execv", "execve", "execvp", "execvpe",
                 "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "popen"):
        if hasattr(os, name):
            setattr(os, name, refuse(f"os.{name}", "a GPU job runs its one script (another program would be unguarded)"))
    subprocess.Popen = refuse("subprocess", "a GPU job runs its one script (another program would be unguarded)")
    try:
        import _posixsubprocess
        _posixsubprocess.fork_exec = refuse("_posixsubprocess", "a GPU job runs its one script")
    except ImportError:
        pass
    if torch is not None:  # grad mode is per thread: every new thread starts with gradients off
        import _thread
        import threading

        start = _thread.start_new_thread

        def start_new_thread(function, args, kwargs=None):
            def run(*a, **k):
                torch._C._set_grad_enabled(False)
                return function(*a, **k)
            return start(run, args, kwargs or {})

        _thread.start_new_thread = start_new_thread
        threading._start_new_thread = start_new_thread
        if hasattr(threading, "_start_joinable_thread"):  # Python 3.13+
            joinable = threading._start_joinable_thread

            def start_joinable_thread(function, *a, **k):
                def run():
                    torch._C._set_grad_enabled(False)
                    return function()
                return joinable(run, *a, **k)

            threading._start_joinable_thread = start_joinable_thread


def main():
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    site = os.environ.get("RSI_GPU_SITE")
    if site:
        sys.path.append(site)
    guard()
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
