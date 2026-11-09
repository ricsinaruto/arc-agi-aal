"""RL1_FIRST_CALLS: compile / autotune timers (fake triton and CuTe modules)."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))


def fakes(monkeypatch):
    calls = []

    class JITFunction:
        def _do_compile(self, key):
            calls.append(("compile", key))
            return key

    class Autotuner:
        def _bench(self, config):
            calls.append(("bench", config))
            return 1.0

    def compile_(fn):
        calls.append(("cute", fn))
        return fn

    jit, tuner = types.ModuleType("triton.runtime.jit"), types.ModuleType("triton.runtime.autotuner")
    jit.JITFunction, tuner.Autotuner = JITFunction, Autotuner
    runtime, triton = types.ModuleType("triton.runtime"), types.ModuleType("triton")
    runtime.jit, runtime.autotuner, triton.runtime = jit, tuner, runtime
    cute, cutlass = types.ModuleType("cutlass.cute"), types.ModuleType("cutlass")
    cute.compile, cutlass.cute = compile_, cute
    for name, module in {"triton": triton, "triton.runtime": runtime, "triton.runtime.jit": jit,
                         "triton.runtime.autotuner": tuner, "cutlass": cutlass, "cutlass.cute": cute}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls, JITFunction, Autotuner, cute


def test_timers_count_calls_and_pass_results(monkeypatch):
    import rl1_firstcall as fc

    monkeypatch.setattr(fc, "INSTALLED", [])
    monkeypatch.setattr(fc, "STATS", {})
    monkeypatch.setattr(fc, "wrap_hub", lambda: None)  # hub_fetch: tests/test_rl1_hubk.py
    calls, JITFunction, Autotuner, cute = fakes(monkeypatch)
    assert fc.install({}) == [] and fc.report() is None  # off without the flag
    assert set(fc.install({"RL1_FIRST_CALLS": "1"})) == {"triton_compile", "autotune_bench", "cute_compile"}
    assert fc.install({"RL1_FIRST_CALLS": "1"}) == fc.INSTALLED  # once
    assert JITFunction()._do_compile("k1") == "k1" and JITFunction()._do_compile("k2") == "k2"
    assert Autotuner()._bench("c") == 1.0 and cute.compile("f") == "f"
    rows = fc.report()
    assert rows["triton_compile"]["calls"] == 2 and rows["autotune_bench"]["calls"] == 1
    assert rows["cute_compile"]["calls"] == 1 and len(calls) == 4
