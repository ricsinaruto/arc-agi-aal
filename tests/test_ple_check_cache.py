"""ARC_PLE_CHECK_SKIP / ARC_SK_CHECK_CACHE (MFU-2 boot, 24 Oct): a passed load-time self-check is skipped on a later
boot with the same identity; a failed one writes no marker; another identity runs again. The runtime module needs
Triton, so its two functions are exec'd from source with stubs."""

import ast
import json
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/arc_infra/serving_kernels_runtime.py"


def functions(config):
    tree = ast.parse(SRC.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_check_identity", "cached_check")]
    logs = []
    torch = types.SimpleNamespace(__version__="2.x", version=types.SimpleNamespace(cuda="13.0"),
                                  cuda=types.SimpleNamespace(is_available=lambda: False))
    ns = {"json": json, "os": os, "Path": Path, "torch": torch, "__file__": str(SRC),
          "load_config": lambda: config, "log": lambda kind, payload: logs.append((kind, payload))}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(SRC), "exec"), ns)
    return ns, logs


def test_skip_after_pass_and_rerun_on_new_identity(tmp_path):
    ns, logs = functions({"ple_uva": True})
    env = {"ARC_SK_CHECK_CACHE": str(tmp_path / "c")}
    calls = []
    assert ns["cached_check"]("ple_uva", {"t": 1}, lambda: calls.append(1), environ=env) is True
    assert ns["cached_check"]("ple_uva", {"t": 1}, lambda: calls.append(2), environ=env) is False  # skipped
    assert calls == [1] and logs[-1][0] == "CHECK_SKIPPED"
    assert ns["cached_check"]("ple_uva", {"t": 2}, lambda: calls.append(3), environ=env) is True  # other identity
    ns2, _ = functions({"ple_uva": True, "embed_host": True})  # other settings: other identity
    assert ns2["cached_check"]("ple_uva", {"t": 1}, lambda: calls.append(4), environ=env) is True
    assert calls == [1, 3, 4]


def test_failure_writes_no_marker_and_off_always_runs(tmp_path):
    ns, logs = functions({})
    env = {"ARC_SK_CHECK_CACHE": str(tmp_path / "c")}

    def bad():
        raise RuntimeError("self-check failed")

    with pytest.raises(RuntimeError):
        ns["cached_check"]("ple_uva", {}, bad, environ=env)
    assert not list((tmp_path / "c").glob("*.ok")) if (tmp_path / "c").exists() else True
    calls = []
    for _ in range(2):
        assert ns["cached_check"]("ple_uva", {}, lambda: calls.append(1), environ={}) is True
    assert calls == [1, 1] and logs[-1][0] == "CHECK_SECONDS"
