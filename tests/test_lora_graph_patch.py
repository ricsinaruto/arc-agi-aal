"""lora_graph_patch (MFU-2 B300 boot, 24 Oct): edits on the pinned V2 runner files, capture cases and dispatch of a
base-only batch under ARC_LORA_GRAPH_ONE=1, unchanged behaviour without it, idempotence and refusal."""

import ast
import shutil
import types
from pathlib import Path

import pytest

from arc_infra import lora_graph_patch as patch

PINNED = Path(__import__("os").environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
pytestmark = pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")


@pytest.fixture
def site(tmp_path):
    for rel in patch.FILES:
        (tmp_path / "site" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PINNED / rel, tmp_path / "site" / rel)
    return tmp_path / "site"


def function(source, name, cls=None):
    tree = ast.parse(source)
    body = tree.body
    if cls:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    for a in fn.args.args + fn.args.kwonlyargs:
        a.annotation = None
    fn.returns = None
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), ns)
    return ns[name]


def cases(site, monkeypatch, on):
    monkeypatch.setenv(patch.ENV, "1") if on else monkeypatch.delenv(patch.ENV, raising=False)
    fn = function((site / "vllm/v1/worker/gpu/lora_utils.py").read_text(), "get_lora_capture_cases")
    fn.__globals__["get_captured_lora_counts"] = lambda m, s: [m + 1]
    lora = types.SimpleNamespace(max_loras=3, specialize_active_lora=False)
    return fn(lora, types.SimpleNamespace(cudagraph_specialize_lora=True))


def resolve(site, monkeypatch, on, capture_cases, n):
    monkeypatch.setenv(patch.ENV, "1") if on else monkeypatch.delenv(patch.ENV, raising=False)
    fn = function((site / "vllm/v1/worker/gpu/cudagraph_utils.py").read_text(), "_resolve_effective_loras",
                  cls="CudaGraphManager")
    top = max(c for c in capture_cases if c > 0)
    obj = types.SimpleNamespace(lora_capture_cases=capture_cases, _max_lora_case=top,
                                _lora_dispatch_map={i: top for i in range(1, top + 1)})
    return fn(obj, n)


def test_pinned_and_behaviour(site, tmp_path, monkeypatch):
    assert cases(site, monkeypatch, on=True) == [0, 4]  # unpatched: both cases even with the env
    identity = patch.apply(site, tmp_path / "out")
    assert all(v["pinned"] for v in identity.values())
    assert (tmp_path / "out" / "lora-graph-patch.json").is_file()
    assert cases(site, monkeypatch, on=True) == [4]
    assert cases(site, monkeypatch, on=False) == [0, 4]
    assert resolve(site, monkeypatch, True, [4], 0) == 4  # base-only batch -> the LoRA graph
    assert resolve(site, monkeypatch, True, [4], 2) == 4
    assert resolve(site, monkeypatch, True, [0, 4], 0) == 0  # a 0 case exists: unchanged
    assert resolve(site, monkeypatch, False, [4], 0) == 0
    assert resolve(site, monkeypatch, False, [0, 4], 2) == 4


def test_idempotent_and_refusal(site, tmp_path):
    first = patch.apply(site, tmp_path / "out")
    data = {rel: (site / rel).read_bytes() for rel in patch.FILES}
    assert patch.apply(site, tmp_path / "out") == first
    assert {rel: (site / rel).read_bytes() for rel in patch.FILES} == data
    other = tmp_path / "other"
    for rel in patch.FILES:
        (other / rel).parent.mkdir(parents=True, exist_ok=True)
        (other / rel).write_text("x = 1\n")
    assert patch.apply(other, tmp_path / "o2") is None
    assert (other / next(iter(patch.FILES))).read_text() == "x = 1\n"
