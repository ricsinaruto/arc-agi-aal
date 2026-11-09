"""sleep_keep_kv_patch: anchors hold on the pinned vLLM, apply is both-or-neither and idempotent, and the patched code
offloads kv_cache and keeps the prefix cache only under ARC_SLEEP_KEEP_KV=1 at level 1 (CPU; no vLLM import)."""

import json
import sys
import types
from pathlib import Path

import pytest

from arc_infra import sleep_keep_kv_patch as patch

PINNED = Path(__import__("os").environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
need_pinned = pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")


def patched(rel):
    return patch.patched_source(rel, (PINNED / rel).read_bytes()).decode()


@need_pinned
def test_anchors_on_pinned():
    for rel in patch.EDITS:
        text = patched(rel)
        assert text.count(patch.TAG) == 1
    assert 'clear_prefix_cache = level >= 1 and not' in patched(patch.CORE)


@need_pinned
@pytest.mark.parametrize("env,level,tags,clear", [(None, 1, ("weights",), True), ("1", 1, ("weights", "kv_cache"), False),
                                                  ("0", 1, ("weights",), True), ("1", 2, (), True)])
def test_behaviour(monkeypatch, env, level, tags, clear):
    if env is None:
        monkeypatch.delenv(patch.ENV, raising=False)
    else:
        monkeypatch.setenv(patch.ENV, env)
    # backend: run the patched CuMemBackend.suspend against a recording allocator
    calls = []

    class Allocator:
        def sleep(self, offload_tags=None):
            calls.append(offload_tags)

    for name in ("vllm", "vllm.logger", "vllm.device_allocator"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.logger"].init_logger = lambda name: None
    sys.modules["vllm.device_allocator"].get_mem_allocator_instance = Allocator
    ns = {"__name__": "sleep_mode_backend_under_test"}
    exec(compile(patched(patch.BACKEND), patch.BACKEND, "exec"), ns)
    backend = ns["CuMemBackend"].__new__(ns["CuMemBackend"])
    backend.suspend(level)
    assert calls == [tags]
    # core: the edited line of EngineCore.sleep, evaluated with this level
    line = next(x for x in patched(patch.CORE).splitlines() if "clear_prefix_cache = level >= 1" in x)
    scope = {"level": level}
    exec(line.strip(), scope)
    assert scope["clear_prefix_cache"] is clear


@need_pinned
def test_apply_idempotent_and_skip(tmp_path, capsys):
    site, out = tmp_path / "site", tmp_path / "out"
    for rel in patch.EDITS:
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_bytes((PINNED / rel).read_bytes())
    first = patch.apply(site, out)
    assert first and json.loads((out / "sleep-keep-kv-patch.json").read_text()) == first
    assert patch.apply(site, out) == first
    for rel in patch.EDITS:
        (site / rel).write_bytes((PINNED / rel).read_bytes())
    (site / patch.CORE).write_text("def sleep():\n    pass\n")
    before = (site / patch.BACKEND).read_bytes()
    assert patch.apply(site, tmp_path / "out2") is None
    assert "ARC_SLEEP_KEEP_KV_PATCH_SKIPPED" in capsys.readouterr().out
    assert (site / patch.BACKEND).read_bytes() == before  # both files or neither


class _Buf:
    def __init__(self, size, pinned):
        self.size, self.pinned, self.ptr = size, pinned, id(self)

    def numel(self):
        return self.size

    def data_ptr(self):
        return self.ptr


def _helper_scope(monkeypatch, env, exact=None, fail=False):
    for name, value in ((patch.ENV, env), ("ARC_SLEEP_EXACT_PIN", exact)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    log = []

    class Cudart:
        def cudaHostRegister(self, ptr, size, flags):
            log.append(("register", size))
            return 1 if fail else 0

        def cudaHostUnregister(self, ptr):
            log.append(("unregister",))

    def check_error(res):
        if res != 0:
            raise RuntimeError(f"cuda error {res}")

    torch = types.SimpleNamespace(uint8="u8", empty=lambda size, dtype, device, pin_memory=False: _Buf(size, pin_memory),
                                  cuda=types.SimpleNamespace(cudart=Cudart, check_error=check_error,
                                                             current_device=lambda: 0, set_device=lambda d: None))
    scope = {"os": __import__("os"), "torch": torch, "PIN_MEMORY": True}
    exec(patch.HELPER.split("\n", 1)[1], scope)  # the helper without its import line
    return scope, log


@need_pinned
def test_cumem_anchor_uses_exact_buffer():
    text = patched(patch.CUMEM)
    assert text.count("cpu_backup_tensor = _arc_host_buf(ptr, size_in_bytes)") == 1
    assert "pin_memory=PIN_MEMORY,\n                )" not in text


@pytest.mark.parametrize("env,exact", [(None, None), ("0", None), ("1", "0")])
def test_exact_pin_off_keeps_torch_pinned(monkeypatch, env, exact):
    scope, log = _helper_scope(monkeypatch, env, exact)
    a, b = scope["_arc_host_buf"](7, 100), scope["_arc_host_buf"](7, 100)
    assert a.pinned and b.pinned and a is not b and log == [] and scope["_ARC_HOST_BUFS"] == {}


def test_exact_pin_registers_once_and_reuses(monkeypatch, capsys):
    scope, log = _helper_scope(monkeypatch, "1")
    buf = scope["_arc_host_buf"]
    a = buf(7, 100)
    assert not a.pinned and a.size == 100 and buf(7, 100) is a
    c = buf(8, 50)
    assert c is not a and buf(8, 50) is c
    d = buf(7, 200)  # size changed: old buffer unregistered, new one registered
    assert d.size == 200 and scope["_ARC_HOST_BUFS"] == {7: d, 8: c}
    assert log == [("register", 100), ("register", 50), ("unregister",), ("register", 200)]
    assert capsys.readouterr().out.count("ARC_SLEEP_EXACT_PIN registered") == 1


def test_exact_pin_register_failure_falls_back(monkeypatch, capsys):
    scope, log = _helper_scope(monkeypatch, "1", fail=True)
    a = scope["_arc_host_buf"](7, 100)
    assert a.pinned and scope["_ARC_HOST_BUFS"] == {}
    assert "ARC_SLEEP_EXACT_PIN_FALLBACK" in capsys.readouterr().out


@need_pinned
def test_cumem_verify_hooks_wrap_both_copies():
    text = patched(patch.CUMEM)
    sleep_at = text.index('_arc_verify(ptr, size_in_bytes, "sleep")')
    assert text.index("libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)") < sleep_at < text.index(
        "unmap_and_release(handle)", sleep_at)
    assert text.index("libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)") < text.index(
        '_arc_verify(ptr, size_in_bytes, "wake")')


def _verify_scope(monkeypatch, env="1"):
    torch = pytest.importorskip("torch")
    if env is None:
        monkeypatch.delenv("ARC_SLEEP_VERIFY", raising=False)
    else:
        monkeypatch.setenv("ARC_SLEEP_VERIFY", env)
    scope = {"os": __import__("os"), "torch": torch, "PIN_MEMORY": False}
    exec(patch.HELPER.split("\n", 1)[1], scope)
    mem = {1: torch.arange(1000, dtype=torch.int32), 2: torch.full((300,), -7, dtype=torch.int32)}
    scope["_arc_view"] = lambda ptr, size: mem[ptr]
    real = scope["_arc_digest"]
    scope["_arc_digest"] = lambda t: real(t, step=128)  # several chunks per allocation
    return scope, mem


def _cycle(scope, mem, change=None):
    for ptr in mem:
        scope["_arc_verify"](ptr, mem[ptr].numel() * 4, "sleep")
    if change:
        change(mem)
    for ptr in mem:
        scope["_arc_verify"](ptr, mem[ptr].numel() * 4, "wake")


def _swap(mem):  # two words exchanged: plain sums agree, the position-weighted sum does not
    mem[1][10:12] = mem[1][10:12].flip(0).clone()


@pytest.mark.parametrize("change,bad", [(None, 0), (lambda m: m[1].__setitem__(500, 0), 1), (_swap, 1)])
def test_verify_counts_restored_mismatches(monkeypatch, capsys, change, bad):
    scope, mem = _verify_scope(monkeypatch)
    _cycle(scope, mem)
    _cycle(scope, mem, change)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("ARC_SLEEP_VERIFY")]
    assert lines[0] == "ARC_SLEEP_VERIFY checked=2 gib=0.00 bad=0 aa_bad=0"
    assert lines[1] == f"ARC_SLEEP_VERIFY checked=2 gib=0.00 bad={bad} aa_bad=0"
    assert scope["_ARC_SUMS"] == {}


def test_verify_off_is_a_no_op(monkeypatch, capsys):
    scope, mem = _verify_scope(monkeypatch, env=None)
    _cycle(scope, mem, lambda m: m[1].__setitem__(5, 0))
    assert "ARC_SLEEP_VERIFY" not in capsys.readouterr().out and scope["_ARC_SUMS"] == {}


@need_pinned
def test_worker_prepins_after_graph_capture():
    text = patched(patch.WORKER)
    assert text.index("self.model_runner.capture_model()") < text.index("_arc_prepin()")
    assert text.count("_arc_prepin()") == 1


def _alloc(scope, entries):
    class Data:
        def __init__(self, tag, size, asleep=False):
            self.tag, self.handle, self.is_asleep = tag, (None, size), asleep

    scope["CuMemAllocator"] = types.SimpleNamespace(instance=types.SimpleNamespace(
        pointer_to_data={ptr: Data(*e) for ptr, e in entries.items()}))


def test_prepin_registers_offloadable_buffers_once(monkeypatch, capsys):
    scope, log = _helper_scope(monkeypatch, "1")
    _alloc(scope, {1: ("weights", 100), 2: ("kv_cache", 50), 3: ("default", 70), 4: ("kv_cache", 30, True)})
    scope["_arc_prepin"]().join(5)
    assert sorted(scope["_ARC_HOST_BUFS"]) == [1, 2] and sorted(log) == [("register", 50), ("register", 100)]
    assert "ARC_SLEEP_PREPIN done n=2" in capsys.readouterr().out
    assert scope["_arc_host_buf"](2, 50) is scope["_ARC_HOST_BUFS"][2] and len(log) == 2  # sleep reuses it


@pytest.mark.parametrize("env,prepin", [(None, None), ("1", "0")])
def test_prepin_off(monkeypatch, env, prepin):
    scope, log = _helper_scope(monkeypatch, env)
    if prepin is None:
        monkeypatch.delenv("ARC_SLEEP_PREPIN", raising=False)
    else:
        monkeypatch.setenv("ARC_SLEEP_PREPIN", prepin)
    _alloc(scope, {1: ("weights", 100)})
    assert scope["_arc_prepin"]() is None and log == []
