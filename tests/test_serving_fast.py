"""serving_fast overlays (MFU-2, 22 Oct): patch application on the pinned files, restart and refusal rules, the
image cache's LRU. Equivalence of replies is checked on the real runtime (Modal CPU: original vs patched modules on
R12 token streams and images; then byte-identical replies on GPU)."""

import hashlib
import shutil
import types
from pathlib import Path

import pytest

from arc_infra import image_cache_runtime as cache
from arc_infra import serving_fast_patch as patch

PINNED = Path("./.claude/worktrees/throughput-arm/keep/vllm-full")  # same build (g8e685d198)
ALL = {"logprob_fast": True, "image_cache": True}


def pinned_site(tmp_path):
    site = tmp_path / "site"
    for pin in (patch.LOGPROBS, patch.CHAT, patch.IMAGE, patch.RENDERER, patch.ASYNC_LLM, patch.GC_UTILS):
        dest = site / pin["relative"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PINNED / pin["relative"], dest)
    return site


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_overlay_on_the_pinned_files_once_and_restart(tmp_path):
    site = pinned_site(tmp_path)
    identity = patch.apply(site, tmp_path / "out", ALL)
    lp = (site / patch.LOGPROBS["relative"]).read_text()
    assert lp.count("elif self.arc_sampled_only:") == 1 and "made.arc_sampled_only = bool(" in lp
    chat = (site / patch.CHAT["relative"]).read_text()
    assert chat.count('"arc_sampled_only": True') == 1 and chat.count("ChatCompletionLogProb.model_construct(") == 1
    assert "from vllm.arc_image_cache import cached" in (site / patch.IMAGE["relative"]).read_text()
    assert (site / patch.IMAGE_HELPER).read_bytes() == Path(cache.__file__).read_bytes()
    assert patch.apply(site, tmp_path / "out", ALL) == identity  # restart in the same working dir
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "other", ALL)  # patched files without this run's marker
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "out", {"logprob_fast": True, "image_cache": False})  # another overlay


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_switches_are_separate(tmp_path):
    site = pinned_site(tmp_path)
    patch.apply(site, tmp_path / "out", {"logprob_fast": False, "image_cache": True})
    assert (site / patch.LOGPROBS["relative"]).read_bytes() == (PINNED / patch.LOGPROBS["relative"]).read_bytes()
    assert (site / patch.CHAT["relative"]).read_bytes() == (PINNED / patch.CHAT["relative"]).read_bytes()
    assert "arc_image_cache" in (site / patch.IMAGE["relative"]).read_text()
    assert patch.apply(site, tmp_path / "o2", {"logprob_fast": False, "image_cache": False}) is None
    assert patch.apply(site, tmp_path / "o3", None) is None


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_mm_warmup_overlap(tmp_path):
    site = pinned_site(tmp_path)
    identity = patch.apply(site, tmp_path / "out", {"mm_warmup_overlap": True})
    assert set(identity["files"]) == {patch.RENDERER["relative"], patch.ASYNC_LLM["relative"]}
    base = (site / patch.RENDERER["relative"]).read_text()
    assert base.count("def arc_start_mm_warmup(self)") == 1 and base.count("def _arc_warmup_mm(self)") == 1
    assert base.count("arc_early.join()") == 1 and base.count("self._arc_warmup_mm()") == 1
    # both processors still warmed, each exactly once, in the moved block
    assert base.count('log_prefix="Multi-modal"') == 1 and base.count('log_prefix="Readonly multi-modal"') == 1
    llm = (site / patch.ASYNC_LLM["relative"]).read_text()
    first, call = llm.index("renderer_from_config(self.vllm_config)"), llm.index("renderer.arc_start_mm_warmup()")
    assert first < call < llm.index("EngineCoreClient.make_async_mp_client(")
    for pin in (patch.LOGPROBS, patch.CHAT, patch.IMAGE):
        assert (site / pin["relative"]).read_bytes() == (PINNED / pin["relative"]).read_bytes()


def test_overlay_rejects_unknown_file(tmp_path, monkeypatch):
    site = tmp_path / "site"
    target = site / patch.IMAGE["relative"]
    target.parent.mkdir(parents=True)
    target.write_text("# drift\n")
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "out", {"image_cache": True})


def test_validate():
    assert patch.validate(None) is None
    assert patch.validate({"image_cache": True}) == {"logprob_fast": False, "image_cache": True}
    assert patch.validate({"mm_warmup_overlap": True}) == {"logprob_fast": False, "image_cache": False,
                                                           "mm_warmup_overlap": True}
    for bad in ({"logprob_fast": 1}, {"other": True}, ["logprob_fast"]):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_serving_configure_rejects_bad_serving_fast():
    from arena import serving

    with pytest.raises(ValueError):
        serving.configure(types.SimpleNamespace(), "/tmp/x", serving_fast={"logprob_fast": "yes"})


class FakeIO:
    def __init__(self, kwargs=None):
        self.image_mode, self.rgba_background_color, self.kwargs = "RGB", (255, 255, 255), kwargs or {}


FakeIO.__name__ = "ImageMediaIO"


class Img:
    def __init__(self, n):
        self.width, self.height, self.n = n, n, n

    def getbands(self):
        return ("R", "G", "B")


def media(n):
    return types.SimpleNamespace(media=Img(n), original_bytes=b"x" * 10)


def test_image_cache_hits_and_evicts(monkeypatch):
    monkeypatch.setattr(cache, "LIMIT", 3 * (100 * 100 * 3 + 10))
    cache._CACHE.clear()
    cache._SIZE[0] = 0
    calls = []

    def load(n):
        calls.append(n)
        return media(100)

    first = cache.cached(FakeIO(), "a", lambda: load("a"))
    assert cache.cached(FakeIO(), "a", lambda: load("a")) is first and calls == ["a"]
    for key in "bcd":
        cache.cached(FakeIO(), key, lambda k=key: load(k))
    assert "a" not in [k[0] for k in cache._CACHE] and len(cache._CACHE) == 3  # LRU evicted the oldest
    cache.cached(FakeIO({"x": 1}), "b", lambda: load("kw"))  # custom decode kwargs: never cached
    assert calls[-1] == "kw"
    other = FakeIO()
    other.image_mode = "L"
    cache.cached(other, "b", lambda: load("mode"))  # another decode mode: another key
    assert calls[-1] == "mode"


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
@pytest.mark.parametrize("early", [True, False])
def test_mm_warmup_overlap_behaviour(tmp_path, early, monkeypatch):
    """The patched renderer methods on a stand-in: an early start warms both processors once in a thread and
    warmup() waits for it; without the early start, warmup() warms them inline, as before."""
    import ast
    import contextlib
    import threading
    import time as _time

    site = pinned_site(tmp_path)
    patch.apply(site, tmp_path / "out", {"mm_warmup_overlap": True})
    tree = ast.parse((site / patch.RENDERER["relative"]).read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and any(
        isinstance(f, ast.FunctionDef) and f.name == "arc_start_mm_warmup" for f in n.body))
    wanted = [f for f in cls.body if isinstance(f, ast.FunctionDef)
              and f.name in ("warmup", "arc_start_mm_warmup", "_arc_warmup_mm")]
    calls = []

    @contextlib.contextmanager
    def threads(n=None):
        yield

    class ChatTemplateResolutionError(Exception):
        pass

    chat_utils = types.ModuleType("vllm.entrypoints.chat_utils")
    chat_utils.ChatTemplateResolutionError = ChatTemplateResolutionError
    import sys as _sys
    for name in ("vllm", "vllm.entrypoints"):
        if name not in _sys.modules:
            monkeypatch.setitem(_sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(_sys.modules, "vllm.entrypoints.chat_utils", chat_utils)
    ns = {"set_default_torch_num_threads": threads, "time": _time, "ChatParams": object,
          "logger": types.SimpleNamespace(debug=lambda *a: None, warning=lambda *a, **k: None, info=lambda *a: None)}
    mod = ast.Module(body=[ast.ClassDef(name="R", bases=[], keywords=[], body=wanted, decorator_list=[])],
                     type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "patched", "exec"), ns)

    class Stand(ns["R"]):
        mm_processor = "mm"
        _readonly_mm_processor = "ro"

        def render_chat(self, *a):
            calls.append(("chat", threading.current_thread().name))

        def _warmup_mm_processor(self, proc, log_prefix):
            _time.sleep(0.05)
            calls.append((proc, threading.current_thread().name))

        def clear_mm_cache(self):
            calls.append(("clear", threading.current_thread().name))

        def _clear_processor_cache(self, proc):
            calls.append(("clear_ro", threading.current_thread().name))

    r = Stand()
    if early:
        r.arc_start_mm_warmup()
    r.warmup(None)
    procs = [c for c in calls if c[0] in ("mm", "ro", "clear", "clear_ro")]
    assert [c[0] for c in procs] == ["mm", "clear", "ro", "clear_ro"]
    assert {c[1] for c in procs} == ({"arc-mm-warmup"} if early else {"MainThread"})
    assert [c for c in calls if c[0] == "chat"] == [("chat", "MainThread")]


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_api_gc_overlay_and_behaviour(tmp_path, monkeypatch, capsys):
    """Patched freeze_gc_heap (run from the patched file) freezes, then tunes once; other files untouched."""
    import gc
    import importlib.util
    import json
    import sys

    site = tmp_path / "site"
    site.mkdir()
    for pin in (patch.LOGPROBS, patch.CHAT, patch.IMAGE, patch.RENDERER, patch.ASYNC_LLM, patch.GC_UTILS):
        dest = site / pin["relative"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PINNED / pin["relative"], dest)
    identity = patch.apply(site, tmp_path / "out", {"api_gc": True})
    assert set(identity["files"]) == {patch.GC_UTILS["relative"]} and "gc_helper_sha256" in identity
    assert (site / patch.GC_HELPER).read_bytes() == (Path(patch.__file__).with_name("api_gc_runtime.py")).read_bytes()
    assert patch.apply(site, tmp_path / "out", {"api_gc": True}) == identity
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "out", {"api_gc": True, "image_cache": True})
    for pin in (patch.LOGPROBS, patch.CHAT, patch.IMAGE, patch.RENDERER, patch.ASYNC_LLM):
        assert (site / pin["relative"]).read_bytes() == (PINNED / pin["relative"]).read_bytes()
    # run the patched function body with stand-in vllm modules
    spec = importlib.util.spec_from_file_location("vllm.arc_api_gc", site / patch.GC_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    tree = __import__("ast").parse((site / patch.GC_UTILS["relative"]).read_text())
    fn = next(n for n in tree.body if getattr(n, "name", "") == "freeze_gc_heap")
    code = compile(__import__("ast").Module(body=[fn], type_ignores=[]), "gc_utils", "exec")
    vllm = type(sys)("vllm")
    vllm.arc_api_gc = helper
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.arc_api_gc", helper)
    monkeypatch.setenv("ARC_API_GC", "40000,10,1000000000,0")
    monkeypatch.setenv("ARC_API_GC_LOG_S", "0")
    calls = []
    fake_gc = types.SimpleNamespace(collect=lambda g: calls.append(("collect", g)), freeze=lambda: calls.append("freeze"))
    ns = {"gc": fake_gc}
    exec(code, ns)
    before = gc.get_threshold()
    try:
        ns["freeze_gc_heap"]()
        assert calls == [("collect", 0), ("collect", 1), ("collect", 2), "freeze"]
        assert gc.get_threshold() == (40000, 10, 1000000000)
        assert helper.tune() is False  # once per process
        line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("ARC_API_GC ")]
        assert json.loads(line[0][11:])["after"] == [40000, 10, 1000000000]
        gc.collect()
        r = helper.row(helper._STATE["stats"], 1.0)
        assert r["full_per_min"] >= 60 and r["gc_s"] > 0 and r["rss_gib"] > 0
        assert helper.row(helper._STATE["stats"], 1.0)["full_per_min"] == 0  # counters reset per line
    finally:
        gc.callbacks.remove(helper._STATE["stats"].callback)
        gc.set_threshold(*before)


def test_api_gc_parse():
    from arc_infra import api_gc_runtime as rt

    assert rt.parse(rt.DEFAULT) == (50000, 10, 10, 0.0)
    for bad in ("1,2,3", "0,10,10,0", "50000,10,10,-1"):
        with pytest.raises(ValueError):
            rt.parse(bad)
    assert patch.validate({"api_gc": True})["api_gc"] is True
    assert "api_gc" not in patch.validate({"image_cache": True})
