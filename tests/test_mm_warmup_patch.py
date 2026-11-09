"""mm_warmup_patch (MFU-2 RTX boot, 24 Oct): the edit on the pinned renderers/base.py, restart and refusal rules,
and the warmup body skipping only the multi-modal processor warmups under ARC_MM_WARMUP_SKIP=1."""

import ast
import contextlib
import shutil
import types
from pathlib import Path

import pytest

from arc_infra import mm_warmup_patch as patch

PINNED = Path(__import__("os").environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))  # same runtime build


def warmup_fn(source):
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
               and any(getattr(f, "name", "") == "warmup" for f in n.body))
    fn = next(f for f in cls.body if getattr(f, "name", "") == "warmup")
    fn.decorator_list = []
    ns = {"set_default_torch_num_threads": lambda n: contextlib.nullcontext(),
          "logger": types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
          "time": __import__("time"), "ChatParams": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "base", "exec"), ns)
    return ns["warmup"]


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_patch_restart_refusal_and_skip(tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "vllm/renderers").mkdir(parents=True)
    shutil.copy(PINNED / patch.RELATIVE, site / patch.RELATIVE)
    identity = patch.apply(site, tmp_path / "out")
    assert patch.apply(site, tmp_path / "out") == identity
    assert patch.apply(site, tmp_path / "other")["patched_sha256"] == identity["patched_sha256"]  # no double edit
    with pytest.raises(RuntimeError):
        patch.patched_source(b"not vllm")
    (site / "vllm/renderers/base.py").write_text("x = 1\n")
    assert patch.apply(site, tmp_path / "third") is None  # unrecognized: skipped, never raises
    shutil.copy(PINNED / patch.RELATIVE, site / patch.RELATIVE)
    patch.apply(site, tmp_path / "out2")
    import sys
    chat_utils = types.ModuleType("vllm.entrypoints.chat_utils")
    chat_utils.ChatTemplateResolutionError = type("E", (Exception,), {})
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", chat_utils)

    def run(fn):
        calls = []
        r = types.SimpleNamespace(render_chat=lambda *a: calls.append("chat"), mm_processor=object(),
                                  _readonly_mm_processor=object(), clear_mm_cache=lambda: calls.append("clear"),
                                  _clear_processor_cache=lambda p: calls.append("clear_ro"),
                                  _warmup_mm_processor=lambda p, log_prefix: calls.append(log_prefix))
        fn(r, None)
        return calls

    original = warmup_fn((PINNED / patch.RELATIVE).read_text())
    patched = warmup_fn((site / patch.RELATIVE).read_text())
    full = ["chat", "Multi-modal", "clear", "Readonly multi-modal", "clear_ro"]
    monkeypatch.delenv(patch.ENV, raising=False)
    assert run(original) == full and run(patched) == full  # off: unchanged
    monkeypatch.setenv(patch.ENV, "1")
    assert run(patched) == ["chat"] and run(original) == full


def method_fn(source, cls_has, name):
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
               and any(getattr(f, "name", "") == cls_has for f in n.body))
    fn = next(f for f in cls.body if getattr(f, "name", "") == name)
    fn.decorator_list = []
    ns = {"set_default_torch_num_threads": lambda n: contextlib.nullcontext(),
          "logger": types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
          "time": __import__("time"), "ChatParams": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "base", "exec"), ns)
    return ns[name]


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_hero_runtime_file_after_serving_fast_overlap(tmp_path, monkeypatch):
    """The B300 hero applies serving_fast (mm_warmup_overlap edits renderers/base.py) before this patch (v6f boot)."""
    sf = pytest.importorskip("arc_infra.serving_fast_patch")

    site = tmp_path / "site"
    (site / "vllm/renderers").mkdir(parents=True)
    hero = sf.edited((PINNED / patch.RELATIVE).read_bytes(), sf.RENDERER, sf.WARMUP_EDITS)
    (site / patch.RELATIVE).write_bytes(hero)
    identity = patch.apply(site, tmp_path / "out")
    assert identity is not None and not identity["pinned"]
    text = (site / patch.RELATIVE).read_text()
    assert text.count('"ARC_MM_WARMUP_SKIP") != "1"') == 2 and "_arc_warmup_mm" in text
    calls = []
    r = types.SimpleNamespace(mm_processor=object(), _readonly_mm_processor=object(),
                              clear_mm_cache=lambda: calls.append("clear"),
                              _clear_processor_cache=lambda p: calls.append("clear_ro"),
                              _warmup_mm_processor=lambda p, log_prefix: calls.append(log_prefix))
    fn = method_fn(text, "_arc_warmup_mm", "_arc_warmup_mm")
    monkeypatch.setenv(patch.ENV, "1")
    fn(r)
    assert calls == []
    monkeypatch.delenv(patch.ENV)
    fn(r)
    assert calls == ["Multi-modal", "clear", "Readonly multi-modal", "clear_ro"]


def test_patch_check_exit_fails_on_a_skipped_patch(tmp_path, monkeypatch):
    for key in ("TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_WORKING_DIR", "TAAF_KAGGLE_SETUP_ENV"):
        monkeypatch.setenv(key, str(tmp_path / key))
    from arc_infra.serving_setup import patch_check_exit

    lines = []
    (tmp_path / "mm-warmup-skip.json").write_text("{}")
    with pytest.raises(SystemExit) as done:
        patch_check_exit({"ple": True, "mm_warmup": {"patched_sha256": "x"}}, {}, tmp_path,
                         emit=lambda s, **k: lines.append(s))
    assert done.value.code == 0 and lines[-1].startswith("ARC_PATCH_CHECK OK") and "mm-warmup-skip.json" in lines[-1]
    with pytest.raises(SystemExit) as done:
        patch_check_exit({"ple": True, "mm_warmup": None}, {}, tmp_path, emit=lambda s, **k: lines.append(s))
    assert done.value.code == 3 and lines[-1].startswith("ARC_PATCH_CHECK FAIL")
