"""MFU-2 (25 Oct): server-2-only patch flags reach serving_setup (v6n's ARC_SLEEP_KEEP_KV in timeshare.env alone never
patched the runtime that SecondServer copies)."""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap(monkeypatch):
    """server2_export alone (bootstrap.py imports the bundle's own bootstrap module beside it)."""
    import ast
    import types

    tree = ast.parse((ROOT / "scripts/arena/bootstrap.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "server2_export")
    module = types.SimpleNamespace()
    scope = {"json": json}
    exec(compile(ast.Module([fn], []), "bootstrap.py", "exec"), scope)
    module.server2_export = scope["server2_export"]
    return module


def _setup(monkeypatch, tmp_path):
    for name in ("TAAF_KAGGLE_WORKING_DIR", "TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_SETUP_ENV"):
        monkeypatch.setenv(name, str(tmp_path))
    sys.modules.pop("arc_infra.serving_setup", None)
    return importlib.import_module("arc_infra.serving_setup")


def test_server2_export(monkeypatch):
    b = _bootstrap(monkeypatch)
    assert b.server2_export({}) == {}
    assert b.server2_export({"rl_loop": {"mg": {"timeshare": {"env": {}}}}}) == {}
    study = {"rl_loop": {"mg": {"timeshare": {"env": {"ARC_SLEEP_KEEP_KV": "1", "ARC_X": 2}}}}}
    out = json.loads(b.server2_export(study)["ARC_SERVER2_ENV"])
    assert out == {"ARC_SLEEP_KEEP_KV": "1", "ARC_X": "2"}


def test_bootstrap_wires_export_into_both_server_paths():
    src = (ROOT / "scripts/arena/bootstrap.py").read_text()
    assert "extra = {**server2_export(study), **extra}" in src  # CPU gate patch check
    assert "return {**server2_export(study), **{str(k)" in src  # real server start


def test_patch_flag(monkeypatch, tmp_path):
    s = _setup(monkeypatch, tmp_path)
    s2 = json.dumps({"ARC_SLEEP_KEEP_KV": "1", "ARC_SEGMENT_TOKEN_CACHE": "1", "ARC_MOE_LORA_BM16": "1"})
    assert s.patch_flag("ARC_SLEEP_KEEP_KV", {"ARC_SLEEP_KEEP_KV": "1"})
    assert not s.patch_flag("ARC_SLEEP_KEEP_KV", {})
    assert s.patch_flag("ARC_SLEEP_KEEP_KV", {"ARC_SERVER2_ENV": s2})
    assert s.patch_flag("ARC_SEGMENT_TOKEN_CACHE", {"ARC_SERVER2_ENV": s2})
    assert not s.patch_flag("ARC_MOE_LORA_BM16", {"ARC_SERVER2_ENV": s2})  # traced: both servers or neither
    assert not s.patch_flag("ARC_SLEEP_KEEP_KV", {"ARC_SERVER2_ENV": "not json"})
    src = (ROOT / "src/arc_infra/serving_setup.py").read_text()
    assert 'if patch_flag("ARC_SLEEP_KEEP_KV"):' in src and 'if patch_flag("ARC_SEGMENT_TOKEN_CACHE"):' in src


def test_aot_root_env(monkeypatch, tmp_path):
    s = _setup(monkeypatch, tmp_path)
    assert s.aot_root_env({}, {}) == {}
    env = s.aot_root_env({}, {"ARC_BOOT_CACHE_CARRY": "1", "ARC_MARLIN_BATCH": "1"})
    assert Path(env["VLLM_CACHE_ROOT"]).parent == s.CACHE_ROOT / "vllm-env"
