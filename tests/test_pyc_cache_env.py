"""ARC_PYC_CACHE=1 (MFU-2 boot, 24 Oct): the server env writes bytecode under CACHE_ROOT/pyc (inside the boot cache's
roots) instead of PYTHONDONTWRITEBYTECODE=1; off unless set."""

import pytest


@pytest.fixture
def ss(tmp_path, monkeypatch):
    from arena import runtime_cache as rc

    for n in ("TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_WORKING_DIR", "TAAF_KAGGLE_SETUP_ENV"):
        monkeypatch.delenv(n, raising=False)
    (tmp_path / "datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1").mkdir(parents=True)
    return rc.pinned_recipe(tmp_path)[0]


def test_off_by_default(ss):
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    assert ss.pyc_cache_env(dict(env), environ={}) == env


def test_on_moves_bytecode_into_boot_cache_root(ss):
    env = ss.pyc_cache_env({"PYTHONDONTWRITEBYTECODE": "1", "X": "y"}, environ={"ARC_PYC_CACHE": "1"})
    assert "PYTHONDONTWRITEBYTECODE" not in env and env["X"] == "y"
    assert env["PYTHONPYCACHEPREFIX"] == str(ss.CACHE_ROOT / "pyc")
    assert ss.CACHE_ROOT in (ss.CACHE_ROOT / "pyc").parents  # tarred by boot_cache (roots CACHE_ROOT, COMPILE_CACHE_ROOT)


def test_check_cache_env(ss):
    assert ss.check_cache_env({}, environ={}) == {}
    env = ss.check_cache_env({}, environ={"ARC_PLE_CHECK_SKIP": "1"})
    assert env["ARC_SK_CHECK_CACHE"] == str(ss.CACHE_ROOT / "arc-checks")
