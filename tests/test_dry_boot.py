"""arena.dry_boot: the CPU boot through the engine start (RL 25 Oct, mtpcheck boot deaths)."""
import sys
import time
import types
from pathlib import Path

import pytest

from arena import dry_boot
from arena import serving as S


def fake_recipe(root, calls):
    site = root / dry_boot.SITE
    r = types.SimpleNamespace(RUNTIME_ROOT=root, CACHE_ROOT=root / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")

    def extract(*a, **k):
        (site / "vllm").mkdir(parents=True, exist_ok=True)
        (site / "vllm/__init__.py").write_text("")
        (site / "vllm/pristine.py").write_text("x = 1\n")
        old = time.time() - 60
        import os
        for p in site.rglob("*.py"):
            os.utime(p, (old, old))
        return {"mode": "fake"}

    def gpu_inventory():
        raise FileNotFoundError("nvidia-smi")

    def server_command(model_dir, *, enable_chunked_prefill=True, tuning=None):
        return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir), "--max-num-seqs", "8"]

    def start_server(*a, **k):
        raise AssertionError("the real engine start ran in a dry boot")

    def main():  # recipe.main's order: inventory, extract, (recipe patches), environment, start
        calls.append(("inventory", r.gpu_inventory()))
        r.verify_and_extract_runtime(root)
        env, _ = r.runtime_environment()
        r.start_server(Path("/model"), env, tuning={"t": 1})

    r.verify_and_extract_runtime = extract
    r.gpu_inventory = gpu_inventory
    r.server_command = server_command
    r.start_server = start_server
    r.resolve_vllm_tuning = lambda **k: {"t": 1}
    r.runtime_environment = lambda **k: ({"PYTHONPATH": str(site), "ARC_X": "1"}, {})
    r.main = main
    r.SUPPORTED_GPUS = {"rtx pro 6000": "12.0", "b200": "10.0"}
    r._CUDA_ARCH = None

    def cuda_arch():
        raise FileNotFoundError("nvidia-smi")

    def find_driver():
        raise RuntimeError("No real non-stub libcuda.so.1 was found.")

    r.cuda_arch = cuda_arch
    r._find_real_cuda_driver = find_driver
    return r


def base_patch(site, working, **k):  # stands in for configure's base serving patch (arena.serving.apply)
    (Path(site) / "vllm/patched.py").write_text("y = 2\n")


def test_dry_boot_runs_the_wrapped_boot_to_the_engine_start(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(dry_boot.ENV, "1")
    monkeypatch.setattr(S, "apply", base_patch)
    calls = []
    recipe = fake_recipe(tmp_path / "rt", calls)
    working = tmp_path / "w"
    working.mkdir()
    S.configure(recipe, working, renderer_num_workers=2)
    with pytest.raises(SystemExit) as e:
        recipe.main()
    out = capsys.readouterr().out
    assert e.value.code == 0, out
    assert calls[0][1]["skipped"].startswith("dry boot")
    assert "ARC_DRY_BOOT engine start (1 patched files" in out
    # configure's own argv wrapper ran on the command the fake engine received
    assert "--renderer-num-workers" in out
    row = __import__("json").loads((working / "dry-boot.json").read_text())
    assert row["patched_files"] == ["vllm/patched.py"] and row["imports"] == {"vllm.patched": "ok"}
    assert row["env_keys"] == ["ARC_X"]
    # the GPU-only lookups inside runtime_environment have stand-ins (v6 -db leg)
    assert recipe._CUDA_ARCH == "12.0"
    driver = recipe._find_real_cuda_driver()
    assert driver.name == "libcuda.so.1" and driver.is_file()


def test_dry_boot_fails_on_a_patched_file_that_does_not_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(dry_boot.ENV, "1")
    monkeypatch.setattr(S, "apply", base_patch)
    recipe = fake_recipe(tmp_path / "rt", [])
    working = tmp_path / "w"
    working.mkdir()
    S.configure(recipe, working)
    wrapped = recipe.verify_and_extract_runtime  # configure's extract wrapper (marks the tree, then patches)

    def extract_then_bad_patch(*a, **k):
        out = wrapped(*a, **k)
        (tmp_path / "rt" / dry_boot.SITE / "vllm/bad.py").write_text("import vllm.nowhere\n")
        return out

    recipe.verify_and_extract_runtime = extract_then_bad_patch
    with pytest.raises(SystemExit) as e:
        recipe.main()
    assert e.value.code == 3
    assert "ARC_DRY_BOOT FAIL" in capsys.readouterr().out and "vllm.bad" in (working / "dry-boot.json").read_text()


def test_dry_boot_off_leaves_the_recipe_alone(tmp_path, monkeypatch):
    monkeypatch.delenv(dry_boot.ENV, raising=False)
    recipe = fake_recipe(tmp_path / "rt", [])
    start = recipe.start_server
    S.configure(recipe, tmp_path)
    assert recipe.start_server is start


def test_model_entry_imported_before_its_package_modules(tmp_path):
    """r3off -db (26 Oct): nvidia/qsa.py imported cold is a circular import (it imports .model, which imports qsa's
    class); the server loads model.py first, so the dry boot does too, patched or not."""
    pkg = tmp_path / "vllm/models/flash/nvidia"
    (pkg / "ops").mkdir(parents=True)
    (pkg / "model.py").write_text("")
    mods = ["vllm.arc_x", "vllm.models.flash.nvidia.ops.qsa", "vllm.models.flash.nvidia.qsa", "vllm.models.other.a.b"]
    assert dry_boot.entry_first(mods, tmp_path) == ["vllm.arc_x", "vllm.models.flash.nvidia.model",
                                                    "vllm.models.flash.nvidia.ops.qsa", "vllm.models.flash.nvidia.qsa",
                                                    "vllm.models.other.a.b"]
    assert dry_boot.entry_first(["vllm.models.flash.nvidia.model", "vllm.models.flash.nvidia.qsa"], tmp_path) == \
        ["vllm.models.flash.nvidia.model", "vllm.models.flash.nvidia.qsa"]
