"""MFU-2 boot items: the restored compile cache survives the patch-time clears of the container's first start."""

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arc_infra import serving_kernels_patch as patch

ROOT = Path(__file__).resolve().parents[1]


def recipe(tmp_path):
    return NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve"],
              RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")


@pytest.fixture
def staged(tmp_path, monkeypatch):
    from arena import serving

    applied = []
    monkeypatch.setattr(serving, "apply", lambda *a, **k: applied.append("layout"))
    monkeypatch.setattr(serving, "apply_serving_kernels", lambda site, working, spec: applied.append("serving") or {})
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: applied.append("clear") or [str(root)])
    for name in serving.KEEP_BLOCKING_ENV + (serving.COMPILE_KEEP_ENV,):
        monkeypatch.delenv(name, raising=False)
    return serving, applied


def start(serving, tmp_path):
    r = recipe(tmp_path)
    serving.configure(r, tmp_path, serving_kernels={"topk_sampler": True})
    r.verify_and_extract_runtime()
    return json.loads((tmp_path / "serving-kernels-start.json").read_text())["cleared_compile_cache"]


def test_without_a_restored_archive_the_clear_runs(staged, tmp_path):
    serving, applied = staged
    assert start(serving, tmp_path) == [str(tmp_path / "cache")]
    assert applied == ["layout", "clear", "serving"] and not (tmp_path / "compile-cache-kept.json").exists()


def test_a_restored_archive_is_kept_once(staged, tmp_path, monkeypatch):
    serving, applied = staged
    monkeypatch.setenv(serving.COMPILE_KEEP_ENV, "5281567aa1d13715")
    assert start(serving, tmp_path) == []
    assert applied == ["layout", "serving"]
    assert json.loads((tmp_path / "compile-cache-kept.json").read_text()) == {"key": "5281567aa1d13715"}
    assert serving.COMPILE_KEEP_ENV not in os.environ  # consumed: a later start in this process clears again
    applied.clear()
    assert start(serving, tmp_path) == [str(tmp_path / "cache")] and applied == ["layout", "clear", "serving"]


@pytest.mark.parametrize("lab", ["ARENA_SERVING_KERNELS", "ARENA_DENSE_KERNELS", "ARENA_EXPERT_PRUNE"])
def test_a_lab_override_outside_the_key_clears(staged, tmp_path, monkeypatch, lab):
    serving, applied = staged
    monkeypatch.setenv(serving.COMPILE_KEEP_ENV, "k")
    monkeypatch.setenv(lab, "off")
    assert serving.take_compile_keep() is None and serving.COMPILE_KEEP_ENV not in os.environ


def modal_rl1():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("modal_rl1_boot", ROOT / "scripts/posttrain/modal_rl1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(folder, key):
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / f"{key}.tar.gz", "w:gz") as tar:
        for name in ("tmp/qwen38-flash-next-vllm-cache/vllm/torch_compile_cache/torch_aot_compile/h/rank_0_0/model",
                     "tmp/other/ignored"):
            data = b"x"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("keep", [False, True])
def test_restore_sets_the_keep_key_only_when_asked(tmp_path, monkeypatch, keep):
    M = modal_rl1()
    monkeypatch.delenv("ARC_COMPILE_KEEP", raising=False)
    archive(tmp_path / "arch", "abc")
    identity = {}
    M.compile_restore("abc", identity, keep=keep, archives=tmp_path / "arch", root=tmp_path / "root")
    assert identity["compile_cache"]["restored"] and identity["compile_cache"]["files"] == 1
    assert (tmp_path / "root/tmp/qwen38-flash-next-vllm-cache/vllm/torch_compile_cache/torch_aot_compile/h/rank_0_0/model").is_file()
    assert not (tmp_path / "root/tmp/other").exists()
    assert os.environ.get("ARC_COMPILE_KEEP") == ("abc" if keep else None)
    os.environ.pop("ARC_COMPILE_KEEP", None)  # not monkeypatch.delenv: its teardown would restore "abc" for later tests
    missing = {}
    M.compile_restore("nope", missing, keep=True, archives=tmp_path / "arch", root=tmp_path / "root")
    assert missing["compile_cache"] == {"key": "nope", "restored": False} and "ARC_COMPILE_KEEP" not in os.environ


def test_patch_settings_stay_canonical():
    assert patch.validate({"ple_uva": True})["ple_uva"] is True


def test_prefetch_reads_every_shard_once_and_reports(tmp_path, capsys):
    M = modal_rl1()
    for i, size in enumerate((5, 70, 0)):
        (tmp_path / f"model-{i:05d}.safetensors").write_bytes(b"a" * size)
    (tmp_path / "config.json").write_text("{}")
    row = {"threads": 2}
    M.prefetch_weights(row, 2, root=tmp_path, block=16)
    assert row["files"] == 3 and row["gib"] == round(75 / 2**30, 2) and "error" not in row
    assert '"event": "weights_prefetched"' in capsys.readouterr().out
    capped = {}
    M.prefetch_weights(capped, 1, root=tmp_path, limit=2)
    assert capped["files"] == 2
    broken = {}
    M.prefetch_weights(broken, 0, root=tmp_path)  # a bad setting never raises: the loads read the volume
    assert "error" in broken


def test_boot_settings_default_off(tmp_path):
    M = modal_rl1()
    assert M.boot_settings(tmp_path) == {}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "boot.json").write_text('{"prefetch_threads": 16}')
    assert M.boot_settings(tmp_path) == {"prefetch_threads": 16}


def fake_recipe(tmp_path):
    """A recipe whose fast-mode extraction writes a small tree (files, a nested dir, a symlink, an executable, an
    empty file) into its RUNTIME_ROOT global, read at call time like the real one."""
    import hashlib
    import types

    runtime = tmp_path / "runtime-ds"
    runtime.mkdir()
    (runtime / "runtime-manifest.json").write_text('{"m": 1}')
    sha = hashlib.sha256((runtime / "runtime-manifest.json").read_bytes()).hexdigest()
    rec = types.SimpleNamespace(VLLM_RUNTIME_MANIFEST_SHA256=sha, EXPECTED_RUNTIME_LAYERS={8: ("aa", 1)},
                                FAST_RUNTIME_WHITEOUTS={8: ("x/.wh.y",)}, RUNTIME_ROOT=tmp_path / "live", calls=[])

    def extract(runtime_dir, *, full_layer_hashes=True, scan_extracted_caches=True):
        root = Path(rec.RUNTIME_ROOT)
        rec.calls.append(str(root))
        (root / "usr/lib/pkg").mkdir(parents=True)
        (root / "usr/lib/pkg/a.py").write_text("print(1)\n")
        (root / "usr/lib/pkg/big.bin").write_bytes(os.urandom(200_000))
        (root / "usr/lib/pkg/empty").write_bytes(b"")
        (root / "usr/bin").mkdir(parents=True)
        (root / "usr/bin/tool").write_text("#!/bin/sh\n")
        (root / "usr/bin/tool").chmod(0o755)
        (root / "usr/lib/link").symlink_to("pkg/a.py")
        return {"manifest_sha256": sha, "layers": [8], "verification_mode": "fast", "extract_seconds": 9.9}

    rec.verify_and_extract_runtime = extract
    return rec, runtime


def test_runtime_cache_builds_checks_and_restores_the_same_tree(tmp_path):
    from arena import runtime_cache as rc

    rec, runtime = fake_recipe(tmp_path)
    cache = tmp_path / "cache"
    assert rc.restore(rec, cache, (runtime,), {"full_layer_hashes": False, "scan_extracted_caches": False}) is None
    built = rc.build(rec, runtime, cache, work=tmp_path / "work", parts=3)
    assert built["built"] and built["built_tree"] == built["restored_tree"] and built["built_tree"]["files"] == 4
    assert rec.RUNTIME_ROOT == tmp_path / "live"  # the module global is put back
    assert rc.build(rec, runtime, cache, work=tmp_path / "work")["why"] == "exists"
    out = rc.restore(rec, cache, (runtime,), {"full_layer_hashes": False, "scan_extracted_caches": False})
    assert out["runtime_cache"]["parts"] == 4 and out["manifest_sha256"] == rec.VLLM_RUNTIME_MANIFEST_SHA256
    assert rc.tree_digest(tmp_path / "live") == built["built_tree"]
    assert os.access(tmp_path / "live/usr/bin/tool", os.X_OK) and os.readlink(tmp_path / "live/usr/lib/link") == "pkg/a.py"
    # full-hash extractions and another manifest are never served from the cache
    assert rc.restore(rec, cache, (runtime,), {"full_layer_hashes": True, "scan_extracted_caches": True}) is None
    (runtime / "runtime-manifest.json").write_text('{"m": 2}')
    assert rc.restore(rec, cache, (runtime,), {"full_layer_hashes": False, "scan_extracted_caches": False}) is None


def test_runtime_cache_broken_entry_falls_back(tmp_path, capsys):
    from arena import runtime_cache as rc

    rec, runtime = fake_recipe(tmp_path)
    cache = tmp_path / "cache"
    built = rc.build(rec, runtime, cache, work=tmp_path / "work", parts=2)
    (cache / built["key"] / "part-01.tar").write_bytes(b"not a tar")
    assert rc.restore(rec, cache, (runtime,), {"full_layer_hashes": False, "scan_extracted_caches": False}) is None
    assert "ARC_RUNTIME_CACHE_FAILED" in capsys.readouterr().out


def test_serving_extract_uses_the_cache_only_when_set(staged, tmp_path, monkeypatch):
    serving, applied = staged
    from arena import runtime_cache as rc

    calls = []
    monkeypatch.setattr(rc, "restore", lambda recipe, cache, a, k: calls.append(cache) or {"cached": True})
    r = recipe(tmp_path)
    r.verify_and_extract_runtime = lambda *a, **k: {"cached": False}
    serving.configure(r, tmp_path)
    assert r.verify_and_extract_runtime() == {"cached": False} and calls == []
    monkeypatch.setenv(rc.ENV, str(tmp_path / "c"))
    r2 = recipe(tmp_path)
    r2.verify_and_extract_runtime = lambda *a, **k: {"cached": False}
    serving.configure(r2, tmp_path)
    assert r2.verify_and_extract_runtime() == {"cached": True} and calls == [str(tmp_path / "c")]


def test_pinned_recipe_imports_without_the_serving_environment(tmp_path, monkeypatch):
    from arena import runtime_cache as rc

    for n in ("TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_WORKING_DIR", "TAAF_KAGGLE_SETUP_ENV"):
        monkeypatch.delenv(n, raising=False)
    (tmp_path / "datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1").mkdir(parents=True)
    module, runtime = rc.pinned_recipe(tmp_path)
    assert callable(module.verify_and_extract_runtime) and runtime.name == "qwen38-flash-next-vllm-nvfp4-runtime-v1"
    assert "TAAF_KAGGLE_BUNDLE_DIR" not in os.environ
    assert rc.cache_key(module, runtime, False, False) is None  # no pinned manifest: never cached


def test_local_weights_shadow_tree_and_staging(tmp_path, capsys):
    M = modal_rl1()
    src = tmp_path / "assets"
    model = src / "models/k/m/pytorch/v/1"
    model.mkdir(parents=True)
    (src / "datasets/runtime").mkdir(parents=True)
    (src / "datasets/runtime/blob").write_bytes(b"rt")
    (src / "models/k/other").mkdir(parents=True)
    names = ["model-00010.safetensors", "model-00002.safetensors", "layer-00001-experts.safetensors"]
    for i, n in enumerate(names):
        (model / n).write_bytes(bytes([i]) * (1000 + i))
    (model / "config.json").write_text("{}")
    shadow = M.shadow_input(src, tmp_path / "kaggle/input", model)
    target = tmp_path / "kaggle/input"
    assert shadow == target / "models/k/m/pytorch/v/1" and not shadow.is_symlink()
    assert (target / "datasets").is_symlink() and (target / "datasets/runtime/blob").read_bytes() == b"rt"
    assert (target / "models/k/other").is_symlink()
    assert sorted(p.name for p in shadow.iterdir()) == sorted(names + ["config.json"])
    assert all((shadow / n).resolve() == (model / n).resolve() for n in names)  # volume until staged
    row = M.stage_local_weights({}, 2, shadow, root=model, local=tmp_path / "local")
    assert row["files"] == 3 and "error" not in row
    for n in names:
        assert (shadow / n).resolve() == (tmp_path / "local" / n).resolve()
        assert (shadow / n).read_bytes() == (model / n).read_bytes()
    assert (shadow / "config.json").resolve() == (model / "config.json").resolve()
    assert not list((tmp_path / "local").glob(".*")) and not list(shadow.glob(".*"))
    assert '"event": "weights_staged_local"' in capsys.readouterr().out
    assert [p.name for p in sorted(model.glob("*.safetensors"), key=lambda p: M._natural(p.name))] == [
        "layer-00001-experts.safetensors", "model-00002.safetensors", "model-00010.safetensors"]


def test_local_weights_failure_leaves_volume_links(tmp_path, monkeypatch):
    M = modal_rl1()
    model = tmp_path / "assets/m"
    model.mkdir(parents=True)
    (model / "a.safetensors").write_bytes(b"x" * 10)
    shadow = M.shadow_input(tmp_path / "assets", tmp_path / "in", model)
    import shutil

    monkeypatch.setattr(shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"x"))  # a short copy
    row = M.stage_local_weights({}, 1, shadow, root=model, local=tmp_path / "local")
    assert "short copy" in row["error"] and (shadow / "a.safetensors").resolve() == (model / "a.safetensors").resolve()


@pytest.mark.parametrize("compact", [False, True])
def test_expert_mask_compact_clears_even_a_kept_archive(staged, tmp_path, monkeypatch, compact):
    serving, applied = staged
    monkeypatch.setattr(serving.expert_mask_patch, "apply", lambda *a: applied.append("mask") or {"ok": True})
    monkeypatch.setenv(serving.COMPILE_KEEP_ENV, "k")
    if compact:
        monkeypatch.setenv(serving.expert_mask_patch.COMPACT_ENV, "1")
    else:
        monkeypatch.delenv(serving.expert_mask_patch.COMPACT_ENV, raising=False)
    r = recipe(tmp_path)
    serving.configure(r, tmp_path, expert_mask=(tmp_path / "m.json", "0" * 64))
    r.verify_and_extract_runtime()
    got = json.loads((tmp_path / "expert-mask-start.json").read_text())["cleared_compile_cache"]
    assert got == ([str(tmp_path / "cache")] if compact else []) and applied[-1] == "mask"


def test_restart_guard_from_the_bundle_blocks_a_crash_restart_and_resumes_a_preemption(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_RL1_BUNDLE", str(ROOT))
    M = modal_rl1()
    monkeypatch.setattr(M, "runs", NS(commit=lambda: None))
    monkeypatch.setenv("ARC_LAUNCH_ID", "call-1")
    monkeypatch.delenv("ARC_RESTART_ACK", raising=False)
    dead = tmp_path / "validation" / "r1"
    dead.mkdir(parents=True)
    assert M.dead_step(dead) == 0
    (dead / "v7_state.json").write_text(json.dumps({"step": 3}))
    assert M.dead_step(dead) == 3
    g = M.restart_guard("validation/r1", root=tmp_path)
    assert not g.start(0).restart  # first attempt
    g2 = M.restart_guard("validation/r1", root=tmp_path)  # the same call restarted, no signal: blocked
    blocked = sys.modules["arc_runops.restart"].RestartBlocked
    with pytest.raises(blocked):
        g2.start(M.dead_step(dead))
    monkeypatch.setenv("ARC_RESTART_ACK", "crash read in the platform log")
    monkeypatch.setenv("ARC_LAUNCH_ID", "call-2")
    g3 = M.restart_guard("validation/r1", root=tmp_path)
    assert g3.start(3).restart  # an acked relaunch
    g3.record_preemption()
    monkeypatch.delenv("ARC_RESTART_ACK")
    g4 = M.restart_guard("validation/r1", root=tmp_path)
    start = g4.start(3)  # same call (call-2) after a preemption with a checkpoint: auto-resume
    assert start.resume and start.incident["resumed"]
    monkeypatch.setenv("ARC_RL1_BUNDLE", str(tmp_path))  # a bundle without runops
    assert modal_rl1().restart_guard("validation/r1", root=tmp_path) is None
