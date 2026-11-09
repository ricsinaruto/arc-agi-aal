"""Boot caches (ARC_BOOT_CACHE_IN / ARC_BOOT_CACHE_SAVE): save then restore round trip, refusal of members outside
the cache roots, and the never-raising environment hooks wired into serving_setup.main."""

import io
import json
import tarfile
from pathlib import Path

from arc_infra import boot_cache as bc

ROOT = Path(__file__).resolve().parents[1]


def test_save_then_restore_round_trips_the_cache_roots(tmp_path):
    a, b = tmp_path / "cache", tmp_path / "compile"
    (a / "vllm/torch_compile_cache/x").mkdir(parents=True)
    (a / "vllm/torch_compile_cache/x/aot.bin").write_bytes(b"\x01" * 1000)
    (b / "triton/k").mkdir(parents=True)
    (b / "triton/k/kernel.cubin").write_bytes(b"cubin")
    (b / "triton/link").symlink_to(b / "triton/k/kernel.cubin")
    out = tmp_path / "out"
    rows = []
    k = bc.key(["serve", "x"], tmp_path)
    bc.save(out, (a, b, tmp_path / "missing"), emit=lambda r, **kw: rows.append(json.loads(r)), cache_key=k)
    name = bc.archive_name(k)
    assert rows[-1]["files"] == 2 and (out / name).is_file() and not (out / (name + ".part")).exists()
    for p in (a, b):
        import shutil
        shutil.rmtree(p)
    assert bc.restore(out, (a, b), emit=lambda *x, **kw: None, cache_key="0" * 16)["found"] is False  # other key
    row = bc.restore(out, (a, b), emit=lambda r, **kw: rows.append(json.loads(r)), cache_key=k)
    assert row["found"] and row["files"] == 2 and row["refused"] == 0
    assert (a / "vllm/torch_compile_cache/x/aot.bin").read_bytes() == b"\x01" * 1000
    assert (b / "triton/k/kernel.cubin").read_bytes() == b"cubin"


def test_restore_refuses_members_outside_the_roots_and_links(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    with tarfile.open(src / bc.NAME, "w") as tar:
        for name, data in ((str(root / "ok.bin").lstrip("/"), b"ok"), ("etc/evil", b"x"),
                           (str(root / "../escape").lstrip("/"), b"y")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(str(root / "l").lstrip("/"))
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    row = bc.restore(src, (root,), emit=lambda *a, **k: None)  # unkeyed name
    assert row["files"] == 1 and row["refused"] == 3
    assert (root / "ok.bin").read_bytes() == b"ok" and not (tmp_path / "escape").exists()
    assert not (root / "l").exists()


def test_key_follows_the_argv_and_every_overlay_identity(tmp_path):
    (tmp_path / "dense-kernels-patch.json").write_text('{"a": 1}')
    k1 = bc.key(["serve", "m"], tmp_path)
    assert k1 == bc.key(["serve", "m"], tmp_path) != bc.key(["serve", "m", "--x"], tmp_path)
    (tmp_path / "serving-kernels-start.json").write_text('{"b": 2}')
    k2 = bc.key(["serve", "m"], tmp_path)
    (tmp_path / "dense-kernels-patch.json").write_text('{"a": 2}')
    assert len({k1, k2, bc.key(["serve", "m"], tmp_path)}) == 3


def test_environment_hooks_are_off_by_default_and_never_raise(tmp_path):
    assert bc.restore_from_environment((tmp_path,), "k", environ={}) is None
    assert bc.save_from_environment((tmp_path,), tmp_path, "k", environ={}) is None
    rows = []
    row = bc.restore_from_environment((tmp_path,), "k", environ={"ARC_BOOT_CACHE_IN": str(tmp_path / "none")},
                                      emit=lambda r, **k: rows.append(json.loads(r)))
    assert row["found"] is False
    (tmp_path / "c").mkdir()
    (tmp_path / "c/f").write_text("1")
    bc.save_from_environment((tmp_path / "c",), tmp_path / "o", "k", environ={"ARC_BOOT_CACHE_SAVE": "1"},
                             emit=lambda r, **k: rows.append(json.loads(r)), wait=True)
    assert rows[-1]["event"] == "boot_cache_save" and (tmp_path / "o" / bc.archive_name("k")).is_file()
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / bc.archive_name("k")).write_bytes(b"not a tar")
    assert bc.restore_from_environment((tmp_path,), "k", environ={"ARC_BOOT_CACHE_IN": str(bad)},
                                       emit=lambda r, **k: rows.append(json.loads(r))) is None
    assert rows[-1]["event"] == "boot_cache_restore_error"


def test_main_restores_before_the_server_starts_and_saves_after_ready():
    src = (ROOT / "src/arc_infra/serving_setup.py").read_text()
    main = src[src.index("def main() -> None:"):]
    r = main.index("restore_from_environment((CACHE_ROOT, COMPILE_CACHE_ROOT), cache_key)")
    s = main.index("server_identity = start_server(model_dir, env, tuning=tuning)")
    w = main.index("readiness = wait_for_server(server_identity, setup_deadline)")
    v = main.index("save_from_environment((CACHE_ROOT, COMPILE_CACHE_ROOT), WORKING_DIR, cache_key)")
    assert main.index("boot_cache_key(server_command(model_dir, tuning=tuning), WORKING_DIR)") < r
    assert r < s < w < v


def test_save_runs_in_a_detached_process_that_outlives_its_parent(tmp_path):
    """rtxmark-v1: serving_setup exits right after ready; the archive is written by a process of its own."""
    import subprocess
    import sys
    import time

    (tmp_path / "c").mkdir()
    (tmp_path / "c/f").write_text("1")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from arc_infra import boot_cache as bc; "
            "bc.save_from_environment((sys.argv[2],), sys.argv[3], 'k', environ={'ARC_BOOT_CACHE_SAVE': '1'})")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT / "src"), str(tmp_path / "c"), str(tmp_path / "o")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "boot_cache_save_started" in r.stdout  # the parent returned at once
    archive = tmp_path / "o" / bc.archive_name("k")
    for _ in range(100):
        if archive.is_file():
            break
        time.sleep(0.1)
    assert archive.is_file()
    rows = []
    bc.restore(tmp_path / "o", (tmp_path / "c",), emit=lambda r, **k: rows.append(json.loads(r)), cache_key="k")
    assert rows[-1]["files"] == 1


def test_save_defaults_to_the_run_output_folder(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "c/f").write_text("1")
    rows = []
    env = {"ARC_BOOT_CACHE_SAVE": "1", "ARC_VALIDATION_OUTPUT": str(tmp_path / "run")}
    bc.save_from_environment((tmp_path / "c",), tmp_path / "serving", "k", environ=env,
                             emit=lambda r, **k: rows.append(json.loads(r)), wait=True)
    assert (tmp_path / "run" / "boot-cache" / bc.archive_name("k")).is_file() and not (tmp_path / "serving").exists()
    bc.save_from_environment((tmp_path / "c",), tmp_path / "serving", "k", wait=True,
                             environ={**env, "ARC_BOOT_CACHE_OUT": str(tmp_path / "x")}, emit=lambda r, **k: None)
    assert (tmp_path / "x" / bc.archive_name("k")).is_file()  # an explicit folder wins


def _aot(root, fp, h, mtime, body=b"m"):
    d = root / "vllm-env" / fp / "torch_compile_cache/torch_aot_compile" / h
    (d / "rank_0_0").mkdir(parents=True)
    (d / "rank_0_0/model").write_bytes(body)
    (d / "inductor_cache/ab").mkdir(parents=True)
    (d / "inductor_cache/ab/k.py").write_text(h)
    import os
    os.utime(d / "rank_0_0/model", (mtime, mtime))
    return d


def test_carry_restores_key_independent_caches_only(tmp_path):
    """MFU-2 (25 Oct): a key miss (new overlays) carries AOT graphs (newest AOT_KEEP, by compile time), kernel caches
    and bytecode from the newest other archive; never other cache content (arc-checks, hf, old vllm root)."""
    import os
    import shutil
    a, b = (tmp_path / "cache").resolve(), (tmp_path / "compile").resolve()
    for i, h in enumerate(["h0", "h1", "h2", "h3", "h4"]):
        _aot(a, "fpA", h, 1_000 + i)
    (a / "vllm/torch_compile_cache/torch_aot_compile/old/rank_0_0").mkdir(parents=True)
    (a / "vllm/torch_compile_cache/torch_aot_compile/old/rank_0_0/model").write_bytes(b"x")  # no fingerprint
    (a / "arc-checks").mkdir()
    (a / "arc-checks/pass").write_text("1")
    (a / "pyc").mkdir()
    (a / "pyc/m.pyc").write_bytes(b"pyc")
    for d in ("triton", "torchinductor", "cuda", "flashinfer"):
        (b / d).mkdir(parents=True)
        (b / d / "f").write_text(d)
    out = tmp_path / "out"
    bc.save(out, (a, b), emit=lambda *x, **k: None, cache_key="1" * 16)
    os.utime(out / bc.archive_name("1" * 16), (5_000, 5_000))
    bc.save(out, (b,), emit=lambda *x, **k: None, cache_key="2" * 16)  # older, other content
    os.utime(out / bc.archive_name("2" * 16), (4_000, 4_000))
    shutil.rmtree(a)
    shutil.rmtree(b)
    rows = []
    env = {"ARC_BOOT_CACHE_IN": str(out), "ARC_BOOT_CACHE_CARRY": "1"}
    row = bc.restore_from_environment((a, b), "3" * 16, environ=env, emit=lambda r, **k: rows.append(json.loads(r)))
    assert row["found"] is False and row["carry"]["name"] == bc.archive_name("1" * 16)
    assert sorted(row["carry"]["aot"]) == ["h1", "h2", "h3", "h4"]  # h0 is the oldest of five
    aot = a / "vllm-env/fpA/torch_compile_cache/torch_aot_compile"
    assert (aot / "h4/rank_0_0/model").is_file() and (aot / "h4/inductor_cache/ab/k.py").read_text() == "h4"
    assert not (aot / "h0").exists() and not (a / "vllm").exists() and not (a / "arc-checks").exists()
    assert (aot / "h4/rank_0_0/model").stat().st_mtime == 1_004  # compile time kept for the next carry
    assert (a / "pyc/m.pyc").is_file()
    assert {d for d in ("triton", "torchinductor", "cuda", "flashinfer") if (b / d / "f").is_file()} == {
        "triton", "torchinductor", "cuda"}
    # an exact key never carries; carry off never carries
    assert "carry" not in bc.restore_from_environment((a, b), "1" * 16, environ=env, emit=lambda *x, **k: None)
    off = {"ARC_BOOT_CACHE_IN": str(out)}
    assert "carry" not in bc.restore_from_environment((a, b), "4" * 16, environ=off, emit=lambda *x, **k: None)


def test_env_fingerprint_splits_on_graph_flags_only():
    base = {"ARC_MARLIN_BATCH": "1", "ARC_VALIDATION_OUTPUT": "/runs/v/a", "ARC_RL1_RESUME": "x", "PATH": "/bin",
            "ARC_BOOT_CACHE_IN": "/a", "ARC_SERVER2_ENV": "{}"}
    fp = bc.env_fingerprint(base)
    assert fp == bc.env_fingerprint({**base, "ARC_VALIDATION_OUTPUT": "/runs/v/b", "ARC_RL1_RESUME": "y", "PATH": "/x",
                                     "ARC_SERVER2_ENV": '{"ARC_MOE_LORA_BM16": "1"}'})
    assert fp != bc.env_fingerprint({**base, "ARC_MOE_LORA_BM16": "1"})
    assert fp != bc.env_fingerprint({**base, "ARENA_SERVING_KERNELS": "{}"})
    assert bc.vllm_cache_root("/c", base) == Path("/c/vllm-env") / fp


def test_serving_setup_sets_the_fingerprinted_vllm_root():
    src = (ROOT / "src/arc_infra/serving_setup.py").read_text()
    assert "    aot_root_env(env)\n" in src and 'env["VLLM_CACHE_ROOT"] = str(vllm_cache_root(CACHE_ROOT, environ))' in src


def test_carry_prefers_an_archive_with_this_servers_graph(tmp_path):
    """Other consumers save to the same folder: the newest archive may hold no graph for this fingerprint."""
    import os
    import shutil
    a, b = (tmp_path / "cache").resolve(), (tmp_path / "compile").resolve()
    env = {"ARC_MARLIN_BATCH": "1"}
    mine = bc.env_fingerprint(env)
    _aot(a, mine, "hm", 1_000)
    (b / "triton").mkdir(parents=True)
    (b / "triton/old").write_text("old")
    out = tmp_path / "out"
    bc.save(out, (a, b), emit=lambda *x, **k: None, cache_key="1" * 16)
    os.utime(out / bc.archive_name("1" * 16), (4_000, 4_000))
    shutil.rmtree(a)
    shutil.rmtree(b)
    _aot(a, "otherfp", "ho", 2_000)
    (b / "triton").mkdir(parents=True)
    (b / "triton/new").write_text("new")
    bc.save(out, (a, b), emit=lambda *x, **k: None, cache_key="2" * 16)  # newest, another consumer
    os.utime(out / bc.archive_name("2" * 16), (5_000, 5_000))
    shutil.rmtree(a)
    shutil.rmtree(b)
    full = {**env, "ARC_BOOT_CACHE_IN": str(out), "ARC_BOOT_CACHE_CARRY": "1"}
    row = bc.restore_from_environment((a, b), "3" * 16, environ=full, emit=lambda *x, **k: None)
    assert row["carry"]["name"] == bc.archive_name("1" * 16) and row["carry"]["preferred"]
    assert (a / "vllm-env" / mine / "torch_compile_cache/torch_aot_compile/hm/rank_0_0/model").is_file()
    shutil.rmtree(a)
    shutil.rmtree(b)
    other = {"ARC_X": "1", "ARC_BOOT_CACHE_IN": str(out), "ARC_BOOT_CACHE_CARRY": "1"}  # no archive has its graph
    row = bc.restore_from_environment((a, b), "3" * 16, environ=other, emit=lambda *x, **k: None)
    assert row["carry"]["name"] == bc.archive_name("2" * 16) and not row["carry"]["preferred"]
