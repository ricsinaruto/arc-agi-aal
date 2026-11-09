"""modal_rl1 teacher_cache_* (MFU-2, 23 Oct): the stream teacher's quack + hub caches persist under a key of the
teacher venv's versions, never seed across a version change, copy hub symlinks as plain files without blobs, and save
once the teacher log holds 3 pieces. Lifted from the source (modal_rl1 builds Modal images at import)."""
import ast
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "modal_rl1.py"
NAMES = ("teacher_cache_key", "teacher_cache_seed", "teacher_cache_save", "teacher_cache_saver", "stream_teacher_env",
         "teacher_cache_env")
CONSTS = ("TEACHER_CACHE", "TEACHER_CACHE_DISTS", "QUACK_SAVE_PIECES", "CACHE_SKIP", "CHAINS_OFF", "CACHE_DIRS")


def lifted():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in CONSTS for t in n.targets))]
    commits = []
    ns = {"Path": Path, "json": json, "hashlib": hashlib, "time": time,
          "runs": SimpleNamespace(commit=lambda: commits.append(1))}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    ns["commits"] = commits
    return ns


def fake_run(versions, noise="sitecustomize: hello", code=0):
    def run(cmd, **kw):
        assert cmd[1] == "-c" and json.loads(cmd[3])
        return SimpleNamespace(returncode=code, stdout=f"{noise}\nARC_VERSIONS {json.dumps(versions)}\n", stderr="")
    return run


def test_key_is_stable_and_changes_with_any_version():
    ns = lifted()
    v = {"python": "3.12.3", "machine": "x86_64", "torch": "2.13.0+cu130", "quack-kernels": "0.4.1", "kernels": "0.9"}
    k1, got = ns["teacher_cache_key"]("py", run=fake_run(v))
    assert got == v and len(k1) == 16 and ns["teacher_cache_key"]("py", run=fake_run(v))[0] == k1
    for dist, new in (("quack-kernels", "0.4.2"), ("torch", "2.13.1+cu130"), ("kernels", None)):
        assert ns["teacher_cache_key"]("py", run=fake_run({**v, dist: new}))[0] != k1
    with pytest.raises(RuntimeError, match="no torch"):
        ns["teacher_cache_key"]("py", run=fake_run({**v, "torch": None}))
    with pytest.raises(RuntimeError, match="failed"):
        ns["teacher_cache_key"]("py", run=fake_run(v, code=1))


def test_key_script_runs_in_a_real_interpreter():
    ns = lifted()
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.run(cmd, **kw)

    with pytest.raises(RuntimeError, match="no torch"):  # the script ran and printed its line: only torch is missing
        ns["teacher_cache_key"](sys.executable, dists=("pytest",), run=run)
    r = subprocess.run(calls[0], capture_output=True, text=True)
    row = json.loads(r.stdout.split("ARC_VERSIONS ", 1)[1])
    assert row["pytest"] == pytest.__version__ and row["python"] == sys.version.split()[0]


def make_local(tmp):
    q, hub = tmp / "q", tmp / "hub"
    (q / "k").mkdir(parents=True)
    (q / "k" / "kernel.o").write_bytes(b"o" * 64)
    (q / "k" / "lock").write_text("")
    repo = hub / "models--kernels-community--sonic-moe"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "abc").write_bytes(b"so" * 32)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("deadbeef")
    snap = repo / "snapshots" / "deadbeef" / "build"
    snap.mkdir(parents=True)
    (snap / "_ops.so").symlink_to("../../../blobs/abc")
    (repo / "snapshots" / "deadbeef" / "x.incomplete").write_text("")
    cute = tmp / "cute"
    cute.mkdir()
    (cute / "abc.mlir.bc").write_bytes(b"c" * 16)
    return {"quack": q, "hub": hub, "cute": cute}


def test_seed_save_roundtrip_and_version_guard(tmp_path):
    ns = lifted()
    root, v = tmp_path / "vol", {"torch": "2.13.0", "quack-kernels": "0.4.1"}
    d0, seeded = ns["teacher_cache_seed"]("k1", v, local=tmp_path / "l0", root=root)
    assert not seeded and sorted(d0) == ["cute", "hub", "quack"] and all(d.is_dir() for d in d0.values())
    assert not (root / "k1").exists()  # empty start
    dirs = make_local(tmp_path)
    assert ns["teacher_cache_save"]("k1", v, dirs, root=root) == {"quack": 1, "hub": 2, "cute": 1}
    assert ns["commits"] == [1]
    vol = root / "k1"
    assert json.loads((vol / "versions.json").read_text()) == v
    so = vol / "hub" / "models--kernels-community--sonic-moe" / "snapshots" / "deadbeef" / "build" / "_ops.so"
    assert so.is_file() and not so.is_symlink() and so.read_bytes() == b"so" * 32
    assert not (vol / "hub" / "models--kernels-community--sonic-moe" / "blobs").exists()
    assert not list(vol.rglob("lock")) and not list(vol.rglob("*.incomplete"))
    d2, seeded = ns["teacher_cache_seed"]("k1", v, local=tmp_path / "l2", root=root)
    q2, hub2 = d2["quack"], d2["hub"]
    assert seeded and (q2 / "k" / "kernel.o").read_bytes() == b"o" * 64
    assert (d2["cute"] / "abc.mlir.bc").read_bytes() == b"c" * 16
    assert (hub2 / "models--kernels-community--sonic-moe" / "refs" / "main").read_text() == "deadbeef"
    assert (hub2 / "models--kernels-community--sonic-moe" / "snapshots" / "deadbeef" / "build" / "_ops.so").is_file()
    # a key dir whose versions.json differs (a hash clash or a hand copy) seeds nothing
    d3, seeded = ns["teacher_cache_seed"]("k1", {**v, "torch": "2.14.0"}, local=tmp_path / "l3", root=root)
    assert not seeded and not any(list(d.iterdir()) for d in d3.values())


def test_save_failure_is_logged_not_raised(tmp_path, capsys):
    ns = lifted()
    dirs = make_local(tmp_path)
    (tmp_path / "vol").write_text("a file where the volume dir should be")
    assert ns["teacher_cache_save"]("k1", {"torch": "x"}, dirs, root=tmp_path / "vol") is False
    assert "teacher cache save failed" in capsys.readouterr().out


def test_saver_saves_after_three_pieces_then_after_the_timer(tmp_path):
    ns = lifted()
    log, saves, done = tmp_path / "teacher-stream.log", [], threading.Event()
    piece = '{"t": 1, "event": "stream_piece", "file": "x"}\n'
    log.write_text("loading\n" + piece * 2)
    t = threading.Thread(target=ns["teacher_cache_saver"], args=(log, lambda: saves.append(time.time())),
                         kwargs={"poll": 0.02, "done": done, "after": 0.6})
    t.start()
    time.sleep(0.2)
    assert saves == []  # two pieces: not yet
    with log.open("a") as f:
        f.write(piece)
    time.sleep(0.2)
    assert len(saves) == 1  # the third piece: one save, not one per poll
    t.join(2)
    assert not t.is_alive() and len(saves) == 2  # the timer's save, then the thread ends


def test_saver_stops_on_done_and_tolerates_a_missing_log(tmp_path):
    ns = lifted()
    saves, done = [], threading.Event()
    t = threading.Thread(target=ns["teacher_cache_saver"], args=(tmp_path / "none.log", lambda: saves.append(1)),
                         kwargs={"poll": 0.02, "done": done})
    t.start()
    time.sleep(0.1)
    done.set()
    t.join(1)
    assert not t.is_alive() and saves == []


def test_teacher_env_flag_and_round_wiring():
    ns = lifted()
    assert ns["stream_teacher_env"](1, "RL1_EXPERTS_FP8=1,ARC_TEACHER_CACHE=1")["ARC_TEACHER_CACHE"] == "1"
    src = SRC.read_text()
    start = src.index('        work = Path("/tmp/heal-stream")')  # stream_round's teacher()
    body = src[start:src.index("saved = save_q()\n", start) + 17]
    assert 'teacher_extra.get("ARC_TEACHER_CACHE") == "1"' in body
    assert 'local_q, hub_env = cdirs["quack"], teacher_cache_env(cdirs)' in body and "HF_HUB_CACHE" not in body
    assert '"QUACK_CACHE_DIR": str(local_q)' in body and "**hub_env" in body
    assert "teacher_cache_saver, args=(logs(0), save_q)" in body
    assert src.count("            saved = save_q()\n") == 1  # the round's end saves on either path


def test_preflight_check_passes_on_the_round_source(tmp_path, monkeypatch):
    """heal_preflight._teacher_cache_check (run when ARC_HEAL_TEACHER_ENV has ARC_TEACHER_CACHE=1), against this
    source; a stand-in torch dist-info gives the interpreter a torch version."""
    pf = SRC.parent / "heal_preflight.py"
    tree = ast.parse(pf.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_teacher_cache_check"]
    ns = {"Path": Path, "json": json, "sys": sys, "MODAL_RL1": str(SRC)}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(pf), "exec"), ns)
    site = tmp_path / "site"
    (site / "torch-9.9.9.dist-info").mkdir(parents=True)
    (site / "torch-9.9.9.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 9.9.9\n")
    monkeypatch.setenv("PYTHONPATH", str(site))
    ok, info = ns["_teacher_cache_check"](tmp_path / "pf")
    assert ok, info
    assert info["torch"] == "9.9.9" and info["commits"] == 1 and info["volume_root"] == "/runs/teacher-cache"


def test_cache_env_names_what_each_package_reads(tmp_path):
    """kernels reads KERNELS_CACHE (kernels/hf_hub.py; c6 showed HF_KERNELS_CACHE ignored), cutlass-dsl reads
    CUTE_DSL_CACHE_DIR (base_dsl/cache_helpers.py), quack QUACK_CACHE_DIR (autotuner.py)."""
    ns = lifted()
    dirs, _ = ns["teacher_cache_seed"]("k", {"torch": "1"}, local=tmp_path / "l", root=tmp_path / "vol")
    env = ns["teacher_cache_env"](dirs)
    assert env == {"QUACK_CACHE_DIR": str(dirs["quack"]), "KERNELS_CACHE": str(dirs["hub"]),
                   "CUTE_DSL_CACHE_DIR": str(dirs["cute"])}
    assert '"HF_KERNELS_CACHE"' not in SRC.read_text()  # named in a comment only, never set
