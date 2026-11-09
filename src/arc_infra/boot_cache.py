"""Compile / autotune caches carried between play-server boots (RTX launch time, 23 Oct).

A fresh Nebius VM rebuilds torch.compile's AOT artifacts (~25-30 s), Triton / Inductor kernels and FlashInfer's
autotune results on every boot. ``restore`` unpacks ``boot-cache.tar`` from ARC_BOOT_CACHE_IN (an asset folder)
into the server's own cache roots before vLLM starts; ``save`` packs those roots into ``<out>/boot-cache.tar`` once
the server is ready (in a thread, after the fact, so the next launch can carry it as an asset). vLLM's AOT compile-cache
key does not see our runtime overlays (arena.serving clears the cache when it applies them), so the archive is named
by ``key``: the exact server argv plus every overlay identity in the working dir; only an archive with the same key is
restored. Members outside the cache roots, links and devices are refused."""

import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

NAME = "boot-cache.tar"


KEY_ENV = "ARC_BOOT_CACHE_KEY"
# argv flags that only place the server (transport, its name, memory shares): never compile inputs; vLLM's own AOT and
# autotune keys re-check every config factor that is one
KEY_SKIP_FLAGS = ("--port", "--host", "--served-model-name", "--kv-cache-memory-bytes", "--gpu-memory-utilization")


def stable_argv(argv):
    """The argv without KEY_SKIP_FLAGS and their values (``--flag value`` or ``--flag=value``)."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in KEY_SKIP_FLAGS:
            skip = True
            continue
        if arg.split("=", 1)[0] in KEY_SKIP_FLAGS and "=" in arg:
            continue
        out.append(arg)
    return out


# Fields an overlay file records about the container's state, not about the compiled code (arena.serving writes
# which stale compile dirs it found and cleared): keyed as their empty value, so run history never moves the key.
OBSERVED_FIELDS = ("cleared_compile_cache",)


def keyed_bytes(path):
    """The overlay file's bytes with OBSERVED_FIELDS reset to [] (json.dumps indent=2, as arena.serving writes them),
    so a file whose fields were already empty keys exactly as before (26 Oct: t4-v1 missed t3-v3's archive because
    serving-kernels-start.json listed the torch_compile_cache dir it cleared)."""
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(data, dict) or not any(data.get(f) for f in OBSERVED_FIELDS):
        return raw
    return json.dumps({k: ([] if k in OBSERVED_FIELDS else v) for k, v in data.items()}, indent=2).encode()


def key(argv, working_dir, environ=None):
    """16 hex of sha256 over the server argv and the overlay identities (*-patch.json, *-start.json) the working dir
    holds when the server starts. ARC_BOOT_CACHE_KEY=2 (stage H, 26 Oct): the argv without KEY_SKIP_FLAGS (N servers
    on N ports and a KV share change share one archive) plus the server's flag fingerprint (env_fingerprint: flags our
    patches read inside traced code, which v1 missed). Observed-state fields are excluded (keyed_bytes)."""
    environ = os.environ if environ is None else environ
    v2 = environ.get(KEY_ENV) == "2"
    h = hashlib.sha256(json.dumps(stable_argv(argv) if v2 else list(argv)).encode())
    for path in sorted(Path(working_dir).glob("*-patch.json")) + sorted(Path(working_dir).glob("*-start.json")):
        h.update(path.name.encode() + b"\0" + keyed_bytes(path))
    if v2:
        h.update(b"env\0" + env_fingerprint(environ).encode())
    return ("k2-" if v2 else "") + h.hexdigest()[:16]


def archive_name(cache_key=None):
    return NAME if cache_key is None else f"boot-cache-{cache_key}.tar"


def _roots(roots):
    return [Path(r).resolve() for r in roots]


def _inside(path, roots):
    p = Path(path).resolve()
    return any(p == r or r in p.parents for r in roots)


def restore(source_dir, roots, emit=print, cache_key=None):
    """Unpack ``<source_dir>/boot-cache-<key>.tar`` (absolute member paths) into ``roots``; returns a summary row."""
    archive = Path(source_dir) / archive_name(cache_key)
    if not archive.is_file():
        row = {"event": "boot_cache_restore", "found": False, "path": str(archive)}
        emit(json.dumps(row), flush=True)
        return row
    began, roots, files, refused = time.time(), _roots(roots), 0, 0
    with tarfile.open(archive) as tar:
        for member in tar:
            target = Path("/") / member.name.lstrip("/")
            if not (member.isfile() or member.isdir()) or not _inside(target, roots):
                refused += 1
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            with open(target, "wb") as handle:
                while chunk := source.read(1 << 20):
                    handle.write(chunk)
            os.chmod(target, member.mode & 0o777)
            os.utime(target, (member.mtime, member.mtime))  # carry() ranks AOT artifacts by their compile time
            files += 1
    row = {"event": "boot_cache_restore", "found": True, "name": archive.name, "files": files, "refused": refused,
           "bytes": archive.stat().st_size, "seconds": round(time.time() - began, 1)}
    emit(json.dumps(row), flush=True)
    return row


def save(out_dir, roots, emit=print, cache_key=None):
    """Pack the existing ``roots`` into ``<out_dir>/boot-cache-<key>.tar`` (a temporary name, then renamed)."""
    began = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = archive_name(cache_key)
    tmp = out / (name + ".part")
    files = 0
    with tarfile.open(tmp, "w") as tar:
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    tar.add(path, arcname=str(path).lstrip("/"), recursive=False)
                    files += 1
    tmp.replace(out / name)
    row = {"event": "boot_cache_save", "name": name, "files": files, "bytes": (out / name).stat().st_size,
           "seconds": round(time.time() - began, 1)}
    emit(json.dumps(row), flush=True)
    return row


CARRY_ENV = "ARC_BOOT_CACHE_CARRY"
# Not graph inputs: paths, run identity, cache plumbing, diagnostics (anything else ARC_* splits the AOT root).
FINGERPRINT_SKIP = ("ARC_RL1_", "ARC_BOOT_CACHE_", "ARC_VALIDATION_", "ARC_RUNTIME_CACHE", "ARC_REAL_NVIDIA_SMI",
                    "ARC_SK_CHECK_CACHE", "ARC_SERVER2_ENV", "ARC_PREFLIGHT", "ARC_COMPILE_KEEP", "ARC_LOAD_PROFILE",
                    "ARC_PYC_CACHE", "ARC_FORCE_CHROOT", "ARC_PATCH_CHECK", "ARC_SERVING_PATCH_CHECK", "ARC_RESTART_ACK",
                    "ARC_PLE_CHECK_SKIP", "ARC_DRAFT_LOAD_FILTER", "ARC_PINNED_LOAD_FIXED")  # the last two: load only
FINGERPRINT_EXTRA = ("ARENA_DENSE_KERNELS", "ARENA_SERVING_KERNELS", "ARENA_STACK_KERNELS", "ARENA_EXPERT_PRUNE",
                     "ARENA_EXPERT_OVERLAY")
CARRY_COMPILE = ("triton", "torchinductor", "cuda", "flashinfer")  # content-keyed caches under the compile root
# FlashInfer autotune results (T3 boot: 80 s of 609), under each fingerprint root; vLLM names their folder by the same
# aot_compile_hash_factors as the AOT graph (flashinfer_autotune_cache.flashinfer_autotune_cache_hash), so a carried
# result loads only where the graph would
CARRY_AUTOTUNE = "flashinfer_autotune_cache"
CARRY_CACHE = ("pyc",)  # bytecode, invalidated by source mtime/size (patched files recompile)
AOT_KEEP = 4  # newest AOT artifacts carried (two B300 SKUs x two configs)


def env_fingerprint(environ=None):
    """12 hex over the server's ARC_* flags (FINGERPRINT_SKIP aside) and the kernel-overlay ARENA_* settings. vLLM's AOT
    key covers its own env, config, torch/Inductor system (device name) and the traced files' contents, but not flags
    our patches read at import inside traced code: a graph traced under other flags
    would load silently. Server 2 inherits server 1's root (it loads server 1's graph)."""
    environ = os.environ if environ is None else environ
    items = sorted((k, v) for k, v in environ.items()
                   if (k.startswith("ARC_") and not k.startswith(FINGERPRINT_SKIP)) or k in FINGERPRINT_EXTRA)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12]


def vllm_cache_root(cache_root, environ=None):
    """ARC_BOOT_CACHE_CARRY=1: VLLM_CACHE_ROOT per flag fingerprint, inside the boot cache's CACHE_ROOT."""
    return Path(cache_root) / "vllm-env" / env_fingerprint(environ)


def _aot_stamps(members):
    """{<...>/torch_aot_compile/<hash>: newest model mtime} over tar members."""
    stamps = {}
    for m in members:
        parts = (Path("/") / m.name.lstrip("/")).parts
        # /<cache_root>/vllm-env/<fp>/torch_compile_cache/torch_aot_compile/<hash>/rank_*/model
        if m.isfile() and parts[-1] == "model" and "torch_aot_compile" in parts:
            at = parts.index("torch_aot_compile")
            if at + 1 < len(parts):
                key = Path(*parts[:at + 2])
                stamps[key] = max(stamps.get(key, 0), m.mtime)
    return stamps


def carry(source_dir, roots, emit=print, exclude=None, keep=AOT_KEEP, prefer=None, scan=3):
    """Key miss: restore only what stays valid across overlay changes: the AOT artifacts (vLLM re-checks traced sources
    and its key on load; the root carries the flag fingerprint), at most ``keep`` newest by compile time, with their
    Inductor caches, plus Triton/Inductor/CUDA kernel caches and bytecode. Source: the newest of the ``scan`` newest
    other archives holding a graph under ``prefer`` (this server's VLLM_CACHE_ROOT; other consumers save to the same
    folder), else the newest."""
    began = time.time()
    archives = sorted((p for p in Path(source_dir).glob("boot-cache-*.tar") if p.name != exclude),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        row = {"event": "boot_cache_carry", "found": False}
        emit(json.dumps(row), flush=True)
        return row
    cache_root, compile_root = (Path(r).resolve() for r in roots[:2])
    fixed = [compile_root / d for d in CARRY_COMPILE] + [cache_root / d for d in CARRY_CACHE]
    env_root = cache_root / "vllm-env"
    prefer = Path(prefer).resolve() if prefer else None
    chosen, members, stamps = None, None, None
    for archive in archives[:max(1, scan)]:
        with tarfile.open(archive) as tar:
            listed = [m for m in tar.getmembers() if m.isfile() or m.isdir()]
        found = _aot_stamps(listed)
        if chosen is None:
            chosen, members, stamps = archive, listed, found
        if prefer is not None and any(prefer in p.parents for p in found):
            chosen, members, stamps = archive, listed, found
            break
    ranked = sorted(stamps.items(), key=lambda kv: (prefer is not None and prefer in kv[0].parents, kv[1]),
                    reverse=True)
    aot = [p for p, _ in ranked if env_root in p.parents][:keep]
    tunes = sorted({Path(*parts[:parts.index(CARRY_AUTOTUNE) + 1]) for m in members
                    for parts in [(Path("/") / m.name.lstrip("/")).parts] if CARRY_AUTOTUNE in parts})
    tunes = [t for t in tunes if env_root in t.parents]
    wanted = fixed + aot + tunes
    files = 0
    with tarfile.open(chosen) as tar:
        for m in members:
            target = Path("/") / m.name.lstrip("/")
            if not any(target == w or w in target.parents for w in wanted) or not _inside(target, [cache_root, compile_root]):
                continue
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(m)
            with open(target, "wb") as handle:
                while chunk := source.read(1 << 20):
                    handle.write(chunk)
            os.chmod(target, m.mode & 0o777)
            os.utime(target, (m.mtime, m.mtime))
            files += 1
    row = {"event": "boot_cache_carry", "found": True, "name": chosen.name, "files": files,
           "aot": [p.name for p in aot], "autotune": len(tunes),
           "preferred": bool(prefer and any(prefer in p.parents for p in aot)),
           "seconds": round(time.time() - began, 1)}
    emit(json.dumps(row), flush=True)
    return row


def restore_from_environment(roots, cache_key, environ=None, emit=print):
    """ARC_BOOT_CACHE_IN=<dir>: restore before the server starts. Never raises (a failure only loses speed)."""
    environ = os.environ if environ is None else environ
    if not environ.get("ARC_BOOT_CACHE_IN"):
        return None
    try:
        row = restore(environ["ARC_BOOT_CACHE_IN"], roots, emit=emit, cache_key=cache_key)
        if not row["found"] and environ.get(CARRY_ENV) == "1":  # ARC_BOOT_CACHE_CARRY=1: the key-independent part
            row["carry"] = carry(environ["ARC_BOOT_CACHE_IN"], roots, emit=emit, exclude=archive_name(cache_key),
                                 prefer=vllm_cache_root(roots[0], environ))
        return row
    except Exception as error:  # noqa: BLE001
        emit(json.dumps({"event": "boot_cache_restore_error", "error": repr(error)[:300]}), flush=True)
        return None


def save_from_environment(roots, default_out, cache_key, environ=None, emit=print, wait=False):
    """ARC_BOOT_CACHE_SAVE=1: pack the caches into ARC_BOOT_CACHE_OUT (default: <ARC_VALIDATION_OUTPUT>/boot-cache, else
    the serving working dir) once the server is ready, in a detached process (rtxmark-v1, 23 Oct: a daemon thread died with
    serving_setup, which exits right after ready, and no archive was written); ``wait``: in this process (tests).
    Returns the process (or None when off). Never raises."""
    import subprocess
    import sys

    environ = os.environ if environ is None else environ
    if environ.get("ARC_BOOT_CACHE_SAVE") != "1":
        return None
    # the run's own output (synced whole at the end; the runner copies only *.json/*.log of the serving dir, which
    # dropped rtxmark-v2's archive), else the serving working dir
    run_out = environ.get("ARC_VALIDATION_OUTPUT")
    out = Path(environ.get("ARC_BOOT_CACHE_OUT") or (Path(run_out) / "boot-cache" if run_out else default_out))
    if wait:
        try:
            return save(out, roots, emit=emit, cache_key=cache_key)
        except Exception as error:  # noqa: BLE001
            emit(json.dumps({"event": "boot_cache_save_error", "error": repr(error)[:300]}), flush=True)
            return None
    try:
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "boot-cache-save.log", "ab")
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "save", str(out), cache_key or "",
                                 *map(str, roots)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, close_fds=True)
        emit(json.dumps({"event": "boot_cache_save_started", "pid": proc.pid, "out": str(out)}), flush=True)
        return proc
    except Exception as error:  # noqa: BLE001
        emit(json.dumps({"event": "boot_cache_save_error", "error": repr(error)[:300]}), flush=True)
        return None


def count_files(roots):
    return sum(1 for root in roots if Path(root).is_dir() for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())


def save_twice(out, roots, cache_key, after, sleep=time.sleep, emit=print):
    """The ready-time save, then (ARC_BOOT_CACHE_SAVE_AFTER=<s>) one more after ``after`` seconds of play when the roots
    gained files: kernels JIT-compiled lazily by the first requests (LoRA / MoE shapes the warm-up never ran). The
    second archive replaces the first by rename; a container that ends first keeps the ready-time one."""
    first = save(out, roots, emit=emit, cache_key=cache_key)
    if not after:
        return first, None
    sleep(float(after))
    if count_files(roots) <= first["files"]:
        emit(json.dumps({"event": "boot_cache_resave", "skipped": "no new files"}), flush=True)
        return first, None
    second = save(out, roots, emit=emit, cache_key=cache_key)
    emit(json.dumps({"event": "boot_cache_resave", "files": second["files"], "added": second["files"] - first["files"]}),
         flush=True)
    return first, second


if __name__ == "__main__":  # python boot_cache.py save <out> <key or ""> <root>...
    import sys

    if len(sys.argv) >= 4 and sys.argv[1] == "save":
        try:
            save_twice(sys.argv[2], sys.argv[4:], sys.argv[3] or None, os.environ.get("ARC_BOOT_CACHE_SAVE_AFTER"))
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"event": "boot_cache_save_error", "error": repr(error)[:300]}), flush=True)
            raise SystemExit(1)
