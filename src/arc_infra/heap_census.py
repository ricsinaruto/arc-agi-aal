"""Heap census of the vLLM processes (real-v7 27 Oct: EngineCore/API heaps grew ~20 GiB per M generated tokens; a
CPU repro of per-request routed arrays under glibc stayed flat, so something is retained). ``install(python)`` puts
``arc_heap_census.py`` and a ``.pth`` that imports it into that interpreter's site-packages; every process of that
venv started with HEAP_CENSUS_DIR set (a server's card_env; not ARC_*, so the compile-cache fingerprint is unchanged)
appends a row every HEAP_CENSUS_EVERY_S to ``<dir>/heap-<pid>.jsonl``: RSS and pymalloc blocks; HEAP_CENSUS_TRACE=N
starts tracemalloc (N frames) at interpreter start, or HEAP_CENSUS_TRACE_AFTER_S later, and adds the top allocation sites by line and by stack (numpy
buffers included); HEAP_CENSUS_TRACE_MATCH=s traces only processes whose cmdline (now or at start) contains s; HEAP_CENSUS_TRIM=1 also calls malloc_trim(0) and records RSS after it. Never walks objects. Off without the variable; never raises into the host process."""
import subprocess
from pathlib import Path

MODULE = r'''
import os
if os.environ.get("HEAP_CENSUS_DIR"):
    import threading

    def _rss():
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS"):
                    return round(int(line.split()[1]) / 2**20, 3)
        except OSError:
            return None

    def _cmd():  # the process title as ps shows it (vLLM's setproctitle: VLLM::EngineCore, VLLM::Worker...)
        try:
            return open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            return ""

    _cmd0 = _cmd()  # at interpreter start, before a setproctitle (L1b 28 Oct: "spawn_main" never matched VLLM::EngineCore)
    _match = os.environ.get("HEAP_CENSUS_TRACE_MATCH") or ""
    _frames = int(os.environ.get("HEAP_CENSUS_TRACE") or 0)
    _after = float(os.environ.get("HEAP_CENSUS_TRACE_AFTER_S") or 0)
    if _frames > 0 and _after <= 0 and not _match:  # from interpreter start: every later allocation is attributed
        import tracemalloc
        tracemalloc.start(_frames)

    def _native():
        """Native candidates at no throttle cost: the smaps split (gVisor has no smaps_rollup),
        mapping count and the biggest anonymous mappings, and torch's pinned host allocator if torch is loaded."""
        out = {}
        try:
            kb = {"anon": 0, "shmem": 0, "file": 0}
            maps, big, cur = 0, [], None
            for line in open("/proc/self/smaps"):
                head = line.split()
                if not head:
                    continue
                if "-" in head[0] and not head[0].endswith(":"):
                    maps += 1
                    cur = [head[-1] if len(head) >= 6 else "[anon]", 0]
                elif head[0] == "Anonymous:":
                    kb["anon"] += int(head[1])
                    if cur is not None:
                        cur[1] += int(head[1])
                        if int(head[1]) >= 262144:  # >= 256 MiB anonymous in one mapping
                            big.append(cur)
                elif head[0] == "Rss:" and cur is not None and ("/dev/shm" in cur[0] or "SYSV" in cur[0] or "memfd" in cur[0]):
                    kb["shmem"] += int(head[1])
                elif head[0] == "Rss:" and cur is not None and cur[0].startswith("/"):
                    kb["file"] += int(head[1])
            out["smaps_gib"] = {k: round(v / 2**20, 3) for k, v in kb.items()}
            out["maps"] = maps
            big = sorted(big, key=lambda x: -x[1])[:5]
            out["big_anon"] = [(n[:40], round(v / 2**20, 2)) for n, v in big]
        except Exception as error:
            out["smaps_error"] = repr(error)[:80]
        import sys
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                fn = getattr(getattr(torch.cuda, "memory", None), "host_memory_stats", None) or getattr(
                    torch._C, "_cuda_hostMemoryStats", None)
                st = fn() if fn else None
                if st:
                    out["pinned"] = {k: (round(v / 2**30, 3) if "bytes" in k else v) for k, v in st.items()
                                     if isinstance(v, (int, float)) and any(w in k for w in ("allocated", "reserved", "segment", "alloc"))
                                     and ("current" in k or "peak" in k or "." not in k)}
            except Exception as error:
                out["pinned_error"] = repr(error)[:80]
        return out

    def _row(top=25, big=10):
        # No object walk: gc.get_objects() from this thread held a reference to every tracked object and broke
        # _PyTuple_Resize's refcount-1 rule in the main thread (heap legs A/B 27 Oct: SystemError tupleobject.c:909).
        import sys, time
        row = {"t": time.time(), "pid": os.getpid(), "argv": " ".join(sys.argv)[:120], "rss_gib": _rss(),
               "blocks": sys.getallocatedblocks()}
        cmd = _cmd()
        row["cmd"] = cmd[:80]
        row.update(_native())  # every row, untraced too: the growth is native (L1b: pymalloc blocks flat, RSS up)
        if _frames > 0 and _match and _match not in cmd and _match not in _cmd0:
            return row  # HEAP_CENSUS_TRACE_MATCH: trace only the processes whose cmdline has it (the API stalled traced)
        if _frames > 0:
            import tracemalloc
            if not tracemalloc.is_tracing():  # HEAP_CENSUS_TRACE_AFTER_S: tracing from then on (heap legs 27 Oct:
                tracemalloc.start(_frames)    # traced from start, torch.compile took ~4x and the boot ~35 min)
                row["trace_started"] = True
                return row
            cur, peak = tracemalloc.get_traced_memory()
            row["trace_own_gib"] = round(tracemalloc.get_tracemalloc_memory() / 2**30, 3)  # tracing's own RSS share
            snap = tracemalloc.take_snapshot()
            row.update(traced_gib=round(cur / 2**30, 3), traced_peak_gib=round(peak / 2**30, 3),
                       lines=[(round(st.size / 2**20, 2), st.count, f"{st.traceback[0].filename}:{st.traceback[0].lineno}")
                              for st in snap.statistics("lineno")[:top]],
                       stacks=[(round(st.size / 2**20, 2), st.count,
                                [f"{f.filename}:{f.lineno}" for f in st.traceback])
                               for st in snap.statistics("traceback")[:big]])
            del snap
        return row

    def _loop():
        import json, time
        every = float(os.environ.get("HEAP_CENSUS_EVERY_S") or 300)
        path = os.path.join(os.environ["HEAP_CENSUS_DIR"], f"heap-{os.getpid()}.jsonl")
        time.sleep(float(os.environ.get("HEAP_CENSUS_FIRST_S") or (_after if _after > 0 else min(every, 60.0))))
        while True:
            try:
                row = _row()
                if os.environ.get("HEAP_CENSUS_TRIM") == "1":  # free-but-unreturned heap: RSS across malloc_trim(0)
                    try:
                        import ctypes
                        t0 = time.time()
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                        row.update(trim_s=round(time.time() - t0, 3), rss_after_trim_gib=_rss())
                    except Exception as error:
                        row["trim_error"] = repr(error)[:120]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception:
                pass
            time.sleep(every)

    threading.Thread(target=_loop, name="arc-heap-census", daemon=True).start()
'''


def site_packages(python):
    out = subprocess.run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def install(python, site=None):
    """Idempotent; returns the site-packages directory written to."""
    site = Path(site) if site is not None else site_packages(python)
    (site / "arc_heap_census.py").write_text(MODULE)
    (site / "arc_heap_census.pth").write_text("import arc_heap_census\n")
    return site


def install_for(argv, env):
    """Before a server's Popen: installs into argv[0]'s venv when its env asks for the census; never raises."""
    if not env.get("HEAP_CENSUS_DIR") or not argv:
        return None
    try:
        return str(install(argv[0]))
    except Exception:  # noqa: BLE001 - a diagnostic only
        return None
