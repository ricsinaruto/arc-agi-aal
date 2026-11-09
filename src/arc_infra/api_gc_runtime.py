"""ARC (serving_fast.api_gc): Python GC thresholds set right after vLLM freezes its heap (``freeze_gc_heap``: API
server, engine core, GPU worker). MFU-2 ts1/gc1 (23 Oct, B300, 75 streams): at the default (700, 10, 10) the API
process spent 0.56 s/s in GC (1.6 full collections/s, pauses to 0.69 s) and requests waited 4.1 s before reaching
the engine; gen0 50,000 cut that to 0.54 s.

Thresholds (50000, 10, 10), no timed full collection. Every LOG_S seconds each tuned process prints one ``ARC_API_GC {json}`` line: t (epoch), pid, proc, rss_gib, gc_s (seconds in GC per second),
gc_max_s, full_per_min. Installed as vllm/arc_api_gc.py."""

import gc
import json
import multiprocessing
import os
import threading
import time

THRESHOLDS = (50000, 10, 10)
LOG_S = 60.0
_STATE = {"applied": False}


def rss_gib():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes, peak (tests only)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.start = None
        self.s = self.max = 0.0
        self.full = 0

    def callback(self, phase, info):
        if phase == "start":
            self.start = time.perf_counter()
        elif self.start is not None:
            took = time.perf_counter() - self.start
            self.start = None
            with self.lock:
                self.s += took
                self.max = max(self.max, took)
                self.full += info.get("generation") == 2

    def take(self):
        with self.lock:
            out = (self.s, self.max, self.full)
            self.s = self.max = 0.0
            self.full = 0
        return out


def row(stats, seconds):
    s, worst, full = stats.take()
    return {"t": round(time.time(), 1), "pid": os.getpid(), "proc": multiprocessing.current_process().name,
            "rss_gib": round(rss_gib(), 3),
            "gc_s": round(s / seconds, 4), "gc_max_s": round(worst, 4), "full_per_min": round(full * 60 / seconds, 2),
            "threshold": list(gc.get_threshold())}


def tune(start_threads=True):
    """Set the thresholds once per process (after gc.freeze); start the log thread."""
    if _STATE["applied"]:
        return False
    before = gc.get_threshold()
    gc.set_threshold(*THRESHOLDS)
    stats = Stats()
    gc.callbacks.append(stats.callback)
    _STATE.update(applied=True, before=before, stats=stats)
    print("ARC_API_GC " + json.dumps({"pid": os.getpid(), "proc": multiprocessing.current_process().name,
                                      "before": list(before), "after": list(gc.get_threshold()),
                                      "full_every_s": 0}), flush=True)
    if not start_threads:
        return True

    def log():
        while True:
            time.sleep(LOG_S)
            print("ARC_API_GC " + json.dumps(row(stats, LOG_S)), flush=True)

    threading.Thread(target=log, daemon=True, name="arc-api-gc-log").start()
    return True
