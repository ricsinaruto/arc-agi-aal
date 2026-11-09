"""ARC (serving_fast.api_gc): Python GC thresholds set right after vLLM freezes its heap (``freeze_gc_heap``: API
server, engine core, GPU worker). MFU-2 ts1/gc1 (23 Oct, B300, 75 streams): at the default (700, 10, 10) the API
process spent 0.56 s/s in GC (1.6 full collections/s, pauses to 0.69 s) and requests waited 4.1 s before reaching
the engine; gen0 50,000 cut that to 0.54 s.

``ARC_API_GC="<gen0>,<gen1>,<gen2>,<full every s>"`` (default below; gen2 1e9 = no automatic full collection, then
<full every s> > 0 runs gc.collect() that often). Every ``ARC_API_GC_LOG_S`` seconds (default 60, 0 = off) each
tuned process prints one ``ARC_API_GC {json}`` line: t (epoch), pid, proc, rss_gib, gc_s (seconds in GC per second),
gc_max_s, full_per_min. Installed as vllm/arc_api_gc.py."""

import gc
import json
import multiprocessing
import os
import threading
import time

DEFAULT = "50000,10,10,0"
_STATE = {"applied": False}


def parse(text):
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("ARC_API_GC is <gen0>,<gen1>,<gen2>,<full every s>")
    g0, g1, g2, period = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
    if min(g0, g1, g2) < 1 or period < 0:
        raise ValueError("ARC_API_GC thresholds >= 1, period >= 0")
    return g0, g1, g2, period


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


def tune(environ=None, start_threads=True):
    """Set the thresholds once per process (after gc.freeze); start the full-collect and log threads."""
    environ = os.environ if environ is None else environ
    if _STATE["applied"]:
        return False
    g0, g1, g2, period = parse(environ.get("ARC_API_GC") or DEFAULT)
    log_s = float(environ.get("ARC_API_GC_LOG_S", "60"))
    before = gc.get_threshold()
    gc.set_threshold(g0, g1, g2)
    stats = Stats()
    gc.callbacks.append(stats.callback)
    _STATE.update(applied=True, before=before, period=period, stats=stats)
    print("ARC_API_GC " + json.dumps({"pid": os.getpid(), "proc": multiprocessing.current_process().name,
                                      "before": list(before), "after": list(gc.get_threshold()),
                                      "full_every_s": period}), flush=True)
    if not start_threads:
        return True

    def full():
        while True:
            time.sleep(period)
            gc.collect()

    def log():
        while True:
            time.sleep(log_s)
            print("ARC_API_GC " + json.dumps(row(stats, log_s)), flush=True)

    if period > 0:
        threading.Thread(target=full, daemon=True, name="arc-api-gc-full").start()
    if log_s > 0:
        threading.Thread(target=log, daemon=True, name="arc-api-gc-log").start()
    return True
