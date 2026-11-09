"""FE_GC="<gen0>,<gen1>,<gen2>,<full every s>" (with FE_BENCH=1): API-process GC tuning. vLLM's API server freezes its
heap at start (vllm.utils.gc_utils.freeze_gc_heap); right after that the thresholds are set to (gen0, gen1, gen2)
(gen2 = 1e9 turns automatic full collections off) and, when <full every s> > 0, a daemon thread runs gc.collect()
that often so cyclic garbage in old objects is still freed. Only the process that calls freeze_gc_heap (the API server)
is changed. One "gc_tuned" row goes to /tmp/fe/ts-<pid>.jsonl (fe_ts writer) or stderr."""
import gc
import importlib.machinery
import os
import sys
import threading
import time

STATE = {"applied": False}


def spec():
    g0, g1, g2, period = (os.environ.get("FE_GC") or "50000,20,1000000000,10").split(",")
    return int(g0), int(g1), int(g2), float(period)


def apply_now(note=print):
    if STATE["applied"]:
        return False
    g0, g1, g2, period = spec()
    before = gc.get_threshold()
    gc.set_threshold(g0, g1, g2)
    STATE.update(applied=True, before=before, after=gc.get_threshold(), period=period, full=[])

    def full():
        while True:
            time.sleep(period)
            t0 = time.perf_counter()
            n = gc.collect()
            STATE["full"].append((round(time.time(), 1), round(time.perf_counter() - t0, 3), n))

    if period > 0:
        threading.Thread(target=full, daemon=True, name="fe-gc-full").start()
    note({"pid": os.getpid(), "gc_before": before, "gc_after": gc.get_threshold(), "full_every_s": period})
    return True


def _emit(row):
    try:
        import fe_ts
        fe_ts.emit("gc_tuned", row)
    except Exception:  # noqa: BLE001
        print("FE_GC " + repr(row), file=sys.stderr, flush=True)


def p_gc_utils(module):
    orig = module.freeze_gc_heap

    def freeze_gc_heap(*a, **kw):
        out = orig(*a, **kw)
        apply_now(_emit)
        return out

    freeze_gc_heap.__wrapped__ = orig
    module.freeze_gc_heap = freeze_gc_heap


def p_server_utils(module):  # server_utils imported freeze_gc_heap by name before our wrap: rebind it there too
    gcu = sys.modules.get("vllm.utils.gc_utils")
    if gcu is not None and hasattr(module, "freeze_gc_heap"):
        module.freeze_gc_heap = gcu.freeze_gc_heap


PATCHES = {"vllm.utils.gc_utils": p_gc_utils, "vllm.entrypoints.serve.utils.server_utils": p_server_utils}
DONE = set()


class _Finder:
    @staticmethod
    def find_spec(name, path, target=None):
        if name not in PATCHES or name in DONE:
            return None
        found = importlib.machinery.PathFinder.find_spec(name, path)
        if found is None or found.loader is None or not hasattr(found.loader, "exec_module"):
            return found
        orig = found.loader.exec_module

        def exec_module(module):
            orig(module)
            DONE.add(name)
            try:
                PATCHES[name](module)
            except Exception as exc:  # noqa: BLE001
                _emit({"patch_error": name, "error": repr(exc)[:200]})

        found.loader.exec_module = exec_module
        return found


def install():
    sys.meta_path.insert(0, _Finder)
