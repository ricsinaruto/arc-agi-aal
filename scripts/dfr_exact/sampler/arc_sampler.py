"""In-process stack sampler, loaded as a vLLM general plugin in every server process (22 Oct: py-spy 0.4.x cannot
attach to the GPU worker). A daemon thread waits for trigger files go-<tag> in ARC_SAMPLER_DIR; on each it samples the
main thread's Python stack every ~4 ms for 60 s and writes collapsed stacks (one 'frame;frame;... weight' per line).
The switch interval drops to 1 ms during a window so a busy main thread yields the GIL to the sampler; each sample is
weighted by the microseconds since the previous one (a busy thread is reached later than a waiting one)."""
import collections
import os
import sys
import threading
import time

DIR = os.environ.get("ARC_SAMPLER_DIR", "")
SECONDS = 60.0
PERIOD = 0.004
_started = False


def _name(code, cache={}):
    key = id(code)
    hit = cache.get(key)
    if hit is None or hit[0] is not code:
        parts = code.co_filename.rsplit("/", 2)
        hit = cache[key] = (code, f"{code.co_name} ({'/'.join(parts[-2:])})")
    return hit[1]


def _sample(tag, main):
    counts = collections.Counter()
    lines = collections.Counter()
    end = time.monotonic() + SECONDS
    t0 = last = time.monotonic()
    n = 0
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.001)
    try:
        while time.monotonic() < end:
            frame = sys._current_frames().get(main)
            now = time.monotonic()
            weight = int((now - last) * 1e6)
            last = now
            stack = []
            top = None
            while frame is not None:
                if top is None:
                    top = f"{_name(frame.f_code)}:{frame.f_lineno}"
                stack.append(_name(frame.f_code))
                frame = frame.f_back
            counts[";".join(reversed(stack))] += weight
            lines[top] += weight
            n += 1
            time.sleep(PERIOD)
    finally:
        sys.setswitchinterval(old_interval)
    try:
        proc = open("/proc/self/cmdline", "rb").read().split(b"\0")[0].decode(errors="replace")[:60]
    except OSError:
        proc = "?"
    safe = "".join(c if c.isalnum() else "_" for c in proc)[:40]
    path = os.path.join(DIR, f"stacks-{tag}-{os.getpid()}-{safe}.txt")
    with open(path + ".part", "w") as f:
        f.write(f"# pid {os.getpid()} proc {proc} samples {n} seconds {time.monotonic() - t0:.1f}\n")
        for stack, c in counts.most_common():
            f.write(f"{stack} {c}\n")
        f.write("# top lines\n")
        for line, c in lines.most_common(300):
            f.write(f"#L {c} {line}\n")
    os.replace(path + ".part", path)


AB_FLAG = "/tmp/arc-ab-flag"
AUDIT_SECONDS = 60.0


def _sync_audit(tag):
    """60 s of torch.cuda sync debug mode (warn): every synchronizing CUDA op becomes a warning; each is counted by its
    Python stack (innermost 10 frames) and the A/B flag state. Plus CUDA caching-allocator counter deltas."""
    import json
    import traceback
    import warnings
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    counts = collections.Counter()
    by_flag = collections.Counter()
    old_show = warnings.showwarning
    old_filters = list(warnings.filters)

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return old_show(message, category, filename, lineno, file, line)
        stack = [f for f in traceback.extract_stack()[:-1] if not f.filename.endswith("warnings.py")]
        key = ";".join(f"{f.filename.rsplit('/', 2)[-2]}/{f.filename.rsplit('/', 1)[-1]}:{f.name}:{f.lineno}"
                       for f in stack[-10:])
        on = os.path.exists(AB_FLAG)
        counts[key] += 1
        by_flag[str(on)] += 1

    keys = ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_sync_all_streams", "num_ooms")
    before = {k: torch.cuda.memory_stats().get(k) for k in keys}
    warnings.filterwarnings("always", message=".*synchroniz.*")
    warnings.showwarning = show
    t0 = time.monotonic()
    torch.cuda.set_sync_debug_mode(1)
    try:
        time.sleep(AUDIT_SECONDS)
    finally:
        torch.cuda.set_sync_debug_mode(0)
        warnings.showwarning = old_show
        warnings.filters[:] = old_filters
    after = {k: torch.cuda.memory_stats().get(k) for k in keys}
    path = os.path.join(DIR, f"syncs-{tag}-{os.getpid()}.json")
    with open(path + ".part", "w") as f:
        json.dump({"pid": os.getpid(), "seconds": round(time.monotonic() - t0, 1), "total": sum(counts.values()),
                   "by_flag": by_flag, "allocator_before": before, "allocator_after": after,
                   "stacks": counts.most_common(300)}, f, indent=1)
    os.replace(path + ".part", path)


def _audit_loop(at):
    t0 = None
    while t0 is None:  # play start = the trial loop's first A/B flag
        if os.path.exists(AB_FLAG):
            t0 = time.monotonic()
        time.sleep(0.5)
    for s in at:
        delay = t0 + s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        try:
            _sync_audit(str(int(s)))
        except Exception as error:  # never fatal to the server
            try:
                open(os.path.join(DIR, f"error-sync-{int(s)}-{os.getpid()}.txt"), "w").write(repr(error))
            except OSError:
                pass


def register():
    global _started
    if not DIR or _started:
        return
    _started = True
    main = threading.main_thread().ident
    try:  # load marker: proves the plugin ran in this process
        os.makedirs(DIR, exist_ok=True)
        open(os.path.join(DIR, f"loaded-{os.getpid()}"), "w").write(" ".join(sys.argv)[:200])
    except OSError:
        pass

    def loop():
        done = set()
        while True:
            time.sleep(1.0)
            try:
                tags = [n[3:] for n in os.listdir(DIR) if n.startswith("go-") and n not in done]
            except OSError:
                continue
            for tag in tags:
                done.add("go-" + tag)
                try:
                    _sample(tag, main)
                except Exception as error:  # never fatal to the server
                    try:
                        open(os.path.join(DIR, f"error-{tag}-{os.getpid()}.txt"), "w").write(repr(error))
                    except OSError:
                        pass

    threading.Thread(target=loop, daemon=True, name="arc-sampler").start()
    at = sorted(float(x) for x in os.environ.get("ARC_SYNC_AUDIT_AT", "").split(",") if x.strip())
    if at:
        threading.Thread(target=_audit_loop, args=(at,), daemon=True, name="arc-sync-audit").start()
