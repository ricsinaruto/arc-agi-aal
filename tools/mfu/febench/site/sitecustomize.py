"""MFU-2 front-end bench hooks (FE_BENCH=1 only): the vLLM serve process records its command and environment (for
the same-container relaunch of the other arms); the EngineCore process samples its scheduler's running/waiting counts
and KV use once a second into /tmp/fe/engine-<pid>.jsonl."""
import os
import sys

if os.environ.get("FE_BENCH") == "1" and os.environ.get("FE_TS") == "1":
    try:
        import fe_ts
        fe_ts.install()
    except Exception:
        pass

if os.environ.get("FE_BENCH") == "1" and os.environ.get("FE_GC"):
    try:
        import fe_gc
        fe_gc.install()
    except Exception:
        pass

if os.environ.get("FE_BENCH") == "1":
    import json
    import threading
    import time

    if "serve" in sys.argv and "--served-model-name" in sys.argv:
        try:
            os.makedirs("/tmp/fe", exist_ok=True)
            with open(f"/tmp/fe/serve-{os.getpid()}.json", "w") as _f:
                json.dump({"argv": list(sys.argv), "orig_argv": list(getattr(sys, "orig_argv", [])),
                           "executable": sys.executable, "cwd": os.getcwd(), "env": dict(os.environ)}, _f)
        except Exception:
            pass

    def _engine_sampler():
        mod = "vllm.v1.core.sched.scheduler"
        while mod not in sys.modules:
            time.sleep(5)
        time.sleep(5)
        try:
            if b"EngineCore" not in open("/proc/self/cmdline", "rb").read():
                return
        except OSError:
            return
        import gc
        sched = None
        while sched is None:
            time.sleep(10)
            for o in gc.get_objects():
                try:
                    if hasattr(o, "running") and hasattr(o, "waiting") and hasattr(o, "kv_cache_manager") \
                            and "Scheduler" in type(o).__name__:
                        sched = o
                        break
                except Exception:
                    pass
        with open(f"/tmp/fe/engine-{os.getpid()}.jsonl", "a") as f:
            while True:
                try:
                    try:
                        kv = round(float(sched.kv_cache_manager.usage), 4)
                    except Exception:
                        kv = None
                    f.write(json.dumps([round(time.time(), 2), len(sched.running), len(sched.waiting), kv]) + "\n")
                    f.flush()
                except Exception as exc:
                    f.write(json.dumps(["err", str(exc)[:200]]) + "\n")
                time.sleep(1)

    def _profiler():
        """py-spy is denied in Modal containers: while /tmp/fe/prof.on names a future epoch, sample every thread's
        Python stack at 50 Hz; then write /tmp/fe/prof-<pid>-<tag>.json (stack -> count, thread names, cmdline)."""
        me = threading.get_ident()
        while True:
            time.sleep(1)
            try:
                spec = json.load(open("/tmp/fe/prof.on"))
            except Exception:
                continue
            until, tag = spec["until"], spec["tag"]
            if time.time() >= until or os.path.exists(f"/tmp/fe/prof-{os.getpid()}-{tag}.json"):
                continue
            counts, names, n = {}, {}, 0
            while time.time() < until:
                for tid, frame in sys._current_frames().items():
                    if tid == me:
                        continue
                    parts = []
                    while frame is not None and len(parts) < 48:
                        c = frame.f_code
                        parts.append(f"{c.co_name} ({'/'.join(c.co_filename.rsplit('/', 2)[-2:])}:{c.co_firstlineno})")
                        frame = frame.f_back
                    key = str(tid) + ";" + ";".join(reversed(parts))
                    counts[key] = counts.get(key, 0) + 1
                n += 1
                time.sleep(0.02)
            for t in threading.enumerate():
                names[str(t.ident)] = t.name
            try:
                cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:200]
            except OSError:
                cmd = ""
            with open(f"/tmp/fe/prof-{os.getpid()}-{tag}.json", "w") as f:
                json.dump({"pid": os.getpid(), "cmd": cmd, "ticks": n, "threads": names, "counts": counts}, f)

    def _digest_hook():
        """FE_DIGEST=1 (load bench): wrap DefaultModelLoader.load_weights; after it, a digest of every parameter and
        buffer of the model (two GPU checksums over its bytes) to /tmp/fe/digest-<pid>.json."""
        if os.environ.get("FE_DIGEST") != "1":
            return
        mod = "vllm.model_executor.model_loader.default_loader"
        while mod not in sys.modules:
            time.sleep(0.05)
        cls = getattr(sys.modules[mod], "DefaultModelLoader", None)
        while cls is None:
            time.sleep(0.05)
            cls = getattr(sys.modules[mod], "DefaultModelLoader", None)
        orig = cls.load_weights

        def load_weights(self, model, *a, **k):
            t0 = time.time()
            out = orig(self, model, *a, **k)
            t1 = time.time()
            try:
                _digest(model, t0, t1)
            except Exception as exc:  # noqa: BLE001 - instrumentation never breaks the load
                with open(f"/tmp/fe/digest-{os.getpid()}.err", "w") as f:
                    f.write(repr(exc))
            return out

        def _digest(model, t0, t1):
            import torch
            rows = {}
            with torch.no_grad():
                for name, t in list(model.named_parameters()) + list(model.named_buffers()):
                    try:
                        u = t.data.reshape(-1).view(torch.uint8) if t.data.is_contiguous() else \
                            t.data.contiguous().reshape(-1).view(torch.uint8)
                    except Exception as exc:  # noqa: BLE001
                        rows[name] = ["err", str(exc)[:100]]
                        continue
                    s1 = s2 = 0
                    step = 1 << 26
                    for i in range(0, u.numel(), step):
                        c = u[i:i + step].to(torch.int64)
                        w = torch.arange(i, i + c.numel(), device=c.device, dtype=torch.int64) % 65521 + 1
                        s1 += int(c.sum())
                        s2 += int((c * w).sum())
                    rows[name] = [str(t.dtype), list(t.shape), u.numel(), s1, s2]
            with open(f"/tmp/fe/digest-{os.getpid()}.json", "w") as f:
                json.dump({"load_weights_s": round(t1 - t0, 2), "digest_s": round(time.time() - t1, 2),
                           "params": rows}, f)

        cls.load_weights = load_weights

    def _start():  # the gc-scan engine sampler is off: it held the EngineCore GIL for ~1 s every 10 s (fe2)
        threading.Thread(target=_profiler, daemon=True).start()
        threading.Thread(target=_digest_hook, daemon=True).start()

    _start()
    os.register_at_fork(after_in_child=_start)  # a forked EngineCore gets its own sampler (threads do not survive fork)
