"""FE_TS=1 (with FE_BENCH=1): per-request timestamps through the vLLM front end and the EngineCore input path, into
/tmp/fe/ts-<pid>.jsonl (wall clock time.time(); `*_m` = time.monotonic()). Patches attach right after each target module
executes (meta-path hook), so they precede the renderer's make_async closures. Every patch is best-effort: a failed one
is logged ("patch" rows) and the server runs unpatched there.

API process, one "req" row per /chat/completions request (keys in order of the path):
  asgi0 ASGI call; h0/h1 OpenAIServingChat.create_chat_completion; arr0/arr1 render_chat_async (arr0 = vLLM's arrival);
  r0/r1 render_messages_async, p0/p1 parse_chat_messages_async inside it; tk0/tk1 tokenize_prompts_async;
  pe0/pe1 process_for_engine_async (mm processor); a0/a1 AsyncLLM.add_request; s0/s1 _add_request (send to engine);
  f_w/f_m/f_eng first output batch with this request (frontend wall/mono, engine monotonic stamp of the batch);
  z_w/z_eng its last; g0/g1 chat_completion_full_generator; asgi1 response sent; x = executor jobs
  [name, queue wait s, run s] (make_async); rid internal request id; pt prompt tokens; mm items.
  "loop" rows each second: event-loop lag (50 ms probe), executor queue depths, open requests, gc pauses,
  process_outputs calls/time and engine->frontend output lag.
EngineCore process: "eng" rows per request (dec0/dec1 msgpack decode, pre0/pre1 preprocess_add_request, add/add_m
  scheduler add); "estep" rows each second: steps, step time, input-queue handling time, gc pauses.
"""
import asyncio
import contextvars
import gc
import importlib.machinery
import json
import os
import sys
import threading
import time

OUT = os.environ.get("FE_TS_DIR", "/tmp/fe")
PATCHED = {}
REC = contextvars.ContextVar("fe_ts_rec", default=None)
BY_RID = {}
OPEN = set()
STATE = {"renderer": None, "sampler": False}
AGG = {"po_calls": 0, "po_outputs": 0, "po_s": 0.0, "lag_sum": 0.0, "lag_n": 0, "lag_max": 0.0,
       "gc_n": 0, "gc_s": 0.0, "gc_max": 0.0, "gc2_n": 0,
       "steps": 0, "step_s": 0.0, "step_max": 0.0, "inq_s": 0.0, "inq_max": 0.0}
_lock = threading.Lock()
_file = {}
_tl = threading.local()


def emit(kind, rec):
    try:
        line = json.dumps({"k": kind, **rec}, separators=(",", ":"), default=str)
        with _lock:
            f = _file.get(os.getpid())
            if f is None:
                os.makedirs(OUT, exist_ok=True)
                f = _file[os.getpid()] = open(f"{OUT}/ts-{os.getpid()}.jsonl", "a", buffering=1)
                f.write(json.dumps({"k": "clock", "pid": os.getpid(), "wall": time.time(), "mono": time.monotonic(),
                                    "argv": " ".join(sys.argv)[:200]}) + "\n")
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 - instrumentation never breaks serving
        pass


def take(keys):
    with _lock:
        out = {k: AGG[k] for k in keys}
        for k in keys:
            AGG[k] = 0 if isinstance(AGG[k], int) else 0.0
    return out


def _gc_cb(phase, info):
    if phase == "start":
        _tl.gc0 = time.perf_counter()
        return
    t0 = getattr(_tl, "gc0", None)
    if t0 is None:
        return
    d = time.perf_counter() - t0
    AGG["gc_n"] += 1
    AGG["gc_s"] += d
    AGG["gc_max"] = max(AGG["gc_max"], d)
    if info.get("generation") == 2:
        AGG["gc2_n"] += 1


def _rss_gib():
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30, 3)
    except Exception:  # noqa: BLE001
        return None


def stamp(key):
    rec = REC.get()
    if rec is not None:
        rec[key] = time.time()
    return rec


def wrap_async(cls, name, k0, k1, before=None):
    orig = getattr(cls, name)

    async def w(self, *a, **kw):
        rec = stamp(k0)
        if before is not None:
            before(self, rec)
        try:
            return await orig(self, *a, **kw)
        finally:
            if rec is not None:
                rec[k1] = time.time()

    w.__name__, w.__qualname__, w.__wrapped__ = orig.__name__, getattr(orig, "__qualname__", name), orig
    setattr(cls, name, w)


# --- API process -------------------------------------------------------------------------------------------------
def p_async_utils(m):
    orig = m.make_async
    from functools import partial

    def make_async(func, executor=None):
        name = getattr(func, "__name__", None) or getattr(getattr(func, "func", None), "__name__", "?")

        def _async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            rec = REC.get()
            if rec is None:
                return loop.run_in_executor(executor, partial(func, *args, **kwargs))
            ctx = contextvars.copy_context()
            sub = time.perf_counter()

            def run():
                t0 = time.perf_counter()
                try:
                    return ctx.run(func, *args, **kwargs)
                finally:
                    t1 = time.perf_counter()
                    rec.setdefault("x", []).append([name, round(t0 - sub, 4), round(t1 - t0, 4)])
            return loop.run_in_executor(executor, run)

        return _async_wrapper

    make_async.__wrapped__ = orig
    m.make_async = make_async


def p_starlette(m):
    cls = m.Starlette
    orig = cls.__call__

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not str(scope.get("path", "")).endswith("/chat/completions"):
            return await orig(self, scope, receive, send)
        rec = {"asgi0": time.time()}
        OPEN.add(id(rec))
        token = REC.set(rec)
        try:
            return await orig(self, scope, receive, send)
        finally:
            rec["asgi1"] = time.time()
            OPEN.discard(id(rec))
            REC.reset(token)
            emit("req", rec)

    cls.__call__ = __call__


def _start_sampler():
    if STATE["sampler"]:
        return
    STATE["sampler"] = True

    async def sampler():
        lags, last = [], time.time()
        while True:
            t = time.monotonic()
            await asyncio.sleep(0.05)
            lags.append(time.monotonic() - t - 0.05)
            if time.time() - last >= 1.0:
                last = time.time()
                r = STATE["renderer"]

                def q(ex):
                    try:
                        return ex._work_queue.qsize()
                    except Exception:  # noqa: BLE001
                        return None
                a = take(["po_calls", "po_outputs", "po_s", "lag_sum", "lag_n", "lag_max", "gc_n", "gc_s", "gc_max",
                          "gc2_n"])
                emit("loop", {"t": round(last, 3), "lag_max": round(max(lags), 4),
                              "lag_mean": round(sum(lags) / len(lags), 4), "ex_q": q(getattr(r, "_executor", None)),
                              "mm_q": q(getattr(r, "_mm_executor", None)), "open": len(OPEN),
                              "wait_first": sum("f_w" not in v for v in list(BY_RID.values())),
                              "po_calls": a["po_calls"], "po_outputs": a["po_outputs"], "po_s": round(a["po_s"], 4),
                              "out_lag_mean": round(a["lag_sum"] / a["lag_n"], 4) if a["lag_n"] else None,
                              "out_lag_max": round(a["lag_max"], 4), "gc_n": a["gc_n"], "gc_s": round(a["gc_s"], 4),
                              "gc_max": round(a["gc_max"], 4), "gc2_n": a["gc2_n"], "rss_gib": _rss_gib()})
                lags = []

    asyncio.get_running_loop().create_task(sampler())


def p_chat_serving(m):
    cls = m.OpenAIServingChat
    orig = cls.create_chat_completion

    async def create_chat_completion(self, request, raw_request=None, *a, **kw):
        _start_sampler()
        rec = REC.get()
        if rec is None:  # not through the patched ASGI entry
            rec = {}
            REC.set(rec)
        rec["h0"] = time.time()
        try:
            rec["n_msg"] = len(request.messages)
        except Exception:  # noqa: BLE001
            pass
        try:
            return await orig(self, request, raw_request, *a, **kw)
        finally:
            rec["h1"] = time.time()

    create_chat_completion.__wrapped__ = orig
    cls.create_chat_completion = create_chat_completion
    wrap_async(cls, "chat_completion_full_generator", "g0", "g1")


def p_renderer_base(m):
    cls = m.BaseRenderer
    wrap_async(cls, "render_chat_async", "arr0", "arr1")
    wrap_async(cls, "tokenize_prompts_async", "tk0", "tk1")
    wrap_async(cls, "process_for_engine_async", "pe0", "pe1")


def p_renderer_hf(m):
    def keep(self, rec):
        STATE["renderer"] = self
    wrap_async(m.HfRenderer, "render_messages_async", "r0", "r1", before=keep)
    orig = m.parse_chat_messages_async

    async def parse_chat_messages_async(*a, **kw):
        rec = stamp("p0")
        try:
            return await orig(*a, **kw)
        finally:
            if rec is not None:
                rec["p1"] = time.time()

    m.parse_chat_messages_async = parse_chat_messages_async


def p_async_llm(m):
    cls = m.AsyncLLM
    wrap_async(cls, "add_request", "a0", "a1")
    orig = cls._add_request

    async def _add_request(self, request, *a, **kw):
        rec = stamp("s0")
        if rec is not None:
            rid = request.request_id
            rec["rid"] = rid
            try:
                rec["pt"] = len(request.prompt_token_ids or [])
                rec["mm"] = len(request.mm_features or [])
            except Exception:  # noqa: BLE001
                pass
            BY_RID[rid] = rec
        try:
            return await orig(self, request, *a, **kw)
        finally:
            if rec is not None:
                rec["s1"] = time.time()

    cls._add_request = _add_request


def p_output_processor(m):
    cls = m.OutputProcessor
    orig = cls.process_outputs

    def process_outputs(self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None, *a, **kw):
        w, mo = time.time(), time.monotonic()
        if BY_RID:
            for o in engine_core_outputs:
                rec = BY_RID.get(o.request_id)
                if rec is None:
                    continue
                if "f_w" not in rec:
                    rec.update(f_w=w, f_m=mo, f_eng=engine_core_timestamp)
                if getattr(o, "finish_reason", None) is not None:
                    rec.update(z_w=w, z_eng=engine_core_timestamp)
                    BY_RID.pop(o.request_id, None)
        t0 = time.perf_counter()
        try:
            return orig(self, engine_core_outputs, engine_core_timestamp, iteration_stats, *a, **kw)
        finally:
            d = time.perf_counter() - t0
            AGG["po_calls"] += 1
            AGG["po_outputs"] += len(engine_core_outputs)
            AGG["po_s"] += d
            if engine_core_timestamp:
                lag = mo - engine_core_timestamp
                AGG["lag_sum"] += lag
                AGG["lag_n"] += 1
                AGG["lag_max"] = max(AGG["lag_max"], lag)

    process_outputs.__wrapped__ = orig
    cls.process_outputs = process_outputs


# --- EngineCore process ------------------------------------------------------------------------------------------
def p_serial(m):
    cls = m.MsgpackDecoder
    orig = cls.decode

    def decode(self, bufs):
        t0 = time.time()
        out = orig(self, bufs)
        if hasattr(out, "prompt_token_ids") and hasattr(out, "request_id"):
            _tl.dec = (t0, time.time())
        return out

    decode.__wrapped__ = orig
    cls.decode = decode


def _estep_thread():
    while True:
        time.sleep(1.0)
        a = take(["steps", "step_s", "step_max", "inq_s", "inq_max", "gc_n", "gc_s", "gc_max", "gc2_n"])
        emit("estep", {"t": round(time.time(), 3), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                       for k, v in a.items()}})


def p_core(m):
    pre_orig = m.EngineCore.preprocess_add_request

    def preprocess_add_request(self, request):
        t0 = time.time()
        out = pre_orig(self, request)
        dec = getattr(_tl, "dec", None)
        _tl.dec = None
        PENDING[request.request_id] = {"rid": request.request_id, "dec0": dec[0] if dec else None,
                                       "dec1": dec[1] if dec else None, "pre0": t0, "pre1": time.time(),
                                       "pt": len(request.prompt_token_ids or [])}
        return out

    preprocess_add_request.__wrapped__ = pre_orig
    m.EngineCore.preprocess_add_request = preprocess_add_request

    add_orig = m.EngineCore.add_request

    def add_request(self, request, *a, **kw):
        t, mo = time.time(), time.monotonic()
        if not STATE.get("estep"):  # the EngineCore process only (the API process imports this module too)
            STATE["estep"] = True
            threading.Thread(target=_estep_thread, daemon=True, name="fe-ts-estep").start()
        try:
            return add_orig(self, request, *a, **kw)
        finally:
            row = PENDING.pop(request.request_id, None) or {"rid": request.request_id}
            row.update(add=t, add_m=mo, add1=time.time())
            try:
                row["inq"] = self.input_queue.qsize()
            except Exception:  # noqa: BLE001
                pass
            emit("eng", row)

    add_request.__wrapped__ = add_orig
    m.EngineCore.add_request = add_request

    def timed(cls, name, total, peak, count=None):
        orig = getattr(cls, name)

        def w(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **kw)
            finally:
                d = time.perf_counter() - t0
                AGG[total] += d
                AGG[peak] = max(AGG[peak], d)
                if count:
                    AGG[count] += 1

        w.__wrapped__ = orig
        setattr(cls, name, w)

    timed(m.EngineCore, "step_with_batch_queue", "step_s", "step_max", "steps")
    timed(m.EngineCore, "step", "step_s", "step_max", "steps")
    timed(m.EngineCoreProc, "_process_input_queue", "inq_s", "inq_max")


PENDING = {}
PATCHES = {
    "vllm.utils.async_utils": p_async_utils,
    "starlette.applications": p_starlette,
    "vllm.entrypoints.openai.chat_completion.serving": p_chat_serving,
    "vllm.renderers.base": p_renderer_base,
    "vllm.renderers.hf": p_renderer_hf,
    "vllm.v1.engine.async_llm": p_async_llm,
    "vllm.v1.engine.output_processor": p_output_processor,
    "vllm.v1.serial_utils": p_serial,
    "vllm.v1.engine.core": p_core,
}


def _apply(name, module):
    try:
        PATCHES[name](module)
        PATCHED[name] = True
    except Exception as exc:  # noqa: BLE001
        PATCHED[name] = f"{type(exc).__name__}: {exc}"[:200]
    emit("patch", {"module": name, "ok": PATCHED[name]})


class _Finder:
    @staticmethod
    def find_spec(name, path, target=None):
        if name not in PATCHES or name in PATCHED:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            _apply(name, module)

        spec.loader.exec_module = exec_module
        return spec


def install():
    for name in PATCHES:  # modules imported before this ran (none in a fresh interpreter)
        if name in sys.modules:
            _apply(name, sys.modules[name])
    sys.meta_path.insert(0, _Finder)
    gc.callbacks.append(_gc_cb)
