"""Virtual time for asyncio loop tests (25 Oct: the rl_async [True] cases flaked under CI load on real-time sleeps).

``vclock(monkeypatch, modules)`` makes every ``asyncio.run`` in the test use a loop whose clock only moves when every
task is waiting on a timer: it then jumps to the next timer. ``time.time()`` / ``time.monotonic()`` inside the listed
modules follow that clock, so wall-clock bars (block lengths, idle floors, deadlines) are measured in virtual seconds
and CPU load cannot reorder events. Work in threads (``asyncio.to_thread``) runs in real time; the clock waits for it.
"""
import asyncio
import selectors
import time as _time
import types


class VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selectors.SelectSelector())
        self._vt = 0.0
        self._threads = 0

    def time(self):
        return self._vt

    def run_in_executor(self, executor, func, *args):
        fut = super().run_in_executor(executor, func, *args)
        self._threads += 1

        def done(_):
            self._threads -= 1

        fut.add_done_callback(done)
        return fut

    async def shutdown_default_executor(self, timeout=None):  # asyncio.run's 300 s join bar would expire at once
        return await super().shutdown_default_executor(None)

    def _run_once(self):
        if not self._ready and self._scheduled:
            self._process_events(self._selector.select(0))  # I/O and thread completions first, without waiting
            while not self._ready and self._threads:  # real work in a thread: wait for it in real time
                self._process_events(self._selector.select(0.005))
            if not self._ready:
                head = min(h for h in self._scheduled if not h._cancelled) if any(
                    not h._cancelled for h in self._scheduled) else None
                if head is not None and head._when > self._vt:
                    self._vt = head._when
        super()._run_once()


class _Policy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        return VirtualLoop()


def vclock(monkeypatch, modules):
    """Install virtual time: new event loops are VirtualLoops and ``time`` in ``modules`` follows the running loop's
    clock (outside a loop: the real clock). Returns the previous policy; the caller restores it."""
    base_wall, base_mono = _time.time(), _time.monotonic()

    def offset():
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return None

    shim = types.SimpleNamespace(**{k: getattr(_time, k) for k in dir(_time) if not k.startswith("__")})
    shim.time = lambda: base_wall + o if (o := offset()) is not None else _time.time()
    shim.monotonic = lambda: base_mono + o if (o := offset()) is not None else _time.monotonic()
    shim.perf_counter = shim.monotonic
    for m in modules:
        monkeypatch.setattr(m, "time", shim)
    old = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(_Policy())
    return old


def fixture_modules():
    from arena import level_bench, rl_async, rl_loop, rl_single

    return [rl_async, rl_single, level_bench, rl_loop]
