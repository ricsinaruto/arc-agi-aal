"""Multi-process arena drivers for rl_seq (``mg.redesign.drivers`` = N > 0; 0 keeps the in-process path).

On a B300 one asyncio process drove ~280 tries (one level bench per group) at ~85% CPU and the engine saw 29-207 of
them running. Here N spawned worker processes each build their own Loop (same study, trial and paths) and play the
groups the controller sends them; the backlog, hook, harvest, pump, phases and trainer stay in the controller.

Static partition, no per-request IPC: each worker has its own slot pool per server URL (ceil(active_tries / N)
slots), its own KV gate at 1/N of the server's capacity (SharedSlots.capacity_share) and its own PhaseGate. The
controller's ``ProxyGate`` fans PhaseController's close/open out to every worker and returns once each has acked
(closed: its in-flight requests aborted and drained, try clocks paused).

IPC: one Pipe per worker, pickled dicts, read with ``add_reader`` on the connection's fd and sent from a thread, so
neither side's event loop blocks on the other.
  C->W: play, gate (closed/open + epoch), cancel, knobs, stop.
  W->C: ready, try_done (-> the run's PROGRESS hook), group_done, group_error (deadline: Seq.play's TimeoutError
  path), gate_ack, feed (-> the one feed.jsonl), state (every STATE_S s: pools, loop lag), fatal.
A worker whose pipe closes, whose process exits or that sends nothing for DEAD_S s is dead: a ``driver died`` alert,
its groups fail (Seq.play's crashed-bench path), it is killed. With no live worker the run stops (``drivers_dead``).
"""
import asyncio
import importlib
import math
import multiprocessing
import os
import queue
import threading
import time
import weakref
from pathlib import Path

from . import level_bench, phase

DEAD_S = 30.0  # no message (state rows come every STATE_S) for this long: the worker is dead
STATE_S = 5.0
READY_S = 180.0  # a worker's start (spawn, imports, its Loop)
STOP_S = 10.0  # a worker's own exit after "stop", before terminate / kill
DRIVER_INDEX = None  # in a worker process: its index (factories may read it)


class DriverError(RuntimeError):
    """A group a driver could not play (its bench failed, or the driver died)."""


def count(cfg):
    n = (cfg or {}).get("drivers") or 0
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("mg.redesign.drivers is a non-negative integer (0: in-process play)")
    return n


def loop_args(loop):
    """The constructor arguments of ``loop`` (rl_loop.Loop): a worker builds its own Loop from them."""
    return {"study": loop.study, "trial": loop.trial,
            "kwargs": {"source": str(loop.source), "games_dir": loop.games_dir, "output": str(loop.output),
                       "server_dir": loop.server_dir, "analysis_python": loop.analysis_python,
                       "base_url": loop.base_url, "deadline_epoch": loop.deadline}}


def real_loop(study, trial, kwargs):
    from .rl_loop import Loop

    return Loop(study, trial, **kwargs)


def resolve(factory):
    if callable(factory):
        return factory
    module, name = factory.split(":")
    return getattr(importlib.import_module(module), name)


class Channel:
    """One end of a worker pipe: sends from a thread (a full pipe never blocks the event loop); reads on the loop."""

    def __init__(self, conn):
        self.conn, self.out, self.fd, self.aloop = conn, queue.SimpleQueue(), conn.fileno(), None
        self.thread = threading.Thread(target=self._send, daemon=True, name="rl-driver-send")
        self.thread.start()

    def _send(self):
        while (msg := self.out.get()) is not None:
            try:
                self.conn.send(msg)
            except (OSError, ValueError):  # the other end is gone (its reader sees EOF)
                return

    def send(self, msg):
        self.out.put(msg)

    def listen(self, aloop, on_msg, on_eof):
        self.aloop = aloop

        def ready():
            try:
                for _ in range(256):  # then other callbacks run; the reader fires again while data waits
                    if not self.conn.poll():
                        break
                    msg = self.conn.recv()
                    try:
                        on_msg(msg)
                    except Exception as exc:  # noqa: BLE001 - one bad message never stops the reader
                        print(f"rl_drivers: message {msg.get('op') if isinstance(msg, dict) else '?'} failed: "
                              f"{exc!r}"[:300], flush=True)
            except (EOFError, OSError):
                self.unlisten()
                on_eof()

        aloop.add_reader(self.fd, ready)

    def unlisten(self):
        if self.aloop is not None:
            try:
                self.aloop.remove_reader(self.fd)
            except (ValueError, OSError, RuntimeError):
                pass
            self.aloop = None

    def close(self, join=1.0):
        self.unlisten()
        self.out.put(None)
        self.thread.join(join)
        try:
            self.conn.close()
        except OSError:
            pass


# ---- worker side -------------------------------------------------------------------------------------------------

class ForwardFeed:
    """A worker Loop's feed: every row goes to the controller's feed.jsonl."""

    def __init__(self, ch):
        self.ch, self.status = ch, {}

    def emit(self, kind, **data):
        self.last = time.time()
        self.ch.send({"op": "feed", "kind": kind, "fields": data})
        return {"kind": kind, **data}


def worker_entry(conn, index, n, factory, args, setup):
    global DRIVER_INDEX
    DRIVER_INDEX = index
    asyncio.run(worker_main(conn, index, n, factory, args, setup))


def pool_state(pool):
    """A worker pool's admit_row fields, contexts as a list (the controller takes the spread over every worker)."""
    from .rl_seq import contexts_of

    row = {"slots": pool.slots, "used": pool.used, "live": len(pool.live), "sending": len(pool.sending),
           "kv_waits": pool.kv_waits, "turn_waits": pool.turn_waits, "contexts": contexts_of(pool.live),
           "capacity_share": pool.capacity_share}
    if pool.gate is not None:
        snap = pool.gate.snapshot()
        row.update(gate_inflight=snap["inflight"], gate_waiting=snap["waiting"], gate_reserved=snap["reserved"],
                   gate_capacity=snap["capacity"])
    from . import rl_seq

    dwell = getattr(rl_seq, "dwell_snapshot", None)  # the RL arm's dwell export, when present
    if dwell is not None:
        try:
            row["dwell"] = dwell(pool)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            row["dwell"] = {"error": repr(exc)[:200]}
    return row


async def worker_main(conn, index, n, factory, args, setup):
    ch = Channel(conn)
    aloop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    tasks, lag = {}, {"max": 0.0}
    try:
        loop = resolve(factory)(**args)
        if setup.get("cfg") is not None:
            loop.cfg = setup["cfg"]  # the controller's settings (preflight shrink, mg_sets) as they stand
        loop.feed = ForwardFeed(ch)
        gpu = getattr(loop, "gpu", None)
        if gpu is not None:  # the controller's sampler covers the cards
            gpu.stop.set()
        maker = getattr(loop, "driver_gate", None)  # tests: a gate over a fake upstream
        gate = maker() if callable(maker) else phase.PhaseGate(leader_first=(setup.get("mg") or {}).get("leader_first"))
        if not setup.get("open", True):
            gate.close()
            phase.TIME.pause()
        level_bench.PHASE_GATE = gate
        level_bench.ENGINE_POLL = False  # the controller's guard is the one periodic /metrics reader
        level_bench.PROGRESS_HOOK = lambda bench, row: ch.send({"op": "try_done", "bench": bench, "row": row})
        mg = setup.get("mg") or {}
        if mg.get("active_tries"):
            level_bench.SHARED_SLOTS_BY_URL = {
                u: level_bench.SharedSlots(math.ceil(int(mg["active_tries"]) / n), mg.get("active_kv_fill"),
                                           project=mg.get("kv_project"), stagger_s=mg.get("admit_stagger_s") or 0.0,
                                           shared_gate=True, capacity_share=1.0 / n) for u in setup["urls"]}
    except Exception as exc:  # noqa: BLE001 - the controller marks this worker dead
        ch.send({"op": "fatal", "error": repr(exc)[:1000]})
        ch.close(join=5.0)
        return
    gate_lock = asyncio.Lock()

    async def play(msg):
        gid = msg["gid"]
        if msg.get("mg_set") is not None:
            loop.cfg.setdefault("mg_sets", {})[msg["spec"]["name"]] = msg["mg_set"]
        try:
            folder, rows, got = await loop.mg_play(msg["name"], msg["spec"], msg["starts"], msg["tries"],
                                                   model=msg["model"], seed=None, harvest=msg.get("harvest", True),
                                                   **({"base_url": msg["url"]} if msg.get("url") else {}))
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            ch.send({"op": "group_error", "gid": gid, "kind": "deadline", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the controller fails the group
            ch.send({"op": "group_error", "gid": gid, "kind": "error", "error": repr(exc)[:1000]})
        else:
            ch.send({"op": "group_done", "gid": gid, "folder": None if folder is None else str(folder), "rows": rows,
                     "play": got})
        finally:
            tasks.pop(gid, None)

    async def gate_op(msg):
        async with gate_lock:
            aborted = 0
            if msg["state"] == "closed":
                aborted = gate.close()
                phase.TIME.pause()
                await gate.drained()
            else:
                phase.TIME.resume()
                gate.open()
            ch.send({"op": "gate_ack", "epoch": msg["epoch"], "aborted": aborted})

    def state():
        pools = {u: pool_state(p) for u, p in (level_bench.SHARED_SLOTS_BY_URL or {}).items()}
        ch.send({"op": "state", "t": time.time(), "pools": pools, "loop_lag": round(lag["max"], 3),
                 "groups": len(tasks), "gate": {"open": gate.is_open, "inflight": len(gate.inflight),
                                                **dict(gate.stats), "leader_first": bool(getattr(gate, "leader_first", None)),
                                                **({"plan": gate.plans[-1]} if getattr(gate, "plans", None) else {})}})
        lag["max"] = 0.0

    def on_msg(msg):
        op = msg.get("op")
        if op == "play":
            tasks[msg["gid"]] = asyncio.ensure_future(play(msg))
        elif op == "gate":
            asyncio.ensure_future(gate_op(msg))
        elif op == "cancel":
            task = tasks.get(msg["gid"])
            if task is not None:
                task.cancel()
        elif op == "knobs":
            k = msg.get("knobs") or {}
            if k.get("active_tries"):
                for pool in (level_bench.SHARED_SLOTS_BY_URL or {}).values():
                    pool.slots = math.ceil(int(k["active_tries"]) / n)
            for url, on in (k.get("kv_hold") or {}).items():  # rl_seq kv_cap: the server's KV at the cap
                pool = (level_bench.SHARED_SLOTS_BY_URL or {}).get(url)
                if pool is not None and pool.gate is not None:
                    pool.gate.hold(on)
            for url in k.get("hold") or ():  # the guard is blind on url: this worker admits nothing past its live tries
                pool = (level_bench.SHARED_SLOTS_BY_URL or {}).get(url)
                if pool is not None:
                    pool.slots = max(1, min(pool.slots, pool.used))
        elif op == "ping":
            state()
        elif op == "stop":
            stopped.set()

    async def beat():
        every = float(setup.get("state_s") or STATE_S)
        tick, last = min(1.0, every), aloop.time()
        state()
        while True:
            began = aloop.time()
            await asyncio.sleep(tick)
            lag["max"] = max(lag["max"], aloop.time() - began - tick)
            if aloop.time() - last >= every:
                state()
                last = aloop.time()

    ch.listen(aloop, on_msg, stopped.set)  # EOF: the controller is gone, so is this worker
    beat_task = asyncio.ensure_future(beat())
    ch.send({"op": "ready", "pid": os.getpid(), "index": index})
    try:
        await stopped.wait()
    finally:
        beat_task.cancel()
        for task in list(tasks.values()):
            task.cancel()
        await asyncio.gather(beat_task, *tasks.values(), return_exceptions=True)
        try:
            await gate.shutdown()
        except Exception:  # noqa: BLE001
            pass
        ch.close(join=5.0)


# ---- controller side ---------------------------------------------------------------------------------------------

class Worker:
    def __init__(self, index, proc, ch):
        self.index, self.proc, self.ch = index, proc, ch
        self.alive, self.ready, self.pid = True, None, proc.pid
        self.last = time.monotonic()
        self.groups = {}  # gid -> {"url", "tries", "done", "name", "future"}
        self.acks = {}  # gate epoch -> future
        self.state = None

    def unfinished(self, url):
        return sum(g["tries"] - g["done"] for g in self.groups.values() if g["url"] == url)


class ProxyGate:
    """PhaseController's gate over the workers' gates: close/open return once every live worker acked."""

    def __init__(self, pool):
        self.pool, self.open_now = pool, True
        self.stats = {"closes": 0, "opens": 0, "aborted": 0}

    @property
    def is_open(self):
        return self.open_now

    async def close(self):
        self.open_now = False
        self.stats["closes"] += 1
        acks = await self.pool.fan("closed")
        aborted = sum(int((a or {}).get("aborted") or 0) for a in acks)
        self.stats["aborted"] += aborted
        return aborted

    async def drained(self):  # a worker acks "closed" only after its own gate drained
        return None

    async def open(self):
        self.open_now = True
        self.stats["opens"] += 1
        await self.pool.fan("open")

    async def aclose(self):
        return None

    async def shutdown(self):
        return None


class DriverPool:
    """N worker processes playing rl_seq's groups (module doc). ``loop``: the controller's Loop; a worker builds its
    own from ``loop_args(loop)``, or from ``loop.driver_factory`` = (factory, kwargs) (tests)."""

    def __init__(self, n, loop, cfg):
        self.n, self.loop, self.cfg = int(n), loop, cfg
        factory = getattr(loop, "driver_factory", None)
        if factory is None:
            a = loop_args(loop)
            factory = ("arena.rl_drivers:real_loop", {"study": a["study"], "trial": a["trial"], "kwargs": a["kwargs"]})
        self.factory, self.args = factory
        self.workers, self.bench = [], {}  # bench name -> (worker, gid)
        self.gate = ProxyGate(self)
        self.epoch = 0
        self.on_try = None  # the run's PROGRESS hook (bench, row)
        self.on_all_dead = None
        self.watch_task = None
        self.stopped = False
        self._final = None

    # ---- start / stop

    def setup(self, urls):
        mg = self.loop.cfg.get("mg") or {}
        return {"urls": list(urls), "cfg": self.loop.cfg, "open": self.gate.is_open, "state_s": STATE_S,
                "mg": {k: mg.get(k) for k in ("active_tries", "active_kv_fill", "kv_project", "admit_stagger_s",
                                                "leader_first")}}

    async def start(self, urls, on_try=None, on_all_dead=None):
        self.on_try, self.on_all_dead = on_try, on_all_dead
        aloop = asyncio.get_running_loop()
        ctx = multiprocessing.get_context("spawn")
        setup = self.setup(urls)
        began = time.time()
        for i in range(self.n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker_entry, args=(child, i, self.n, self.factory, self.args, setup),
                               name=f"rl-driver-{i}", daemon=False)  # daemon processes may not have children
            proc.start()
            child.close()
            w = Worker(i, proc, Channel(parent))
            w.ready = aloop.create_future()
            w.ch.listen(aloop, lambda msg, w=w: self.on_message(w, msg), lambda w=w: self.dead(w, "pipe closed"))
            self.workers.append(w)
        self._final = weakref.finalize(self, kill_all, [w.proc for w in self.workers])  # no orphans on any exit
        done, _ = await asyncio.wait([w.ready for w in self.workers], timeout=READY_S)
        for w in self.workers:
            if w.alive and not w.ready.done():
                self.dead(w, f"not ready after {READY_S:.0f} s")
        live = [w for w in self.workers if w.alive]
        self.watch_task = asyncio.ensure_future(self.watch())
        self.loop.feed.emit("seq_drivers", drivers=self.n, live=len(live), pids=[w.pid for w in self.workers],
                            slots=setup["mg"]["active_tries"] and math.ceil(int(setup["mg"]["active_tries"]) / self.n),
                            capacity_share=round(1.0 / self.n, 4), seconds=round(time.time() - began, 1))
        if not live:
            raise DriverError("rl_drivers: no driver started")
        return len(live)

    async def stop(self):
        if self.stopped:
            return
        self.stopped = True
        try:
            if self.watch_task is not None:
                self.watch_task.cancel()
            for w in self.workers:
                if w.proc.is_alive():
                    w.ch.send({"op": "stop"})
            await asyncio.to_thread(join_all, [w.proc for w in self.workers], STOP_S)
        finally:
            kill_all([w.proc for w in self.workers])
            for w in self.workers:
                w.alive = False
                w.ch.close()
                self.fail(w, "driver stopped")
            if self._final is not None:
                self._final.detach()

    # ---- messages

    def on_message(self, w, msg):
        w.last = time.monotonic()
        if self.stopped:  # the run's teardown: no hook or feed row after its tasks were gathered
            return
        op = msg.get("op")
        if op == "try_done":
            got = self.bench.get(msg["bench"])
            if got is not None and got[1] in got[0].groups:
                got[0].groups[got[1]]["done"] += 1
            if self.on_try is not None:
                self.on_try(msg["bench"], msg["row"])
        elif op in ("group_done", "group_error"):
            g = w.groups.get(msg["gid"])
            if g is None or g["future"].done():
                return
            if op == "group_done":
                folder = None if msg["folder"] is None else Path(msg["folder"])
                g["future"].set_result((folder, msg["rows"], msg["play"]))
            elif msg["kind"] == "deadline":
                g["future"].set_exception(TimeoutError(msg["error"]))
            else:
                g["future"].set_exception(DriverError(f"driver {w.index}: {msg['error']}"))
        elif op == "gate_ack":
            fut = w.acks.pop(msg["epoch"], None)
            if fut is not None and not fut.done():
                fut.set_result(msg)
        elif op == "feed":
            self.loop.feed.emit(msg["kind"], **{**msg["fields"], "driver": w.index})
        elif op == "state":
            w.state = msg
        elif op == "ready":
            if w.ready is not None and not w.ready.done():
                w.ready.set_result(msg)
        elif op == "fatal":
            self.dead(w, f"fatal: {msg.get('error')}")

    # ---- liveness

    async def watch(self, every=None):
        every = every or min(1.0, DEAD_S / 4)
        while True:
            await asyncio.sleep(every)
            for w in self.workers:
                if not w.alive:
                    continue
                if not w.proc.is_alive():
                    self.dead(w, f"exited ({w.proc.exitcode})")
                elif time.monotonic() - w.last > DEAD_S:
                    self.dead(w, f"silent for {time.monotonic() - w.last:.0f} s")

    def fail(self, w, why):
        for gid, g in list(w.groups.items()):
            if not g["future"].done():
                g["future"].set_exception(DriverError(f"driver {w.index} {why}"))
        for fut in w.acks.values():
            if not fut.done():
                fut.set_result(None)
        w.acks.clear()
        if w.ready is not None and not w.ready.done():
            w.ready.set_result(None)

    def dead(self, w, why):
        if not w.alive or self.stopped:
            return
        w.alive = False
        w.ch.unlisten()
        if w.proc.is_alive():  # silent or fatal: never left running
            w.proc.kill()
        self.loop.feed.emit("alert", why="driver died", driver=w.index, pid=w.pid, cause=why,
                            exitcode=w.proc.exitcode, groups=len(w.groups))
        self.fail(w, f"died ({why})")
        if not any(x.alive for x in self.workers) and self.on_all_dead is not None:
            self.loop.feed.emit("alert", why="every driver died: the run stops")
            self.on_all_dead()

    # ---- play

    def live(self):
        return [w for w in self.workers if w.alive]

    def assign(self, gid, name, tries, url):
        """The live worker with the fewest unfinished tries on ``url`` (index order on ties); None: none alive."""
        live = self.live()
        if not live:
            return None
        w = min(live, key=lambda x: (x.unfinished(url), x.index))
        w.groups[gid] = {"url": url, "tries": int(tries), "done": 0, "name": name,
                         "future": asyncio.get_running_loop().create_future()}
        self.bench[name] = (w, gid)
        return w.index

    async def play(self, gid, name, spec, starts, tries, *, model, url, mg_set=None, harvest=True):
        """One group on its assigned worker -> (folder, rows, play), as Loop.mg_play. A worker's bench deadline
        raises TimeoutError (its message), any other failure DriverError."""
        w = next((x for x in self.workers if gid in x.groups), None)
        if w is None:  # not assigned at launch (rl_seq assigns there)
            if self.assign(gid, name, tries, url) is None:
                raise DriverError("no live driver")
            w = next(x for x in self.workers if gid in x.groups)
        g = w.groups[gid]
        if not w.alive:
            w.groups.pop(gid, None)
            self.bench.pop(name, None)
            if g["future"].done():
                g["future"].exception()  # retrieved: raised below as this group's failure
            raise DriverError(f"driver {w.index} is dead")
        w.ch.send({"op": "play", "gid": gid, "name": name, "spec": spec, "starts": starts, "tries": int(tries),
                   "model": model, "url": url, "harvest": harvest, "mg_set": mg_set})
        try:
            return await g["future"]
        except asyncio.CancelledError:
            if w.alive:
                w.ch.send({"op": "cancel", "gid": gid})
            raise
        finally:
            w.groups.pop(gid, None)
            self.bench.pop(name, None)

    # ---- gate and knobs

    async def fan(self, state):
        self.epoch += 1
        aloop = asyncio.get_running_loop()
        futs = []
        for w in self.live():
            fut = aloop.create_future()
            w.acks[self.epoch] = fut
            w.ch.send({"op": "gate", "state": state, "epoch": self.epoch})
            futs.append(fut)
        return list(await asyncio.gather(*futs)) if futs else []

    def knobs(self, knobs):
        for w in self.live():
            w.ch.send({"op": "knobs", "knobs": dict(knobs)})

    # ---- admission rows

    def pool_row(self, url):
        """admit_row's pool fields for ``url`` summed over the live workers' latest state rows."""
        from .rl_seq import spread

        rows = [((w.state or {}).get("pools") or {}).get(url) for w in self.live()]
        rows = [r for r in rows if r]
        if not rows:
            return {}
        out = {k: sum(int(r.get(k) or 0) for r in rows) for k in ("slots", "used", "live", "sending", "kv_waits",
                                                                  "turn_waits")}
        out["contexts"] = spread([c for r in rows for c in r.get("contexts") or []])
        if any("gate_capacity" in r for r in rows):
            out.update({k: sum(int(r.get(k) or 0) for r in rows)
                        for k in ("gate_inflight", "gate_waiting", "gate_reserved", "gate_capacity")})
        dwells = [r["dwell"] for r in rows if isinstance(r.get("dwell"), dict) and "now" in r["dwell"]]
        if dwells:  # rl_seq.dwell_snapshot summed over the workers
            out["dwell"] = {part: {k: round(sum(d[part].get(k, 0) for d in dwells), 1) for k in dwells[0][part]}
                            for part in ("now", "live_s")}
        return out

    def driver_rows(self):
        now = time.monotonic()
        return [{"driver": w.index, "alive": w.alive, "groups": len(w.groups),
                 "loop_lag": (w.state or {}).get("loop_lag"), "age_s": round(now - w.last, 1),
                 **({"plan": ((w.state or {}).get("gate") or {})["plan"]} if "plan" in ((w.state or {}).get("gate") or {})
                    else {})} for w in self.workers]


def join_all(procs, timeout):
    end = time.monotonic() + timeout
    for p in procs:
        p.join(max(0.0, end - time.monotonic()))


def kill_all(procs):
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except (OSError, ValueError, AssertionError):
            pass
    join_all([p for p in procs if _alive(p)], 2.0)
    for p in procs:
        if _alive(p):
            try:
                p.kill()
                p.join(2.0)
            except (OSError, ValueError, AssertionError):
                pass


def _alive(p):
    try:
        return p.is_alive()
    except (ValueError, AssertionError):
        return False
