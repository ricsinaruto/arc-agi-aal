"""Sequential play/train phases (RL redesign): every GPU plays, then every GPU trains.

At a phase stop the model gate (``PhaseGate``, the play clients' httpx transport) holds new requests, aborts the
in-flight ones (the client disconnect aborts them in vLLM) and resends them once play resumes; the caller sees one
slow reply. Harness, tool and game processes stay alive. ``TIME`` is the try clocks' time source: it stops while
training, so wall clocks, wall caps and the runtime status the model reads see no gap.
"""
import asyncio
import hashlib
import inspect
import itertools
import json
import time
import uuid
from collections import OrderedDict

import httpx

from arc_infra import KEEPALIVE_EXPIRY


class PhaseTime:
    """Monotonic seconds with the training phases cut out (frozen while paused)."""

    def __init__(self, now=time.monotonic):
        self.now, self.paused_total, self.paused_at, self.pauses = now, 0.0, None, 0

    def monotonic(self):
        return (self.now() if self.paused_at is None else self.paused_at) - self.paused_total

    @property
    def paused(self):
        return self.paused_at is not None

    def pause(self):
        if self.paused_at is None:
            self.paused_at, self.pauses = self.now(), self.pauses + 1

    def resume(self):
        if self.paused_at is not None:
            self.paused_total += max(0.0, self.now() - self.paused_at)
            self.paused_at = None


TIME = PhaseTime()


def monotonic():
    """``time.monotonic()`` minus training time; equal to it in any run that never pauses."""
    return TIME.monotonic()


def sticky_key(content):
    """A request's try key: its first two messages (system + opening user turn). Tries of one start share it, so a
    group lands on one server and shares its prefix cache."""
    try:
        head = json.loads(content or b"{}").get("messages", [])[:2]
    except (ValueError, AttributeError):
        return None
    return hashlib.sha1(json.dumps(head, sort_keys=True, default=str).encode()).hexdigest() if head else None


WARM_ID = "warm-"  # X-Request-Id of a leader-first warm-up: vLLM names it chatcmpl-warm-... (rl_seq.served_prune)
LEADER_FIRST = {"min_share": 0.5, "min_chars": 20000, "wait_s": 300.0}
WARM_DROP = ("stream", "stream_options", "logprobs", "top_logprobs", "prompt_logprobs", "min_tokens", "n")


def prompt_chain(content):
    """(cumulative chars, chain hashes) over a chat request's head (model, tools, template kwargs), then each message;
    entry k covers the head and the first k messages. None without messages."""
    try:
        body = json.loads(content or b"{}")
    except (ValueError, TypeError):
        return None
    msgs = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(msgs, list) or not msgs:
        return None
    head = json.dumps({k: body.get(k) for k in ("model", "tools", "chat_template_kwargs")}, sort_keys=True, default=str)
    h = hashlib.sha1(head.encode())
    chars, hashes = [len(head)], [h.hexdigest()]
    for m in msgs:
        part = json.dumps(m, sort_keys=True, default=str)
        h.update(part.encode())
        chars.append(chars[-1] + len(part))
        hashes.append(h.hexdigest())
    return chars, hashes


def leader_plan(contents, min_share=0.5, min_chars=20000, **_):
    """Requests held over a train phase -> clusters (index lists, shortest prompt first = the leader): a request whose
    deepest message prefix shared with any other held request covers >= max(min_chars, min_share x its prompt) joins
    the cluster of that prefix (a carried group's tries: their restored start). The rest are left out."""
    chains = [prompt_chain(c) for c in contents]
    seen = {}
    for i, ch in enumerate(chains):
        if ch is not None:
            for k in range(1, len(ch[1])):
                seen.setdefault(ch[1][k], []).append(i)
    clusters = {}
    for i, ch in enumerate(chains):
        if ch is None:
            continue
        chars, hashes = ch
        for k in range(len(hashes) - 1, 0, -1):
            if len(seen[hashes[k]]) > 1:
                if chars[k] >= max(min_chars, min_share * chars[-1]):
                    clusters.setdefault(hashes[k], []).append(i)
                break
    return [sorted(c, key=lambda i: (chains[i][0][-1], i)) for c in clusters.values() if len(c) > 1]


def warm_body(content, rid=None):
    """A held leader's body as a prefill-only request (one token): its prompt lands in the prefix cache."""
    body = json.loads(content)
    if rid is not None:
        body["request_id"] = rid
    for k in WARM_DROP:
        body.pop(k, None)
    body["max_tokens"] = 1
    if "max_completion_tokens" in body:
        body["max_completion_tokens"] = 1
    return json.dumps(body).encode()


class PhaseGate(httpx.AsyncBaseTransport):
    """httpx transport of the play clients over N play servers.

    ``backends``: server base URLs; a request is sent to the one its try key maps to (fewest in flight at first sight,
    then sticky), or to its own URL when ``backends`` is empty. ``close()`` holds new requests and aborts in-flight
    ones; ``drained()`` waits until none is on a server; ``open()`` sends the held and aborted ones.

    ``leader_first`` (mg.leader_first, RL 29 Oct: phase 16 of real-v15 recomputed 11.75M prompt tokens in a ~14 min
    ramp, siblings of each carried group prefilling their shared restored start side by side): at ``open()`` the held
    requests are clustered by shared message prefix (``leader_plan``); each cluster's shortest prompt is first sent as
    a one-token warm-up, and the cluster is released once it returns (or after ``wait_s``)."""

    def __init__(self, backends=(), inner=None, keys=10000, keepalive_expiry=KEEPALIVE_EXPIRY, leader_first=None):
        # the clients' own limits do not apply to a given transport: the arena's (level_bench KEEPALIVE_EXPIRY) here
        self.inner = inner or httpx.AsyncHTTPTransport(limits=httpx.Limits(
            max_connections=None, max_keepalive_connections=256, keepalive_expiry=keepalive_expiry))
        self.backends = [httpx.URL(b) for b in backends]
        self.route, self.keys = OrderedDict(), int(keys)
        self.load = [0] * len(self.backends)
        self.gate = asyncio.Event()
        self.gate.set()
        self.inflight, self.aborted = set(), set()
        self.idle = asyncio.Event()
        self.idle.set()
        self.stats = {"sent": 0, "held": 0, "aborted": 0, "resent": 0}
        self.leader_first = ({**LEADER_FIRST, **(leader_first if isinstance(leader_first, dict) else {})}
                             if leader_first else None)
        self.waiting, self.follow, self.warming, self.tokens = {}, {}, set(), itertools.count()
        self.plans = []  # per open(): {"held", "clusters", "followers", "warm": [prompt tokens, cached tokens] ...}

    @property
    def is_open(self):
        return self.gate.is_set()

    def close(self):
        """Phase stop: hold new requests; abort every in-flight one (resent at ``open``). Returns the abort count."""
        self.gate.clear()
        for task in list(self.warming):
            task.cancel()
        for ev in list(self.follow.values()):  # back to the closed gate: held, planned again at the next open
            ev.set()
        self.follow.clear()
        n = 0
        for task in list(self.inflight):
            if not task.done():
                self.aborted.add(task)
                task.cancel()
                n += 1
        return n

    async def drained(self):
        await self.idle.wait()
        if self.warming:
            await asyncio.gather(*list(self.warming), return_exceptions=True)

    def open(self):
        if self.leader_first and self.waiting:
            self.plan()
        self.gate.set()

    def plan(self):
        # chat completions only: a held /tokenize call carries the same messages (N1c 29 Oct: it led clusters, its
        # warm-up went to /tokenize and came back without a warm id, so leader_first switched itself off)
        held = [(k, r) for k, r in self.waiting.items() if r.url.path.endswith("/chat/completions")]
        clusters = leader_plan([r.content for _, r in held], **self.leader_first)
        row = {"held": len(held), "clusters": len(clusters), "followers": sum(len(c) for c in clusters), "warm": 0,
               "warm_failed": 0, "warm_prompt": 0, "warm_cached": 0, "follow_timeout": 0, "t0": time.time()}
        for c in clusters:
            ev = asyncio.Event()
            for i in c:
                self.follow[held[i][0]] = ev
            task = asyncio.ensure_future(self.warm(held[c[0]][1], ev, row))
            self.warming.add(task)
            task.add_done_callback(self.warming.discard)
        self.plans.append(row)
        del self.plans[:-8]
        return row

    async def warm(self, request, ev, row):
        try:
            rid = WARM_ID + uuid.uuid4().hex
            req = httpx.Request(request.method, request.url, content=warm_body(request.content, rid),
                                headers=[(k, v) for k, v in request.headers.items() if k.lower() != "content-length"])
            req.headers["X-Request-Id"] = rid
            self.aim(req, self.pick(request))
            response = await self.send(req)
            if response.status_code >= 400:
                raise RuntimeError(f"warm-up {response.status_code}")
            data = response.json() or {}
            if not str(data.get("id", "")).startswith("chatcmpl-" + WARM_ID):
                # the server named it otherwise: rl_seq.served_prune would never delete its R3 capture (~1.7 KB a
                # token of host memory), so no further warm-ups in this run
                row["warm_id_bad"] = row.get("warm_id_bad", 0) + 1
                row.setdefault("warm_id_seen", []).append([rid, str(data.get("id"))[:80], response.status_code,
                                                           sorted(data)[:8]])
                del row["warm_id_seen"][3:]
                self.leader_first = None
            usage = data.get("usage") or {}
            row["warm"] += 1
            row["warm_prompt"] += int(usage.get("prompt_tokens") or 0)
            row["warm_cached"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a failed warm-up only releases its cluster unwarmed
            row["warm_failed"] += 1
        finally:
            ev.set()
            row["t_last"] = time.time()

    def pick(self, request):
        if not self.backends:
            return None
        key = sticky_key(request.content)
        if key is not None and key in self.route:
            self.route.move_to_end(key)
            return self.route[key]
        index = min(range(len(self.backends)), key=lambda i: (self.load[i], i))
        if key is not None:
            self.route[key] = index
            while len(self.route) > self.keys:
                self.route.popitem(last=False)
        return index

    def aim(self, request, index):
        if index is None:
            return
        url = self.backends[index]
        request.url = request.url.copy_with(scheme=url.scheme, host=url.host, port=url.port)
        request.headers["host"] = f"{url.host}:{url.port}" if url.port else url.host

    async def send(self, request):
        response = await self.inner.handle_async_request(request)
        try:
            await response.aread()  # the whole reply inside the phase: an abort loses no partial body
        except BaseException:
            await response.aclose()
            raise
        return response

    async def handle_async_request(self, request):
        request.read()
        index, resend, token = self.pick(request), False, next(self.tokens)
        while True:
            if not self.gate.is_set():
                self.stats["held"] += 1
                self.waiting[token] = request
                try:
                    while not self.gate.is_set():  # woken by open(), it may have closed again before this ran (H5)
                        await self.gate.wait()
                finally:
                    self.waiting.pop(token, None)
            ev = self.follow.pop(token, None)
            if ev is not None:  # leader_first: its cluster's warm-up first
                try:
                    await asyncio.wait_for(ev.wait(), float(self.leader_first["wait_s"]))
                except asyncio.TimeoutError:
                    if self.plans:
                        self.plans[-1]["follow_timeout"] += 1
                if not self.gate.is_set():
                    continue
            self.aim(request, index)
            task = asyncio.ensure_future(self.send(request))
            self.inflight.add(task)
            self.idle.clear()
            if index is not None:
                self.load[index] += 1
            self.stats["resent" if resend else "sent"] += 1
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:  # the caller gave up: so does the server request
                task.cancel()
                raise
            finally:
                self.inflight.discard(task)
                if index is not None:
                    self.load[index] -= 1
                if not self.inflight:
                    self.idle.set()
            if task in self.aborted:
                self.aborted.discard(task)
                self.stats["aborted"] += 1
                resend = True
                continue
            return task.result()

    async def aclose(self):
        """Shared by every bench client: a client's close leaves it open (``shutdown`` closes it)."""

    async def shutdown(self):
        await self.inner.aclose()


class PhaseController:
    """Switches every play server and the trainer between phases.

    ``servers``: objects with async ``sleep()``, ``wake()`` and ``load(name, path)`` (rl_loop.Server). ``park`` and
    ``unpark``: async callables of the trainer. ``gates``: the PhaseGates of every play client. Each switch returns
    its timings (seconds) for the feed."""

    def __init__(self, gates, servers, park, unpark, *, adapter="policy", clock=None, now=time.monotonic, memory=None):
        self.gates, self.servers, self.park, self.unpark = list(gates), list(servers), park, unpark
        # async callable -> per-card used GiB, read with the servers asleep (trainer not yet unparked) and awake
        # (adapter not yet loaded): the vLLM residue a sleep leaves on the cards, per phase (29 Oct: +12 GiB seen)
        self.memory = memory
        self.adapter, self.clock, self.now = adapter, clock or TIME, now
        self.phase, self.switches = "play", 0
        # async callable(adapter_path), awaited after wake and before the load (mg.async_adapter_save: the trainer's
        # background adapter write lands, overlapped with the park and wake)
        self.before_load = None

    async def to_train(self):
        if self.phase != "play":
            raise RuntimeError(f"to_train from {self.phase}")
        t0 = self.now()
        aborted = 0
        for g in self.gates:  # a gate over driver processes (rl_drivers.ProxyGate) returns once each has acked
            n = g.close()
            aborted += (await n) if inspect.isawaitable(n) else n
        self.clock.pause()
        await asyncio.gather(*(g.drained() for g in self.gates))
        t1 = self.now()
        await asyncio.gather(*(s.sleep() for s in self.servers))
        t2 = self.now()
        used = await self.read_memory()
        await self.unpark()
        t3 = self.now()
        self.phase = "train"
        return {"aborted": aborted, "drain_s": round(t1 - t0, 3), "sleep_s": round(t2 - t1, 3),
                "unpark_s": round(t3 - t2, 3), "switch_s": round(t3 - t0, 3),
                **({"asleep_gib": used} if used is not None else {})}

    async def read_memory(self):
        if self.memory is None:
            return None
        try:
            return await self.memory()
        except Exception:  # noqa: BLE001 - a missed reading never stops a switch
            return None

    async def to_play(self, adapter_path=None, before_open=None, recycle=False):
        """``recycle``: every server's processes are replaced (Server.recycle) instead of woken, while the trainer is
        parked off the cards (mg.redesign.recycle)."""
        if self.phase != "train":
            raise RuntimeError(f"to_play from {self.phase}")
        t0 = tp = self.now()
        await self.park()
        t1 = self.now()
        recycled = None
        if recycle:
            recycled = list(await asyncio.gather(*(s.recycle() for s in self.servers)))
        else:
            await asyncio.gather(*(s.wake() for s in self.servers))
        t2 = self.now()
        used = await self.read_memory()
        waited = None
        if adapter_path is not None and self.before_load is not None:
            await self.before_load(adapter_path)
            waited = round(self.now() - t2, 3)
        if adapter_path is not None:
            await asyncio.gather(*(s.load(self.adapter, adapter_path) for s in self.servers))
        if before_open is not None:  # rl_seq's boot dump: the adapter is live, nothing plays yet
            await before_open()
        t3 = self.now()
        self.clock.resume()
        for g in self.gates:
            got = g.open()
            if inspect.isawaitable(got):
                await got
        self.phase, self.switches = "play", self.switches + 1
        return {"park_s": round(t1 - tp, 3), "wake_s": round(t2 - t1, 3), "load_s": round(t3 - t2, 3),
                "switch_s": round(t3 - t0, 3), **({"save_wait_s": waited} if waited is not None else {}),
                **({"recycled": recycled} if recycled is not None else {}),
                **({"awake_gib": used} if used is not None else {})}

    async def train(self, update, recycle=None):
        """One train phase: stop play, run ``update()`` (returns the new adapter path or None), resume play. A failed
        update leaves the train phase standing (the caller rolls back, then ``to_play``). ``recycle``: a callable read
        after the update (True: the servers' processes are replaced at the switch back)."""
        stop = await self.to_train()
        path = await update()
        return {"stop": stop, "start": await self.to_play(path, recycle=bool(recycle and recycle()))}
