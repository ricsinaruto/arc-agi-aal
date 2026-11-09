"""Cache-based admission: admit against the server's real KV use, not reserved worst cases.

The reserving gate (``arena.gate.KVGate``) holds prompt + completion cap per request and on RTX kept the cache at
31-37% with the gate binding. ``CacheGate`` admits while the server's ``vllm:kv_cache_usage_perc`` plus the requests
admitted since that reading (each at its full reserved cost) stays at or under ``target``. With no fresh reading
(``max_age``), it falls back to the reserving rule on the same capacity (fail closed). A preemption on the server
lowers the target by ``step`` (floor ``floor``)."""

import time

from arena.gate import KVGate

DEFAULTS = {"target": 0.8, "max_age": 15.0, "settle": 15.0, "poll": 2.0, "step": 0.05, "floor": 0.5,
            "charge": 0.0}  # charge 1: a request counts its prompt + the mean reply, not the cap (hint); 0: full cost


def settings(raw):
    merged = {**DEFAULTS, **(raw or {})}
    merged.pop("mode", None)
    if set(merged) != set(DEFAULTS):
        raise ValueError(f"Unknown admission settings: {sorted(set(merged) - set(DEFAULTS))}")
    if not 0 < merged["floor"] <= merged["target"] < 1:
        raise ValueError("admission: 0 < floor <= target < 1")
    if min(merged["max_age"], merged["settle"], merged["poll"]) <= 0 or merged["poll"] >= merged["max_age"]:
        raise ValueError("admission: positive times, poll under max_age")
    return {k: float(v) for k, v in merged.items()}


class CacheGate(KVGate):
    def __init__(self, capacity_tokens, slots, *, blocks, target=0.8, max_age=15.0, settle=15.0, step=0.05,
                 floor=0.5, charge=0.0, now=time.monotonic, **kw):
        super().__init__(capacity_tokens, slots, now=now, **kw)
        self.blocks = int(blocks)  # the server's cache blocks (the fraction's denominator)
        self.target, self.max_age, self.settle, self.step, self.floor = target, max_age, settle, step, floor
        self.kv = self.read_at = self.preemptions = None
        self.pending = []  # (admitted at, units): admitted after the reading that will show them
        self.stats = {"kv_admitted": 0, "reserve_admitted": 0, "readings": 0, "stale_reads": 0,
                      "target_cuts": 0, "peak_kv": 0.0, "peak_estimate": 0.0}
        self.charged = bool(charge)
        self.charges, self._asking = {}, None  # game -> units the next admission adds to the real KV use
        self.rank = {}  # game -> 0 (never submitted: admitted strictly first) / 1; set per request by the transport

    def hint(self, game_id, units):
        """``charge``: what admitting ``game_id``'s next request adds to the server's used blocks (its prompt and the
        mean reply, ``Transport``); without a hint it counts at its full reserved cost (prompt + cap)."""
        if self.charged:
            self.charges[game_id] = int(units)

    def _charge(self, game_id, units):
        return min(int(units), self.charges.get(game_id, units)) if game_id is not None else int(units)

    async def acquire(self, game_id, tokens, prompt=None):
        self._asking = game_id  # KVGate.acquire's first fit check runs before it awaits
        return await super().acquire(game_id, tokens, prompt)

    def _pump_focus(self):
        if getattr(self, "lane", None) is not None or self.opening_burst_requests > 1:
            return super()._pump_focus()
        while self.queue:  # by rank (never-submitted first), then FIFO; the chosen request's own charge
            while self.queue and self.queue[0][0].done():
                self.queue.popleft()
            if not self.queue:
                break
            at = min(range(len(self.queue)), key=lambda i: (self.queue[i][0].done(), self.rank.get(self.queue[i][1], 1), i))
            future, game_id, tokens = self.queue[at]
            if not self._fits(tokens, game_id):
                return False  # strict: nothing of a lower rank is admitted past a waiting never-submitted request
            del self.queue[at]
            self._selection(game_id, affinity=False)
            self._take(game_id, tokens, prompt=self._prompt_of.pop(future, 0))
            future.set_result(None)
        return True

    def fresh(self):
        return self.read_at is not None and self.now() - self.read_at <= self.max_age

    def observe(self, engine):
        """A /metrics reading (``trial.engine_metrics``'s dict, or None when unreadable)."""
        kv = (engine or {}).get("kv_cache_fraction")
        if kv is None:
            self.stats["stale_reads"] += 1
            return
        now = self.now()
        self.kv, self.read_at = float(kv), now
        self.stats["readings"] += 1
        self.stats["peak_kv"] = max(self.stats["peak_kv"], self.kv)
        self.pending = [(at, units) for at, units in self.pending if at > now - self.settle]
        preempted = (engine or {}).get("preemptions")
        if preempted is not None:
            if self.preemptions is not None and preempted > self.preemptions and self.target > self.floor:
                self.target = max(self.floor, round(self.target - self.step, 4))
                self.stats["target_cuts"] += 1
                if self.log:
                    self.log({"event": "target_cut", "preemptions": preempted, "target": self.target})
            self.preemptions = preempted
        self._pump()

    def estimate(self, units=0):
        return self.kv + (sum(u for _, u in self.pending) + units) / self.blocks

    def _cache_fits(self, units):
        return self.estimate(units) <= self.target

    def _fits(self, tokens, game_id=None):
        if self.inflight >= self.slots:
            return False
        if self.fresh():
            return self._cache_fits(self._charge(game_id if game_id is not None else self._asking, tokens))
        return self.reserved + tokens <= self.capacity

    def grow_fits(self, cost, held, freed=0, charge=None):
        """Room for a held reservation of ``held`` units to grow to ``cost`` (``Transport.grow``; ``freed``: units
        being released). In cache mode the whole next turn counts (its prompt is back in use), or ``charge``."""
        if self.fresh():
            return self._cache_fits(cost if charge is None or not self.charged else min(cost, int(charge)))
        return self.reserved - freed + cost - held <= self.capacity

    def note_growth(self, units, charge=None):
        if self.fresh():
            self._pending(units if charge is None or not self.charged else min(units, int(charge)))

    def _pending(self, units):
        self.pending.append((self.now(), int(units)))
        self.stats["peak_estimate"] = max(self.stats["peak_estimate"], round(self.estimate(), 4))

    def _take(self, game_id, tokens, lane=False, prompt=0):
        cache = self.fresh()
        super()._take(game_id, tokens, lane=lane, prompt=prompt)
        self.stats["kv_admitted" if cache else "reserve_admitted"] += 1
        charge = self._charge(game_id, tokens)
        self.charges.pop(game_id, None)
        if cache:
            self._pending(charge)

    def snapshot(self):
        return {**super().snapshot(), "admission": "kv", "blocks": self.blocks, "target": self.target,
                "kv": self.kv, "fresh": self.fresh(), **self.stats}


class OverhangGate(KVGate):
    """Study ``arc2.gate_overhang`` {"window": s, "poll": s} (flag, default off; Q6 preemption 28 Oct): v87b's exact
    reserving gate with its capacity cut by the measured overhang, the cache blocks in use beyond every reservation
    (``kv_cache_fraction`` x blocks - reserved, the largest over ``window`` seconds). Those blocks are held outside any
    request (F4: 265-430 of 2,180 blocks by the server's allocation-failure dumps), so the reserving rule alone lets
    reservations + overhang pass the pool and vLLM preempts. With the cut, reservations + overhang <= the gate's
    capacity. Unread /metrics: the last cut stands (it decays only on readings)."""

    def __init__(self, capacity_tokens, slots, *, blocks, window=60.0, **kw):
        super().__init__(capacity_tokens, slots, **kw)
        self.base, self.blocks, self.window = self.capacity, int(blocks), float(window)
        self.seen = __import__("collections").deque()  # (time, overhang blocks)
        self.stats = {"readings": 0, "stale_reads": 0, "peak_overhang": 0, "cut": 0, "min_capacity": self.capacity}

    def observe(self, engine):
        kv = (engine or {}).get("kv_cache_fraction")
        if kv is None:
            self.stats["stale_reads"] += 1
            return
        now = self.now()
        over = max(0, int(round(float(kv) * self.blocks)) - self.reserved)
        self.seen.append((now, over))
        while self.seen and self.seen[0][0] < now - self.window:
            self.seen.popleft()
        cut = max(o for _, o in self.seen)
        self.capacity = max(1, self.base - cut)
        self.stats.update(readings=self.stats["readings"] + 1, cut=cut,
                          peak_overhang=max(self.stats["peak_overhang"], over),
                          min_capacity=min(self.stats["min_capacity"], self.capacity))
        self._pump()

    def snapshot(self):
        return {**super().snapshot(), "admission": "overhang", "base_capacity": self.base, **self.stats}
