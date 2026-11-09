"""v7 asynchronous RL loop core (docs/rl-plan.md, v7): groups pinned to one adapter version, a buffer of finished
groups under a staleness bound, the version book (live adapters, LoRA slots), the held-out read schedule and the
card-2 controller (Mode B play blocks). Pure logic, no I/O and no asyncio: the loop driver calls it and writes
``state_dict()`` into every per-step state, ``load_state`` restores it on resume."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Group:
    """``size`` tries from one start, all on ``version`` (the served version when the group was made)."""
    gid: str
    unit: str
    start: list | None
    version: int
    size: int
    created: float
    done: dict = field(default_factory=dict)  # member index (str) -> {"reward", "path", "ended", ...}
    finished: float | None = None
    status: str = "running"  # running | ready | consumed | dropped
    cap: int | None = None  # the group's per-try token cap when it differs from its set's (curriculum running cap)
    tokens: int = 0  # completion tokens of its finished tries (a lag drop reports what it threw away)

    def member_done(self, index, row, now):
        self.done[str(int(index))] = dict(row)
        if len(self.done) >= self.size and self.finished is None:
            self.finished, self.status = float(now), "ready"
        return self.finished is not None


class Book:
    """Adapter versions: the trainer's lineage, the version server 1 serves new groups on, and the versions still
    used by an in-flight try or read (each held as a LoRA slot). ``serve_every`` k: server 1 takes every k-th version."""

    def __init__(self, max_loras, serve_every=1, version=0, read_slots=0, free=()):
        """``read_slots``: LoRA slots kept for reads beyond ``max_loras`` (v6g's open base read
        pinned a slot and versions 4-6 never served); ``free``: versions that need no slot (the base model, v0)."""
        self.max_loras, self.serve_every = int(max_loras), max(1, int(serve_every))
        self.read_slots, self.free, self.read_users = int(read_slots), {int(x) for x in free}, {}
        if self.max_loras < 2:  # the served version holds a slot while the next one waits: one slot would never free
            raise ValueError(f"max_loras {self.max_loras}: v7 needs >= 2 live versions")
        self.current = self.served = int(version)
        self.lineage = [int(version)]
        self.users = {}
        self.pending = None  # a version due to serve that waits for a slot (max_loras live at most)

    def publish(self, version):
        """The trainer's new version; True when server 1 is to load it (it becomes the version new groups get).
        A due version that would take live versions over ``max_loras`` waits (``take_pending``); new groups keep
        the served one meanwhile (lag still bounded by s_max)."""
        version = int(version)
        if version <= self.current:
            raise ValueError(f"version {version} after {self.current}")
        self.current = version
        self.lineage.append(version)
        if version % self.serve_every == 0:
            self.pending = version
            return self.take_pending() is not None
        return False

    def take_pending(self):
        """The waiting version, now served, once its slot is free (call after a release); else None."""
        v = self.pending
        if v is None or len((set(self.users) | {v}) - self.free) > self.max_loras:  # reads never block a version
            return None
        self.served, self.pending = v, None
        return v

    def rollback(self, version):
        """The lineage after a rollback to ``version`` (dropped versions leave it; their users stay counted)."""
        version = int(version)
        if version not in self.lineage:
            raise ValueError(f"version {version} not in the lineage")
        while self.lineage[-1] != version:
            self.lineage.pop()
        self.current = self.served = version
        self.pending = None

    def acquire(self, version, n=1, read=False):
        book = self.read_users if read else self.users
        book[int(version)] = book.get(int(version), 0) + int(n)

    def release(self, version, n=1, read=False):
        v, book = int(version), (self.read_users if read else self.users)
        left = book.get(v, 0) - int(n)
        if left < 0:
            raise ValueError(f"version {v}: released more than acquired")
        if left:
            book[v] = left
        else:
            book.pop(v, None)

    def live(self):
        """Versions in use: the served one plus every one a try or a read holds."""
        return sorted({self.served} | set(self.users) | set(self.read_users))

    def unloadable(self, loaded):
        return sorted(int(v) for v in loaded if int(v) != self.served and not self.users.get(int(v))
                      and not self.read_users.get(int(v)))

    def slots_ok(self):
        groups = ({self.served} | set(self.users)) - self.free
        reads = set(self.read_users) - groups - self.free
        return len(groups) <= self.max_loras and len(reads) <= self.read_slots

    def lag(self, version):
        """Updates since ``version`` along the lineage (None off the lineage: a rolled-back branch)."""
        v = int(version)
        return self.lineage[::-1].index(v) if v in self.lineage else None

    def state_dict(self):
        return {"max_loras": self.max_loras, "serve_every": self.serve_every, "read_slots": self.read_slots, "current": self.current,
                "served": self.served, "lineage": list(self.lineage), "pending": self.pending}

    @classmethod
    def load_state(cls, data, read_slots=None, max_loras=None, serve_every=None):
        """``read_slots`` / ``max_loras``: the resumed leg's settings over the saved ones (v6j pfR2: a saved 3 held the
        Book at 3 slots under s_max 3)."""
        book = cls(int(data["max_loras"] if max_loras is None else max_loras),
                   int(data.get("serve_every", 1) if serve_every is None else serve_every), data["lineage"][0],
                   int(data.get("read_slots", 0) if read_slots is None else read_slots))
        book.current, book.served, book.lineage = int(data["current"]), int(data["served"]), [int(x) for x in data["lineage"]]
        book.pending = data.get("pending")
        return book


class Buffer:
    """Finished groups, first finished first. ``next_group`` gives the trainer the next group within ``s_max`` of
    its version; older (or off-lineage) groups are dropped whole and counted. A consumed or dropped id never
    returns (resume dedupe)."""

    def __init__(self, s_max, batch_groups):
        self.s_max, self.batch_groups = int(s_max), int(batch_groups)
        self.ready = deque()
        self.closed = {}  # gid -> "consumed" | "dropped"
        self.drops = []

    def add(self, group):
        if group.gid in self.closed or any(g.gid == group.gid for g in self.ready):
            return False
        group.status = "ready"
        self.ready.append(group)
        return True

    def next_group(self, book, now=None):
        while self.ready:
            g = self.ready.popleft()
            lag = book.lag(g.version)
            if lag is None or lag > self.s_max:
                self.close(g, "dropped")
                self.drops.append({"gid": g.gid, "unit": g.unit, "version": g.version, "lag": lag, "at": now,
                                   "trainer_version": book.current, "tokens": int(getattr(g, "tokens", 0) or 0)})
                continue
            return g, lag
        return None, None

    def close(self, group, how):
        group.status = how
        self.closed[group.gid] = how

    def state_dict(self):
        return {"s_max": self.s_max, "batch_groups": self.batch_groups, "ready": [asdict(g) for g in self.ready],
                "closed": dict(self.closed), "drops": list(self.drops[-200:])}

    @classmethod
    def load_state(cls, data, s_max=None, batch_groups=None):
        """Settings: the resumed leg's (``s_max``, ``batch_groups``) over the saved ones; state: ready, closed, drops."""
        buf = cls(data["s_max"] if s_max is None else s_max, data["batch_groups"] if batch_groups is None else batch_groups)
        buf.ready = deque(Group(**g) for g in data["ready"])
        buf.closed, buf.drops = dict(data["closed"]), [{**d, "_seen": True} for d in data.get("drops") or []]
        return buf  # an earlier leg's drops were reported there (the loop emits only unseen ones)


class Scheduler:
    """When to make groups: whenever server 1 has fewer than one group's tries queued, unless ``max_ready`` finished
    groups wait for the trainer (the bounded queue); at start-up at most one group per ``ramp_s`` until
    ``in_flight_target`` tries are in flight (no synchronized finish waves, no start-wave KV burst)."""

    def __init__(self, group_size, max_ready, ramp_s, in_flight_target):
        self.group_size, self.max_ready = int(group_size), int(max_ready)
        self.ramp_s, self.in_flight_target = float(ramp_s), int(in_flight_target)
        self.last_made = None
        self.ramped = False
        self.queue_groups = 1  # groups' worth of tries kept queued (2 while server 2 plays a Mode B block)

    def want(self, now, queued, in_flight, ready_groups):
        if ready_groups >= self.max_ready or queued >= self.group_size * self.queue_groups:
            return 0
        if in_flight >= self.in_flight_target + self.group_size * self.queue_groups:  # hard cap (pfL1: a pool that
            return 0  # reports no queue never stopped group creation)
        if not self.ramped:
            if in_flight >= self.in_flight_target:
                self.ramped = True
            elif self.last_made is not None and now - self.last_made < self.ramp_s:
                return 0
            else:
                return 1
        return 1

    def made(self, now):
        self.last_made = float(now)

    def state_dict(self):
        return {"ramped": self.ramped}


class Reads:
    """Held-out reads every ``every`` steps, pinned to that step's version; a read lands when its last try ends."""

    def __init__(self, every):
        self.every = int(every)
        self.open = {}  # version (str) -> {"step", "tries", "done"}

    def due(self, step):
        return self.every > 0 and step > 0 and step % self.every == 0 and str(int(step)) not in self.open

    def start(self, step, version, tries):
        self.open[str(int(step))] = {"version": int(version), "tries": int(tries), "done": 0}

    def try_done(self, step):
        rec = self.open[str(int(step))]
        rec["done"] += 1
        if rec["done"] >= rec["tries"]:
            return self.open.pop(str(int(step)))
        return None

    def state_dict(self):
        return {"every": self.every, "open": {k: dict(v) for k, v in self.open.items()}}

    @classmethod
    def load_state(cls, data, every=None):
        r = cls(data["every"] if every is None else every)
        r.open = {k: dict(v) for k, v in data["open"].items()}
        return r


class Card2:
    """Mode B: the trainer card plays (server 2) in blocks when the trainer would otherwise idle. At a step boundary:
    projected tries available over ``horizon_s`` (ready now + in-flight groups by their ETAs) against what the trainer
    consumes in that time (C, tries/s from step timings); a projected idle of at least ``min_block_s`` with C/P1 at
    least ``ratio_on`` (P1: server 1's group completions) gives a block of that length, up to ``max_block_s``."""

    def __init__(self, on=True, min_block_s=900.0, max_block_s=1200.0, ratio_on=1.3, horizon_s=1800.0, ema=0.3,
                 switch_s=40.0, start_block=False, end_ready=0, continuous=False, idle_min_s=60.0, idle_end_ready=1,
                 idle_min_serve_s=0.0, hold=False, idle_ready_wait_s=0.0, end_on_batch=False):
        self.on, self.min_block_s, self.max_block_s = bool(on), float(min_block_s), float(max_block_s)
        # end_ready (hero-v3 read: the start block slept 1,760 s while groups were ready from 391 s in): a block ends
        # early once this many finished groups wait for the trainer (0: the planned length)
        self.end_ready = int(end_ready or 0)
        # start_block (hero card-2 gate): before the first step (no trainer rate yet) and with nothing ready, card 2
        # plays until the first in-flight group is due instead of idling through the first groups' play
        self.start_block = bool(start_block)
        # continuous: also mid-batch, once the trainer has waited idle_min_s
        # with nothing ready, card 2 serves until a group is ready (gradients kept) or max_block_s. Group ETAs are not
        # used: a straggler-bound group is overdue by its mean latency yet may take many minutes more
        self.continuous, self.idle_min_s = bool(continuous), float(idle_min_s)
        # v6f2 24 Oct (card 2 at 5 of 128 slots, 54% util): an idle block ends on ``idle_end_ready`` ready groups, and
        # never before ``idle_min_serve_s`` (tries moved back to server 1 at a block's end re-prefill their context)
        self.idle_end_ready, self.idle_min_serve_s = int(idle_end_ready), float(idle_min_serve_s)
        # (v6g step 6: a finished group waited 10.7 min for a second): after this many seconds with
        # a group ready, one ready group ends an idle block (0: off)
        self.idle_ready_wait_s = float(idle_ready_wait_s)
        # end_on_batch (MFU-2, v6i: the first step after the resume waited 4 min on an idle block's 300 s floor with its
        # last two groups ready): an idle block ends, floor or not, once the ready groups complete the trainer's batch
        self.end_on_batch = bool(end_on_batch)
        # hold: at a block's end server 2's tries pause and resume on server 2 at its next block (their
        # context re-prefilled there) instead of moving to server 1 (v6g: the move bypassed card 1's KV, 0.99 + preemptions)
        self.hold = bool(hold)
        self.ratio_on, self.horizon_s, self.ema, self.switch_s = float(ratio_on), float(horizon_s), float(ema), float(switch_s)
        self.c = None  # trainer tries/s while training
        self.done_times = deque(maxlen=64)  # (time, tries) of server-1 group completions

    def observe_step(self, tries, seconds):
        if seconds > 0 and tries > 0:
            rate = tries / seconds
            self.c = rate if self.c is None else (1 - self.ema) * self.c + self.ema * rate

    def observe_group(self, now, tries):
        self.done_times.append((float(now), int(tries)))

    def p1(self, now, window_s=3600.0):
        rows = [(t, n) for t, n in self.done_times if now - t <= window_s]
        if len(rows) < 2:
            return None
        span = max(now - rows[0][0], 1.0)
        return sum(n for _, n in rows) / span

    def decide(self, now, ready_tries, etas):
        """``etas``: (expected finish time, tries) of in-flight groups. Returns ("train", 0) or ("play", seconds)."""
        if not self.on:
            return "train", 0.0
        if self.c is None:
            if not self.start_block or ready_tries:
                return "train", 0.0
            idle = min((t for t, _ in etas), default=now + self.horizon_s) - now - self.switch_s
            cap = max(self.max_block_s, self.horizon_s)  # the start: one long block, not several switches
            return ("play", min(idle, cap)) if idle >= self.min_block_s else ("train", 0.0)
        p1 = self.p1(now)
        if not p1 or self.c / p1 < self.ratio_on:
            return "train", 0.0
        h = self.horizon_s
        supply = ready_tries + sum(n for t, n in etas if t <= now + h)
        idle = h - supply / self.c - self.switch_s
        if idle < self.min_block_s:
            return "train", 0.0
        return "play", min(idle, self.max_block_s)

    def idle(self, ready_groups, waited_s):
        """card2.continuous: the block length card 2 serves while the trainer waits (0: keep waiting)."""
        if not (self.on and self.continuous) or ready_groups or waited_s < self.idle_min_s:
            return 0.0
        return self.max_block_s

    def state_dict(self):
        return {"c": self.c, "done_times": list(self.done_times)}

    def load(self, data):
        self.c = data.get("c")
        self.done_times = deque((tuple(x) for x in data.get("done_times") or []), maxlen=64)


def lag_hist(lags):
    out = {}
    for lag in lags:
        out[str(lag)] = out.get(str(lag), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (kv[0] == "None", int(kv[0]) if kv[0] != "None" else 0)))


def ess(weights):
    """PipelineRL eq. 6: (sum w)^2 / (N sum w^2), 1 for on-policy data."""
    w = [float(x) for x in weights]
    s2 = sum(x * x for x in w)
    return (sum(w) ** 2) / (len(w) * s2) if w and s2 > 0 else None


def balance_scale(up_mass, down_mass):
    """Per-group balance (mg.balance under streamed accumulation): the push-down weights' scale that makes a group's
    push-down mass equal its push-up mass (1 when either side is empty)."""
    return up_mass / down_mass if up_mass > 0 and down_mass > 0 and math.isfinite(up_mass / down_mass) else 1.0
