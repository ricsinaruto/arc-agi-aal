"""KV-aware admission gate: bounded in-flight requests and reserved KV tokens (FIFO).

rl-min: the background lane (fixed-2x-index-lane), the commit lane and the background reservations (keep_alive /
parallel_samples) are gone; no live study builds them. The opening affinity stays: the submission (v8.5p, v87b)
runs it (opening_burst_requests 3) and the held-out read plays whole games through the same gate."""

import asyncio
import time
from collections import deque


class Reservation:
    __slots__ = ("game_id", "tokens", "admitted_at")

    def __init__(self, game_id, tokens, admitted_at):
        self.game_id, self.tokens, self.admitted_at = game_id, tokens, admitted_at


class KVGate:
    """At most ``slots`` requests in flight and ``capacity_tokens`` reserved, first come first served."""

    def __init__(self, capacity_tokens, slots, log=None, now=time.monotonic, budget=None,
                 opening_burst_requests=1, opening=None):
        if type(opening_burst_requests) is not int or not 1 <= opening_burst_requests <= 3:
            raise ValueError("Opening burst must be one to three requests")
        self.opening_burst_requests = opening_burst_requests
        self.opening = opening or (lambda: False)
        self._credits = {}
        self._head_bypasses = 0
        self.affinity_admissions = 0
        self.capacity, self.slots, self.log, self.now = int(capacity_tokens), int(slots), log, now
        self.budget = budget
        self.reserved = 0
        self.inflight = 0
        self.queue = deque()  # (future, game_id, tokens)
        self.peak_reserved = 0
        self.peak_inflight = 0
        self.peak_waiting = 0
        self.admitted = 0
        self.held = False  # rl_seq kv_cap: the engine's KV is at the cap; nothing new while anything is in flight

    def _fits(self, tokens):
        if self.held and self.inflight > 0:
            return False
        return self.inflight < self.slots and self.reserved + tokens <= self.capacity

    def hold(self, on):
        self.held = bool(on)
        if not self.held:
            self._pump()

    async def acquire(self, game_id, tokens):
        tokens = self.budget.cost(int(tokens)) if self.budget else int(min(tokens, self.capacity))
        if tokens > self.capacity:
            units = getattr(self.budget, "units", "cache blocks")
            raise ValueError(f"Single request needs {tokens} {units}; budget is {self.capacity}")
        if not self.queue and self._fits(tokens):
            self._selection(game_id, affinity=False)
            return self._admit(game_id, tokens, waited=0.0)
        future = asyncio.get_running_loop().create_future()
        self.queue.append((future, game_id, tokens))
        self.peak_waiting = max(self.peak_waiting, len(self.queue))
        started = self.now()
        try:
            await future
        except asyncio.CancelledError:
            # Cancelling a task also cancels the future it is awaiting. A cancelled
            # future is done, but was never admitted; only a completed, non-cancelled
            # future carries the reservation allocated by _pump.
            if future.cancelled():
                if self.queue and self.queue[0][0] is future:
                    self._head_bypasses = 0
                self.queue = deque(item for item in self.queue if item[0] is not future)
            else:  # admitted by the pump but cancelled before resuming
                self.inflight -= 1
                self.reserved -= tokens
            self._pump()
            raise
        return self._admit(game_id, tokens, waited=self.now() - started, account=False)

    def _admit(self, game_id, tokens, *, waited, account=True):
        if account:
            self._take(tokens)
        self.admitted += 1
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        if self.log:
            self.log({"event": "admit", "game": game_id, "tokens": tokens, "waited": round(waited, 3),
                      "reserved_after": self.reserved, "inflight_after": self.inflight})
        return Reservation(game_id, tokens, self.now())

    def _take(self, tokens):
        self.inflight += 1
        self.reserved += tokens

    def release(self, reservation):
        self.inflight -= 1
        self.reserved -= reservation.tokens
        if self.log:
            self.log(
                {"event": "release", "game": reservation.game_id, "tokens": reservation.tokens,
                 "held": round(self.now() - reservation.admitted_at, 3),
                 "reserved_after": self.reserved, "inflight_after": self.inflight}
            )
        self._pump()

    def _pump(self):
        """Admit from the head of the FIFO queue while it fits."""
        # Opening-only bounded continuation affinity. Never hold an idle reservation
        # for a game doing tool work. After at most two bypasses, the FIFO head wins.
        while self.queue:
            future, game_id, tokens = self.queue[0]
            if future.done():
                self.queue.popleft()
                self._head_bypasses = 0
                continue
            chosen = 0
            if self.opening() and self._head_bypasses < self.opening_burst_requests - 1:
                for index, (candidate, identity, cost) in enumerate(self.queue):
                    if (not candidate.done() and self._credits.get(identity, 0) > 0
                            and self._fits(cost)):
                        chosen = index
                        break
            future, game_id, tokens = self.queue[chosen]
            if not self._fits(tokens):
                return
            del self.queue[chosen]
            self._head_bypasses = self._head_bypasses + 1 if chosen else 0
            self._selection(game_id, affinity=chosen > 0)
            # Account now so the next head sees the reservation before the waiter resumes.
            self._take(tokens)
            future.set_result(None)

    def _selection(self, game_id, *, affinity):
        if not self.opening() or self.opening_burst_requests == 1:
            self._credits.clear()
            return
        if affinity:
            self._credits[game_id] -= 1
            self.affinity_admissions += 1
        else:
            self._credits[game_id] = self.opening_burst_requests - 1
        if self.log:
            self.log({"event": "opening_selection", "game": game_id,
                      "policy": "continuation" if affinity else "fifo",
                      "credits_remaining": self._credits[game_id],
                      "head_bypasses": self._head_bypasses})

    def snapshot(self):
        return {
            "capacity": self.capacity,
            "units": getattr(self.budget, "units", "cache_blocks") if self.budget else "tokens",
            "slots": self.slots,
            "reserved": self.reserved,
            "inflight": self.inflight,
            "waiting": len(self.queue),
            "peak_reserved": self.peak_reserved,
            "peak_inflight": self.peak_inflight,
            "peak_waiting": self.peak_waiting,
            "admitted": self.admitted,
            "opening_burst_requests": self.opening_burst_requests,
            "affinity_admissions": self.affinity_admissions,
        }


class GateSlot:
    """Per-game async context manager standing in for the request semaphore.

    Entering waits until the game is active, then reserves
    ``min(window, estimate + cap)`` tokens in the gate; leaving releases them. Diagnostics distinguish scheduler
    parking from eligible waiting without changing either admission policy.
    """

    def __init__(self, gate, game, cap, window, reserve=None):
        self.gate, self.game, self.cap, self.window = gate, game, int(cap), int(window)
        # completion tokens reserved per request (level_bench.gate_completion_reserve, e.g. the observed p95) instead
        # of the full cap; a longer completion still runs (vLLM preempts if its KV runs out). None: the cap.
        self.reserve_completion = self.cap if reserve is None else min(self.cap, int(reserve))
        self.next_estimate = 0
        self.reservation = None
        self._stack = []
        self._timing_stack = []
        self._turn_stack = []  # per enter: the pool that counted this request in flight (None: not counted)
        self.turn_pool = None  # level_bench: a time-share pool with a turn gate (SharedSlots.turn) waits here first

    def reserve(self, prompt_estimate):
        self.next_estimate = max(0, int(prompt_estimate))

    def tokens(self):
        if self.gate.budget is not None:
            # Never hide an oversized summary/request behind the agent's target window.
            # The physical layout enforces the actual native server limit separately.
            return self.next_estimate + self.reserve_completion
        return min(self.window, self.next_estimate + self.reserve_completion)

    async def __aenter__(self):
        started = self.gate.now()
        active = self.game.clock.active_seconds()
        parked = self.game.clock.parked_seconds()
        attempt = {
            "attempt": len(self.game.admission_attempts) + 1,
            "status": "waiting",
            "requested_tokens": self.tokens(),
            "started_wall_seconds": round(self.game.clock.global_clock.elapsed(), 3),
            "start_phase": None,
        }
        self.game.admission_attempts.append(attempt)
        self._record("request_wait_started", attempt)
        try:
            if self.turn_pool is not None:
                await self.turn_pool.turn(self)
            reservation = await self.gate.acquire(self.game.game_id, self.tokens())
        except BaseException as exc:
            self._waiting_times(attempt, started, active, parked)
            attempt.update(status="cancelled_wait" if isinstance(exc, asyncio.CancelledError)
                           else "failed_wait", error_type=type(exc).__name__)
            self._record("request_wait_closed", attempt)
            raise
        self._waiting_times(attempt, started, active, parked)
        attempt.update(status="admitted", tokens=reservation.tokens,
                       admitted_wall_seconds=round(self.game.clock.global_clock.elapsed(), 3),
                       admitted_phase=None)
        self._stack.append(reservation)
        pool = self.turn_pool  # counted on this pool and ended there, even if the try moves pools while in flight
        self._turn_stack.append(pool if pool is not None and hasattr(pool, "begin") and pool.begin(self) else None)
        self._timing_stack.append((attempt, self.gate.now(),
                                   self.game.clock.active_seconds(), self.game.clock.parked_seconds()))
        self._record("request_admitted", attempt)
        request = {"tokens": reservation.tokens, "gate_wait_seconds": attempt["wait_seconds"],
                   "wall_seconds": round(self.game.clock.global_clock.elapsed(), 3)}
        self.game.requests.append(request)
        return self

    async def __aexit__(self, *exc):
        pool = self._turn_stack.pop() if self._turn_stack else None
        if pool is not None:
            pool.end(self)  # the turn gate's place in flight
        reservation = self._stack.pop()
        attempt, started, active, parked = self._timing_stack.pop()
        self.gate.release(reservation)
        kind = exc[0] if exc else None
        attempt.update(
            status=("cancelled_context" if kind and issubclass(kind, asyncio.CancelledError)
                    else "failed_context" if kind else "closed"),
            context_seconds=round(max(0.0, self.gate.now() - started), 3),
            context_active_seconds=round(max(0.0, self.game.clock.active_seconds() - active), 3),
            context_parked_seconds=round(max(0.0, self.game.clock.parked_seconds() - parked), 3),
            ended_wall_seconds=round(self.game.clock.global_clock.elapsed(), 3),
        )
        if kind:
            attempt["error_type"] = kind.__name__
        self._record("request_context_closed", attempt)
        return False

    def _waiting_times(self, attempt, started, active, parked):
        total = max(0.0, self.gate.now() - started)
        eligible = max(0.0, self.game.clock.active_seconds() - active)
        paused = max(0.0, self.game.clock.parked_seconds() - parked)
        attempt.update(
            wait_seconds=round(total, 3),
            eligible_wait_seconds=round(eligible, 3),
            parked_wait_seconds=round(paused, 3),
            unclassified_wait_seconds=round(max(0.0, total - eligible - paused), 3),
            wait_ended_wall_seconds=round(self.game.clock.global_clock.elapsed(), 3),
        )

    def _record(self, event, attempt):
        if self.gate.log:
            self.gate.log({"event": event, "game": self.game.game_id, **attempt})
