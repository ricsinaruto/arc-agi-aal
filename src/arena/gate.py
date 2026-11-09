"""KV-aware admission gate: bounded in-flight requests and reserved KV tokens (FIFO)."""

import asyncio
import time
from collections import deque

LANE_DEFAULTS = {"max_inflight": 2, "guard_caps": 2}


def lane_settings(raw=None):
    """Validated background-lane settings (``fixed-2x-index-lane``)."""
    merged = {**LANE_DEFAULTS, **(raw or {})}
    if set(merged) != set(LANE_DEFAULTS):
        raise ValueError(f"Unknown background lane settings: {sorted(set(merged) - set(LANE_DEFAULTS))}")
    if type(merged["max_inflight"]) is not int or merged["max_inflight"] < 1:
        raise ValueError("The lane's max_inflight must be a positive integer")
    if type(merged["guard_caps"]) not in (int, float) or not 1 <= merged["guard_caps"] <= 8:
        raise ValueError("The focus guard must cover one to eight completion caps")
    return merged


class Reservation:
    __slots__ = ("game_id", "tokens", "admitted_at", "lane", "slot", "background")

    def __init__(self, game_id, tokens, admitted_at, lane=False, slot=True, background=False):
        self.game_id, self.tokens, self.admitted_at = game_id, tokens, admitted_at
        self.lane = lane  # admitted under the background-lane rules
        self.slot = slot  # holds a sequence (a request); a block hold of try_acquire(slot=False) does not
        self.background = background  # admitted by try_acquire (background work)


class BackgroundLane:
    """Work-conserving background lane (``fixed-2x-index-lane``, 26 September).

    Phase 2 of the index scheduler runs two focus games, which leave most of the GPU and about
    half of the cache idle. Games outside the focus slots (``GameRecord.lane``: level-0 and parked
    games) may send requests on that leftover capacity at a lower priority:

    - Each focus game (state "active", not in the lane) holds a standing guard of
      ``cost(min(window, latest prompt + guard_caps * cap))``: room for its next request even
      after the current reply (at most one cap) and its tool result are appended.
    - A lane request is admitted only when ``reserved + its cost + sum over focus games of
      max(0, guard - what the game already holds) <= capacity``: its own reservation fits beside
      every in-flight reservation and the unused part of every focus guard. A focus game's
      in-flight reservation is part of its guard (it is released before the game's next request).
    - Queued focus requests always go first; no lane request is admitted while one waits.
    - At most ``max_inflight`` lane requests at once, one per game; lane order is the fewest
      completion tokens spent on the current level, then roster order.

    Preemption: a lane admission requires the plain gate condition ``reserved + cost <= capacity``
    and more (the guard term is never negative), so the sum of reservations never exceeds the
    capacity, ``kv_margin`` x the server's blocks, exactly as without the lane. Each reservation
    is the block cost of the request's full length (prompt + completion cap + the speculative
    lookahead), so under this model every admitted request fits the cache at its maximum length
    and the server never needs to preempt. The guard adds the no-wait property for focus games
    on top: when a focus game's next request is at most its guard, it fits at once whatever lane
    requests are in flight.
    """

    def __init__(self, games, *, window, cap, max_inflight=LANE_DEFAULTS["max_inflight"],
                 guard_caps=LANE_DEFAULTS["guard_caps"]):
        settings = lane_settings({"max_inflight": max_inflight, "guard_caps": guard_caps})
        self.games = {game.game_id: game for game in games}
        self.window, self.cap = int(window), int(cap)
        self.max_inflight, self.guard_caps = settings["max_inflight"], settings["guard_caps"]

    def is_lane(self, game_id):
        game = self.games.get(game_id)
        return game is not None and game.lane

    def order(self, game_id):
        game = self.games[game_id]
        return (game.tokens_on_level(), game.index)

    def focus_games(self):
        return [game for game in self.games.values() if game.state == "active" and not game.lane]

    def guard_tokens(self, game):
        """The focus game's next request at most; the whole window before its first request."""
        if game.latest_prompt_tokens <= 0:
            return self.window
        return int(min(self.window, game.latest_prompt_tokens + self.guard_caps * self.cap))

    def summary(self):
        return {"max_inflight": self.max_inflight, "guard_caps": self.guard_caps,
                "window": self.window, "cap": self.cap}


class KVGate:
    """At most ``slots`` requests in flight and ``capacity_tokens`` reserved, first come first served."""

    def __init__(self, capacity_tokens, slots, log=None, now=time.monotonic, budget=None,
                 opening_burst_requests=1, opening=None, lane=None, background_slots=0):
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
        self._prompt_of = {}  # queued future -> prompt tokens (the commit lane's occupancy estimate)
        self.inflight = 0
        self.queue = deque()  # (future, game_id, tokens)
        self.peak_reserved = 0
        self.peak_inflight = 0
        self.peak_waiting = 0
        self.admitted = 0
        # Background work that yields to a waiting request (arena.parallel_samples); empty: the gate of every earlier
        # build.
        self.queue_hooks = []
        # Sequences for background requests beyond the ``slots`` of the regular ones (arena.parallel_samples: the
        # server's max_num_seqs above the study's generation slots). 0 (every earlier build): background requests
        # share ``slots`` and always leave one free. Otherwise regular requests keep ``slots`` to themselves and
        # background requests use at most ``background_slots``, so the server runs at most slots + background_slots.
        if type(background_slots) is not int or background_slots < 0:
            raise ValueError("background_slots is a non-negative integer")
        self.background_slots = background_slots
        self.background_inflight = 0
        # Background lane (fixed-2x-index-lane); ``None`` keeps the plain FIFO gate.
        self.lane = lane
        if lane is not None:
            self.lane_queue = []  # (future, game_id, tokens), served in lane order
            self.held = {}  # game id -> units reserved by its in-flight requests
            self.lane_inflight = 0
            self._focus_since = {}  # queued future -> when it last became a focus request
            self._mix = {}  # "focus=F,lane=L" -> seconds with F focus and L lane requests in flight
            self._mix_since = self.now()
            self.lane_stats = {"lane_admitted": 0, "lane_wait_seconds": 0.0, "peak_lane_inflight": 0,
                               "opening_admitted": 0, "focus_admitted": 0, "focus_wait_seconds": 0.0, "focus_waits": 0,
                               "focus_max_wait_seconds": 0.0, "focus_queued_beside_lane": 0}
        # Commit lane (fixed-2x-commit, ``arena.commit.CommitLane``): preemptible lane requests admitted on
        # an occupancy estimate; focus requests fit against the focus reservations alone.
        self.commit = lane is not None and bool(getattr(lane, "preemptible", False))
        if self.commit:
            self.lane_reserved = 0  # units reserved by lane requests (not counted against focus requests)
            self.flights = {}  # game id -> [(prompt tokens, admitted at)] of its in-flight requests
            self._timeline = []  # (start, end, focus in flight, lane in flight) for preemption attribution
            self.lane_stats.update(focus_over_estimate=0, peak_lane_estimate_units=0,
                                   lane_admitted_tokens=0)

    def _units(self, tokens):
        return self.budget.cost(tokens) if self.budget else int(min(tokens, self.capacity))

    def _fits(self, tokens):
        # With the commit lane, lane reservations do not hold up focus requests: vLLM (priority policy)
        # preempts lane requests before any focus request, so focus requests keep the plain rule on the
        # focus reservations and never wait at the gate on a lane request.
        reserved = self.reserved - self.lane_reserved if self.commit else self.reserved
        inflight = self.inflight - self.background_inflight if self.background_slots else self.inflight
        return inflight < self.slots and reserved + tokens <= self.capacity

    async def acquire(self, game_id, tokens, prompt=None):
        """``prompt``: the request's prompt tokens (the commit lane's estimate); by default the
        requested length minus the completion cap."""
        requested_tokens = int(tokens)
        tokens = self.budget.cost(requested_tokens) if self.budget else int(min(tokens, self.capacity))
        if tokens > self.capacity:
            units = getattr(self.budget, "units", "cache blocks")
            raise ValueError(f"Single request needs {tokens} {units}; budget is {self.capacity}")
        if self.lane is not None:
            prompt = max(0, int(requested_tokens - self.lane.cap if prompt is None else prompt))
            return await self._acquire_with_lane(game_id, tokens, prompt)
        if not self.queue and self._fits(tokens):
            self._selection(game_id, affinity=False)
            return self._admit(game_id, tokens, waited=0.0)
        future = asyncio.get_running_loop().create_future()
        self.queue.append((future, game_id, tokens))
        self.peak_waiting = max(self.peak_waiting, len(self.queue))
        started = self.now()
        if self.queue_hooks:
            self._yield_to_queue()
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

    def try_acquire(self, game_id, tokens, *, headroom_tokens=0, units=None, event="background_admit", slot=True):
        """Non-blocking reservation for background work (``arena.keep_alive``, ``arena.parallel_samples``), or
        ``None``.

        Admitted only on the plain FIFO gate when no request waits, a slot stays free for the next request and
        ``headroom_tokens`` more (one full request) would still fit beside it. The reservation is the same exact block
        cost as ``acquire``'s (or ``units``, the caller's exact cost of a request that shares blocks another
        reservation already holds), so the sum of reservations stays within the capacity. Logged as ``event``
        (default ``background_admit``; never ``admit``: the request logs' analyses pair admits with gameplay
        requests). ``slot=False``: a block hold that sends no request (no sequence slot is taken).

        Sequences: with ``background_slots`` 0 a background request needs a free slot beside the next regular one
        (``inflight + 1 < slots``); otherwise it needs one of the ``background_slots`` (regular requests keep their
        ``slots``)."""
        if self.lane is not None or self.queue:
            return None
        units = self._units(tokens) if units is None else int(units)
        extra = self._units(headroom_tokens) if headroom_tokens else 0
        if slot:
            full = (self.background_inflight >= self.background_slots if self.background_slots
                    else self.inflight + 1 >= self.slots)
            if full:
                return None
        if self.reserved + units + extra > self.capacity:
            return None
        if slot:
            self.inflight += 1
            self.background_inflight += 1
        self.reserved += units
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        if self.log:
            self.log({"event": event, "game": game_id, "tokens": units,
                      "reserved_after": self.reserved, "inflight_after": self.inflight})
        return Reservation(game_id, units, self.now(), slot=slot, background=True)

    def release_background(self, reservation, event="background_release"):
        if reservation.slot:
            self.inflight -= 1
            self.background_inflight -= 1
        self.reserved -= reservation.tokens
        if self.log:
            self.log({"event": event, "game": reservation.game_id, "tokens": reservation.tokens,
                      "held": round(self.now() - reservation.admitted_at, 3),
                      "reserved_after": self.reserved, "inflight_after": self.inflight})
        self._pump()

    def _yield_to_queue(self):
        """A request waits at the plain gate: ask the background work that can yield (``queue_hooks``, the parallel
        samples' siblings) to free what the head needs. A hook aborts its requests and releases their reservations
        after its abort grace; it returns the ``(units, slots)`` freed or pending release."""
        if not self.queue_hooks or not self.queue:
            return
        tokens = self.queue[0][2]
        deficit = self.reserved + tokens - self.capacity
        need_slot = (self.inflight - self.background_inflight if self.background_slots else self.inflight) >= self.slots
        for hook in list(self.queue_hooks):
            if deficit <= 0 and not need_slot:
                break
            units, slots = hook(max(0, deficit), need_slot)
            deficit -= units
            if slots:
                need_slot = False

    def _admit(self, game_id, tokens, *, waited, account=True, lane=False, focus_wait=None, prompt=0):
        if account:
            self._take(game_id, tokens, prompt=prompt)
        self.admitted += 1
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        entry = {"event": "admit", "game": game_id, "tokens": tokens, "waited": round(waited, 3),
                 "reserved_after": self.reserved, "inflight_after": self.inflight}
        if self.lane is not None:
            self._count_admission(lane, waited if focus_wait is None else focus_wait)
            entry.update(lane=lane, lane_inflight_after=self.lane_inflight)
        if self.log:
            self.log(entry)
        return Reservation(game_id, tokens, self.now(), lane)

    def _take(self, game_id, tokens, lane=False, prompt=0):
        if self.lane is not None:
            self._tally()
            self.held[game_id] = self.held.get(game_id, 0) + tokens
            self.lane_inflight += lane
            if self.commit:
                self.lane_reserved += tokens if lane else 0
                self.flights.setdefault(game_id, []).append((int(prompt), self.now()))
                self.lane.note_admission(game_id)
        self.inflight += 1
        self.reserved += tokens

    def _give(self, game_id, tokens, lane=False):
        if self.lane is not None:
            self._tally()
            self.held[game_id] -= tokens
            if not self.held[game_id]:
                del self.held[game_id]
            self.lane_inflight -= lane
            if self.commit:
                self.lane_reserved -= tokens if lane else 0
                flights = self.flights.get(game_id)
                if flights:
                    flights.pop(0)
                    if not flights:
                        del self.flights[game_id]
        self.inflight -= 1
        self.reserved -= tokens

    def release(self, reservation):
        self._give(reservation.game_id, reservation.tokens, reservation.lane)
        if self.log:
            self.log(
                {"event": "release", "game": reservation.game_id, "tokens": reservation.tokens,
                 "held": round(self.now() - reservation.admitted_at, 3),
                 "reserved_after": self.reserved, "inflight_after": self.inflight}
            )
        self._pump()

    def _pump(self):
        if self.lane is not None:
            self._sort_roles()
            if self._pump_focus():
                (self._pump_commit_lane if self.commit else self._pump_lane)()
            return
        self._pump_focus()

    def _pump_focus(self):
        """The FIFO queue (every request without the lane); True once it is empty."""
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
                return False
            del self.queue[chosen]
            self._head_bypasses = self._head_bypasses + 1 if chosen else 0
            self._selection(game_id, affinity=chosen > 0)
            # Account now so the next head sees the reservation before the waiter resumes.
            self._take(game_id, tokens, prompt=self._prompt_of.pop(future, 0))
            future.set_result(None)
        return True

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

    # --- background lane (fixed-2x-index-lane) ----------------------------------------------

    async def _acquire_with_lane(self, game_id, tokens, prompt=0):
        """``acquire`` with the lane: focus requests keep the FIFO rules (and the opening
        affinity) and never queue behind lane requests; lane requests wait in lane order."""
        self._sort_roles()  # a request queued under a stale role must not hold up this one
        lane = self.lane.is_lane(game_id)
        started = self.now()
        if not lane and not self.queue and self._fits(tokens):
            self._selection(game_id, affinity=False)
            return self._admit(game_id, tokens, waited=0.0, prompt=prompt)
        future = asyncio.get_running_loop().create_future()
        self._prompt_of[future] = prompt
        if lane:
            self.lane_queue.append((future, game_id, tokens))
        else:
            self.queue.append((future, game_id, tokens))
            self._focus_since[future] = started
            # A focus request that queues while lane requests run (it may also wait on the other
            # focus game, as without the lane); the guard should keep this near zero.
            self.lane_stats["focus_queued_beside_lane"] += self.lane_inflight > 0
        self.peak_waiting = max(self.peak_waiting, len(self.queue) + len(self.lane_queue))
        if lane:
            self._pump()  # idle capacity admits it at once
        try:
            admitted_lane = await future
        except asyncio.CancelledError:
            if future.cancelled():
                if self.queue and self.queue[0][0] is future:
                    self._head_bypasses = 0
                self.queue = deque(item for item in self.queue if item[0] is not future)
                self.lane_queue = [item for item in self.lane_queue if item[0] is not future]
            else:  # admitted by the pump but cancelled before resuming
                self._give(game_id, tokens, bool(future.result()))
            self._focus_since.pop(future, None)
            self._prompt_of.pop(future, None)
            self._pump()
            raise
        self._prompt_of.pop(future, None)
        now = self.now()
        since = self._focus_since.pop(future, started)
        return self._admit(game_id, tokens, waited=now - started, account=False,
                           lane=bool(admitted_lane), focus_wait=now - since)

    def refresh(self):
        """Re-evaluate the queues after the scheduler changed roles or guards (every tick)."""
        self._pump()

    def _sort_roles(self):
        """Move queued requests whose game changed role: a promoted lane game's request joins the
        focus queue (from now on it counts as a focus wait); a parked focus game's joins the lane."""
        demoted = [item for item in self.queue if not item[0].done() and self.lane.is_lane(item[1])]
        promoted = [item for item in self.lane_queue if not item[0].done() and not self.lane.is_lane(item[1])]
        if demoted:
            if self.queue and self.queue[0] in demoted:
                self._head_bypasses = 0
            self.queue = deque(item for item in self.queue if item not in demoted)
            for item in demoted:
                self._focus_since.pop(item[0], None)
            self.lane_queue.extend(demoted)
        if promoted:
            self.lane_queue = [item for item in self.lane_queue if item not in promoted]
            now = self.now()
            for item in promoted:
                self._focus_since[item[0]] = now
            self.queue.extend(promoted)

    def guard_deficit(self):
        """Units the focus games may still need beyond what they hold now."""
        deficit = 0
        for game in self.lane.focus_games():
            deficit += max(0, self._units(self.lane.guard_tokens(game)) - self.held.get(game.game_id, 0))
        return deficit

    def _lane_fits(self, tokens):
        return (self.inflight < self.slots and self.lane_inflight < self.lane.max_inflight
                and self.reserved + tokens + self.guard_deficit() <= self.capacity)

    def _pump_lane(self):
        # Lane order: the game with the fewest completion tokens on its level first. One request per
        # lane game (a game already holding a reservation is skipped, never blocks the head); if the
        # first candidate does not fit, nothing behind it goes (no starving it with smaller requests).
        while True:
            self.lane_queue = [item for item in self.lane_queue if not item[0].done()]
            candidates = [item for item in self.lane_queue if item[1] not in self.held]
            if not candidates:
                return
            future, game_id, tokens = min(candidates, key=lambda item: self.lane.order(item[1]))
            if not self._lane_fits(tokens):
                return
            self.lane_queue = [item for item in self.lane_queue if item[0] is not future]
            self._take(game_id, tokens, lane=True)
            future.set_result(True)

    def _tally(self):
        # Seconds per in-flight mix; the opening (every game focus, no lane) is kept apart.
        now = self.now()
        if self.commit and now > self._mix_since:
            self._timeline.append((self._mix_since, now, self.inflight - self.lane_inflight, self.lane_inflight))
        key = f"{'opening:' if self.opening() else ''}focus={self.inflight - self.lane_inflight},lane={self.lane_inflight}"
        self._mix[key] = self._mix.get(key, 0.0) + max(0.0, now - self._mix_since)
        self._mix_since = now

    def _count_admission(self, lane, waited):
        stats = self.lane_stats
        if lane:
            stats["lane_admitted"] += 1
            stats["lane_wait_seconds"] += waited
            stats["peak_lane_inflight"] = max(stats["peak_lane_inflight"], self.lane_inflight)
            return
        if self.opening():
            stats["opening_admitted"] += 1  # the opening's contention is not a phase-2 focus wait
            return
        stats["focus_admitted"] += 1
        if self.commit and self.lane_estimate() > self.lane.budget_units():
            # The focus request (admitted here on the focus reservations) finds the estimated cache past
            # the lane fraction: vLLM may queue it until lane requests yield blocks (server-side wait).
            stats["focus_over_estimate"] += 1
        stats["focus_wait_seconds"] += waited
        stats["focus_waits"] += waited > 0
        stats["focus_max_wait_seconds"] = max(stats["focus_max_wait_seconds"], waited)

    def lane_snapshot(self):
        self._tally()
        occupancy = ({"lane_estimate_units": self.lane_estimate(), "lane_reserved": self.lane_reserved}
                     if self.commit else {"guard_deficit": self.guard_deficit()})
        return {**self.lane.summary(), **{k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in self.lane_stats.items()},
                "lane_inflight": self.lane_inflight, "lane_waiting": len(self.lane_queue),
                **occupancy,
                "inflight_seconds": {k: round(v, 3) for k, v in sorted(self._mix.items())}}

    # --- commit lane (fixed-2x-commit) ------------------------------------------------------

    def lane_estimate(self):
        """Estimated cache units in use or about to be (``arena.commit.CommitLane``): every in-flight
        request's prompt plus what it has generated so far (rate x age, at most the cap), and every focus
        game without a request in flight its latest prompt plus the headroom."""
        now, units, headroom = self.now(), 0, self.lane.headroom()
        for flights in self.flights.values():
            for prompt, started in flights:
                units += self._units(prompt + self.lane.generated(now - started))
        for game in self.lane.focus_games():
            if game.game_id not in self.held and game.latest_prompt_tokens > 0:
                units += self._units(game.latest_prompt_tokens + headroom)
        return units

    def _commit_lane_fits(self, prompt):
        if self.lane_inflight >= self.lane.max_inflight or self.inflight >= self.slots:
            return False
        estimate = self.lane_estimate()
        stats = self.lane_stats
        stats["peak_lane_estimate_units"] = max(stats["peak_lane_estimate_units"], estimate)
        return estimate + self._units(prompt + self.lane.headroom()) <= self.lane.budget_units()

    def _pump_commit_lane(self):
        # Lane order (``CommitLane.order``); one request per lane game (a game already holding a
        # reservation is skipped); if the first candidate does not fit, nothing behind it goes.
        while True:
            self.lane_queue = [item for item in self.lane_queue if not item[0].done()]
            candidates = [item for item in self.lane_queue if item[1] not in self.held]
            if not candidates:
                return
            future, game_id, tokens = min(candidates, key=lambda item: self.lane.order(item[1]))
            prompt = self._prompt_of.get(future, 0)
            if not self._commit_lane_fits(prompt):
                return
            self.lane_queue = [item for item in self.lane_queue if item[0] is not future]
            self._prompt_of.pop(future, None)
            self._take(game_id, tokens, lane=True, prompt=prompt)
            self.lane_stats["lane_admitted_tokens"] += prompt
            future.set_result(True)

    def exposure(self, start, end):
        """The in-flight mixes over ``[start, end]`` (gate clock): ``focus_only`` if at some moment a
        focus request was in flight with no lane request, ``lane`` if a lane request was, ``any`` if
        anything was. Drops the timeline before ``start`` (queries come with increasing ``start``)."""
        self._tally()
        self._timeline = [seg for seg in self._timeline if seg[1] >= start]
        segments = [seg for seg in self._timeline if seg[0] <= end]
        # The current mix (since the last change) counts up to now.
        current = (self.inflight - self.lane_inflight, self.lane_inflight)
        mixes = [(focus, lane) for _, _, focus, lane in segments] + [current]
        return {"focus_only": any(focus > 0 and lane == 0 for focus, lane in mixes),
                "lane": any(lane > 0 for _, lane in mixes),
                "any": any(focus + lane > 0 for focus, lane in mixes)}

    def snapshot(self):
        if self.lane is not None:
            return {**self._snapshot(), "lane": self.lane_snapshot()}
        return self._snapshot()

    def _snapshot(self):
        extra = ({"background_slots": self.background_slots, "background_inflight": self.background_inflight}
                 if self.background_slots else {})
        return {
            **extra,
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

    Entering waits until the game is active (or in the background lane), then reserves
    ``min(window, estimate + cap)`` tokens in the gate; leaving releases them. Diagnostics distinguish scheduler
    parking from eligible waiting without changing either admission policy.
    """

    def __init__(self, gate, game, cap, window):
        self.gate, self.game, self.cap, self.window = gate, game, int(cap), int(window)
        self.next_estimate = 0
        self.reservation = None
        self._stack = []
        self._timing_stack = []
        # A variant's aside request (cross-game notes written at a park) may go out while the
        # game is parked: it still takes a gate reservation but skips the active wait.
        self.allow_parked = False

    def reserve(self, prompt_estimate):
        self.next_estimate = max(0, int(prompt_estimate))

    def tokens(self):
        if self.gate.budget is not None:
            # Never hide an oversized summary/request behind the agent's target window.
            # The physical layout enforces the actual native server limit separately.
            return self.next_estimate + self.cap
        return min(self.window, self.next_estimate + self.cap)

    async def __aenter__(self):
        started = self.gate.now()
        active = self.game.clock.active_seconds()
        parked = self.game.clock.parked_seconds()
        attempt = {
            "attempt": len(self.game.admission_attempts) + 1,
            "status": "waiting",
            "requested_tokens": self.tokens(),
            "started_wall_seconds": round(self.game.clock.global_clock.elapsed(), 3),
            "start_phase": getattr(self.game.observer, "phase", None),
        }
        self.game.admission_attempts.append(attempt)
        self._record("request_wait_started", attempt)
        try:
            while True:
                # A lane game (fixed-2x-index-lane) is parked but may send background requests.
                if not self.allow_parked and not self.game.lane:
                    await self.game.clock.wait_active()
                reservation = await self.gate.acquire(self.game.game_id, self.tokens(), prompt=self.next_estimate)
                # A game may have been parked while queued for GPU admission. It has
                # not sent a request yet, so release the slot and wait for re-admission
                # (with the lane: re-queue as a lane request unless admitted as one).
                if self.allow_parked or not self.game.clock.parked or (reservation.lane and self.game.lane):
                    break
                self.gate.release(reservation)
        except BaseException as exc:
            self._waiting_times(attempt, started, active, parked)
            attempt.update(status="cancelled_wait" if isinstance(exc, asyncio.CancelledError)
                           else "failed_wait", error_type=type(exc).__name__)
            self._record("request_wait_closed", attempt)
            raise
        self._waiting_times(attempt, started, active, parked)
        attempt.update(status="admitted", tokens=reservation.tokens,
                       admitted_wall_seconds=round(self.game.clock.global_clock.elapsed(), 3),
                       admitted_phase=getattr(self.game.observer, "phase", None))
        self._stack.append(reservation)
        self._timing_stack.append((attempt, self.gate.now(),
                                   self.game.clock.active_seconds(), self.game.clock.parked_seconds()))
        self._record("request_admitted", attempt)
        self.game.latest_prompt_tokens = self.next_estimate
        request = {"tokens": reservation.tokens, "gate_wait_seconds": attempt["wait_seconds"],
                   "wall_seconds": round(self.game.clock.global_clock.elapsed(), 3)}
        if self.gate.lane is not None:
            request["lane"] = reservation.lane
            self.game.note_request(lane=reservation.lane)
        if getattr(self.game, "slot", None) is not None:
            request["slot"] = self.game.slot  # fixed-2x-deep-explore: deep or exploration slot
        if hasattr(self.game, "note_slot_request"):
            self.game.note_slot_request()
        self.game.requests.append(request)
        return self

    def request_priority(self):
        """The vLLM ``priority`` of the request being sent under this slot with the commit lane
        (fixed-2x-commit, a server started with ``--scheduling-policy priority``): 1 for a lane
        request, 0 for a focus or opening request (lower is more important). ``None`` otherwise: the
        field is not sent (vLLM rejects a non-zero priority on an FCFS server)."""
        if not getattr(self.gate, "commit", False) or not self._stack:
            return None
        return 1 if self._stack[-1].lane else 0

    async def __aexit__(self, *exc):
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
