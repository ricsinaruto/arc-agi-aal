"""Cache keep-alive for parked games (prefix cache v3 option (a), 29 September; prefix-cache-v3.md).

Parked hero-v2 games wait a median ~3,300 s for re-admission while the server's free block pool turns over in ~4-10
minutes, so a re-admitted game usually re-prefills its whole prefix (1.99M of hero-v2's 9.48M uncached tokens in
phase 2). The server cannot know which parked games come back soon; the arena scheduler can (its backlog order). This
service keeps the next ``games`` parked games' prefixes warm: when such a game's prefix was last touched (its last
request's end, or its last ping) ``every_seconds`` ago, it sends the game's pending request once with
``max_tokens=1`` (the exact prompt the game will send when re-admitted, held since its admission count), which moves
the prefix's blocks to the LRU tail of the server's free queue. The sampled token is discarded; the game's messages,
clock and budgets are untouched.

Refresh only: a prefix untouched for ``max_age_seconds`` or more is presumed evicted and is not pinged (a ping would
re-prefill it, moving work earlier at best), and with ``--enable-prompt-tokens-details`` a ping whose cached tokens
are under half its prompt marks the game as evicted until its next admission.

Gate safety: a ping takes an ordinary gate reservation (``KVGate.try_acquire``) of its exact cost (the admission
count + 1 token, in cache blocks), only when no request is waiting, a generation slot stays free and one full-window
request more still fits. The sum of reservations therefore stays within ``kv_margin`` x the server's blocks, as for
every request, so the server never needs to preempt; a ping never queues, so it never delays a game's request. It
runs only with the plain FIFO gate (not with the background or commit lanes).
"""

import asyncio
import copy
import time

DEFAULTS = {"games": 20, "every_seconds": 120.0, "max_age_seconds": 240.0, "timeout_seconds": 120.0}


def settings(raw):
    """``None`` / ``False`` (off), ``True`` (the defaults) or a dict overriding them."""
    if raw is None or raw is False:
        return None
    merged = dict(DEFAULTS) if raw is True else {**DEFAULTS, **raw} if isinstance(raw, dict) else None
    if merged is None or set(merged) != set(DEFAULTS):
        raise ValueError(f"keep_alive is true or a dict with keys {sorted(DEFAULTS)}")
    if type(merged["games"]) is not int or not 1 <= merged["games"] <= 110:
        raise ValueError("keep_alive.games is an integer in 1..110")
    for key in ("every_seconds", "max_age_seconds", "timeout_seconds"):
        if type(merged[key]) not in (int, float) or merged[key] <= 0:
            raise ValueError(f"keep_alive.{key} is a positive number of seconds")
    if merged["max_age_seconds"] <= merged["every_seconds"]:
        raise ValueError("keep_alive.max_age_seconds must exceed every_seconds")
    return {k: (float(v) if k != "games" else v) for k, v in merged.items()}


class KeepAlive:
    """Pings the first ``games`` parked games of the scheduler's backlog order (see the module docstring)."""

    def __init__(self, scheduler, gate, agents, spec, *, window, log=None, now=time.monotonic):
        if gate.lane is not None:
            raise ValueError("keep_alive runs with the plain FIFO gate only (no background or commit lane)")
        self.scheduler, self.gate, self.agents = scheduler, gate, agents
        self.spec, self.window, self.log, self.now = spec, int(window), log, now
        self.inflight = {}  # game id -> task
        self.evicted = set()  # games whose ping found the prefix gone, until their next admission
        self.admits = {}
        self.stats = {"pings": 0, "ping_prompt_tokens": 0, "ping_cached_tokens": 0, "ping_seconds": 0.0,
                      "found_evicted": 0, "skipped_no_room": 0, "skipped_stale": 0, "errors": 0}

    def targets(self):
        parked = [g for g in self.scheduler.backlog() if g.state == "parked"]
        return parked[:self.spec["games"]]

    def tick(self):
        now = self.now()
        for game in self.scheduler.games:
            admits = sum(e.get("event") == "admit" for e in game.admissions)
            if self.admits.get(game.game_id) != admits:
                self.admits[game.game_id] = admits
                self.evicted.discard(game.game_id)
        for game in self.targets():
            agent = self.agents.get(game.game_id)
            pending = getattr(agent, "cache_ping_request", None)
            if pending is None or game.game_id in self.inflight or game.game_id in self.evicted:
                continue
            touched = getattr(agent, "cache_touched", None)
            if touched is None:
                continue
            age = now - touched
            if age < self.spec["every_seconds"]:
                continue
            if age >= self.spec["max_age_seconds"]:
                self.stats["skipped_stale"] += 1
                continue
            payload, count = pending
            reservation = self.gate.try_acquire(game.game_id, count + 1, headroom_tokens=self.window)
            if reservation is None:
                self.stats["skipped_no_room"] += 1
                return  # no room now: nothing else fits either
            task = asyncio.create_task(self.ping(game, agent, payload, count, reservation, age))
            self.inflight[game.game_id] = task

    async def ping(self, game, agent, payload, count, reservation, age):
        started = self.now()
        body = {**copy.deepcopy(payload), "max_tokens": 1}
        body.pop("priority", None)
        record = {"event": "cache_ping", "game": game.game_id, "prompt_estimate": count, "age_seconds": round(age, 1)}
        try:
            response = await agent.client.client.post("chat/completions", json=body,
                                                      timeout=self.spec["timeout_seconds"])
            response.raise_for_status()
            usage = response.json().get("usage") or {}
            prompt = int(usage.get("prompt_tokens") or 0)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            self.stats["pings"] += 1
            self.stats["ping_prompt_tokens"] += prompt
            if cached is not None:
                self.stats["ping_cached_tokens"] += int(cached)
                if int(cached) < prompt / 2:
                    self.evicted.add(game.game_id)
                    self.stats["found_evicted"] += 1
            record.update(prompt_tokens=prompt, cached_tokens=cached)
            if self.agents.get(game.game_id) is agent:
                agent.cache_touched = self.now()
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        except Exception as exc:  # noqa: BLE001 - a failed ping never affects the game
            self.stats["errors"] += 1
            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self.gate.release_background(reservation)
            seconds = self.now() - started
            self.stats["ping_seconds"] += seconds
            record["seconds"] = round(seconds, 3)
            self.inflight.pop(game.game_id, None)
            if self.log:
                self.log(record)

    async def run(self, interval=1.0):
        while True:
            self.tick()
            await asyncio.sleep(interval)

    async def close(self):
        tasks = list(self.inflight.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def summary(self):
        return {"settings": dict(self.spec), **{k: round(v, 3) if isinstance(v, float) else v
                                                 for k, v in self.stats.items()}}
