"""Parallel samples: per-game latency from spare batch capacity (hero-v2 ``parallel_samples``, 30 September;
docs/validation-20260927/parallel-samples-plans.md).

The score is carried by the few games that go deep, and from level 4 on each level costs 30-120K generated tokens at
~33 tok/s per stream: the bottleneck is the time per level of one game, not the aggregate throughput. Batch capacity is
cheap (~0.32-0.49 ms of engine step per extra stream) and about half of the KV pool sits unused. For a game with at
least ``min_levels`` levels (an in-run signal, no per-game prior), every gameplay request goes out as ``k`` samples of
the same prompt at the run's temperature; the first sample that finishes with a tool call wins and the others are
aborted at once (client disconnect: vLLM aborts the request and frees its slot and blocks). If every sample finishes
without a tool call, the first finished one is used and the usual cap / no-action handling applies. The trajectory is
exactly the winner's reply (append-only, verbatim reasoning), so compaction, the inventory and the receipts behave as
with one sample, and the game's token budget is charged the winner's tokens only (``record_reply`` sees the winner
alone); the other samples' tokens are logged apart.

Prefix sharing (optional, prefix_warmup=True). Before the samples, one ``max_tokens=1`` request with the exact payload (the ping) computes the prompt,
so every sample is admitted on a finished prefix: all of the prompt's full attention blocks are exact hits (the same
hit a continuation request gets with prefix cache v2), and each sample recomputes only its own partial tail block. No
sample depends on blocks another sample is still computing.

By default there is no warm-up, no late admission and no measurement-only completion of losers.
Each sibling reserves its entire request footprint, including the prompt; cached sharing is an engine optimization,
not an admission assumption. This conservative policy may admit fewer siblings, but avoids the serial prefix ping.

KV safety (admission accounting; GPU validation still required). The primary sample runs under the game's ordinary gate reservation, cost(prompt + cap)
blocks, held until the whole group has ended. A sibling is admitted only opportunistically through the exact gate
(``KVGate.try_acquire``): nothing waiting, a generation slot left free, and one full-window request still fitting beside
it. With prefix_warmup=True it reserves its own blocks exactly: cost(prompt + cap) less the prompt's full attention blocks, which are shared
with the primary (prefix hits, reserved once by the primary's reservation). Its recurrent-state blocks (9 per running
request) and its reply blocks are its own. The sum of reservations therefore stays within 0.9 x the served blocks, as
for every request. A regular request that has to wait at the gate makes the running siblings yield (they are aborted),
so siblings never delay a game's own request for longer than the abort grace. An aborted request's reservation is
released ``abort_grace_seconds`` after the disconnect (the engine processes an abort within a step or two); an aborted
primary keeps the game's reservation for the same grace before the winner's reply returns.

Measurement. Per group: the samples launched (late ones included), the winner's index, tokens and latency, each
sample's time, the aborted samples' tokens (estimated at the winner's decode rate: an aborted request returns no usage;
the engine's generation counter gives the exact total), the ping's time and cached tokens and a sibling winner's cached
tokens. Every ``observe_every``-th group of a game lets the losing samples finish (their replies are discarded; the
trajectory is still the winner's), which measures the time saved against the median and the slowest sample on
uncensored groups.
"""

import asyncio
import statistics
import time

DEFAULTS = {
    "k": 4,
    "min_levels": 2,
    "max_num_seqs": 48,
    "abort_grace_seconds": 0.5,
    "late_admission_seconds": 0.0,
    "retry_seconds": 0.25,
    "observe_every": 0,
    "prefix_warmup": False,
}
MAX_K = 8
SEQ_LIMIT = 48  # arc_infra.serving: generation slots 25..48
FULL_WINDOW_OVERHEAD = 256  # arena.capacity.FRESH_OVERHEAD_TOKENS: a full-window request is window + 256 tokens


def settings(raw):
    """``None`` / ``False`` (off), ``True`` (the defaults) or a dict overriding them; validated."""
    if raw is None or raw is False:
        return None
    if raw is True:
        merged = dict(DEFAULTS)
    elif isinstance(raw, dict):
        merged = {**DEFAULTS, **raw}
    else:
        raise ValueError(f"parallel_samples is true or a dict with keys {sorted(DEFAULTS)}")
    if set(merged) != set(DEFAULTS):
        raise ValueError(f"Unknown parallel_samples settings: {sorted(set(merged) - set(DEFAULTS))}")
    if type(merged['prefix_warmup']) is not bool:
        raise ValueError('parallel_samples.prefix_warmup must be a boolean')
    for key in ("k", "min_levels", "max_num_seqs", "observe_every"):
        if type(merged[key]) is not int:
            raise ValueError(f"parallel_samples.{key} is an integer")
    if not 2 <= merged["k"] <= MAX_K:
        raise ValueError(f"parallel_samples.k is 2..{MAX_K} samples per request")
    if not 0 <= merged["min_levels"] <= 20:
        raise ValueError("parallel_samples.min_levels is 0..20 levels")
    if not 26 <= merged["max_num_seqs"] <= SEQ_LIMIT:
        raise ValueError(f"parallel_samples.max_num_seqs is 26..{SEQ_LIMIT}")
    if merged["observe_every"] != 0 and merged["observe_every"] < 2:
        raise ValueError("parallel_samples.observe_every is 0 (never) or every n >= 2 groups of a game")
    for key, low, high in (("abort_grace_seconds", 0.01, 5.0), ("late_admission_seconds", 0.0, 600.0),
                           ("retry_seconds", 0.01, 5.0)):
        value = merged[key]
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f"parallel_samples.{key} is {low}..{high} seconds")
        merged[key] = float(value)
    return merged


def graph_capture_size(max_num_seqs):
    """The decode CUDA-graph capture size covering ``max_num_seqs`` (the pinned sizes step by 8 above 8)."""
    return -(-int(max_num_seqs) // 8) * 8


def attention_block_tokens(budget):
    """The full-attention group's block size in the served layout (0 without one)."""
    layout = getattr(budget, "layout", None) or {}
    for group in layout.get("groups") or ():
        if group.get("type") == "FullAttentionSpec":
            return int(group.get("block_size") or 0)
    return 0


def shared_blocks(budget, prompt_tokens):
    """The prompt's full attention blocks: prefix-cache hits a sibling shares with the primary (0 without a layout)."""
    layout = getattr(budget, "layout", None) or {}
    shared = 0
    for group in layout.get("groups") or ():
        if group.get("type") == "FullAttentionSpec" and int(group.get("block_size") or 0) > 0:
            shared += int(prompt_tokens) // int(group["block_size"])
    return shared


def sibling_units(budget, prompt_tokens, cap):
    """Blocks one sibling sample reserves: cost(prompt + cap) less the shared full prompt blocks (at least 1)."""
    total = budget.cost(int(prompt_tokens) + int(cap))
    return max(1, total - shared_blocks(budget, prompt_tokens))


def has_tool_call(module, message):
    """True when the reply calls a tool: parsed ``tool_calls`` or tool-call markup the public loop recovers."""
    if not isinstance(message, dict):
        return False
    if message.get("tool_calls"):
        return True
    reasoning = module._extract_reasoning_text(message)
    content = module._normalize_message_content(message.get("content", ""))
    if not module._contains_tool_call_markup(reasoning, content):
        return False
    return bool(module._recover_tool_calls_from_markup(reasoning, content))


class Sample:
    """One sample of a group: its task, gate reservation (siblings, observe holds) and outcome."""

    __slots__ = ("index", "task", "reservation", "released", "started", "ended", "status", "response", "exception",
                 "error", "late", "completion_tokens", "cached_tokens", "tool_call", "finish_reason",
                 "estimated_tokens", "yielded")

    def __init__(self, index, reservation, started, late=False):
        self.index, self.reservation, self.started, self.late = index, reservation, started, late
        self.task = self.ended = self.response = self.exception = self.error = None
        self.released = False
        self.status = "running"  # running, finished (2xx or not), error, aborted, won
        self.completion_tokens = self.cached_tokens = self.finish_reason = self.estimated_tokens = None
        self.tool_call = False
        self.yielded = False

    def row(self, origin):
        row = {"i": self.index, "status": self.status, "start": round(self.started - origin, 3),
               "end": None if self.ended is None else round(self.ended - origin, 3)}
        if self.late:
            row["late"] = True
        if self.yielded:
            row["yielded"] = True
        for key in ("completion_tokens", "cached_tokens", "finish_reason", "estimated_tokens", "error"):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        if self.response is not None and not self.response.is_success:
            row["http_status"] = self.response.status_code
        return row


class ParallelSampler:
    """One game's parallel sampling (see the module docstring); ``complete`` replaces one gameplay request."""

    def __init__(self, agent, spec, *, now=time.monotonic):
        self.agent, self.spec, self.now = agent, spec, now
        self.groups = 0
        self.hooked = None  # the gate this sampler's yield hook is registered on
        self.live = set()  # samples holding a sibling or observe reservation that are still running
        self.background = set()  # observed groups still settling after their group returned
        self.stats = {"requests_eligible": 0, "groups": 0, "groups_without_room": 0, "samples_launched": 0,
                      "late_samples": 0, "winner_primary": 0, "winner_sibling": 0, "winner_without_tool_call": 0,
                      "primary_aborted": 0, "siblings_aborted": 0, "siblings_yielded": 0, "observed_groups": 0,
                      "observe_holds": 0, "winner_completion_tokens": 0, "aborted_tokens_estimated": 0,
                      "finished_loser_tokens": 0, "pings": 0, "ping_failures": 0, "ping_seconds": 0.0,
                      "ping_prompt_tokens": 0, "ping_cached_tokens": 0, "sibling_winner_cached_tokens": 0,
                      "sibling_winner_prompt_tokens": 0, "sibling_winner_full_hits": 0,
                      "sibling_winners_with_cache_details": 0, "grace_seconds": 0.0, "grace_holds": 0,
                      "winner_seconds": 0.0,
                      "sample_errors": 0}
        self.saved = {"median": [], "slowest": []}

    # --- eligibility and gate plumbing --------------------------------------------------------------------------

    def eligible(self):
        game = self.agent.game
        return game is not None and int(game.levels) >= int(self.spec["min_levels"])

    @property
    def gate(self):
        return self.agent.client.slots.gate

    def _hook(self):
        gate = self.gate
        hooks = getattr(gate, "queue_hooks", None)
        if hooks is not None and self.hooked is not gate:
            hooks.append(self.yield_to_queue)
            self.hooked = gate

    def _headroom_tokens(self):
        return int(self.agent.window) + FULL_WINDOW_OVERHEAD

    def _try(self, prompt, *, units, event):
        gate = self.gate
        if not hasattr(gate, "try_acquire"):
            return None
        return gate.try_acquire(self.agent.game.game_id, int(prompt) + int(self.agent.cap),
                                headroom_tokens=self._headroom_tokens(), units=units, event=event)

    def _hold(self, units, event):
        """A block hold of ``units`` (no request, so no sequence slot) that stands in for a reservation being released
        (no headroom rule: it replaces the game's reservation, which is at least as large), or None."""
        gate = self.gate
        if not hasattr(gate, "try_acquire"):
            return None
        return gate.try_acquire(self.agent.game.game_id, 0, headroom_tokens=0, units=units, event=event, slot=False)

    def _try_sibling(self, prompt):
        """A sibling's reservation from the gate's spare capacity (its own blocks only), or None."""
        budget = self.gate.budget
        # Without a completed warm-up the new prompt tail may still be computing in
        # another request. Never subtract speculative shared blocks from admission.
        units = (sibling_units(budget, prompt, self.agent.cap)
                 if budget is not None and self.spec['prefix_warmup'] else None)
        return self._try(prompt, units=units, event="sibling_admit")

    def _release(self, sample, delay):
        """Release a sample's own reservation now (its request ended) or ``delay`` s after its abort (once)."""
        if sample.reservation is None or sample.released:
            return
        sample.released = True
        self.live.discard(sample)
        gate, reservation = self.gate, sample.reservation
        if delay <= 0:
            gate.release_background(reservation, event="sibling_release")
        else:
            asyncio.get_running_loop().call_later(delay, gate.release_background, reservation, "sibling_release")

    def yield_to_queue(self, deficit_units, need_slot):
        """Gate hook (a regular request waits): abort this game's running siblings and observed losers, newest first,
        until the freed units cover ``deficit_units`` (and a slot, when ``need_slot``). Returns the ``(units, slots)``
        freed or pending release."""
        freed_units = freed_slots = 0
        for sample in sorted(self.live, key=lambda s: -s.started):
            if sample.status == "running":
                sample.yielded = True
                self.stats["siblings_yielded"] += 1
                self._abort(sample)
            freed_units += sample.reservation.tokens
            freed_slots += 1
            if freed_units >= deficit_units and (freed_slots or not need_slot):
                break
        return freed_units, freed_slots

    # --- one request -----------------------------------------------------------------------------------------------

    async def complete(self, payload, base):
        """The gameplay response for ``payload``; ``base(payload, **record)`` posts one request."""
        agent, spec, stats = self.agent, self.spec, self.stats
        stats["requests_eligible"] += 1
        self._hook()
        prompt = int(getattr(agent.client.slots, "next_estimate", 0) or 0)
        # Reserve the siblings before anything is sent (nothing extra is sent when none fits).
        siblings = []
        for _ in range(spec["k"] - 1):
            reservation = self._try_sibling(prompt)
            if reservation is None:
                break
            siblings.append(reservation)
        if not siblings:
            stats["groups_without_room"] += 1
            return await base(payload)
        self.groups += 1
        group = {"id": self.groups, "samples": [], "origin": self.now(), "prompt": prompt, "ping": None,
                 "observe": bool(spec["observe_every"] and self.groups % spec["observe_every"] == 0)}
        if spec['prefix_warmup']:
            # 1. The ping: the exact prompt with max_tokens=1, so every sample starts on a finished prefix.
            started = self.now()
            try:
                ping = await base({**payload, "max_tokens": 1}, **self._record(group, "ping", None))
            except BaseException:
                for reservation in siblings:
                    self.gate.release_background(reservation, event="sibling_release")
                raise
            seconds = self.now() - started
            stats["ping_seconds"] += seconds
            if not ping.is_success:
                # The primary alone: a context rejection or a server error then follows its ordinary path.
                stats["ping_failures"] += 1
                for reservation in siblings:
                    self.gate.release_background(reservation, event="sibling_release")
                self._log(group, {"ping_status": ping.status_code, "k": 1})
                return await base(payload)
            usage = ping.json().get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            group["ping"] = {"seconds": round(seconds, 3), "prompt_tokens": usage.get("prompt_tokens"),
                             "cached_tokens": cached}
            stats["pings"] += 1
            stats["ping_prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            stats["ping_cached_tokens"] += int(cached or 0)
        # 2. The primary under the game's reservation, the reserved siblings beside it.
        try:
            self._launch(group, None, payload, base)
            for reservation in siblings:
                self._launch(group, reservation, payload, base)
            winner = await self._race(group, payload, base)
        except BaseException:
            # Every sample failed (the primary's exception), or the game was cancelled (the wall end): abort what
            # runs, release what ended, and cover the aborts' blocks for the grace when a hold fits (no wait here).
            for sample in group["samples"]:
                if sample.status == "running":
                    self._abort(sample)
                else:
                    self._release(sample, 0)
            budget = self.gate.budget
            self._cover_aborts(group, shared_blocks(budget, group["prompt"])
                               if budget is not None and spec['prefix_warmup'] else 0)
            self._log(group, {"k": len(group["samples"]), "winner": None, "ended_by": "exception",
                              "samples": [s.row(group["origin"]) for s in group["samples"]]})
            raise
        return await self._finish(group, winner)

    def _record(self, group, role, index):
        if not getattr(self.agent.client.client, "records_extra", False):
            return {}
        extra = {"parallel_group": group["id"], "parallel_role": role}
        if index is not None:
            extra["parallel_sample"] = index
        return {"record_extra": extra}

    def _launch(self, group, reservation, payload, base, late=False):
        sample = Sample(len(group["samples"]), reservation, self.now(), late=late)
        role = "primary" if sample.index == 0 else "sibling"
        sample.task = asyncio.create_task(self._run(sample, payload, base, self._record(group, role, sample.index)))
        group["samples"].append(sample)
        if reservation is not None:
            self.live.add(sample)
        self.stats["samples_launched"] += 1
        self.stats["late_samples"] += late
        return sample

    async def _run(self, sample, payload, base, record):
        try:
            response = await base(dict(payload), **record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed sample never decides the group alone
            sample.exception, sample.error = exc, f"{type(exc).__name__}: {exc}"[:300]
            sample.status, sample.ended = "error", self.now()
            self.stats["sample_errors"] += 1
            self.agent.workspace.append("parallel_sample_error", sample=sample.index, error=sample.error,
                                 **record.get("record_extra", {}))
            return None
        sample.ended, sample.response, sample.status = self.now(), response, "finished"
        if response.is_success:
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            sample.completion_tokens = int(usage.get("completion_tokens") or 0)
            sample.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            sample.finish_reason = choice.get("finish_reason")
            sample.tool_call = has_tool_call(self.agent.module, choice.get("message"))
        return response

    async def _race(self, group, payload, base):
        """The winning sample: the first 2xx reply with a tool call; when every sample has ended without one, the
        first 2xx reply, else the primary's own outcome (its non-2xx response, or its exception re-raised)."""
        spec, samples = self.spec, group["samples"]
        launched = self.now()
        while True:
            running = [s.task for s in samples if s.status == "running"]
            if running:
                await asyncio.wait(running, timeout=spec["retry_seconds"], return_when=asyncio.FIRST_COMPLETED)
            ended = sorted((s for s in samples if s.status in ("finished", "error")), key=lambda s: s.ended)
            for sample in ended:
                if sample.status == "finished" and sample.response.is_success and sample.tool_call:
                    return sample
            if not any(s.status == "running" for s in samples):
                ok = [s for s in ended if s.status == "finished" and s.response.is_success]
                if ok:
                    return ok[0]
                primary = samples[0]
                if primary.status == "finished":
                    return primary
                if primary.exception is not None:
                    raise primary.exception
                raise RuntimeError("parallel samples: every sample ended without a reply")
            # Late siblings: only while no sample has ended, nothing yielded and the group is young.
            if (not ended and len(samples) < spec["k"] and not any(s.yielded for s in samples)
                    and spec["late_admission_seconds"] > 0
                    and self.now() - launched <= spec["late_admission_seconds"]):
                while len(samples) < spec["k"]:
                    reservation = self._try_sibling(group["prompt"])
                    if reservation is None:
                        break
                    self._launch(group, reservation, payload, base, late=True)

    def _abort(self, sample):
        """Cancel a running sample (client disconnect); its own reservation is released after the grace."""
        if sample.status != "running":
            return
        sample.task.cancel()
        sample.status, sample.ended = "aborted", self.now()
        self._release(sample, self.spec["abort_grace_seconds"])

    async def _finish(self, group, winner):
        stats, samples = self.stats, group["samples"]
        primary = samples[0]
        kept = []
        for sample in samples:
            if sample is winner:
                continue
            if sample.status == "running":
                if group["observe"] and self._keep_running(group, sample):
                    kept.append(sample)
                    continue
                self._abort(sample)
                stats["primary_aborted" if sample is primary else "siblings_aborted"] += 1
            else:
                self._release(sample, 0)  # its request ended in the engine
        self._release(winner, 0)
        winner.status = "won"
        # The game's reservation (the primary's: the shared prompt blocks + the primary's own) is released when the
        # reply returns. Whatever still holds shared blocks in the engine after that must stay covered: observed
        # siblings running without the primary (a hold of the shared blocks for their lifetime, else they are
        # aborted), and aborted samples until the engine has processed their abort (a hold of the primary's full cost
        # or of the shared blocks for the grace, else the reply waits for the grace on the game's reservation).
        shared = (shared_blocks(self.gate.budget, group["prompt"])
                  if self.gate.budget is not None and self.spec["prefix_warmup"] else 0)
        if kept and all(s.index != 0 for s in kept) and shared:
            hold = self._hold(shared, "observe_hold")
            if hold is None:
                for sample in kept:
                    self._abort(sample)
                    stats["siblings_aborted"] += 1
                kept = []
            else:
                group["shared_hold"] = hold
                stats["observe_holds"] += 1
        grace = self._cover_aborts(group, shared)
        rate = None
        if winner.completion_tokens and winner.ended > winner.started:
            rate = winner.completion_tokens / (winner.ended - winner.started)
        for sample in samples:
            if sample.status == "aborted" and rate is not None:
                sample.estimated_tokens = int(min(self.agent.cap, rate * max(0.0, sample.ended - sample.started)))
                stats["aborted_tokens_estimated"] += sample.estimated_tokens
            elif sample.status == "finished" and sample.completion_tokens:
                stats["finished_loser_tokens"] += sample.completion_tokens
        stats["groups"] += 1
        stats["winner_primary" if winner is primary else "winner_sibling"] += 1
        stats["winner_without_tool_call"] += not winner.tool_call
        stats["winner_completion_tokens"] += int(winner.completion_tokens or 0)
        stats["winner_seconds"] += winner.ended - winner.started
        if winner is not primary and winner.cached_tokens is not None:
            # Prefix sharing: a sibling that starts after the ping hits every full block of the prompt.
            block = attention_block_tokens(self.gate.budget)
            prompt = int((group["ping"] or {}).get("prompt_tokens") or group["prompt"])
            stats["sibling_winners_with_cache_details"] += 1
            stats["sibling_winner_cached_tokens"] += int(winner.cached_tokens)
            stats["sibling_winner_prompt_tokens"] += prompt
            if block and int(winner.cached_tokens) >= prompt // block * block:
                stats["sibling_winner_full_hits"] += 1
        record = {"k": len(samples), "winner": winner.index, "winner_tool_call": winner.tool_call,
                  "winner_seconds": round(winner.ended - winner.started, 3),
                  "winner_completion_tokens": winner.completion_tokens, "observe": group["observe"],
                  "samples": [s.row(group["origin"]) for s in samples]}
        if group["observe"]:
            self._observe(group, winner)
        self._log(group, record)
        if grace:
            stats["grace_seconds"] += grace
            await asyncio.sleep(grace)
        return winner.response

    def _cover_aborts(self, group, shared):
        """Aborted samples keep their blocks until the engine processes the abort, but the game's reservation goes when
        the group returns: hold the aborted primary's full cost, or the shared prompt blocks for aborted siblings, for
        the grace. Returns the seconds the caller must wait on the game's reservation instead (no room for a hold)."""
        spec, samples = self.spec, group["samples"]
        aborted = [s for s in samples if s.status == "aborted"]
        if not aborted:
            return 0.0
        if self.gate.budget is None:
            return spec["abort_grace_seconds"]  # a token gate: no block costs to hold, so wait on the reservation
        cover = shared
        if samples[0].status == "aborted":
            cover = self.gate.budget.cost(group["prompt"] + int(self.agent.cap))
        if not cover:
            return 0.0
        hold = self._hold(cover, "grace_hold")
        if hold is None:
            return spec["abort_grace_seconds"]
        asyncio.get_running_loop().call_later(spec["abort_grace_seconds"], self.gate.release_background, hold,
                                              "grace_release")
        self.stats["grace_holds"] += 1
        return 0.0

    def _keep_running(self, group, sample):
        """Observe mode: a losing sample keeps running only while its blocks stay reserved after the group returns. A
        sibling keeps its own reservation; the primary runs on the game's, which is released at the return, so it needs
        a background hold of its full cost, else it is aborted (a censored observation)."""
        if sample.index != 0:
            return True
        reservation = self._try(group["prompt"], units=None, event="observe_hold")
        if reservation is None:
            return False
        sample.reservation = reservation
        self.live.add(sample)
        self.stats["observe_holds"] += 1
        return True

    def _observe(self, group, winner):
        """Observe mode: the losers still running finish in the background; their times complete the record."""
        running = [s for s in group["samples"] if s.status == "running"]
        self.stats["observed_groups"] += 1
        won = winner.ended - winner.started

        async def settle():
            await asyncio.gather(*(s.task for s in running), return_exceptions=True)
            for sample in running:
                if sample.status in ("finished", "error"):
                    self._release(sample, 0)
                if sample.status == "finished" and sample.completion_tokens:
                    self.stats["finished_loser_tokens"] += sample.completion_tokens
            hold = group.pop("shared_hold", None)
            if hold is not None:
                # The shared prompt blocks stay held until the last observed sample has left the engine.
                delay = self.spec["abort_grace_seconds"] if any(s.status == "aborted" for s in running) else 0
                if delay:
                    asyncio.get_running_loop().call_later(delay, self.gate.release_background, hold, "observe_release")
                else:
                    self.gate.release_background(hold, event="observe_release")
            ended = [s for s in group["samples"] if s.status in ("won", "finished")]
            censored = any(s.status in ("aborted", "error") for s in group["samples"])
            row = {"censored": censored, "durations": [round(s.ended - s.started, 3) for s in ended],
                   "tokens": [s.completion_tokens for s in ended], "tool_calls": [s.tool_call for s in ended]}
            if not censored and len(ended) >= 2:
                durations = [s.ended - s.started for s in ended]
                for key, value in (("median", statistics.median(durations) - won), ("slowest", max(durations) - won)):
                    self.saved[key].append(value)
                    row[f"saved_vs_{key}_seconds"] = round(value, 3)
            self.agent.workspace.append("parallel_observed", step=self.agent.steps, group=group["id"], **row)

        task = asyncio.create_task(settle())
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    def _log(self, group, record):
        self.agent.workspace.append("parallel_samples", step=self.agent.steps, group=group["id"],
                                    prompt_tokens=group["prompt"], ping=group["ping"], **record)

    async def close(self):
        """The game ended: abort observed losers still running (their reservations follow the grace)."""
        for sample in list(self.live):
            self._abort(sample)
        tasks = list(self.background)
        await asyncio.gather(*tasks, return_exceptions=True)

    def summary(self):
        stats = dict(self.stats)
        for key in ("ping_seconds", "grace_seconds", "winner_seconds"):
            stats[key] = round(stats[key], 3)
        for key, values in self.saved.items():
            stats[f"saved_vs_{key}"] = {"groups": len(values),
                                        "mean_seconds": round(sum(values) / len(values), 3) if values else None,
                                        "median_seconds": round(statistics.median(values), 3) if values else None}
        return {"settings": dict(self.spec), **stats}


def aggregate(summaries):
    """The trial's totals over the games' ``summary()`` rows (progress.json and arena.json)."""
    rows = [s for s in summaries if s]
    if not rows:
        return None
    total = {}
    for row in rows:
        for key, value in row.items():
            if key != "settings" and not isinstance(value, dict):
                total[key] = total.get(key, 0) + value
    for key in ("saved_vs_median", "saved_vs_slowest"):
        groups = sum(r[key]["groups"] for r in rows)
        weighted = sum((r[key]["mean_seconds"] or 0) * r[key]["groups"] for r in rows)
        total[key] = {"groups": groups, "mean_seconds": round(weighted / groups, 3) if groups else None}
    total["games_with_groups"] = sum(1 for r in rows if r.get("groups"))
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in total.items()}
