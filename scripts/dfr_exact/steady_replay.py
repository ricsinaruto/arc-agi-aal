"""Steady-state replay bench: the late-run serving load of a c100 arena run in minutes.

A 90-minute arena run reaches its late steady state (contexts 40-115K, KV-gated in-flight, drains and handovers
re-prefilling ~65K-token contexts, ~2 computed prompt tokens per generated token) only after an hour. This bench
starts there: recorded game histories (scripts/steady_replay_inputs.py) at late-run context lengths, sent the way the
Duck harness sends them, through the arena's own KV gate (prompt + reply cap per request) and its cache-budget rule
for the active set (max(GPU, host) blocks x fraction, each active context projected a headroom ahead).

Per active game: request (model generates freely: real reasoning, real reply lengths, real MTP acceptance) ->
append the reply and the game's next recorded tool result -> the recorded late-window tool gap -> repeat. Past the
harness drain trigger (window - reply cap) the history drains to half with a note (the prefix changes, so the kept
context re-prefills, as a real drain) and the game hands over to the waiting game with the fewest generated tokens
(the equal scheduler's swap point). Over the cache budget, the idle active game with the most generated tokens parks
(arena cache trim). Score is not measured; steady-state generated tokens/s is.

Report: engine generated tokens/s in bins, the steady window's mean (after ``warm_seconds``), running / waiting
requests, KV use, computed vs cached prompt tokens, acceptance, drains, handovers, active games.
"""

import asyncio
import json
import random
import time
from pathlib import Path

GEN = "vllm:generation_tokens_total"
COMPUTED = "vllm:prompt_tokens_by_source_total:local_compute"
CACHED = "vllm:prompt_tokens_by_source_total:local_cache_hit"
DRAFTS, ACCEPTED = "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"
RUNNING, WAITING, KV = "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"
PREEMPT = "vllm:num_preemptions_total"
NOTE = ("[Context note {n}] Earlier turns of this game were condensed to save context. Key facts so far are kept "
        "in the interpreter state; continue from the latest observation.")


def parse_arguments(text):
    try:
        value = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        return text
    return json.dumps(value) if isinstance(value, dict) else text


class Game:
    def __init__(self, index, record, rng, start_tokens):
        self.index, self.id = index, record["game"]
        self.tools = record["tools"]
        self.all = record["messages"]
        cuts, tokens = record["cuts"], record["cut_tokens"]
        target = rng.uniform(*start_tokens)
        k = max([c for c, t in zip(cuts, tokens) if t <= target] or [cuts[0]])
        self.messages = [dict(m) for m in self.all[:k]]
        self.estimate = dict(zip(cuts, tokens))[k]
        self.chars_per_token = float(record["chars_per_token"])
        # A history shorter than its start target is extended with its own turns again (from the first assistant
        # turn on, cyclically): the late-run sawtooth needs contexts over the whole drain range from the start.
        body = self.all[2:cuts[-1]]
        i = 0
        while self.estimate < target and body:
            m = dict(body[i % len(body)])
            i += 1
            if not (self.messages[-1]["role"] in ("tool", "user") and m["role"] == "tool"):
                self.messages.append(m)
                self.estimate += len(json.dumps(m)) / self.chars_per_token
            if i > 10 * len(body):
                break
        while self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        self.estimate = int(self.estimate)
        self.extended = i
        later = [m for m in self.all[k:] if m["role"] in ("tool", "user")]
        every = [m for m in self.all if m["role"] in ("tool", "user")][1:]
        self.feed = later + every
        self.feed_at = 0
        self.generated = 0
        self.drains = 0
        self.busy = False

    def next_feed(self):
        m = self.feed[self.feed_at % len(self.feed)]
        self.feed_at += 1
        return m["content"]

    def drain(self, keep_tokens):
        """Half-window drain: system + first user turn + a note + the newest turns that fit ``keep_tokens``."""
        head = [m for m in self.messages[:2]]
        tail, used = [], 0
        for m in reversed(self.messages[2:]):
            size = len(json.dumps(m)) / self.chars_per_token
            if used + size > keep_tokens and tail and tail[-1]["role"] in ("tool", "user"):
                break
            tail.append(m)
            used += size
        tail.reverse()
        while tail and tail[0]["role"] == "tool":  # a tool result needs its call
            tail.pop(0)
        self.drains += 1
        note = {"role": "user", "content": NOTE.format(n=self.drains)}
        self.messages = head + [note] + tail
        if self.messages[-1]["role"] == "assistant":
            self.messages.append({"role": "user", "content": self.next_feed()})
        self.estimate = int(sum(len(json.dumps(m)) for m in self.messages) / self.chars_per_token)


async def steady_replay(bench, client, gate, budget, monitor, spec, *, save=lambda: None):
    """One phase per entry of ``spec["phases"]`` (bench-level overrides, e.g. the reply reserve), each from a fresh
    start on the same server after a cache reset; without phases, one phase with ``spec`` as is."""
    phases = spec.get("phases")
    if not phases:
        return await _phase(bench, client, gate, budget, monitor, spec, save=save)
    from .isolation import isolate_trial

    report = {"phases": []}
    for i, overrides in enumerate(phases):
        if i:
            report["phases"][-1]["isolation"] = await isolate_trial(bench.base_url)
        one = {**{k: v for k, v in spec.items() if k != "phases"}, **overrides}
        result = await _phase(bench, client, gate, budget, monitor, one, save=save)
        result["label"] = overrides.get("label", str(i))
        report["phases"].append(result)
        save()
    return report


async def _phase(bench, client, gate, budget, monitor, spec, *, save=lambda: None):
    data = json.loads((bench.source / spec["contexts_file"]).read_text())
    if data.get("format") != "arc-steady-replay-v1":
        raise ValueError("steady replay: arc-steady-replay-v1 contexts required")
    rng = random.Random(int(spec.get("seed", 0)))
    study = bench.study
    window, cap = int(study["windows"][0]), int(study["completion_cap"])
    trigger, keep = window - cap, (window - cap) // 2
    # KV-gate reply reserve per request (the arena reserves the full reply cap; a lever: the engine preempts if short).
    reserve = int(spec.get("reserve_tokens") or cap)
    games = [Game(i, r, rng, spec["start_tokens"]) for i, r in enumerate(data["games"][:int(spec.get("games", 100))])]
    rng.shuffle(games)
    gaps = data["gaps"] or [1.0]
    completions = data.get("completions") if spec.get("forced", True) else None
    # Cache budget (arena.trial): units are gate blocks of the served layout.
    cb = study["cache_budget"]
    per_block = float(budget.layout["planned_cache_bytes"]) / int(budget.layout["num_blocks"])
    gib = float((study.get("cache_experiment") or {}).get("cpu_cache_gib") or 0)
    host = int(gib * 2**30 / per_block * float(cb["host_margin"]))
    budget_blocks = int(max(int(budget.capacity), host) * float(cb["fraction"]))
    if spec.get("cache_budget") is False:
        # v8.2's admission: a fixed focus set (active_limit), the exact KV gate alone limits what is in flight.
        budget_blocks = 10**9
    headroom, min_slots = int(cb["headroom_tokens"]), int(cb["min_slots"])
    active_limit = int(spec.get("active_limit", study.get("active_limit") or 64))

    def projected(g):
        return budget.cost(min(window, g.estimate + headroom))

    def room(extra=None, without=None):
        act = [g for g in active if g is not without]
        if len(act) < min_slots:
            return True
        return sum(projected(g) for g in act) + (projected(extra) if extra else 0) <= budget_blocks

    active, waiting = [], list(games)
    records, events = [], []
    started = time.monotonic()
    deadline = started + float(spec["warm_seconds"]) + float(spec["measure_seconds"])
    body_common = {"model": bench.model, "temperature": bench.temperature, "top_p": float(spec.get("top_p", 0.95)),
                   "top_k": int(spec.get("top_k", 20)), "max_tokens": cap, "tool_choice": "auto",
                   "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": bench.effort}}
    from arc_infra.model_templates import template_kwargs

    body_common["chat_template_kwargs"] = template_kwargs(bench.model, body_common["chat_template_kwargs"])
    tasks = {}

    def note(kind, g, **extra):
        events.append({"t": round(time.monotonic() - started, 2), "event": kind, "game": g.id, **extra})

    def admit_fill():
        waiting.sort(key=lambda g: (g.generated, g.index))
        while waiting and len(active) < active_limit and room(extra=waiting[0]):
            g = waiting.pop(0)
            active.append(g)
            note("admit", g, est=g.estimate)
            if g.id not in tasks or tasks[g.id].done():  # a parked game's old loop may still be in its gap
                tasks[g.id] = asyncio.create_task(loop(g))

    def park(g, reason):
        if g in active:
            active.remove(g)
            waiting.append(g)
            note("park", g, reason=reason)

    def trim():
        # arena.scheduler._cache_trim: resident load (no headroom) over the budget parks the busiest idle game.
        while len(active) > min_slots and sum(budget.cost(g.estimate) for g in active) > budget_blocks:
            idle = [g for g in active if not g.busy]
            if not idle:
                return
            park(max(idle, key=lambda g: (g.generated, g.index)), "cache_trim")

    async def loop(g):
        try:
            while time.monotonic() < deadline and g in active:
                body = {**body_common, "messages": g.messages, "tools": g.tools}
                if completions:
                    # The recorded late-window reply length, generated in full (the free model's replies shrink on
                    # replayed tool results that do not answer its actions).
                    # min_tokens (EOS held back, no post-EOS junk) keeps the reply re-renderable, so the next turn
                    # hits the cache on it as in the arena; v2's ignore_eos replies missed ~1.5K tokens a turn.
                    n = int(rng.choice(completions) * float(spec.get("length_scale", 1.0)))
                    body.update(min_tokens=max(1, min(n, cap - 1)))
                g.busy = True
                reservation = await gate.acquire(g.id, g.estimate + reserve)
                sent = time.monotonic()
                try:
                    response = await client.post(bench.base_url + "/chat/completions", json=body,
                                                 timeout=max(30.0, deadline - sent + 60))
                finally:
                    gate.release(reservation)
                    g.busy = False
                if response.status_code != 200:
                    records.append({"game": g.id, "status": response.status_code, "text": response.text[:300]})
                    await asyncio.sleep(5)
                    continue
                reply = response.json()
                usage, choice = reply["usage"], reply["choices"][0]
                message = choice["message"]
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                records.append({"game": g.id, "t0": round(sent - started, 2), "t1": round(time.monotonic() - started, 2),
                                "prompt": usage["prompt_tokens"], "cached": cached,
                                "completion": usage["completion_tokens"], "finish": choice.get("finish_reason")})
                g.generated += usage["completion_tokens"]
                assistant = {"role": "assistant", "content": message.get("content") or ""}
                reasoning = message.get("reasoning") or message.get("reasoning_content")
                if reasoning:
                    assistant["reasoning"] = reasoning
                calls = message.get("tool_calls") or []
                if calls:
                    assistant["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function",
                                                "function": {"name": c["function"]["name"],
                                                             "arguments": parse_arguments(c["function"]["arguments"])}}
                                               for i, c in enumerate(calls)]
                g.messages.append(assistant)
                added = len(json.dumps(assistant))
                if calls:
                    for c in assistant["tool_calls"]:
                        result = {"role": "tool", "content": g.next_feed(), "tool_call_id": c["id"]}
                        g.messages.append(result)
                        added += len(result["content"])
                else:
                    result = {"role": "user", "content": g.next_feed()}
                    g.messages.append(result)
                    added += len(result["content"])
                g.estimate = usage["prompt_tokens"] + int(added / g.chars_per_token)
                if g.estimate > trigger:
                    before = g.estimate
                    g.drain(keep)
                    note("drain", g, before=before, after=g.estimate)
                    if waiting:
                        park(g, "drain_handover")
                        admit_fill()
                        return
                trim()
                admit_fill()
                # gap_max (calibration, 19 Oct): the recorded gi3m3l2 gaps reach 981 s (p90 3.6 s); real c100 runs keep
                # every active game in flight (DKb-300K: 42 of 42), so long idle gaps only evict prefixes.
                await asyncio.sleep(min(rng.choice(gaps), float(spec.get("gap_max") or 1e9)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one game's fault is recorded, the bench goes on
            records.append({"game": g.id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            park(g, "error")
            admit_fill()

    admit_fill()
    profiles = []

    async def profile_windows():
        """Torch-profile the live replay (19 Oct, mixed-step profile): at each offset the server's profiler starts
        and stops itself after its max_iterations (ARENA_TORCH_PROFILER_ITERS); the traces move to the output."""
        import shutil

        folder = Path(bench.server_dir) / "torch-profile"
        root = bench.base_url.removesuffix("/v1")
        for i, at in enumerate(spec.get("profile_windows") or []):
            rec = {"at": at}
            profiles.append(rec)
            try:
                await asyncio.sleep(max(0.0, started + float(at) - time.monotonic()))
                before = set(folder.rglob("*")) if folder.is_dir() else set()
                r = await client.post(root + "/start_profile")
                rec["start_status"] = r.status_code
                rec["t_start"] = round(time.monotonic() - started, 2)
                await asyncio.sleep(float(spec.get("profile_seconds", 60)))
                r = await client.post(root + "/stop_profile")
                rec["stop_status"] = r.status_code
                rec["t_stop"] = round(time.monotonic() - started, 2)
                dest = Path(bench.output) / f"profile-steady-{i}"
                dest.mkdir(parents=True, exist_ok=True)
                rec["files"] = []
                for trace in await bench.new_traces(folder, before):
                    shutil.move(str(trace), dest / trace.name)
                    rec["files"].append({"file": trace.name, "bytes": (dest / trace.name).stat().st_size})
                for table in folder.glob("profiler_out_*.txt") if folder.is_dir() else []:
                    shutil.copyfile(table, dest / table.name)
            except Exception as exc:  # noqa: BLE001 - a failed profile must not stop the replay
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            save()

    prof_task = asyncio.create_task(profile_windows()) if spec.get("profile_windows") else None
    report = {"spec": {k: v for k, v in spec.items() if k != "contexts_file"}, "games": len(games),
              "budget_blocks": budget_blocks, "gpu_blocks": int(budget.capacity), "host_blocks": host,
              "trigger_tokens": trigger, "keep_tokens": keep, "source": data["source"]}
    next_save = started + 60
    while time.monotonic() < deadline:
        await asyncio.sleep(5)
        if time.monotonic() >= next_save:
            report["progress"] = {"t": round(time.monotonic() - started), "active": len(active),
                                  "requests": len(records)}
            save()
            next_save += 60
    if prof_task is not None:
        try:
            await asyncio.wait_for(prof_task, timeout=300)
        except Exception as exc:  # noqa: BLE001
            profiles.append({"error": f"profile task: {type(exc).__name__}"})
    for t in list(tasks.values()):
        t.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)
    await monitor.sample()
    samples = [s for s in monitor.samples if s.get("metrics")]
    t_start = samples[0]["elapsed_seconds"] if samples else 0
    rows = []
    for s in samples:
        m = s["metrics"]
        rows.append((s["elapsed_seconds"], m.get(GEN, 0.0), m.get(COMPUTED, 0.0), m.get(CACHED, 0.0),
                     m.get(DRAFTS, 0.0), m.get(ACCEPTED, 0.0), m.get(RUNNING), m.get(WAITING), m.get(KV),
                     m.get(PREEMPT, 0.0)))
    bin_seconds = float(spec.get("bin_seconds", 60))
    # Monitor clock offset: the bench's started maps to the last sample at or before it.
    offset = (samples[-1]["elapsed_seconds"] - (time.monotonic() - started)) if samples else 0.0
    bins = []
    t = float(spec.get("first_bin", 0))
    end = time.monotonic() - started
    while t + bin_seconds <= end:
        a = [r for r in rows if r[0] <= t + offset]
        b = [r for r in rows if r[0] <= t + bin_seconds + offset]
        inside = [r for r in rows if t + offset < r[0] <= t + bin_seconds + offset]
        if a and b and b[-1][0] > a[-1][0]:
            dt = b[-1][0] - a[-1][0]
            drafts = b[-1][4] - a[-1][4]
            bins.append({"t": round(t), "gen_tps": round((b[-1][1] - a[-1][1]) / dt, 1),
                         "computed_tps": round((b[-1][2] - a[-1][2]) / dt, 1),
                         "cached_tps": round((b[-1][3] - a[-1][3]) / dt, 1),
                         "accept_len": round(1 + (b[-1][5] - a[-1][5]) / drafts, 3) if drafts else None,
                         "running": round(sum(r[6] or 0 for r in inside) / len(inside), 1) if inside else None,
                         "waiting": round(sum(r[7] or 0 for r in inside) / len(inside), 1) if inside else None,
                         "kv": round(sum(r[8] or 0 for r in inside) / len(inside), 3) if inside else None,
                         "preemptions": b[-1][9] - a[-1][9]})
        t += bin_seconds
    warm = float(spec["warm_seconds"])
    steady = [b for b in bins if b["t"] >= warm]
    ok = [r for r in records if "prompt" in r]
    late = [r for r in ok if r["t1"] >= warm]
    report.update(
        bins=bins, monitor_start=t_start,
        steady={"bins": len(steady),
                "gen_tps": round(sum(b["gen_tps"] for b in steady) / len(steady), 1) if steady else None,
                "computed_tps": round(sum(b["computed_tps"] for b in steady) / len(steady), 1) if steady else None,
                "running": round(sum(b["running"] or 0 for b in steady) / len(steady), 1) if steady else None,
                "preemptions": sum(b["preemptions"] for b in steady),
                "accept_len": round(sum(b["accept_len"] or 0 for b in steady) / len(steady), 3) if steady else None,
                "requests": len(late),
                "prompt_mean": round(sum(r["prompt"] for r in late) / len(late)) if late else None,
                "completion_mean": round(sum(r["completion"] for r in late) / len(late)) if late else None},
        requests=len(ok), errors=[r for r in records if "prompt" not in r][:20],
        drains=sum(e["event"] == "drain" for e in events), handovers=sum(e.get("reason") == "drain_handover"
                                                                       for e in events),
        trims=sum(e.get("reason") == "cache_trim" for e in events), active_end=len(active))
    if profiles:
        report["profiles"] = profiles
    out = Path(bench.output)
    tag = f"-{spec['label']}" if spec.get("label") else ""
    (out / f"steady-requests{tag}.json").write_text(json.dumps(records))
    (out / f"steady-events{tag}.json").write_text(json.dumps(events))
    save()
    return report
