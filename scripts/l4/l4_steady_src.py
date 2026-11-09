"""Steady-state base12 traffic on the served stack (``steady`` in the options): ``arc2.trial`` hands over here in place
of the play.

The conversations live at one moment of a recorded RTX run start at their recorded messages (``scripts/l4/
steady_snapshot.py``), so the context mix is the run's own and a short kernel need not grow it. Warm: each prompt once
with one token, through the real gate. Then every slot goes on with real replies: a reply replaces the recorded one, the
recorded tool result follows it, reopens and drains come at their recorded rows, and a finished conversation gives its
slot to the next recorded one. The gate, the transport and the served log are the play's own.
"""

import asyncio
import gzip
import json
import random
import time
from collections import deque
from pathlib import Path

from . import tasks

WARM_SHARE = 0.75  # of the wall: the steady part starts by then, warm or not
STAGGER = 20.0  # s: first steady requests spread over this, as a running play's are
ERRORS = 8  # request errors that end the test


COUNTERS = ("generation_tokens_total", "prompt_tokens_total", "iteration_tokens_total_count", "iteration_tokens_total_sum",
            "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
            "prefix_cache_hits_total", "prefix_cache_queries_total", "num_preemptions_total")


async def counters(client):
    """Raw engine counters, for step ms and acceptance by window."""
    from arc_infra.metrics import parse_metrics

    try:
        values = parse_metrics((await client.get(client.base_url.copy_with(path="/metrics"), timeout=10)).text)
    except Exception:  # noqa: BLE001 - telemetry only
        return {}
    return {name: values.get("vllm:" + name) for name in COUNTERS}


def text_of(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def follow(messages, reply, row):
    """The next request's messages: the recorded row, with a plain continuation carrying the live reply."""
    keep, append = int(row["keep"]), row["append"]
    if keep < len(messages) or not append or append[0].get("role") != "assistant":
        return messages[:keep] + append
    message = reply["message"]
    entry = {"role": "assistant", "content": message.get("content") or ""}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if reasoning:
        entry["reasoning"] = reasoning
    rest = [dict(m) for m in append[1:]]
    call = tasks.tool_call(message)
    if call is not None:
        entry["tool_calls"] = [call[0]]
    if rest:
        if call is not None:
            rest[0] = {"role": "tool", "tool_call_id": call[0]["id"], "content": text_of(rest[0])}
        elif rest[0].get("role") == "tool":
            rest[0] = {"role": "user", "content": text_of(rest[0])}
    return messages + [entry] + rest


class Conv:
    def __init__(self, row):
        self.task, self.messages, self.rows = row["task"], row["messages"], deque(row["rows"])
        self.max_tokens, self.n = int(row["max_tokens"]), 0


async def run_steady(trial, source, output, transport, gate, settings, client, *, slots, window, started, wall, live,
                     engine_metrics):
    spec = trial["steady"]
    with gzip.open(Path(source) / spec["file"], "rt") as stream:
        data = json.load(stream)
    conversations = deque(Conv(row) for row in data["conversations"])
    events = (Path(output) / "events.jsonl").open("a", buffering=1)
    state = {"phase": "warm", "replies": 0, "gen": 0, "errors": [], "warmed": 0, "warm_seconds": None, "ended": 0, "gate_timeouts": 0}
    rng = random.Random(1)
    wall_end = started + wall

    def log(conv, reply, phase):
        usage = reply.get("usage") or {}
        gen = int(usage.get("completion_tokens") or 0)
        state["replies"] += 1
        state["gen"] += gen
        events.write(json.dumps({
            "t": round(time.monotonic() - started, 2), "event": "reply", "task": conv.task, "attempt": 1, "reply": conv.n,
            "completion_tokens": gen, "prompt_tokens": int(usage.get("prompt_tokens") or 0), "tokens_at": state["gen"],
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": reply.get("finish_reason"), "seconds": round(reply["t1"] - reply["t0"], 2), "t0": reply["t0"],
            "t1": reply["t1"], "phase": phase}) + "\n")

    def failed(conv, exc):
        state["errors"].append({"task": conv.task, "reply": conv.n, "error": f"{type(exc).__name__}: {exc}"[:300]})

    async def warm(conv):
        count = await transport.count(conv.messages, settings)
        reply = await transport.complete(conv.task, conv.messages, count, 1, {**settings, "reply": 0})
        log(conv, reply, "warm")
        transport.sent[conv.task] = count  # its next request re-sends this prompt: no new tokens for the prefill pacing
        state["warmed"] += 1

    async def play(conv):
        await asyncio.sleep(rng.uniform(0, STAGGER))
        while True:
            count = await transport.count(conv.messages, settings)
            if count >= window - 16:  # the recording drained about here; its next conversation takes the slot
                break
            conv.n += 1
            reply = await transport.complete(conv.task, conv.messages, count, conv.max_tokens, {**settings, "reply": conv.n})
            log(conv, reply, "steady")
            if not conv.rows:
                break
            row = conv.rows.popleft()
            conv.messages, conv.max_tokens = follow(conv.messages, reply, row), int(row["max_tokens"])
            await asyncio.sleep(float(row["gap"]))

    async def slot(conv):
        while conv is not None:
            try:
                await play(conv)
                state["ended"] += 1
            except asyncio.CancelledError:
                raise
            except TimeoutError:  # the wall, or a wait at a full gate: that one is a read, the slot tries again
                if time.monotonic() >= wall_end - 2:
                    raise
                state["gate_timeouts"] += 1
                await asyncio.sleep(1)
                continue
            except Exception as exc:  # noqa: BLE001 - a read: counted, the slot goes on, too many end the test
                failed(conv, exc)
                if len(state["errors"]) >= ERRORS:
                    raise
            transport.drop(conv.task)
            conv = conversations.popleft() if conversations else None

    history = []

    def row():
        return {"trial": trial["name"], "elapsed_seconds": round(time.monotonic() - started, 1), "wall_seconds": wall,
                "gate": gate.snapshot(), "engine": state.get("engine"), "completion_tokens": state["gen"],
                "replies": state["replies"], "request_errors": len(state["errors"]), "steady": {
                    k: state[k] for k in ("phase", "warmed", "warm_seconds", "ended", "gate_timeouts")}}

    async def progress():
        with (Path(output) / "steady-metrics.jsonl").open("a", buffering=1) as stream:
            while True:
                state["engine"] = await engine_metrics(client, history)
                now = {**row(), "counters": await counters(client)}
                stream.write(json.dumps(now) + "\n")
                text = json.dumps(now, indent=1)
                (Path(output) / "progress.json").write_text(text)
                if live is not None:
                    live.mkdir(parents=True, exist_ok=True)
                    (live / f"arc2-{trial['name']}.json").write_text(text)
                await asyncio.sleep(10)

    reporter = asyncio.create_task(progress())
    first = [conversations.popleft() for _ in range(min(slots, len(conversations)))]
    begun = time.monotonic()
    warming = [asyncio.create_task(warm(conv)) for conv in first]
    limit, idle, pending, done = begun + WARM_SHARE * (wall_end - begun), 0, set(warming), set()
    while pending and time.monotonic() < limit and idle < 3:  # idle: the rest wait at a full gate, held by the warmed
        finished, pending = await asyncio.wait(pending, timeout=5.0)
        done |= finished
        idle = idle + 1 if not transport.flying else 0
    for job in pending:  # those still waiting go on as steady requests: their first request does the prefill
        job.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for conv, job in zip(first, warming):
        if job in done and job.exception() is not None:
            failed(conv, job.exception())
    state.update(phase="steady", warm_seconds=round(time.monotonic() - begun, 1), steady_at=round(time.monotonic() - started, 1))
    jobs = [asyncio.create_task(slot(conv)) for conv in first]
    finished, pending = await asyncio.wait(jobs, timeout=max(1.0, wall_end - time.monotonic() - 10),
                                           return_when=asyncio.FIRST_EXCEPTION)
    state["ended_by"] = [repr(job.exception())[:200] for job in finished if job.exception() is not None][:3]
    for job in pending:
        job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
    reporter.cancel()
    report = {"status": "complete" if len(state["errors"]) < ERRORS else "error", "kind": "steady", "slots": slots,
              "replies": state["replies"], "completion_tokens": state["gen"], "request_errors": len(state["errors"]),
              "errors": state["errors"][:ERRORS], "warm_seconds": state["warm_seconds"], "ended_by": state.get("ended_by"), "steady_at": state.get("steady_at"),
              "warmed": state["warmed"], "ended": state["ended"], "gate_timeouts": state["gate_timeouts"], "snapshot": data["meta"], "gate": gate.snapshot(),
              "play_seconds": round(time.monotonic() - started, 1), "elapsed_seconds": round(time.monotonic() - started, 1)}
    (Path(output) / "report.json").write_text(json.dumps(report, indent=1))
    events.close()
    return report
