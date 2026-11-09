"""Arena trial kind ``arc2``: the tasks of one set against the arena's server, through the arena's KV gate.

Scheduling: ``streams`` workers take tasks from a queue ordered by tier, then roster order. A task plays until it ends
or has spent its tier's tokens, then waits for the next tier, so at the wall every task has at least the last finished
tier (the read-out marks each budget as complete or a lower bound). Trial fields: ``tasks`` (ids), ``split``,
``tiers`` (cumulative completion tokens), ``wall_seconds``, optional ``arms`` (setting overrides, assigned to tasks
in turn, or with ``paired`` every task under every arm; the read-out is given per arm as well),
``min_verified_tasks`` (fewer: the trial ends as an error) and ``arm_from`` (play the winning arm of an earlier trial).

Roster trials (the competition's shape, test as the 240-task hidden run is scored) have
``marks`` in place of ``tiers``: named totals of completion tokens for one arm's roster. No task has a cap; the next
reply goes to the task ``Engine.priority`` puts first, and an arm takes no new reply once its finished replies reach
the largest mark (the replies in flight finish, so the token clock is exact). ``copies`` plays every task that many
times as independent entries. The read-out (``roster``) is the competition's score against the roster's total tokens.
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

from arc_infra import KEEPALIVE_EXPIRY
from arena import engine_watchdog
from arena.cache import BlockBudget
from arena.early_stop import stop_paths
from arena.gate import KVGate

from . import sandbox, tasks
from .loop import DEFAULTS, ContextFull, Engine, StopRequested, TaskRun, guesses, readout, roster

PROGRESS_SECONDS = 60
# Graceful stop: ``stop-request`` in the trial folder or at $ARENA_STOP_REQUEST (the runners write it
# when validation/<run>/arena/live/stop-request appears). No reply starts after it; replies in flight finish within
# STOP_GRACE seconds (trial ``stop_grace_seconds``), then are cut as at the wall; the report is written as at the wall.
STOP_GRACE, STOP_POLL = 900.0, 5.0
DRAIN_SECONDS = 600.0  # end drain + write-out room under the notebook limit
# live wall (checklist item 5): {"wall_seconds": s} in this file (default <output>/control.json; the Nebius runner
# mirrors the bucket's arena/live/control.json to $ARENA_CONTROL) sets the wall each tick, capped by the machine deadline
CONTROL_ENV, CONTROL_FILE = "ARENA_CONTROL", "control.json"


def control_read(path, seen):
    """The control file's contents when it changed since ``seen[0]`` (its mtime), else None."""
    try:
        mtime = path.stat().st_mtime
        if mtime == seen[0]:
            return None
        seen[0] = mtime
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def control_wall(path, seen, read=None):
    """The control file's ``wall_seconds`` when the file changed since ``seen[0]`` (its mtime), else None."""
    value = control_read(path, seen) if read is None else read
    try:
        value = float(value["wall_seconds"])
        return value if value > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


LIVE_ADMISSION = ("target", "floor", "step")  # control.json {"admission": {...}, "slots": n}: read every tick
ORDERS = ("balance", "first_submit", "first_submit_asc", "tokens", "confidence")  # control.json "order": the backlog / reopen order


def control_order(engine, read, queue, runs):
    """control.json ``order``: the backlog and reopen order of a conversation
    engine, applied now; waiting backlog entries are re-keyed. Returns the new order or None."""
    order = (read or {}).get("order")
    if order not in ORDERS or not hasattr(engine, "live_order") or order == engine.live_order:
        return None
    engine.live_order = order
    items = getattr(queue, "_queue", None)
    if items is not None:  # backlog entries (priority, n); reopen tokens (n = -1) keep theirs
        items[:] = [((engine.priority(runs[n]) if n >= 0 else key), n) for key, n in items]
        __import__("heapq").heapify(items)
    return order


def control_gate(gate, read, cap):
    """Apply the control file's live gate knobs (checklist item 5): CacheGate target/floor/step and the gate's
    slot ceiling (at most ``cap``, the worker count). Returns what changed."""
    changed = {}
    for key, value in ((read or {}).get("admission") or {}).items():
        if key in LIVE_ADMISSION and hasattr(gate, key):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if 0 < value < 1 and getattr(gate, key) != value:
                setattr(gate, key, value)
                changed[key] = value
    if hasattr(gate, "floor") and gate.floor > gate.target:
        gate.floor = changed["floor"] = gate.target
    slots = (read or {}).get("slots")
    if isinstance(slots, (int, float)) and 1 <= int(slots) and min(int(slots), cap) != gate.slots:
        gate.slots = changed["slots"] = min(int(slots), cap)
    if changed:
        gate._pump()
    return changed


def efforts(study, arms):
    """Every reasoning effort the trial can send: the study's, its settings' and each arm's (effort, confirm_effort,
    hard_effort, late_effort, open_schedule, scout_effort)."""
    found = {study.get("reasoning_effort")}
    for block in [(study.get("arc2") or {}).get("settings") or {}, *arms]:
        found |= {block.get(k) for k in ("effort", "confirm_effort", "hard_effort", "late_effort")}
        found |= {row[0] for row in block.get("open_schedule") or []}
        if block.get("scouts"):
            found.add(block.get("scout_effort", "low"))
    return sorted(e for e in found if e is not None)
STATE_KEYS = ("spent", "prompt_tokens", "attempts", "confirms", "first_verified_attempt", "candidates")
CONV_STATE_KEYS = ("done", "submits", "reopened", "awaiting_reopen", "first_submit_tokens")  # arc2.conv runs


GROW_WAIT = 120  # s an attempt waits for more blocks with its own held; past it it gives them up and queues


CONTINUE = {"add_generation_prompt": False, "continue_final_message": True}
# think_cap (29 Oct; naive12: 331 replies, 26% of tokens, ran 16K inside the think block): a reply still thinking at
# the cap gets this line and the end of the think block appended to its own tokens, and the same turn goes on
# through /v1/completions on the exact token prefix (prefix-cached); the tool call is parsed here (vLLM's parsers
# read a continued turn as reasoning).
THINK_BRIDGE = "\n\nI have thought about this long enough. Now I act: I test my current idea with the python tool.\n"
THINK_END = "\n</think>\n\n"
THINK_ACT = "I act now: I test my current idea with the python tool."  # think_chat: the answer's first line


def parse_tool_calls(text):
    """``(content, calls)`` from a raw Qwen3-Coder reply: the text before ``<tool_call>`` and each complete call as an
    OpenAI tool call (parameter values without their one leading and trailing newline, as vLLM's qwen3_coder)."""
    i = text.find("<tool_call>")
    if i < 0:
        return text.strip(), []
    calls = []
    for m in re.finditer(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", text[i:], re.S):
        args = {p.group(1): p.group(2) for p in re.finditer(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>",
                                                             m.group(2), re.S)}
        calls.append({"id": f"call_t{len(calls)}", "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    return (text[:i].strip() if calls else text.strip()), calls

class Transport:
    """Exact prompt counts from the server tokenizer and gated chat completions, as the arena's Duck requests are
    sent: non-streaming, top_p 0.95, top_k 20, thinking on at the study's effort, earlier reasoning in the prompt."""

    def __init__(self, client, gate, *, model, effort, temperature, deadline, prefill=None, hold=False,
                 grow_wait=GROW_WAIT, reserve=None, template=None):
        self.client, self.gate, self.model = client, gate, model
        self.template = dict(template or {})  # the study's chat_template_extra (candidate models)
        # ``reserve``: reply tokens a request holds at the gate, under its cap (four L4s: 14% of replies reach the
        # 8,192 cap and the mean is about 2,900, so the cap over-reserves about 1.3 of 19.7 blocks a reply; the
        # server's own margin takes the replies that run past). None: the cap, as before.
        self.reserve = None if reserve is None else int(reserve)
        # ``hold`` (a small cache pool, four L4s): a task keeps its blocks between the turns of an attempt and its
        # next turn takes the blocks it needs more before anything that waits (``grow``); ``drop`` gives them back.
        # Without it a turn queued behind the waiting first turns (package rehearsal 3 at 292 blocks: 100 s between
        # turns at the median), its conversation left the cache meanwhile (21 of 145 later turns found it) and 91%
        # of the prompt tokens were computed again.
        # A turn that cannot grow waits with its blocks (``grown``): rehearsal 7 gave them up 23 times, waited a
        # median 181 s at the gate and computed 0.71M prompt tokens again, a third of all it computed.
        self.hold, self.held, self.growers, self.flying = bool(hold), {}, [], 0
        self.grow_wait = float(grow_wait)  # 0: an attempt that cannot grow gives its blocks up at once
        self.effort, self.temperature, self.deadline = effort, temperature, deadline
        # ``prefill`` {"slots": n, "tokens_per_second": r} (four L4s: every request prefilling at once costs about
        # 0.2 GiB of each card): at most n requests inside their prefill, taken as new prompt tokens / r seconds.
        self.stopped = None  # run_arc2 sets its stop check: a reply that waited at the gate is not sent after a stop
        self.stop_flag, self.sent_after_stop = None, 0  # replies sent once the stop was already seen (event count)
        self.prefill = prefill and (asyncio.Semaphore(int(prefill["slots"])), float(prefill["tokens_per_second"]))
        self.sent = {}  # task -> tokens the server has seen of its conversation (prompt + reply), for the above
        self.charge_units, self.reply_mean = {}, 4096.0  # arc2.admission charge: per task, and the mean reply
        self.inflight_started = {}
        self.server_errors = []  # the first 5xx bodies: the retries below end as connection errors once a server died
        self.served = None  # arc2.served.ServedLog: every request's exact input and reply (SFT reuse)

    def kwargs(self, settings):
        return {"enable_thinking": True, "reasoning_effort": settings.get("effort") or self.effort, **self.template}

    def left(self):
        left = self.deadline - time.monotonic()
        if left <= 1:
            raise TimeoutError("the trial's wall")
        return left

    async def count(self, messages, settings):
        url = self.client.base_url.copy_with(path="/tokenize")
        response = await self.client.post(url, json={"model": self.model, "messages": messages,
                                                     "chat_template_kwargs": self.kwargs(settings),
                                                     **({"tools": [tasks.tool_spec(settings)]} if settings.get("tool") else {}),
                                                     **(CONTINUE if settings.get("continue") else {})},
                                          timeout=min(120, self.left()))
        response.raise_for_status()
        return int(response.json()["count"])

    async def complete(self, task_id, messages, count, max_tokens, settings):
        payload = tasks.chat_request(self.model, messages, max_tokens,
                                     temperature=settings.get("temperature") or self.temperature,
                                     effort=self.kwargs(settings)["reasoning_effort"],
                                     tools=[tasks.tool_spec(settings)] if settings.get("tool") else None,
                                     template=self.template)
        if settings.get("continue"):  # a prefilled summary (conv.compact): continue the last assistant turn
            payload.update(CONTINUE)
            if payload.get("tools"):
                payload["tool_choice"] = "none"
        if self.served is not None and self.served.ids:  # exact token ids for SFT (prompt kept as its sha256)
            payload["return_token_ids"] = True
        think_cap = int(settings.get("think_cap") or 0)
        if think_cap and not settings.get("continue") and int(max_tokens) > think_cap:
            payload.update(max_tokens=think_cap, return_token_ids=True)
        else:
            think_cap = 0
        held = count + (int(max_tokens) if self.reserve is None else min(int(max_tokens), self.reserve))
        if settings.get("gate_reserve"):  # direct_answer_reserve: this request holds less than its cap at the gate
            held = count + min(int(max_tokens), int(settings["gate_reserve"]))
        if hasattr(self.gate, "hint"):  # arc2.admission charge: what this turn adds to the server's used blocks.
            # The whole prompt: a cached prefix sits in free (evictable) blocks until the request pins it again, so
            # /metrics' usage rises by the full prompt; only the reply is charged at its mean instead of the cap.
            units = self.gate._units(count + min(int(max_tokens), int(self.reply_mean)))
            self.charge_units[task_id] = units
            self.gate.hint(task_id, units)
        if hasattr(self.gate, "rank"):
            self.gate.rank[task_id] = int(settings.get("gate_rank", 1))
        reservation = self.held.pop(task_id, None)
        if reservation is not None:
            reservation = await self.grown(reservation, held)
        if reservation is None:
            reservation = await self.gate.acquire(task_id, held)
        keep = False
        try:
            if self.prefill:
                lock, rate = self.prefill
                seen = self.sent.get(task_id, 0)
                await lock.acquire()
                asyncio.get_running_loop().call_later((count - seen if count >= seen else count) / rate, lock.release)
            if self.stopped is not None and self.stopped():  # the stop landed while this reply waited at the gate
                raise StopRequested()
            t0 = time.monotonic()
            if self.stop_flag is not None and self.stop_flag():  # counted, never expected: the check above runs first
                self.sent_after_stop += 1
            self.flying += 1
            try:
                delays, outages = [5, 15, None], 0
                while True:
                    seen = len(engine_watchdog.STATE["events"])
                    response = await self.client.post("chat/completions", json=payload, timeout=self.left())
                    if response.status_code >= 500 and len(self.server_errors) < 5:
                        self.server_errors.append({"status": response.status_code, "task": task_id,
                                                   "body": response.text[:4000]})
                    if response.status_code < 500:
                        break
                    # engine watchdog (arena.engine_watchdog): a 5xx while the server is down or being restarted
                    # waits for it (transport errors wait inside the patched client), at most 5 outages a request
                    if (engine_watchdog.enabled() and outages < 5
                            and await engine_watchdog.settle(min(engine_watchdog.OUTAGE_WAIT, self.left() - 1))
                            and len(engine_watchdog.STATE["events"]) > seen):
                        outages += 1
                        continue
                    delay = delays.pop(0)
                    if delay is None:
                        break
                    await asyncio.sleep(delay)
            finally:
                self.flying -= 1
            keep = self.hold and response.status_code == 200
            data = response.json() if response.status_code == 200 else None
            if think_cap and data and data["choices"][0].get("finish_reason") == "length":
                rest = self.think_rest_chat if settings.get("think_chat") else self.think_rest
                data = await rest(data, payload, int(max_tokens) - think_cap)
            if self.served is not None and data is not None:
                self.served.write(task_id, payload, data, t0, time.monotonic(), reply=settings.get("reply_n"))
        finally:
            if keep:
                self.held[task_id] = reservation
            else:
                self.release(reservation)
        if response.status_code == 400 and "context" in response.text.lower():
            raise ContextFull(response.text[:200])
        response.raise_for_status()
        choice = data["choices"][0]
        done = int((data.get("usage") or {}).get("completion_tokens") or 0)
        self.sent[task_id] = count + done
        self.reply_mean += 0.05 * (done - self.reply_mean)  # the charge's expected reply
        raw = await self.raw_text(choice) if settings.get("sandbox_fix") else None
        return {"message": choice["message"], "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage") or {}, "t0": t0, "t1": time.monotonic(),
                **({"think_cut": data["think_cut"]} if data.get("think_cut") else {}), **({"raw": raw} if raw else {})}

    async def raw_text(self, choice):
        """``sandbox_fix`` (base20): the reply's raw text after its thinking (server /detokenize of its token ids) when
        its `python` call came back without code; None otherwise or on any failure (the call's error text stands)."""
        empty = tasks.tool_call(choice.get("message") or {})
        if empty is None or empty[1] is not None or not choice.get("token_ids"):
            return None
        try:
            url = self.client.base_url.copy_with(path="/detokenize")
            response = await self.client.post(url, json={"model": self.model, "tokens": list(choice["token_ids"])},
                                              timeout=min(60, self.left()))
            response.raise_for_status()
            text = str(response.json().get("prompt") or "")
        except Exception:  # noqa: BLE001 - salvage is best effort
            return None
        return text[text.rfind("</think>") + len("</think>"):] if "</think>" in text else text

    async def token_ids(self, text):
        cache = self.__dict__.setdefault("_ids", {})
        if text not in cache:
            url = self.client.base_url.copy_with(path="/tokenize")
            response = await self.client.post(url, json={"model": self.model, "prompt": text, "add_special_tokens": False},
                                              timeout=min(120, self.left()))
            response.raise_for_status()
            cache[text] = list(response.json()["tokens"])
        return cache[text]

    async def think_rest(self, data, payload, budget):
        """``think_cap``: the rest of a reply cut at the cap, in the same assistant turn. Still thinking: THINK_BRIDGE
        and THINK_END are appended to its tokens; cut inside the answer: it goes on as it was. On any failure the cut
        reply stands (the conversation's own cut path)."""
        choice = data["choices"][0]
        ids, prompt_ids = choice.get("token_ids"), data.get("prompt_token_ids")
        if not ids or not prompt_ids or budget <= 0:
            return data
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        end = await self.token_ids("</think>")
        if len(end) == 1 and end[0] in ids:  # cut inside the answer
            url = self.client.base_url.copy_with(path="/detokenize")
            response = await self.client.post(url, json={"model": self.model, "tokens": ids[ids.index(end[0]) + 1:]},
                                              timeout=min(120, self.left()))
            response.raise_for_status()
            head, extra, mode = response.json()["prompt"], [], "answer"
        else:
            head, extra, mode = "", await self.token_ids(THINK_BRIDGE + THINK_END), "think"
            reasoning = reasoning.rstrip() + THINK_BRIDGE.rstrip()
        body = {"model": self.model, "prompt": list(prompt_ids) + list(ids) + list(extra), "max_tokens": int(budget),
                "temperature": payload["temperature"], "top_p": payload["top_p"], "top_k": payload["top_k"],
                "skip_special_tokens": False}
        response = await self.client.post("completions", json=body, timeout=self.left())
        if response.status_code != 200:
            self.server_errors.append({"status": response.status_code, "think_rest": True, "body": response.text[:2000]}) \
                if len(self.server_errors) < 5 else None
            return data
        rest = response.json()
        more = rest["choices"][0]
        text = head + (more.get("text") or "")
        for special in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(special, "")
        cut = more.get("finish_reason") == "length"
        content, calls = (text.strip(), []) if cut else parse_tool_calls(text)
        merged = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
        if calls:
            merged["tool_calls"] = calls
        first, second = data.get("usage") or {}, rest.get("usage") or {}
        usage = {**first, "completion_tokens": int(first.get("completion_tokens") or 0) + int(second.get("completion_tokens") or 0),
                 "think_rest_prompt": second.get("prompt_tokens"),
                 "think_rest_cached": (second.get("prompt_tokens_details") or {}).get("cached_tokens")}
        finish = "length" if cut else "tool_calls" if calls else "stop"
        return {**data, "choices": [{**choice, "message": merged, "finish_reason": finish}], "usage": usage,
                "think_cut": mode}


    async def think_rest_chat(self, data, payload, budget):
        """``think_cap`` + ``think_chat`` (30 Oct; think1's raw-id continuation lost the images): the rest of a reply cut
        at the cap as a chat request on the same messages (images included) plus the cut turn continued
        (continue_final_message). Still thinking: its reasoning + THINK_BRIDGE, closed, then THINK_ACT as the start of
        the answer; cut inside the answer: its reasoning and the answer so far. On any failure the cut reply stands."""
        choice = data["choices"][0]
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        head, mode = (message.get("content") or "", "answer") if (message.get("content") or "").strip() else (
            THINK_ACT, "think")
        if mode == "think":
            reasoning = reasoning.rstrip() + THINK_BRIDGE.rstrip()
        if budget <= 0 or not reasoning.strip():
            return data
        body = {k: v for k, v in payload.items() if k != "return_token_ids"}
        body.update(CONTINUE, max_tokens=int(budget),
                    messages=list(payload["messages"]) + [{"role": "assistant", "content": head, "reasoning": reasoning,
                                                           "reasoning_content": reasoning}])  # conv's key + the template's
        response = await self.client.post("chat/completions", json=body, timeout=self.left())
        if response.status_code != 200:
            self.server_errors.append({"status": response.status_code, "think_rest": True, "body": response.text[:2000]}) \
                if len(self.server_errors) < 5 else None
            return data
        rest = response.json()
        more = rest["choices"][0]
        got = more.get("message") or {}
        calls = list(got.get("tool_calls") or [])
        # the parsers may read a continued turn as reasoning: the text of both fields, tool calls parsed here
        text = (got.get("reasoning_content") or got.get("reasoning") or "") + (got.get("content") or "")
        for special in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(special, "")
        cut = more.get("finish_reason") == "length"
        if not calls and not cut:
            text, calls = parse_tool_calls(text)
        merged = {"role": "assistant", "content": (head + text).strip() or None, "reasoning_content": reasoning}
        if calls:
            merged["tool_calls"] = calls
        first, second = data.get("usage") or {}, rest.get("usage") or {}
        usage = {**first, "completion_tokens": int(first.get("completion_tokens") or 0) + int(second.get("completion_tokens") or 0),
                 "think_rest_prompt": second.get("prompt_tokens"),
                 "think_rest_cached": (second.get("prompt_tokens_details") or {}).get("cached_tokens")}
        finish = "length" if cut else "tool_calls" if calls else "stop"
        return {**data, "choices": [{**choice, "message": merged, "finish_reason": finish}], "usage": usage,
                "think_cut": mode}

    def room(self, cost, held, freed=0, charge=None):
        """Room for a held reservation of ``held`` units to grow to ``cost`` (``freed``: units being released).
        The cache gate (``arc2.admission``) counts the whole next turn, or its ``charge``, against the real KV use."""
        gate = self.gate
        if hasattr(gate, "grow_fits"):
            return gate.grow_fits(cost, held, freed, charge=charge)
        return gate.reserved - freed + cost - held <= gate.capacity

    def grow(self, reservation, tokens):
        """The held reservation at the cost of ``tokens``, if the pool has the difference now."""
        gate = self.gate
        cost = gate.budget.cost(int(tokens)) if gate.budget else int(min(tokens, gate.capacity))
        charge = self.charge_units.get(reservation.game_id)
        if cost > reservation.tokens and not self.room(cost, reservation.tokens, charge=charge):
            return False
        if hasattr(gate, "note_growth"):
            gate.note_growth(cost, charge)
        gate.reserved += cost - reservation.tokens
        reservation.tokens = cost
        gate.peak_reserved = max(gate.peak_reserved, gate.reserved)
        if gate.log:
            gate.log({"event": "grow", "game": reservation.game_id, "tokens": cost, "reserved_after": gate.reserved,
                      "inflight_after": gate.inflight})
        return True

    async def grown(self, reservation, tokens):
        """The held reservation at the cost of ``tokens``: at once if the pool has the difference, else after
        waiting for it with the blocks held, served before anything queued at the gate (``release``). None when
        no reply in flight can free blocks or after ``GROW_WAIT`` s: the blocks were given up, as before."""
        if self.grow(reservation, tokens):
            return reservation
        if self.grow_wait > 0 and self.flying > 0 and self.left() > 1:
            future = asyncio.get_running_loop().create_future()
            row = (future, reservation, int(tokens))
            self.growers.append(row)
            try:
                await asyncio.wait_for(future, min(self.grow_wait, self.left()))
                return reservation
            except TimeoutError:
                pass
            finally:
                if row in self.growers:
                    self.growers.remove(row)
        self.release(reservation)
        return None

    def release(self, reservation):
        """Give a reservation back: its units go first to the attempts waiting to grow, in the order they asked."""
        gate, freed = self.gate, reservation.tokens
        for row in list(self.growers):
            future, held, tokens = row
            cost = gate.budget.cost(tokens) if gate.budget else int(min(tokens, gate.capacity))
            if future.done():
                self.growers.remove(row)
            elif self.room(cost, held.tokens, freed, charge=self.charge_units.get(held.game_id)):
                if hasattr(gate, "note_growth"):
                    gate.note_growth(cost, self.charge_units.get(held.game_id))
                gate.reserved += cost - held.tokens  # above the capacity only until the release two lines below
                held.tokens = cost
                self.growers.remove(row)
                future.set_result(True)
                if gate.log:
                    gate.log({"event": "grow", "game": held.game_id, "tokens": cost, "waited": True,
                              "reserved_after": gate.reserved - freed, "inflight_after": gate.inflight})
        gate.release(reservation)

    def drop(self, task_id):
        reservation = self.held.pop(task_id, None)
        if reservation is not None:
            self.release(reservation)


async def generated_tokens(client):
    """The server's own count of generated tokens (in-flight replies cut at the wall included); None if unreadable."""
    try:
        response = await client.get(client.base_url.copy_with(path="/metrics"), timeout=20)
        return sum(float(line.rsplit(" ", 1)[1]) for line in response.text.splitlines()
                   if line.startswith("vllm:generation_tokens_total"))
    except Exception:  # noqa: BLE001 - telemetry only
        return None


ENGINE_METRICS = {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting",
                  "kv_cache_fraction": "vllm:kv_cache_usage_perc", "preemptions": "vllm:num_preemptions_total"}
PREFIX_WINDOW = 1800  # prefix_hit_rate: hits / queries over the last 30 min


async def engine_metrics(client, history):
    """The live feed's ``engine`` dict from the server's /metrics (ARC-3 style run cards); ``history``
    keeps (time, prefix hits, prefix queries) for the 30-min hit rate. None when /metrics is unreadable."""
    from arc_infra.metrics import parse_metrics

    try:
        response = await client.get(client.base_url.copy_with(path="/metrics"), timeout=10)
        values = parse_metrics(response.text)
    except Exception:  # noqa: BLE001 - telemetry only
        return None
    out = {label: values.get(key) for label, key in ENGINE_METRICS.items()}
    hits, queries = values.get("vllm:prefix_cache_hits_total"), values.get("vllm:prefix_cache_queries_total")
    if hits is not None and queries is not None:
        now = time.monotonic()
        history.append((now, hits, queries))
        while len(history) > 2 and now - history[1][0] >= PREFIX_WINDOW:
            history.pop(0)
        base = history[0]
        asked = queries - base[2]
        out["prefix_hit_rate"] = round((hits - base[1]) / asked, 4) if asked > 0 else None
    return out


def block_budget(server_dir, margin, smallest=False):
    """The served cache layout for the gate. ``smallest`` (four tensor-parallel ranks, ``arc2.l4_serving``): when
    the ranks print different pools, the smallest one instead of the arena's refusal."""
    try:
        return BlockBudget.load(server_dir, margin)
    except RuntimeError as exc:
        if not smallest or "disagree" not in str(exc):
            raise
    layouts = []
    for log in sorted(Path(server_dir).glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "ARC_CACHE_LAYOUT " in line:
                layouts.append(json.JSONDecoder().raw_decode(line.split("ARC_CACHE_LAYOUT ", 1)[1])[0])
    return BlockBudget(min(layouts, key=lambda x: x["num_blocks"]), margin)


def summary(runs, started, arms, marks=None, cuts=None, step=None, extra=()):
    """The read-out with tokens per second, overall and per arm; with ``marks`` each arm's roster read-out.
    ``extra``: runs outside the read-out whose tokens still run the roster clock (``copy3``)."""
    def one(rows, name):
        clock = rows + [r for r in extra if r.arm.get("name") == name] if extra else None
        return {**readout(rows), **({"roster": roster(rows, marks, cut=(cuts or {}).get(name), step=step,
                                                      clock_runs=clock)} if marks else {})}

    seconds = time.monotonic() - started
    out = {**(readout(runs) if len(arms) > 1 else one(runs, arms[0]["name"])), "play_seconds": round(seconds, 1)}
    out["extra_tokens"] = sum(r.spent for r in extra)
    out["tokens_per_second"] = (out["completion_tokens"] + out["extra_tokens"]) / max(1.0, seconds)
    if len(arms) > 1:
        out["per_arm"] = {arm["name"]: one([r for r in runs if r.arm.get("name") == arm["name"]], arm["name"])
                          for arm in arms}
    return out


def winner(report):
    """Roster trials, fixed before launch: the arm with the best mean score over the marks (ties: the trial's
    order). The evaluation bundle is built with it."""
    names = [arm["name"] for arm in report["arms"]]
    rows = {n: [m["score_outputs"] for m in report["per_arm"][n]["roster"]["marks"].values()] for n in names}
    best = max(names, key=lambda n: (sum(rows[n]) / len(rows[n]), -names.index(n)))
    return {"arm": best, "marks": list(report["marks"]), "scores": {n: [round(x, 2) for x in rows[n]] for n in names}}


def winning_arm(study, trials, *, trial, budget):
    """The arm of an earlier trial of this run with the best score at ``budget`` tokens per task (ties: fewer mean
    tokens, then the trial's order), from that trial's report. Fixed in the study before launch."""
    report = json.loads((Path(trials) / trial / "report.json").read_text())
    arms = next(t for t in study["trials"] if t["name"] == trial)["arms"]
    rows = {name: next(c for c in out["curve"] if c["budget"] == budget) for name, out in report["per_arm"].items()}
    best = min(arms, key=lambda arm: (-rows[arm["name"]]["score_outputs"], rows[arm["name"]]["mean_tokens"]))
    return {"arm": best, "budget": budget, "from": trial,
            "scores": {name: [round(row["score_outputs"], 2), round(row["mean_tokens"])] for name, row in rows.items()}}


async def review_replay(items, task_map, transport, execute, *, window, cap, settings, record, workers, deadline):
    """The review turn alone, on stored conversations that end in a verified program (``trial["replay"]``; built
    from a finished run of training tasks). Each item is asked ``samples`` times; a row says whether the reply
    kept the program or sent a correction, and whether the correction passes the pairs and is right on the tests.
    ``kind`` is the stored program's own state: "wrong" items measure the gain, "right" ones the false alarms."""
    queue, rows = asyncio.Queue(), []
    for item in items:
        for k in range(int(item.get("samples", 1))):
            queue.put_nowait((item, k))

    async def one(item, k):
        task = task_map[item["task"]]
        messages = item["messages"] + [{"role": "user", "content": tasks.review_prompt(
            task, item["tests"], settings["review_style"])}]
        row = {"event": "review_replay", "item": item["id"], "task": item["task"], "kind": item["kind"], "sample": k}
        count = await transport.count(messages, settings)
        if window - count < settings["min_room_tokens"]:
            return {**row, "outcome": "no_room", "completion_tokens": 0, "prompt_tokens": 0}
        reply = await transport.complete(f"{item['id']}#{k}", messages, count,
                                         min(settings["cap"] or cap, window - count), settings)
        usage, content = reply["usage"], reply["message"].get("content") or ""
        row.update(completion_tokens=int(usage.get("completion_tokens") or 0),
                   prompt_tokens=int(usage.get("prompt_tokens") or 0),
                   cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   finish=reply.get("finish_reason"), seconds=round(reply["t1"] - reply["t0"], 2), content=content,
                   reasoning=reply["message"].get("reasoning") or reply["message"].get("reasoning_content") or "")
        code = tasks.extract_code(content)
        if code is None:
            return {**row, "outcome": "cut" if reply.get("finish_reason") == "length" else "keep"}
        result = await execute(code, [p["input"] for p in task["train"]] + [t["input"] for t in task["test"]])
        verdict = tasks.judge(task, result)
        return {**row, "outcome": "corrected" if verdict["verified"] else "correction_fails_pairs", "code": code,
                "passed": verdict["passed"], "pairs": verdict["pairs"], "changed": verdict["tests"] != item["tests"],
                "right": verdict["tests"] == [t["output"] for t in task["test"]],
                "tests": verdict["tests"], "accuracy": verdict["accuracy"], "verified": verdict["verified"]}

    async def worker():
        while True:
            item, k = await queue.get()
            try:
                if time.monotonic() < deadline:
                    rows.append(await one(item, k))
                    record(rows[-1])
            except (TimeoutError, httpx.TimeoutException):
                pass  # the wall
            except Exception as exc:  # noqa: BLE001 - one item's failure never ends the trial
                rows.append({"event": "review_replay", "item": item["id"], "kind": item["kind"], "sample": k,
                             "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:300],
                             "completion_tokens": 0, "prompt_tokens": 0})
                record(rows[-1])
            finally:
                queue.task_done()

    jobs = [asyncio.create_task(worker()) for _ in range(workers)]
    try:
        await asyncio.wait_for(queue.join(), timeout=max(1.0, deadline - time.monotonic()))
    except TimeoutError:
        pass
    for job in jobs:
        job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
    return rows


def replay_summary(rows, items):
    """Counts per kind of stored program: samples, what the review did, and what a correction was worth."""
    out = {}
    for kind in sorted({item["kind"] for item in items}):
        mine = [r for r in rows if r["kind"] == kind]
        fixed = [r for r in mine if r["outcome"] == "corrected"]
        out[kind] = {
            "items": sum(item["kind"] == kind for item in items), "samples": len(mine),
            "outcomes": {name: sum(r["outcome"] == name for r in mine) for name in sorted({r["outcome"] for r in mine})},
            "corrected_right": sum(r["right"] for r in fixed),
            "corrected_other_output_wrong": sum(r["changed"] and not r["right"] for r in fixed),
            "corrected_same_output": sum(not r["changed"] for r in fixed),
            "items_with_a_right_correction": len({r["item"] for r in fixed if r["right"]}),
            "completion_tokens": sum(r["completion_tokens"] for r in mine)}
    return out


SERVED_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")


def served_identity(model):
    """The served model's template identity (SFT reuse): its folder, the sha256 of the files that set tokenisation
    and the chat template, and the vLLM server's command line."""
    root = Path(str(model.get("root") or ""))
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in SERVED_FILES if root.is_dir() and (root / name).is_file()}
    argv = None
    for proc in Path("/proc").glob("[0-9]*/cmdline") if Path("/proc").is_dir() else []:
        try:
            parts = proc.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any("vllm" in part for part in parts[:3]) and any(p in ("serve", "vllm.entrypoints.openai.api_server")
                                                            for p in parts):
            argv = [p for p in parts if p]
            break
    return {"model_id": model.get("id"), "model_root": str(root), "model_files_sha256": files, "server_argv": argv}


def engine_settings_hint(arms):
    return [{k: v for k, v in arm.items() if k != "name"} for arm in arms]


def graded(run):
    """Per candidate, each test output right (None without the answer), and the final two guesses per output with
    whether each is right (SFT reuse: which attempt was submitted and how it scored)."""
    truth = [t.get("output") for t in run.task["test"]]
    known = all(x is not None for x in truth)
    cands = [{"code": c.get("code"), "submitted": c.get("submitted"), "submit_n": c.get("submit_n"),
              "submit_at": c.get("submit_at"), "reply": c.get("reply"), "tokens_at": c.get("tokens_at"),
              "right": [None if not known or g is None else g == x for g, x in zip(c["tests"], truth)]}
             for c in run.candidates]
    picks = guesses(run.candidates, [t["input"] for t in run.task["test"]])
    final = [[{"source": src, "right": None if not known else g == x} for g, src in pick] for pick, x in zip(picks, truth)]
    return {"candidates": cands, "final": final}


def rescore(baseline, replay, data):
    """A finished roster run rescored with a review replay of its own first verified programs: each review's
    program joins its entry's candidates as the loop would have had it (``review``: after an independent program
    with as many votes) and the guesses are picked again. Counts only; ``data`` is the split's task folder.
    Slot 2 of an output in the baseline: "empty, one program", "empty, programs agree" (two or more verified
    programs, one output), "unverified" or "verified"."""
    import hashlib

    reviews = {}
    for row in map(json.loads, (Path(replay) / "events.jsonl").read_text().splitlines()):
        if row.get("event") == "review_replay":
            reviews[row["item"].split(":")[0]] = row
    out = {"entries": 0, "reviews": len(reviews), "outputs": 0, "baseline_right": 0, "with_review_right": 0,
           "review_tokens": sum(r["completion_tokens"] for r in reviews.values()),
           "review_outcomes": {}, "answer_1_wrong_and_review_right": 0, "of_those_submitted": 0, "lost": 0,
           "by_slot_2": {}}
    for row in reviews.values():
        out["review_outcomes"][row["outcome"]] = out["review_outcomes"].get(row["outcome"], 0) + 1
    for row in map(json.loads, (Path(baseline) / "tasks.jsonl").read_text().splitlines()):
        task = json.loads((Path(data) / f"{row['task'].split('#')[0]}.json").read_text())
        inputs, cands, review = [t["input"] for t in task["test"]], row["candidates"], reviews.get(row["task"])
        more = list(cands)
        if review and review.get("code") and len(review.get("tests") or []) == len(inputs):
            first = min((c for c in cands if c["verified"]), key=lambda c: c["tokens_at"])
            more.append({"code": hashlib.sha256(review["code"].encode()).hexdigest()[:16], "review": True,
                         "passed": review["passed"], "pairs": review["pairs"], "accuracy": review["accuracy"],
                         "verified": review["verified"], "tests": review["tests"],
                         "tokens_at": first["tokens_at"] + review["completion_tokens"]})
        out["entries"] += 1
        for k, (old, new, test) in enumerate(zip(guesses(cands, inputs), guesses(more, inputs), task["test"])):
            before, after = (any(g == test["output"] for g, _ in picks) for picks in (old, new))
            programs = {c["code"] for c in cands if c["verified"] and c["tests"][k] is not None}
            slot = ("no verified answer" if old[0][1] != "verified" else
                    "verified" if len(old) == 2 and old[1][1] == "verified" else
                    "unverified" if len(old) == 2 else
                    "empty, programs agree" if len(programs) >= 2 else "empty, one program")
            cell = out["by_slot_2"].setdefault(slot, {"outputs": 0, "baseline_right": 0, "with_review_right": 0})
            cell["outputs"] += 1
            cell["baseline_right"] += before
            cell["with_review_right"] += after
            out["outputs"] += 1
            out["baseline_right"] += before
            out["with_review_right"] += after
            out["lost"] += before and not after
            if len(more) > len(cands) and old[0][0] != test["output"] and more[-1]["tests"][k] == test["output"]:
                out["answer_1_wrong_and_review_right"] += 1
                out["of_those_submitted"] += after
    out["baseline_score"] = 100 * out["baseline_right"] / out["outputs"]
    out["with_review_score"] = 100 * out["with_review_right"] / out["outputs"]
    out["gain_points"] = out["with_review_score"] - out["baseline_score"]
    return out


def submission(runs):
    """The competition's ``submission.json``: per task, per test input, two attempts: the two best guesses
    (``loop.guesses``), the one guess twice, or the input itself while there is none. Independent copies of a task
    (``copies``: entries ``id#k``) are merged: their candidates pooled, each augmented copy counted per entry, so two
    entries agreeing on a reading count as two copies."""
    pooled = {}
    for run in runs:
        entry = run.id.split("@")[0]
        task, cands = pooled.setdefault(entry.split("#")[0], (run.task, []))
        cands.extend({**c, "copy": (entry, c.get("copy", 0))} for c in run.candidates)
    out = {}
    for name, (task, cands) in pooled.items():
        picks = guesses(cands, [t["input"] for t in task["test"]])
        out[name] = [{"attempt_1": pick[0][0], "attempt_2": pick[-1][0]} for pick in picks]
    return out


async def run_arc2(study, trial, *, source, output, server_dir, base_url, deadline_epoch, python=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = study["arc2"]
    # The end drain and write-out need room inside the notebook limit: DRAIN_SECONDS + 10% (a quarter of what is left
    # in short runs). C5, 20 Oct: 180 s was not enough; the process limit cut the trial before tasks.jsonl was written.
    left = deadline_epoch - time.time()
    wall = min(float(trial["wall_seconds"]), left - min(1.1 * DRAIN_SECONDS, 0.25 * left))
    wall_clipped = wall < float(trial["wall_seconds"])
    wall_max = left - min(1.1 * DRAIN_SECONDS, 0.25 * left)  # the machine deadline less the drain: the live wall's cap
    control, control_seen, wall_changes = Path(os.environ.get(CONTROL_ENV) or output / CONTROL_FILE), [None], []
    started = time.monotonic()
    deadline = started + wall
    check = sandbox.self_test(python)
    if not check["ok"]:  # e.g. an environment the unprivileged uid cannot read: the looser worker, recorded
        check = {**sandbox.self_test(python, strict=False), "first": check}
    (output / "sandbox-check.json").write_text(json.dumps(check, indent=1))
    if not check["ok"]:
        return {"status": "error", "task_errors": ["sandbox self-test failed"], "sandbox": check["checks"]}
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], trial["tasks"])
    arms, chosen = trial.get("arms") or [{"name": "base"}], None
    if trial.get("arm_from"):
        chosen = winning_arm(study, output.parent, **trial["arm_from"])
        arms = [chosen["arm"]]
    copies = int(trial.get("copies", 1))  # independent copies of every task, as the c110 roster's clones
    entries = [(f"{i}#{k + 1}" if copies > 1 else i, i) for i in trial["tasks"] for k in range(copies)]
    if trial.get("copy_major"):  # every task's copy 1, then every copy 2, ...: a cut run keeps every task
        entries = [(f"{i}#{k + 1}", i) for k in range(copies) for i in trial["tasks"]]
    if trial.get("paired"):  # every task under every arm
        runs = [TaskRun(f"{name}@{arm['name']}", task_map[i], arm) for name, i in entries for arm in arms]
    else:
        runs = [TaskRun(name, task_map[i], arms[n % len(arms)]) for n, (name, i) in enumerate(entries)]
    if trial.get("rotate_copies"):  # copy k + 1 of a task starts on augmented view k (fixed-attempt designs: one
        for run in runs:            # attempt an entry, the views of a roster task's attempts 1, 2, ...)
            run.copy_offset = int(run.id.split("@")[0].split("#")[1]) - 1 if "#" in run.id else 0
    if trial.get("skip"):  # a resumed fixed-attempt trial: entries an earlier run finished are not played again
        skip = set(trial["skip"])
        runs = [run for run in runs if run.id not in skip]
    # ``twins`` 2 (arm setting): every task also gets a second run that starts on the next copy and works at the same
    # time; the two share candidates and replies (the read-out counts the first run only).
    sched = list(runs)
    if int((arms[0] or {}).get("twins", 1)) == 2:
        for run in runs:
            twin = TaskRun(run.id + "~2", run.task, run.arm)
            twin.candidates, twin.replies, twin.copy_offset = run.candidates, run.replies, 1
            run.shared = twin.shared = True
            sched.append(twin)
    # A play that continues an earlier one (arc2.kaggle after a server restart): what each task had, a minute old
    # at most. Its open conversation is gone: the next reply starts an attempt.
    state = Path(trial["state"]) if trial.get("state") else None
    if state is not None and state.is_file():
        saved = json.loads(state.read_text())
        for run in runs:
            for key in STATE_KEYS:
                if run.id in saved:
                    setattr(run, key, saved[run.id][key])
            conv = (saved.get(run.id) or {}).get("conv")
            if conv:  # ``conversation``: a submitted task stays closed; the rest go on numbering their submits
                run.resume = conv
                run.submits, run.first_submit_tokens = int(conv.get("submits") or 0), conv.get("first_submit_tokens")
                run.reopened, run.awaiting_reopen = bool(run.submits), False  # saved again as is if never started
                if conv.get("done") == "submitted":
                    run.done = "submitted"
    # Roster trial: ``marks`` are total completion tokens of one arm's roster; the arm stops at the largest.
    marks = {name: float(tokens) for name, tokens in (trial.get("marks") or {}).items()}
    stop, spent = max(marks.values(), default=None), {arm["name"]: 0 for arm in arms}
    budget = block_budget(server_dir, float(study["kv_margin"]), smallest=spec.get("cache_layout") == "smallest")
    window, cap = int(study["windows"][0]), int(study["completion_cap"])
    # A draft's lookahead comes off the top of the served context (27B with MTP 3 on a 65,536 context: the cost
    # table ends at 65,536 and a full window plus 3 is past it).
    window = min(window, int(budget.layout["max_model_len"]) - int(budget.layout["lookahead"]))
    if budget.cost(window) > budget.capacity:
        raise RuntimeError("The study window does not fit the served cache")
    admission = (output / "admission.jsonl").open("a", buffering=1)  # by the line: a cut run keeps its log
    capacity = budget.capacity
    if spec.get("gate_match"):  # another server's pressure: as many requests of that size as its gate held
        match = spec["gate_match"]
        capacity = min(capacity, round(float(match["requests"]) * budget.cost(int(match["tokens"]))))
    gate_log = lambda row: admission.write(json.dumps({"t": round(time.monotonic() - started, 2), **row}) + "\n")  # noqa: E731
    cache_admission = spec.get("admission") and spec["admission"].get("mode") == "kv"
    if cache_admission:  # arc2.admission: real KV use; the reserving rule above when /metrics goes unread
        from .admission import CacheGate
        from .admission import settings as admission_settings

        knobs = admission_settings(spec["admission"])
        poll = knobs.pop("poll")
        gate = CacheGate(capacity, slots=int(study["generation_slots"]), budget=budget, log=gate_log,
                         blocks=int(budget.layout["num_blocks"]), **knobs)
    elif spec.get("gate_overhang"):  # flag (default off): v87b's gate minus the measured overhang
        from .admission import OverhangGate

        over = {"window": 60.0, "poll": 2.0, **spec["gate_overhang"]}
        poll = float(over.pop("poll"))
        gate = OverhangGate(capacity, slots=int(study["generation_slots"]), budget=budget, log=gate_log,
                            blocks=int(budget.layout["num_blocks"]), window=float(over["window"]))
    else:
        gate = KVGate(capacity, slots=int(study["generation_slots"]), budget=budget, log=gate_log)
    events = (output / "events.jsonl").open("a")
    client = httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/", trust_env=False,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128, keepalive_expiry=KEEPALIVE_EXPIRY))
    engine_watchdog.start(base_url, server_dir)  # armed by arc2.kaggle (ARENA_ENGINE_WATCHDOG), else a no-op
    listing = await client.get("models", timeout=30)
    listing.raise_for_status()
    if trial.get("score_items"):  # arc2.score: likelihood of recorded readings, no play
        from .score import run_scores

        await client.aclose()
        return await run_scores({**trial, "score_source": source}, task_map, output, base_url, deadline,
                                model=listing.json()["data"][0]["id"], template=spec.get("chat_template_extra"))
    transport = Transport(client, gate, model=listing.json()["data"][0]["id"], effort=study["reasoning_effort"],
                          temperature=float(study["temperature"]), deadline=deadline, prefill=spec.get("prefill"),
                          hold=spec.get("hold_attempt_blocks"), grow_wait=spec.get("grow_wait", GROW_WAIT),
                          reserve=spec.get("reply_reserve"), template=spec.get("chat_template_extra"))
    if spec.get("served_log", True):  # every run reusable for SFT
        from .served import ServedLog

        transport.served = ServedLog(output, ids=spec.get("served_token_ids", True))
        manifest = Path(source) / "MANIFEST.json"
        meta = json.loads(manifest.read_text()) if manifest.is_file() else {}
        (output / "served-meta.json").write_text(json.dumps({
            "study": study.get("name"), "trial": trial.get("name"), "arms": arms, "settings": engine_settings_hint(arms),
            "bundle_source": meta.get("source"), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.is_file() else None, "files": {k: v for k, v in (meta.get("files") or {}).items()
                                                       if k.startswith(("src/arc2/", "configs/"))},
            **served_identity(listing.json()["data"][0])}, indent=1) + "\n")
    # C4 effort guard (20 Oct; "high" got 400s on 13 and 17 Oct): the served chat template renders every effort this
    # trial can send before any play
    for effort in efforts(study, arms):
        try:
            await transport.count([{"role": "user", "content": "effort check"}], {"effort": effort})
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"effort guard: the served chat template rejects reasoning_effort {effort!r}: "
                               f"{exc.response.text[:200]}") from exc
    served_before = await generated_tokens(client)
    runner = asyncio.Semaphore(int(spec.get("sandbox_processes", 8)))

    async def execute(code, inputs, **kw):
        async with runner:
            return await asyncio.to_thread(sandbox.run, code, inputs, python=python, strict=check["strict"], **kw)

    async def explore(cells, code, task, **kw):
        async with runner:
            return await asyncio.to_thread(sandbox.run_cells, cells, code, task, python=python, strict=check["strict"],
                                           **kw)

    def record(row):
        events.write(json.dumps({"t": round(time.monotonic() - started, 2), **row}) + "\n")
        events.flush()

    if trial.get("replay"):
        items = json.loads((Path(source) / trial["replay"]).read_text())
        rows = await review_replay(items, task_map, transport, execute, window=window, cap=cap, record=record,
                                   settings={**DEFAULTS, **(spec.get("settings") or {}), **arms[0]},
                                   workers=int(study["active_limit"]), deadline=deadline)
        served = await generated_tokens(client)
        await client.aclose()
        seconds, errors = time.monotonic() - started, [r["error"] for r in rows if r["outcome"] == "error"]
        tokens = sum(r["completion_tokens"] for r in rows)
        asked = sum(int(item.get("samples", 1)) for item in items)
        report = {"trial": trial["name"], "split": trial["split"], "arms": arms, "window": window,
                  "replay": replay_summary(rows, items), "samples_asked": asked, "replies": len(rows) - len(errors),
                  "request_errors": len(errors), "completion_tokens": tokens,
                  "prompt_tokens": sum(r["prompt_tokens"] for r in rows), "play_seconds": round(seconds, 1),
                  "tokens_per_second": tokens / max(1.0, seconds), "gate": gate.snapshot(),
                  "server_errors": transport.server_errors, "task_errors": errors[:5]}
        healthy = report["replies"] > 0 and len(errors) <= max(3, len(rows) // 20)
        report["status"] = "complete" if healthy else "error"
        if not healthy and not errors:
            report["task_errors"] = [f"{report['replies']} replies of {asked} samples"]
        if served is not None and served_before is not None:
            report["server_generated_tokens"] = served - served_before
        (output / "report.json").write_text(json.dumps(report, indent=1))
        admission.close()
        events.close()
        return report
    if any(a.get("conversation") for a in arms) or (spec.get("settings") or {}).get("conversation"):
        from .conv import (  # one conversation and one persistent interpreter per task
            ConvEngine,
            overwrite_marks,
        )
        from .repl import Repl

        engine = ConvEngine(complete=transport.complete, count=transport.count, execute=execute, window=window,
                            cap=cap, settings=spec.get("settings"), deadline=deadline, record=record,
                            drop=transport.drop,
                            repl=lambda run: Repl(getattr(run, "shown", None) or run.task, python=python,
                                                      strict=check["strict"], chars=int(
                                                          {**(spec.get("settings") or {}), **run.arm}.get("tool_chars")
                                                          or tasks.TOOL_CHARS),
                                                      memo=bool({**(spec.get("settings") or {}), **run.arm}.get("memo")),
                                                      one_answer=bool({**(spec.get("settings") or {}), **run.arm}.get("one_answer")),
                                                      programs=bool({**(spec.get("settings") or {}), **run.arm}.get("submit_check")),
                                                      import_globals=bool({**(spec.get("settings") or {}),
                                                                           **run.arm}.get("tool_fix")),
                                                      sandbox_fix=bool({**(spec.get("settings") or {}),
                                                                        **run.arm}.get("sandbox_fix")),
                                                      check_fn=bool({**(spec.get("settings") or {}),
                                                                     **run.arm}.get("check_fn")),
                                                      restore_names=bool({**(spec.get("settings") or {}),
                                                                          **run.arm}.get("restore_names", True))))
    else:
        engine = Engine(complete=transport.complete, count=transport.count, execute=execute, window=window, cap=cap,
                        settings=spec.get("settings"), deadline=deadline, record=record, explore=explore,
                        drop=transport.drop)
    library = Path(source) / spec["data"] / "library.json"
    if library.is_file():  # ``examples``: programs verified on training tasks only (checked when it was built)
        engine.library = json.loads(library.read_text())
    if trial.get("repair_seed"):  # repair replay: {entry: test outputs of a recorded verified program}; an entry
        # starts with that program as its one verified candidate (arm setting ``repair`` then gives one repair reply);
        # entries without a seed end before play
        seeds = json.loads((Path(source) / trial["repair_seed"]).read_text())
        for run in sched:
            tests = seeds.get(run.id.split("@")[0])
            if tests is None:
                run.done = "no_seed"
                continue
            n = len(run.task["train"])
            run.candidates.append({"code": "seed", "attempt": 0, "reply": 0, "tokens_at": 0, "t": 0.0, "passed": n,
                                   "pairs": n, "accuracy": 1.0, "verified": True, "tests": tests, "seed": True})
    if trial.get("pool_seed"):  # ``pool``: earlier programs per task, {task: [{view, code, passed, pairs, accuracy, ...}]}
        seed = json.loads((Path(source) / trial["pool_seed"]).read_text())
        engine.pools.update({("", task): rows for task, rows in seed.items()})
    tiers, sticky = [int(t) for t in trial.get("tiers") or []], bool(spec.get("sticky_attempts"))
    queue = asyncio.PriorityQueue()
    order = runs
    runs = sched  # the workers' list; ``order`` (one run a task) is what is read out
    base_len = len(runs)  # ``copy3`` appends third copies after these (read out apart: tasks.jsonl, the clock)
    fork_tokens = 0.0
    if trial.get("fork"):  # arc2.fork: a reopen variant starts from a base run's first passes
        from arc2 import fork as fork_mod
        fork_tokens = fork_mod.apply(engine, order, fork_mod.load(str(Path(source) / trial["fork"])))
        for name in spent:
            spent[name] += fork_tokens
    for n in range(len(runs)):  # a continued play (``state``) starts with the tasks that have had least
        if trial.get("fork"):
            if runs[n].done is None:
                queue.put_nowait(engine.queue_item(runs[n], n))
            continue
        queue.put_nowait((engine.priority(runs[n]) if marks else 0, n))
    if trial.get("fork") and marks and hasattr(engine, "attempts_due"):  # fork + b2: every entry has started (or
        for extra in engine.attempts_due(order, int(gate.slots)):  # ended in the base); attempts fill the slots
            runs.append(extra)
            queue.put_nowait(engine.queue_item(extra, len(runs) - 1))
    if fork_tokens:  # the base's first-pass tokens run the roster clock before every live reply (read out apart)
        base = TaskRun("fork-base", order[0].task, order[0].arm)
        base.spent, base.done, base.replies = fork_tokens, "fork_base", [
            {"completion_tokens": fork_tokens, "t0": time.monotonic() - 2.0, "t1": time.monotonic() - 1.0}]
        runs.append(base)
    failures = []
    stop_files, stopping = stop_paths(output), {}

    def stop_seen():
        if not stopping and any(path.exists() for path in stop_files):
            stopping.update(seen_seconds=round(time.monotonic() - started, 1), arm_tokens=dict(spent),
                            inflight=len(engine.inflight))  # events.jsonl stays replies only
        return bool(stopping)

    transport.stopped, transport.stop_flag = stop_seen, lambda: bool(stopping)

    tasks_out = (output / "tasks.jsonl").open("a")  # each task's row as it ends; rewritten whole at the end
    written = set()

    def task_row(run):
        return {"task": run.id, "arm": run.arm.get("name"), "spent": run.spent, "prompt_tokens": run.prompt_tokens,
                "attempts": run.attempts, "ended": run.done, "errors": run.errors, "candidates": run.candidates,
                "graded": graded(run), **({"scout": run.scout} if run.scout else {})}

    def cancelled():  # the wall's cancel can meet a request timeout at the same moment and arrive as TimeoutError
        task = asyncio.current_task()
        return bool(task is not None and getattr(task, "cancelling", lambda: 0)())

    idle = [0]  # workers waiting on the backlog (``multi_attempt``: free slots)
    # arc2.live_slots_max: workers above the starting slots, so control.json "slots" can raise the gate's ceiling live
    n_workers = max(int(study["active_limit"]), int((study.get("arc2") or {}).get("live_slots_max") or 0))

    async def worker():
        while True:
            if cancelled():  # that cancel was swallowed by ``except TimeoutError`` below: end now, not at the drain's
                return  # timeout (v51 preview: 300 s past the wall, the trial killed at wall + 120)
            idle[0] += 1
            try:
                tier, n = await queue.get()
            finally:
                idle[0] -= 1
            if n < 0:  # ``submit_mode`` "reopen": an anonymous token; the lowest-confidence submitted task now
                n = engine.pick_reopen(runs)
            try:
                if marks:  # one reply for the task that is next; no new reply once its arm has its tokens
                    if hasattr(engine, "holds_round") and engine.holds_round(runs[n], order):
                        await asyncio.sleep(2)  # ``rounds_after_all_submitted``: back to the backlog, the slot idles
                        queue.put_nowait(engine.queue_item(runs[n], n) if hasattr(engine, "queue_item")
                                         else (engine.priority(runs[n]), n))
                        continue
                    arm = runs[n].arm["name"]
                    while spent[arm] < stop and time.monotonic() < deadline and not stop_seen():
                        before, folds = runs[n].spent, getattr(runs[n], "compactions", 0)
                        await engine.advance(runs[n], float("inf"), replies=1)
                        spent[arm] += runs[n].spent - before
                        if engine.holds:  # ``conversation``: the task keeps its slot until it
                            # compacts (its prefix changes anyway) or ends; parked, it keeps its summary, recent tail
                            # and interpreter, and the slot takes the open task with the fewest tokens
                            compacted = getattr(runs[n], "compactions", 0) != folds and (
                                not hasattr(engine, "swaps_at_compaction") or engine.swaps_at_compaction(runs[n]))
                            if (runs[n].done is not None or compacted
                                    or getattr(runs[n], "parked", False)):  # and at a submit (P, R)
                                runs[n].parked = False
                                break
                            continue
                        # ``sticky_attempts`` (a small cache pool, four L4s): the attempt's next reply follows at
                        # once, while the blocks its last reply freed are still cached; in roster order a task
                        # waits minutes for its turn and every reply prefills the whole conversation again.
                        if not sticky or runs[n].done is not None or runs[n].messages is None:
                            break
                    if runs[n].done is None and spent[arm] < stop and time.monotonic() < deadline and not stop_seen():
                        queue.put_nowait(engine.queue_item(runs[n], n) if hasattr(engine, "queue_item")
                                         else (engine.priority(runs[n]), n))
                    elif runs[n].done is not None and hasattr(engine, "copy3_for"):
                        third = engine.copy3_for(runs[n], runs)
                        if third is not None:
                            runs.append(third)
                            queue.put_nowait((engine.priority(third), len(runs) - 1))
                    if hasattr(engine, "attempts_due") and not stop_seen() and spent[arm] < stop:  # b2
                        for extra in engine.attempts_due(order, idle[0] + 1 - queue.qsize() - (n_workers - gate.slots)):
                            runs.append(extra)
                            queue.put_nowait(engine.queue_item(extra, len(runs) - 1) if hasattr(engine, "queue_item")
                                             else (engine.priority(extra), len(runs) - 1))
                    if hasattr(engine, "seconds_due") and not stop_seen():  # ``second_attempt`` (B)
                        for second in engine.seconds_due(order):
                            runs.append(second)
                            queue.put_nowait(engine.queue_item(second, len(runs) - 1) if hasattr(engine, "queue_item")
                                             else (engine.priority(second), len(runs) - 1))
                    continue
                if stop_seen():
                    continue
                await engine.advance(runs[n], tiers[tier])
                if runs[n].done is None and time.monotonic() < deadline and not stop_seen():
                    if tier + 1 < len(tiers):
                        queue.put_nowait((tier + 1, n))
                    else:
                        runs[n].done = "budget"
            except (TimeoutError, httpx.TimeoutException):
                pass  # the wall
            except Exception as exc:  # noqa: BLE001 - one task's failure never ends the trial
                failures.append(f"{runs[n].id}: {type(exc).__name__}: {exc}"[:300])
                runs[n].done = "failed"
            finally:
                transport.drop(runs[n].id)  # the attempt ended, or the task leaves this worker
                if runs[n].done is not None and runs[n].id not in written:
                    written.add(runs[n].id)
                    tasks_out.write(json.dumps(task_row(runs[n])) + "\n")
                    tasks_out.flush()
                queue.task_done()

    live = os.environ.get("ARC_VALIDATION_OUTPUT")
    live = Path(live) / "arena" / "live" if live else None
    from .gpu import GpuSampler

    gpu = GpuSampler()
    gpu.mark("play")

    def cuts():  # per arm, the request time of the earliest reply still in flight
        arm_of = {run.id: run.arm.get("name") for run in runs}  # runs grows (copy3 thirds)
        times = {}
        for task, t in engine.inflight.items():
            times[arm_of[task]] = min(t, times.get(arm_of[task], t))
        return times

    def publish():
        if state is not None:
            tasks.write_json(state, {run.id: {**{key: getattr(run, key) for key in STATE_KEYS}, **({"conv": {
                key: getattr(run, key, None) for key in CONV_STATE_KEYS}} if hasattr(run, "submits") else {})}
                for run in order})
        if trial.get("submission"):  # the packaged run: the answers so far, every minute
            tasks.write_json(trial["submission"], {**json.loads(Path(trial["submission"]).read_text()), **submission(order)}
                       if Path(trial["submission"]).is_file() else submission(order))
        row = {"trial": trial["name"], "elapsed_seconds": round(time.monotonic() - started, 1), "wall_seconds": wall,
               "gate": gate.snapshot(), "failures": len(failures), "arm_tokens": spent, "gpu": gpu.summary(),
               "stop_request": stopping or None, "engine": engine_now.get("engine"),
               **({"overwrites": overwrite_marks(order, marks)} if marks and hasattr(engine, "queue_item") else {}),
               **summary(order, started, arms, marks, cuts(), extra=runs[base_len:])}
        text = json.dumps(row, indent=1)
        (output / "progress.json").write_text(text)
        if live is not None:
            live.mkdir(parents=True, exist_ok=True)
            (live / f"arc2-{trial['name']}.json").write_text(text)

    engine_now, engine_history = {}, []

    async def progress():
        while True:
            await asyncio.sleep(PROGRESS_SECONDS)
            engine_now["engine"] = await engine_metrics(client, engine_history)
            publish()

    async def kv_reader():  # the cache gate's readings
        history = []
        while True:
            gate.observe(await engine_metrics(client, history))
            await asyncio.sleep(poll)

    publish()
    workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
    ticker = asyncio.create_task(progress())
    if cache_admission or spec.get("gate_overhang"):
        workers.append(asyncio.create_task(kv_reader()))
    try:
        joined, end = asyncio.ensure_future(queue.join()), started + max(1.0, wall)
        while not joined.done():
            now = time.monotonic()
            read = control_read(control, control_seen)
            asked = control_wall(control, control_seen, read=read) if read is not None else None
            knobs = control_gate(gate, read, n_workers) if read is not None else {}
            if read is not None and marks and engine.holds and control_order(engine, read, queue, runs):
                knobs = {**knobs, "order": engine.live_order}
            if knobs:
                print(json.dumps({"event": "gate_changed", "at_seconds": round(now - started, 1), **knobs}), flush=True)
                gate_log({"event": "control", **knobs})
            if asked is not None and min(asked, wall_max) != wall:
                wall = min(asked, wall_max)
                deadline, end = started + wall, started + max(1.0, wall)
                transport.deadline = engine.deadline = deadline
                wall_changes.append({"at_seconds": round(now - started, 1), "asked": asked, "wall": round(wall, 1)})
                print(json.dumps({"event": "wall_changed", **wall_changes[-1]}), flush=True)
            if stop_seen() and "end_seconds" not in stopping:  # also when every worker is inside a long reply
                stopping["end_seconds"] = round(min(end, now + float(trial.get("stop_grace_seconds", STOP_GRACE)))
                                                - started, 1)
            limit = started + stopping["end_seconds"] if stopping else end
            if now >= limit:
                break
            await asyncio.wait({joined}, timeout=min(STOP_POLL, limit - now))
        if not joined.done():
            joined.cancel()
        if stopping:
            stopping.update(ended_seconds=round(time.monotonic() - started, 1), cut_inflight=len(engine.inflight),
                            sent_after=transport.sent_after_stop)
    except TimeoutError:
        pass
    for job in (*workers, ticker):
        job.cancel()
    try:  # a worker stuck in a cancelled call must not eat the write-out's time
        await asyncio.wait_for(asyncio.gather(*workers, ticker, return_exceptions=True), timeout=DRAIN_SECONDS / 2)
    except TimeoutError:
        pass
    tasks_out.close()
    for run in runs:  # ``conversation``: the persistent interpreters of tasks still open at the end
        if getattr(run, "interp", None) is not None:
            run.interp.close()
            run.interp = None
    served = await generated_tokens(client)
    await client.aclose()
    report = {"engine_watchdog": {key: engine_watchdog.STATE[key] for key in ("enabled", "restarts", "outage_seconds")},
              "trial": trial["name"], "split": trial["split"], "tiers": tiers, "window": window, "completion_cap": cap,
              "streams": int(study["active_limit"]), "settings": engine.settings, "arms": arms, "chosen_arm": chosen,
              "sandbox": {"strict": check["strict"], **check["checks"]}, "gate": gate.snapshot(), "gpu": gpu.summary(),
              "task_errors": failures, "server_errors": transport.server_errors, "marks": marks, "stop_tokens": stop, "copies": copies, "arm_tokens": spent,
              "stop_request": stopping or None,
              **summary(order, started, arms, marks, cuts(), trial.get("curve_step"), extra=runs[base_len:])}
    if marks and hasattr(engine, "queue_item"):
        report["overwrites"] = overwrite_marks(order, marks)
    report["wall"] = {"seconds": round(wall, 1), "clipped": wall_clipped, "asked": float(trial["wall_seconds"]),
                      "changes": wall_changes}
    with (output / "tasks.jsonl").open("w") as stream:
        for run in [*order, *runs[base_len:]]:
            stream.write(json.dumps(task_row(run)) + "\n")
    # Complete when the loop worked: replies came back, programs ran, few requests failed and, where the trial sets a
    # floor (the training sample), some tasks verified. A later trial with ``requires`` runs only then.
    verified = sum(any(c["verified"] for c in r.candidates) for r in order)
    closed = bool(order) and all(getattr(r, "resume", None) is not None and r.done == "submitted" for r in order)
    attempts_played = any(r.replies for r in runs[base_len:] if r.id != "fork-base")  # fk-b2: entries never reply
    healthy = ((report["replies"] > 0 or closed or attempts_played) and report["candidates"] > 0 and not failures
               and report["request_errors"] <= max(3, report["replies"] // 20)
               and verified >= int(trial.get("min_verified_tasks", 0)))
    report["status"] = "complete" if healthy else "error"
    if not healthy:
        report["task_errors"] = failures or [f"{report['replies']} replies, {report['candidates']} programs run, "
                                             f"{report['request_errors']} request errors, {verified} tasks verified"]
    if served is not None and served_before is not None:
        # Includes the replies cut at the wall, which no task was charged for.
        report["server_generated_tokens"] = served - served_before
        report["server_tokens_per_second"] = (served - served_before) / max(1.0, report["play_seconds"])
    (output / "report.json").write_text(json.dumps(report, indent=1))
    publish()
    admission.close()
    events.close()
    return report


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "winner":
        sys.exit(print(json.dumps(winner(json.loads(Path(sys.argv[2]).read_text())))))
    if len(sys.argv) == 5 and sys.argv[1] == "rescore":
        sys.exit(print(json.dumps(rescore(*sys.argv[2:]), indent=1)))
    sys.exit("arc2.trial runs through arena.run (trial kind arc2); `winner report.json` names the roster arm; "
             "`rescore <run's trial folder> <replay's trial folder> <task folder>` adds a review replay to a run")
