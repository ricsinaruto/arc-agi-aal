"""Setting ``conversation``: the ARC-AGI-2 harness the ARC-AGI-3 way.

One long conversation per roster entry, told the true aim (the test outputs, two guesses each), with one persistent
interpreter per task (``arc2.repl``), no call or program caps, and the answer given by ``submit(first, second)``, which
ends the task. A reply that ends with a ``transform`` program is verified on the training pairs as in ``arc2.loop``
(the exact differences); every program gets a number, is callable in the interpreter as ``program_N`` and its test
outputs join a ledger of the distinct readings, so later work sees them. At the context limit the model writes a
summary and the conversation continues from it plus the most recent part verbatim (``compact_keep`` of the window).
A task keeps its slot until it compacts or ends; a compacted task goes back to the backlog with its summary, recent
tail and interpreter, and a free slot takes the open task with the fewest tokens so far (as ARC-AGI-3). The model is
never shown a budget. Without a submit by the end, the guesses are ``loop.guesses``
over the conversation's programs.
"""

import asyncio
import hashlib
import json
import time

from . import tasks
from .loop import ContextFull, Engine, StopRequested

SUMMARY_TOKENS = 4096
CHARS_PER_TOKEN = 3.0  # picking whole messages for the kept tail only; the request itself is counted by the server


def approx_tokens(message):
    content = message.get("content")
    if isinstance(content, list):
        size = sum(len(p.get("text", "")) if p.get("type") == "text" else 1200 for p in content)
    else:
        size = len(content or "")
    size += len(message.get("reasoning") or "") + len(json.dumps(message.get("tool_calls") or ""))
    return size / CHARS_PER_TOKEN


def tail(messages, budget):
    """The most recent whole messages within ``budget`` tokens, never starting on a tool result whose call is cut."""
    kept, used = [], 0.0
    for message in reversed(messages):
        cost = approx_tokens(message)
        if kept and used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    while kept and kept[0]["role"] == "tool":
        kept.pop(0)
    return kept


class ConvEngine(Engine):
    holds = True

    def __init__(self, *, repl=None, **kw):
        super().__init__(**kw)
        self.repl = repl  # task -> arc2.repl.Repl (the trial's factory); None: no tool

    def phase(self, run):
        return "submitted" if run.done == "submitted" else "open"

    def priority(self, run):
        return run.spent

    def start(self, run, settings):
        run.attempts, run.attempt_replies = 1, 0
        run.view = run.shown = run.task
        run.programs, run.compactions = [], 0
        first = tasks.task_prompt(run.task)
        if settings["images"]:
            first = [{"type": "text", "text": first}] + tasks.image_parts(run.task, int(settings["image_side"]),
                                                                          bool(settings["image_changes"]))
        run.head = [{"role": "system", "content": tasks.CONV_SYSTEM}, {"role": "user", "content": first}]
        run.messages = list(run.head)
        if self.repl is not None and getattr(run, "interp", None) is None:
            run.interp = self.repl(run)

    async def tool(self, run, method, *args):
        return await asyncio.to_thread(getattr(run.interp, method), *args)

    def ledger(self, run):
        """The distinct test outputs of the verified programs, and which programs gave them."""
        readings = {}
        for c in run.candidates:
            if c.get("verified") and c.get("program"):
                readings.setdefault(json.dumps(c["tests"]), []).append(c["program"])
        if not readings:
            return "No program passes every training pair yet."
        lines = [f"Programs that pass every training pair give {len(readings)} distinct test output"
                 f"{'s' if len(readings) > 1 else ''}:"]
        for n, (_, programs) in enumerate(readings.items()):
            lines.append(f"- reading {chr(65 + n)}: program{'s' if len(programs) > 1 else ''} "
                         + ", ".join(str(p) for p in programs))
        return "\n".join(lines)

    async def compact(self, run, settings):
        """One tool-less summary reply, then head + summary + the most recent ``compact_keep`` of the window."""
        asked = run.messages + [{"role": "user", "content": tasks.CONV_COMPACT}]
        count = await self.count(asked, {**settings, "tool": False})
        if count + 512 > self.window:  # no room even to ask: drop the oldest half of the body first
            body = run.messages[len(run.head):]
            asked = run.head + tail(body, (self.window - 2 * SUMMARY_TOKENS) / 2) + [asked[-1]]
            count = await self.count(asked, {**settings, "tool": False})
        self.inflight[run.id] = time.monotonic()
        reply = await self.complete(run.id, asked, count, min(SUMMARY_TOKENS, self.window - count),
                                    {**settings, "tool": False, "effort": settings["summary_effort"]})
        self.inflight.pop(run.id, None)
        message, usage = reply["message"], reply.get("usage") or {}
        summary = (message.get("content") or "").strip() or (message.get("reasoning") or "")[-6000:]
        run.spent += int(usage.get("completion_tokens") or 0)
        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        kept = tail(run.messages[len(run.head):], float(settings["compact_keep"]) * self.window)
        run.messages = run.head + [{"role": "user", "content": tasks.CONV_RESUMED.format(summary=summary)
                                    + "\n\n" + self.ledger(run)}] + \
            ([{"role": "assistant", "content": "(continuing)"}] if kept and kept[0]["role"] == "user" else []) + kept
        run.compactions += 1
        self.keep(run, {"event": "reply", "task": run.id, "attempt": 1, "reply": len(run.replies) + 1,
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0), "tokens_at": run.spent,
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        "finish": reply.get("finish_reason"), "seconds": 0, "t0": reply.get("t0", 0),
                        "t1": reply.get("t1", 0), "phase": "open", "reasoning_chars": len(message.get("reasoning") or ""),
                        "content_chars": len(message.get("content") or ""), "outcome": "compaction",
                        "summary": summary[:4000], "kept_messages": len(kept)})

    async def advance(self, run, budget, replies=None):
        settings = {**self.settings, **run.arm}
        self.known[run.id] = run
        last = None if replies is None else len(run.replies) + replies
        while run.done is None and run.spent < budget and (last is None or len(run.replies) < last):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return
            if run.messages is None:
                self.start(run, settings)
            try:
                count = await self.count(run.messages, settings)
                if count + min(self.cap, int(settings["cap"] or self.cap)) + 256 > self.window:
                    await self.compact(run, settings)
                    continue
                cap = min(int(settings["cap"] or self.cap), self.window - count)
                self.inflight[run.id] = time.monotonic()
                reply = await self.complete(run.id, run.messages, count, cap,
                                            {**settings, "tool": self.repl is not None})
                self.inflight.pop(run.id, None)
            except asyncio.CancelledError:
                raise
            except StopRequested:
                self.inflight.pop(run.id, None)
                return
            except ContextFull:
                self.inflight.pop(run.id, None)
                await self.compact(run, settings)
                continue
            except Exception as exc:  # noqa: BLE001 - a failed request is retried on the same conversation
                self.inflight.pop(run.id, None)
                if self.deadline is not None and time.monotonic() >= self.deadline - 5:
                    return
                run.errors += 1
                self.record({"event": "request_error", "task": run.id, "error": f"{type(exc).__name__}: {exc}"[:400]})
                if run.errors >= settings["errors_per_task"]:
                    run.done = "errors"
                await asyncio.sleep(min(30, 2 ** run.errors))
                continue
            await self.take(run, reply, settings)
        if run.done is not None and getattr(run, "interp", None) is not None:
            await self.tool(run, "close")
            run.interp = None

    async def take(self, run, reply, settings):
        message, usage = reply["message"], reply.get("usage") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        run.spent += int(usage.get("completion_tokens") or 0)
        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        entry = {"role": "assistant", "content": content}
        if reasoning:
            entry["reasoning"] = reasoning
        run.messages.append(entry)
        t0, t1 = reply.get("t0", 0), reply.get("t1", 0)
        row = {"event": "reply", "task": run.id, "attempt": 1, "reply": len(run.replies) + 1,
               "completion_tokens": int(usage.get("completion_tokens") or 0),
               "prompt_tokens": int(usage.get("prompt_tokens") or 0), "tokens_at": run.spent,
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "finish": reply.get("finish_reason"), "seconds": round(t1 - t0, 2), "t0": t0, "t1": t1, "phase": "open",
               "reasoning_chars": len(reasoning), "content_chars": len(content), "content": content,
               "reasoning": reasoning}
        call = tasks.tool_call(message) if self.repl is not None else None
        if call is not None:
            kept, snippet = call
            entry["tool_calls"] = [kept]
            if snippet is None:
                text, result = "Error: the call needs one string argument `code`.", {}
            else:
                result = await self.tool(run, "run", snippet)
                text = tasks.tool_text(result, tasks.TOOL_CHARS)
                if result.get("notice"):
                    text = result["notice"] + "\n" + text
            submitted = result.get("submitted")
            if submitted is not None:
                problem = self.check_submit(run, submitted)
                if problem:
                    text = f"submit: {problem}"
                else:
                    self.take_submit(run, submitted)
                    text = "Submitted. The puzzle is finished."
                    row.update(submitted=True)
            run.messages.append({"role": "tool", "tool_call_id": kept["id"], "content": text})
            row.update(outcome="submit" if run.done == "submitted" else "tool", tool_code=snippet, tool_result=text)
            self.keep(run, row)
            return
        code = tasks.extract_code(content)
        if code is None:
            cut = reply.get("finish_reason") == "length"
            row["outcome"] = "truncated" if cut else "no_code"
            run.messages.append({"role": "user", "content": tasks.CONV_CUT if cut else tasks.CONV_NOTHING})
            self.keep(run, row)
            return
        n = len(run.programs) + 1
        run.programs.append(code)
        inputs = [p["input"] for p in run.task["train"]] + [t["input"] for t in run.task["test"]]
        result = await self.execute(code, inputs)
        verdict = tasks.judge(run.task, result)
        tests = verdict["tests"]
        earlier = {json.dumps(c["tests"]): c["program"] for c in run.candidates if c.get("verified") and c.get("program")}
        run.candidates.append({
            "code": hashlib.sha256(code.encode()).hexdigest()[:16], "attempt": 1, "reply": len(run.replies) + 1,
            "tokens_at": run.spent, "t": t1, "passed": verdict["passed"], "pairs": verdict["pairs"],
            "accuracy": verdict["accuracy"], "verified": verdict["verified"], "tests": tests, "program": n})
        if self.repl is not None and not result.get("error"):
            await self.tool(run, "define", f"program_{n}", code)
        if verdict["verified"]:
            same = earlier.get(json.dumps(tests))
            lines = [f"Program {n} passes every training pair (`program_{n}` in python)."]
            lines.append(f"Its test output is the same as program {same}'s." if same else
                         "Its test output is a new reading:\n" + "\n\n".join(
                             tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in tests))
            text = "\n".join(lines)
        else:
            text = f"Program {n}: " + tasks.feedback(run.task, result, verdict).replace(
                tasks.FIX_LINE, "Fix the rule and send the corrected program, or keep exploring in python.")
        text += "\n\n" + self.ledger(run)
        row.update(outcome="verified" if verdict["verified"] else "run_error" if result.get("error") else "wrong",
                   passed=verdict["passed"], pairs=verdict["pairs"], accuracy=round(verdict["accuracy"], 4), code=code,
                   program=n)
        run.messages.append({"role": "user", "content": text})
        self.keep(run, row)

    def check_submit(self, run, guesses):
        first = guesses[0]
        if first is None:
            return "the first guess is missing"
        for guess in (g for g in guesses if g is not None):
            if len(guess) != len(run.task["test"]) or not all(tasks.valid(g) for g in guess):
                return (f"each guess needs {len(run.task['test'])} valid grid(s) (lists of lists of ints 0-9, "
                        "rectangular, at most 30x30), one per test input. Nothing was submitted.")
        return None

    def take_submit(self, run, guesses):
        verified = {json.dumps(c["tests"]) for c in run.candidates if c.get("verified")}
        for rank, guess in enumerate(guesses, 1):
            if guess is None:
                continue
            run.candidates.append({"code": f"submit-{rank}", "attempt": 1, "reply": len(run.replies) + 1,
                                   "tokens_at": run.spent, "t": time.monotonic(), "passed": 0, "pairs": 0,
                                   "accuracy": 0.0, "verified": json.dumps(guess) in verified, "tests": guess,
                                   "submitted": rank})
        run.done = "submitted"
