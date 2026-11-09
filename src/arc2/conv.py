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

from . import multi, novelty, tasks
from .loop import SYMMETRIES, ContextFull, Engine, StopRequested, TaskRun, augment_back, augment_spec, augment_view

SUMMARY_TOKENS = 4096
# Churn rule (P's late collapse: replies 3,118 -> 291 tokens, submit turns x15): setting ``churn``.
# A task goes stale after CHURN_SAME repeats of the same submitted answer (three identical submits in a row) or
# CHURN_REPLIES replies in a row under CHURN_SHORT tokens without an answer change; stale, it ranks last when it
# next leaves its slot (no swap of its own: ); stale a second time, it is closed.
CHURN_SAME, CHURN_REPLIES, CHURN_SHORT, CHURN_CLOSE = 2, 8, 1000, 2
CHARS_PER_TOKEN = 3.0  # picking whole messages for the kept tail only; the request itself is counted by the server


def approx_tokens(message):
    content = message.get("content")
    if isinstance(content, list):
        size = sum(len(p.get("text", "")) if p.get("type") == "text" else 1200 for p in content)
    else:
        size = len(content or "")
    size += len(message.get("reasoning") or "") + len(json.dumps(message.get("tool_calls") or ""))
    return size / CHARS_PER_TOKEN


DRAIN_SPEC = {"drain_tokens": 56960, "turn_cap": 150, "turn_drain": 30, "coalesce": True, "note": True,
              "inventory_tokens": 1536, "receipts": 12}  # v8.5p (sabl-v85p-c100-300k-v1): drain_half + harness_fixes
INVENTORY_CHARS_PER_TOKEN = 3.5  # arena.harness.CHARS_PER_TOKEN (the inventory's character budget)


def progress(run):
    """The loop rule's action count: programs sent plus submits (an ARC-2 puzzle's "game actions")."""
    return len(getattr(run, "programs", None) or []) + int(getattr(run, "submits", 0) or 0)


def gate_rank(run, settings):
    """0 for a never-submitted entry (strictly first at the KV gate, ahead of reopen passes, resumed
    submitted holds and attempts), else 1. ``ns_first_gate`` off: 1 for all (FIFO)."""
    return 0 if settings.get("ns_first_gate") and not getattr(run, "submits", 0) \
        and not (getattr(run, "arm", None) or {}).get("attempt_k") else 1


def tail(messages, budget, clip_call=True):
    """The most recent whole messages within ``budget`` tokens, never starting on a tool result whose call is cut."""
    kept, used = [], 0.0
    for message in reversed(messages):
        cost = approx_tokens(message)
        if kept and used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    cut = len(messages) - len(kept)
    if clip_call and kept and kept[0]["role"] == "tool" and cut and messages[cut - 1].get("tool_calls"):
        # the call alone is past the budget (conv1 21897d95#1 kept nothing): keep it, its text cut to the room left
        kept.insert(0, clip(messages[cut - 1], max(0.0, budget - used)))
    while kept and kept[0]["role"] == "tool":
        kept.pop(0)
    return kept


def clip(message, budget):
    """``message`` with its reasoning and text content cut to their last ``budget`` tokens (tool calls kept)."""
    room = int(budget * CHARS_PER_TOKEN)
    out = dict(message)
    for key in ("reasoning", "content"):
        text = out.get(key)
        if isinstance(text, str) and len(text) > room:
            out[key] = "[... earlier text cut]\n" + text[len(text) - room:] if room else ""
        if isinstance(text, str):
            room = max(0, room - len(out[key]))
    return out


def view_of(run):
    """``copy_views`` (conv1 analysis, 28 Oct): copy k >= 2 of a task works on augmented view k - 1 (symmetry and
    colour map, ``loop.augment_view``); 0 = the task itself."""
    return int(getattr(run, "view_k", 0) or 0)


def to_task(run, grid):
    """A grid of the copy's view in the task's own frame (candidates and submits are stored in it)."""
    return augment_back(grid, run.id, view_of(run)) if view_of(run) and grid is not None else grid


def to_view(run, grid):
    """A grid in the task's frame shown in the copy's view."""
    if not view_of(run) or grid is None:
        return grid
    sym, colours = augment_spec(run.id, view_of(run))
    return [[colours[c] for c in row] for row in SYMMETRIES[sym][0](grid)]


def balance(run):
    """Balanced order, lowest first: tokens spent so far / tokens to the first submit. Tokens
    after the first submit go to tasks in proportion to how long they took to submit (harder, likelier wrong); at
    equal first-submit tokens it is fewest tokens first. Each turn raises the head's key, so no task holds the head."""
    return run.spent / max(1, getattr(run, "first_submit_tokens", 0) or run.spent or 1)


# ROUNDS (setting ``swap`` "budget", order "rounds"): round 1 runs to the first submit
# (compactions swap as before); each later round has a budget of the task's tokens-to-first-submit (``budget_floor``:
# at least one reply cap) and ends at that budget or when the same verified answer is submitted twice, not at a
# compaction. ``compact_on_swap``: a budget swap compacts first, so the next round re-enters short.
# ``force_submit_after`` N: a task with no submit after N compactions gets its ranked guesses as a provisional submit.
# (P4/R4: every later round ended at its first resubmit, 2 replies vs ~40K re-prefill):
# ``same_answer_scope`` "round": the identical-answer swap needs the same answer twice inside the current round (the
# round-1 submit does not count); ``budget_context``: a later round's budget is at least the context it resumes with;
# ``compacted_first``: a never-submitted task that compacts (round 1 ends) goes to the head of the backlog.


class ConvEngine(Engine):
    holds = True

    def __init__(self, *, repl=None, **kw):
        super().__init__(**kw)
        self.live_order = None  # control.json "order" (checklist item 5): overrides open_order and reopen_order
        self.repl = repl  # task -> arc2.repl.Repl (the trial's factory); None: no tool

    def phase(self, run):
        return "submitted" if getattr(run, "submits", 0) else "open"

    def priority(self, run):
        """Backlog order: tasks with no submit() first, then fewest tokens (``submit_mode`` "provisional");
        ``open_order`` "first_submit": submitted tasks by longest tokens-to-first-submit first.
        Stale tasks (``churn``) after all others."""
        settings = {**(self.settings or {}), **run.arm}
        rank = (1 if getattr(run, "submits", 0) else 0) + (2 if getattr(run, "stale", False) else 0)
        if (settings.get("third") or settings.get("sure")) and getattr(run, "submits", 0):  # base18/19: strict levels;
            if getattr(run, "select_due", None) is not None:  # a pass parked mid-way (a drain) goes first in its level
                return (3, run.spent)
            return (getattr(run, "reopen_count", 0), -1)
        if settings.get("midpass_first") and getattr(run, "submits", 0) and getattr(run, "reopened", False) \
                and not getattr(run, "awaiting_reopen", False) and not getattr(run, "stale", False):
            return (1, -1)  # a pass that left its slot (compaction) goes on before new reopen tokens ((1, 0), -1)
        order = self.live_order or settings.get("open_order")
        if order == "fewest":  # base15c: fewest tokens spent, submitted or not, as ARC-3; base17: within
            return (getattr(run, "try_n", 1) - 1, run.spent)  # a round, a parked try goes before any later try
        if order == "first_submit" and getattr(run, "submits", 0):
            return (rank, -(getattr(run, "first_submit_tokens", 0) or 0))
        if order == "balance" and getattr(run, "submits", 0):
            return (rank, balance(run))
        if order == "rounds" and getattr(run, "submits", 0):  # every task gets round k before any gets k + 1
            return (rank, getattr(run, "round", 1), -(getattr(run, "first_submit_tokens", 0) or 0))
        if settings.get("compacted_first") and not getattr(run, "submits", 0) and getattr(run, "compactions", 0) \
                and not getattr(run, "stale", False):  # picked up at the next free slot
            return (-1, getattr(run, "compacted_at", 0.0))
        return (rank, run.spent)

    def stall_end(self, run, settings):
        """Backstop (base15e240 cb2d8a2c#1: 3,149 replies after shadowing submit): an entry ends
        after ``stall_end`` replies in a row with no recorded submit and no program (base15e-family max 118)."""
        moved = run.replies[-1].get("outcome") in ("submit", "verified", "wrong") or run.replies[-1].get("submitted")
        run.stall_streak = 0 if moved else getattr(run, "stall_streak", 0) + 1
        if run.stall_streak < int(settings["stall_end"]):
            return False
        self.record({"event": "stall_end", "task": run.id, "spent": run.spent, "submits": run.submits,
                     "streak": run.stall_streak})
        if settings.get("fresh_tries"):
            self.new_try(run, settings, "stall")
            run.stall_streak = 0
        else:
            run.done = "stalled"
        return True

    def idle_end(self, run, settings):
        """Backstop (base15b's print('ok') entry): an entry with a submit ends after ``idle_end``
        replies in a row under ``idle_tokens`` completion tokens."""
        small = int(run.replies[-1].get("completion_tokens") or 0) < int(settings.get("idle_tokens") or 128)
        run.idle_streak = getattr(run, "idle_streak", 0) + 1 if small else 0
        if not (run.submits or settings.get("idle_any")) or run.idle_streak < int(settings["idle_end"]):
            return False
        self.record({"event": "idle_end", "task": run.id, "spent": run.spent, "submits": run.submits,
                     "streak": run.idle_streak})
        if settings.get("fresh_tries"):
            self.new_try(run, settings, "idle")
        else:
            run.done = "idle"
        return True

    def new_try(self, run, settings, cause):
        """base17: the try's submits stay with the task; the next try is blank (new conversation, new
        interpreter) and comes back through the backlog."""
        self.record({"event": "try_end", "task": run.id, "try": getattr(run, "try_n", 1), "cause": cause,
                     "spent": run.spent, "try_tokens": run.spent - getattr(run, "try_start", 0),
                     "submits": run.submits - getattr(run, "try_base", 0)})
        run.try_n, run.try_start, run.try_base = getattr(run, "try_n", 1) + 1, run.spent, run.submits
        run.idle_streak, run.parked = 0, True
        self.refresh(run, settings)

    def queue_item(self, run, n):
        """The backlog entry of a parked task: ``submit_mode`` "reopen" queues a submitted task as an anonymous reopen
        token (-1); the worker that draws it takes the lowest-confidence waiting task (``pick_reopen``)."""
        if getattr(run, "awaiting_reopen", False) and ({**(self.settings or {}), **run.arm}.get("third")
                                                       or {**(self.settings or {}), **run.arm}.get("sure")):
            return ((1 + getattr(run, "reopen_count", 0), 0), -1)  # base18: second passes before third passes
        return ((1, 0), -1) if getattr(run, "awaiting_reopen", False) else (self.priority(run), n)

    def confidence(self, run):
        """Higher = more likely right (conv1 traces, 27 Oct: both copies' answers agree 97% right when submitted by
        60K tokens; they differ 32-41%; submitted after 60K 44%)."""
        first = lambda r: next((c["tests"] for c in r.candidates if c.get("submitted") == 1
                                and c.get("submit_n") == r.submits), None)
        task, _, copy = run.id.split("@")[0].partition("#")
        partner = next((r for r in self.known.values() if r is not run and r.id.split("@")[0].partition("#")[0] == task
                        and getattr(r, "submits", 0)), None)
        agree = 0 if partner is None else (2 if first(partner) == first(run) else -2)
        late = -1 if (getattr(run, "first_submit_tokens", 0) or 0) > 60000 else 0
        return (agree + late, -(getattr(run, "first_submit_tokens", 0) or 0))

    def pick_reopen(self, runs):
        """The waiting submitted task with the lowest confidence; it gets the check pass now."""
        waiting = [n for n, r in enumerate(runs) if getattr(r, "awaiting_reopen", False)]
        order = self.live_order or {**(self.settings or {}), **runs[waiting[0]].arm}.get("reopen_order", "confidence")
        passes = lambda i: getattr(runs[i], "reopen_count", 0)  # repeated passes (reopen_passes 0): fewest first
        if order == "try_cap":  # no accepted submit first (fewest tries), then mean tokens to
            # submit across tries (a cap-ended try counts its whole length), longest first
            mean = lambda r: sum(r.try_tokens) / len(r.try_tokens) if getattr(r, "try_tokens", None) else 0  # noqa: E731
            n = min(waiting, key=lambda i: (bool(runs[i].submits), 0 if runs[i].submits else getattr(runs[i], "try_n", 1),
                                            -mean(runs[i])))
        elif order == "rounds":
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), getattr(runs[i], "round", 1),
                                            -(getattr(runs[i], "first_submit_tokens", 0) or 0)))
        elif order == "balance":  # fewest tokens and longest first submit, balanced
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), balance(runs[i])))
        elif order == "indep":  # ind20: the naive reopens as first_submit, then a class's next try
            # waits for the classes before it: fit with open answers, no fit, a repeated answer (``try_class``)
            level = lambda i: passes(i) + (self.try_class(runs[i]) if passes(i) else 0)  # noqa: E731
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), level(i),
                                            -(getattr(runs[i], "first_submit_tokens", 0) or 0)))
        elif order == "never_first":  # base13b: an entry with no accepted submit first, then as first_submit
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), bool(runs[i].submits), passes(i),
                                            -(getattr(runs[i], "first_submit_tokens", 0) or 0)))
        elif order == "first_submit_asc":  # base15: shortest tokens-to-first-submit first
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), passes(i),
                                            getattr(runs[i], "first_submit_tokens", 0) or 0))
        elif order == "first_submit":  # longest tokens-to-first-submit first (conv1: > 60K 44% all-right, < 15K 93%)
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), passes(i),
                                            -(getattr(runs[i], "first_submit_tokens", 0) or 0)))
        else:
            n = min(waiting, key=lambda i: (getattr(runs[i], "stale", False), passes(i), self.confidence(runs[i])))
        run = runs[n]
        run.awaiting_reopen, run.reopened = False, True
        run.reopen_count = getattr(run, "reopen_count", 0) + 1
        settings = {**(self.settings or {}), **run.arm}
        if settings.get("submit_mode") == "fresh":
            self.refresh(run, settings)
        elif not settings.get("round_prompt"):  # with ``round_prompt`` the round text comes in ``advance``
            run.messages.append({"role": "user", "content": self.reopen_text(run)})
        self.record({"event": "reopen", "task": run.id, "confidence": list(self.confidence(run)), "spent": run.spent,
                     "pass": run.reopen_count, "fresh": settings.get("submit_mode") == "fresh",
                     **({"try_class": self.try_class(run)} if order == "indep" else {})})
        return n

    def independent(self, run, settings):
        """ind20: the passes after the first ``indep_after`` reopens are independent tries (fresh2's)."""
        after = int(settings.get("indep_after") or 0)
        return bool(after) and getattr(run, "reopen_count", 0) > after

    def try_class(self, run):
        """ind20's in-run stand-in for fit-wrong / no-fit: 0 a train-fitting function was seen and no first guess was
        submitted twice; 1 none seen; 2 a first guess was submitted twice (base12c: 57 of 62 right; differing 19 of 40)."""
        firsts = [json.dumps(c["tests"]) for c in run.candidates if str(c.get("code", "")).startswith("submit-")
                  and str(c["code"]).endswith("-1")]
        if len(set(firsts)) < len(firsts):
            return 2
        return 0 if getattr(run, "fit_seen", False) else 1

    def pick_args(self, run, settings):
        """take_submit's keep_first and pick: an independent try's submit is fresh2's vote, keep_first off."""
        if self.independent(run, settings):
            return {"keep_first": False, "pick": "vote"}
        return {"keep_first": settings.get("keep_first"), "pick": settings.get("pick")}

    def answers18(self, run):
        """base18: the accepted answers in submit order (a1, a2, a3)."""
        subs = [c for c in run.candidates if c.get("submitted") and not c.get("vote")
                and str(c.get("code", "")).startswith("submit-")]
        return [c["tests"] for c in sorted(subs, key=lambda c: c.get("submit_at", c.get("submit_n")))]

    def same_run(self, run):
        """base19: how many of the latest accepted submits in a row give the same grids."""
        got = [json.dumps(a) for a in self.answers18(run)]
        n = 1
        while n < len(got) and got[-1 - n] == got[-1]:
            n += 1
        return n

    def b19_log(self, run, run_of):
        got = self.answers18(run)
        truth = [t.get("output") for t in run.task["test"]]
        self.record({"event": "b19_submit", "task": run.id, "submit": len(got), "pass": getattr(run, "reopen_count", 0),
                     "grids": got[-1], "right": [g == t if t is not None else None for g, t in zip(got[-1], truth)],
                     "same_run": run_of, "spent": run.spent})

    def pick18(self, run, second=None):
        """base18 guesses: guess 1 the first answer, guess 2 the selector's pick, else the second answer."""
        got = self.answers18(run)
        self.write_pick(run, [got[0], second if second is not None else got[1]], only=True)

    def select_ready(self, run):
        """base18: the outputs where the second and third answers differ each get a selector call."""
        a = self.answers18(run)
        run.select_due = [k for k in range(len(run.task["test"])) if json.dumps(a[1][k]) != json.dumps(a[2][k])]
        run.select_pick = [a[1][k] for k in range(len(run.task["test"]))]
        run.parked = True
        if not run.select_due:
            self.select_end(run)

    def select_end(self, run):
        self.pick18(run, run.select_pick)
        truth = [t.get("output") for t in run.task["test"]]
        a = self.answers18(run)
        self.record({"event": "b18", "task": run.id, "a1": a[0], "a2": a[1], "a3": a[2], "pick": run.select_pick,
                     "right": [[x[k] == truth[k] if truth[k] is not None else None for x in (a[0], a[1], a[2])]
                               for k in range(len(truth))],
                     "choices": getattr(run, "select_log", [])})
        run.select_due, run.done = None, "submitted"

    async def select(self, run, settings):
        """base18 selector: a fresh conversation (the usual system prompt and puzzle) per output, no tools, the two
        later answers in a fixed per-output order; ``PICK: A`` / ``PICK: B``; anything else keeps answer 2."""
        import hashlib
        import re
        if not run.select_due:
            self.select_end(run)
            return
        k = run.select_due[0]
        a = self.answers18(run)
        swap = hashlib.sha256(f"{run.id}|{k}".encode()).digest()[0] % 2 == 1  # half the outputs show answer 3 first
        first, second = (a[2][k], a[1][k]) if swap else (a[1][k], a[2][k])
        many = len(run.task["test"]) > 1
        ask = tasks.SELECT_ASK.format(which=f" for test input {k + 1}" if many else "",
                                      a=tasks.text(first) if tasks.valid(first) else tasks.describe(first),
                                      b=tasks.text(second) if tasks.valid(second) else tasks.describe(second))
        messages = list(run.head) + [{"role": "user", "content": ask}]
        ask_settings = {**settings, "tool": False, "gate_rank": gate_rank(run, settings), "reply_n": len(run.replies) + 1}
        count = await self.count(messages, ask_settings)
        cap = min(int(settings["cap"] or self.cap), self.window - count)
        self.inflight[run.id] = time.monotonic()
        try:
            reply = await self.complete(run.id, messages, count, cap, ask_settings)
        except asyncio.CancelledError:
            raise
        except StopRequested:
            self.inflight.pop(run.id, None)
            run.select_due = []  # stopping: answer 2 stays
            return
        except Exception as exc:  # noqa: BLE001 - one retry budget as a play request; then answer 2 stays
            self.inflight.pop(run.id, None)
            run.errors += 1
            self.record({"event": "request_error", "task": run.id, "error": f"{type(exc).__name__}: {exc}"[:400],
                         "selector": k})
            if run.errors >= settings["errors_per_task"]:
                run.select_due.pop(0)
            await asyncio.sleep(min(30, 2 ** run.errors))
            return
        self.inflight.pop(run.id, None)
        message, usage = reply["message"], reply.get("usage") or {}
        content = message.get("content") or ""
        found = re.findall(r"PICK:\s*\**\s*([AB])\b", content)
        letter = found[-1] if found else None
        chosen = None if letter is None else (first if letter == "A" else second)
        if chosen is not None:
            run.select_pick[k] = chosen
        run.select_due.pop(0)
        run.select_log = getattr(run, "select_log", []) + [{"output": k, "letter": letter, "third_first": swap,
                                                             "picked": None if chosen is None else (
                                                                 3 if json.dumps(chosen) == json.dumps(a[2][k]) else 2)}]
        run.spent += int(usage.get("completion_tokens") or 0)
        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.keep(run, {"event": "reply", "task": run.id, "attempt": 1, "reply": len(run.replies) + 1,
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0), "tokens_at": run.spent,
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        "finish": reply.get("finish_reason"), "seconds": round(reply.get("t1", 0) - reply.get("t0", 0), 2),
                        "t0": reply.get("t0", 0), "t1": reply.get("t1", 0), "phase": "selector", "outcome": "selector",
                        "selector_output": k, "letter": letter, "content": content[-2000:],
                        "reasoning_chars": len(message.get("reasoning") or ""), "content_chars": len(content)})
        if not run.select_due:
            self.select_end(run)

    def refresh(self, run, settings):
        """``submit_mode`` "fresh": the pass is a new conversation that shows only this copy's
        own last program and submitted answer (no other copy's), with the instruction to check and improve them.
        ``fresh_text`` "blank" (targets.md naive designs, "just doubling"): a completely fresh attempt that sees nothing
        of the earlier passes; keep_first keeps submission 1's first guess as guess 2."""
        run.pass_start = run.spent
        run.alt_asked = False
        if self.independent(run, settings):
            settings = {**settings, "fresh_text": "blank", "fresh_interp": True}
        if settings.get("try_cap"):
            run.try_n = getattr(run, "try_n", 1) + 1
        if settings.get("try_cap") and not run.submits:  # try_cap: no accepted submit yet, a blank try (new interpreter)
            if getattr(run, "interp", None) is not None:
                run.interp.close()
                run.interp = None
            self.begin(run, settings)
            if self.repl is not None:
                run.interp = self.repl(run)
            return
        run.note_asked = False
        if settings.get("try_notes"):  # notes1: a new try in a new interpreter
            # opens with the playbook of the earlier tries (their notes), never their programs or answers
            self.notes_head(run, settings)
            return
        if settings.get("fresh_text") == "blank":
            if settings.get("pass_clean"):  # base13b: no earlier program number, ledger line or receipt in this pass
                run.pass_base = len(run.programs)
                run.__dict__.pop("playbook_cut", None)
                run.__dict__.pop("playbook_due", None)
            if settings.get("fresh_interp") and not settings.get("fresh_view"):  # fresh2: a completely
                if getattr(run, "interp", None) is not None:  # fresh try: nothing carried over, not even the session
                    run.interp.close()
                    run.interp = None
                if self.repl is not None:
                    run.interp = self.repl(run)
            if settings.get("fresh_view"):  # naive5: pass k on augmented view k, in a new interpreter
                if getattr(run, "interp", None) is not None:
                    run.interp.close()
                    run.interp = None
                run.view_k = int(getattr(run, "reopen_count", 0) or 0)
                run.view = run.shown = augment_view(run.task, run.id, run.view_k) if run.view_k else run.task
                if self.repl is not None:
                    run.interp = self.repl(run)
            self.begin(run, settings)
            return
        sub = max((c for c in run.candidates if c.get("submitted") and not c.get("vote")),
                  key=lambda c: (c.get("submit_n") or 0, -(c.get("submitted") or 9)), default=None)
        guesses = sorted((c for c in run.candidates if c.get("submitted") and sub is not None
                          and c.get("submit_n") == sub.get("submit_n")), key=lambda c: c["submitted"])
        programs = [c for c in run.candidates if c.get("program")]
        prog = max((c for c in programs if c.get("verified")), key=lambda c: c.get("t", 0), default=None) or \
            max(programs, key=lambda c: c.get("t", 0), default=None)
        fmt = lambda grids: "\n\n".join(tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in grids)  # noqa: E731
        parts = [tasks.CONV_NAIVE if settings.get("fresh_text") in ("naive", "hyp") else tasks.CONV_FRESH]
        repair = getattr(run, "repair_source", None)
        if repair:  # rep1 (30 Oct): the pass after the first submit checks that answer's rule on the test input
            parts = [tasks.CONV_REPAIR.format(notes=novelty.report(getattr(run, "shown", None) or run.task)),
                     f"\n\nThe function:\n```python\n{repair}\n```"]
            run.repair_source = None
        elif prog is not None and 0 < prog["program"] <= len(run.programs):
            parts.append(f"\n\nYour last {'verified ' if prog.get('verified') else ''}program:\n```python\n"
                         f"{run.programs[prog['program'] - 1]}\n```")
        for c in guesses:
            parts.append(f"\n\nYour submitted {'first' if c['submitted'] == 1 else 'second'} guess:\n"
                         f"{fmt([to_view(run, g) for g in c['tests']])}")
        if settings.get("memo") and getattr(run, "interp", None) is not None:
            run.interp.inventory()
            book = str((run.interp.memo or {}).get("text") or "").strip()
            if book:
                parts.append("\n\n" + tasks.PLAYBOOK_SHOWN + book)
        if settings.get("try_cap"):  # the earlier guesses were wrong; a new, all-different submission
            parts.append(tasks.CONV_TRY_WRONG)
        alt = getattr(run, "alt_source", None)
        if alt:  # altR: the train-fitting function from the last pass, and the ask for a different fitting rule
            parts.append(f"\n\nA function that reproduces every training pair:\n```python\n{alt}\n```"
                         + tasks.CONV_ALT_REOPEN)
            run.alt_source = None
        if settings.get("fresh_text") == "hyp":  # hyp1 (29 Oct): this pass compares several readings before coding
            parts.append(tasks.CONV_HYP)
        self.begin(run, settings, "".join(parts))

    def notes_head(self, run, settings):
        """notes1: the earlier tries' playbook (read from the old interpreter, seeded into the new one) and a one-line
        ledger of the tries; no program, no answer grid."""
        book = ""
        if getattr(run, "interp", None) is not None:
            run.interp.inventory()
            book = str((run.interp.memo or {}).get("text") or "").strip()
            run.interp.close()
            run.interp = None
        if self.repl is not None:
            run.interp = self.repl(run)
            if book:
                run.interp.run(f"playbook({book!r})")
                run.interp.inventory()
        run.memo_base = int((getattr(run.interp, "memo", None) or {}).get("writes") or 0) if run.interp else 0
        fitted = any(c.get("verified") for c in run.candidates if c.get("submitted"))
        parts = [tasks.CONV_NOTES_HEAD.format(
            tries=run.submits, fitted="was" if fitted else "was not")]
        parts.append(("\n\n" + tasks.PLAYBOOK_SHOWN + book) if book else ("\n\n" + tasks.CONV_NOTES_NONE))
        self.begin(run, settings, "".join(parts))

    def noted(self, run):
        """notes1: the playbook was rewritten in this try."""
        if getattr(run, "interp", None) is None or run.interp.inventory() is None:
            return True  # no interpreter to ask: never hold a try open for its notes
        return int((run.interp.memo or {}).get("writes") or 0) > int(getattr(run, "memo_base", 0) or 0)

    def vote_pick(self, run):
        """fresh2 (``pick`` "vote"): the final two guesses per output from every submission's first guess: the most
        submitted grid (ties: the latest), then the most submitted different one, else the latest submit's own second
        guess. Written as candidates that rank above the latest submit (``loop.guesses``)."""
        firsts = sorted((c for c in run.candidates if str(c.get("code", "")).startswith("submit-")
                         and str(c["code"]).endswith("-1")), key=lambda c: c.get("submit_at", c.get("submit_n")) or 0)
        seconds = sorted((c for c in run.candidates if str(c.get("code", "")).startswith("submit-")
                          and str(c["code"]).endswith("-2")), key=lambda c: c.get("submit_at", c.get("submit_n")) or 0)
        n_out = len(run.task["test"])
        picks = [[], []]
        for k in range(n_out):
            grids = [json.dumps(c["tests"][k]) for c in firsts if c["tests"][k] is not None]
            count, last = {}, {}
            for i, g in enumerate(grids):
                count[g] = count.get(g, 0) + 1
                last[g] = i
            order = sorted(count, key=lambda g: (-count[g], -last[g]))
            if len(order) < 2:
                own = next((json.dumps(c["tests"][k]) for c in reversed(seconds) if c["tests"][k] is not None
                            and json.dumps(c["tests"][k]) not in order), None)
                order += [own] if own else []
            picks[0].append(json.loads(order[0]) if order else None)
            picks[1].append(json.loads(order[1]) if len(order) > 1 else None)
        self.write_pick(run, picks)

    def all_pick(self, run):
        """base13b (``pick`` "vote_all"): per output, the two grids in the most passes among all guesses of
        all passes; ties a grid that was a first guess over one that never was, then the oldest. Candidates as
        ``vote_pick``."""
        passes, firsts = {}, set()
        tries = {**(self.settings or {}), **run.arm}.get("fresh_tries")  # base17: a tie goes to a try's first submit
        for c in run.candidates:
            code = str(c.get("code", "")).split("-")
            if code[0] == "submit":
                passes.setdefault(int(code[1]), {})[int(code[2])] = c["tests"]
                if c.get("try_first") or not tries:
                    firsts.add(int(code[1]))
        picks = [[], []]
        for k in range(len(run.task["test"])):
            count, first, oldest = {}, set(), {}
            for n in sorted(passes):
                here = {rank: json.dumps(t[k]) for rank, t in passes[n].items() if t is not None and t[k] is not None}
                for grid in dict.fromkeys(here[rank] for rank in sorted(here)):  # a full tie: first guess first, not hash order
                    count[grid] = count.get(grid, 0) + 1
                    oldest.setdefault(grid, n)
                first.update({here[1]} if 1 in here and n in firsts else set())
            top = sorted(count, key=lambda g: (-count[g], g not in first, oldest[g]))
            picks[0].append(json.loads(top[0]) if top else None)
            picks[1].append(json.loads(top[1]) if len(top) > 1 else None)
        self.write_pick(run, picks)

    def write_pick(self, run, picks, only=False):
        verified = {json.dumps(c["tests"]) for c in run.candidates if c.get("verified")}
        now = time.monotonic()
        for rank, tests in enumerate(picks, 1):
            if all(t is None for t in tests):
                continue
            run.candidates.append({"code": f"vote-{run.submits}-{rank}", "attempt": 1, "reply": len(run.replies) + 1,
                                   "tokens_at": run.spent, "t": now, "passed": 0, "pairs": 0, "accuracy": 0.0,
                                   "tests": tests, "verified": json.dumps(tests) in verified, "vote": True,
                                   "submitted": rank, "submit_n": run.submits + 0.5, "submit_at": run.submits,
                                   **({"only": True} if only or {**(self.settings or {}), **run.arm}.get("fresh_tries")
                                      else {})})

    def begin(self, run, settings, prior=""):
        """A new conversation head: system prompt, the task (and ``prior``), images when set."""
        shown = getattr(run, "shown", None) or run.task
        first = tasks.task_prompt(shown) + prior
        if settings["images"]:
            first = [{"type": "text", "text": first}] + tasks.image_parts(shown, int(settings["image_side"]),
                                                                          bool(settings["image_changes"]))
        run.head = [{"role": "system", "content": tasks.conv_system(settings)}, {"role": "user", "content": first}]
        run.loop_detector = run.loop_fire = None  # the loop rule counts within one conversation
        run.sb = {"fails": 0, "since_program": 0, "same": {}, "last": None, "reason": None, "replies": 0}
        run.messages = list(run.head)

    def reopen_text(self, run):
        text = self.reopen_words(run)
        settings = {**(self.settings or {}), **run.arm}
        answers = self.answers18(run) if settings.get("reopen_picture") and settings.get("images") else None
        if answers:  # base20: each test input beside the latest submitted answer, after the ask
            return [{"type": "text", "text": text}] + tasks.answer_parts(
                getattr(run, "shown", None) or run.task, answers[-1], int(settings["image_side"]))
        return text

    def reopen_words(self, run):
        if {**(self.settings or {}), **run.arm}.get("strong_ask"):  # base19w256: base19's second-answer ask only
            return tasks.CONV_WRONG_STRONG
        if {**(self.settings or {}), **run.arm}.get("sure"):  # base19
            return tasks.CONV_WRONG_STRONG if getattr(run, "submits", 0) <= 1 else tasks.CONV_SURE
        if {**(self.settings or {}), **run.arm}.get("reopen_wrong"):  # base15
            return tasks.CONV_REOPEN_WRONG
        first = lambda r: next((c["tests"] for c in r.candidates if c.get("submitted") == 1
                                and c.get("submit_n") == r.submits), None)
        task = run.id.split("@")[0].partition("#")[0]
        partner = next((r for r in self.known.values() if r is not run and r.id.split("@")[0].partition("#")[0] == task
                        and getattr(r, "submits", 0)), None)
        other = first(partner) if partner is not None else None
        mine = first(run)
        extra = ""
        if other is not None and other != mine and {**(self.settings or {}), **run.arm}.get("partner_text", True):
            extra = ("\n\nAnother independent attempt at this puzzle submitted a different answer:\n" + "\n\n".join(
                tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in other))
        return tasks.CONV_REOPEN + extra

    def copy3_for(self, run, runs):
        """Arm setting ``copy3`` "split": once both copies of a task have ended, a third
        copy starts in a fresh conversation. On a task-hash half it is "informed" (it sees both copies' submitted
        answers and last verified programs), on the other half "blind" (the control). None: no third copy now."""
        if {**(self.settings or {}), **run.arm}.get("copy3") != "split" or run.done is None:
            return None
        base = run.id.split("@")[0].partition("#")[0]
        mates = [r for r in runs if r.id.split("@")[0].partition("#")[0] == base]
        if len(mates) != 2 or any(r.done is None for r in mates):
            return None
        informed = int(hashlib.sha1(base.encode()).hexdigest()[:8], 16) % 2 == 0
        third = TaskRun(f"{base}#3", run.task, {**run.arm, "copy3_mode": "informed" if informed else "blind"})
        if informed:
            parts = [tasks.CONV_PRIOR]
            for k, mate in enumerate(sorted(mates, key=lambda r: r.id), 1):
                sub = max((c for c in mate.candidates if c.get("submitted")), key=lambda c: c.get("submit_n") or 0,
                          default=None)
                prog = max((c for c in mate.candidates if c.get("verified") and c.get("program")),
                           key=lambda c: c.get("t", 0), default=None)
                answer = ("\n\n".join(tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in sub["tests"])
                          if sub and sub.get("tests") else "(no submit)")
                parts.append(f"\n\nAttempt {k} submitted:\n{answer}")
                if prog is not None and 0 < prog["program"] <= len(getattr(mate, "programs", None) or []):
                    # 28 Oct: was prog["code"], the program's 16-hex hash, not its source (copy3 / copy3f informed)
                    parts.append(f"\nIts last verified program:\n```python\n{mate.programs[prog['program'] - 1]}\n```")
            third.prior = "".join(parts)
        self.record({"event": "copy3", "task": third.id, "mode": third.arm["copy3_mode"]})
        return third

    def holds_round(self, run, order):
        """Setting ``rounds_after_all_submitted`` (p7: after 10M rounds >= 2 took 58-84% of the
        tokens while ~25 entries had not submitted): a submitted task's later round waits until every entry still
        playing has a first submit."""
        settings = {**(self.settings or {}), **run.arm}
        return bool(settings.get("rounds_after_all_submitted") and getattr(run, "submits", 0) and run.done is None
                    and any(not getattr(r, "submits", 0) and r.done is None for r in order))

    def attempts_due(self, order, n):
        """Setting ``multi_attempt`` (b2, ``arc2.multi``): once every entry has started, up to
        ``n`` (the free slots) new fresh attempts, each for the backlog entry ``multi.pick`` ranks first."""
        out = []
        settings = {**(self.settings or {}), **(order[0].arm if order else {})}
        if n <= 0 or not order or not settings.get("multi_attempt") \
                or any(r.messages is None and r.done is None for r in order):  # fork: base entries ended unplayed
            return out
        for _ in range(n):
            i = multi.pick([self.attempt_infos(r) for r in order], ns_first=bool(settings.get("multi_ns_first")))
            if i is None:
                break
            entry = order[i]
            runs = getattr(entry, "attempt_runs", None) or [entry]
            k = len(runs) + 1
            base, at, arm = entry.id.partition("@")
            name, _, copy = base.partition("#")
            attempt = TaskRun(f"{name}#b{k}x{copy or 1}{at}{arm}", entry.task, {**entry.arm, "attempt_k": k})
            attempt.entry, entry.attempt_runs = entry, runs + [attempt]
            self.record({"event": "attempt", "task": entry.id, "attempt": attempt.id, "k": k,
                         "key": multi.entry_key(self.attempt_infos(entry)[:-1])})
            out.append(attempt)
        return out

    @staticmethod
    def attempt_infos(entry):
        return [{"k": j + 1, "running": a.done is None, "submitted": bool(getattr(a, "submits", 0)),
                 "first_submit_tokens": getattr(a, "first_submit_tokens", None), "spent": a.spent}
                for j, a in enumerate(getattr(entry, "attempt_runs", None) or [entry])]

    def merge_attempts(self, run):
        """b2: the entry's guesses from all its attempts' latest first guesses (``multi.merge``), as a new merge
        generation that ranks above every earlier candidate (earlier generations stay for the matched-token marks)."""
        entry = getattr(run, "entry", None) or run
        attempts = getattr(entry, "attempt_runs", None) or [entry]
        if len(attempts) < 2:
            return
        def latest(r):
            return max((c for c in r.candidates if c.get("submitted") == 1 and not c.get("mg")),
                       key=lambda c: c.get("submit_n") or 0, default=None)
        answers = [{"tests": c["tests"], "verified": bool(c.get("verified"))} if c else {"tests": None}
                   for c in map(latest, attempts)]
        inputs = [t["input"] for t in entry.task["test"]]
        ranked = multi.merge(answers, len(inputs))
        if not any(ranked):
            return
        before = guesses_of(entry.candidates, inputs)
        gen = getattr(entry, "merge_gen", 0) + 1
        entry.merge_gen, now = gen, time.monotonic()
        for rank in (1, 2):
            tests = [r[rank - 1] if len(r) >= rank else None for r in ranked]
            if any(t is not None for t in tests):
                entry.candidates.append({"code": f"mg{gen}-{rank}", "mg": True, "attempt": 1, "reply": 0,
                                         "tokens_at": entry.spent, "t": now, "passed": 0, "pairs": 0, "accuracy": 0.0,
                                         "tests": tests, "verified": False, "submitted": rank, "submit_n": 10 ** 6 + gen})
        after = guesses_of(entry.candidates, inputs)
        truth = [t.get("output") for t in entry.task["test"]]
        row = {"event": "attempt_merge", "task": entry.id, "gen": gen, "attempts": len(attempts),
               "answered": sum(a["tests"] is not None for a in answers),
               "verified": sum(bool(a.get("verified")) for a in answers),
               "agree": [max((sum(1 for a in answers if a["tests"] is not None and a["tests"][i] == g) for g in r),
                             default=0) for i, r in enumerate(ranked)], "changed": [a != b for a, b in zip(after, before)]}
        if all(x is not None for x in truth):
            row["before"] = [any(g == x for g in b) for b, x in zip(before, truth)]
            row["after"] = [any(g == x for g in a) for a, x in zip(after, truth)]
        self.record(row)

    def seconds_due(self, order):
        """Setting ``second_attempt`` > 0: once every entry has started, an entry that
        submitted after more than that many tokens, or has not submitted after them, gets one independent second
        attempt: a fresh conversation that sees nothing of the first, spawned from the token budget (not the roster's
        other copy). Its answer joins the entry's guesses by ``merge_second``."""
        out = []
        if not order or any(r.messages is None for r in order):
            return out
        for run in order:
            over = int({**(self.settings or {}), **run.arm}.get("second_attempt") or 0)
            if not over or getattr(run, "pair", None) is not None or run.arm.get("b_attempt"):
                continue
            ttfs = getattr(run, "first_submit_tokens", None)
            if (run.done == "submitted" and (ttfs or 0) > over) or (
                    run.done is None and not getattr(run, "submits", 0) and run.spent > over):
                base, at, arm = run.id.partition("@")
                sid = (base.replace("#", "#b", 1) if "#" in base else base + "#b") + at + arm
                second = TaskRun(sid, run.task, {**run.arm, "b_attempt": True})
                run.pair, second.pair = second, run
                self.record({"event": "second_attempt", "task": run.id, "second": second.id, "ttfs": ttfs,
                             "spent": run.spent, "done": run.done})
                out.append(second)
        return out

    def merge_second(self, run):
        """B's selection, redone at every submit of either attempt: the second attempt's latest first guess joins the
        entry's candidates. Agree: nothing changes. Differ: it is guess 1 when it is verified and the entry's own
        first guess is not (or the entry has none), else guess 2 (ahead of the entry's own second guess)."""
        orig = run.pair if run.arm.get("b_attempt") else run
        sec = getattr(orig, "pair", None)
        if sec is None:
            return
        orig.candidates = [c for c in orig.candidates if not c.get("b2")]
        latest = lambda r: max((c for c in r.candidates if c.get("submitted") == 1 and not c.get("b2")),  # noqa: E731
                               key=lambda c: c.get("submit_n") or 0, default=None)
        s, a = latest(sec), latest(orig)
        if s is None or (a is not None and json.dumps(a["tests"]) == json.dumps(s["tests"])):
            self.record({"event": "second_merge", "task": orig.id, "agree": s is not None, "first": None})
            return
        lead = a is None or (s.get("verified") and not a.get("verified"))
        n = (a.get("submit_n") or 0) if a is not None else 0
        orig.candidates.append({**s, "code": f"b2-{s['code']}", "b2": True, "submitted": 1 if lead else 1.5,
                                "submit_n": n + 0.5 if lead else n})
        self.record({"event": "second_merge", "task": orig.id, "agree": False, "first": "second" if lead else "entry"})

    def start(self, run, settings):
        run.attempts, run.attempt_replies = 1, 0
        run.view = run.shown = run.task
        run.held, run.refusals, run.held_checks = None, {}, 0
        if settings.get("hold_one") is not None and len(run.task["train"]) >= 2:  # base16: one pair never served
            pairs, name = run.task["train"], run.id.split("@")[0].partition("#")[0]
            k = int(hashlib.sha256(f"{settings['hold_one']}|{name}".encode()).hexdigest(), 16) % len(pairs)
            run.held = pairs[k]
            run.view = run.shown = {**run.task, "train": pairs[:k] + pairs[k + 1:], "extra": [pairs[k]["input"]]}
            self.record({"event": "held_pair", "task": run.id, "index": k, "pairs": len(pairs)})
        copy = run.id.split("@")[0].rpartition("#")[2]
        run.view_k = int(copy) - 1 if settings.get("copy_views") and copy.isdigit() and int(copy) >= 2 else 0
        if settings.get("b_attempt") and settings.get("second_view"):  # B: the second attempt on augmented view 1
            run.view_k = 1
        if settings.get("attempt_k") and settings.get("multi_view"):  # b2: attempt k on augmented view k - 1
            run.view_k = int(settings["attempt_k"]) - 1
        if run.view_k:
            run.view = run.shown = augment_view(run.task, run.id, run.view_k)
        run.programs, run.compactions = [], 0
        run.submits, run.parked, run.bad_calls, run.overwrites = 0, False, 0, []
        run.awaiting_reopen, run.reopened, run.first_submit_tokens = False, False, None
        run.try_n, run.try_tokens, run.pass_start, run.try_refused, run.cap_ends = 1, [], run.spent, 0, 0
        resume = getattr(run, "resume", None)
        if resume:  # a continued play (trial ``state``): submit numbers go on (the latest still ranks first) and a
            run.resume = None  # task submitted before the break gets one more submit, which closes it
            run.submits = int(resume.get("submits") or 0)
            run.reopened = bool(run.submits)
            run.first_submit_tokens = resume.get("first_submit_tokens")
        self.begin(run, settings, getattr(run, "prior", None) or "")
        if self.repl is not None and getattr(run, "interp", None) is None:
            run.interp = self.repl(run)

    async def tool(self, run, method, *args):
        return await asyncio.to_thread(getattr(run.interp, method), *args)

    def ledger(self, run):
        """The distinct test outputs of the verified programs, and which programs gave them."""
        readings = {}
        base = getattr(run, "pass_base", 0)
        for c in run.candidates:
            if c.get("verified") and c.get("program") and c["program"] > base:
                readings.setdefault(json.dumps(c["tests"]), []).append(c["program"] - base)
        if not readings:
            return "No program passes every training pair yet."
        lines = [f"Programs that pass every training pair give {len(readings)} distinct test output"
                 f"{'s' if len(readings) > 1 else ''}:"]
        for n, (_, programs) in enumerate(readings.items()):
            lines.append(f"- reading {chr(65 + n)}: program{'s' if len(programs) > 1 else ''} "
                         + ", ".join(str(p) for p in programs))
        return "\n".join(lines)

    async def drain_note(self, run, messages, tokens, loop=None):
        """v8.5p's ``drain_note``: what was dropped (or the loop rule's line), that the interpreter survives, and the
        harness inventory of live definitions with the newest program results as its receipts."""
        from arena.context_drain import PYTHON_NOTE
        from arena.harness_fixes import render_inventory

        text = tasks.LOOP_NOTE.format(streak=loop["streak"]) if loop else tasks.DRAIN_NOTE.format(messages=messages,
                                                                                                    tokens=tokens)
        text += PYTHON_NOTE.lstrip() if loop else PYTHON_NOTE
        data = await self.tool(run, "inventory") if getattr(run, "interp", None) is not None else None
        if {**(self.settings or {}), **run.arm}.get("memo"):  # v8.5p memo_style 2: the playbook in full, no cap
            memo = getattr(run.interp, "memo", None) or {}
            book = str(memo.get("text") or "").strip()
            text += "\n\n" + (tasks.PLAYBOOK_SHOWN + book if book else tasks.PLAYBOOK_EMPTY)
            run.playbook_cut = (getattr(run, "drains", 0) + getattr(run, "loop_clears", 0) + 1,
                                int(memo.get("writes") or 0))  # writes when the next fill begins
            self.record({"event": "playbook_shown", "task": run.id, "chars": len(book),
                         "writes": int(memo.get("writes") or 0)})
        base = getattr(run, "pass_base", 0)
        receipts = [f"- program {c['program'] - base}: {c['passed']}/{c['pairs']} training pairs"
                    + (" (passes all)" if c.get("verified") else "")
                    for c in run.candidates if c.get("program") and c["program"] > base][-DRAIN_SPEC["receipts"]:]
        text += "\n\n" + render_inventory(data, receipts, max_tokens=DRAIN_SPEC["inventory_tokens"],
                                           chars_per_token=INVENTORY_CHARS_PER_TOKEN)
        return {"role": "user", "content": text}

    def drain_trigger(self, run, jitter):
        """v52 drain jitter (port of v84-sub 77e18eef0, arena.context_drain.trigger_draw): the trigger of this
        conversation's next drain (index = drains so far), uniform integer in [low, high], a pure function of
        sha256(seed|task|index); drawn once per index and logged (``drain_trigger``)."""
        import hashlib

        index = getattr(run, "drains", 0)
        triggers = run.__dict__.setdefault("drain_triggers", [])
        if len(triggers) <= index:
            low, high = int(jitter["low"]), int(jitter["high"])
            digest = hashlib.sha256(f"{int(jitter.get('seed', 0))}|{run.id}|{index}".encode()).digest()
            triggers.append(low + int.from_bytes(digest[:8], "big") * (high - low + 1) // 2 ** 64)
            self.record({"event": "drain_trigger", "task": run.id, "drain_index": index,
                         "trigger_tokens": triggers[index], "keep_tokens": jitter.get("keep_tokens")})
        return triggers[index]

    async def drain(self, run, settings, count):
        """naive12: ``arena.context_drain.drain_history`` on the messages after the head
        (system prompt and the puzzle, never dropped, as v8.5p's system prompt): the oldest whole blocks go until the
        request is ``drain_tokens`` below the budget, reasoning verbatim, the latest user message and what follows it
        kept; the turn cap (150 -> 120 replies); a note in front. A pending loop-rule fire clears the history instead.
        Returns the request's new prompt tokens."""
        from arena.context_drain import drain_history

        spec = {**DRAIN_SPEC, **(settings.get("drain") or {})}
        head, history = run.head, run.messages[len(run.head):]
        fire = getattr(run, "loop_fire", None)
        if fire is not None:
            run.loop_fire = None
            if history:
                note = await self.drain_note(run, len(history), 0, loop=fire)
                run.messages = head + [note]
                after = await self.count(run.messages, settings)
                run.loop_clears = getattr(run, "loop_clears", 0) + 1
                self.record({"event": "loop_rule_fresh_context", "task": run.id, "streak": fire["streak"],
                             "reason": fire["reason"], "tokens_before": int(count), "tokens_after": int(after),
                             "messages_dropped": len(history)})
                return after
        budget = self.window - min(self.cap, int(settings["cap"] or self.cap)) - 256
        jitter, trigger = spec.get("jitter"), None
        if jitter:  # v52: this drain's own trigger in place of the window's budget (never above it); keep_tokens kept
            trigger = self.drain_trigger(run, {**jitter, "keep_tokens": spec.get("keep_tokens")})
            budget = max(1, min(budget, trigger))
        # v8.5p's unit is a turn (an observation and the replies after it); here an exchange is one reply and its tool
        # result or feedback. drain_history runs on proxies with an exchange's reply as the "user" message, so blocks
        # are whole exchanges (a call never loses its result) and the protected "observation" is the latest one.
        proxies = [{"role": "user" if m.get("role") == "assistant" else "assistant", "i": i}
                   for i, m in enumerate(history)]

        def estimate(items):
            return sum(approx_tokens(history[p["i"]]) for p in items)

        kept_p, info = drain_history(proxies, count=count, budget=budget, estimate=estimate, spec=spec)
        if info is None:
            if settings.get("memo") and count >= 0.8 * budget and history and getattr(run, "interp", None) is not None:
                fill = getattr(run, "drains", 0) + getattr(run, "loop_clears", 0)
                if getattr(run, "playbook_due", None) != fill:
                    await self.tool(run, "inventory")
                    writes = int((getattr(run.interp, "memo", None) or {}).get("writes") or 0)
                    since = getattr(run, "playbook_cut", (fill, 0))
                    last = history[-1].get("content")
                    if writes == (since[1] if since[0] == fill else 0) and history[-1].get("role") in ("tool", "user") \
                            and (isinstance(last, str) or isinstance(last, list) and settings.get("reopen_picture")):
                        run.playbook_due = fill
                        if isinstance(last, list):  # base20's reopen message with its pictures
                            history[-1]["content"] = last + [{"type": "text", "text": "\n\n" + tasks.PLAYBOOK_OPEN_DUE}]
                        else:
                            history[-1]["content"] += "\n\n" + tasks.PLAYBOOK_OPEN_DUE
                        self.record({"event": "playbook_due", "task": run.id, "prompt_tokens": int(count)})
                        return await self.count(run.messages, settings)
            return count
        kept = [history[p["i"]] for p in kept_p]
        note = await self.drain_note(run, info["messages_dropped"], max(0, estimate(proxies) - estimate(kept_p))) \
            if spec["note"] else None
        run.messages = head + ([note] if note else []) + kept
        after, rounds = await self.count(run.messages, settings), 1
        while after > budget and rounds < 4:  # v8.5p: drain again from the measure, the estimate rescaled
            previous = kept_p
            measured, estimated = count - after, estimate(proxies) - estimate(previous)
            scale = measured / estimated if measured > 0 and estimated > 0 else 1.0
            kept_p, extra = drain_history(previous, count=after, budget=budget, spec=spec, force=True,
                                          estimate=lambda items, scale=scale: int(estimate(items) * scale))
            if extra is None:
                break
            kept = [history[p["i"]] for p in kept_p]
            run.messages, rounds = head + ([note] if note else []) + kept, rounds + 1
            info["messages_dropped"] += extra["messages_dropped"]
            info["turn_exchanges"] += extra["turn_exchanges"]
            after = await self.count(run.messages, settings)
        run.drains = getattr(run, "drains", 0) + 1
        self.record({"event": "context_drain", "task": run.id, "tokens_before": int(count), "tokens_after": int(after),
                     "rounds": rounds, "note_chars": len(note["content"]) if note else 0,
                     **({"drain_index": run.drains - 1, "trigger_tokens": trigger} if trigger is not None else {}),
                     **{k: v for k, v in info.items() if k != "prompt_tokens"}})
        return after

    async def compact(self, run, settings):
        """One tool-less summary reply, then head + summary + the most recent ``compact_keep`` of the window."""
        prefill = settings.get("summary_prefill")
        # v8.5p's "prefilled-empty" (arena.compaction): the play request (tools rendered, play effort) plus an assistant
        # turn after the template's empty think block, continued: the summary reuses the cached prefix, no reasoning.
        # Without it: a tool-less request at summary_effort (the system block changes, so all of it re-prefills).
        ask = {**settings, "continue": True, "tool_choice": "none"} if prefill else \
            {**settings, "tool": False, "effort": settings["summary_effort"]}
        ask["gate_rank"] = gate_rank(run, settings)
        suffix = [{"role": "user", "content": tasks.CONV_COMPACT}] + \
            ([{"role": "assistant", "content": prefill}] if prefill else [])
        asked = run.messages + suffix
        count = await self.count(asked, ask)
        if count + 512 > self.window:  # no room even to ask: drop the oldest half of the body first
            body = run.messages[len(run.head):]
            asked = run.head + tail(body, (self.window - 2 * SUMMARY_TOKENS) / 2, settings.get("tail_clip", True)) + suffix
            count = await self.count(asked, ask)
        self.inflight[run.id] = time.monotonic()
        reply = await self.complete(run.id, asked, count, min(SUMMARY_TOKENS, self.window - count), ask)
        self.inflight.pop(run.id, None)
        message, usage = reply["message"], reply.get("usage") or {}
        if prefill:  # the parser may return the continuation as reasoning (no </think> in the output)
            parts = []
            for key in ("reasoning_content", "reasoning", "content"):
                part = (message.get(key) or "").strip() if isinstance(message.get(key), str) else ""
                if part and part not in parts:
                    parts.append(part)
            summary = (prefill + " " + "\n".join(parts))[:24000] if parts else ""
        else:
            summary = (message.get("content") or "").strip() or (message.get("reasoning") or "")[-6000:]
        run.spent += int(usage.get("completion_tokens") or 0)
        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        kept = tail(run.messages[len(run.head):], float(settings["compact_keep"]) * self.window,
                    settings.get("tail_clip", True))
        run.messages = run.head + [{"role": "user", "content": tasks.CONV_RESUMED.format(summary=summary)
                                    + "\n\n" + self.ledger(run)}] + \
            ([{"role": "assistant", "content": "(continuing)"}] if kept and kept[0]["role"] == "user" else []) + kept
        run.compactions += 1
        run.compacted_at = time.monotonic()  # ``compacted_first``: first compacted, first resumed
        after = int(settings.get("force_submit_after") or 0)
        if after and not run.submits and run.compactions >= after:  # coordinator (c): a provisional answer now
            picks = guesses_of(run.candidates, [t["input"] for t in run.task["test"]])
            if picks and all(picks):
                self.take_submit(run, [[p[0] for p in picks], [p[-1] for p in picks]])
                self.record({"event": "forced_submit", "task": run.id, "compactions": run.compactions})
                if settings.get("swap") == "budget":
                    self.end_round(run, "forced_submit")
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
            if getattr(run, "select_due", None) is not None:  # base18: one selector call per output
                await self.select(run, settings)
                continue
            if run.messages is None:
                self.start(run, settings)
            if settings.get("round_prompt") and getattr(run, "submits", 0) and getattr(run, "round", 1) >= 2 \
                    and getattr(run, "prompted_round", None) != run.round:
                run.messages.append({"role": "user", "content": self.round_text(run, settings)})
                run.prompted_round = run.round
                self.record({"event": "round_prompt", "task": run.id, "round": run.round, "spent": run.spent,
                             "context": getattr(run, "round_context", None),
                             "budget": self.round_budget(run, settings)})
            try:
                if settings.get("resolve_cap") and getattr(run, "submits", 0):  # cap1 (30 Oct): re-solves keep the
                    settings = {**settings, "cap": int(settings["resolve_cap"])}  # base reply cap, pass 1 a larger one
                count = await self.count(run.messages, settings)
                if settings.get("context_mode") == "drain":  # naive12: v8.5p's drain_half instead of a summary
                    fill = getattr(run, "drains", 0) + getattr(run, "loop_clears", 0)
                    count = await self.drain(run, settings, count)
                    if settings.get("park_drain") and getattr(run, "drains", 0) + getattr(run, "loop_clears", 0) != fill:
                        # base15c: the prefix changed anyway; the slot takes the fewest-tokens entry
                        run.parked = True
                        self.record({"event": "park", "task": run.id, "spent": run.spent, "submits": run.submits})
                        return
                if settings.get("context_mode") == "cap":  # try_cap: the context is the try's cap, never compacted
                    if self.window - count - 256 < int(settings.get("try_min_reply") or 1024):
                        self.end_try(run, settings, "cap", count)
                        break
                elif count + min(self.cap, int(settings["cap"] or self.cap)) + 256 > self.window:
                    await self.compact(run, settings)
                    continue
                cap = min(int(settings["cap"] or self.cap), self.window - count - (256 if settings.get("context_mode") == "cap" else 0))
                self.inflight[run.id] = time.monotonic()
                lean = settings.get("resolve_effort") if getattr(run, "submits", 0) else None  # lean (30 Oct): re-solves
                under = settings.get("resolve_effort_under")  # leanS: only for entries whose first submit came early
                if lean and under and (getattr(run, "first_submit_tokens", None) or 0) >= int(under):
                    lean = None
                reply = await self.complete(run.id, run.messages, count, cap,  # after the first submit think less
                                            {**settings, "tool": self.repl is not None, "gate_rank": gate_rank(run, settings),
                                             "reply_n": len(run.replies) + 1, **({"effort": lean} if lean else {}),
                                             **({"temperature": float(settings["resolve_temperature"])}  # t10r
                                                if settings.get("resolve_temperature") and getattr(run, "submits", 0)
                                                else {})})
                self.inflight.pop(run.id, None)
            except asyncio.CancelledError:
                raise
            except StopRequested:
                self.inflight.pop(run.id, None)
                return
            except ContextFull:
                self.inflight.pop(run.id, None)
                if settings.get("context_mode") == "cap":
                    self.end_try(run, settings, "cap", None)
                    break
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
            if getattr(run, "try_over", None):
                self.new_try(run, settings, run.try_over)
                run.try_over = None
                return
            if settings.get("loop_exit") and self.loop_exit(run, settings):
                break
            if settings.get("idle_end") and self.idle_end(run, settings):
                break
            if settings.get("stall_end") and self.stall_end(run, settings):
                break
            if settings.get("step_back"):
                self.step_back(run)
            self.churn_check(run, settings)
            self.pass_check(run, settings)
            await self.budget_check(run, settings)
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
        if reply.get("think_cut"):  # think_cap: the reply went on past the cap in the same turn
            row.update(think_cut=reply["think_cut"], think_rest_cached=usage.get("think_rest_cached"),
                       think_rest_prompt=usage.get("think_rest_prompt"))
        call = tasks.tool_call(message) if self.repl is not None else None
        if settings.get("loop_rule"):  # v8.5p loop_rule (arena.loop_rule.LoopDetector) on progress, not game actions
            from arena.loop_rule import DEFAULTS as LOOP_DEFAULTS, LoopDetector
            if getattr(run, "loop_detector", None) is None:
                run.loop_detector = LoopDetector({**LOOP_DEFAULTS, **(settings["loop_rule"] if isinstance(
                    settings["loop_rule"], dict) else {})})
            fire = run.loop_detector.note(progress(run), usage.get("completion_tokens"), message)
            if fire is not None:
                run.loop_fire = fire
                row.update(loop_rule=fire["reason"])
        if call is not None and settings.get("drop_malformed"):
            from arena.context_drain import MALFORMED_CALL_FOLLOWUP, TRUNCATED_CALL_FOLLOWUP, malformed_calls
            if malformed_calls(message):
                cut = reply.get("finish_reason") == "length"
                run.messages.append({"role": "user", "content": (TRUNCATED_CALL_FOLLOWUP if cut
                                                                 else MALFORMED_CALL_FOLLOWUP).strip()})
                run.bad_calls += 1
                row.update(outcome="malformed_dropped", bad_args=call[0]["function"]["arguments"][:2000])
                self.keep(run, row)
                return
        if call is not None:
            kept, snippet = call
            if snippet is None and settings.get("sandbox_fix") and reply.get("raw"):  # base20: the parser lost it
                snippet = tasks.salvage_code(reply["raw"])
                if snippet is not None:
                    row.update(salvaged=True, bad_args=kept["function"]["arguments"][:2000])
                    kept = {**kept, "function": {**kept["function"], "arguments": json.dumps({"code": snippet})}}
            entry["tool_calls"] = [kept]
            if snippet is None:  # conv1: unparseable arguments looped 200-1,180 times in 4 entries
                raw = kept["function"]["arguments"]
                run.bad_calls += 1
                row.update(bad_args=raw[:2000])
                if not settings.get("bad_call_fix", True):  # conv1's text, arguments kept as sent
                    text, result = "Error: the call needs one string argument `code`.", {}
                else:
                    text, result = (f"Error: the call needs one string argument `code`; received {raw[:300]!r}.", {})
                    args = json.loads(raw) if _json_ok(raw) else None
                    if settings.get("tool_fix") and args is not None and not str(args.get("code") or "").strip():
                        text = tasks.CONV_EMPTY_CALL.format(raw=raw[:120])
                        row.update(empty_call=True)
                    if run.bad_calls >= 3:
                        text += " " + tasks.CONV_BAD_CALLS
                    kept["function"]["arguments"] = json.dumps({"code": ""})
            else:
                run.bad_calls = 0
                result = await self.tool(run, "run", snippet)
                text = tasks.tool_text(result, int(settings.get("tool_chars") or tasks.TOOL_CHARS))
                if result.get("restored"):  # restore_names: the call rebound a harness name
                    row.update(restored=result["restored"])
                if result.get("notice"):
                    text = result["notice"] + "\n" + text
            submitted = result.get("submitted")
            if submitted is None and snippet is not None and settings.get("alt_fit") == "pass" \
                    and not getattr(run, "alt_asked", False):  # altP (30 Oct): at the pass's first train-fit
                fits = await self.fits(run, row)
                if fits:
                    text += tasks.CONV_ALT_NOW.format(name=fits[0]["name"])
                    run.alt_asked = True
                    row.update(alt_asked=True)
            if submitted is not None and settings.get("aug_check") and result.get("submit_check") is not None \
                    and not getattr(run, "uncertain", False):  # chk5: a failed transform marks the entry uncertain
                aug = await self.tool(run, "aug", list(settings["aug_check"]))
                failed = sorted(k for k, v in (aug or {}).items() if v is False)
                row.update(aug_failed=failed, aug_checked=len(aug or {}))
                run.uncertain = bool(failed)
            if submitted is not None:
                problem = self.submit_problem(run, result, settings, row) or self.check_submit(run, submitted) or (
                    self.try_problem(run, submitted, row) if settings.get("try_cap") else None) or (
                    self.repeat_problem(run, submitted, row) if (settings.get("no_repeat") or settings.get("second_differs"))
                    and run.submits > getattr(run, "try_base", 0)
                    and not (settings.get("sure") and run.submits >= 2) else None)  # base19: later repeats accepted
                if problem is None and getattr(run, "held", None) is not None:
                    problem = await self.held_problem(run, submitted, row)
                if problem:
                    text = f"submit: {problem}"
                    if settings.get("submit_check") and settings.get("one_answer"):  # refusals by cause, per entry
                        cause = "held" if row.get("held_passed") is False else "repeat" if row.get("repeat_refused") \
                            else "grids" if row.get("submit_fn") is False else "train" if row.get("submit_passed") != row.get("submit_pairs") else "invalid"
                        run.refusals[cause] = run.refusals.get(cause, 0) + 1
                        row.update(refused=cause, refusals=dict(run.refusals))
                    if getattr(run, "repeat_twice", False) and settings.get("repeat_ends_round") \
                            and settings.get("swap") == "budget":  # targets.md line, pending the maintainer (off by default:
                        run.repeat_twice = False  # with refusal it can recreate P4/R4's 2-reply rounds)
                        self.end_round(run, "same_answer")
                elif settings.get("try_cap"):
                    text = self.try_accept(run, submitted, settings, row)
                else:
                    self.take_submit(run, submitted, **self.pick_args(run, settings))
                    mode = settings["submit_mode"]
                    if settings.get("indep_after") and await self.fits(run, row, 8.0):
                        run.fit_seen = True
                    # a task leaves its slot only at a compaction or at its FIRST
                    # submit (prefix hits); a later submit never swaps, the model keeps iterating until its next
                    # compaction. A task that ends (close mode, R's single pass) still frees its slot.
                    first = run.submits - getattr(run, "try_base", 0) == 1
                    vc = await self.vcheck_text(run, row, submitted) if (
                        mode == "fresh" and first and settings.get("vcheck")) else None
                    if mode == "fresh" and first and settings.get("repair") == "fresh":  # rep1: the next pass repairs
                        await self.repair_ready(run, row)
                    if mode == "fresh" and settings.get("alt_fit") == "reopen":  # altR (30 Oct): keep a train-fitting
                        await self.alt_ready(run, row)  # function's source for the next pass's ask
                    legacy = settings.get("legacy_swap")  # replay of runs before ce971a80a (28 Oct 08:32 UTC):
                    # every submit left the slot; fresh / reopen-passes-0 passes ended at every submit
                    if mode == "continue":  # base15c / d: the conversation runs on in its slot
                        if settings.get("recheck"):
                            text = {"wrong": tasks.CONV_RECHECK_WRONG, "reopen": tasks.CONV_REOPEN_WRONG}.get(
                                settings["recheck"], tasks.CONV_RECHECK)
                            if settings["recheck"] != "reopen" and getattr(run, "same_submits", 0) >= 2:
                                # the same answer three submits in a row; base15c2 has no such stop
                                text, run.done = "Submitted. The puzzle is finished.", "identical"
                        elif settings.get("fresh_tries") and (not first or settings.get("submit_two")):
                            text, run.try_over = "Submitted. The puzzle is finished.", "submits"  # base17: not the task
                        elif first:  # base16b: base15b's text
                            text = tasks.CONV_REOPEN_WRONG if settings.get("reopen_wrong") else tasks.CONV_ALTERNATIVE
                        else:
                            text, run.done = "Submitted. The puzzle is finished.", "submitted"
                    elif mode == "provisional":  # no task stops at a submit
                        text = (tasks.CONV_FIRM if settings.get("firm") else tasks.CONV_PROVISIONAL).format(n=run.submits)
                        run.parked = first or bool(legacy)
                    elif legacy and (mode == "fresh" or (mode == "reopen" and (
                            not run.reopened or not int(settings.get("reopen_passes", 1))))):
                        text, run.parked, run.awaiting_reopen = "Submitted. The puzzle is finished.", True, True
                    elif mode == "fresh" and first and settings.get("triage_close") is not None \
                            and run.spent <= int(settings["triage_close"]):  # hyp1: a fast first submit closes the entry
                        text, run.done = "Submitted. The puzzle is finished.", "submitted"
                        row.update(triage_closed=True)
                    elif mode == "fresh" and settings.get("agree_close") and self.agrees(run):  # agree1 (30 Oct):
                        # a pass whose first guess repeats an earlier pass's on every output closes the task
                        text, run.done = "Submitted. The puzzle is finished.", "agreed"
                        row.update(agree_closed=True)
                    elif mode == "fresh" and first and settings.get("repair") == "inpass":  # rep2 (30 Oct): the
                        # first answer is kept (guess 2 via keep_first); the pass goes on to an edge-case check
                        text = tasks.CONV_REPAIR_NOW.format(notes=novelty.report(getattr(run, "shown", None) or run.task))
                        row.update(repair_asked=True)
                    elif vc:  # vcheck: the first answer is kept (guess 2 via keep_first); the pass goes on
                        text = vc
                    elif mode == "fresh" and settings.get("try_notes") and not getattr(run, "note_asked", False) \
                            and not self.noted(run):  # notes1: a try ends only with its notes written for the next
                        text = tasks.CONV_TRY_NOTE  # (v8.5p LEVEL_NOTE); the next submit ends it
                        run.note_asked = True
                        row.update(note_asked=True)
                    elif mode == "fresh" and (first or settings.get("fresh_each")):  # once the
                        # backlog is empty, the task comes back fresh; ``fresh_each`` (naive): after every submit
                        # back in a new conversation (``refresh``) with its own program and answer
                        text, run.parked, run.awaiting_reopen = "Submitted. The puzzle is finished.", True, True
                    elif mode == "reopen" and first:  # R: back for its check pass once the backlog is empty
                        text, run.parked, run.awaiting_reopen = "Submitted. The puzzle is finished.", True, True
                    elif mode == "reopen" and settings.get("sure"):  # base19: every later answer is followed by the
                        # same question in the same conversation, as its next pass; 5 identical submits in a row close it
                        self.write_pick(run, [self.answers18(run)[0], self.answers18(run)[-1]], only=True)
                        run_of = self.same_run(run)
                        self.b19_log(run, run_of)
                        if run_of >= 5:
                            text, run.done = "Submitted. The puzzle is finished.", "identical5"
                        else:
                            text, run.parked, run.awaiting_reopen = "Submitted. The puzzle is finished.", True, True
                    elif mode == "reopen" and settings.get("third") and run.submits - getattr(run, "try_base", 0) == 2:
                        # base18: the third answer is asked as the second was, once every second pass ran
                        text, run.parked, run.awaiting_reopen = "Submitted. The puzzle is finished.", True, True
                        self.pick18(run)
                    elif mode == "reopen" and settings.get("third") and run.submits - getattr(run, "try_base", 0) == 3:
                        text = "Submitted. The puzzle is finished."  # base18: the selector picks guess 2 (``select``)
                        self.select_ready(run)
                    elif mode == "fresh" or (mode == "reopen" and not int(settings.get("reopen_passes", 1))):
                        # a later submit of a reopened task: recorded; it keeps iterating until its next compaction
                        text = tasks.CONV_FIRM.format(n=run.submits)
                    else:
                        text, run.done = "Submitted. The puzzle is finished.", "submitted"
                    if (settings.get("no_repeat") or settings.get("second_differs")) and run.done is None and mode != "continue" \
                            and getattr(run, "select_due", None) is None:
                        # no "submit it again" texts
                        text = (tasks.CONV_RECORDED if first or ((settings.get("third") or settings.get("sure"))
                                                                 and run.awaiting_reopen)
                                else tasks.CONV_NEXT).format(n=run.submits)
                        text = tasks.pinned(text) if settings.get("keep_first") else text
                    if settings.get("swap") == "budget" and run.done is None:  # ROUNDS
                        waiting = getattr(run, "awaiting_reopen", False)
                        if first:
                            self.end_round(run, "first_submit")
                        elif settings.get("same_answer_swap") and (
                                getattr(run, "round_repeat", False) if settings.get("same_answer_scope") == "round"
                                else not run.answer_changed):  # round scope 
                            self.end_round(run, "same_answer")
                        elif getattr(run, "verified_twice", False):
                            self.end_round(run, "verified_twice")
                        run.awaiting_reopen = waiting
                    row.update(submitted=True, submit_n=run.submits)
            run.messages.append({"role": "tool", "tool_call_id": kept["id"], "content": text})
            row.update(outcome="submit" if row.get("submitted") else "tool", tool_code=snippet, tool_result=text)
            self.keep(run, row)
            return
        code = tasks.extract_code(content)
        if code is None:
            cut = reply.get("finish_reason") == "length"
            row["outcome"] = "truncated" if cut else "no_code"
            cut_text = tasks.CONV_CUT_CODE if settings.get("cut_mode") == "code_now" else tasks.CONV_CUT
            run.messages.append({"role": "user", "content": cut_text if cut else tasks.nothing_text(settings)})
            self.keep(run, row)
            return
        n = len(run.programs) + 1
        base = getattr(run, "pass_base", 0)  # ``pass_clean``: numbers restart with the pass, ``program`` stays absolute
        label = n - base
        run.programs.append(code)
        shown = run.shown or run.task
        inputs = [p["input"] for p in shown["train"]] + [t["input"] for t in shown["test"]]
        result = await self.execute(code, inputs)
        verdict = tasks.judge(shown, result)
        shown_tests, tests = verdict["tests"], [to_task(run, t) for t in verdict["tests"]]
        earlier = {json.dumps(c["tests"]): c["program"] - base for c in run.candidates
                   if c.get("verified") and c.get("program") and c["program"] > base}
        run.candidates.append({
            "code": hashlib.sha256(code.encode()).hexdigest()[:16], "attempt": 1, "reply": len(run.replies) + 1,
            "tokens_at": run.spent, "t": t1, "passed": verdict["passed"], "pairs": verdict["pairs"],
            "accuracy": verdict["accuracy"], "verified": verdict["verified"], "tests": tests, "program": n,
            **({"last_program": settings["fallback_last"]} if settings.get("fallback_last") else {})})
        if self.repl is not None and not result.get("error"):
            await self.tool(run, "define", f"program_{label}", code)
        if verdict["verified"]:
            same = earlier.get(json.dumps(tests))
            lines = [f"Program {label} passes every training pair (`program_{label}` in python)."]
            lines.append(f"Its test output is the same as program {same}'s." if same else
                         "Its test output is a new reading:\n" + "\n\n".join(
                             tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in shown_tests))
            text = "\n".join(lines)
        else:
            text = f"Program {label}: " + tasks.feedback(shown, result, verdict).replace(
                tasks.FIX_LINE, "Fix the rule and send the corrected program, or keep exploring in python.")
        text += "\n\n" + self.ledger(run)
        row.update(outcome="verified" if verdict["verified"] else "run_error" if result.get("error") else "wrong",
                   passed=verdict["passed"], pairs=verdict["pairs"], accuracy=round(verdict["accuracy"], 4), code=code,
                   program=n)
        run.messages.append({"role": "user", "content": text})
        self.keep(run, row)

    STEP_BACK = {"fails": 6, "since_program": 12, "same": 3, "every": 8, "gap": 3}  # naive12 triggers

    def step_back(self, run):
        """naive12: v8.5p's reflect STEP BACK (style 2) with ARC-2 triggers, counted within one conversation and reset
        at a submit: ``fails`` program replies in a row failing a training pair, ``since_program`` replies without a
        new program, or a failing program's outputs seen ``same`` times; again every ``every`` replies while the same
        trigger holds, at least ``gap`` replies after the previous block. The block goes after the reply's result."""
        sb, row = getattr(run, "sb", None), (run.replies[-1] if run.replies else {})
        if sb is None:
            return
        sb["replies"] += 1
        out = row.get("outcome")
        if row.get("submitted"):
            sb.update(fails=0, since_program=0, same={}, last=None, reason=None)
            return
        if out in ("verified", "wrong", "run_error"):
            c = run.candidates[-1]
            sb["since_program"] = 0
            if c.get("verified"):
                sb["fails"] = 0
            else:
                sb["fails"] += 1
                key = json.dumps([c.get("passed"), round(float(c.get("accuracy") or 0), 4), c.get("tests")])
                sb["same"][key] = sb["same"].get(key, 0) + 1
        else:
            sb["since_program"] += 1
        limits, why, reason = self.STEP_BACK, None, None
        top = max(sb["same"].values(), default=0)
        if sb["fails"] >= limits["fails"]:
            reason, why = "fails", f"your last {sb['fails']} programs each failed a training pair"
        elif sb["since_program"] >= limits["since_program"]:
            reason, why = "since_program", f"{sb['since_program']} replies without a new program"
        elif top >= limits["same"] and out in ("wrong", "run_error") and sb["same"].get(key) == top:
            reason, why = "same", f"a program gave the same training outputs as an earlier failing one ({top} times)"
        if reason is None:
            return
        if sb["last"] is not None and sb["replies"] - sb["last"] < (limits["every"] if reason == sb["reason"]
                                                                      else limits["gap"]):
            return
        last = run.messages[-1] if run.messages else None
        if last is None or last.get("role") not in ("tool", "user") or not isinstance(last.get("content"), str):
            return
        programs = [c for c in run.candidates if c.get("program")]
        shown = run.shown or run.task
        last["content"] += "\n\n" + tasks.STEP_BACK.format(
            why=why, replies=sb["replies"], tokens=run.spent, programs=len(programs),
            verified=sum(bool(c.get("verified")) for c in programs),
            best=max((int(c.get("passed") or 0) for c in programs), default=0), pairs=len(shown["train"]))
        sb["last"], sb["reason"] = sb["replies"], reason
        self.record({"event": "step_back", "task": run.id, "reason": reason, "reply": len(run.replies)})

    def swaps_at_compaction(self, run):
        """Whether a compaction gives up the slot (``swap`` "budget": only in round 1)."""
        return {**(self.settings or {}), **run.arm}.get("swap") != "budget" or not getattr(run, "submits", 0)

    def end_round(self, run, why):
        run.parked, run.round, run.round_start = True, getattr(run, "round", 1) + 1, None
        run.round_answers, run.round_context = set(), None
        self.record({"event": "round_end", "task": run.id, "why": why, "round": run.round - 1, "spent": run.spent})

    def round_budget(self, run, settings):
        budget = int(getattr(run, "first_submit_tokens", 0) or 0)
        if settings.get("budget_context"):  # at least the context it re-entered with (prefill : generation <= 1)
            budget = max(budget, int(getattr(run, "round_context", None) or 0))
        return max(budget, int(settings["cap"] or self.cap)) if settings.get("budget_floor") else max(1, budget)

    async def budget_check(self, run, settings):
        """``swap`` "budget": a later round ends when its tokens reach the round budget."""
        if settings.get("swap") != "budget" or not getattr(run, "submits", 0) or run.done is not None or run.parked:
            return
        if getattr(run, "round_start", None) is None:
            run.round_start = run.spent - (run.replies[-1]["completion_tokens"] if run.replies else 0)
            if getattr(run, "round_context", None) is None:  # ``round_text`` may have set it at the resume
                run.round_context = run.replies[-1].get("prompt_tokens") if run.replies else None
        if run.spent - run.round_start >= self.round_budget(run, settings):
            over = int(settings.get("compact_over") or 0)  # compact only a context over 64K
            if settings.get("compact_on_swap") and (not over or await self.count(run.messages, settings) > over):
                await self.compact(run, settings)
            self.end_round(run, "budget")

    def pass_check(self, run, settings):
        """``pass_budget`` (bounded passes, 29 Oct): a fresh-mode pass that reaches this many tokens without a submit
        ends; the pass's best program output (verified, then accuracy, then latest) is submitted for it, and the
        entry waits for a fresh pass as after a submit."""
        cap = settings.get("pass_budget")
        if not cap or run.done is not None or run.parked or settings.get("submit_mode") != "fresh":
            return
        start = getattr(run, "pass_start", 0)
        if run.spent - start < int(cap):
            return
        pool = [c for c in run.candidates if c.get("tokens_at", 0) > start and c.get("program") is not None
                and not c.get("submitted") and c.get("tests") and all(t is not None for t in c["tests"])]
        best = max(pool, key=lambda c: (bool(c.get("verified")), c.get("accuracy") or 0, c.get("tokens_at", 0)),
                   default=None)
        if best is not None:
            self.take_submit(run, [[to_view(run, g) for g in best["tests"]]], **self.pick_args(run, settings))
        run.parked, run.awaiting_reopen = True, True
        self.record({"event": "pass_cut", "task": run.id, "pass": getattr(run, "reopen_count", 0), "spent": run.spent,
                     "pass_tokens": run.spent - start, "submitted": best is not None,
                     "verified": bool(best and best.get("verified")), "accuracy": best and best.get("accuracy")})

    def churn_check(self, run, settings):
        """Setting ``churn``: see CHURN_SAME. Called after each reply."""
        if not settings.get("churn") or run.done is not None or not run.replies or not getattr(run, "submits", 0):
            return  # never-submitted tasks are kept indefinitely
        last = run.replies[-1]
        if last.get("outcome") == "submit" and getattr(run, "answer_changed", False):
            run.short_streak = 0
        elif int(last.get("completion_tokens") or 0) < CHURN_SHORT:
            run.short_streak = getattr(run, "short_streak", 0) + 1
        else:
            run.short_streak = 0
        same = getattr(run, "same_submits", 0)
        if same < CHURN_SAME and run.short_streak < CHURN_REPLIES:
            return
        run.stale_count = getattr(run, "stale_count", 0) + 1
        why = "same_answer" if same >= CHURN_SAME else "short_replies"
        run.short_streak = run.same_submits = 0
        if run.stale_count >= CHURN_CLOSE:
            run.done, run.awaiting_reopen = "stale", False
        else:  # no swap: it ranks last when it next leaves its slot (compaction)
            run.stale = True
            run.parked = run.parked or bool(settings.get("legacy_swap"))
        self.record({"event": "churn", "task": run.id, "why": why, "stale": run.stale_count, "spent": run.spent,
                     "closed": run.done == "stale"})

    def submit_problem(self, run, result, settings, row):
        """``submit_check``: a submitted function must reproduce every training pair; grids alone are refused. The
        exact differences come back as for a program reply. ``allow_force`` (chk1-chk6 only): ``force=True`` passes
        grids or a failing function (removed by default)."""
        run.submit_checked = run.submit_checked2 = False
        if settings.get("submit_two"):
            return self.submit_two_problem(run, result, settings, row)
        if not settings["submit_check"]:
            return None
        if settings.get("submit_check2") and not self.check2_exited(run, settings) and self.check2_due(run, settings, row):
            problem = self.submit_problem2(run, result, row)
            if problem is not None:
                run.check2_refusals = getattr(run, "check2_refusals", 0) + 1
                if getattr(run, "check2_from", None) is None:  # the exit budget: the entry's tokens so far
                    run.check2_from, run.check2_budget = run.spent, max(run.spent, int(settings["cap"] or self.cap))
                row.update(check2_refused=run.check2_refusals)
                return problem
            return None
        check, asked = result.get("submit_check"), bool(result.get("forced"))
        allow = bool(settings.get("allow_force"))
        forced = asked and allow
        row.update(submit_forced=forced, submit_fn=check is not None, **({"force_refused": True} if asked and not allow else {}))
        if check is None:
            return None if forced else (tasks.CONV_SUBMIT_GRIDS_FORCE if allow else tasks.CONV_SUBMIT_GRIDS_ONE
                                        if settings.get("one_answer") else tasks.CONV_SUBMIT_GRIDS)
        shown = run.shown or run.task
        verdict = tasks.judge(shown, {"outputs": check})
        row.update(submit_passed=verdict["passed"], submit_pairs=verdict["pairs"])
        if verdict["passed"] == verdict["pairs"]:
            run.submit_checked = True
            return None
        if forced:
            return None
        return ("your function does not reproduce every training pair, so nothing was submitted.\n"
                + tasks.feedback(shown, {"outputs": check}, verdict).replace(
                    tasks.FIX_LINE, tasks.CONV_SUBMIT_FIX_FORCE if allow else tasks.CONV_SUBMIT_FIX))

    async def held_problem(self, run, submitted, row):
        """base16: the function must also reproduce the held-back pair; the model hears pass or fail only."""
        run.held_checks += 1
        passed = await self.tool(run, "held", run.held["input"]) == run.held["output"]
        truth = [t.get("output") for t in run.task["test"]]
        tests = [to_task(run, g) for g in submitted[0]]
        self.record({"event": "held_check", "task": run.id, "n": run.held_checks, "passed": passed, "spent": run.spent,
                     "accepted": run.submits, "right": [a == b for a, b in zip(tests, truth)]})
        row.update(held_passed=passed)
        return None if passed else tasks.CONV_HELD_FAIL

    def submit_two_problem(self, run, result, settings, row):
        """base13b (``submit_two``): every submit has two guesses with different test outputs; an entry
        past ``check2_late`` needs both to be functions that reproduce every training pair, on all its submits. No
        refusal cap, no token exit, no force: a refusal loop ends with the pass at the context cap."""
        guesses = result.get("submitted") or []
        if settings.get("check2_late") is not None and self.check2_due(run, settings, row):
            problem = self.submit_problem2(run, result, row)
            if problem is not None:  # nothing is told up front: the refusal says what passes
                detail = problem.removeprefix("nothing was submitted: ")
                problem = tasks.CONV_SUBMIT_VERIFY if problem == tasks.CONV_SUBMIT_TWO else (
                    f"{tasks.CONV_SUBMIT_VERIFY}\n\n{detail[:1].upper()}{detail[1:]}")
        elif len(guesses) < 2 or guesses[0] is None or guesses[1] is None:
            problem = tasks.CONV_SUBMIT_TWO_GUESSES
        elif settings.get("fresh_tries") and any(json.dumps(a) == json.dumps(b) for a, b in zip(guesses[0], guesses[1])):
            problem = tasks.CONV_SUBMIT_SAME_OUTPUT
        elif json.dumps(guesses[0]) == json.dumps(guesses[1]):
            problem = tasks.CONV_SUBMIT_SAME_GUESSES
        else:
            problem = None
        if problem is not None:
            run.two_refusals = getattr(run, "two_refusals", 0) + 1
            row.update(two_refused=run.two_refusals)
            if settings.get("score_refused") and not run.submits:
                self.keep_refused(run, guesses)
        return problem

    def keep_refused(self, run, guesses):
        """``score_refused``: until an entry has an accepted submit its last refused pair stands as its answer
        (harness side, the model is never told); ``take_submit`` drops it. Grids ``check_submit`` would refuse are not
        kept."""
        good = [(rank, g) for rank, g in enumerate(guesses[:2], 1) if g is not None and len(g) == len(run.task["test"])
                and all(tasks.valid(x) for x in g)]
        if not good:
            return
        run.candidates = [c for c in run.candidates if not c.get("refused")]
        now = time.monotonic()
        for rank, guess in good:
            run.candidates.append({"code": f"refused-{rank}", "attempt": 1, "reply": len(run.replies) + 1,
                                   "tokens_at": run.spent, "t": now, "passed": 0, "pairs": 0, "accuracy": 0.0,
                                   "tests": [to_task(run, x) for x in guess], "verified": False, "refused": True,
                                   "submitted": rank, "submit_n": 0})

    @staticmethod
    def check2_due(run, settings, row):
        """chk5: with ``check2_late`` set, check2 applies only to an entry whose tokens to its
        first submit exceed it (counted to now while it has none) or that an augmentation check marked uncertain;
        other submits follow the chk1 rule. Unset: always (chk2-chk4)."""
        late = settings.get("check2_late")
        if late is None:
            return True
        ttfs = (getattr(run, "first_submit_tokens", None) if getattr(run, "submits", 0) else None) or run.spent
        reason = "late" if ttfs > int(late) else "aug" if getattr(run, "uncertain", False) else None
        row.update(check2_reason=reason)
        return reason is not None

    @staticmethod
    def check2_exited(run, settings):
        """``submit_check2``: after 3 refusals, or once the tokens since the first refusal
        reach the entry's tokens before it (floor one full reply), the chk1 rule applies (one verified function and
        an unchecked second guess)."""
        if getattr(run, "check2_refusals", 0) >= int(settings.get("check2_max_refusals") or 3):  # chk4: 10
            return True
        factor = settings.get("check2_token_exit", 1.0)  # tokens since the first refusal / tokens before it; None: off
        start = getattr(run, "check2_from", None)
        return factor is not None and start is not None and \
            run.spent - start >= float(factor) * getattr(run, "check2_budget", 0)

    def submit_problem2(self, run, result, row):
        """Both submitted functions must reproduce every training pair and give different test outputs."""
        check, check2 = result.get("submit_check"), result.get("submit_check2")
        row.update(submit_fn=check is not None, submit_fn2=check2 is not None)
        if check is None or check2 is None:
            return tasks.CONV_SUBMIT_TWO
        shown = run.shown or run.task
        n_train = len(shown["train"])
        for k, (outs, fix) in enumerate(((check, tasks.CONV_SUBMIT2_FIX), (check2, tasks.CONV_SUBMIT_SECOND_FIX))):
            verdict = tasks.judge(shown, {"outputs": outs})
            row.update(**{f"submit_passed{k + 1}": verdict["passed"], "submit_pairs": verdict["pairs"]})
            if verdict["passed"] != verdict["pairs"]:
                which = "first" if k == 0 else "second"
                return (f"your {which} function does not reproduce every training pair, so nothing was submitted.\n"
                        + tasks.feedback(shown, {"outputs": outs}, verdict).replace(tasks.FIX_LINE, fix))
        if json.dumps(check[n_train:]) == json.dumps(check2[n_train:]):
            return tasks.CONV_SUBMIT_SAME
        run.submit_checked = run.submit_checked2 = True
        row.update(check2_accepted=True)
        return None

    async def fits(self, run, row, seconds=5.0):
        """The interpreter's train-fitting functions (REPL op fit), noted on the reply row; [] when none or down."""
        interp = getattr(run, "interp", None)
        if interp is None or not hasattr(interp, "fit"):
            return []
        fits = await self.tool(run, "fit", seconds) or []
        row.update(fit_n=len(fits), fit_outputs=len({json.dumps(f["tests"]) for f in fits}))
        return fits

    async def vcheck_text(self, run, row, submitted):
        """vcheck (30 Oct): the train-fitting function that produced the submitted first guess, checked for the
        transforms every training pair keeps; the check text when one is broken on the test input, else None."""
        interp = getattr(run, "interp", None)
        if interp is None or not hasattr(interp, "fit") or not submitted or submitted[0] is None:
            return None
        fits = await self.tool(run, "fit", 8.0, 1.0) or []
        mine = [f for f in fits if f.get("tests") == submitted[0]]
        row.update(vcheck_fits=len(fits), vcheck_mine=len(mine))
        for f in mine:
            broken = [(n, v) for n, v in (f.get("equiv") or {}).items() if v and v[0] and v[1] is False]
            broken.sort(key=lambda x: x[0].startswith("color"))  # a broken symmetry of the grid first
            for name, v in broken:
                cells = v[2] if len(v) > 2 else None
                k = next((i for i, c in enumerate(cells or []) if c), None)
                if k is None:
                    continue
                row.update(vcheck=name, vcheck_cells=len(cells[k]))
                which = "the test output" if len(run.task["test"]) == 1 else f"test output {k + 1}"
                shown = ", ".join(f"({a}, {b})" for a, b in cells[k][:25]) + (
                    f" and {len(cells[k]) - 25} more" if len(cells[k]) > 25 else "")
                return tasks.CONV_VCHECK.format(name=f["name"], what=tasks.TRANSFORM_NAMES.get(name, name),
                                                which=which, cells=shown)
        return None

    async def repair_ready(self, run, row):
        """rep1: at the first submit, the source of a train-fitting function for the next pass's repair head."""
        await self.alt_ready(run, row)
        run.repair_source, run.alt_source = run.alt_source, None
        row.update(repair_ready=bool(run.repair_source))

    async def alt_ready(self, run, row):
        """altR: at a submit, the source of a train-fitting function (the cell that defined it) for the next pass's
        ask; None when nothing fits."""
        fits = await self.fits(run, row)
        source = None
        for f in fits:
            source = next((c for c in reversed(getattr(run.interp, "codes", None) or [])
                           if f"def {f['name']}(" in str(c or "")), None)  # submit cells too (not in history)
            if source:
                break
        run.alt_source = tasks.clip(source, 4000) if source else None
        row.update(alt_ready=bool(source))

    @staticmethod
    def agrees(run):
        """The latest submit's first guess equals an earlier submit's first guess on every output."""
        firsts = {c["code"]: json.dumps(c["tests"]) for c in run.candidates if str(c.get("code", "")).startswith("submit-")
                  and c["code"].endswith("-1")}
        latest = firsts.get(f"submit-{run.submits}-1")
        return latest is not None and any(v == latest for k, v in firsts.items() if k != f"submit-{run.submits}-1")

    def try_problem(self, run, guesses, row):
        """``try_cap`` (): the first accepted submit needs two different guesses; a later
        one needs only guess 1 to differ from every earlier guess (guess 2 is free, None included: the final answer is
        S1 g1 + S2 g1). Refused: the try goes on."""
        key = lambda g: json.dumps([to_task(run, x) for x in g])  # noqa: E731
        old = {json.dumps(c["tests"]) for c in run.candidates if c.get("submit_at") is not None}
        if run.submits and key(guesses[0]) in old:
            why, text = "earlier", tasks.CONV_TRY_OLD
        elif run.submits:
            return None
        elif len(guesses) < 2 or guesses[1] is None:
            why, text = "second_missing", tasks.CONV_TRY_SECOND
        elif key(guesses[0]) == key(guesses[1]):
            why, text = "same", tasks.CONV_TRY_SAME
        else:
            return None
        run.try_refused = getattr(run, "try_refused", 0) + 1
        row.update(try_refused=why, try_n=getattr(run, "try_n", 1))
        return text

    def try_accept(self, run, guesses, settings, row):
        """An accepted submit ends the try. The second accepted submit ends the task with guess 1 = S1 guess 1 and
        guess 2 = S2 guess 1 (S2 guess 2 is kept unsubmitted); after the first, the task waits for its next try."""
        self.take_submit(run, guesses)
        k, tokens = run.submits, run.spent - getattr(run, "pass_start", 0)
        run.try_tokens = getattr(run, "try_tokens", []) + [tokens]
        row.update(submitted=True, submit_n=k, try_n=getattr(run, "try_n", 1), try_tokens=tokens)
        if k >= 2:
            for c in run.candidates:
                if c.get("submit_at") == 1 and c.get("submitted") == 1:
                    c["submit_n"] = k
                elif c.get("submit_at") == k and c.get("submitted") == 1:
                    c["submitted"] = 2
                elif c.get("submit_at") == k and c.get("submitted") == 2:
                    c["submitted"], c["s2_second"] = None, True
            run.done = "submitted"
        else:
            run.parked, run.awaiting_reopen = True, True
        self.record({"event": "try_end", "task": run.id, "cause": "accepted", "try": getattr(run, "try_n", 1),
                     "tokens": tokens, "accepted": k, "spent": run.spent, "refused": getattr(run, "try_refused", 0)})
        return "Submitted. The puzzle is finished."

    def end_try(self, run, settings, cause, count):
        """``context_mode`` "cap": the context is full; the try ends without a submit and the task waits for its next
        try (a blank one if it has no accepted submit)."""
        tokens = run.spent - getattr(run, "pass_start", 0)
        run.try_tokens = getattr(run, "try_tokens", []) + [tokens]
        run.cap_ends = getattr(run, "cap_ends", 0) + 1
        if tokens <= 0:  # the try's opening prompt alone fills the window: another try would end the same way
            run.done = "context_full"
        else:
            run.parked, run.awaiting_reopen = True, True
        self.record({"event": "try_end", "task": run.id, "cause": cause, "try": getattr(run, "try_n", 1),
                     "tokens": tokens, "context": count, "accepted": run.submits, "spent": run.spent,
                     "refused": getattr(run, "try_refused", 0)})

    def loop_exit(self, run, settings):
        """``loop_exit`` (base14): loop_rule only acts inside a drain, so a no-drain pass had no way out
        (base13b 7491f3cf: 112K of ``{"code": ""}``; tc128 dd6b8c4b: 84 bad calls). A rule fire, or this many bad calls
        in a row, ends the pass as the context cap does."""
        fire, run.loop_fire = getattr(run, "loop_fire", None), None
        if run.done is not None or run.parked:
            return False
        cause = "loop" if fire else "bad_calls" if run.bad_calls >= int(settings["loop_exit"]) else None
        if cause is None:
            return False
        run.bad_calls, run.loop_detector = 0, None  # the detector fires once per streak; a new pass has its own
        self.end_try(run, settings, cause, None)
        return True

    def repeat_problem(self, run, guesses, row):
        """``no_repeat``: a later submit whose grids, output by output, were all submitted before is
        refused (it would add nothing to the two guesses); the round goes on. Partial overlap is accepted."""
        n = len(run.task["test"])
        new = [{json.dumps(to_task(run, g[k])) for g in guesses if g is not None} for k in range(n)]
        only_first = {**(self.settings or {}), **run.arm}.get("differs_first")  # base15c2: a later answer may come back
        old = [{json.dumps(c["tests"][k]) for c in run.candidates if c.get("submitted") and c["tests"][k] is not None
                and (not only_first or (c.get("submit_at", c.get("submit_n")) == getattr(run, "try_base", 0) + 1
                                       and not c.get("vote")))}
               for k in range(n)]
        repeats = [bool(a) and a <= b for a, b in zip(new, old)]
        row.update(repeat_outputs=sum(repeats))
        if not all(repeats):
            return None
        row.update(repeat_refused=True)
        run.repeats_refused = getattr(run, "repeats_refused", 0) + 1
        key = json.dumps([None if g is None else [to_task(run, x) for x in g] for g in guesses])
        seen = getattr(run, "round_answers", None)
        if seen is None:
            seen = run.round_answers = set()
        run.repeat_twice = key in seen  # the same answer already tried in this round (refused or accepted)
        seen.add(key)
        if {**(self.settings or {}), **run.arm}.get("one_answer"):
            return tasks.CONV_REPEAT_ONE
        text = tasks.CONV_REPEAT.format(which="this answer is" if n == 1 else "every output of this answer is")
        return tasks.pinned(text) if {**(self.settings or {}), **run.arm}.get("keep_first") else text

    def round_text(self, run, settings):
        """``round_prompt``: the start of a later round: the earlier submissions with their verification, the doubts to
        test, a different answer to submit, and the round's token budget."""
        if run.replies and getattr(run, "round_context", None) is None:
            run.round_context = int(run.replies[-1].get("prompt_tokens") or 0) + int(run.replies[-1].get("completion_tokens") or 0)
        fmt = lambda grids: "\n\n".join(tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in grids)  # noqa: E731
        at = lambda c: c.get("submit_at", c.get("submit_n"))  # noqa: E731  the submit it came from (``keep_first``)
        subs = sorted({at(c) for c in run.candidates if c.get("submitted")}, reverse=True)[:2]
        if settings.get("keep_first") and subs and 1 not in subs:  # the latest and the pinned submission 1
            subs = [subs[0], 1]
        parts = [tasks.CONV_ROUND.format(round=run.round, budget=self.round_budget(run, settings))]
        if settings.get("keep_first"):
            parts = [tasks.pinned(parts[0])]
        for n in subs:
            for c in sorted((c for c in run.candidates if c.get("submitted") and at(c) == n),
                            key=lambda c: c["submitted"]):
                status = ("produced by a program that reproduces every training pair" if c.get("verified")
                          else "not verified against the training pairs")
                parts.append(f"\n\nSubmission {n}, {'first' if c['submitted'] in (1, 1.5) else 'second'} guess ({status}):\n"
                             + fmt([to_view(run, g) for g in c["tests"]]))
        return "".join(parts)

    def check_submit(self, run, guesses):
        first = guesses[0]
        if first is None:
            return "the first guess is missing"
        for guess in (g for g in guesses if g is not None):
            if len(guess) != len(run.task["test"]) or not all(tasks.valid(g) for g in guess):
                return (f"each guess needs {len(run.task['test'])} valid grid(s) (lists of lists of ints 0-9, "
                        "rectangular, at most 30x30), one per test input. Nothing was submitted.")
        return None

    def take_submit(self, run, guesses, keep_first=False, pick=None):
        """The submit's grids become the task's answer (the latest submit ranks first in ``loop.guesses``); an
        overwrite of an earlier answer is logged with what it did to each output, when the outputs are known."""
        inputs = [t["input"] for t in run.task["test"]]
        guesses = [None if g is None else [to_task(run, x) for x in g] for g in guesses]  # the task's frame
        run.candidates = [c for c in run.candidates if not c.get("refused")]
        before = guesses_of(run.candidates, inputs) if run.submits else None
        answer = json.dumps(guesses)
        run.answer_changed = answer != getattr(run, "last_answer", None)
        verified_now = json.dumps(guesses[0]) in {json.dumps(c["tests"]) for c in run.candidates if c.get("verified")} \
            or bool(getattr(run, "submit_checked", False))
        run.verified_twice = not run.answer_changed and verified_now and getattr(run, "last_verified", False)
        run.last_verified = verified_now
        run.same_submits = 0 if run.answer_changed else getattr(run, "same_submits", 0) + 1
        seen = getattr(run, "round_answers", None)
        if seen is None:
            seen = run.round_answers = set()
        run.round_repeat = answer in seen  # the same answer already submitted in this round
        seen.add(answer)
        run.last_answer = answer
        run.submits += 1
        if run.first_submit_tokens is None:
            run.first_submit_tokens = run.spent
        verified = {json.dumps(c["tests"]) for c in run.candidates if c.get("verified")}
        checked = getattr(run, "submit_checked", False)
        now = time.monotonic()
        if keep_first and run.submits > 1:  # submission 1's first guess stays guess 2 on every later
            # submit (p7: S1 right, then [S2, S1], then [S3, S2] dropped it); a new first equal to it keeps the
            # model's own second guess (``loop.guesses`` dedups the grid)
            pin = next((c for c in run.candidates if c.get("pinned") or (
                c.get("submitted") == 1 and c.get("submit_at", c.get("submit_n")) == 1 and not c.get("b2"))), None)
            if pin is not None:
                pin.update(submit_n=run.submits, submitted=1.5, pinned=True)
        for rank, guess in enumerate(guesses, 1):
            if guess is None:
                continue
            n = run.submits
            run.candidates.append({"code": f"submit-{run.submits}-{rank}", "attempt": 1, "reply": len(run.replies) + 1,
                                   "tokens_at": run.spent, "t": now, "passed": 0, "pairs": 0,
                                   "accuracy": 0.0, "tests": guess,
                                   "verified": json.dumps(guess) in verified or (rank == 1 and checked)
                                   or (rank == 2 and getattr(run, "submit_checked2", False)),
                                   "submitted": rank, "submit_n": n, "submit_at": run.submits,
                                   **({"try_first": True} if run.submits - getattr(run, "try_base", 0) == 1 else {}),
                                   **({"ends": True} if pick == "first_last" else {})})
        if pick == "vote":
            self.vote_pick(run)
        elif pick == "vote_all":
            self.all_pick(run)
        if before is not None:
            after = guesses_of(run.candidates, inputs)
            truth = [t.get("output") for t in run.task["test"]]
            row = {"event": "overwrite", "task": run.id, "submit": run.submits, "spent": run.spent, "t": now,
                   "pass": getattr(run, "reopen_count", 0), "new_answer": run.answer_changed,
                   "later_repeat": run.round_repeat, "changed": [a != b for a, b in zip(after, before)]}
            if all(x is not None for x in truth):
                row["before"] = [any(g == x for g in b) for b, x in zip(before, truth)]
                row["after"] = [any(g == x for g in a) for a, x in zip(after, truth)]
            run.overwrites.append(row)
            self.record(row)
        if getattr(run, "pair", None) is not None:
            self.merge_second(run)
        if getattr(run, "entry", None) is not None or getattr(run, "attempt_runs", None):
            self.merge_attempts(run)


def _json_ok(raw):
    try:
        return isinstance(json.loads(raw), dict)
    except (ValueError, TypeError):
        return False


def guesses_of(candidates, inputs):
    """Per test output, the two grids ``loop.guesses`` picks now."""
    from .loop import guesses

    return [[g for g, _ in pick] for pick in guesses(candidates, inputs)]


def overwrite_marks(runs, marks):
    """Per mark (roster tokens, by ``loop.roster``'s clock): overwrites so far and what they did to the outputs."""
    from .loop import token_clock

    clock = token_clock([x for r in runs for x in r.replies])
    rows = [(clock(o["t"]), o) for r in runs for o in getattr(r, "overwrites", [])]
    out = {}
    for name, tokens in marks.items():
        seen = [o for c, o in rows if c <= tokens + 1e-6]
        pairs = [(b, a) for o in seen if "before" in o for b, a in zip(o["before"], o["after"])]
        out[name] = {"overwrites": len(seen), "outputs_changed": sum(sum(o["changed"]) for o in seen),
                     "right_to_wrong": sum(b and not a for b, a in pairs),
                     "wrong_to_right": sum(a and not b for b, a in pairs)}
    return out
