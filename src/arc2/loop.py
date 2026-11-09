"""One task's verify-and-refine loop, its guesses and the score-against-tokens read-out.

An attempt is a fresh conversation: task -> program -> exact mismatches -> corrected program ..., until a program is
verified (every training pair exact and a valid grid for every test input), the attempt's replies are used up or the
context window is full. Then a new, independent attempt starts. A task ends when two verified programs disagree on a
test output, when ``confirm_attempts`` further attempts ended after the first verified program, or at its token
budget. Tokens are completion tokens (reasoning included), as in the arena's level budgets.

``review`` (an arm's setting): after the task's first verified program, one more reply in the same conversation sees
that program's test outputs and either keeps them or sends a corrected program. A corrected program is a candidate
like any other, ranked after an independent one with as many votes, and does not count as the second verified
program that settles a task (the confirming attempt still runs). ``review_style`` "check+second": both turns, the
check first, then the second reading of the outputs the check left (its verified correction's, else the program's).

Guesses per test input: candidates (every program that returned a valid grid for it) ranked by verified, training
pairs passed, mean cell accuracy on the training pairs, number of candidates with the same grid, a review's
correction last, then earliest; the two first distinct grids. With no candidate the guess is the test input itself ("fallback").

Roster runs (the competition's shape: one token budget for all tasks) never end a task. The next reply goes to the
task with the lowest ``spent + delay``, delay 0 without a verified program, ``confirm_delay`` while the confirming
attempts run and ``settled_delay`` after them. The order does not depend on the budget, so the state at any total
token count is what a run stopped there would have had: one run gives the whole score-against-total-tokens curve.
"""

import asyncio
import bisect
import hashlib
import json
import random
import re
import time

from . import tasks

DEFAULTS = {"attempt_replies": 6, "confirm_attempts": 1, "min_room_tokens": 4096, "errors_per_task": 4,
            "effort": None, "temperature": None, "cap": None,  # cap: a reply cap below the study's
            "confirm_delay": 0, "settled_delay": None,  # settled_delay None: a settled task ends
            "review": False, "review_style": "check",
            "tool": False, "tool_calls": 12,  # tool: the persistent `python` tool; calls per attempt
            "truncated_continue": False,  # a reply cut at the cap: "continue" (reasoning kept) instead of tasks.TRUNCATED
            "conversation": False, "compact_keep": 0.10, "summary_effort": "low",  # arc2.conv
            "cut_program": 0, "cut_effort": "low",  # > 0: after a cut reply, one program reply (no tools, this cap,
            # cut_effort); unless it verifies, the attempt ends
            "tool_fixes": False,  # C1 (tasks.TOOL_FIXES): output start and end, no tools after the last call, ...
            "carry": False, "note_tokens": 1024, "notes_kept": 3,
            # pool k: from attempt pool_after, the attempt sees up to k earlier programs of the task on its view
            # (distinct verified readings first, then the best others; Engine.pick_pool); pool_live records this run's
            # programs, else only the trial's pool_seed is used
            # pool_style "check" (item 2): one verified program and its test outputs; the attempt is asked what is new
            # in the test input and for the most plausible other reading there
            "pool": 0, "pool_after": 3, "pool_live": True, "pool_style": "programs",  # carry: notes + best program into the next attempt
            "confirm_style": None,  # "other": the attempt after a verified program asks for the other reading
            "prompt": None,  # "arc2" / "arc2b": tasks.ARC2_HINT / ARC2_HINT_B after the system prompt
            "lessons": False, "lessons_kept": 5, "lesson_tokens": 1024,  # cross-task lessons (tasks.LESSON)
            "minds": 1,  # 2: even attempts in the second style (tasks.MIND_B)
            "settle_agree": False,  # True: two verified programs settle a task even when they agree (with confirm_attempts > 1)
            "progress_weight": 0.0,
            "augment": 1, "helpers": False, "views": False, "images": False, "image_side": 448, "image_feedback": False, "image_changes": False, "show": False, "examples": 0, "confirm_effort": None, "hard_effort": None, "hard_after": 0, "direct_every": 0, "direct_after": False, "direct_tokens": 8192, "repair": False, "anchor": False, "late_after": 0, "late_effort": None, "late_cap": None, "holdout": False, "hypotheses": False, "two_readings": False, "trace": False, "columns": False, "scouts": 0, "scout_tokens": 4096, "scout_effort": "low", "scout_judge_tokens": 8192, "scout_unfinished": False, "pictures": False, "reject_known": 0, "known_k": 3, "open_schedule": None, "max_attempts": 0, "direct_answer": False, "direct_answer_effort": "xhigh", "direct_answer_share": False, "direct_answer_reserve": None}  # n > 1: attempt a works on copy (a - 1) % n of the task (augment_view); copies' agreement ranks  # roster order: an open task's spent counts (1 - w * best share of training pairs passed)
EMPTY_FREE = 3  # tool_fixes: empty `python` calls an attempt answered without counting against the limit
BUDGETS = (5000, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 150000, 200000, 300000)

# Augmented copies (setting ``augment``): copy 0 is the task; copy k > 0 one of the seven other symmetries of the
# square and a permutation of colours 1-9 (0, the usual background, kept), both drawn from the task id and k. The
# model sees the copy; its program's test outputs are mapped back before they become candidates.
SYMMETRIES = (
    (lambda g: [list(r) for r in g], lambda g: [list(r) for r in g]),
    (lambda g: [list(r) for r in zip(*g[::-1])], lambda g: [list(r) for r in zip(*g)][::-1]),  # rotate 90
    (lambda g: [r[::-1] for r in g[::-1]], lambda g: [r[::-1] for r in g[::-1]]),  # rotate 180
    (lambda g: [list(r) for r in zip(*g)][::-1], lambda g: [list(r) for r in zip(*g[::-1])]),  # rotate 270
    (lambda g: [r[::-1] for r in g], lambda g: [r[::-1] for r in g]),  # mirror left-right
    (lambda g: [list(r) for r in g[::-1]], lambda g: [list(r) for r in g[::-1]]),  # mirror up-down
    (lambda g: [list(r) for r in zip(*g)], lambda g: [list(r) for r in zip(*g)]),  # transpose
    (lambda g: [list(r) for r in zip(*g[::-1])][::-1], lambda g: [list(r) for r in zip(*g[::-1])][::-1]),  # anti
)


def augment_spec(task_id, k):
    """(symmetry index, colour map) of copy ``k`` of the task (k = 0: the task itself)."""
    if k == 0:
        return 0, list(range(10))
    seed = int(hashlib.sha256(f"{task_id.split('#')[0]}:{k}".encode()).hexdigest()[:12], 16)
    sym = 1 + (seed % 7)
    colours = list(range(1, 10))
    for n in range(len(colours) - 1, 0, -1):  # Fisher-Yates from the seed
        seed, r = divmod(seed * 6364136223846793005 + 1442695040888963407 & (2**64 - 1), n + 1)
        colours[n], colours[r] = colours[r], colours[n]
    return sym, [0] + colours


def augment_view(task, task_id, k):
    """Copy ``k`` of the task: training pairs and test inputs (no test outputs)."""
    sym, colours = augment_spec(task_id, k)
    fwd = SYMMETRIES[sym][0]
    grid = lambda g: [[colours[c] for c in row] for row in fwd(g)]
    return {"train": [{"input": grid(p["input"]), "output": grid(p["output"])} for p in task["train"]],
            "test": [{"input": grid(t["input"])} for t in task["test"]]}


def augment_back(out, task_id, k):
    """A copy-k test output in the task's own frame (None stays None)."""
    if out is None or k == 0:
        return out
    sym, colours = augment_spec(task_id, k)
    inverse = {v: i for i, v in enumerate(colours)}
    return SYMMETRIES[sym][1]([[inverse.get(c, c) for c in row] for row in out])



class ContextFull(Exception):
    """The server refused the request for its length: the attempt ends."""


class StopRequested(Exception):
    """A stop request was seen after the reply passed its stop check but before it was sent (it waited at the gate):
    the reply is not sent and the task stops where it is."""


class TaskRun:
    def __init__(self, task_id, task, arm=None):
        self.id, self.task, self.arm = task_id, task, arm or {}
        self.view, self.copy = task, 0  # augment: the copy the current attempt works on
        self.spent = self.prompt_tokens = self.attempts = self.errors = 0
        self.candidates, self.replies = [], []
        self.messages, self.attempt_replies, self.confirms = None, 0, 0
        self.first_verified_attempt = None
        self.cut_pending = False  # cut_program: the next reply is the cut attempt's last program
        self.review_pending = False  # the review message is the conversation's last; its reply ends the attempt
        self.review_next, self.review_tests = [], None  # "check+second": the styles still to ask, the outputs shown
        self.cells, self.attempt_tools = [], 0  # the tool arm: this attempt's clean snippets and its calls
        self.attempt_empty = 0  # tool_fixes: empty calls answered without counting (EMPTY_FREE an attempt)
        self.done = None  # the reason once finished
        self.best, self.notes, self.note_pending = None, [], False  # carry: best unverified program, notes
        self.lesson_pending = False  # lessons: the reply after the first verified program is a lesson
        self.direct = False  # direct_every: this attempt answers the test grids itself, one reply
        self.direct_next = False  # direct_after without room left: the next attempt is a fresh direct one
        self.answered = False  # ``direct_answer``: the task's one direct reply has been sent
        self.shown, self.hidden = task, None  # holdout: the view the prompt shows, the hidden pair's index
        self.pooled = 0  # ``pool``: programs shown to the current attempt
        self.judged = None  # ``pool_style`` "judge": [reading hash, attempts] shown to the current attempt
        self.ranking = None  # and the reply's two best readings (indexes into judged)
        self.repaired, self.predicted = False, 0  # repair: done once; anchor: the attempt whose prediction is kept
        # ``twins`` (arm setting, arc2.trial): a second TaskRun works the same task at once from another copy; the
        # two share ``candidates`` and ``replies``; ``copy_offset`` shifts this run's copies.
        self.copy_offset, self.shared = 0, False
        self.scout = None  # ``scouts``: this attempt's rule statements, the judge's pick, their tokens
        self.known, self.rejects = None, 0  # ``reject_known``: readings earlier attempts found; rejections this attempt

    def verified_outputs(self):
        return {json.dumps(c["tests"]) for c in self.candidates if c["verified"] and not (c.get("review") or c.get("second"))}


def copy_index(task_id):
    """Roster copy number of an entry id ("t#2" -> 2; no suffix -> 1)."""
    part = task_id.split("@")[0].split("#")
    return int(part[1]) if len(part) > 1 and part[1].isdigit() else 1


def guesses(candidates, test_inputs, limit=None, key="tokens_at"):
    """Per test input: up to two ``(grid, source)`` with source "verified", "unverified" or "fallback". ``limit``
    keeps the candidates with ``key`` (the task's own tokens, or "clock": the roster's) at most that."""
    pool = [c for c in candidates if limit is None or c[key] <= limit]
    answers = [c for c in pool if c.get("answer")]  # ``direct_answer``: second answer only, never over a verified one
    pool = [c for c in pool if not c.get("answer")]
    result = []
    for k, test_input in enumerate(test_inputs):
        usable = [(c, json.dumps(c["tests"][k])) for c in pool if c["tests"][k] is not None]
        votes, copies = {}, {}
        predicted = {grid for c, grid in usable if c.get("predict")}  # setting ``anchor``
        for c, grid in usable:
            votes.setdefault(grid, set()).add(c["code"])
            if c["verified"] and not (c.get("review") or c.get("second")):
                copies.setdefault(grid, set()).add(c.get("copy", 0))
        ranked = sorted(usable, key=lambda x: (x[0].get("submitted") or 9, -x[0]["verified"], -len(copies.get(x[1], ())), -(x[1] in predicted),
                                               -(len(votes[x[1]]) if x[0].get("direct") else 0), -x[0]["passed"], -x[0]["accuracy"],
                                               -len(votes[x[1]]), bool(x[0].get("review") or x[0].get("second")), x[0]["tokens_at"]))
        picked, seen = [], set()
        for c, grid in ranked:
            if grid not in seen:
                seen.add(grid)
                picked.append((c["tests"][k], "verified" if c["verified"] else "direct" if c.get("direct") else "unverified"))
            if len(picked) == 2:
                break
        answer = next((c["tests"][k] for c in answers if c["tests"][k] is not None), None)
        if answer is not None and (len(picked) < 2 or picked[1][1] != "verified") and (not picked or picked[0][0] != answer):
            picked = picked[:1] + [(answer, "answer")]
        result.append(picked or [(test_input, "fallback")])
    return result


def score(run, limit=None, key="tokens_at"):
    """``(correct test outputs, test outputs, source of the first guess per output, correct per output)``."""
    picks = guesses(run.candidates, [t["input"] for t in run.task["test"]], limit, key)
    correct = [any(g == t["output"] for g, _ in pick) for pick, t in zip(picks, run.task["test"])]
    return sum(correct), len(correct), [pick[0][1] for pick in picks], correct


def scores(runs, limit=None, key="tokens_at"):
    """The competition's score (share of test outputs with a correct guess among two) and its breakdown."""
    rows = [score(r, limit, key) for r in runs]
    outputs = sum(r[1] for r in rows)
    by = {}
    for _, _, sources, correct in rows:
        for s, ok in zip(sources, correct):
            by.setdefault(s, [0, 0])
            by[s][0] += ok
            by[s][1] += 1
    return {"score_outputs": 100 * sum(r[0] for r in rows) / outputs,
            "score_tasks": 100 * sum(r[0] / r[1] for r in rows) / len(rows),
            "verified_tasks": sum(any(c["verified"] and (limit is None or c[key] <= limit)
                                      for c in r.candidates) for r in runs) / len(runs),
            "by_source": {s: {"outputs": n, "correct": ok, "accuracy": ok / n} for s, (ok, n) in sorted(by.items())}}


def token_clock(replies):
    """``tokens(t)``: completion tokens generated by time ``t`` over all replies, each at a uniform rate between its
    request and its end. Unlike a sum over finished replies it counts the replies in flight, as the hardware does."""
    events = []
    for x in replies:
        n, a, b = x["completion_tokens"], x["t0"], x["t1"]
        events += [(a, n / (b - a), 0), (b, -n / (b - a), 0)] if b > a else [(b, 0.0, n)]
    events.sort()
    times, totals, slopes = [], [], []
    total = slope = 0.0
    for t, change, jump in events:
        if times:
            total += slope * (t - times[-1])
        total, slope = total + jump, slope + change
        times.append(t)
        totals.append(total)
        slopes.append(max(0.0, slope))

    def tokens(t):
        k = bisect.bisect_right(times, t) - 1
        return 0.0 if k < 0 else totals[k] + slopes[k] * (min(t, times[-1]) - times[k])
    return tokens


def roster(runs, marks, *, cut=None, step=None):
    """Score against the roster's total tokens: at each named mark (``{name: tokens}``) and, with ``step``, every
    ``step`` tokens. A candidate counts at a mark when its reply had ended by the time the roster had generated that
    many tokens. ``cut``: the request time of the earliest reply the wall cancelled; the clock misses that reply's
    tokens, so marks beyond it are not exact. A mark the run did not reach scores what the run had at its end."""
    clock = token_clock([x for r in runs for x in r.replies])
    for r in runs:
        for c in r.candidates:
            c["clock"] = clock(c["t"])
    total = clock(float("inf"))
    exact = total if cut is None else clock(cut)

    def row(tokens):
        return {"tokens": tokens, "tokens_per_task": tokens / len(runs), "reached": tokens <= exact + 1e-6,
                **scores(runs, tokens + 1e-6, "clock")}
    out = {"tokens": total, "exact_through": exact, "marks": {name: row(tokens) for name, tokens in marks.items()}}
    if step:
        out["curve"] = [row(step * k) for k in range(1, int(max(total, *marks.values()) // step) + 1)]
    return out


def readout(runs, budgets=BUDGETS):
    """The fixed read-out over finished and unfinished tasks. A budget row is ``complete`` when every task either
    ended or spent at least that much; otherwise its score is a lower bound."""
    def at(limit):
        return scores(runs, limit)

    tokens = sum(r.spent for r in runs)
    replies = [x for r in runs for x in r.replies]
    split, phases = {}, {}
    for x in replies:
        for table, name in ((split, x["outcome"]), (phases, x["phase"])):
            row = table.setdefault(name, {"replies": 0, "tokens": 0})
            row["replies"] += 1
            row["tokens"] += x["completion_tokens"]
    by_attempt = {}
    for x in replies:
        key = str(min(x["attempt"], 4)) + ("+" if x["attempt"] >= 4 else "")
        row = by_attempt.setdefault(key, {"tokens": 0, "attempts": set(), "verified": set()})
        row["tokens"] += x["completion_tokens"]
        row["attempts"].add((x["task"], x["attempt"]))
        if x["outcome"] == "verified":
            row["verified"].add((x["task"], x["attempt"]))
    solved = {r.id for r in runs if score(r)[0] == score(r)[1]}
    reasoning = sum(x["reasoning_chars"] for x in replies)
    return {
        "tasks": len(runs), "test_outputs": sum(len(r.task["test"]) for r in runs),
        "final": at(None),
        # mean_tokens: what a run capped at this budget spends per task (tasks that end earlier spend less).
        "curve": [{"budget": b, **at(b), "mean_tokens": sum(min(b, r.spent) for r in runs) / len(runs),
                   "complete": all(r.done or r.spent >= b for r in runs),
                   "tasks_reached": sum(bool(r.done) or r.spent >= b for r in runs)} for b in budgets],
        "completion_tokens": tokens, "prompt_tokens": sum(r.prompt_tokens for r in runs),
        "tokens_per_task": {"mean": tokens / len(runs), "max": max(r.spent for r in runs),
                            "min": min(r.spent for r in runs)},
        "ended": {reason: sum(r.done == reason for r in runs) for reason in sorted({r.done for r in runs if r.done})}
        | {"unfinished": sum(r.done is None for r in runs)},
        "where_tokens_go": {
            "by_reply_outcome": split, "by_phase": phases,
            "by_attempt": {k: {"tokens": v["tokens"], "attempts": len(v["attempts"]),
                               "verified": len(v["verified"])} for k, v in sorted(by_attempt.items())},
            "reasoning_share_by_characters": reasoning / max(1, reasoning + sum(x["content_chars"] for x in replies)),
            "on_tasks_solved": sum(r.spent for r in runs if r.id in solved),
            "on_tasks_not_solved": sum(r.spent for r in runs if r.id not in solved),
            "after_first_verified": sum(x["completion_tokens"] for r in runs for x in r.replies
                                        if r.first_verified_attempt and x["attempt"] > r.first_verified_attempt),
            "reply_tokens": percentiles([x["completion_tokens"] for x in replies]),
            "truncated_replies": sum(x["finish"] == "length" for x in replies),
        },
        "replies": len(replies), "candidates": sum(len(r.candidates) for r in runs),
        "request_errors": sum(r.errors for r in runs),
    }


def percentiles(values):
    values = sorted(values)
    return {p: values[min(len(values) - 1, int(len(values) * q))] for p, q in
            (("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("max", 1.0))} if values else {}


class Engine:
    """Runs task replies against the server. ``complete(messages, max_tokens, settings)`` and ``count(messages,
    settings)`` are the transport (``arc2.trial`` supplies the gated HTTP ones, the tests a scripted one);
    ``execute(code, inputs)`` runs a program."""

    holds = False  # True (``arc2.conv``): a roster task keeps its slot until it compacts or ends

    def __init__(self, *, complete, count, execute, window, cap, settings=None, deadline=None, record=None,
                 explore=None, drop=None):
        self.complete, self.count, self.execute, self.explore = complete, count, execute, explore
        self.drop = drop or (lambda request_id: None)  # gives back a held reservation (``scouts``' side requests)
        self.window, self.cap = int(window), int(cap)
        self.settings = {**DEFAULTS, **(settings or {})}
        self.deadline, self.record = deadline, record or (lambda row: None)
        self.inflight = {}  # task -> request time
        self.lessons = []  # (task id without the copy suffix, lesson), shared by every task of the engine
        self.known = {}  # every task advanced so far (late_after counts the roster's tokens over them)
        self.library = []  # ``examples``: verified training-task programs (arc2.trial loads data/arc2/library.json)
        self.answers = {}  # ``direct_answer_share``: task (no copy suffix) -> copy 1's direct-answer candidate
        self.pools = {}  # ``pool``: (arm, task) -> programs seen so far; ("", task) from the trial's pool_seed

    def start_attempt(self, run):
        run.attempts += 1
        run.attempt_replies = 0
        run.cells, run.attempt_tools, run.attempt_empty = [], 0, 0
        settings = {**self.settings, **run.arm}
        tool = settings["tool"] and self.explore is not None
        run.copy = (run.attempts - 1 + run.copy_offset) % int(settings["augment"])
        run.view = augment_view(run.task, run.id, run.copy) if run.copy else run.task
        run.hidden, run.shown = None, run.view
        if settings["direct_answer"] and not run.answered and (not settings["direct_answer_share"] or copy_index(run.id) == 1):  # one direct reply first (official ARC way:
            # DIRECT_SYSTEM, text grids in the task's own frame, no tool, ``direct_answer_effort``, no cap below the
            # window); its grid is a second answer only (guesses). The harness's copy rotation is unchanged.
            run.answered, run.copy_offset = True, run.copy_offset - 1
            run.direct, run.copy, run.view, run.shown = "answer", 0, run.task, run.task
            run.messages = [{"role": "system", "content": tasks.DIRECT_SYSTEM}, {"role": "user", "content": tasks.task_prompt(run.task)}]
            return
        if settings["holdout"] and len(run.view["train"]) >= 3:  # one training pair kept back from the prompt
            run.hidden = (run.attempts - 1) % len(run.view["train"])
            run.shown = {**run.view, "train": [p for i, p in enumerate(run.view["train"]) if i != run.hidden]}
        pictures = bool(settings["pictures"] and settings["images"] and tool)  # tasks.PICTURES_HEAD: no text grids
        first = tasks.pictures_prompt(run.shown) if pictures else \
            tasks.task_prompt(run.shown, views=settings["views"], columns=settings["columns"])
        if settings["repair"] and not run.repaired and len(run.verified_outputs()) == 1:
            tests = json.loads(next(iter(run.verified_outputs())))
            if all(tasks.valid(g) for g in tests):  # one reply: the test grids with the verified output in view
                run.repaired, run.direct, run.copy, run.view = True, "repair", 0, run.task
                first = tasks.repair_prompt(run.task, tests)
                if settings["images"]:
                    side = int(settings["image_side"])
                    first = ([{"type": "text", "text": first}] + tasks.image_parts(run.task, side, bool(settings["image_changes"]))
                             + tasks.review_parts(run.task, tests, side))
                run.messages = [{"role": "system", "content": tasks.DIRECT_SYSTEM}, {"role": "user", "content": first}]
                return
        every = int(settings["direct_every"])
        open_task = not any(c["verified"] and not (c.get("review") or c.get("second")) for c in run.candidates)
        run.direct = "every" if open_task and ((every and run.attempts % every == 0) or run.direct_next) else False
        run.direct_next = False
        if run.direct:  # transduction: the model gives the test outputs, no program, no tool
            run.messages = [{"role": "system", "content": tasks.DIRECT_SYSTEM}, {"role": "user", "content": first}]
            return
        verified = next((c for c in run.candidates if c["verified"] and not (c.get("review") or c.get("second"))), None)
        if verified is not None and settings["confirm_style"] == "diverse":
            readings = []  # every distinct verified reading, in this copy's frame
            for c in run.candidates:
                if c["verified"] and not (c.get("review") or c.get("second")) and c["tests"] not in readings:
                    readings.append(c["tests"])
            if run.copy:
                readings = [[augment_view({"train": [], "test": [{"input": g}]}, run.id, run.copy)["test"][0]["input"]
                             for g in r] for r in readings]
            first = tasks.diverse_prompt(run.shown, readings)
        elif verified is not None and settings["confirm_style"] == "other":
            first = tasks.other_prompt(run.shown, verified["tests"])
        elif verified is None and settings["carry"] and (run.notes or run.best):
            first = tasks.carry_prompt(run.shown, run.notes[-settings["notes_kept"]:], run.best)
        run.pooled, run.judged, run.ranking = 0, None, None
        run.rejects, run.known = 0, None
        if settings["reject_known"] and run.hidden is None:  # earlier attempts' readings: drawn like judge_pool
            run.known = {row["tests"] if isinstance(row["tests"], str) else json.dumps(row["tests"])
                         for row, _ in self.judge_pool(run, int(settings["known_k"]))}
            seen = [c["tests"] for c in run.candidates if c["verified"] and c["attempt"] < run.attempts]
            run.known |= {json.dumps(t) for t in seen}  # and this run's own earlier verified readings
        if settings["pool"] and run.attempts >= int(settings["pool_after"]) and run.hidden is None:
            check = settings["pool_style"] == "check"
            picked = [] if settings["pool_style"] == "judge" else \
                self.pick_pool(run, 1 if check else int(settings["pool"]), verified_only=check)
            if settings["pool_style"] == "judge":
                groups = self.judge_pool(run, int(settings["pool"]))
                if groups:
                    shown = []
                    for row, n in groups:
                        outs = json.loads(row["tests"]) if isinstance(row["tests"], str) else row["tests"]
                        if run.copy:
                            outs = [augment_view({"train": [], "test": [{"input": g}]}, run.id, run.copy)["test"][0]["input"]
                                    for g in outs]
                        shown.append((row["code"], outs, n))
                    first, run.pooled = tasks.judge_prompt(run.shown, shown), sum(n for _, n in groups)
                    run.judged = [[hashlib.sha256((row["tests"] if isinstance(row["tests"], str) else json.dumps(
                        row["tests"])).encode()).hexdigest()[:12], n] for row, n in groups]
            if picked and check:
                outs = json.loads(picked[0]["tests"]) if isinstance(picked[0]["tests"], str) else picked[0]["tests"]
                if run.copy:
                    outs = [augment_view({"train": [], "test": [{"input": g}]}, run.id, run.copy)["test"][0]["input"]
                            for g in outs]
                first, run.pooled = tasks.check_prompt(run.shown, picked[0]["code"], outs), 1
            elif picked:
                first, run.pooled = tasks.pool_prompt(run.shown, picked), len(picked)
        if settings["examples"] and self.library:  # worked examples from training tasks, nearest first
            first = tasks.examples_prompt(tasks.pick_examples(run.task, self.library, int(settings["examples"]))) + first
        if settings["lessons"]:
            own = run.id.split("#")[0]  # a roster copy of the same task never sees the other copy's lesson
            kept = [text for task, text in self.lessons if task != own][-settings["lessons_kept"]:]
            if kept:
                first = tasks.LESSONS.format(lessons="\n".join(f"- {t}" for t in kept)) + first
        system = (tasks.SYSTEM_TOOL if tool else tasks.SYSTEM) + ({"arc2": tasks.ARC2_HINT, "arc2b": tasks.ARC2_HINT_B.format(s="s")}.get(settings["prompt"], "") if settings["prompt"] != "rotate" else tasks.LENSES[(run.attempts - 1) % len(tasks.LENSES)])
        if settings["tool_fixes"] and tool:
            for old, new in tasks.TOOL_FIXES:
                system = system.replace(old, new, 1)
        if settings["helpers"]:
            system = system.replace("- A run on one grid", tasks.HELPERS + "\n- A run on one grid", 1)
        if settings["views"]:
            system = system.replace("\n- numpy", tasks.VIEWS_NOTE + "\n- numpy", 1)
        if settings["show"] and tool:
            system = system.replace("\n- numpy", tasks.SHOW_NOTE + "\n- numpy", 1)
        if settings["minds"] == 2 and run.attempts % 2 == 0:
            system += tasks.MIND_B
        if run.hidden is not None:
            first += tasks.HOLDOUT_NOTE
        if settings["two_readings"]:  # each reply may end on a second program for the other reading (Engine.take_reply)
            first += tasks.TWO_READINGS.format(s="s" if len(run.view["test"]) > 1 else "")
        if settings["hypotheses"]:
            first += tasks.HYPOTHESES.format(s="s" if len(run.view["test"]) > 1 else "")
        if run.scout and run.scout.get("rule"):  # ``scouts``: the judged rule, to check before the program
            first += tasks.SCOUT_PICK.format(rule=run.scout["rule"])
        if settings["anchor"]:
            first += tasks.ANCHOR.format(s="s" if len(run.view["test"]) > 1 else "")
        if settings["images"]:  # the puzzle also as pictures (tasks.image_parts)
            first = [{"type": "text", "text": first}] + tasks.image_parts(run.shown, int(settings["image_side"]), bool(settings["image_changes"]),
                                                                          note=tasks.PICTURES_NOTE if pictures else None)
        run.messages = [{"role": "system", "content": system},
                        {"role": "user", "content": first}]

    async def scout_rules(self, run, settings):
        """``scouts`` (n): before the attempt, n independent short replies state the rule in words (no code, effort
        ``scout_effort``, every second one without the pictures), then one reply picks the most plausible of them
        (the attempt's effort); the attempt's first message carries the pick (tasks.SCOUT_PICK). Their completion
        tokens count as the task's."""
        task, s = run.task, "s" if len(run.task["test"]) > 1 else ""
        text = tasks.task_prompt(task) + tasks.SCOUT.format(s=s)
        sent = {**settings, "tool": False, "effort": settings["scout_effort"]}

        async def ask(key, content, cap, how):
            messages = [{"role": "system", "content": tasks.SCOUT_SYSTEM}, {"role": "user", "content": content}]
            try:
                count = await self.count(messages, how)
                reply = await self.complete(key, messages, count, min(int(cap), max(256, self.window - count)), how)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a lost scout is left out
                self.record({"event": "scout_error", "task": run.id, "error": f"{type(exc).__name__}: {exc}"[:300]})
                return None
            finally:
                self.drop(key)
            usage = reply.get("usage") or {}
            run.spent += int(usage.get("completion_tokens") or 0)
            run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            return reply["message"]

        jobs = []
        for i in range(int(settings["scouts"])):
            content = text
            if settings["images"] and i % 2 == 0:
                content = [{"type": "text", "text": text}] + tasks.image_parts(task, int(settings["image_side"]),
                                                                              bool(settings["image_changes"]))
            jobs.append(ask(f"{run.id}~s{run.attempts + 1}.{i}", content, settings["scout_tokens"], sent))
        before = run.spent
        rules = []
        for message in await asyncio.gather(*jobs):
            rule = ((message or {}).get("content") or "").strip()
            thought = ((message or {}).get("reasoning") or (message or {}).get("reasoning_content") or "").strip()
            if not rule and thought and settings["scout_unfinished"]:  # cut while thinking: its last lines
                rule = tasks.SCOUT_UNFINISHED + thought[-tasks.SCOUT_CHARS:]
            if rule:
                rules.append(rule[:tasks.SCOUT_CHARS + len(tasks.SCOUT_UNFINISHED)])
        pick = 0 if len(rules) == 1 else None
        if len(rules) > 1:
            message = await ask(f"{run.id}~j{run.attempts + 1}", tasks.scout_judge_prompt(task, rules),
                                settings["scout_judge_tokens"], {**settings, "tool": False})
            for field in ("content", "reasoning", "reasoning_content"):
                found = re.findall(r"PICK:\s*\**\s*([A-Z])\b", (message or {}).get(field) or "")
                if found and ord(found[-1]) - 65 < len(rules):
                    pick = ord(found[-1]) - 65
                    break
        run.scout = {"rules": rules, "pick": pick, "rule": rules[pick] if pick is not None else None,
                     "tokens": run.spent - before}

    def judge_pool(self, run, k):
        """``pool_style`` "judge" (item 3): k earlier verified programs of the task on this view drawn at random (a
        reading's share is its share of attempts), grouped by test reading: [(one program, attempts)], shuffled."""
        base, arm = run.id.split("#")[0].split("@")[0], run.arm.get("name", "")
        rows = [r for r in self.pools.get(("", base), []) + (self.pools.get((arm, base), []) if arm else [])
                if r["view"] == run.copy and r["verified"]]
        rng = random.Random(f"{run.id}/{run.attempts}")
        groups = {}
        for r in rng.sample(rows, min(k, len(rows))):
            key = r["tests"] if isinstance(r["tests"], str) else json.dumps(r["tests"])
            groups.setdefault(key, []).append(r)
        out = [(rs[0], len(rs)) for rs in groups.values()]
        rng.shuffle(out)
        return out

    def pick_pool(self, run, k, verified_only=False):
        """Up to ``k`` earlier programs of the task on this attempt's view: one per distinct verified test reading
        (shuffled by the run's id and attempt), then the others by training pairs passed and cell accuracy."""
        base, arm = run.id.split("#")[0].split("@")[0], run.arm.get("name", "")
        rows = [r for r in self.pools.get(("", base), []) + (self.pools.get((arm, base), []) if arm else []) if r["view"] == run.copy]
        rng = random.Random(f"{run.id}/{run.attempts}")
        readings, others, seen = {}, [], set()
        for r in rows:
            if r["code"] in seen:
                continue
            seen.add(r["code"])
            if r["verified"]:
                key = r["tests"] if isinstance(r["tests"], str) else json.dumps(r["tests"])
                readings.setdefault(key, []).append(r)
            else:
                others.append(r)
        picked = [rng.choice(v) for v in readings.values()]
        rng.shuffle(picked)
        if verified_only:
            return picked[:k]
        others.sort(key=lambda r: (-r["passed"] / max(1, r["pairs"]), -r["accuracy"]))
        return (picked + others)[:k]

    def phase(self, run):
        """"open" without a verified program, "confirm" while the confirming attempts run, then "settled"."""
        if run.first_verified_attempt is None and not (run.shared and run.verified_outputs()):
            return "open"  # twins: the other run's verified program counts here too
        settings = {**self.settings, **run.arm}  # scheduler keys may come from the arm too (a roster `set`)
        settled = len(run.verified_outputs()) >= 2 or run.confirms >= settings["confirm_attempts"]
        if int(settings["augment"]) > 1:  # two copies' verified programs agree: settled
            agree = {}
            for c in run.candidates:
                if c["verified"] and not (c.get("review") or c.get("second")):
                    agree.setdefault(json.dumps(c["tests"]), set()).add(c.get("copy", 0))
            settled = settled or any(len(v) >= 2 for v in agree.values())
        if settings["settle_agree"]:
            settled = settled or sum(c["verified"] and not (c.get("review") or c.get("second")) for c in run.candidates) >= 2
        return "settled" if settled else "confirm"

    def priority(self, run):
        """Roster order: the lowest goes next (module docstring)."""
        settings = {**self.settings, **run.arm}
        delay = {"open": 0, "confirm": settings["confirm_delay"], "settled": settings["settled_delay"]}
        phase = self.phase(run)
        if phase == "open" and settings["progress_weight"] and run.candidates:
            share = max((c["passed"] / c["pairs"] for c in run.candidates if c["pairs"]), default=0)
            return run.spent * (1 - settings["progress_weight"] * share)
        return run.spent + delay[phase]

    def end_attempt(self, run, verified):
        """Close the attempt; decide whether the task ends (module docstring)."""
        run.messages, run.review_pending, run.cut_pending = None, False, False
        if run.first_verified_attempt is None and verified:
            run.first_verified_attempt = run.attempts
        elif run.first_verified_attempt is not None or (run.shared and run.verified_outputs()):
            run.confirms += 1  # twins: an attempt after the other run's verified program confirms
        if {**self.settings, **run.arm}["settled_delay"] is None and self.phase(run) == "settled":
            run.done = "verified"

    def next_review(self, run, settings):
        """``review_style`` "check+second": after the check's reply, the next style's turn on the outputs it left;
        False when no turn is left (the attempt ends)."""
        if not run.review_next:
            return False
        style = run.review_next.pop(0)
        content = tasks.review_prompt(run.view, run.review_tests, style)
        pictures = tasks.review_parts(run.view, run.review_tests, int(settings["image_side"])) \
            if settings["images"] else []
        run.messages.append({"role": "user", "content": [{"type": "text", "text": content}] + pictures
                             if pictures else content})
        return True

    def attach_answer(self, run, settings):
        """``direct_answer_share``: copy 1's direct grid is also the other copies' second answer, attached when the
        copy is next advanced (at its own spend then, so never earlier on the roster clock than the grid existed)."""
        if not settings["direct_answer_share"] or copy_index(run.id) == 1 or run.answered:
            return
        found = self.answers.get(run.id.split("#")[0])
        if found is not None:
            run.answered = True
            run.candidates.append({**found, "tokens_at": run.spent, "shared": True})

    async def advance(self, run, budget, replies=None):
        """Replies until the task ends, has spent ``budget`` completion tokens (checked between replies) or has
        taken ``replies`` more replies. With ``replies`` (a roster turn) it also returns when an attempt ends
        without a verified program, before the next attempt starts: a worker that keeps an attempt to its end
        (``sticky_attempts``) otherwise never gives an unsolved task back, and with every worker on one the
        rest of the roster is never started (the full-length L4 run: 98 of 240 tasks, 17 tasks with 57% of
        the tokens)."""
        settings = {**self.settings, **run.arm}
        self.known[run.id] = run
        self.attach_answer(run, settings)
        last = None if replies is None else len(run.replies) + replies
        while run.done is None and run.spent < budget and (last is None or len(run.replies) < last):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return
            if run.messages is None:
                if settings["max_attempts"] and run.attempts >= int(settings["max_attempts"]):
                    run.done = "attempts"  # arc2.rl_loop: an episode is one attempt
                    return
                if (settings["scouts"] and not run.direct_next
                        and (run.attempts + run.copy_offset) % int(settings["augment"]) == 0):  # rules in the task's own frame
                    await self.scout_rules(run, settings)
                self.start_attempt(run)
            try:
                count = await self.count(run.messages, settings)
                room = self.window - count
                if run.attempt_replies >= settings["attempt_replies"] or room < settings["min_room_tokens"]:
                    if run.attempt_replies == 0 and not run.attempt_tools:
                        run.done = "task_too_long"  # the bare task does not fit the window
                        return
                    if (settings["direct_after"] and not run.direct and run.first_verified_attempt is None
                            and not run.review_pending and not run.lesson_pending and not run.note_pending
                            and room < settings["direct_tokens"] + 256):
                        run.direct_next = True  # no room for the grid reply here: a fresh one-reply attempt next
                    if (settings["direct_after"] and not run.direct and run.first_verified_attempt is None
                            and not run.review_pending and not run.lesson_pending and not run.note_pending
                            and room >= settings["direct_tokens"] + 256):
                        run.direct = "after"  # one reply: the test grids without a program, then the attempt ends
                        run.messages.append({"role": "user", "content": tasks.DIRECT_AFTER})
                    elif (settings["carry"] and not run.note_pending and run.first_verified_attempt is None
                            and not run.review_pending and not run.lesson_pending and room >= settings["note_tokens"] + 256
                            and any(c["attempt"] == run.attempts for c in run.candidates)):
                        run.note_pending = True  # one short reply for the next attempt, then the attempt ends
                        run.messages.append({"role": "user", "content": tasks.NOTE})
                    elif run.lesson_pending and room >= settings["lesson_tokens"] + 256:
                        pass  # the lesson reply still runs (the attempt's replies are its programs)
                    elif not run.review_pending or room < settings["min_room_tokens"]:
                        run.note_pending = False
                        verified = run.review_pending or run.lesson_pending
                        run.lesson_pending = False
                        self.end_attempt(run, verified)  # no room for the review: the program stands
                        if replies is not None:
                            return  # roster order: the next attempt waits its turn (docstring)
                        continue
                started = self.inflight[run.id] = time.monotonic()
                cap = (settings["note_tokens"] if run.note_pending else settings["lesson_tokens"] if run.lesson_pending
                       else settings["direct_tokens"] if run.direct == "after" else room if run.direct == "answer"
                       else settings["cap"] or self.cap)
                late = settings["late_after"] and sum(r.spent for r in self.known.values()) >= settings["late_after"]
                if late and settings["late_cap"] and not (run.note_pending or run.lesson_pending or run.direct in ("after", "answer")):
                    cap = int(settings["late_cap"])  # the roster is past late_after tokens: longer replies
                sent = {**settings, "tool": False} if run.direct in ("every", "repair") else settings
                if run.direct == "answer":
                    sent = {**settings, "tool": False, "effort": settings["direct_answer_effort"] or settings["effort"],
                            "gate_reserve": settings["direct_answer_reserve"]}
                if settings["confirm_effort"] and run.first_verified_attempt is not None:
                    sent = {**sent, "effort": settings["confirm_effort"]}  # confirming attempts think less
                elif (settings["hard_effort"] and run.first_verified_attempt is None
                        and run.attempts > int(settings["hard_after"])):
                    sent = {**sent, "effort": settings["hard_effort"]}  # a task still open after hard_after attempts
                if settings["open_schedule"] and run.first_verified_attempt is None and not run.direct:
                    effort, step_cap = settings["open_schedule"][min(run.attempts, len(settings["open_schedule"])) - 1]
                    sent = {**sent, "effort": effort}  # open tasks: effort and cap by attempt number
                    if not (run.note_pending or run.lesson_pending):
                        cap = int(step_cap)
                if late and settings["late_effort"]:
                    sent = {**sent, "effort": settings["late_effort"]}  # and deeper thinking, every task
                if settings["tool_fixes"] and run.attempt_tools >= int(settings["tool_calls"]):
                    sent = {**sent, "tool": False}  # the attempt's calls are used up: no `tools` offered
                if run.cut_pending:
                    cap = min(cap, int(settings["cut_program"]))
                    sent = {**sent, "tool": False, "effort": settings["cut_effort"] or sent["effort"]}
                reply = await self.complete(run.id, run.messages, count, min(cap, room), sent)
                self.inflight.pop(run.id)
            except asyncio.CancelledError:
                raise  # the wall: the request stays in ``inflight``
            except StopRequested:
                self.inflight.pop(run.id, None)
                return
            except ContextFull:
                self.inflight.pop(run.id, None)
                if run.attempt_replies == 0 and not run.attempt_tools:
                    run.done = "task_too_long"
                    return
                run.note_pending = False
                verified = run.review_pending or run.lesson_pending
                run.lesson_pending = False
                self.end_attempt(run, verified)
                if replies is not None:
                    return
                continue
            except Exception as exc:  # noqa: BLE001 - a failed request is retried on the same conversation
                if self.deadline is None or time.monotonic() < self.deadline - 5:
                    self.inflight.pop(run.id, None)
                if self.deadline is not None and time.monotonic() >= self.deadline - 5:
                    return  # cut by the wall, not a failure
                run.errors += 1
                self.record({"event": "request_error", "task": run.id, "error": f"{type(exc).__name__}: {exc}"[:400]})
                if run.errors >= settings["errors_per_task"]:
                    run.done = "errors"
                await asyncio.sleep(min(30, 2 ** run.errors))
                continue
            await self.take_reply(run, reply, reply.get("t0", started), reply.get("t1", time.monotonic()))

    async def take_reply(self, run, reply, t0, t1):
        message, usage = reply["message"], reply.get("usage") or {}
        phase = self.phase(run)
        content = message.get("content") or ""
        if run.judged and run.ranking is None:  # pool_style "judge": the reply's RANKING line, letters -> readings
            pattern = r"RANKING:\s*([A-Z])(?:\s*,\s*([A-Z]))?"
            found = re.search(pattern, content) or ([None] + list(re.finditer(
                pattern, message.get("reasoning") or message.get("reasoning_content") or "")))[-1]
            if found:
                run.ranking = [ord(x) - 65 for x in found.groups() if x and ord(x) - 65 < len(run.judged)]
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        run.spent += int(usage.get("completion_tokens") or 0)
        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        run.attempt_replies += 1
        entry = {"role": "assistant", "content": content}
        if reasoning:
            entry["reasoning"] = reasoning  # earlier reasoning stays in the context, as in the arena
        run.messages.append(entry)
        row = {"event": "reply", "task": run.id, "attempt": run.attempts, "reply": run.attempt_replies,
               "completion_tokens": int(usage.get("completion_tokens") or 0),
               "prompt_tokens": int(usage.get("prompt_tokens") or 0), "tokens_at": run.spent,
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "finish": reply.get("finish_reason"), "seconds": round(t1 - t0, 2), "t0": t0, "t1": t1, "phase": phase,
               "reasoning_chars": len(reasoning), "content_chars": len(content), "content": content,
               "reasoning": reasoning, **({"copy": run.copy} if run.copy else {})}
        settings = {**self.settings, **run.arm}
        salvage, run.cut_pending = run.cut_pending, False
        if run.note_pending:  # the attempt's closing note (carry): kept for the next attempt, the attempt ends
            run.note_pending = False
            note = " ".join((content or reasoning[-800:]).split())[:900]
            if note:
                run.notes.append(note)
            row.update(outcome="note", note=note)
            self.keep(run, row)
            self.end_attempt(run, False)
            return
        if run.lesson_pending:  # the lesson after the first verified program (lessons): shared, the attempt ends
            run.lesson_pending = False
            lesson = " ".join((content or reasoning[-600:]).split())[:500]
            if lesson:
                self.lessons.append((run.id.split("#")[0], lesson))
            row.update(outcome="lesson", lesson=lesson)
            self.keep(run, row)
            self.end_attempt(run, True)
            return
        if run.direct:  # one reply: its grids are a candidate (never verified), the attempt ends
            grids = tasks.direct_grids(content, len(run.view["test"]))
            if grids is not None and any(g is not None for g in grids):
                run.candidates.append({
                    "code": f"direct-{run.attempts}",  # one vote per attempt that gave the grid
                    "attempt": run.attempts, "reply": run.attempt_replies, "tokens_at": run.spent, "t": t1,
                    "passed": 0, "pairs": 0, "accuracy": 0.0, "verified": False, "direct": True,
                    "tests": [augment_back(g, run.id, run.copy) for g in grids],
                    **({"copy": run.copy} if run.copy else {}), **({"repair": True} if run.direct == "repair" else {}),
                    **({"answer": True} if run.direct == "answer" else {})})
                if run.direct == "answer":
                    self.answers[run.id.split("#")[0]] = run.candidates[-1]
            row.update(outcome="direct" if grids is not None else
                       "truncated" if reply.get("finish_reason") == "length" else "no_code")
            self.keep(run, row)
            self.end_attempt(run, False)
            return
        if settings["anchor"] and run.predicted != run.attempts and not run.review_pending:
            grids = tasks.predicted_grids(content, len(run.view["test"]))
            if grids is not None and any(g is not None for g in grids):  # the attempt's prediction: a candidate
                run.predicted = run.attempts
                run.candidates.append({
                    "code": f"predict-{run.attempts}", "attempt": run.attempts, "reply": run.attempt_replies,
                    "tokens_at": run.spent, "t": t1, "passed": 0, "pairs": 0, "accuracy": 0.0, "verified": False,
                    "direct": True, "predict": True,
                    "tests": [augment_back(g, run.id, run.copy) if g is not None else None for g in grids],
                    **({"copy": run.copy} if run.copy else {})})
                row["predicted"] = sum(g is not None for g in grids)
        call = tasks.tool_call(message) if settings["tool"] and self.explore and not run.review_pending \
            and not salvage else None
        if call is not None:  # a `python` call: run it in the attempt's session, the result goes back
            kept, snippet = call
            entry["tool_calls"] = [kept]
            run.attempt_tools += 1
            if settings["tool_fixes"] and snippet is None and run.attempt_empty < EMPTY_FREE:
                run.attempt_empty += 1
                run.attempt_tools -= 1  # an empty call is answered without counting against the limit
            if run.attempt_tools > settings["tool_calls"]:
                text = tasks.TOOL_LIMIT  # counts as a reply: the attempt cannot go on calling
            else:
                run.attempt_replies -= 1  # the attempt's replies are its programs
                if snippet is None:
                    text = "Error: the call needs one string argument `code`."
                else:
                    result = await self.explore(run.cells, snippet, run.shown,
                                                **({"helpers": True} if settings["helpers"] else {}),
                                                **({"show": True} if settings["show"] else {}),
                                                **({"fixes": True, "chars": tasks.TOOL_CHARS}
                                                   if settings["tool_fixes"] else {}))
                    if not result.get("error"):
                        run.cells.append(snippet)
                    elif result.get("kept"):
                        run.cells.append(result["kept"])  # tool_fixes: what ran before the failing statement
                    text = tasks.tool_text(result, tasks.TOOL_CHARS if settings["tool_fixes"] else 0)
                if settings["tool_fixes"] and run.attempt_tools == int(settings["tool_calls"]):
                    text += "\n\n" + tasks.TOOL_LAST
            run.messages.append({"role": "tool", "tool_call_id": kept["id"], "content": text})
            pictures = tasks.shown_parts(result.get("shown") or []) if call is not None and snippet is not None \
                and run.attempt_tools <= settings["tool_calls"] else None
            if pictures:  # setting ``show``: the grids the snippet showed, as pictures
                run.messages.append({"role": "user", "content": pictures})
                row["shown"] = len(pictures) - 1
            row.update(outcome="tool", tool_code=snippet, tool_result=text)
            self.keep(run, row)
            return
        code, second = tasks.extract_code(content), None
        if settings["two_readings"] and not run.review_pending:
            blocks = tasks.extract_codes(content)
            if len(blocks) >= 2 and blocks[-1] != blocks[-2]:
                code, second = blocks[-2], blocks[-1]
        review = run.review_pending
        if review and code is None:  # KEEP, or cut at the cap: the verified program stands
            row.update(outcome="review_cut" if reply.get("finish_reason") == "length" else "review_keep", review=True)
            self.keep(run, row)
            if not self.next_review(run, settings):
                self.end_attempt(run, True)
            return
        if code is None:
            row["outcome"] = "truncated" if reply.get("finish_reason") == "length" else "no_code"
            if salvage:  # cut_program: the cut attempt's last reply gave no program
                row["salvage"] = True
                self.keep(run, row)
                self.end_attempt(run, False)
                return
            if row["outcome"] == "truncated" and settings["cut_program"] and not settings["truncated_continue"]:
                run.cut_pending = True
            run.messages.append({"role": "user", "content": (tasks.CONTINUE if settings["truncated_continue"]
                                                             else tasks.TRUNCATED) if row["outcome"] == "truncated"
                                 else tasks.NO_CODE_TOOL if call is None and settings["tool"] and self.explore
                                 and not (settings["tool_fixes"] and run.attempt_tools >= int(settings["tool_calls"]))
                                 else tasks.NO_CODE})
            self.keep(run, row)
            return
        inputs = [p["input"] for p in run.view["train"]] + [t["input"] for t in run.view["test"]]
        result = await self.execute(code, inputs, **({"helpers": True} if settings["helpers"] else {}),
                                    **({"fixes": True} if settings["tool_fixes"] else {}))
        verdict = tasks.judge(run.view, result)
        run.candidates.append({
            "code": hashlib.sha256(code.encode()).hexdigest()[:16], "attempt": run.attempts,
            "reply": run.attempt_replies, "tokens_at": run.spent, "t": t1, "passed": verdict["passed"],
            "pairs": verdict["pairs"], "accuracy": verdict["accuracy"], "verified": verdict["verified"],
            "tests": [augment_back(t, run.id, run.copy) for t in verdict["tests"]],
            **({"copy": run.copy} if run.copy else {}), **({"review": True} if review else {}),
            **({"pooled": run.pooled} if run.pooled else {}), **({"judged": run.judged} if run.judged else {}),
            **({"ranking": run.ranking} if run.ranking is not None else {})})
        if second is not None:  # ``two_readings``: the other reading's program is a candidate; feedback is on the first
            other = await self.execute(second, inputs, **({"helpers": True} if settings["helpers"] else {}),
                                       **({"fixes": True} if settings["tool_fixes"] else {}))
            seen2 = tasks.judge(run.view, other)
            run.candidates.append({
                "code": hashlib.sha256(second.encode()).hexdigest()[:16], "attempt": run.attempts,
                "reply": run.attempt_replies, "tokens_at": run.spent, "t": t1, "passed": seen2["passed"],
                "pairs": seen2["pairs"], "accuracy": seen2["accuracy"], "verified": seen2["verified"],
                "tests": [augment_back(t, run.id, run.copy) for t in seen2["tests"]], "second": True,
                **({"copy": run.copy} if run.copy else {})})
            row.update(second_verified=seen2["verified"], second_code=second)
        if settings["pool"] and settings["pool_live"] and not review:
            self.pools.setdefault((run.arm.get("name", ""), run.id.split("#")[0].split("@")[0]), []).append({
                "view": run.copy, "code": code, "passed": verdict["passed"], "pairs": verdict["pairs"],
                "accuracy": verdict["accuracy"], "verified": verdict["verified"],
                "tests": json.dumps(run.candidates[-1]["tests"])})
        row.update(outcome="verified" if verdict["verified"] else "run_error" if result.get("error") else "wrong",
                   passed=verdict["passed"], pairs=verdict["pairs"], accuracy=round(verdict["accuracy"], 4),
                   code=code, run_error=result.get("error"))
        if review:  # one reply: a correction is a candidate whether or not it still passes the pairs
            row.update(outcome="review_" + row["outcome"], review=True)
            self.keep(run, row)
            if verdict["verified"]:
                run.review_tests = verdict["tests"]
            if not self.next_review(run, settings):
                self.end_attempt(run, True)
            return
        if (verdict["verified"] and run.known and run.rejects < int(settings["reject_known"])
                and json.dumps(run.candidates[-1]["tests"]) in run.known):
            run.rejects += 1  # a reading already found: the program stays a candidate; ask for a different rule
            run.candidates[-1]["known"] = True
            row.update(outcome="verified_known")
            run.messages.append({"role": "user", "content": tasks.KNOWN_READING.format(
                s="s" if len(run.view["test"]) > 1 else "")})
            self.keep(run, row)
            return
        if verdict["verified"]:
            self.keep(run, row)
            if {**self.settings, **run.arm}["review"] and run.first_verified_attempt is None:
                run.review_pending = True
                first, *run.review_next = str({**self.settings, **run.arm}["review_style"]).split("+")
                run.review_tests = verdict["tests"]
                content = tasks.review_prompt(run.view, verdict["tests"], first)
                pictures = tasks.review_parts(run.view, verdict["tests"], int(settings["image_side"])) \
                    if settings["images"] else []
                run.messages.append({"role": "user", "content": [{"type": "text", "text": content}] + pictures
                                     if pictures else content})
                return
            if settings["lessons"] and run.first_verified_attempt is None:
                run.lesson_pending = True
                run.messages.append({"role": "user", "content": tasks.LESSON})
                return
            self.end_attempt(run, True)
            return
        if salvage:  # cut_program: the cut attempt's program did not verify, the attempt ends
            row["salvage"] = True
            self.keep(run, row)
            self.end_attempt(run, False)
            return
        shown, seen = run.view, verdict  # holdout: feedback on the shown pairs only, the hidden one never shown
        if run.hidden is not None and not result.get("error") and result.get("outputs"):
            outs = result["outputs"]
            shown = run.shown
            result = {**result, "outputs": outs[:run.hidden] + outs[run.hidden + 1:]}
            seen = tasks.judge(shown, result)
        if run.hidden is not None and seen["verified"]:
            row["feedback"] = tasks.HOLDOUT_FAIL.format(n=len(shown["train"]))
            row["holdout_fail"] = True
        else:
            row["feedback"] = tasks.feedback(shown, result, seen)
            failing = [i for i, (ok, _, _) in enumerate(seen["rows"]) if not ok] if not result.get("error") else []
            if settings["trace"] and failing and run.hidden is None:  # the program's steps on its first failing pair
                traced = await self.execute(code, [shown["train"][failing[0]]["input"]], trace=0,
                                            **({"helpers": True} if settings["helpers"] else {}),
                                            **({"fixes": True} if settings["tool_fixes"] else {}))
                if traced.get("trace"):
                    row["feedback"] = row["feedback"].replace(tasks.FIX_LINE, tasks.TRACE.format(
                        i=failing[0] + 1, trace=traced["trace"]) + tasks.FIX_LINE)
                    row["traced"] = True
        if settings["carry"] and (run.best is None or (verdict["passed"], verdict["accuracy"])
                                  > (run.best["passed"], run.best["accuracy"])):
            run.best = {"code": code, "passed": verdict["passed"], "pairs": verdict["pairs"],
                        "accuracy": verdict["accuracy"], "feedback": row["feedback"]}
        pictures = tasks.feedback_parts(shown, seen, int(settings["image_side"])) \
            if settings["images"] and settings["image_feedback"] and not result.get("error") and not row.get("holdout_fail") else []
        run.messages.append({"role": "user", "content": [{"type": "text", "text": row["feedback"]}] + pictures
                             if pictures else row["feedback"]})
        self.keep(run, row)

    def keep(self, run, row):
        run.replies.append({k: row[k] for k in ("task", "attempt", "reply", "completion_tokens", "prompt_tokens",
                                                "tokens_at", "finish", "seconds", "t0", "t1", "phase",
                                                "reasoning_chars", "content_chars", "outcome")})
        self.record(row)
