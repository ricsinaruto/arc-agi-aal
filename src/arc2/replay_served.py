"""Rebuild a finished conversation run's served inputs (SFT reuse of runs from before arc2.served).

The run's own harness (``src`` of its bundle commit) replays every task's conversation with the recorded replies in
place of the server: the interpreter and the program checks run again, ``count`` returns the recorded prompt sizes
(and the drain's before/after sizes), reopens happen where the run reopened. Every request the engine would have sent
goes to an ``arc2.served`` log (``served-*.jsonl``), so the result reads like a live run's. Checks per request: the
replayed tool text equals the recorded ``tool_result``; the request count equals the recorded replies.

usage: python -m arc2.replay_served RUN_FOLDER STUDY_JSON DATA_DIR OUT [--tasks a,b]"""
import asyncio
import json
import sys
from pathlib import Path

from . import tasks as T
from .served import ServedLog


def tool_call(row, n):
    code = row.get("tool_code")
    args = row["bad_args"] if row.get("bad_args") is not None else json.dumps({"code": code or ""})
    if unrecorded(row):  # conv1 era: a call without a string ``code``; its arguments were not recorded
        args = "{}"
    if row.get("bad_args") is not None and row.get("outcome") != "malformed_dropped":
        try:  # bad_args is cut at 2,000 characters: a long call the server parsed must stay parseable here
            json.loads(args)
        except ValueError:
            args = json.dumps({"_cut": args})
    return {"id": f"replay-{n}", "type": "function", "function": {"name": "python", "arguments": args}}


def unrecorded(row):
    return row.get("bad_args") is None and row.get("tool_code") in (None, "None", "") \
        and str(row.get("tool_result") or "").startswith("Error: the call needs one string argument")


def message_of(row, n):
    msg = {"role": "assistant", "content": row.get("content") or None, "reasoning": row.get("reasoning") or ""}
    if row.get("tool_code") is not None or row.get("bad_args") is not None or unrecorded(row):
        msg["tool_calls"] = [tool_call(row, n)]
    return msg


class Replay:
    def __init__(self, rows, other, log, model, effort, temperature, template):
        self.rows, self.other, self.log = rows, other, log  # per task: reply rows; non-reply events in order
        self.model, self.effort, self.temperature, self.template = model, effort, temperature, template
        self.pos, self.mismatch, self.extra, self.queues, self.window = {}, [], [], {}, 0
        self.substituted = []  # tool texts replaced by the recorded ones
        self.probe = None  # --probe-fit: a stream for the train-fit probe rows
        self.equiv = 0.0  # --probe-equiv S: equivariance verdicts at submits, S seconds a function
        self.sweep = 0.0  # --probe-sweep S: parameter sweep of fitting functions at submits, S seconds a function

    def next_row(self, task):
        i = self.pos.get(task, 0)
        return self.rows[task][i] if i < len(self.rows[task]) else None

    def counts(self, task):
        """The sizes the run's count() returned before the next request: a drain's or loop clear's before/after, a
        cap end's context, then the request's own prompt size."""
        i = self.pos.get(task, 0)
        rows = self.rows[task]
        lo = rows[i - 1]["_k"] if i > 0 else -1
        hi = rows[i]["_k"] if i < len(rows) else float("inf")
        out = []
        for e in self.other.get(task, []):
            if lo < e["_k"] < hi:
                if e.get("event") in ("context_drain", "loop_rule_fresh_context"):
                    out += [int(e["tokens_before"]), int(e["tokens_after"])]
                elif e.get("event") == "try_end" and e.get("context") is not None:
                    out.append(int(e["context"]))
        return out

    async def count(self, messages, settings):  # the recorded prompt size of the request about to be sent
        task = settings.get("_task")
        if not task:
            return 0
        key = (task, self.pos.get(task, 0))
        queue = self.queues.setdefault(key, self.counts(task))
        if queue:
            return queue.pop(0)
        row = self.next_row(task)
        return int(row["prompt_tokens"]) if row else int(self.window)

    async def complete(self, task_id, messages, count, max_tokens, settings):
        row = self.next_row(task_id)
        if row is None:
            self.extra.append(task_id)
            raise asyncio.CancelledError()
        self.pos[task_id] = self.pos.get(task_id, 0) + 1
        payload = T.chat_request(self.model, messages, max_tokens,
                                 temperature=settings.get("temperature") or self.temperature,
                                 effort=settings.get("effort") or self.effort,
                                 tools=[T.tool_spec(settings)] if settings.get("tool") else None, template=self.template)
        message = message_of(row, self.pos[task_id])
        if unrecorded(row):  # the call's own text is unknown: the requests after it differ by its tokens
            self.mismatch.append({"task": task_id, "reply": row.get("reply"), "kind": "bad_args_unrecorded"})
        if len(row.get("bad_args") or "") >= 2000 and row.get("outcome") != "malformed_dropped" \
                and not settings.get("bad_call_fix", True):  # the cut arguments went back into the conversation
            self.mismatch.append({"task": task_id, "reply": row.get("reply"), "kind": "bad_args_cut"})
        usage = {"prompt_tokens": row.get("prompt_tokens"), "completion_tokens": row.get("completion_tokens"),
                 "prompt_tokens_details": {"cached_tokens": row.get("cached_tokens")}}
        self.log.write(task_id, payload, {"choices": [{"message": message, "finish_reason": row.get("finish")}],
                                          "usage": usage}, row.get("t0"), row.get("t1"), reply=row.get("reply"))
        return {"message": message, "finish_reason": row.get("finish"), "usage": usage,
                "t0": row.get("t0", 0), "t1": row.get("t1", 0)}


async def replay_task(engine, run, rp):
    """Advance one task through its recorded replies, reopening where it waited for a reopen."""
    while rp.next_row(run.id) is not None and run.done is None:
        if getattr(run, "awaiting_reopen", False):
            engine.pick_reopen([run])
        run.parked = False
        before = rp.pos.get(run.id, 0)
        seen = {id(m) for m in run.messages or []}
        try:
            await engine.advance(run, float("inf"), replies=1)
        except asyncio.CancelledError:  # a request past the last recorded reply (e.g. the compaction summary a
            if rp.next_row(run.id) is not None:  # budget check starts after it, in flight when the run stopped)
                raise
            break
        if rp.pos.get(run.id, 0) == before and not getattr(run, "awaiting_reopen", False):
            break  # the engine ended the task (or a pass) without a request: nothing left to replay
        row = rp.rows[run.id][rp.pos[run.id] - 1] if rp.pos.get(run.id) else None
        if rp.probe is not None and row is not None and row.get("tool_code") and getattr(run, "interp", None) \
                and (not rp.equiv or row.get("submitted")):  # --probe-equiv reads submits only
            # fit1 sim: the model's train-fitting functions after this reply; at submits (rp.equiv > 0) each one's
            # equivariance verdicts too (verifier probe, 30 Oct)
            eq = rp.equiv if row.get("submitted") else 0.0
            fit = run.interp.fit(5.0 + 20 * eq, equiv=eq) if eq else run.interp.fit(5.0)
            truth = [t.get("output") for t in run.task["test"]]
            if rp.sweep and row.get("submitted"):  # item 2 screen: the fitting functions' constants, one at a time
                done = []
                for f in (fit or [])[:6]:
                    source = next((c for c in reversed(getattr(run.interp, "codes", None) or [])
                                   if f"def {f['name']}(" in str(c or "")), None)
                    got = run.interp.sweep(f["name"], source, rp.sweep) if source else None
                    if got is not None:
                        done.append({"name": f["name"], "tests": f["tests"], "tried": got["tried"], "variants": got["tests"]})
                rp.probe.write(json.dumps({"task": run.id, "reply": row.get("reply"), "tokens_at": row.get("tokens_at"),
                                           "sweep": done, "truth": truth, "submit_n": row.get("submit_n")}) + "\n")
            rp.probe.write(json.dumps({"task": run.id, "reply": row.get("reply"), "tokens_at": row.get("tokens_at"),
                                       "submit": bool(row.get("submitted")), "n_fit": None if fit is None else len(fit),
                                       "fit_right": None if not fit else [f["tests"] == truth for f in fit],
                                       "fit_outputs": None if not fit else len({json.dumps(f["tests"]) for f in fit}),
                                       **({"fits": [{"right": [a == b for a, b in zip(f["tests"], truth)],
                                                     "tests": f["tests"], "equiv": f.get("equiv")} for f in fit]}
                                          if eq and fit else {})})
                           + "\n")
        if row is not None and row.get("tool_result") is not None:
            msg = next((m for m in reversed(run.messages or []) if m.get("role") == "tool" and id(m) not in seen), None)
            sent = msg["content"] if msg else None
            if sent is None or not sent.startswith(row["tool_result"]):  # a step-back block may follow the result
                rp.mismatch.append({"task": run.id, "reply": row.get("reply"),  # a bad call's text quotes the cut
                                    "kind": "bad_call_forced" if row.get("bad_args") else "tool_result" if msg is not None
                                    else "tool_result_unserved"})  # args; unserved: the entry or try ended there
                if msg is not None:  # served as recorded from here on
                    msg["content"] = row["tool_result"]


def replay_all(rows, only, data, arm, engine, rp, report, TaskRun, fork=None, copy3_hash=False):
    """Every entry in id order (a copy3 third after its two copies). Drivers: b2 attempts (``#bKxC``) carry
    ``attempt_k``; a copy3 third is built by the engine from the replayed copies (informed: their answers and
    programs); a fork trial's entries start in the base snapshot's state (``fork``)."""
    from . import fork as fork_mod
    done = {}
    for task_id in sorted(rows):
        if only and task_id not in only:  # a copy3 third needs its two copies in the same call
            continue
        name = task_id.split("@")[0].split("#")[0]
        task = json.loads((data / f"{name}.json").read_text())
        copy = task_id.split("@")[0].partition("#")[2]
        run = TaskRun(task_id, task, arm)
        if copy.startswith("b") and "x" in copy:  # b2 attempt k of copy c
            run = TaskRun(task_id, task, {**arm, "attempt_k": int(copy[1:].split("x")[0])})
        elif copy == "3":
            mates = [done[k] for k in sorted(done) if k.split("@")[0].partition("#")[0] == name
                     and k.split("@")[0].partition("#")[2] in ("1", "2")]
            if copy3_hash:  # copy3 (27 Oct): the informed prior showed the program's 16-hex hash, not its source
                for mate in mates:
                    mate.programs = [next((c["code"] for c in mate.candidates if c.get("program") == k + 1), "")
                                     for k in range(len(getattr(mate, "programs", None) or []))]
            third = engine.copy3_for(mates[-1], mates) if len(mates) == 2 else None
            if third is not None and third.id == task_id:
                run = third
            else:
                rp.mismatch.append({"task": task_id, "reply": 0, "kind": "copy3_unbuilt"})
        if fork is not None:
            fork_mod.apply(engine, [run], fork)
        asyncio.run(replay_task(engine, run, rp))
        if getattr(run, "interp", None) is not None:
            run.interp.close()
        done[task_id] = run
        report[task_id] = {"replies": len(rows[task_id]), "replayed": rp.pos.get(task_id, 0)}


def main(argv):
    folder, study_path, data, out = map(Path, argv[:4])
    only = set(argv[argv.index("--tasks") + 1].split(",")) if "--tasks" in argv else None
    probe = "--probe-fit" in argv or "--probe-equiv" in argv or "--probe-sweep" in argv
    from .conv import ConvEngine
    from .loop import TaskRun
    from .repl import Repl
    from . import sandbox

    study = json.loads(study_path.read_text())
    if isinstance(study.get("study"), dict):  # a run's bucket study.json: the comparison config it ran
        study = study["study"]
    trial, spec = study["trials"][0], study.get("arc2") or {}
    arm = trial["arms"][0]
    for i, flag in enumerate(argv):  # --set key=json: the rules a run ran under where today's defaults differ
        if flag == "--set":  # (conv1: bad_call_fix=false; before ce971a80a, 28 Oct 08:32 UTC: legacy_swap=true)
            key, _, value = argv[i + 1].partition("=")
            arm = {**arm, key: json.loads(value)}
    rows, other = {}, {}
    for k, line in enumerate((folder / "events.jsonl").read_text().splitlines()):
        e = {**json.loads(line), "_k": k}
        if e.get("event", "reply") == "reply":
            rows.setdefault(e["task"], []).append(e)
        elif e.get("task"):
            other.setdefault(e["task"], []).append(e)
    log = ServedLog(out, ids=False)
    rp = Replay(rows, other, log, model="replay", effort=study["reasoning_effort"],
                temperature=float(study["temperature"]), template=spec.get("chat_template_extra"))
    settings = spec.get("settings")
    if probe:
        out.mkdir(parents=True, exist_ok=True)
        rp.equiv = float(argv[argv.index("--probe-equiv") + 1]) if "--probe-equiv" in argv else 0.0
        rp.sweep = float(argv[argv.index("--probe-sweep") + 1]) if "--probe-sweep" in argv else 0.0
        rp.equiv = rp.equiv or (0.01 if rp.sweep else 0.0)  # submits only
        rp.probe = (out / "fit-probe.jsonl").open("w")
    rp.window = int(study["windows"][0])

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, strict=False, **kw)

    async def count(messages, s):
        return await rp.count(messages, s)

    engine = ConvEngine(complete=rp.complete, count=count, execute=execute, window=int(study["windows"][0]),
                        cap=int(study["completion_cap"]), settings=settings, record=lambda row: None,
                        repl=lambda run: Repl(getattr(run, "shown", None) or run.task, strict=False,
                                              chars=int({**(settings or {}), **run.arm}.get("tool_chars")
                                                        or T.TOOL_CHARS),
                                              memo=bool({**(settings or {}), **run.arm}.get("memo")),
                                              one_answer=bool({**(settings or {}), **run.arm}.get("one_answer")),
                                              import_globals=bool({**(settings or {}), **run.arm}.get("tool_fix"))))
    original = engine.advance

    async def advance(run, budget, replies=None):  # count needs the task: carry it in the settings view
        run.arm = {**run.arm, "_task": run.id}
        return await original(run, budget, replies)

    engine.advance = advance
    # the served text is the recorded one: an interpreter output that differs on replay (memory addresses, paths,
    # timing) is replaced by what the model saw; the rest of the conversation follows from it
    from . import conv as C
    real_text, real_tool = C.tasks.tool_text, engine.tool

    def recorded(run):
        i = rp.pos.get(run.id, 0)
        return rp.rows[run.id][i - 1].get("tool_result") if i else None

    current = {}

    async def tool(run, method, *args):
        result = await real_tool(run, method, *args)
        if method == "run" and isinstance(result, dict) and recorded(run) is not None:
            current["run"] = run
            result = {**result, "notice": None}
        return result

    def tool_text(result, chars=T.TOOL_CHARS):
        run = current.pop("run", None)
        text = recorded(run) if run is not None else None
        if text is not None:
            if text != real_text(result, chars):
                rp.substituted.append({"task": run.id, "reply": rp.pos.get(run.id, 0)})
            return text
        return real_text(result, chars)

    engine.tool, C.tasks.tool_text = tool, tool_text
    report = {}
    try:
        replay_all(rows, only, data, arm, engine, rp, report, TaskRun, copy3_hash="--copy3-hash" in argv,
                   fork=json.loads(Path(argv[argv.index("--fork") + 1]).read_text()) if "--fork" in argv else None)
    finally:
        C.tasks.tool_text = real_text
    summary = {"tasks": len(report), "replies": sum(r["replies"] for r in report.values()),
               "replayed": sum(r["replayed"] for r in report.values()), "tool_mismatch": len(rp.mismatch),
               "mismatch": rp.mismatch[:50], "extra_requests": len(rp.extra), "substituted": len(rp.substituted)}
    (out / "replay-report.json").write_text(json.dumps({"summary": summary, "tasks": report}, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1:])
