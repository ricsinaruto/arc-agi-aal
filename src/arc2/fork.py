"""Fork a reopen variant from a base run's first passes (first-pass noise between runs is
~20 outputs, larger than the reopen effects being read). ``snapshot`` keeps, per entry of a finished conversation run,
the state at its first submit (submission 1's guesses, the programs and candidates before it, its tokens); ``apply``
starts a trial in that state: every submitted entry waits for its first reopen pass, entries that never submitted keep
the base's result and do not play, and the roster clock starts at the base's first-pass tokens. Passes run in a new
interpreter (the base's interpreter state is not restored)."""
import json

FORK_T = -1e9  # base candidates sit before every live reply: they count at every mark


def snapshot(rows, events):
    """``rows``: the base's tasks.jsonl rows; ``events``: its events.jsonl rows. Returns {entries, first_pass_tokens}."""
    replies = {}
    for e in events:
        if e.get("event", "reply") == "reply" and "program" in e:
            replies.setdefault(e["task"], []).append(e)
    entries, total = {}, 0
    for row in rows:
        cands = row["candidates"]
        s1 = [c for c in cands if c.get("submitted") and (c.get("submit_at") or c.get("submit_n")) == 1
              and not c.get("mg")]
        if not s1:
            entries[row["task"]] = {"submitted": False, "spent": row["spent"],
                                    "candidates": [{**c, "t": FORK_T} for c in cands]}
            total += row["spent"]
            continue
        at = min(c["tokens_at"] for c in s1)
        guesses = [{**{k: v for k, v in c.items() if k not in ("pinned", "clock")}, "t": FORK_T, "submit_n": 1,
                    "submitted": 1 if c.get("pinned") or c["submitted"] == 1 else c["submitted"]} for c in s1]
        before = [{**{k: v for k, v in c.items() if k != "clock"}, "t": FORK_T} for c in cands
                  if not c.get("submitted") and c.get("tokens_at", 0) <= at]
        programs = {}
        for e in replies.get(row["task"], []):
            if int(e.get("tokens_at") or 0) <= at and e.get("code"):
                programs[int(e["program"])] = e["code"]
        n = max([c.get("program") or 0 for c in before] + [0])
        entries[row["task"]] = {"submitted": True, "spent": at, "first_submit_tokens": at,
                                "candidates": before + guesses, "programs": [programs.get(k, "") for k in range(1, n + 1)]}
        total += at
    return {"entries": entries, "first_pass_tokens": total}


def apply(engine, order, snap):
    """Put the trial's entries in the snapshot's state. Returns the base's first-pass tokens (the roster clock's
    start). Entries missing from the snapshot play from the start."""
    for run in order:
        s = snap["entries"].get(run.id)
        if s is None:
            continue
        settings = {**(engine.settings or {}), **run.arm}
        if not s["submitted"]:
            run.candidates[:] = [dict(c) for c in s["candidates"]]
            run.spent = s["spent"]
            if not settings.get("fork_ns_play"):  # ``fork_ns_play`` (fk-rounds): plays on in a new conversation
                run.done = "fork_base"
            continue
        if settings.get("submit_mode", "close") == "close":  # b2 (close mode): the entry ended at its first submit;
            run.candidates[:] = [dict(c) for c in s["candidates"]]  # later work is fresh attempts (multi_attempt)
            run.programs = list(s["programs"])
            run.spent, run.first_submit_tokens, run.submits = s["spent"], s["first_submit_tokens"], 1
            run.done, run.forked = "submitted", True
            continue
        engine.start(run, settings)
        run.candidates[:] = [dict(c) for c in s["candidates"]]
        run.programs = list(s["programs"])
        run.spent, run.first_submit_tokens, run.submits = s["spent"], s["first_submit_tokens"], 1
        if settings.get("swap") == "budget":  # ROUNDS (fk-rounds): round 2 from the backlog in rounds order, in a new
            run.round, run.forked = 2, True  # conversation (the base's context is not saved) opened by the round
            continue  # prompt; its budget is then the tokens to first submit (floor one reply)
        run.awaiting_reopen = True  # not parked: the worker clears that at a submit; set, every pass stopped after
        # one reply and went behind all reopen tokens (fk-naive1, 29 Oct)
        run.forked = True
    return float(snap["first_pass_tokens"])


def load(path):
    return json.loads(open(path).read())
