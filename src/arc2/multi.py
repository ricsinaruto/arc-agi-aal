"""Adaptive multi-attempt design (b2), shared by the RTX harness (arc2.conv) and the 4xL4 port.

An entry is one roster task-copy; its attempts are independent conversations on it (attempt 1 = the entry's own run,
attempt k >= 2 a fresh one on augmented view k - 1, spawned from the token budget, never the roster's other copy).

- ``pick``: once every entry has started, a freed slot starts the next attempt of the backlog entry with the largest
  key, one scale for all: the tokens to first submit of its most recent attempt, or, while no attempt
  has submitted, the tokens it has spent so far. An entry with an attempt still running beyond attempt 1 is not in
  the backlog (attempt 1 that never submitted keeps running beside its fresh attempt, rule).
- ``merge``: per test output, the distinct answers of all attempts' latest first guesses, ranked by (verified, number
  of attempts that agree, earliest attempt); the top two are guesses 1 and 2. With two attempts this is b1's rule."""
import json


def entry_key(attempts):
    """The backlog key of an entry (higher = sooner), or None when it is not in the backlog. ``attempts``: dicts
    {k, running, submitted, first_submit_tokens, spent} in attempt order (attempt 1 first)."""
    if any(a["running"] for a in attempts[1:]):
        return None
    if not any(a["submitted"] for a in attempts):
        return attempts[0]["spent"]
    last = attempts[-1]
    if last["running"]:  # attempt 1 still running after it submitted (not in close mode): wait for it
        return None
    return last["first_submit_tokens"] or 0


def pick(entries, ns_first=False):
    """The index of the backlog entry to give the next attempt (``entries``: per entry its attempts as for
    ``entry_key``), or None. ``ns_first`` (fk-b2, the ns rule): entries no attempt has submitted
    rank above every submitted one."""
    keys = [((not any(x["submitted"] for x in a)) if ns_first else 0, entry_key(a), i) for i, a in enumerate(entries)]
    keys = [(top, k, i) for top, k, i in keys if k is not None]
    return max(keys)[2] if keys else None


def merge(answers, n_outputs):
    """``answers``: per attempt, in attempt order, {tests: [grid per output] (None = no answer), verified: bool}.
    Returns per output the ranked distinct grids (at most two)."""
    out = []
    for i in range(n_outputs):
        seen = {}
        for k, a in enumerate(answers):
            grid = a["tests"][i] if a["tests"] is not None and i < len(a["tests"]) else None
            if grid is None:
                continue
            key = json.dumps(grid)
            row = seen.setdefault(key, {"grid": grid, "verified": False, "votes": 0, "first": k})
            row["votes"] += 1
            row["verified"] = row["verified"] or bool(a.get("verified"))
        ranked = sorted(seen.values(), key=lambda r: (not r["verified"], -r["votes"], r["first"]))
        out.append([r["grid"] for r in ranked[:2]])
    return out
