"""Harness-verified negatives (``harness_fixes`` sub-fix ``negative_check``, off by default).

When a gameplay reply (its reasoning, its text or a ``playbook(...)`` write) states a negative about the game, the next
user turn carries the harness's own record for the level, once per distinct claim:

- a click target named by coordinates, ``(row, col)`` or ``row R, col C``, in a sentence saying it does nothing, has
  no effect, is blocked, refused, not clickable, cannot be filled / moved / ... -> the record of clicks at or within
  ``RADIUS`` cells of it;
- an offered action named in capitals (``UP``, ``SPACE``, ...) before the negative in such a sentence -> that
  action's record (targets are read within ``NEAR`` characters of the negative);
- such a sentence naming neither -> the actions of the batch before the reply that changed nothing or were refused;
- the level called unsolvable or impossible -> the level's totals.

A record is: times executed on this level, in how many distinct board states, how many of those changed the board, and
how many attempts the stale-state guard refused (never executed). Nothing is said when the record supports the claim:
executed at least twice, in at least two distinct board states, never changing the board. No advice: the facts and one
neutral line. No game-specific text or logic.
"""

import re

RADIUS = 2
MAX_LINES = 4
NEAR = 80  # characters between the negative and the target it is read with
_NO = r"(?:n't|\s+not|\s+never)"
NEGATIVE = re.compile(
    r"\b(?:do(?:es)?" + _NO + r"\s+(?:do\s+anything|change|work|respond|react|fill|move|register)"
    r"|did" + _NO + r"\s+(?:do\s+anything|change|work|respond|react|fill|move|register)"
    r"|do(?:es)?\s+nothing|did\s+nothing|ha(?:s|ve|d)\s+no\s+effect|no\s+effect|no-?op"
    r"|not\s+(?:clickable|fillable|interactive|selectable|movable|removable|usable|reachable|a\s+slot)"
    r"|(?:can(?:not|'t)|could" + _NO + r")\s+be\s+(?:filled|clicked|moved|removed|changed|selected|placed|entered|opened"
    r"|pushed|used|reached|passed|crossed)"
    r"|(?:is|are|was|were|stays?|remains?)\s+(?:blocked|locked|refused|inert|immovable|unclickable|unreachable)"
    r"|blocked\s+by|(?:was|were|is|got)\s+refused)\b", re.I)
VERDICT = re.compile(
    r"\b(?:unsolvable|unwinnable|not\s+solvable|no\s+(?:valid\s+|possible\s+)?solution"
    r"|impossible\s+to\s+(?:solve|complete|win|reach|satisfy|achieve)"
    r"|(?:level|puzzle|game|goal|task|arrangement|configuration|target)\b[^.!?\n]{0,60}?\b(?:is|are|looks|seems|would\s+be)\s+"
    r"(?:\w+\s+)?impossible"
    r"|(?:level|puzzle|game|goal|task|arrangement|configuration|target)\b[^.!?\n]{0,60}?\b(?:cannot|can't)\s+be\s+"
    r"(?:solved|completed|satisfied|reached|achieved|won))\b", re.I)
DENIED = re.compile(r"(?:\bnot|n't|\bnothing|\bnever|\bno longer|\bnone)\s+(?:\w+\s+)?$", re.I)
CELL = re.compile(r"\(\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)|\brow\s*=?\s*(\d{1,2})\s*,?\s*col(?:umn)?\s*=?\s*(\d{1,2})", re.I)
CLICK = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")
PLAYBOOK = re.compile(r"(?<![a-z_])playbook\(")
HEAD = "Harness record check (this level's log; you wrote that something has no effect or cannot be done):"
TAIL = "A negative that was never executed, or executed once, is not established."


def reply_text(message):
    """The text a claim is read from: reasoning, reply text and the code of a tool call that writes the playbook."""
    parts = [message.get("reasoning") or message.get("reasoning_content") or "", message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        arguments = str((call.get("function") or {}).get("arguments") or "")
        if PLAYBOOK.search(arguments):
            parts.append(arguments.replace("\\n", "\n"))
    return "\n".join(p for p in parts if isinstance(p, str) and p)


def claims(text, keys=()):
    """Distinct claims in ``text``, in order: ("cell", row, col), ("key", NAME), ("recent",), ("level",)."""
    found = []
    names = [k for k in keys if k and k.upper() == k and k != "MOUSE"]
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        for negative in NEGATIVE.finditer(sentence):
            if DENIED.search(sentence[max(0, negative.start() - 14):negative.start()]):
                continue  # "nothing is blocked", "not refused"
            lo, hi = negative.start() - NEAR, negative.end() + NEAR
            targets = [("cell", int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4)))
                       for m in CELL.finditer(sentence) if lo <= m.start() <= hi]
            targets += [("key", name) for name in names
                        for m in re.finditer(rf"(?<![A-Za-z_]){re.escape(name)}(?![A-Za-z_])", sentence)
                        if lo <= m.start() <= negative.start()]  # an action is the subject: named before the negative
            found.extend(t for t in targets if t[0] != "cell" or (t[1] < 64 and t[2] < 64))
            if not targets:
                found.append(("recent",))
        if VERDICT.search(sentence):
            found.append(("level",))
    return list(dict.fromkeys(found))


def target(label):
    """A record key for an executed or refused action's display label."""
    click = CLICK.fullmatch(str(label).strip())
    return ("cell", int(click.group(1)), int(click.group(2))) if click else ("key", str(label).strip().upper())


class Record:
    """What was executed and refused, per level."""

    def __init__(self):
        self.levels = {}
        self.recent, self.previous = [], []

    def _row(self, level, key):
        return self.levels.setdefault(level, {}).setdefault(key, {"n": 0, "boards": set(), "changed": 0, "refused": 0})

    def executed(self, level, label, board, changed):
        key = target(label)
        row = self._row(level, key)
        row["n"] += 1
        row["boards"].add(board)
        row["changed"] += bool(changed)
        if not changed:
            self.recent.append(key)

    def refused(self, level, label):
        key = target(label)
        self._row(level, key)["refused"] += 1
        self.recent.append(key)

    def mark(self):
        """A reply arrived: what happened since the previous one is what its claims are about."""
        self.previous, self.recent = list(dict.fromkeys(self.recent)), []

    def stats(self, level, claim):
        rows = self.levels.get(level, {})
        if claim[0] == "cell":
            near = [v for k, v in rows.items() if k[0] == "cell" and max(abs(k[1] - claim[1]), abs(k[2] - claim[2])) <= RADIUS]
        else:
            near = [rows[claim]] if claim in rows else []
        boards = set().union(*(v["boards"] for v in near)) if near else set()
        return {"n": sum(v["n"] for v in near), "boards": len(boards), "changed": sum(v["changed"] for v in near),
                "refused": sum(v["refused"] for v in near)}

    def totals(self, level, offered=()):
        rows = self.levels.get(level, {})
        ran = {k: v for k, v in rows.items() if v["n"]}
        keys = {k[1] for k in ran if k[0] == "key"}
        return {"actions": sum(v["n"] for v in ran.values()), "distinct": len(ran),
                "changed": sum(v["changed"] > 0 for v in ran.values()), "once": sum(v["n"] == 1 for v in ran.values()),
                "refused": sum(v["refused"] for v in rows.values()),
                "unused": [k for k in offered if k != "MOUSE" and k not in keys]
                + (["MOUSE"] if "MOUSE" in offered and not any(k[0] == "cell" for k in ran) else [])}


def supported(stats):
    return stats["n"] >= 2 and stats["boards"] >= 2 and stats["changed"] == 0


def times(n):
    return "once" if n == 1 else f"{n} times"


def line(claim, stats):
    name = f"MOUSE at or within {RADIUS} cells of row {claim[1]}, col {claim[2]}" if claim[0] == "cell" else claim[1]
    if not stats["n"]:
        ran = "never executed on this level"
    else:
        ran = (f"executed {times(stats['n'])} on this level, in {stats['boards']} distinct board "
               f"state{'s' if stats['boards'] != 1 else ''}; {stats['changed']} of those changed the board")
    refused = (f"; {stats['refused']} further attempt{'s were' if stats['refused'] != 1 else ' was'} refused by the "
               "stale-state guard and never ran" if stats["refused"] else "")
    return f"- {name}: {ran}{refused}."


def respond(record, level, found, answered, offered=()):
    """``(text or None, details)`` for the claims of one reply; ``answered`` (a set) keeps what was said already."""
    lines, details = [], []
    expanded = []
    for claim in found:
        expanded.extend(record.previous[:3] if claim == ("recent",) else [claim])
    for claim in dict.fromkeys(expanded):
        if (level, claim) in answered or (claim[0] == "cell" and any(
                lv == level and c[0] == "cell" and max(abs(c[1] - claim[1]), abs(c[2] - claim[2])) <= RADIUS
                for lv, c in answered)):
            continue  # said already (for a click: about this spot)
        if claim == ("level",):
            t = record.totals(level, offered)
            if not t["actions"]:
                continue
            answered.add((level, claim))
            unused = f"; offered and never executed on this level: {', '.join(t['unused'])}" if t["unused"] else ""
            lines.append(f"- This level so far: {t['actions']} actions executed over {t['distinct']} distinct actions or "
                         f"cells; {t['changed']} of those changed the board at least once, {t['once']} were executed only "
                         f"once; {t['refused']} attempts were refused by the stale-state guard and never ran{unused}.")
            details.append({"claim": "level", **{k: v for k, v in t.items()}})
            continue
        stats = record.stats(level, claim)
        if supported(stats):
            details.append({"claim": list(claim), **stats, "quiet": True})
            answered.add((level, claim))
            continue
        answered.add((level, claim))
        lines.append(line(claim, stats))
        details.append({"claim": list(claim), **stats})
        if len(lines) >= MAX_LINES:
            break
    if not lines:
        return None, details
    return "\n".join([HEAD, *lines, TAIL]), details
