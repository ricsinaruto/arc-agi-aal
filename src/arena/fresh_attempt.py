"""hx4 ``fresh_attempt`` (7 October): a stuck level is restarted with the earlier attempt's beliefs as suspects.

Why (docs: hx4 LOG, Gate 1): on stuck DEV-family levels the primary cause in 16 of 33 traces was a mechanic inferred
from one confounded observation and never re-tested (a refused move read as "immovable"), in 3 a wrong goal; the
chance of clearing in the next 100K tokens halves after 200K on a level, while another clone clears the same level in
a fraction of the tokens. The agent does not doubt its own conclusions inside the context that produced them.

Mechanism: when a level has taken ``tokens`` generated gameplay tokens since it started (or since the last fresh
attempt; the n-th cutoff is ``tokens * growth**n``, at most ``max`` times per level when ``max`` > 0), the harness
- collects what the agent believes (its carried world model and its ``note()`` entries),
- resets the level (``reset``),
- clears the conversation to ``[system, this note, current observation]`` and the carried world model
  (the python session, ``history``, ``transitions`` and registered tools stay),
- and hands the beliefs back as suspects (``beliefs`` = "suspect") with a fixed order of work: name the beliefs the
  failed plans depended on, test each with the cheapest live experiment, check the goal against earlier wins, then
  plan. ``beliefs`` = "drop" omits them (plain restart).
``at_start`` applies it once before the first turn (the stuck-level bench: the attempt starts from a recorded stuck
state with the source agent's beliefs). Generic: no per-game information, in-run signals only. Needs ``drain_half``.
"""

DEFAULTS = {"tokens": 60000, "growth": 1.0, "max": 0, "reset": True, "beliefs": "suspect", "at_start": False}
BELIEFS = ("suspect", "drop")
FIELDS = (("goal_model", "Goal"), ("world_model", "World"), ("action_model", "Actions"),
          ("recent_findings", "Recent findings"), ("open_questions", "Open questions"), ("current_plan", "Plan"),
          ("cross_level_notes", "From earlier levels"))
FIELD_CHARS, NOTE_CHARS, TOTAL_CHARS = 700, 300, 4500

HEAD = (
    "[Fresh attempt] {spent} went into level {level} without clearing it. The conversation was cleared{reset}. Your "
    "python session is unchanged: `history`, `transitions`, your variables, functions and registered tools are all "
    "still there.\n\n"
    "The level is solvable with the actions you have; nothing about it is impossible. "
)
SUSPECT = (
    "The earlier attempt did not solve it, so at least one thing it took for granted is wrong. This is what it "
    "believed when it stopped. Treat every line as a suspect, not as a fact:\n{beliefs}\n\n"
    "Do not resume its plan. Work in this order:\n"
    "1. Name the two or three beliefs above that its failed plans depended on most. Include anything it judged "
    "impossible, immovable, unreachable, random or irrelevant, and what it took the goal to be.\n"
    "2. For each, run the cheapest experiment on the live board that could prove it wrong. If an action was refused "
    "or changed nothing, repeat it from a different position, on a different object, or after a different action "
    "before concluding anything: one refusal says little about its cause.\n"
    "3. Check the goal against evidence: use `history` to see what the board looked like at the moment each earlier "
    "level was won, and what on this board corresponds to it. If an on-board indicator changes while you act, find "
    "out what changes it.\n"
    "4. Only then plan. Record with note(...) which earlier beliefs you refuted and which you confirmed."
)
DROP = (
    "The earlier attempt's conclusions were discarded because they did not lead to a solution. Re-derive the "
    "mechanics and the goal from the board and from `history` (what each action did, and what the board looked like "
    "at the moment each earlier level was won); test with a few actions rather than reasoning at length. If an "
    "action is refused or changes nothing, repeat it from a different position, on a different object, or after a "
    "different action before concluding anything."
)


def settings(value):
    """The study block (None/False: off; True: ``DEFAULTS``; a dict of overrides)."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError(f"fresh_attempt is true/false or a dict over {sorted(DEFAULTS)}")
    result = {**DEFAULTS, **value}
    if type(result["tokens"]) is not int or result["tokens"] < 1000:
        raise ValueError("fresh_attempt.tokens is an integer >= 1000")
    if type(result["growth"]) not in (int, float) or result["growth"] < 1:
        raise ValueError("fresh_attempt.growth is a number >= 1")
    if type(result["max"]) is not int or result["max"] < 0:
        raise ValueError("fresh_attempt.max is a non-negative integer")
    if result["beliefs"] not in BELIEFS:
        raise ValueError(f"fresh_attempt.beliefs is one of {BELIEFS}")
    for key in ("reset", "at_start"):
        if type(result[key]) is not bool:
            raise ValueError(f"fresh_attempt.{key} is a boolean")
    return result


def cutoff(spec, count):
    """Generated tokens on the level before fresh attempt number ``count`` (0-based), or None when none is left."""
    if spec["max"] and count >= spec["max"]:
        return None
    return int(spec["tokens"] * float(spec["growth"]) ** count)


def clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 12].rstrip() + " [truncated]"


def belief_lines(knowledge, notes):
    """The earlier attempt's beliefs as bullet lines (the carried world model, then its notes), bounded."""
    lines = [f"- {label}: {clip(knowledge[key], FIELD_CHARS)}" for key, label in FIELDS if (knowledge or {}).get(key)]
    for key, text in (notes or {}).items():
        lines.append(f"- Note {clip(key, 40)}: {clip(text, NOTE_CHARS)}")
    kept, used = [], 0
    for line in lines:
        if used + len(line) > TOTAL_CHARS:
            break
        kept.append(line)
        used += len(line)
    return kept


def note_text(spec, *, level, tokens, actions, reset, knowledge=None, notes=None):
    spent = (f"{int(tokens):,} generated tokens and {int(actions)} actions" if tokens else
             "A long earlier attempt")
    text = HEAD.format(spent=spent, level=level,
                       reset=" and the level was reset to its starting board (shown below)" if reset else "")
    lines = belief_lines(knowledge, notes) if spec["beliefs"] == "suspect" else []
    return text + (SUSPECT.format(beliefs="\n".join(lines)) if lines else DROP)


def aggregate(rows):
    rows = [r for r in rows if r]
    return {"games": sum(1 for r in rows if r.get("attempts")), "attempts": sum(int(r.get("attempts", 0)) for r in rows),
            "resets": sum(int(r.get("resets", 0)) for r in rows)}
