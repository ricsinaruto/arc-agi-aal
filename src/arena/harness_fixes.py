"""hero-v2 ``harness_fixes``: the harness bugs of the hero-v2 2 h audit, fixed behind one opt-in switch.

User decision (29 September): "bug fixes + inventory". The audit is docs/validation-20260927/hero-v2-analysis/
harness.md (numbers in harness-audit.json; replay in scripts/arena_analysis/harness_fixes_replay.py). The study key
``harness_fixes`` (``settings``) is None on every earlier build; with it on, ``ArenaDuckMixin`` applies the listed
sub-fixes (``FIXES``), each named in study.json:

- ``exact_trigger``: compact on the exact prompt count of the would-be gameplay request (the admission's /tokenize
  count, cached so the admission reuses it), not the character estimate x 1.1; same budget W - cap - 512.
- ``summary_guard``: when the exact count + ``summary_room_tokens`` exceeds the window, the summary request covers the
  messages up to the latest observation; the summary's output cap comes from the exact count, not the estimate.
- ``inventory``: after the summary, a harness-written block with the live python definitions (functions with
  signatures and first docstring lines, classes, modules, variables with types and short reprs) and the last
  ``inventory_receipts`` action receipts, capped at ``inventory_max_tokens``.
- ``cap_followup``: a reply cut off at the completion cap is followed by "your previous reply was cut off ..." instead
  of the public "You have not acted yet. Investigate first."; the cap and the verbatim reasoning are unchanged.
- ``stuck_notice``: the notice reports active seconds (parked time excluded) and generated tokens on the level; it
  fires on a no-change streak or on the turn counter (thresholds from the audit's level distributions); RESET is
  suggested only on a no-change streak.
- ``sandbox``: harmless standard-library modules (``hashlib``, ``textwrap``, ``traceback``; ``time``, ``pickle`` and
  ``inspect`` as restricted shims) and the ``globals``/``locals``/``vars``/``id`` builtins; no file access.
- ``timeout_interrupt``: a python call past the 30 s tool timeout is interrupted (SIGUSR1 -> an exception in the
  running statement) instead of killing the interpreter; the kill stays as the fallback.
- ``prompt_text``: the system prompt no longer claims that earlier images are omitted or that a world model is always
  carried; the empty "end of world model." line is dropped; the GAME_OVER note no longer points at "the mechanics
  above" when nothing is above it; the state line uses the image label's step.

Opt-in sub-fixes (``OPTIONAL_FIXES``; 30 September, docs/validation-20260927/compaction-overshoot-fix.md): named in a
list only, never switched on by ``true``, so every build made with ``true`` (runs A and B) keeps its recorded
``fixes`` and thresholds byte for byte:

- ``summary_headroom``: when the summary of the whole context would have less than ``summary_room_tokens`` (plus
  its own prompt) of room in the window, the summary covers the messages before the latest message boundary that
  leaves that room, and the rest (the reply that overshot, its tool result and the observation) stays verbatim
  after the summary: nothing is dropped or truncated. At 64K the exact trigger lets a context reach W - 512 + the
  next observation (a 16K reply landing on a prompt just under W - cap - 512); the cut before that reply leaves
  the summary at least cap + 512 of room, since that reply's own prompt passed the trigger.
- ``inventory_imports``: the inventory lists the imported modules and classes first (they were the lowest-priority
  section and were dropped whenever the variables filled the cap), says that anything not listed must be imported,
  says ``inspect`` needs ``import inspect`` (its header's ``inspect.getsource`` hint caused NameErrors), and names
  the python view of the receipts (``history[i].action``, ``history[i].frame.step``).
- ``text_count`` (B200, 5 October): the exact counts (trigger and admission) are made with the images sent as the text
  the template renders for them plus each image's token count (``arc_infra.model`` ``text_images``), so the serving
  frontend's single multimodal thread processes each request once, not ~2.5 times (b2-run1-12h-v1 stalled at 88
  streams from ~1,300 s: ~60 s per /tokenize). Each reply's prompt_tokens is checked against the admission count.
"""

from duck_transfer.feedback import StuckTracker

# --- prompt audit fixes (9 October; docs/harness-arm.md 5.7), each its own opt-in sub-fix -------------------------
# solvable_line (item 2): one system line. death_turn_v2 (item 3): one text on a level's first death, one line on
# later ones; a step-budget death is named as such (``death_text``). findings_keep (item 4): a GAME_OVER no longer
# writes the carried "Recent findings" line (the model's own stays; the user turn's GAME_OVER line is the one-turn
# notice). tool_call_followup (item 1) and stuck_streak (item 10): prepared, in no build yet.
# negative_check: a stated negative is answered once with the level's record.
SOLVABLE_ANCHOR = "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
SOLVABLE_LINE = (
    "- Every level can be completed with the actions offered. If your analysis shows a level cannot be completed, "
    "a rule in your model of the game is wrong: find and test it.\n"
)
TOOL_CALL_FOLLOWUP = (
    "Your previous reply contained no tool call, so nothing was run. Reply with one `python` tool call."
)
DEATH_FIRST = (
    "GAME_OVER at step {step} after {fatal}: the attempt on level {level} ended and the level was reset "
    "automatically; the board below is its fresh start. Completed levels are kept and what you learned about this "
    "level's mechanics still applies. The attempt lasted {actions} actions{edge}. Identify the cause before acting "
    "again; there are two kinds. Step budget: a line of cells along a border that changes on most steps is a "
    "step-budget bar, and when it runs out the attempt ends whatever the last action was. Look at it in the death "
    "frame (`history[-2].frame`; `history[-3].frame` is the board before the last action): if it was used up, the "
    "route was too long, not wrong, so work out how many actions one attempt allows and how fast the bar moves, "
    "then finish in fewer or find what refills it. Fatal action: if the bar still had room, that action from that "
    "position ends the attempt; keep the rest of what you know and change that step."
)
DEATH_LATER = (
    "GAME_OVER at step {step} after {fatal}: death {count} on level {level}, which was reset to its start. Attempts "
    "lasted {lengths} actions{verdict}."
)
DEATH_BUDGET = (": the same length with different actions, so this is a step budget of about {actions} actions per "
                "attempt, not a fatal action; the solution has to fit in fewer, or the bar has to be refilled")
DEATH_REPEAT = ": the same actions ended the same way; change the plan before its last step"
DEATH_OPEN = "; identify the cause (step budget used up, or that action from that position) before repeating the plan"
DEATH_IMAGE_LABEL = (
    "Death frame image (the board at the moment the attempt ended, before the automatic reset): check the border "
    "bar here."
)


def death_text(deaths, *, step, fatal, level):
    """The death turn's one text (``death_turn_v2``). ``deaths``: this level's deaths, oldest first, each
    ``{"actions", "edge", "sequence"}`` (actions of the attempt, how many of them changed a border-ring cell, a
    digest of the attempt's actions)."""
    last = deaths[-1]
    if len(deaths) == 1:
        edge = ""
        if last["actions"] >= 4 and 2 * last["edge"] >= last["actions"]:
            edge = f"; cells in the border ring changed on {last['edge']} of them"
        return DEATH_FIRST.format(step=step, fatal=fatal, level=level, actions=last["actions"], edge=edge)
    before = deaths[-2]
    if before["sequence"] == last["sequence"]:
        verdict = DEATH_REPEAT
    elif abs(before["actions"] - last["actions"]) <= 1 and last["actions"] >= 4:
        verdict = DEATH_BUDGET.format(actions=last["actions"])
    else:
        verdict = DEATH_OPEN
    lengths = ", ".join(str(d["actions"]) for d in deaths[-4:])
    return DEATH_LATER.format(step=step, fatal=fatal, level=level, count=len(deaths), lengths=lengths, verdict=verdict)

FIXES = ("exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox",
         "timeout_interrupt", "prompt_text")
# Named in a list only; ``true`` (every build so far) means ``FIXES``.
OPTIONAL_FIXES = ("summary_headroom", "inventory_imports", "text_count", "solvable_line", "death_turn_v2",
                  "findings_keep", "tool_call_followup", "stuck_streak", "negative_check")
KNOWN_FIXES = FIXES + OPTIONAL_FIXES

# Thresholds (docs/validation-20260927/hero-v2-analysis/harness.md, "Fixes implemented"): on the 182 level segments
# of the hero-v2 2 h run, the longest no-change streak per level was at most 3 and cleared levels took 7 acting
# turns (median; p90 14, max 25), so a 5-action streak and 16 turns (then every 8) fire only past what normal play
# shows; the recorded 8-then-4 rule fired 149 notices, 55 of them on levels that were cleared later.
DEFAULTS = {
    "summary_room_tokens": 4096,
    "inventory_max_tokens": 1536,
    "inventory_receipts": 12,
    "stuck_turns": 16,
    "stuck_notice_turns": 8,
    "stuck_no_change_actions": 5,
    "interrupt_grace_seconds": 3.0,
}


def settings(value):
    """The study block for ``value`` (False/None: off; True: every sub-fix of ``FIXES``; a list of sub-fix names from
    ``KNOWN_FIXES``; or a dict with ``fixes`` and threshold overrides), or None when off."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    elif isinstance(value, (list, tuple)):
        value = {"fixes": list(value)}
    if not isinstance(value, dict):
        raise ValueError("harness_fixes is true/false, a list of sub-fixes or a settings object")
    unknown = set(value) - {"fixes", *DEFAULTS}
    if unknown:
        raise ValueError(f"Unknown harness_fixes settings {sorted(unknown)}; known: fixes, {sorted(DEFAULTS)}")
    fixes = list(value.get("fixes", FIXES))
    bad = [f for f in fixes if f not in KNOWN_FIXES]
    if bad or not fixes or len(set(fixes)) != len(fixes):
        raise ValueError(f"harness_fixes fixes are distinct names from {KNOWN_FIXES}; got {fixes}")
    if "inventory_imports" in fixes and "inventory" not in fixes:
        raise ValueError("harness_fixes inventory_imports changes the inventory: it needs the inventory sub-fix")
    result = {"fixes": [f for f in KNOWN_FIXES if f in fixes]}
    for key, default in DEFAULTS.items():
        item = value.get(key, default)
        if type(item) is bool or not isinstance(item, (int, float)) or item <= 0:
            raise ValueError(f"harness_fixes {key} is a positive number")
        result[key] = type(default)(item)
    return result


# --- cap-hit follow-up --------------------------------------------------------------------------------------------

PUBLIC_FOLLOWUP_PREFIX = "You have not acted yet. Investigate first. "
CAP_FOLLOWUP = (
    "Your previous reply was cut off at the {cap:,}-token output limit before it called a tool; its reasoning is kept "
    "above. Do not restart the analysis: state the decision it reached in one or two sentences, then call `python` "
    "with the action (or the one check that decision still needs)."
)
# Thinking off (hx_harness no_think): there is no reasoning above to point at.
CAP_FOLLOWUP_NO_THINK = (
    "Your previous reply was cut off at the {cap:,}-token output limit before it called a tool. State your decision "
    "in one or two sentences, then call `python` with the action (or the one check that decision still needs)."
)
CAP_TOOL_NOTE = (
    "Your previous reply reached the {cap:,}-token output limit, so this tool call was cut off. State the decision "
    "briefly and send the call again, complete and short."
)


# --- stuck notice ---------------------------------------------------------------------------------------------------

class ActiveStuckTracker(StuckTracker):
    """The stuck notice on active time and generated tokens (``stuck_notice``).

    ``active_seconds`` and ``completion_tokens`` are the game clock's (``ActiveClock``): parked time never counts, and
    the tokens are every completed request's on the level. Due on a no-change streak of ``stuck_no_change_actions``
    (again after as many more) or from ``stuck_turns`` turns on, every ``stuck_notice_turns`` turns."""

    def __init__(self, cfg, *, active_seconds, completion_tokens, rule):
        self.active_seconds, self.completion_tokens = active_seconds, completion_tokens
        self.rule = rule
        super().__init__(cfg)
        self.no_change_limit = int(rule["stuck_no_change_actions"])
        self.turn_limit = int(rule["stuck_turns"])
        self.interval = int(rule["stuck_notice_turns"])

    def reset_level(self, now=None):
        super().reset_level(now)
        self.level_active = self.active_seconds()
        self.level_tokens = self.completion_tokens()
        self.streak_noticed = 0
        self.fired = {"streak": 0, "turns": 0}

    def note_action(self, board_changed):
        super().note_action(board_changed)
        if board_changed:
            self.streak_noticed = 0

    def due(self):
        """``"streak"``, ``"turns"`` or None."""
        if self.no_change >= self.no_change_limit and self.no_change >= self.streak_noticed + self.no_change_limit:
            return "streak"
        if "stuck_streak" in self.rule["fixes"]:
            return None  # stuck_streak: the turn counter alone is not a stuck signal
        if self.turns >= self.turn_limit and (self.last_notice is None or self.turns - self.last_notice >= self.interval):
            return "turns"
        return None

    def notice(self, level, tried, now=None):
        reason = self.due()
        if reason is None:
            return None
        self.last_notice, self.notices = self.turns, self.notices + 1
        self.fired[reason] += 1
        if reason == "streak":
            self.streak_noticed = self.no_change
        seconds = int(max(0.0, self.active_seconds() - self.level_active))
        tokens = max(0, int(self.completion_tokens() - self.level_tokens))
        tried_text = ", ".join(tried[:12]) + (" ..." if len(tried) > 12 else "") if tried else "none"
        text = (
            f"Stuck check: {self.turns} turns, {self.actions} actions and {tokens:,} generated tokens on level {level} "
            f"({seconds}s of active play), {self.no_change} consecutive actions produced no board change. "
            f"Tried on this exact board with no effect: {tried_text}. "
            "Re-derive the goal from `current_frame.segmentation` (which object differs from the others, what a "
            "target or slot looks like), probe an untried object or action type and read `transition` in the receipt."
        )
        if reason == "streak":
            text += (" If the level is bricked, call `action(['RESET'])` to restart it (RESET is refused only as the "
                     "very first action of a fresh level).")
        return text

    def summary(self):
        return {"notices": self.notices, **self.fired}


# --- post-compaction inventory ---------------------------------------------------------------------------------------

INVENTORY_HEADER = (
    "[Harness inventory] Written by the harness from the live python session, not by you. These definitions still "
    "exist in the interpreter: call them instead of redefining them (check one with `inspect.getsource(name)` or "
    "`inspect.signature(name)` if unsure)."
)
# ``inventory_imports``: the header's hint named ``inspect`` without saying it must be imported (run A: both
# post-compaction ``inspect`` NameErrors were that hint followed literally).
INVENTORY_HEADER_IMPORTS = (
    "[Harness inventory] Written by the harness from the live python session, not by you. These definitions still "
    "exist in the interpreter: call them instead of redefining them (to check one, `import inspect` first, then "
    "`inspect.getsource(name)` or `inspect.signature(name)`)."
)
IMPORTS_NOTE = "Only the modules and names listed here are imported; import anything else before using it."
NO_IMPORTS_NOTE = "No modules are imported in the interpreter; import what you use."
RECEIPTS_TITLE_IMPORTS = "Last actions (newest last; in python `history[i].action` and `history[i].frame.step`):"
FRESH_SESSION = (
    "[Harness inventory] The python interpreter holds no definitions of yours (it was never used or it was reset): "
    "redefine the helpers you need."
)


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def receipt_lines(history, results, count):
    """One line per executed action, newest last: the action and whether the board changed."""
    rows = [entry for entry in history if getattr(entry, "action", "")][-int(count):]
    lines = []
    for entry in rows:
        step = entry.frame.step
        result = results.get(step) or {}
        label = result.get("action_display") or entry.action
        if result.get("level_completed"):
            effect = "level completed"
        elif result.get("game_over"):
            effect = "GAME_OVER"
        elif entry.action == "RESET":
            effect = "level restarted"
        elif "board_changed" not in result:
            effect = "executed"
        elif result.get("board_changed"):
            transition = result.get("transition") or {}
            cells, bbox = transition.get("changed_cells"), transition.get("change_bbox")
            effect = "changed" + (f" {cells} cells" if cells is not None else "") + (f" in {list(bbox)}" if bbox else "")
        else:
            effect = "no change"
        lines.append(f"- step {step} {label}: {effect}")
    return lines


def render_inventory(data, receipts, *, max_tokens, chars_per_token, reset_note=None, imports=False):
    """The inventory block within ``max_tokens`` (characters budgeted with the calibrated ratio).

    Truncation: each entry is clipped (functions 160, variables 110 characters); sections fill in priority order
    (functions, receipts, variables, classes and modules), each keeping its newest entries; what does not fit is
    counted in one "+N more" line with the dropped names (clipped).

    ``imports`` (``inventory_imports``): the classes and modules lines (clipped to 300 characters each) move into the
    head, before every section, with the note that anything not listed must be imported; the header says how to use
    ``inspect`` and the receipts' title names their python view."""
    budget = max(400, int(max_tokens * max(1.0, float(chars_per_token))))
    receipts_title = RECEIPTS_TITLE_IMPORTS if imports else "Last actions (newest last):"
    if not data:
        head = [FRESH_SESSION]
        functions = variables = extra = []
    else:
        head = [INVENTORY_HEADER_IMPORTS if imports else INVENTORY_HEADER]
        functions = [_clip(f"- {name}{signature}" + (f" -- {doc}" if doc else ""), 160)
                     for name, signature, doc in data.get("functions") or []]
        variables = [_clip(f"- {name}: {kind}" + (f" len={size}" if size is not None else "") + f" = {text}", 110)
                     for name, kind, size, text in data.get("values") or []]
        extra = []
        if data.get("classes"):
            extra.append("- classes: " + ", ".join(data["classes"]))
        if data.get("modules"):
            extra.append("- imported modules and callables: " + ", ".join(data["modules"]))
        extra = [_clip(line, 300) for line in extra]
        if data.get("error"):
            head.append(_clip(f"(inventory incomplete: {data['error']})", 200))
        elif imports:
            head.extend(extra)
            head.append(IMPORTS_NOTE if data.get("modules") or data.get("classes") else NO_IMPORTS_NOTE)
            extra = []
    if reset_note:
        head.append(reset_note)
    used = sum(len(line) + 1 for line in head)
    sections = {}
    for key, title, lines in (("functions", "Functions:", functions), ("receipts", receipts_title, receipts),
                              ("variables", "Variables:", variables), ("extra", "Other names:", extra)):
        if not lines:
            continue
        room = budget - used - len(title) - 1 - 120  # 120: the "+N more" line
        kept = []
        for line in reversed(lines):
            if room - len(line) - 1 < 0:
                break
            kept.append(line)
            room -= len(line) + 1
        kept.reverse()
        dropped = len(lines) - len(kept)
        if not kept and budget - used < len(title) + 120:
            continue
        block = [title, *kept]
        if dropped:
            names = [line[2:].split("(", 1)[0].split(":", 1)[0] for line in lines[:dropped]]
            block.append(_clip(f"(+{dropped} more, oldest first: {', '.join(names)})", 118))
        sections[key] = block
        used += sum(len(line) + 1 for line in block)
    lines = list(head)
    for key in ("functions", "variables", "extra", "receipts"):
        lines.extend(sections.get(key, []))
    return "\n".join(lines)


# --- prompt text ----------------------------------------------------------------------------------------------------

SYSTEM_PROMPT_FIXES = {
    # The arena's append-only history keeps every image (up to 26 per request in hero-v2 2 h).
    "- The newest user turn includes an image of the board as it was before this turn's actions (labelled with its "
    "step); earlier turns keep only a note that their image was omitted.\n":
        "- Each user turn includes an image of the board as it was before that turn's actions (labelled with its "
        "step). Earlier turns, their images and your earlier replies stay in the context verbatim until a context "
        "compaction replaces them with your own summary.\n",
    # The carried world model is filled only from reply text (content), empty in ~95% of turns.
    "(applies to every turn; each user turn carries only the outcome of the previous sequence, the state line, the "
    "valid actions and the world model you carried)":
        "(applies to every turn; each user turn carries the outcome of the previous sequence, the state line, the "
        "valid actions and, when your previous reply's text had one, the world model you carried)",
    "The user turn shows the world model carried from your previous turn; copy it and add or remove items based on "
    "the evidence you gathered.":
        "When your previous reply's text (not its reasoning) had labelled world-model lines, the user turn shows them "
        "as the carried world model; update it from the evidence you gathered.",
}
SANDBOX_PROMPT_FIXES = {
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, "
    "itertools, json, math, operator, random, re, statistics, string.\n":
        "- The only importable modules are these standard-library ones: bisect, collections, copy, fractions, "
        "functools, hashlib, heapq, inspect (signature, getsource of your own functions), itertools, json, math, "
        "operator, pickle (dumps/loads of plain data), random, re, statistics, string, textwrap, time (clocks only; "
        "no sleep: the game advances only when you act), traceback. No third-party packages (numpy is not "
        "installed) and no file access.\n",
}
INTERRUPT_PROMPT_FIXES = {
    "variables you define stay available until the game ends or a timeout resets the interpreter (the tool result "
    "says so).":
        "variables you define stay available until the game ends; a call that runs past the tool timeout is "
        "interrupted and its earlier definitions are kept (the tool result says so, and says if the interpreter ever "
        "had to be reset).",
}
EMPTY_WORLD_MODEL_LINE = "end of world model. "
WORLD_MODEL_HEADER = "Working world model carried from earlier turns:"


def replace_pinned(text, replacements, what):
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"harness_fixes: pinned {what} text changed")
        text = text.replace(old, new, 1)
    return text
