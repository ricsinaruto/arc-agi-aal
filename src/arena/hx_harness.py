"""hero-v2 ``hx_harness``: the harness-evolution arm's components (3 October 2026; docs/research-harness/LOG.md),
at the reference setting (``SETTING``: toolbox, reflect style 2, economy style 1, the v8.2 playbook). All signals are
in-run (this game's own history); every constant is generic (no per-game information).

- ``toolbox`` (interface expansion): the persistent sandbox gains ``register_tool(fn, note=None)`` and
  ``register_observer(fn, note=None)``. A registered tool is listed (name, signature, note) at the end of every later
  user turn, so it survives context drains. An observer is called as ``fn(before_frame, after_frame)`` after every
  executed ``action()`` call; its short text goes into that call's result and into the next user turn.
- ``reflect`` (stuck detection): per level, the harness counts model requests, actions, actions that changed the board
  interior, deaths and revisits of identical boards. When the level stalls (``reflect_stall`` requests since the last
  interior change, ``reflect_no_change`` interior no-ops in a row, or a board seen ``reflect_revisits`` times; never on
  a death turn), the next user turn carries a STEP BACK block.
- ``economy``: the per-turn action count; ``memo``: the playbook (``playbook(text)``), shown in full after every cut
  and at each level start.
"""

import hashlib
import re
from collections import Counter

from .harness_fixes import exact

# The reference block (rl-min: every other value is refused). The code reads the reflect thresholds, tool_list_max and
# observer_chars; the other keys are the removed components' switches, kept off.
SETTING = {
    "toolbox": True, "reflect": True, "perceive": False, "probe": False, "economy": True, "kit": False, "opener": False,
    "guide": False, "grid": False, "lattice": False, "coverage": False, "merges": False, "autopilot": False,
    "worldmodel": False, "memo": True, "autoexplore": False, "stuckexplore": False, "no_stuck_notice": False,
    "reflect_requests": 8, "reflect_every": 8, "reflect_no_change": 5, "reflect_revisits": 3, "reflect_deaths": 2,
    "reflect_min_gap": 3, "reflect_style": 2, "reflect_stall": 5, "tool_list_max": 12, "observer_chars": 300,
    "digest_actions": 8, "autoexplore_actions": 10, "autoexplore_clicks": 6, "stuck_explore_actions": 8,
    "scene_objects": 14, "economy_style": 1, "memo_once": 1, "reflect_skip_death": 1, "memo_style": 2, "memo_limits": 0,
    "economy_spend": 0, "keep_rules": 0, "playbook_log": 0, "no_think": 0}
BORDER = 4  # the interior rule of dfr_harness gameplay_changed (timer bars sit in the outer ring)


def settings(value):
    return exact("hx_harness", value, SETTING)


# --- toolbox: system paragraph, sandbox child code, user-turn listing ------------------------------------------------

TOOLBOX_SYSTEM = (
    "Self-built tools (you can extend your own interface):\n"
    "- `register_tool(fn, note=None)` lists your python function `fn` (name, signature and `note` or the first "
    "docstring line) at the end of every later user turn. Definitions persist across turns and context drains, so a "
    "registered helper is never forgotten: build reusable helpers (object locators, action-effect tables, a "
    "simulator of the rules checked against `transitions`, a BFS planner, probes) once, register them, and reuse "
    "them instead of re-deriving the same facts every turn.\n"
    "- `register_observer(fn, note=None)`: after every executed `action(...)` call, `fn(before_frame, after_frame)` "
    "runs automatically and its short text result appears in that call's result under `observers` and in the next "
    "user turn. Use it to have the harness tell you exactly what you care about after each action (e.g. where the "
    "object you control is, a counter, whether a target changed).\n"
    "- `unregister(name)` removes a tool or observer.\n"
)
# v8.2: the memory framing, and the playbook without note() and without a length limit.
PLAYBOOK_OPEN_SYSTEM = (
    "- Memory: to keep requests short the oldest part of the conversation is dropped from time to time, and your "
    "reasoning in the dropped part is gone. Three things survive every cut: your playbook (your conclusions), "
    "`history` and `transitions` (every action and board of this game) and your python session (the variables and "
    "functions you defined). Before you spend an action to test something, check whether `history` already answers "
    "it.\n"
    "- Your playbook: `playbook(text)` replaces it and `playbook()` returns it. It has no length limit, but it is "
    "shown to you in full after every cut and at the start of each level, so write it as a briefing for yourself, "
    "not a journal. Two parts. Working model: the controls, the mechanics and the objective as you now understand "
    "them, each point marked `checked` (you saw it in `transitions`) or `assumed`; do not build a long plan on "
    "assumed points, and when the evidence contradicts a point or a plan built on it turns out impossible, that "
    "point is what is wrong: re-derive it instead of adding a mechanic to rescue it. Working memory: the level, "
    "where things are, your plan and next step, and one `ruled out: ...` line per idea this level's evidence has "
    "falsified. Rewrite it whenever you settle or drop something; at the end of a level fold what you learned into "
    "the working model and clear the working memory.\n"
)


PLAYBOOK_SHOWN = "Your playbook, as you last wrote it:\n\n"
PLAYBOOK_OPEN_DUE = ("The oldest part of the conversation will be dropped soon and you have not written your "
                     "playbook since the last cut. If it is out of date, rewrite it now with `playbook(text)`.")


def playbook_block(state):
    """Style 2: the playbook as a cut note and a level start show it."""
    text = str((state or {}).get("playbook") or "").strip()
    return PLAYBOOK_SHOWN + text if text else PLAYBOOK_EMPTY


PLAYBOOK_EMPTY = "(You have not written a playbook yet. Write one with `playbook(text)`.)"
PLAYBOOK_LEVEL = ("Level cleared. Before acting on the new level, rewrite your playbook: fold what this level taught "
                  "you into the working model and clear the working memory.")
SESSION_ANCHOR = "\n\n\nTool session rules:\n"
# economy: the cleared levels of the recorded runs took ~0.2-1x the human actions (the 115 cap) while each request costs
# ~1,500 tokens (~60-80 s at 38 streams): actions are cheap, deliberation is not.
ECONOMY_OLD = "- Optimize for as few in-game actions as possible while still being reliable.\n"
ECONOMY_NEW = (
    "- Your thinking time is the scarce resource, not game actions. A cleared level is scored by its action count "
    "against a human's, and humans typically need 20-60 actions for an early level, so the first ~20 actions on a "
    "level cost almost nothing while every long deliberation costs real time. When unsure what something does or "
    "which hypothesis is right, test it directly with a few actions (batch several experiments in one python call) "
    "instead of reasoning at length; once the mechanics are clear, execute an efficient plan.\n"
)


def system_prompt(text, block):
    """The system prompt with the economy line, and the toolbox and playbook paragraphs before the tool session
    rules."""
    if text.count(ECONOMY_OLD) != 1:
        raise ValueError("hx_harness: pinned system prompt line (as few in-game actions) changed")
    text = text.replace(ECONOMY_OLD, ECONOMY_NEW, 1)
    section = TOOLBOX_SYSTEM + PLAYBOOK_OPEN_SYSTEM
    if text.count(SESSION_ANCHOR) != 1:
        raise ValueError("hx_harness: pinned system prompt anchor (tool session rules) changed")
    return text.replace(SESSION_ANCHOR, "\n\n\n" + section.rstrip("\n") + SESSION_ANCHOR, 1)

CHILD_BLOCK = r'''
    _HX = {"tools": {}, "observers": {}, "out": {}, "out_step": None, "errors": 0, "loops": 0, "loop_actions": 0,
           "playbook": "", "playbook_step": None, "playbook_writes": 0, "kept": [], "kept_fns": {},
           "sim": None, "sim_report": None, "sim_seen": 0, "notes": {}}
    _HX_OBS_CHARS = __HX_OBS_CHARS__
    def _hx_key(fn, name):
        key = str(name or getattr(fn, "__name__", "") or "tool").strip()[:40]
        return key if key and key != "<lambda>" else "tool_%d" % (len(_HX["tools"]) + len(_HX["observers"]) + 1)

    def _hx_sig(fn):
        import inspect as _inspect
        try:
            return str(_inspect.signature(fn))[:120]
        except (TypeError, ValueError):
            return "(...)"

    def _hx_note(fn, note):
        text = note if note is not None else ((getattr(fn, "__doc__", None) or "").strip().split("\n")[0])
        return " ".join(str(text).split())[:160]

    def register_tool(fn, note=None, name=None):
        """List fn in every later user turn (it persists across turns and context drains); returns its name."""
        if not callable(fn):
            raise TypeError("register_tool(fn, note=None): fn must be a function")
        key = _hx_key(fn, name)
        _HX["tools"][key] = {"sig": _hx_sig(fn), "note": _hx_note(fn, note)}
        runtime_globals[key] = fn
        return key

    def register_observer(fn, note=None, name=None):
        """Call fn(before_frame, after_frame) after every executed action() call; its short text is reported."""
        if not callable(fn):
            raise TypeError("register_observer(fn, note=None): fn must be a function of (before_frame, after_frame)")
        key = _hx_key(fn, name)
        _HX["observers"][key] = {"fn": fn, "note": _hx_note(fn, note)}
        return key

    def unregister(name):
        """Remove a registered tool or observer by name."""
        found = _HX["tools"].pop(str(name), None) is not None
        found = _HX["observers"].pop(str(name), None) is not None or found
        return found

    _hx_inner_action = runtime_globals["action"]

    def action(actions):
        before = runtime_globals.get("current_frame")
        result = _hx_inner_action(actions)
        if _HX["observers"] and isinstance(result, dict) and result.get("executed"):
            after = runtime_globals.get("current_frame")
            outputs = {}
            for key, item in list(_HX["observers"].items()):
                try:
                    text = item["fn"](before, after)
                except Exception as exc:  # an observer must never break the action call
                    _HX["errors"] += 1
                    text = "observer error: %s: %s" % (type(exc).__name__, str(exc)[:120])
                outputs[key] = str(text)[:_HX_OBS_CHARS]
            result["observers"] = outputs
            _HX["out"] = dict(outputs)
            _HX["out_step"] = getattr(after, "step", None)
        return result

    runtime_globals["action"] = action

    def playbook(text=None):
        """Your playbook: playbook(text) replaces it, playbook() returns it. It is what survives when the
        conversation is cut."""
        if text is not None:
            text = str(text).strip()
            print("[playbook] saved, %d characters" % len(text))
            _HX["playbook"] = text
            _HX["playbook_writes"] += 1
            _HX["playbook_step"] = getattr(runtime_globals.get("current_frame"), "step", None)
        return _HX["playbook"]

    _hx_refresh_inner = _refresh_state

    def _refresh_state(state_payload):
        _hx_refresh_inner(state_payload)
        runtime_globals["register_tool"] = register_tool
        runtime_globals["register_observer"] = register_observer
        runtime_globals["unregister"] = unregister
        runtime_globals["playbook"] = playbook

    def _hx_take():
        return {"tools": {k: dict(v) for k, v in _HX["tools"].items()},
                "observers": {k: {"note": v["note"]} for k, v in _HX["observers"].items()},
                "out": dict(_HX["out"]), "out_step": _HX["out_step"], "errors": _HX["errors"],
                "loops": _HX["loops"], "loop_actions": _HX["loop_actions"], "sim": _HX["sim_report"],
                "playbook": _HX["playbook"], "playbook_step": _HX["playbook_step"],
                "playbook_writes": _HX["playbook_writes"], "notes": dict(_HX["notes"]),
                "kept": [{k: v for k, v in e.items() if k != "seen"} for e in _HX["kept"]]}

'''
ANCHOR = "    request = initial\n    while True:\n"
RESULT_FIELD = '"action_results": _json_safe(action_results),'
RUNTIME_NAMES = '"last_action_result", "action", "result"'
# the fixes loop's inventory skips these names (the removed helpers' too: a model's own `probe` or `note` stays out of
# the inventory as on v8.5p; harness parity)
HX_RUNTIME_NAMES = ('"register_tool", "register_observer", "unregister", "probe", "crop", "objects", "at", "changes", '
                    '"lattice", "autopilot", "explore", "check_sim", "register_sim", "plan", "note", "playbook"')


def with_toolbox(bootstrap, block):
    """``bootstrap`` with the toolbox child code (after any dfr_harness code: its action wrapper wraps theirs) and the
    ``hx`` field in the call's final and error messages."""
    if "_HX = {" in bootstrap:
        raise ValueError("hx_harness: the toolbox code is already installed")
    if bootstrap.count(ANCHOR) != 1:
        raise ValueError("hx_harness: pinned Duck sandbox loop changed")
    child = CHILD_BLOCK.replace("__HX_OBS_CHARS__", str(int(block["observer_chars"])))
    text = bootstrap.replace(ANCHOR, child.lstrip("\n") + ANCHOR, 1)
    if text.count(RESULT_FIELD) != 2:
        raise ValueError("hx_harness: pinned Duck sandbox result messages changed")
    text = text.replace(RESULT_FIELD, RESULT_FIELD + ' "hx": _hx_take(),')
    if text.count(RUNTIME_NAMES) == 1:  # the fixes loop's inventory skips runtime names
        text = text.replace(RUNTIME_NAMES, RUNTIME_NAMES + ", " + HX_RUNTIME_NAMES, 1)
    return text


def toolbox_lines(state, block):
    """The user-turn footer for the child's last ``hx`` report (None: nothing registered)."""
    if not state:
        return None
    lines = []
    tools = list((state.get("tools") or {}).items())
    if tools:
        shown = tools[-block["tool_list_max"]:]
        lines.append("Your registered tools (persist across turns; call them from python): " + "; ".join(
            f"{name}{item.get('sig', '')}" + (f" - {item['note']}" if item.get("note") else "") for name, item in shown)
            + (f"; (+{len(tools) - len(shown)} older)" if len(tools) > len(shown) else ""))
    observers = state.get("observers") or {}
    out = state.get("out") or {}
    if observers:
        if out:
            lines.append(f"Your observers after the last action (step {state.get('out_step')}): " + " | ".join(
                f"{name}: {text}" for name, text in out.items()))
        else:
            lines.append("Your registered observers: " + ", ".join(observers))
    return "\n".join(lines) or None


# --- reflect: in-run stuck detection and the STEP BACK block ---------------------------------------------------------

def _hash(grid):
    return hashlib.blake2s(repr(grid).encode(), digest_size=10).hexdigest()


def _interior_changed(before, after):
    rows = len(after)
    if before is None or len(before) != rows:
        return True
    for r in range(BORDER, rows - BORDER):
        a, b = before[r], after[r]
        if a[BORDER:len(a) - BORDER] != b[BORDER:len(b) - BORDER]:
            return True
    return False


ACTION_KIND = re.compile(r"^MOUSE\(")


class Reflector:
    """Per-level counters over the game's own history (``duck_history`` entries and their receipts)."""

    def __init__(self, block):
        self.block = block
        self.seen = 0
        self.level = None
        self.notices = 0
        self.last_reason = None
        self.fired = Counter()
        self.reset_level(0, 0, 0)

    def reset_level(self, levels, requests, tokens):
        self.level = levels
        self.requests0, self.tokens0 = requests, tokens
        self.actions = self.changed = self.deaths = self.no_change = 0
        self.boards = Counter()
        self.effects = {}  # action kind -> [tried, interior changes, deaths]
        self.last_notice = None
        self.change_request = 0  # style 2: the level's request count at the last interior change
        self.revisit_noticed = 0
        self.streak_noticed = 0
        self.died_now = False  # a death among the entries of the latest update (reflect_skip_death)

    def update(self, history, results, levels, requests, tokens):
        """Consume the new history entries; ``levels`` = levels cleared now."""
        if levels != self.level:
            # A clear (or the first call): the new level starts at the newest board; the entries up to it (the
            # clearing action) belong to the previous level.
            self.reset_level(levels, requests, tokens)
            if history:
                self.boards[_hash(history[-1].frame.grid)] += 1
                self.seen = len(history)
        previous = history[self.seen - 1].frame.grid if 0 < self.seen <= len(history) else None
        deaths_before = self.deaths
        for index in range(self.seen, len(history)):
            entry = history[index]
            grid = entry.frame.grid
            if entry.action and entry.action != "RESET":
                result = results.get(entry.frame.step) or {}
                kind = "MOUSE" if ACTION_KIND.match(entry.action) else entry.action
                changed = _interior_changed(previous, grid)
                died = bool(result.get("game_over"))
                row = self.effects.setdefault(kind, [0, 0, 0])
                row[0] += 1
                row[1] += changed
                row[2] += died
                self.actions += 1
                self.changed += changed
                if changed:
                    self.change_request = requests - self.requests0
                self.deaths += died
                self.no_change = 0 if changed else self.no_change + 1
            self.boards[_hash(grid)] += 1
            previous = grid
        self.died_now = self.deaths > deaths_before
        self.seen = len(history)
        self.requests = requests - self.requests0
        self.tokens = tokens - self.tokens0
        self.current = _hash(history[-1].frame.grid) if history else None

    def due(self):
        b = self.block
        if self.last_notice is not None and self.requests - self.last_notice < b["reflect_min_gap"]:
            return None
        if self.died_now:  # the death turn has its own text (reflect_skip_death)
            return None
        if self.no_change >= b["reflect_no_change"] and self.no_change >= self.streak_noticed + b["reflect_no_change"]:
            return "no_change"
        visits = self.boards.get(self.current, 0)
        if visits >= b["reflect_revisits"] and visits > self.revisit_noticed:
            return "revisits"
        stalled = self.requests - max(self.change_request, self.last_notice or 0)
        return "stall" if stalled >= b["reflect_stall"] else None

    def notice(self, level_number):
        reason = self.due()
        if reason is None:
            return None
        self.last_notice = self.requests
        self.last_reason = reason
        self.notices += 1
        self.fired[reason] += 1
        if reason == "no_change":
            self.streak_noticed = self.no_change
        if reason == "revisits":
            self.revisit_noticed = self.boards.get(self.current, 0)
        why = {"no_change": f"your last {self.no_change} actions changed nothing inside the board",
               "revisits": f"this exact board has now been seen {self.boards.get(self.current, 0)} times on this level",
               "stall": f"{self.requests - self.change_request} model requests since an action last changed the board "
                        "interior"}[reason]
        table = ", ".join(f"{kind} {tried}x ({changed} changed the board{f', {died} deaths' if died else ''})"
                          for kind, (tried, changed, died) in sorted(self.effects.items())) or "no actions yet"
        return (
            f"STEP BACK (harness check: {why}). Level {level_number}: {self.requests} requests, "
            f"{self.tokens:,} generated tokens, {self.actions} actions ({self.changed} changed the interior), "
            f"{self.deaths} deaths, {len(self.boards)} distinct boards; effects: {table}. The current approach is "
            "not working and more deliberation will not fix it. In this reply: name the belief you are least sure "
            "of and the cheapest experiment that would refute it; if your tools or your view of the board are the "
            "bottleneck, ask how you can modify your own code or your interface to the environment: write the "
            "helper or observer you lack and register it (`register_tool(fn)`, `register_observer(fn)`); then run "
            "the experiment now (several actions in one call are fine) instead of planning further."
        )

    def summary(self):
        return {"notices": self.notices, **dict(self.fired)}


def economy_line(level_actions, level_number):
    """The per-turn action count on the level (economy)."""
    return (f"Action economy: {level_actions} actions so far on level {level_number} (the first ~20 cost almost "
            "nothing in score; experiments are cheaper than long deliberation).")

