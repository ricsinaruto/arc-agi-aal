"""hero-v2 ``hx_harness``: the harness-evolution arm's components (3 October 2026; docs/research-harness/LOG.md).

One opt-in study key ``hx_harness`` (``settings``): None on every earlier build, so their sandboxes, prompts and
payloads stay byte for byte. True switches every flag on with the defaults; a dict selects flags (booleans over
``FLAGS``) and overrides numeric settings (``DEFAULTS``). All signals are in-run (this game's own history); every
constant is generic (no per-game information).

- ``toolbox`` (interface expansion): the persistent sandbox gains ``register_tool(fn, note=None)`` and
  ``register_observer(fn, note=None)``. A registered tool is listed (name, signature, note) at the end of every later
  user turn, so it survives context drains and the agent's own helpers become part of its interface. An observer is
  called as ``fn(before_frame, after_frame)`` after every executed ``action()`` call; its short text goes into that
  call's result (``result['observers']``) and into the next user turn, so the agent chooses what it is told after
  each action. One system paragraph explains both.
- ``reflect`` (stuck detection + self-modification prompt): from the game's own history the harness counts, per
  level, model requests, actions, actions that changed the board interior, deaths and revisits of identical boards.
  When the level stalls (``reflect_requests`` requests without a clear, again every ``reflect_every``; or
  ``reflect_no_change`` interior-no-op actions in a row; or a repeated death; or a board seen ``reflect_revisits``
  times), the next user turn carries a STEP BACK block: the counts, an action-effect table, and the questions what
  each attempt taught, which untested goal hypotheses remain, and "how can I modify my own code or my interfaces to
  the environment to improve?" (write and register a helper or an observer, then use it).

The history stays append-only between evictions: every addition is text in the new user turn.
"""

import hashlib
import re
from collections import Counter

FLAGS = ("toolbox", "reflect", "perceive", "probe", "economy", "kit", "opener", "guide", "grid", "lattice",
         "coverage", "merges", "autopilot", "worldmodel", "memo", "autoexplore", "stuckexplore", "no_stuck_notice")
DEFAULTS = {
    "reflect_requests": 8,      # model requests on a level without a clear before the first STEP BACK
    "reflect_every": 8,         # requests between STEP BACK blocks while the level stays uncleared
    "reflect_no_change": 5,     # consecutive executed actions without an interior change
    "reflect_revisits": 3,      # an identical board seen this many times on the level
    "reflect_deaths": 2,        # the second death on a level (and every later one)
    "reflect_min_gap": 3,       # requests at least between two STEP BACK blocks, whatever the trigger
    "reflect_style": 1,         # 1: the v1 block (request-count trigger, three questions); 2: progress-aware triggers
                                # only (stall, streak, revisits, deaths) and a short act-now block
    "reflect_stall": 5,         # style 2: requests since the last action that changed the board interior
    "tool_list_max": 12,        # registered tools listed in a user turn
    "observer_chars": 300,      # characters kept from one observer's output
    "digest_actions": 8,        # executed actions described one by one in the next user turn
    "autoexplore_actions": 10,  # autoexplore: harness-played actions at the start of each level (0 tokens)
    "autoexplore_clicks": 6,    # autoexplore: distinct small interior objects clicked (click games)
    "stuck_explore_actions": 8,  # stuckexplore: harness-played untried actions / clicks when the level stalls
    "scene_objects": 14,        # object groups listed in a level's scene summary
    "economy_style": 1,         # 2: also replace the "do not ration tool calls" lines (inspect and act in one call);
                                # 3: economy plus a lean-reasoning line after it
    "memo_once": 0,             # 1: show the notes line only when it changed, after a drain/compaction or on a new level
                                # (0: every user turn, which repeats ~250-500 tokens per turn into the history)
    "reflect_skip_death": 0,    # 1 (prompt audit item 6): no STEP BACK on a death turn and no deaths trigger (the
                                # death turn has its own text)
    "memo_style": 0,            # 1 (hx7 trim bed, arm F): one playbook text of 5,000 characters in place of the note
                                # slots; it is what survives a trim (``trim.summary: playbook``), not shown per turn
                                # 2 (v8.2): the playbook replaces note() and the notes block: no length limit, shown
                                # in full after every context cut and at a level start, a receipt line per write
    "memo_limits": 0,           # 1 (item 8): the system line states the 12 x 400 limits and note() prints a cut or
                                # a dropped key
    "economy_spend": 0,         # 1: the per-turn line also reports the requests, generated tokens and active seconds
                                # spent since the previous turn (in-run feedback on deliberation cost)
    "keep_rules": 0,            # (10 October, with memo_style 2) 1: at each clear the model writes what completed the
                                # level with won(text); it is kept beside the playbook, which cannot change it.
                                # 2: also keep_rule(text, test): a mechanic kept once its test holds on the recorded
                                # transitions, re-checked on every later one and marked when one breaks it
    "playbook_log": 0,          # 1: the playbook is logged at each level start, first action and 100K tokens on a level
    "no_think": 0,              # 1: the run serves with thinking off. The reply text is the
                                # only place to reason: a system line and a per-turn line ask for a short
                                # belief-and-plan text before each call, and the texts that speak of hidden
                                # reasoning or deliberation are reworded (playbook framing, STEP BACK)
}
ZERO_OK = ("economy_spend", "memo_once", "reflect_skip_death", "memo_limits", "memo_style", "keep_rules", "playbook_log",
           "no_think")
BORDER = 4  # the interior rule of dfr_harness gameplay_changed (timer bars sit in the outer ring)


def settings(value):
    """The study block (None when off)."""
    if value is None or value is False:
        return None
    if value is True:
        value = {flag: True for flag in FLAGS}
    if not isinstance(value, dict):
        raise ValueError("hx_harness is true/false or a dict over " + ", ".join(FLAGS + tuple(DEFAULTS)))
    unknown = sorted(set(value) - set(FLAGS) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown hx_harness keys {unknown}")
    block = {flag: bool(value.get(flag, False)) for flag in FLAGS}
    for flag in FLAGS:
        if flag in value and type(value[flag]) is not bool:
            raise ValueError(f"hx_harness.{flag} is a boolean")
    for key, default in DEFAULTS.items():
        raw = value.get(key, default)
        if type(raw) is not int or raw < (0 if key in ZERO_OK else 1):
            raise ValueError(f"hx_harness.{key} is a positive integer" + (" or 0" if key in ZERO_OK else ""))
        block[key] = raw
    return block if any(block[flag] for flag in FLAGS) else None


def describe(block):
    return {"flags": {f: block[f] for f in FLAGS}, "settings": {k: block[k] for k in DEFAULTS},
            "module": "arena.hx_harness"}


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
PROBE_SYSTEM = (
    "- `probe(actions)` runs a list of actions one at a time as deliberate experiments (an interior no-op does not "
    "stop the rest) and returns one row per action: `action`, `gameplay_changed`, `changed_cells`, `change_bbox`, "
    "counts of `moved`/`appeared`/`disappeared` objects and `animated`; it stops after a game over, a level clear or "
    "the win. Use it to test several untried actions in one call, e.g. `probe(['UP', 'DOWN', 'SPACE'])`. Every "
    "probed action really changes the game state, so in games where clicks toggle or select things probe one to "
    "three targets at a time and read the rows before probing more.\n"
)
LATTICE_SYSTEM = (
    "- Many boards are drawn on a coarse lattice (each logical cell is a k x k block of pixels, often with an offset). "
    "`lattice(pitch=None, origin=None, mode='center')` returns that lattice's cell map, one letter per cell, with the "
    "pixel <-> cell formulas; by default it infers the cell size from how far objects moved per action. Use it to "
    "plan moves on cells instead of pixels.\n"
)
AUTOPILOT_SYSTEM = (
    "- Closed-loop control (one python call can play many actions; you are woken only when something unexpected "
    "happens): `autopilot(policy, max_actions=60, until=None, expect=None, max_noops=3)` repeatedly executes "
    "`policy(current_frame)` (an action name, a MOUSE dict, or None to stop) and stops on None, `until(frame)`, "
    "`expect(before, after, action)` returning False (a surprise), repeated no-ops, a cleared level or a game over. "
    "`explore(max_actions=30, clicks=8)` tries, state by state, every action and a click on each distinct small "
    "object not yet tried there (walking known transitions back to states with untried options) and reports what "
    "each kind of action did. Once you know what to do, encode it as a policy and let autopilot run it instead of "
    "deciding each action in a new turn; when you do not know what anything does, explore() is cheaper than "
    "reasoning about it.\n"
)
WORLDMODEL_SYSTEM = (
    "- Executable world model: write `sim(rows, action) -> rows` (rows: `current_frame.ascii.split('\\n')`, a list "
    "of strings; action: a name or a MOUSE dict) that predicts the next board. `check_sim(sim)` replays it on every "
    "recorded transition of this level and shows the first wrong cells; `register_sim(sim)` keeps it checked against "
    "each new action (reported in your next turns). Once it predicts exactly, `plan(sim, goal, actions=None, "
    "max_depth=40)` breadth-first searches for the shortest action list with `goal(rows) == True`; execute the "
    "path, and if a step contradicts the sim, fix the sim rather than guessing.\n"
)
MEMO_SYSTEM = (
    "- Notes that outlive the context: `note(key, text)` stores a short entry (mechanics, goal, controls, what "
    "failed and why, what worked); every user turn shows your notes, also after the conversation history is cut "
    "and on later levels. Keep them current: they are your long-term memory of this game.\n"
)
# After Retrodict's playbook.md (externals/Retrodict/src/arc3/prompts.py:32-56).
PLAYBOOK_CHARS = 5000
PLAYBOOK_SYSTEM = (
    "- Your playbook: to keep requests short the conversation is cut from time to time: everything except the "
    "newest few messages is dropped, and your reasoning in the dropped part is gone. What survives is the python "
    "session and your playbook. `playbook(text)` replaces it (at most 5,000 characters); `playbook()` returns it. "
    "Write it as a briefing for yourself after the cut, in two parts. Working model: the controls, the mechanics "
    "and the objective as you now understand them, each point marked `checked` (you saw it in `transitions`) or "
    "`assumed`; do not build a long plan on assumed points, and when the evidence contradicts a point or a plan "
    "built on it turns out impossible, that point is what is wrong: re-derive it instead of adding a mechanic to "
    "rescue it. Working memory: the level, where things are, your plan and next step, and one `ruled out: ...` "
    "line per idea this level's evidence has falsified. Rewrite it whenever you settle or drop something; at the "
    "end of a level fold what you learned into the working model and clear the working memory. Keep it a "
    "briefing, not a journal.\n"
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
KEEP_SYSTEM = (
    "- Kept lines: when you clear a level, the first thing you do on the next one is call `won(text)`: one or two "
    "sentences on what the winning action brought about on the board and why that completed the level, in words "
    "that also fit every level you won before (compare the boards before and after each winning step in "
    "`transitions`; if the sentence does not fit one of those wins, it is not right yet). It is stored once per "
    "level, `playbook(text)` cannot change it, and it is shown with your playbook after every cut. When your "
    "playbook and a kept line disagree, the kept line was written from the evidence of a win: re-check the "
    "playbook.\n"
)
KEEP_RULES_SYSTEM = (
    "- `keep_rule(text, test)` keeps a mechanic the same way once the evidence supports it. `test(t)` takes one "
    "entry of `transitions` (`t.action`, `t.before_frame`, `t.after_frame`) and returns True where the rule held, "
    "False where it was broken and None where it says nothing about that step. The rule is kept only if it held "
    "on every recorded step it applies to (at least 5); it is checked again on every later step, and a kept rule "
    "that a later step breaks is marked BROKEN. Keep the rules your plans rest on (what each action does, what "
    "blocks a move, what cannot move); a simulator that predicts the next board is the strongest test. A rule you "
    "have not kept is a guess: do not build a long plan on it.\n"
)
KEEP_LEVEL = ("Level cleared. First call `won(text)`: what the winning action brought about on the board and why that "
              "completed the level, in words that also fit every level you won before (compare the boards around "
              "each winning step in `transitions`). Then rewrite your playbook: fold what this level taught you into "
              "the working model and clear the working memory.")
KEEP_DUE = "You have not kept what completed level {level} yet: call `won(text)` before you go on."
KEEP_BROKEN = ("Kept rule {name!r} no longer holds: step {step} broke it. Whatever in your playbook or plan rests on "
               "it is in doubt; re-derive it from `transitions`.")
KEPT_HEAD = "Kept lines (written at a win or after a test; `playbook(text)` cannot change them):"


def kept_block(state):
    """The kept lines as shown beside the playbook; '' without any."""
    kept = (state or {}).get("kept") or []
    lines = []
    for entry in kept:
        if entry.get("kind") == "win":
            lines.append(f"- Level {entry.get('level')} was completed by: {entry.get('text')}")
        elif entry.get("broken") is not None:
            lines.append(f"- BROKEN at step {entry.get('broken')} (re-derive it): {entry.get('text')}")
        else:
            lines.append(f"- Rule (held on {entry.get('held')} steps): {entry.get('text')}")
    return KEPT_HEAD + "\n" + "\n".join(lines) if lines else ""


def _with_kept(text, state):
    kept = kept_block(state)
    return text + "\n\n" + kept if kept else text


# no_think: thinking is off, so the visible reply is the model's only reasoning.
NO_THINK_SYSTEM = (
    "- You are running without a hidden reasoning phase: the text of your reply is the only place where you can "
    "reason, and only what you write is kept. Before every `python` call write two to four short lines of plain "
    "text: what the last result showed, what you now believe about the rule or the goal, and what this call tests "
    "or does with the result you expect. Then make exactly one `python` call. Never send a bare call.\n"
)
NO_THINK_TURN = ("Reply format: two to four short lines first (what the last result showed, what you now believe, "
                 "what this call does and the result you expect), then one `python` call.")
NO_THINK_REWORDS = (
    ("and your reasoning in the dropped part is gone.", "and what you wrote in the dropped part is gone."),
)
STEP_BACK_DELIBERATION = " and more deliberation will not fix it"


def no_think_text(text):
    """A system text with the phrases about hidden reasoning reworded (``no_think``)."""
    for old, new in NO_THINK_REWORDS:
        if text.count(old) != 1:
            raise ValueError(f"hx_harness: pinned text for no_think changed: {old!r}")
        text = text.replace(old, new, 1)
    return text


PLAYBOOK_SHOWN = "Your playbook, as you last wrote it:\n\n"
PLAYBOOK_OPEN_DUE = ("The oldest part of the conversation will be dropped soon and you have not written your "
                     "playbook since the last cut. If it is out of date, rewrite it now with `playbook(text)`.")


def playbook_block(state):
    """Style 2: the playbook as a cut note and a level start show it."""
    text = str((state or {}).get("playbook") or "").strip()
    return _with_kept(PLAYBOOK_SHOWN + text if text else PLAYBOOK_EMPTY, state)


PLAYBOOK_HANDOFF = (
    "[Context cut] The conversation before the messages below was dropped to keep requests short. Your python "
    "session is unchanged. Your playbook, as you last wrote it:\n\n"
)
PLAYBOOK_EMPTY = "(You have not written a playbook yet. Write one with `playbook(text)`.)"
PLAYBOOK_NOTES = "(No playbook yet; these are your note() entries.)\n"
PLAYBOOK_LEVEL = ("Level cleared. Before acting on the new level, rewrite your playbook: fold what this level taught "
                  "you into the working model and clear the working memory.")
PLAYBOOK_DUE = ("The conversation will be cut soon; only your playbook and the newest messages survive. If the "
                "playbook is out of date, rewrite it now with `playbook(text)`.")


def playbook_text(state):
    """What a trim carries: the playbook, else the note entries, else a line saying there is none."""
    state = state or {}
    text = str(state.get("playbook") or "").strip()
    if text:
        return _with_kept(text, state)
    notes = state.get("notes") or {}
    if notes:
        return PLAYBOOK_NOTES + "\n".join(f"- {key}: {value}" for key, value in notes.items())
    return _with_kept(PLAYBOOK_EMPTY, state)


MEMO_LIMITS_SYSTEM = (
    "- Notes that outlive the context: `note(key, text)` stores an entry (mechanics, goal, controls, what failed and "
    "why, what worked); at most 12 entries of 400 characters: a longer text is cut and a 13th key drops the oldest "
    "entry (`note(key)` deletes one, the same key replaces it). Your notes are shown when they change, after the "
    "conversation history is cut and on later levels. Keep them current: they are your long-term memory of this "
    "game.\n"
)
KIT_SYSTEM = (
    "- Built-in board helpers (use them instead of writing your own parsers): `crop(r0, r1, c0, c1, frame=None)` "
    "prints rows r0..r1-1 / cols c0..c1-1 with row numbers and a column ruler; `objects(color=None, min_px=1, "
    "max_px=None, frame=None)` lists objects with `bbox` [r0, c0, r1, c1] (inclusive), `height`, `width`, `center`; "
    "`at(r, c)` returns the object containing a cell; `changes(before=None, after=None)` lists the changed cells "
    "(default `previous_frame` -> `current_frame`). Colours are the letter codes.\n"
)
PERCEIVE_SYSTEM = (
    "- Each user turn also carries a harness digest: what each action since your previous turn changed (cells, "
    "moved/appeared/disappeared objects, animations, HUD-only changes), a per-level ledger of the action types and "
    "click targets you have tried and what they did, the action types not yet tried on this level, on a new level "
    "the objects that were not on the previous level, a scene summary at the start of each level (regions, object "
    "groups with positions, border strips), and a note when the board returned to the level's start. "
    "Read it before writing inspection code; an action type that did nothing in one situation may still matter "
    "in another (another target, another state).\n"
)
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


# grid (coordinate grounding, harness.md section 9): every image of the current observation (board, diff, death
# frame) gets a dark top and left margin with a tick per cell and row / column numbers every 8 cells, so positions in
# the image map to `row` / `col` without counting; the board pixels themselves are untouched.
GRID_MARGIN = 32
GRID_MARGIN_RGB = (56, 40, 24)
GRID_SYSTEM = (
    "- Every board image (and diff image) has a dark brown margin on its top and left with a tick per cell and the row / "
    "column numbers 0, 8, 16, ..., 56: read coordinates from them (row = down, col = right) instead of counting "
    "cells.\n"
)
GRID_ANCHOR = "- The image and `current_frame.ascii` are two representations of the same current frame.\n"


def with_coordinates(url, cells=64):
    """A board-style PNG data URL with the coordinate margin (unchanged if it is not a square multiple of cells)."""
    import base64
    import io

    from PIL import Image, ImageDraw, ImageFont

    head, _, data = url.partition(",")
    if not head.startswith("data:image/png"):
        return url
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    if image.width != image.height or image.width % cells:
        return url
    unit, m = image.width // cells, GRID_MARGIN
    canvas = Image.new("RGB", (image.width + m, image.height + m), GRID_MARGIN_RGB)  # off-palette dark brown
    canvas.paste(image, (m, m))
    draw = ImageDraw.Draw(canvas)
    draw.line([(m - 1, m - 1), (canvas.width - 1, m - 1)], fill=(150, 150, 150))
    draw.line([(m - 1, m - 1), (m - 1, canvas.height - 1)], fill=(150, 150, 150))
    try:
        font = ImageFont.load_default(size=13)
    except (TypeError, OSError):  # Pillow without FreeType
        font = ImageFont.load_default()
    for i in range(cells):
        major = i % 8 == 0
        length = 8 if major else 3
        x = m + i * unit + unit // 2
        draw.line([(x, m - length), (x, m - 1)], fill=(200, 200, 200))
        draw.line([(m - length, x), (m - 1, x)], fill=(200, 200, 200))
        if major:
            draw.text((x, 2), str(i), fill=(235, 235, 235), font=font, anchor="ma")
            draw.text((m - 10, x), str(i), fill=(235, 235, 235), font=font, anchor="rm")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# guide: generic goal and mechanic archetypes of grid puzzle games (no game-specific rule), to widen the hypotheses the
# model considers before it commits; inserted before the visual-game guidance.
GUIDE = (
    "Field guide (generic patterns of these games; use them to generate hypotheses, then test):\n"
    "- Typical goals: bring a piece onto a target or into a slot of the same shape or colour; make one region match a "
    "reference pattern shown elsewhere on the board (often a small panel or key); connect two endpoints with a path; "
    "collect or remove every item of a kind; set switches or cells so that indicators are satisfied; arrange, sort or "
    "rotate pieces until they fit.\n"
    "- Typical mechanics: you may control one piece, several pieces in lockstep, a cursor, or nothing directly (clicks "
    "change cells or objects); pieces may slide until blocked, move along rails, push other pieces, or wrap around; "
    "clicking may select an object and a second click may place or swap it; a bar or row of cells that shrinks each "
    "action is a step budget (running out is a game over); some objects appear only as transient animation frames.\n"
    "- A level clear usually keeps the earlier mechanics and adds one new element: find what is new and what it does.\n"
    "- When nothing obvious happens, the goal is often signalled by a matching colour or shape between a movable thing "
    "and a fixed thing.\n"
)
GUIDE_ANCHOR = "\n\n\nVisual-game guidance:\n"
LEAN_REASONING = (
    "- Keep your reasoning lean: state the current hypothesis, the test and what you expect in a few lines. Do not "
    "re-derive facts you already established, do not enumerate every alternative before acting, and when two "
    "readings of the board compete, let an action settle it rather than more analysis.\n"
)
CALLS_OLD = (
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear "
    "probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between "
    "frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
)
CALLS_NEW = (
    "- Every python call costs a full model turn of thinking time. Combine inspection and action: inspect, then act "
    "or run an experiment in the same call whenever you can. A call that only prints is worth it only when its "
    "answer decides between different next actions; the action receipt already reports what changed "
    "(`transition`, `gameplay_changed`, `board_changed`).\n"
)


def system_prompt(text, block):
    """The system prompt with the on flags' paragraphs (the toolbox section before the tool session rules)."""
    if block["grid"]:
        if text.count(GRID_ANCHOR) != 1:
            raise ValueError("hx_harness: pinned system prompt line (image and ascii) changed")
        text = text.replace(GRID_ANCHOR, GRID_ANCHOR + GRID_SYSTEM, 1)
    if block["guide"]:
        if text.count(GUIDE_ANCHOR) != 1:
            raise ValueError("hx_harness: pinned system prompt anchor (visual-game guidance) changed")
        text = text.replace(GUIDE_ANCHOR, "\n\n\n" + GUIDE.rstrip("\n") + GUIDE_ANCHOR, 1)
    if block["economy"]:
        if text.count(ECONOMY_OLD) != 1:
            raise ValueError("hx_harness: pinned system prompt line (as few in-game actions) changed")
        text = text.replace(ECONOMY_OLD, ECONOMY_NEW + (LEAN_REASONING if block["economy_style"] == 3 else ""), 1)
        if block["economy_style"] == 2:
            if text.count(CALLS_OLD) != 1:
                raise ValueError("hx_harness: pinned system prompt lines (do not ration tool calls) changed")
            text = text.replace(CALLS_OLD, CALLS_NEW, 1)
    if block["no_think"]:
        anchor = ECONOMY_NEW if block["economy"] else ECONOMY_OLD
        if text.count(anchor) != 1:
            raise ValueError("hx_harness: pinned system prompt line (in-game actions) changed")
        text = text.replace(anchor, anchor + NO_THINK_SYSTEM, 1)
    section = ""
    if block["toolbox"]:
        section += TOOLBOX_SYSTEM
    playbook_open = no_think_text(PLAYBOOK_OPEN_SYSTEM) if block["no_think"] else PLAYBOOK_OPEN_SYSTEM
    extra = ((PROBE_SYSTEM if block["probe"] else "") + (KIT_SYSTEM if block["kit"] else "")
             + (LATTICE_SYSTEM if block["lattice"] else "") + (AUTOPILOT_SYSTEM if block["autopilot"] else "")
             + (WORLDMODEL_SYSTEM if block["worldmodel"] else "") + ((playbook_open if block["memo_style"] == 2 else PLAYBOOK_SYSTEM if block["memo_style"]
                 else MEMO_LIMITS_SYSTEM if block["memo_limits"] else MEMO_SYSTEM)
                if block["memo"] else "")
             + ((KEEP_SYSTEM + (KEEP_RULES_SYSTEM if block["keep_rules"] == 2 else ""))
                if block["memo"] and block["memo_style"] == 2 and block["keep_rules"] else "")
             + (PERCEIVE_SYSTEM if block["perceive"] else ""))
    if extra:
        if not section:
            section = "Harness aids:\n"
        section += extra
    if section:
        if text.count(SESSION_ANCHOR) != 1:
            raise ValueError("hx_harness: pinned system prompt anchor (tool session rules) changed")
        text = text.replace(SESSION_ANCHOR, "\n\n\n" + section.rstrip("\n") + SESSION_ANCHOR, 1)
    return text

CHILD_BLOCK = r'''
    _HX = {"tools": {}, "observers": {}, "out": {}, "out_step": None, "errors": 0, "loops": 0, "loop_actions": 0,
           "playbook": "", "playbook_step": None, "playbook_writes": 0, "kept": [], "kept_fns": {},
           "sim": None, "sim_report": None, "sim_seen": 0, "notes": {}}
    _HX_OBS_CHARS = __HX_OBS_CHARS__
    _HX_FLAGS = __HX_FLAGS__

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

    def probe(actions):
        """Execute each action in turn as a deliberate experiment (no stale-state refusal between them) and return
        one effect row per action; stops after a game over, a level clear or the win."""
        if isinstance(actions, (str, dict)):
            actions = [actions]
        rows = []
        for item in list(actions):
            try:
                _DFR["stale"] = None  # dfr_harness latch: a probe's no-op is expected, not a stale belief
            except NameError:
                pass
            result = runtime_globals["action"]([item])
            if not isinstance(result, dict):
                break
            transition = result.get("transition") or {}
            counts = (transition.get("objects") or {}).get("counts") or {}
            executed = result.get("executed_actions") or []
            row = {"action": executed[-1] if executed else str(item), "executed": bool(result.get("executed")),
                   "gameplay_changed": result.get("gameplay_changed", result.get("board_changed")),
                   "changed_cells": transition.get("changed_cells"), "change_bbox": transition.get("change_bbox"),
                   "moved": counts.get("moved", 0), "appeared": counts.get("appeared", 0),
                   "disappeared": counts.get("disappeared", 0), "animated": bool(result.get("animation"))}
            for flag in ("game_over", "level_completed", "done"):
                if result.get(flag):
                    row[flag] = True
            if result.get("observers"):
                row["observers"] = result["observers"]
            if not result.get("executed"):
                row["error"] = result.get("error") or result.get("stop_reason")
            rows.append(row)
            if not result.get("executed") or any(result.get(f) for f in ("game_over", "level_completed", "done")):
                break
        return rows

    def _hx_frame(frame):
        return frame if frame is not None else runtime_globals.get("current_frame")

    def crop(r0, r1, c0, c1, frame=None):
        """Rows r0..r1-1 and cols c0..c1-1 of a frame (default current_frame) with row numbers and a column ruler."""
        lines = _hx_frame(frame).ascii.split("\n")
        r0, c0 = max(0, int(r0)), max(0, int(c0))
        r1, c1 = min(len(lines), int(r1)), min(len(lines[0]) if lines else 0, int(c1))
        tens = "".join(str((c // 10) % 10) for c in range(c0, c1))
        ones = "".join(str(c % 10) for c in range(c0, c1))
        return "\n".join(["    " + tens, "    " + ones] + ["%3d %s" % (r, lines[r][c0:c1]) for r in range(r0, r1)])

    def objects(color=None, min_px=1, max_px=None, frame=None):
        """Segmentation objects (4-connected, same colour) as dicts with id, color, pixels, inclusive bbox
        [r0, c0, r1, c1], height, width and center; filter by colour letter(s) and size."""
        out = []
        for n in _hx_frame(frame).segmentation.get("nodes", []):
            if color is not None and n["color"] not in color:
                continue
            if n["pixels"] < min_px or (max_px is not None and n["pixels"] > max_px):
                continue
            pts = n.get("boundary") or []
            if not pts:
                continue
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            out.append({"id": n["id"], "color": n["color"], "pixels": n["pixels"], "bbox": [r0, c0, r1, c1],
                        "height": r1 - r0 + 1, "width": c1 - c0 + 1,
                        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0), "hash": n.get("hash")})
        return out

    def at(r, c, frame=None):
        """The 4-connected same-colour component containing cell (r, c): color, pixels, inclusive bbox, center."""
        lines = _hx_frame(frame).ascii.split("\n")
        r, c = int(r), int(c)
        if not (0 <= r < len(lines) and 0 <= c < len(lines[r])):
            return None
        colour, seen, stack = lines[r][c], {(r, c)}, [(r, c)]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if (ny, nx) not in seen and 0 <= ny < len(lines) and 0 <= nx < len(lines[ny]) \
                        and lines[ny][nx] == colour:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
        rows = [p[0] for p in seen]
        cols = [p[1] for p in seen]
        return {"color": colour, "pixels": len(seen), "bbox": [min(rows), min(cols), max(rows), max(cols)],
                "center": ((min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0)}

    def changes(before=None, after=None, limit=60):
        """Cells that differ between two frames (default previous_frame -> current_frame): count, inclusive bbox
        and up to ``limit`` (row, col, old, new) tuples."""
        before = before if before is not None else runtime_globals.get("previous_frame")
        after = _hx_frame(after)
        if before is None or after is None:
            return {"count": 0, "cells": [], "note": "no previous frame"}
        a, b = before.ascii.split("\n"), after.ascii.split("\n")
        cells = [(r, c, a[r][c], b[r][c]) for r in range(min(len(a), len(b)))
                 for c in range(min(len(a[r]), len(b[r]))) if a[r][c] != b[r][c]]
        if not cells:
            return {"count": 0, "cells": []}
        rows = [x[0] for x in cells]
        cols = [x[1] for x in cells]
        return {"count": len(cells), "bbox": [min(rows), min(cols), max(rows), max(cols)], "cells": cells[:limit]}

    def _hx_gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _hx_lattice_guess(limit=40):
        """(pitch, origin) from the displacements of objects moved by recent actions (None, None if inconsistent)."""
        steps, origin = [], None
        for t in list(runtime_globals.get("transitions") or [])[-limit:]:
            result = getattr(t, "result", None) or {}
            moved = ((result.get("transition") or {}).get("objects") or {}).get("moved") or []
            for m in moved:
                if not m.get("from") or not m.get("bbox"):
                    continue
                dr, dc = abs(m["bbox"][0] - m["from"][0]), abs(m["bbox"][1] - m["from"][1])
                steps.extend(v for v in (dr, dc) if v)
                origin = (m["bbox"][0], m["bbox"][1])
        pitch = 0
        for v in steps:
            pitch = _hx_gcd(pitch, v)
        if len(steps) < 2 or not 3 <= pitch <= 16:
            return None, None
        return pitch, origin

    def lattice(pitch=None, origin=None, frame=None, mode="center"):
        """One character per lattice cell of a board drawn on a pitch x pitch grid (default: inferred from how far
        objects moved per action). Returns pitch, origin (the pixel row/col where a cell starts), rows, cols, the
        cell map (``map``: list of strings; ``mode`` "center" samples each cell's centre pixel, "majority" its most
        common colour) and the pixel <-> cell formulas."""
        lines = _hx_frame(frame).ascii.split("\n")
        if pitch is None or origin is None:
            guess_pitch, guess_origin = _hx_lattice_guess()
            pitch = pitch if pitch is not None else guess_pitch
            origin = origin if origin is not None else guess_origin
        if not pitch:
            return {"error": "no consistent movement stride yet: pass pitch=<cell size> and origin=(row, col)"}
        pr, pc = (pitch, pitch) if isinstance(pitch, int) else (int(pitch[0]), int(pitch[1]))
        orr, occ = (int(origin[0]) % pr, int(origin[1]) % pc) if origin is not None else (0, 0)
        rows = []
        for r0 in range(orr, len(lines) - pr + 1, pr):
            row = []
            for c0 in range(occ, len(lines[r0]) - pc + 1, pc):
                if mode == "majority":
                    counts = {}
                    for r in range(r0, r0 + pr):
                        for ch in lines[r][c0:c0 + pc]:
                            counts[ch] = counts.get(ch, 0) + 1
                    row.append(max(counts, key=counts.get))
                else:
                    row.append(lines[r0 + pr // 2][c0 + pc // 2])
            rows.append("".join(row))
        return {"pitch": (pr, pc), "origin": (orr, occ), "rows": len(rows), "cols": len(rows[0]) if rows else 0,
                "map": rows, "cell_of_pixel": "cell (i, j) = ((row - %d) // %d, (col - %d) // %d)" % (orr, pr, occ, pc),
                "pixel_of_cell": "top-left pixel = (%d + i * %d, %d + j * %d)" % (orr, pr, occ, pc)}

    # --- phase 2: closed-loop control (autopilot, explore), model-written world models (sim), notes -------------

    def _hx_unlatch():
        try:
            _DFR["stale"] = None  # dfr_harness latch: these loops expect interior no-ops
        except NameError:
            pass

    def _hx_rows(frame):
        return list(frame.ascii.split("\n"))

    def _hx_action_obj(label):
        text = str(label)
        if text.startswith("MOUSE(") and "row=" in text and "col=" in text:
            try:
                row = int(text.split("row=")[1].split(",")[0].split(")")[0])
                col = int(text.split("col=")[1].split(")")[0])
                return {"action": "MOUSE", "row": row, "col": col}
            except ValueError:
                return text
        return text

    def _hx_akey(item):
        if isinstance(item, dict):
            return "MOUSE(%s,%s)" % (item.get("row"), item.get("col"))
        return str(item).strip().upper()

    def _hx_terminal(result):
        return any(result.get(f) for f in ("level_completed", "game_over", "done", "run_complete"))

    def autopilot(policy, max_actions=60, until=None, expect=None, max_noops=3):
        """Closed loop: repeatedly a = policy(current_frame) (an action, a MOUSE dict, or None to stop) and execute
        it, without a model turn per action. Stops on None, `until(current_frame)` true, `expect(before, after,
        a)` false (a surprise), `max_noops` actions in a row that change nothing inside the board, a cleared level,
        a game over, or `max_actions`. Returns the stop reason and a compact per-action log."""
        log, noops, reason = [], 0, "max_actions"
        for _ in range(int(max_actions)):
            frame = runtime_globals.get("current_frame")
            if until is not None and until(frame):
                reason = "until"
                break
            a = policy(frame)
            if a is None:
                reason = "policy_done"
                break
            _hx_unlatch()
            result = runtime_globals["action"]([a])
            if not isinstance(result, dict) or not result.get("executed"):
                reason = "not_executed: %s" % ((result or {}).get("error") or (result or {}).get("stop_reason"))
                break
            changed = result.get("gameplay_changed", result.get("board_changed"))
            noops = 0 if changed else noops + 1
            log.append((_hx_akey(a), bool(changed)))
            if _hx_terminal(result):
                reason = [f for f in ("level_completed", "game_over", "done", "run_complete") if result.get(f)][0]
                break
            if expect is not None and not expect(frame, runtime_globals.get("current_frame"), a):
                reason = "surprise"
                break
            if noops >= int(max_noops):
                reason = "no_change"
                break
        _HX["loops"] += 1
        _HX["loop_actions"] += len(log)
        return {"stop": reason, "actions": len(log), "log": log[-12:]}

    _HX_GRAPH = {"edges": {}, "cands": {}}

    def _hx_state_key(frame):
        # The interior only: a timer bar in the outer 4-cell ring would make every action a new state.
        import hashlib as _hashlib
        rows = frame.ascii.split("\n")
        inner = "\n".join(r[4:len(r) - 4] for r in rows[4:len(rows) - 4])
        return _hashlib.md5(inner.encode()).hexdigest()[:12]

    def _hx_candidates(frame, clicks):
        valid = list(runtime_globals.get("valid_actions") or [])
        out = [a for a in valid if a not in ("MOUSE", "RESET", "UNDO", "ACTION7")]
        if "MOUSE" in valid and clicks:
            seen = set()
            nodes = sorted(frame.segmentation.get("nodes", []), key=lambda n: n.get("pixels", 0))
            for n in nodes:
                pts = n.get("boundary") or []
                if not pts or n.get("pixels", 0) >= 400:
                    continue
                rows_ = [p[0] for p in pts]
                cols_ = [p[1] for p in pts]
                if max(rows_) < 4 or min(rows_) > 59 or max(cols_) < 4 or min(cols_) > 59:
                    continue  # objects inside the outer ring (HUD / timer bars)
                sig = (n.get("color"), n.get("hash"))
                if sig in seen:
                    continue
                seen.add(sig)
                rows = [p[0] for p in pts]
                cols = [p[1] for p in pts]
                r, c = (min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2
                out.append({"action": "MOUSE", "row": int(r), "col": int(c)})
                if len(out) >= clicks + 8:
                    break
        return out

    def explore(max_actions=30, clicks=8):
        """Systematic exploration without a model turn per action: at each board state try an action (or a click on
        a distinct small object) not yet tried there; when every candidate of the state is tried, walk the recorded
        transition graph to the nearest state with untried candidates. Records every state's effects across calls.
        Stops on a cleared level, a game over or `max_actions`. Returns per-action-kind effects and new states."""
        from collections import deque as _deque

        edges, cands = _HX_GRAPH["edges"], _HX_GRAPH["cands"]
        effects, new_states, reason, done, hits = {}, 0, "max_actions", 0, []
        while done < int(max_actions):
            frame = runtime_globals.get("current_frame")
            s = _hx_state_key(frame)
            if s not in cands:
                cands[s] = _hx_candidates(frame, clicks)
                new_states += 1
            tried = edges.setdefault(s, {})
            untried = [a for a in cands[s] if _hx_akey(a) not in tried]
            if untried:
                # Breadth over action kinds: the kind tried least often so far first (keys before clicks on ties).
                counts = _HX_GRAPH.setdefault("kind_counts", {})
                a = min(untried, key=lambda c: counts.get("MOUSE" if isinstance(c, dict) else str(c), 0))
            else:
                # BFS over recorded edges to a state with untried candidates.
                prev, queue, target = {s: None}, _deque([s]), None
                while queue and target is None:
                    u = queue.popleft()
                    for akey, (a2, v) in edges.get(u, {}).items():
                        if v in prev:
                            continue
                        prev[v] = (u, a2)
                        if any(_hx_akey(c) not in edges.get(v, {}) for c in cands.get(v, [])):
                            target = v
                            break
                        queue.append(v)
                if target is None:
                    reason = "exhausted"
                    break
                step = target
                while prev[step][0] != s:
                    step = prev[step][0]
                a = prev[step][1]
            _hx_unlatch()
            result = runtime_globals["action"]([a])
            if not isinstance(result, dict) or not result.get("executed"):
                reason = "not_executed"
                break
            done += 1
            after = runtime_globals.get("current_frame")
            tried[_hx_akey(a)] = (a, _hx_state_key(after))
            kind = "MOUSE" if isinstance(a, dict) else str(a)
            _HX_GRAPH.setdefault("kind_counts", {})[kind] = _HX_GRAPH["kind_counts"].get(kind, 0) + 1
            changed = bool(result.get("gameplay_changed", result.get("board_changed")))
            row = effects.setdefault(kind, [0, 0])
            row[0] += 1
            row[1] += changed
            if isinstance(a, dict) and changed:
                hits.append("(%s,%s)" % (a.get("row"), a.get("col")))
            if _hx_terminal(result):
                reason = [f for f in ("level_completed", "game_over", "done", "run_complete") if result.get(f)][0]
                break
        _HX["loops"] += 1
        _HX["loop_actions"] += done
        return {"stop": reason, "actions": done, "new_states": new_states, "known_states": len(cands),
                "effects": {k: "%d tried, %d changed the board interior" % (v[0], v[1]) for k, v in effects.items()},
                "clicks_that_changed_something": hits[:12]}

    def _hx_compare(pred, actual, limit=6):
        if not isinstance(pred, (list, tuple)) or len(pred) != len(actual):
            return -1, ["sim returned %s, expected a list of %d strings" % (type(pred).__name__, len(actual))]
        diffs = []
        for r, (p, a) in enumerate(zip(pred, actual)):
            if p != a:
                for c in range(max(len(p), len(a))):
                    pc = p[c] if c < len(p) else "?"
                    ac = a[c] if c < len(a) else "?"
                    if pc != ac:
                        diffs.append((r, c, pc, ac))
        return len(diffs), ["(%d,%d) predicted %s actual %s" % d for d in diffs[:limit]]

    def check_sim(sim, last=None):
        """Replay `sim(rows, action) -> rows` (rows: list of ascii row strings; action: a name or a MOUSE dict) on
        the recorded transitions of the current level (or the `last` n) and report exact predictions and the first
        mismatching cells."""
        ts = [t for t in list(runtime_globals.get("transitions") or []) if str(t.action) != "RESET"]
        level = getattr(runtime_globals.get("current_frame"), "level", None)
        ts = [t for t in ts if getattr(t.before_frame, "level", None) == level == getattr(t.after_frame, "level", None)]
        if last:
            ts = ts[-int(last):]
        exact, first = 0, None
        for t in ts:
            try:
                pred = sim(_hx_rows(t.before_frame), _hx_action_obj(t.action))
                n, cells = _hx_compare(pred, _hx_rows(t.after_frame))
            except Exception as exc:
                n, cells = -1, ["%s: %s" % (type(exc).__name__, str(exc)[:120])]
            if n == 0:
                exact += 1
            elif first is None:
                first = {"step": getattr(t.after_frame, "step", None), "action": str(t.action),
                         "cells_wrong": n, "examples": cells}
        return {"checked": len(ts), "exact": exact, "first_mismatch": first}

    def register_sim(sim):
        """Keep `sim` checked against every new transition; the next user turns report its accuracy."""
        _HX["sim"] = sim
        _HX["sim_report"] = check_sim(sim)
        return _HX["sim_report"]

    def plan(sim, goal, actions=None, max_depth=40, max_nodes=20000, start=None, seconds=20):
        """Breadth-first search with your sim from `start` (default: the current board's rows) for the shortest
        action list reaching `goal(rows) == True`. `actions`: candidate actions (default: the non-MOUSE valid
        actions). Returns {'path': [...] or None, 'nodes': n, 'reason': ...}."""
        import time as _time
        from collections import deque as _deque

        begin = _time.monotonic()
        rows0 = list(start) if start is not None else _hx_rows(runtime_globals.get("current_frame"))
        if actions is None:
            actions = [a for a in (runtime_globals.get("valid_actions") or []) if a not in ("MOUSE", "RESET")]
        key0 = "\n".join(rows0)
        if goal(rows0):
            return {"path": [], "nodes": 1, "reason": "start satisfies goal"}
        prev, queue, nodes = {key0: None}, _deque([(rows0, 0)]), 0
        while queue:
            rows, depth = queue.popleft()
            if depth >= int(max_depth):
                continue
            for a in actions:
                nodes += 1
                if nodes > int(max_nodes) or _time.monotonic() - begin > float(seconds):
                    return {"path": None, "nodes": nodes, "reason": "budget exhausted"}
                try:
                    nxt = list(sim(list(rows), a))
                except Exception as exc:
                    return {"path": None, "nodes": nodes, "reason": "sim error: %s: %s" % (type(exc).__name__, exc)}
                key = "\n".join(nxt)
                if key in prev:
                    continue
                prev[key] = ("\n".join(rows), a)
                if goal(nxt):
                    path, k = [], key
                    while prev[k] is not None:
                        k, a2 = prev[k]
                        path.append(a2)
                    return {"path": path[::-1], "nodes": nodes, "reason": "found"}
                queue.append((nxt, depth + 1))
        return {"path": None, "nodes": nodes, "reason": "no path within max_depth"}

    def note(key, text=None):
        """Your persistent notes: note(key, text) sets or replaces an entry, note(key) deletes it. They are shown in
        every later user turn and survive context drains and levels (at most 12 entries, 400 characters each)."""
        key = str(key).strip()[:40]
        if text is None:
            _HX["notes"].pop(key, None)
        else:
            text = " ".join(str(text).split())
            if len(text) > 400 and _HX_FLAGS.get("memo_limits"):
                print("[note] %r was cut from %d to 400 characters" % (key, len(text)))
            _HX["notes"][key] = text[:400]
            while len(_HX["notes"]) > 12:
                dropped = next(iter(_HX["notes"]))
                _HX["notes"].pop(dropped)
                if _HX_FLAGS.get("memo_limits"):
                    print("[note] 12 entries at most: the oldest, %r, was dropped" % dropped)
        return dict(_HX["notes"])

    def playbook(text=None):
        """Your playbook: playbook(text) replaces it, playbook() returns it. It is what survives when the
        conversation is cut."""
        if text is not None:
            text = str(text).strip()
            if _HX_FLAGS.get("playbook_open"):
                print("[playbook] saved, %d characters" % len(text))
            elif len(text) > 5000:
                print("[playbook] cut from %d to 5,000 characters; write a shorter one" % len(text))
                text = text[:5000]
            _HX["playbook"] = text
            _HX["playbook_writes"] += 1
            _HX["playbook_step"] = getattr(runtime_globals.get("current_frame"), "step", None)
        return _HX["playbook"]

    def won(text=None):
        """won(text): keep, once, what completed the level you just cleared; won() lists the kept wins."""
        if text is None:
            return [dict(k) for k in _HX["kept"] if k["kind"] == "win"]
        frame = runtime_globals.get("current_frame")
        level = int(getattr(frame, "level", 0) or 0) - 1
        if level < 1:
            print("[kept] no level cleared yet")
            return None
        if any(k["kind"] == "win" and k["level"] == level for k in _HX["kept"]):
            print("[kept] level %d is already kept and stays as written" % level)
            return None
        text = " ".join(str(text).split())
        if len(text) > 700:
            print("[kept] cut from %d to 700 characters" % len(text))
            text = text[:700]
        _HX["kept"].append({"kind": "win", "level": level, "text": text, "step": getattr(frame, "step", None)})
        print("[kept] level %d: saved" % level)
        return level

    def _hx_rule_run(fn, ts):
        held, broke = 0, None
        for t in ts:
            r = fn(t)
            if r is None:
                continue
            if r:
                held += 1
            elif broke is None:
                broke = getattr(t.after_frame, "step", None)
        return held, broke

    def keep_rule(text, test, name=None):
        """keep_rule(text, test): keep a mechanic whose test(t) holds on every recorded transition it applies to."""
        if not callable(test):
            raise TypeError("keep_rule(text, test): test is a function of one transition")
        rules = [k for k in _HX["kept"] if k["kind"] == "rule"]
        key = str(name or getattr(test, "__name__", "") or "").strip()[:40]
        if not key or key == "<lambda>":
            key = "rule_%d" % (len(rules) + 1)
        if len([k for k in rules if k["broken"] is None]) >= 12 and not any(k["name"] == key for k in rules):
            print("[kept] 12 kept rules at most; a kept rule leaves only when a later step breaks it")
            return None
        every = list(runtime_globals.get("transitions") or [])
        try:
            held, broke = _hx_rule_run(test, [t for t in every if str(t.action) != "RESET"])
        except Exception as exc:
            print("[kept] not kept: the test raised %s: %s" % (type(exc).__name__, str(exc)[:160]))
            return None
        if broke is not None:
            print("[kept] not kept: the test fails at step %s (it held on %d other steps)" % (broke, held))
            return None
        if held < 5:
            print("[kept] not kept: the test applies to %d recorded steps; it needs 5" % held)
            return None
        entry = {"kind": "rule", "name": key, "text": " ".join(str(text).split())[:300], "held": held, "broken": None,
                 "step": getattr(runtime_globals.get("current_frame"), "step", None), "seen": len(every)}
        _HX["kept"] = [k for k in _HX["kept"] if not (k["kind"] == "rule" and k["name"] == key)] + [entry]
        _HX["kept_fns"][key] = test
        print("[kept] rule %r: held on all %d steps it applies to; it is checked again on every later step" % (key, held))
        return key

    def _hx_recheck():
        """Kept rules against the transitions recorded since their last check (at most about a second per call)."""
        import time as _time
        every = list(runtime_globals.get("transitions") or [])
        started = _time.time()
        for entry in _HX["kept"]:
            if entry["kind"] != "rule" or entry["broken"] is not None or entry["seen"] >= len(every):
                continue
            if _time.time() - started > 1.0:
                break
            fn = _HX["kept_fns"].get(entry["name"])
            new = [t for t in every[entry["seen"]:] if str(t.action) != "RESET"]
            entry["seen"] = len(every)
            if fn is None:
                continue
            try:
                held, broke = _hx_rule_run(fn, new)
            except Exception:
                held, broke = 0, getattr(runtime_globals.get("current_frame"), "step", None)
            entry["held"] += held
            entry["broken"] = broke

    _hx_refresh_inner = _refresh_state

    def _refresh_state(state_payload):
        _hx_refresh_inner(state_payload)
        if _HX_FLAGS["toolbox"]:
            runtime_globals["register_tool"] = register_tool
            runtime_globals["register_observer"] = register_observer
            runtime_globals["unregister"] = unregister
        if _HX_FLAGS["probe"]:
            runtime_globals["probe"] = probe
        if _HX_FLAGS["kit"]:
            runtime_globals.update(crop=crop, objects=objects, at=at, changes=changes)
        if _HX_FLAGS["lattice"]:
            runtime_globals["lattice"] = lattice
        if _HX_FLAGS["autopilot"]:
            runtime_globals.update(autopilot=autopilot, explore=explore)
        if _HX_FLAGS["worldmodel"]:
            runtime_globals.update(check_sim=check_sim, register_sim=register_sim, plan=plan)
        if _HX_FLAGS["memo"] and not _HX_FLAGS.get("playbook_open"):
            runtime_globals["note"] = note
        if _HX_FLAGS.get("memo_style"):
            runtime_globals["playbook"] = playbook
        if _HX_FLAGS.get("keep"):
            runtime_globals["won"] = won
        if _HX_FLAGS.get("keep") == 2:
            runtime_globals["keep_rule"] = keep_rule

    def _hx_take():
        if _HX["sim"] is not None:
            try:
                _HX["sim_report"] = check_sim(_HX["sim"])
            except Exception as exc:  # a broken sim must never break the call
                _HX["sim_report"] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
        if _HX_FLAGS.get("keep") == 2:
            try:
                _hx_recheck()
            except Exception:  # a kept rule's test must never break the call
                pass
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
HX_RUNTIME_NAMES = '"register_tool", "register_observer", "unregister"'


def with_toolbox(bootstrap, block):
    """``bootstrap`` with the toolbox child code (after any dfr_harness code: its action wrapper wraps theirs) and the
    ``hx`` field in the call's final and error messages."""
    if "_HX = {" in bootstrap:
        raise ValueError("hx_harness: the toolbox code is already installed")
    if bootstrap.count(ANCHOR) != 1:
        raise ValueError("hx_harness: pinned Duck sandbox loop changed")
    flags = {key: bool(block[key]) for key in ("toolbox", "probe", "kit", "lattice", "autopilot", "worldmodel", "memo",
                                               "memo_limits", "memo_style")}
    flags["playbook_open"] = block["memo_style"] == 2
    flags["keep"] = block.get("keep_rules", 0) if block["memo"] and block["memo_style"] == 2 else 0
    child = CHILD_BLOCK.replace("__HX_OBS_CHARS__", str(int(block["observer_chars"]))).replace(
        "__HX_FLAGS__", repr(flags))
    text = bootstrap.replace(ANCHOR, child.lstrip("\n") + ANCHOR, 1)
    if text.count(RESULT_FIELD) != 2:
        raise ValueError("hx_harness: pinned Duck sandbox result messages changed")
    text = text.replace(RESULT_FIELD, RESULT_FIELD + ' "hx": _hx_take(),')
    if text.count(RUNTIME_NAMES) == 1:  # the fixes loop's inventory skips runtime names
        text = text.replace(RUNTIME_NAMES, RUNTIME_NAMES + ", " + HX_RUNTIME_NAMES
                            + ', "probe", "crop", "objects", "at", "changes", "lattice", "autopilot", "explore", '
                            '"check_sim", "register_sim", "plan", "note", "playbook"'
                            # won / keep_rule only exist with keep_rules: else a model's own `won` stays in the inventory
                            # as on v8.5p (harness parity)
                            + (', "won"' if flags["keep"] else "") + (', "keep_rule"' if flags["keep"] == 2 else ""), 1)
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
        self.deaths_noticed = 0
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

    def skip_to(self, history):
        """Harness-played probes (autoexplore / stuckexplore) are not the model's attempts: consume them without
        counting actions, no-op streaks, deaths or board visits."""
        self.seen = len(history)
        self.no_change = 0

    def due(self):
        b = self.block
        if self.last_notice is not None and self.requests - self.last_notice < b["reflect_min_gap"]:
            return None
        if b["reflect_skip_death"]:
            if self.died_now:
                return None
        elif self.deaths >= b["reflect_deaths"] and self.deaths > self.deaths_noticed:
            return "deaths"
        if self.no_change >= b["reflect_no_change"] and self.no_change >= self.streak_noticed + b["reflect_no_change"]:
            return "no_change"
        visits = self.boards.get(self.current, 0)
        if visits >= b["reflect_revisits"] and visits > self.revisit_noticed:
            return "revisits"
        if b["reflect_style"] == 2:
            stalled = self.requests - max(self.change_request, self.last_notice or 0)
            return "stall" if stalled >= b["reflect_stall"] else None
        if self.requests >= b["reflect_requests"] and (
                self.last_notice is None or self.requests - self.last_notice >= b["reflect_every"]):
            return "requests"
        return None

    def notice(self, level_number):
        reason = self.due()
        if reason is None:
            return None
        self.last_notice = self.requests
        self.last_reason = reason
        self.notices += 1
        self.fired[reason] += 1
        if reason == "deaths":
            self.deaths_noticed = self.deaths
        if reason == "no_change":
            self.streak_noticed = self.no_change
        if reason == "revisits":
            self.revisit_noticed = self.boards.get(self.current, 0)
        why = {"deaths": f"{self.deaths} deaths on this level",
               "no_change": f"your last {self.no_change} actions changed nothing inside the board",
               "revisits": f"this exact board has now been seen {self.boards.get(self.current, 0)} times on this level",
               "requests": f"{self.requests} model requests on this level without clearing it",
               "stall": f"{self.requests - self.change_request} model requests since an action last changed the board "
                        "interior"}[reason]
        table = ", ".join(f"{kind} {tried}x ({changed} changed the board{f', {died} deaths' if died else ''})"
                          for kind, (tried, changed, died) in sorted(self.effects.items())) or "no actions yet"
        if self.block["reflect_style"] == 2:
            return (
                f"STEP BACK (harness check: {why}). Level {level_number}: {self.requests} requests, "
                f"{self.tokens:,} generated tokens, {self.actions} actions ({self.changed} changed the interior), "
                f"{self.deaths} deaths, {len(self.boards)} distinct boards; effects: {table}. The current approach is "
                f"not working{'' if self.block['no_think'] else STEP_BACK_DELIBERATION}. In this reply: name the belief you "
                "are least sure "
                "of and the cheapest experiment that would refute it; if your tools or your view of the board are the "
                "bottleneck, ask how you can modify your own code or your interface to the environment: write the "
                "helper or observer you lack and register it (`register_tool(fn)`, `register_observer(fn)`); then run "
                "the experiment now (several actions in one call are fine) instead of planning further."
            )
        return (
            f"STEP BACK (harness check: {why}). Level {level_number} so far: {self.requests} requests, "
            f"{self.tokens:,} generated tokens, {self.actions} actions ({self.changed} changed the board interior), "
            f"{self.deaths} deaths, {len(self.boards)} distinct boards. Action effects on this level: {table}.\n"
            "Before acting, answer briefly in your reply text:\n"
            "1. What did each kind of attempt teach you, and which belief about the goal or the controls is most "
            "likely wrong?\n"
            "2. Name two goal hypotheses you have not tested yet and the cheapest action sequence that tells them "
            "apart.\n"
            "3. How can you modify your own code or your interfaces to the environment to improve? If your tools "
            "make you re-derive the same facts every turn, or you cannot see what matters, expand the interface now: "
            "write a reusable python helper (an action-effect table from `transitions`, a locator for what you "
            "control, a simulator of the rules you believe checked against the recorded transitions, a planner/BFS "
            "over it, or a probe that tries each untried action or object once) and register it with "
            "`register_tool(fn)`, or `register_observer(fn)` so that `fn(before_frame, after_frame)` reports what "
            "matters after every action. Then use it to choose your next actions."
        )

    def summary(self):
        return {"notices": self.notices, **dict(self.fired)}


# --- perceive: the per-turn harness digest ----------------------------------------------------------------------------

COLOR_CHARS = "WwgGcBMPRbSYOrNp"  # inference.utils.grid_utils.ARC_COLOR_CHARS
MOUSE_RE = re.compile(r"^MOUSE\(row=(-?\d+), col=(-?\d+)\)$")


def _color(value):
    try:
        return COLOR_CHARS[int(value)]
    except (IndexError, ValueError, TypeError):
        return str(value)


def _span(bbox):
    r0, c0, r1, c1 = bbox
    return f"rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1}"


def _obj(entry):
    r0, c0, r1, c1 = entry["bbox"]
    return f"{_color(entry['color'])} {entry['pixels']}px at ({r0},{c0})"


def describe_effect(label, result, edge_only):
    """One line for one executed action's receipt."""
    if result.get("game_over"):
        tail = " -> GAME OVER (level restarted)"
    elif result.get("level_completed"):
        tail = " -> LEVEL CLEARED"
    else:
        tail = ""
    transition = result.get("transition") or {}
    cells = transition.get("changed_cells")
    animation = result.get("animation") or {}
    anim = ""
    if animation.get("frames"):
        anim = f"; animated over {animation['frames']} frames"
        if animation.get("transient_pixels"):
            anim += f" with {animation['transient_pixels']} cells that changed and changed back"
    if not cells:
        return f"{label}: no change" + anim + tail
    if edge_only:
        return f"{label}: only the outer edge ring changed ({cells} cells; HUD/timer?)" + anim + tail
    parts = [f"{cells} cells changed in {_span(transition['change_bbox'])}"] if transition.get("change_bbox") else [
        f"{cells} cells changed"]
    objects = transition.get("objects") or {}
    for key, verb in (("moved", "moved"), ("appeared", "appeared"), ("disappeared", "disappeared")):
        listed = objects.get(key) or []
        if listed:
            shown = []
            for entry in listed[:2]:
                if key == "moved" and entry.get("from"):
                    shown.append(f"{_color(entry['color'])} {entry['pixels']}px ({entry['from'][0]},{entry['from'][1]})"
                                 f"->({entry['bbox'][0]},{entry['bbox'][1]})")
                else:
                    shown.append(_obj(entry))
            more = (objects.get("counts") or {}).get(key, len(listed)) - len(shown)
            parts.append(f"{verb}: " + ", ".join(shown) + (f" +{more}" if more > 0 else ""))
    return f"{label}: " + "; ".join(parts) + anim + tail


LARGE = 400  # cells: a component this big is a region (background, wall mass), not an object


def scene_summary(grid, segmentation, limit):
    """A level's opening scene from the host segmentation: large regions, object groups (colour + shape, with
    positions), and thin strips flush with the border (likely HUD / timer bars)."""
    nodes = (segmentation or {}).get("nodes") or []
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    regions = sorted((n for n in nodes if n["pixels"] >= LARGE), key=lambda n: -n["pixels"])
    strips, groups = [], {}
    for n in nodes:
        if n["pixels"] >= LARGE:
            continue
        r0, c0, r1, c1 = n["bbox"]
        thin = min(r1 - r0, c1 - c0) <= 2 and max(r1 - r0, c1 - c0) >= 8
        if thin and (r0 <= 1 or c0 <= 1 or r1 >= rows - 1 or c1 >= cols - 1):
            strips.append(n)
            continue
        groups.setdefault((n["color"], n["hash"]), []).append(n)
    parts = []
    if regions:
        parts.append("large regions: " + ", ".join(f"{_color(n['color'])} {n['pixels']} cells" for n in regions[:5]))
    ordered = sorted(groups.values(), key=lambda g: (len(g), -g[0]["pixels"]))
    listed = []
    for group in ordered[:limit]:
        n = group[0]
        h = n["bbox"][2] - n["bbox"][0]
        w = n["bbox"][3] - n["bbox"][1]
        where = ", ".join(f"({m['bbox'][0]},{m['bbox'][1]})" for m in group[:4]) + (" ..." if len(group) > 4 else "")
        listed.append(f"{len(group)}x {_color(n['color'])} {n['pixels']}px {h}x{w} at {where}")
    if listed:
        parts.append("objects (count, colour, cells, height x width, top-left): " + "; ".join(listed)
                     + (f"; +{len(ordered) - limit} more groups" if len(ordered) > limit else ""))
    if strips:
        parts.append("thin strips at the border (likely HUD/timer bars): " + ", ".join(
            f"{_color(n['color'])} {n['pixels']}px {_span(n['bbox'])}" for n in strips[:4]))
    return ("Scene (harness segmentation, 4-connected same-colour components): " + "; ".join(parts) + ".") if parts else None


class Perceiver:
    """The digest's state: per-level ledger, the level's start board, the previous level's start objects."""

    def __init__(self, block, objects):
        self.block = block
        self.objects = objects  # grid (tuple rows) -> segmentation dict with nodes (host TransitionCache)
        self.seen = 0
        self.level = None
        self.start_hash = None
        self.start_kinds = None
        self.ledger = {}
        self.clicks = {}
        self.level_actions = 0
        self.new_level_note = None
        self.scene_note = None
        self.reset_noted = False
        self.digests = 0
        self.chars = 0

    def kinds(self, grid):
        nodes = (self.objects(grid) or {}).get("nodes") or []
        return Counter((n["color"], n["hash"]) for n in nodes if n.get("pixels", 0) < 1500)

    def start_level(self, levels, grid):
        kinds = self.kinds(grid)
        note = None
        if self.start_kinds is not None:
            new = [k for k in kinds if k not in self.start_kinds]
            gone = [k for k in self.start_kinds if k not in kinds]
            nodes = {(n["color"], n["hash"]): n for n in (self.objects(grid) or {}).get("nodes") or []}
            if new or gone:
                shown = [f"{_color(c)} {nodes[(c, h)]['pixels']}px at ({nodes[(c, h)]['bbox'][0]},"
                         f"{nodes[(c, h)]['bbox'][1]})" for c, h in new[:6]]
                note = ("New level: objects whose colour and shape were not on the previous level's start: "
                        + (", ".join(shown) + (f" +{len(new) - 6}" if len(new) > 6 else "") if new else "none")
                        + f"; {len(gone)} earlier object kinds are gone. New elements often carry the new mechanic: "
                        "test them early.")
        self.level, self.start_hash, self.start_kinds = levels, _hash(grid), kinds
        self.ledger, self.clicks, self.level_actions, self.reset_noted = {}, {}, 0, False
        self.new_level_note = note
        try:
            self.scene_note = scene_summary(grid, self.objects(grid), self.block["scene_objects"])
        except Exception:  # noqa: BLE001 - a summary must never break the turn
            self.scene_note = None

    def turn(self, history, results, levels, available):
        """The digest lines for the new user turn."""
        if not history:
            return []
        if self.level is None:
            self.start_level(levels, history[-1].frame.grid)
            self.seen = len(history)
            return [self.scene_note] if self.scene_note else []
        effects = []
        cleared = False
        previous = history[self.seen - 1].frame.grid if 0 < self.seen <= len(history) else None
        for index in range(self.seen, len(history)):
            entry = history[index]
            grid = entry.frame.grid
            if entry.action and entry.action != "RESET":
                result = results.get(entry.frame.step) or {}
                interior = _interior_changed(previous, grid) if previous is not None else True
                cells = (result.get("transition") or {}).get("changed_cells") or 0
                effects.append(describe_effect(entry.action, result, edge_only=bool(cells) and not interior))
                kind = "MOUSE" if ACTION_KIND.match(entry.action) else entry.action
                row = self.ledger.setdefault(kind, [0, 0, 0])
                row[0] += 1
                row[1] += interior
                row[2] += bool(result.get("game_over"))
                match = MOUSE_RE.match(entry.action)
                if match and previous is not None:
                    r, c = int(match.group(1)), int(match.group(2))
                    if 0 <= r < len(previous) and 0 <= c < len(previous[r]):
                        target = self.clicks.setdefault(_color(previous[r][c]), [0, 0])
                        target[0] += 1
                        target[1] += interior
                self.level_actions += 1
                if result.get("level_completed"):
                    cleared = True
            previous = grid
        self.seen = len(history)
        lines = []
        if effects:
            limit = self.block["digest_actions"]
            shown = effects if len(effects) <= limit else effects[:limit // 2] + [
                f"... {len(effects) - limit} more ..."] + effects[-(limit - limit // 2):]
            lines.append("Harness digest - effects since your previous turn: " + " | ".join(shown))
        if cleared or levels != self.level:
            self.start_level(levels, history[-1].frame.grid)
            if self.new_level_note:
                lines.append(self.new_level_note)
            if self.scene_note:
                lines.append(self.scene_note)
            return lines
        if self.ledger:
            parts = []
            for kind, (tried, changed, died) in sorted(self.ledger.items()):
                text = f"{kind} {tried}x ({changed} changed the interior{f', {died} deaths' if died else ''})"
                if kind == "MOUSE" and self.clicks:
                    text += " [clicked on: " + ", ".join(
                        f"{color} {n}x/{ch} changed" for color, (n, ch) in sorted(self.clicks.items())) + "]"
                parts.append(text)
            untried = [a for a in available if a not in self.ledger and a != "RESET"]
            lines.append("Ledger for this level: " + "; ".join(parts)
                         + (f". Not yet tried on this level: {', '.join(untried)}." if untried else "."))
        current = _hash(history[-1].frame.grid)
        if current == self.start_hash and self.level_actions and not self.reset_noted:
            self.reset_noted = True
            lines.append("Note: the board is identical to this level's starting board again (after "
                         f"{self.level_actions} actions on the level): a restart or lost life may have happened.")
        elif current != self.start_hash:
            self.reset_noted = False
        return lines

    def summary(self):
        return {"digests": self.digests, "chars": self.chars}


def opener_line(level_number, available, probe, explore=False):
    """The first user turn of each level (opener): test every action type once before any long analysis."""
    if explore:
        return (f"Level opener (first turn on level {level_number}): unless the goal is already clear, start with ONE "
                "python call that runs `explore(max_actions=12)` and reads its effects table (it tries each action "
                "and a click on each distinct object, state by state); only then analyse. A dozen test actions are "
                "cheap; long analysis before any evidence is not.")
    keys = [a for a in available if a not in ("MOUSE", "RESET")]
    parts = []
    if keys:
        parts.append("each of " + ", ".join(keys) + " once")
    if "MOUSE" in available:
        parts.append("MOUSE on the two or three most distinctive objects")
    tool = "`probe([...])`" if probe else "a loop of single `action([...])` calls"
    return (f"Level opener (first turn on level {level_number}): unless the goal is already clear, start with ONE python "
            f"call that glances at the board and then tests {' and '.join(parts) or 'the available actions'} with "
            f"{tool}, reading what each did; only then analyse. A few test actions are cheap; long analysis before any "
            "evidence is not.")


def economy_line(level_actions, level_number, spend=None):
    """The per-turn action count on the level (economy); ``spend`` = (requests, tokens, seconds) since the previous
    turn (economy_spend)."""
    text = (f"Action economy: {level_actions} actions so far on level {level_number} (the first ~20 cost almost "
            "nothing in score; experiments are cheaper than long deliberation).")
    if spend is not None:
        requests, tokens, seconds = spend
        text += (f" Since your previous turn: {requests} model request{'s' if requests != 1 else ''}, {tokens:,} "
                 f"generated tokens, {seconds:.0f} s of play time.")
    return text


# --- lattice (host side): a one-time hint when the movement stride is consistent ------------------------------------

def stride_of(receipts):
    """(pitch, origin) from the moved-object displacements in ``receipts`` (None when < 2 steps or not 3..16)."""
    import math

    steps, origin = [], None
    for result in receipts:
        moved = ((result.get("transition") or {}).get("objects") or {}).get("moved") or []
        for m in moved:
            if not m.get("from") or not m.get("bbox"):
                continue
            dr, dc = abs(m["bbox"][0] - m["from"][0]), abs(m["bbox"][1] - m["from"][1])
            steps.extend(v for v in (dr, dc) if v)
            origin = (m["bbox"][0], m["bbox"][1])
    pitch = 0
    for v in steps:
        pitch = math.gcd(pitch, v)
    if len(steps) < 2 or not 3 <= pitch <= 16:
        return None
    return pitch, origin


def lattice_hint(pitch, origin):
    r0, c0 = origin
    return (f"Harness: the objects that moved so far moved by multiples of {pitch} pixels per action, so this board is "
            f"probably a lattice of {pitch} x {pitch}-pixel cells (one cell starts at row {r0 % pitch}, col "
            f"{c0 % pitch}). `lattice()` returns the cell map at that pitch (one letter per cell) and the pixel <-> "
            "cell formulas; plan on cells.")


# --- coverage: actions that did nothing, by how many distinct boards they were tried on ------------------------------

class Coverage:
    """Per level: for each action kind, the distinct boards it was tried on and whether it ever changed the interior.
    The line lists the no-effect kinds tried on few boards while the board has changed since (cd82: UP / DOWN tried
    only at one ring position)."""

    def __init__(self):
        self.seen = 0
        self.level = None
        self.reset()

    def reset(self):
        self.tried = {}   # kind -> set of board hashes it was tried on
        self.worked = set()
        self.last_tried = {}  # kind -> board hash of its last try
        self.noted = None

    def update(self, history, levels):
        if levels != self.level:
            self.level = levels
            self.reset()
            self.seen = len(history)
            return
        previous = history[self.seen - 1].frame.grid if 0 < self.seen <= len(history) else None
        for index in range(self.seen, len(history)):
            entry = history[index]
            grid = entry.frame.grid
            if entry.action and entry.action != "RESET" and previous is not None:
                kind = "MOUSE" if ACTION_KIND.match(entry.action) else entry.action
                key = _hash(previous)
                self.tried.setdefault(kind, set()).add(key)
                self.last_tried[kind] = key
                if _interior_changed(previous, grid):
                    self.worked.add(kind)
            previous = grid
        self.seen = len(history)

    def line(self, current_grid, limit=3):
        here = _hash(current_grid)
        stale = [(kind, len(boards)) for kind, boards in sorted(self.tried.items())
                 if kind not in self.worked and kind != "MOUSE" and len(boards) <= limit and here not in boards]
        if not stale:
            return None
        text = ("Coverage: " + ", ".join(f"{kind} did nothing on the {n} board{'s' if n != 1 else ''} where it was "
                                         "tried" for kind, n in stale)
                + "; the board has changed since, and a no-op in one position is not evidence that the action does "
                  "nothing elsewhere - retest it here when it could matter.")
        if text == self.noted:
            return None
        self.noted = text
        return text


def detect_lattice(grid, pitches=range(3, 13), min_rate=0.95, min_blocks=6):
    """(pitch, origin, rate) of a block-drawn board: the pitch and offset whose blocks are most often a single colour,
    counted over the blocks that are not the dominant (background) colour; None when no pitch reaches ``min_rate``
    on at least ``min_blocks`` such blocks. Exact k-x upscales (e.g. an 11 x 11 board drawn at 5x) score 1.0 at their
    pitch and offset; free-form scenes do not."""
    import numpy as np

    g = np.asarray(grid)
    if g.ndim != 2:
        return None
    values, counts = np.unique(g, return_counts=True)
    background = values[np.argmax(counts)]
    best = None
    for p in pitches:
        for r0 in range(p):
            for c0 in range(p):
                rows, cols = (g.shape[0] - r0) // p, (g.shape[1] - c0) // p
                if rows < 2 or cols < 2:
                    continue
                blocks = g[r0:r0 + rows * p, c0:c0 + cols * p].reshape(rows, p, cols, p).swapaxes(1, 2)
                flat = blocks.reshape(rows, cols, p * p)
                uniform = (flat == flat[:, :, :1]).all(axis=2)
                foreground = ~(flat == background).all(axis=2)
                n = int(foreground.sum())
                if n < min_blocks:
                    continue
                rate = float((uniform & foreground).sum()) / n
                key = (rate, p)
                if rate >= min_rate and (best is None or key > (best[2], best[0])):
                    best = (p, (r0, c0), rate)
    return best


def lattice_scene_hint(found):
    p, (r0, c0), rate = found
    return (f"Harness: this board looks drawn on a lattice of {p} x {p}-pixel cells starting at row {r0}, col {c0} "
            f"({rate:.0%} of the non-background cells are single-coloured blocks). `lattice(pitch={p}, "
            f"origin=({r0}, {c0}))` returns the cell map (one letter per cell); plan on cells.")


# --- merges: same-colour objects that touched (one component) or came apart ------------------------------------------

def merge_notes(before_nodes, after_nodes, label, limit=2):
    """Lines for colours whose 4-connected component count changed while their total cell count stayed the same
    (objects moved into contact or apart: segmentation shows touching same-colour objects as ONE component)."""
    def by_colour(nodes):
        out = {}
        for n in nodes:
            if n.get("pixels", 0) >= LARGE:
                continue
            count, cells = out.get(n["color"], (0, 0))
            out[n["color"]] = (count + 1, cells + n["pixels"])
        return out

    before, after = by_colour(before_nodes), by_colour(after_nodes)
    notes = []
    for colour in sorted(set(before) & set(after)):
        (nb, cb), (na, ca) = before[colour], after[colour]
        if cb != ca or nb == na:
            continue
        if na < nb:
            notes.append(f"after {label}, {nb - na + 1} separate {_color(colour)} objects now touch and show as one "
                         f"component ({ca} cells): they are still distinct objects; contact is not necessarily "
                         "overlap or the goal state")
        else:
            notes.append(f"after {label}, a {_color(colour)} component split into {na - nb + 1} parts ({ca} cells): "
                         "objects that touched have separated")
        if len(notes) >= limit:
            break
    return notes


# --- phase 2 host lines: sim accuracy, notes ---------------------------------------------------------------------------

def sim_line(report):
    if not report:
        return None
    if report.get("error"):
        return f"Your registered sim failed: {report['error']}."
    text = f"Your registered sim predicts {report['exact']}/{report['checked']} transitions of this level exactly"
    first = report.get("first_mismatch")
    if first:
        text += (f"; first wrong: step {first['step']} {first['action']}: {first['cells_wrong']} cells, e.g. "
                 + "; ".join(first.get("examples") or [])[:300])
    return text + "."


def memo_turn(line, marker, shown):
    """``(line or None, shown)`` for memo_once: the notes line goes out when its text or ``marker`` (drains,
    compactions, level) differs from what was last shown; otherwise the history already holds it."""
    if not line:
        return None, shown
    key = (line, marker)
    return (None, shown) if key == shown else (line, key)


def notes_line(notes):
    if not notes:
        return None
    return ("Your notes (persist across turns, levels and context drains; update with note(key, text)): "
            + " | ".join(f"{k}: {v}" for k, v in notes.items()))


LEVEL_NOTE = ("Level cleared. Before acting on the new level, record in note(key, text) what you learned that will "
              "matter later (the mechanics, the goal, what worked, what to avoid).")


# --- autoexplore: harness-played probes at the start of each level (no model tokens) ---------------------------------

def explore_candidates(available, nodes, clicks, rows=64, cols=64):
    """The level-start probe list: every key action once (not RESET / UNDO), then a click on each distinct small
    interior object (smallest first; HUD strips in the outer ring excluded)."""
    keys = [a for a in available if a not in ("MOUSE", "RESET", "UNDO")]
    out = [{"action": a} for a in keys]
    if "MOUSE" in available and clicks:
        seen = set()
        for n in sorted(nodes, key=lambda n: n.get("pixels", 0)):
            if n.get("pixels", 0) >= LARGE or not n.get("bbox"):
                continue
            r0, c0, r1, c1 = n["bbox"]
            if r1 <= BORDER or r0 >= rows - BORDER or c1 <= BORDER or c0 >= cols - BORDER:
                continue
            sig = (n["color"], n["hash"])
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"action": "MOUSE", "row": (r0 + r1 - 1) // 2, "col": (c0 + c1 - 1) // 2})
            if len(out) >= len(keys) + clicks:
                break
    return out


def autoexplore_note(level_number, lines, stop):
    head = (f"Harness exploration of level {level_number} before your first turn on it (each action, then a click on "
            "each distinct small object, once, in this order, 0 model tokens): ")
    tail = f" ({stop})" if stop else ""
    return head + " | ".join(lines) + tail + ". The board you see is the result; use these effects as evidence."


def interior_key(grid):
    rows = len(grid)
    return _hash(tuple(tuple(r[BORDER:len(r) - BORDER]) for r in grid[BORDER:rows - BORDER]))


def tried_here(history, grid):
    """Action labels already executed on a board with this interior (any time in the history)."""
    here = interior_key(grid)
    out = set()
    for prev, entry in zip(history, history[1:]):
        if entry.action and entry.action != "RESET" and interior_key(prev.frame.grid) == here:
            out.add(entry.action)
    return out


def label_of(item):
    if item.get("action") == "MOUSE":
        return f"MOUSE(row={item['row']}, col={item['col']})"
    return item["action"]


def stuck_note(level_number, why, lines, stop):
    tail = f" ({stop})" if stop else ""
    return (f"Harness, level {level_number} stalled ({why}): it played actions and clicks not yet tried on this exact "
            "board (0 model tokens): " + (" | ".join(lines) if lines else "nothing untried was left") + tail
            + ". Use these effects as new evidence; a mechanic you have not seen yet is often behind them.")
