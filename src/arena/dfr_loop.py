"""hero-v2 ``dfr_loop`` (v5): the agent loop matched more closely to dfranzen's harness.

Ported from dfranzen/arc-agi-3-milestone-2-solution (Apache-2.0; his ``setup_env``: LOCAL_ANALYZER_YIELD_TOKENS=2048,
ARC3_MEMORY_SECTIONS=off, ARC3_ACTION_ECHO=1, the default "full" resume opener). One opt-in study key ``dfr_loop``
(None on every earlier build, so their prompts and payloads stay byte for byte); true switches every flag on, a dict
selects some (``FLAGS``; ``yield_tokens`` is a positive integer or 0/False):

- ``prompt_fix``: the system line that said earlier turns stay "until a context compaction replaces them with your own
  summary" describes drain_half instead (needs drain_half).
- ``yield_tokens``: an analyzer invocation that generated this many tokens without executing an action ends at his
  three check points (before a request, after a reply without a tool call, after a tool call that executed nothing);
  the next opener is the full opener with the board image, its outcome block replaced by his stale lines ("Your last
  python snippet executed NO actions ..." or "Your previous exchange this turn produced no tool call ..."). A
  resumption is not a new turn for the stuck notice or the refusal count.
- ``no_world_model``: his ARC3_MEMORY_SECTIONS=off: no carried block, no "MUST give the revised world model" lines,
  no cross-level hypothesis carry, his world-model-free system line and new-level prompt.
- ``action_feedback``: his receipts: the per-call echo line printed by ``action()``, ``action_trace`` for every batch,
  game over and refusal, the game-over opener's per-action trace, ``last_action_call_result`` (alias of
  ``last_action_result``), per-action ``.result`` with ``automatic``, ``run_elapsed_seconds`` /
  ``time_remaining_seconds``, no ``reward``, and his flag-semantics line. Kept from ours (no conflict): ``transition``,
  ``tried_no_effect_here``, ``skipped_repeats``, the no-change-repeat and RESET refusals and their stop reasons.
- ``inventory_animation`` (our bug fix): the drain note's inventory runs ``len``/``repr`` on user variables, and one
  bound to a lazy animation view (``tl = last_animation_timeline``) asked the host for frames; the inventory accepted
  only its own reply and discarded the interpreter (85 of 86 v3 interpreter resets, 15 of 15 in v4). The inventory
  now answers those requests.
"""

import re

from . import dfr_harness as dfr

FLAGS = ("prompt_fix", "yield_tokens", "no_world_model", "action_feedback", "inventory_animation")
YIELD_TOKENS = 2048  # his LOCAL_ANALYZER_YIELD_TOKENS
BOOLEANS = ("prompt_fix", "no_world_model", "action_feedback", "inventory_animation")


def settings(value):
    """The study block (None/False: off; True: every flag, 2,048 yield tokens; a dict over ``FLAGS``), or None."""
    if value is None or value is False:
        return None
    if value is True:
        value = {flag: True for flag in FLAGS}
    if not isinstance(value, dict):
        raise ValueError("dfr_loop is true/false or a dict over " + ", ".join(FLAGS))
    unknown = sorted(set(value) - set(FLAGS))
    if unknown:
        raise ValueError(f"Unknown dfr_loop flags {unknown}; known: {list(FLAGS)}")
    bad = sorted(k for k in BOOLEANS if k in value and type(value[k]) is not bool)
    if bad:
        raise ValueError(f"dfr_loop flags {bad} are booleans")
    tokens = value.get("yield_tokens", False)
    if tokens is True:
        tokens = YIELD_TOKENS
    if tokens is False or tokens is None:
        tokens = 0
    if type(tokens) is not int or tokens < 0:
        raise ValueError("dfr_loop yield_tokens is true/false or a non-negative integer")
    block = {"prompt_fix": bool(value.get("prompt_fix", False)), "yield_tokens": tokens,
             "no_world_model": bool(value.get("no_world_model", False)),
             "action_feedback": bool(value.get("action_feedback", False)),
             "inventory_animation": bool(value.get("inventory_animation", False))}
    return block if any(block.values()) else None


def describe(block):
    return {**block, "source": "dfranzen/arc-agi-3-milestone-2-solution (Apache-2.0)", "module": "arena.dfr_loop"}


# --- prompt_fix ------------------------------------------------------------------------------------------------------

COMPACTION_LINE = (
    "- Each user turn includes an image of the board as it was before that turn's actions (labelled with its step). "
    "Earlier turns, their images and your earlier replies stay in the context verbatim until a context compaction "
    "replaces them with your own summary.\n"
)
PUBLIC_IMAGE_LINE = (
    "- The newest user turn includes an image of the board as it was before this turn's actions (labelled with its "
    "step); earlier turns keep only a note that their image was omitted.\n"
)
FRACTION_LINE = (
    "- Each user turn includes an image of the board as it was before that turn's actions (labelled with its step). "
    "When context fills, the oldest 90% is replaced with your summary; the newest 10% stays verbatim, rounded to "
    "complete observation/tool exchanges. The persistent Python interpreter and its variables survive. "
    "The degenerate-loop escape hatch can clear the conversation while preserving Python state.\n"
)
DRAIN_LINE = (
    "- Each user turn includes an image of the board as it was before that turn's actions (labelled with its step). "
    "Earlier turns, their images and your earlier replies stay in the context verbatim until the context fills; then "
    "the oldest turns are dropped, a note marks the drop, and the extra images (diff, death frame) of the earlier "
    "turns that remain become a placeholder.\n"
)


# --- yield_tokens: his stale opener (``_stale_summary_lines``) ---------------------------------------------------------

def stale_lines(summary):
    """His rendering of a step summary carried over from an exchange that executed nothing."""
    if summary.get("stale_reason") == "no_tool_call":
        lines = ["Your previous exchange this turn produced no tool call and executed no actions."]
    else:
        lines = ["Your last python snippet executed NO actions (it only computed or inspected state - that is fine)."]
    executed_actions = summary.get("executed_actions")
    rendered = ""
    if isinstance(executed_actions, list):
        rendered = ", ".join(str(name).strip() for name in executed_actions[:10] if str(name).strip())
    if summary.get("game_over"):
        fatal = str(summary.get("fatal_action") or "")
        lines.append(
            "Reminder of the still-pending outcome from earlier this turn: the "
            f"executed sequence ({rendered or 'see earlier message'}) ended in "
            "GAME OVER"
            + (f" after '{fatal}'" if fatal else "")
            + "; the level was auto-reset and you are planning the fresh attempt "
            "now. Do NOT re-submit that sequence - the full diagnosis "
            "instructions were given in an earlier message this turn."
        )
    elif summary.get("level_transition"):
        lines.append("Reminder: the earlier sequence this turn completed the level; you are now on the new level.")
    elif rendered:
        lines.append(f"For reference, the most recent executed sequence earlier this turn was: {rendered}.")
    return lines


NO_SUMMARY_LINE = "No previous action sequence was captured."


# --- no_world_model --------------------------------------------------------------------------------------------------

WORLD_MODEL_SYSTEM = {
    # Python tool guidance: his WORLD_MODEL_FREE_ADDENDUM in place of the two world-model lines.
    "- Maintain a compact working world model: what entities or regions exist, what actions seem to do, what the goal "
    "likely is, what remains uncertain, and what plan best fits the evidence so far.\n": "",
    "- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. "
    "If confidence is low, program a discriminating probe and revise the world model from the result.\n":
        "- Optimize for the shortest reliable sequence that advances the current goal. If confidence is low, program "
        "a discriminating probe and revise your approach from the result.\n",
    # Per-turn discipline (ours): no carried model, no revised-model demand; his opener sentence.
    "(applies to every turn; each user turn carries the outcome of the previous sequence, the state line, the valid "
    "actions and, when your previous reply's text had one, the world model you carried)":
        "(applies to every turn; each user turn carries the outcome of the previous sequence, the state line and the "
        "valid actions)",
    "- Use Python to inspect the evidence, refine the world model from the newest history, and search or score "
    "candidate actions or short sequences against the current goal.\n":
        "- Use Python to inspect the evidence from the newest history, and search or score candidate actions or short "
        "sequences against the current goal as you currently understand it.\n",
}
WORLD_MODEL_DISCIPLINE_RE = re.compile(r"- Maintain a compact working world model: what the level contains, .*?"
                                       r"keep that text short\.\n")
EMPTY_WORLD_MODEL_LINE = "end of world model. "


def world_model_free(text):
    for old, new in WORLD_MODEL_SYSTEM.items():
        if text.count(old) != 1:
            raise ValueError("dfr_loop: pinned world-model system text changed")
        text = text.replace(old, new, 1)
    text, count = WORLD_MODEL_DISCIPLINE_RE.subn("", text)
    if count != 1:
        raise ValueError("dfr_loop: pinned world-model discipline line changed")
    return text


# --- action_feedback -------------------------------------------------------------------------------------------------

FEEDBACK_SYSTEM = {
    "- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n":
        "- Each history entry exposes `.action`, `.frame`, and `.result`; entries are not subscriptable like "
        "`entry['action']`.\n",
    "`.frame` (alias of `.after_frame`), and `.result`.\n":
        "`.frame` (alias of `.after_frame`), and `.result`. Each `.result` describes that individual action and is "
        "retained with its history entry.\n",
    "- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions "
    "may have an empty `.result`. For before/after diffs,":
        "- `last_transition` is `transitions[-1]` or `None`. A refused action creates no transition and does not "
        "overwrite existing transition results. For before/after diffs,",
    "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground on "
    "the next turn.\n":
        "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground on "
        "the next turn.\n"
        "- Flag semantics: `game_over` = this attempt FAILED (death or a limit ran out) — the level auto-resets to its "
        "initial state and completed levels are kept; it never means the run is won. `level_completed` = advanced one "
        "level. `run_complete`/`done` = the whole game is won. Deaths come either from the action itself (e.g., a "
        "hazard cell) or from a step/time budget depleting — watch for a bar at the grid border that shrinks each "
        "action, and count remaining budget into your plans.\n",
    "When the final executed action of the latest `action(...)` call animated, `last_action_result['animation']`":
        "When the final executed action of the latest model-issued call animated, "
        "`last_action_call_result['animation']`",
}
LAST_RESULT_LINE_RE = re.compile(r"- `last_action_result` is the persisted result dict from the most recent "
                                 r"`action\(\.\.\.\)` call\..*?\n")
LAST_CALL_RESULT_LINES = (
    "- `last_action_call_result` contains the result of the latest model-issued `action(...)` call, including batch "
    "totals, executed/skipped actions, and stop reasons. `action(...)` returns this result. It remains available "
    "across later Python inspection calls and analyzer turns, and is `{}` before any action call result exists.\n"
    "- On the next analyzer turn after automatic RESET, `current_frame` shows the restarted level. The fatal action and "
    "RESET are separate transitions. Each transition result has `automatic=True` for a harness-initiated action and "
    "`automatic=False` for a model-issued action. Automatic RESET does not overwrite `last_action_call_result`: it "
    "still describes the call that caused death.\n"
)


def feedback_system(text):
    for old, new in FEEDBACK_SYSTEM.items():
        if text.count(old) != 1:
            raise ValueError("dfr_loop: pinned action-feedback system text changed")
        text = text.replace(old, new, 1)
    text, count = LAST_RESULT_LINE_RE.subn(lambda _: LAST_CALL_RESULT_LINES, text)
    if count != 1:
        raise ValueError("dfr_loop: pinned last_action_result line changed")
    # Every remaining mention (the refresh list, the per-turn tool line, "Inspection-only Python calls ...").
    return text.replace("`last_action_result`", "`last_action_call_result`")


BLOCKED_NOOP = "BLOCKED (known no-op in this exact board state; never executed)"
BLOCKED = {"reset_refused": "BLOCKED (RESET as a level's first action; never executed)",
           "invalid_action": "BLOCKED (not a valid action right now; never executed)"}
SKIPPED = "SKIPPED (never executed)"
GAME_OVER_TRACE = "Per-action trace of the previous sequence: {trace}."


def action_trace(requested, verdicts, blocked=None):
    """One line per submitted action (his ``_build_action_trace``): executed actions get the verdict, our refused
    repeats his known-no-op block, the position a stop refused (``blocked``: its stop reason) its block, the rest
    SKIPPED."""
    trace = []
    for position, name in enumerate(requested, 1):
        if position <= len(verdicts):
            payload = verdicts[position - 1][1]
            trace.append(f"{position}: {name} -> " + (dfr.verdict(payload) if payload is not None else BLOCKED_NOOP))
        elif blocked and position == len(verdicts) + 1:
            trace.append(f"{position}: {name} -> {BLOCKED.get(blocked, 'BLOCKED (never executed)')}")
        else:
            trace.append(f"{position}: {name} -> {SKIPPED}")
    return trace


def action_echo(requested, executed_payloads, stop_reason):
    """His ``_build_action_echo``: one line per ``action(...)`` call, printed by the sandbox."""
    if not requested:
        return ""
    if len(requested) == 1:
        if executed_payloads:
            return f"[action] {requested[0]} -> {dfr.verdict(executed_payloads[0])}"
        reason = f" | stop_reason: {stop_reason}" if stop_reason else ""
        return f"[action] {requested[0]} -> not executed{reason}"
    ran = ", ".join(str(item.get("action_display") or item.get("action_name") or "?") for item in executed_payloads)
    line = f"[action] batch of {len(requested)} actions, {len(executed_payloads)} executed"
    if ran:
        line += f": {ran}"
    if stop_reason:
        line += f" | stop_reason: {stop_reason}"
    return line


def turn_action_trace(results):
    """His ``_turn_action_trace``: the snippet's per-call traces stitched and renumbered (synthesized for a call
    without one)."""
    entries = []
    for item in results:
        trace = item.get("action_trace")
        if isinstance(trace, list) and trace:
            entries.extend(re.sub(r"^\d+:\s*", "", str(line)) for line in trace)
            continue
        names = item.get("executed_actions")
        if not isinstance(names, list) or not names:
            display = str(item.get("action_display") or "").strip()
            names = [display] if display else []
        entries.extend(f"{name} -> {dfr.verdict(item)}" for name in names)
    return [f"{position}: {entry}" for position, entry in enumerate(entries, 1)]


def summary_extras(action_results):
    """His step-summary fields the openers read: ``game_over_trace`` and ``fatal_action``."""
    executed = [item for item in action_results if isinstance(item, dict) and item.get("executed")]
    over = any(item.get("game_over") for item in executed)
    return {"game_over_trace": turn_action_trace(executed) if over else [],
            "fatal_action": next((str(item.get("fatal_action") or item.get("action_display") or "")
                                  for item in executed if item.get("game_over")), "")}


# The persistent sandbox (after dfr_harness's child block): ``action()`` prints and pops the echo, and
# ``last_action_call_result`` is the per-call result (the same dict as ``last_action_result``).
CHILD_BLOCK = r'''
    _v5_inner_action = runtime_globals["action"]

    def action(actions):
        result = _v5_inner_action(actions)
        if isinstance(result, dict):
            echo = result.pop("action_echo", None)
            if echo:
                print(echo)
        return result

    runtime_globals["action"] = action
    _v5_refresh_inner = _refresh_state

    def _refresh_state(state_payload):
        _v5_refresh_inner(state_payload)
        last = runtime_globals.get("last_action_result")
        if isinstance(last, dict):
            last.pop("action_echo", None)
        runtime_globals["last_action_call_result"] = last

'''
RUNTIME_NAMES = '"last_action_result", "action", "result"'


def with_feedback(bootstrap):
    """``bootstrap`` with the child block before the request loop (after dfr_harness's, which it wraps)."""
    if "_v5_inner_action" in bootstrap:
        raise ValueError("dfr_loop: the sandbox code is already installed")
    if bootstrap.count(dfr.ANCHOR) != 1:
        raise ValueError("dfr_loop: pinned Duck sandbox loop changed")
    text = bootstrap.replace(dfr.ANCHOR, CHILD_BLOCK.lstrip("\n") + dfr.ANCHOR, 1)
    if text.count(RUNTIME_NAMES) == 1:  # the fixes loop's inventory skips runtime names
        text = text.replace(RUNTIME_NAMES, RUNTIME_NAMES + ', "last_action_call_result"', 1)
    return text


def aggregate(rows):
    keys = ("yields", "resumes", "stale_openers", "stale_no_tool_call", "echoes", "traces", "gameover_traces")
    rows = [r for r in rows if r]
    return {key: sum(int(r.get(key) or 0) for r in rows) for key in keys}
