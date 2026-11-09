"""Prompt layout of the bridged Duck analyzer (v5).

The public per-turn user prompt repeats about 400 tokens of constant instructions, and the
retained history keeps every earlier user turn verbatim: the carried world model, that
boilerplate and a base64 board image, up to thirty times per request. Here the constant
lines move into the (cached) system prompt once, the newest user turn alone carries the
world model and an image labelled with its step, and every earlier user turn is reduced to
a one-line stub. The same override explains an automatic level reset after GAME_OVER and
appends the stuck notice. All public text is matched exactly and must occur once.
"""

import re

from .sandbox import _replace_all

# Lines removed from every user turn (they are constant) and restated once below.
USER_PROMPT_REMOVALS = (
    "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, and `action(actions)`.",
    "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
    "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
    "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
    "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
    "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
    "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`.",
    "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
    "If you include assistant text before a tool call, keep it short and use it to update the world model. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.",
)
OPTIONAL_REMOVALS = ("If you use MOUSE, include integer row and col arguments.",)

TURN_DISCIPLINE = (
    "\n\nPer-turn discipline (applies to every turn; each user turn carries only the outcome of "
    "the previous sequence, the state line, the valid actions and the world model you carried):\n"
    "- Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, "
    "`transitions`, `last_transition`, `valid_actions`, `last_action_result`, and `action(actions)`.\n"
    "- Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs "
    "are not available.\n"
    "- Keep tool output compact: use `current_frame.segmentation` as the primary view, and "
    "`current_frame.ascii` only for a small specific region; never print full boards.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or "
    "`last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the "
    "current frame, not the previous one.\n"
    "- Use Python to inspect the evidence, refine the world model from the newest history, and "
    "search or score candidate actions or short sequences against the current goal.\n"
    "- Maintain a compact working world model: what the level contains, what actions do, what "
    "the goal is, what is uncertain, and the best current plan. The user turn shows the world "
    "model carried from your previous turn; copy it and add or remove items based on the evidence "
    "you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE "
    "WORLD MODEL, using the prefixes `World model:`, `Goal model:`, `Action model:`, "
    "`Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`; keep that text short.\n"
    "- When ready, call `action(actions)` from inside `python` with the best valid action or "
    "ordered batch selected by your code; batch a reliable short sequence in one call. You may "
    "call `action(actions)` more than once in one snippet; stop immediately if a receipt reports "
    "`game_over`, `run_complete`, `level_completed`, or `done`.\n"
    "- If you use MOUSE, include integer row and col arguments.\n"
)
SYSTEM_PROMPT_ADDITIONS = {
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n":
        "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
        + TURN_DISCIPLINE,
    "- User turns include an attached image of the current ARC grid.\n":
        "- The newest user turn includes an image of the board as it was before this turn's "
        "actions (labelled with its step); earlier turns keep only a note that their image was omitted.\n",
    "and `last_action_result['valid_actions']`.\n":
        "and `last_action_result['valid_actions']`.\n"
        "- Every executed action's receipt also carries `transition`: `changed_cells`, "
        "`change_bbox` as `[row0, col0, row1, col1]` (exclusive) and, when the board was "
        "segmented, `objects` with `appeared`/`disappeared`/`moved` (counts plus up to three "
        "entries with color, pixels, bbox and `from`), `background_reshaped` and "
        "`no_change_repeat`; `transitions[i].result['transition']` keeps it for every executed "
        "action. `tried_no_effect_here` lists actions already tried on this exact board without "
        "any change; repeating one is refused with `stop_reason='no_change_repeat'`.\n",
}

GAME_OVER_LINES = "You are still on the same level.\nThe game is over."
IMAGE_LABEL = "Grid image of the board at step {step} (before this turn's actions):"
IMAGE_LABEL_RE = re.compile(r"Grid (?:image|text) of the board at step (\d+) \(before this turn's actions\):")
STATE_RE = re.compile(r"Current state: step (\d+), level (\d+)")
STUB_PREFIXES = (
    "The code executed",
    "Executed actions",
    "You have completed",
    "You have progressed",
    "You are still",
    "The game is over.",
    "GAME_OVER at step",
    "No previous",
    "Current state:",
)


def system_prompt(text):
    return _replace_all(text, SYSTEM_PROMPT_ADDITIONS, "system prompt layout")


def reduce_user_prompt(text):
    """Drop the constant lines (each pinned to occur once) from a public user prompt."""
    from inference.agent.prompts import TOOL_CALL_FORMAT_GUIDANCE

    lines = text.split("\n")
    for line in (*USER_PROMPT_REMOVALS, TOOL_CALL_FORMAT_GUIDANCE):
        if lines.count(line) != 1:
            raise ValueError("Pinned Duck user prompt changed")
        lines.remove(line)
    for line in OPTIONAL_REMOVALS:
        while line in lines:
            lines.remove(line)
    return "\n".join(lines)


def game_over_note(auto_reset, level):
    actions = ", ".join(str(a) for a in (auto_reset.get("actions") or [])[-10:]) or "none recorded"
    return (
        f"GAME_OVER at step {auto_reset.get('step')} after actions [{actions}]; the level was "
        f"reset automatically and the board below is the fresh start of level {level}. What you "
        "learned about this level's mechanics still applies; only your plan needs revising."
    )


def turn_stub(text):
    """One line summarising an earlier user turn (its image and world model are dropped)."""
    kept = [line for line in text.split("\n") if line.startswith(STUB_PREFIXES)]
    image = IMAGE_LABEL_RE.search(text)
    kind = 'text' if image and image.group(0).startswith('Grid text') else 'image'
    note = f" grid {kind} at step {image.group(1)} omitted." if image else ""
    return "[earlier turn: " + " ".join(kept) + note + "]"


class PromptMixin:
    """Overrides of the public ToolAgent's prompt assembly; needs ``_owner``."""

    _owner = None

    def _build_user_message(self, user_prompt, current_frame):
        message = super()._build_user_message(user_prompt, current_frame)
        content = message.get("content")
        if isinstance(content, list) and content and content[0].get("type") == "text":
            text = content[0]["text"]
            suffix = "\n\nCurrent grid image:"
            if text.count(suffix) != 1:
                raise ValueError("Pinned Duck image label changed")
            step = current_frame.step if current_frame is not None else 0
            content[0]["text"] = text.replace(suffix, "\n\n" + IMAGE_LABEL.format(step=step))
        # The arena's dfr_harness (larger board image, diff and game-over images); absent here.
        extend = getattr(self._owner, "observation_message", None) if self._owner is not None else None
        if extend is not None:
            message = extend(message, current_frame)
        return message

    def _build_user_prompt(self, action_num, **kwargs):
        owner = self._owner
        # The arena's dfr_loop: a summary carried over an invocation that executed nothing opens with its stale lines.
        stale = getattr(owner, "stale_opener_lines", None) if owner is not None else None
        stale = stale(kwargs.get("previous_step_summary")) if stale is not None else None
        if stale:
            kwargs = {**kwargs, "previous_step_summary": None}
        text = reduce_user_prompt(super()._build_user_prompt(action_num, **kwargs))
        if stale:
            from arena.dfr_loop import NO_SUMMARY_LINE

            lines = text.split("\n")
            if lines.count(NO_SUMMARY_LINE) != 1:
                raise ValueError("Pinned Duck no-summary line changed")
            at = lines.index(NO_SUMMARY_LINE)
            text = "\n".join([*lines[:at], *stale, *lines[at + 1:]])
        if owner is not None and "The game is over." in text and owner.obs.state != "GAME_OVER":
            note = game_over_note(
                owner.auto_reset or {"step": None, "actions": []},
                max(1, min(owner.obs.total_levels, owner.obs.levels + 1)),
            )
            if text.count(GAME_OVER_LINES) != 1:
                raise ValueError("Pinned Duck game-over lines changed")
            text = text.replace(GAME_OVER_LINES, note)
        rewrite = getattr(owner, "rewrite_user_prompt", None) if owner is not None else None
        if rewrite is not None:
            # The arena's harness_fixes prompt text (empty world-model line, one step convention); identity otherwise.
            text = rewrite(text, kwargs.get("current_frame"))
        if owner is not None:
            tried = owner.tried_no_effect_here()
            notice = owner.stuck.notice(max(1, owner.obs.levels + 1), tried)
            if notice:
                owner.workspace.append("stuck_notice", step=owner.steps, text=notice)
                text += "\n" + notice
            # One-off notices from an arena harness (harness v2's new-level notice); absent here.
            extra = getattr(owner, "turn_notice", None)
            extra = extra() if extra is not None else None
            if extra:
                text += "\n" + extra
        return text

    def _persistent_history_messages(self, messages, *, tools=None):
        kept = super()._persistent_history_messages(messages, tools=tools)
        observations = [
            index
            for index, message in enumerate(kept)
            if str(message.get("role", "")) == "user" and _has_image_label(message)
        ]
        # The next turn builds a fresh observation, so every retained one becomes a stub.
        for index in observations:
            kept[index] = {"role": "user", "content": _stub_text(kept[index])}
        return kept


def _message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _has_image_label(message):
    return IMAGE_LABEL_RE.search(_message_text(message)) is not None


def _stub_text(message):
    return turn_stub(_message_text(message))
