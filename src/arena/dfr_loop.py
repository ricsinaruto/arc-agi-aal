"""hero-v2 ``dfr_loop`` (v5): the agent loop matched more closely to dfranzen's harness
(dfranzen/arc-agi-3-milestone-2-solution, Apache-2.0):

- ``prompt_fix``: the system line that said earlier turns stay "until a context compaction replaces them with your own
  summary" describes drain_half instead.
- ``no_world_model``: his ARC3_MEMORY_SECTIONS=off: no carried block, no "MUST give the revised world model" lines,
  no cross-level hypothesis carry, his world-model-free system line and new-level prompt; the no-tool-call follow-up
  loses its sentence on the labelled lines.
- ``inventory_animation`` (our bug fix): the drain note's inventory answers a lazy animation view's frame requests
  instead of discarding the interpreter.
"""

import re

from .harness_fixes import exact

SETTING = {"prompt_fix": True, "yield_tokens": 0, "no_world_model": True, "action_feedback": False,
           "inventory_animation": True}


def settings(value):
    return exact("dfr_loop", value, SETTING)


# --- prompt_fix ------------------------------------------------------------------------------------------------------

# the image/history line as harness_fixes prompt_text leaves it (the drain replaces it)
COMPACTION_LINE = (
    "- Each user turn includes an image of the board as it was before that turn's actions (labelled with its step). "
    "Earlier turns, their images and your earlier replies stay in the context verbatim until a context compaction "
    "replaces them with your own summary.\n"
)
DRAIN_LINE = (
    "- Each user turn includes an image of the board as it was before that turn's actions (labelled with its step). "
    "Earlier turns, their images and your earlier replies stay in the context verbatim until the context fills; then "
    "the oldest turns are dropped, a note marks the drop, and the extra images (diff, death frame) of the earlier "
    "turns that remain become a placeholder.\n"
)


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
# The public no-tool-call follow-up's sentence on the labelled lines (nothing parses them under no_world_model).
FOLLOWUP_LABELS = (
    "If helpful, include short world-model update lines such as `World model:`, `Goal model:`, `Action model:`, "
    "`Recent findings:`, `Open questions:`, `Plan:`, or `Cross-level notes:`. "
)


def world_model_free(text):
    for old, new in WORLD_MODEL_SYSTEM.items():
        if text.count(old) != 1:
            raise ValueError("dfr_loop: pinned world-model system text changed")
        text = text.replace(old, new, 1)
    text, count = WORLD_MODEL_DISCIPLINE_RE.subn("", text)
    if count != 1:
        raise ValueError("dfr_loop: pinned world-model discipline line changed")
    return text
