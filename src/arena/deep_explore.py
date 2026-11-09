"""Depth-first scheduling with a wide exploration rotation (``fixed-2x-deep-explore``).

ideas.md: "more streams at lower context + depthwise scheduling: always keep playing the first game that clears in
1 slot, then rotate the others in the other slots to find the next game to clear a level and so on", with mixed
context sizes. The state machine lives in ``arena.scheduler.Scheduler`` (``budget_policy="deep-explore"``):

- Deep slots (``deep_slots``, window ``deep_window``): the first game to clear a level takes a free deep slot at once
  and keeps playing level after level while its completion tokens on the current level stay within
  ``max(deep_min_tokens, deep_multiple x median tokens of the levels it has cleared in this run)``. Past that budget it
  is demoted (``deep_demote``: parked, back in the exploration rotation, its window cut to ``explore_window`` so the
  existing compaction path fits it at its next request) only when a promotion candidate is queued; otherwise it keeps
  playing (a deep slot is never idled).
- Promotion queue: exploration games that cleared a level, most recent clear first (ties: fewer tokens on their
  cleared levels). A free deep slot takes the queue head (``deep_promote``); its window grows to ``deep_window``.
- Exploration slots (``explore_slots``, window ``explore_window``): games rotate with a fixed quantum of
  ``explore_quantum_tokens`` completion tokens per turn; never-played games first (roster order), then fewest total
  completion tokens; games with ``explore_max_tokens`` or more on level 1 and no clear last. A game that clears in an
  exploration slot keeps the slot until its quantum ends, then parks into the promotion queue (``explore_queue``) and is
  not rotated while queued.
- No opening phase: the exploration slots run from t = 0; the deep slot fills at the first clear.
"""

import math

DEFAULTS = {
    "deep_slots": 1,
    "deep_window": 131072,
    "explore_slots": 8,
    "explore_window": 32768,
    "deep_min_tokens": 25_000,
    "deep_multiple": 2.5,
    "explore_quantum_tokens": 18_000,
    "explore_max_tokens": 60_000,
}


def settings(raw=None):
    """Validated ``deep_explore`` settings."""
    merged = {**DEFAULTS, **(raw or {})}
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown deep_explore settings: {sorted(unknown)}")

    def number(key, low, high=math.inf, integer=False):
        value = merged[key]
        kinds = (int,) if integer else (int, float)
        if type(value) not in kinds or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"deep_explore.{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")

    number("deep_slots", 1, 4, integer=True)
    number("explore_slots", 1, 24, integer=True)
    number("deep_window", 4096, integer=True)
    number("explore_window", 4096, merged["deep_window"], integer=True)
    number("deep_min_tokens", 1, integer=True)
    number("deep_multiple", 0.5, 20)
    number("explore_quantum_tokens", 1, integer=True)
    number("explore_max_tokens", 1, integer=True)
    return merged
