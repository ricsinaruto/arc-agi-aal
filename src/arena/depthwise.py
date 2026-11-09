"""Depth-first slots with fixed-2x budgets and a level-0 rotation (``fixed-2x-depthwise``).

After ``fixed-2x-deep-explore`` (0.55% on the 110-clone hour: fast first clears, but its single deep slot took 15% of
the tokens and its stall rule demoted deep games after 250-700 s), every slot can hold a depth game. The state machine
lives in ``arena.scheduler.Scheduler`` (``budget_policy="depthwise"``):

- ``slots`` slots from the release, no opening phase. A game that has cleared no level plays in any free slot at the
  ``level0_window`` (the agent's compaction threshold follows the slot window, ``GameRecord.window``) for a turn of
  ``level0_quantum_tokens`` completion tokens, then goes back to the rotation (``level0_rotate``) if any game waits;
  otherwise the turn is renewed (``level0_renew``).
- A game that clears level 1 during its turn keeps its slot (``level0_clear``) and from then on uses ``deep_window``.
  Its level budget is fixed-2x on the busy clock: level k -> k+1 gets 2 x (k+1) x T1 (``Scheduler.level_budget``),
  granted in full at every clear (``depth_level``).
- An exhausted depth game parks (``depth_park``) only if some game waits (a level-0 game in the rotation counts);
  otherwise it keeps playing (``depth_keep``, logged once per grant).
- A freed slot goes to (1) parked games whose clear landed while parked (``depth_parked_clear``), then (2) the other
  parked games with at least one level, fastest level-1 clear first (smallest T1; ``depth_readmit``), then (3) level-0
  games: never played first (roster order), then fewest completion tokens (``level0_rotate``). A game parked in this
  tick does not take back its own slot.
- Re-admission grant (``regrant``): ``"spent"`` (default, the user's rule text): the rest of the level's fixed-2x
  allowance 2 x (k+1) x T1 - spent if positive (a parked clear starts a fresh level), else
  max(1.5 x spent, spent + 30) with spent = busy seconds on the current level; ``"full"``: the fixed-2x code's
  re-admission, the full 2 x (k+1) x T1 again.
- Refill order (``backlog``): ``"depth-first"`` (default, the order above: every game with a level ahead of every
  level-0 game) or ``"fixed-2x"`` (after the slots fill, the same order as the fixed-2x
  scheduler's steady state, ``backlog_policy="readmissions"``): a clear that landed while parked first, then fewest
  admissions, then most levels, then roster order. A parked depth game then waits behind every game admitted fewer
  times, as a parked focus game does in fixed-2x (c110-depthwise-mtp3-v1 spent 44% of its tokens readmitting six
  stalled depth games under the depth-first order).
"""

import math

DEFAULTS = {
    "slots": 6,
    "deep_window": 65536,
    "level0_window": 32768,
    "level0_quantum_tokens": 18_000,
    "regrant": "spent",
    "backlog": "depth-first",
}

REGRANTS = ("spent", "full")
BACKLOGS = ("depth-first", "fixed-2x")


def settings(raw=None):
    """Validated ``depthwise`` settings."""
    merged = {**DEFAULTS, **(raw or {})}
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown depthwise settings: {sorted(unknown)}")

    def integer(key, low, high=math.inf):
        value = merged[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"depthwise.{key} must be an integer in [{low}, {high}]")

    integer("slots", 1, 24)
    integer("deep_window", 4096)
    integer("level0_window", 4096, merged["deep_window"])
    integer("level0_quantum_tokens", 1)
    if merged["regrant"] not in REGRANTS:
        raise ValueError(f"depthwise.regrant must be one of {REGRANTS}")
    if merged["backlog"] not in BACKLOGS:
        raise ValueError(f"depthwise.backlog must be one of {BACKLOGS}")
    return merged
