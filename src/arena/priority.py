"""Priority scheduling (2026; ``budget_policy="priority"``), ported from dfranzen's ARC3-Inference.

Source: dfranzen, ARC3-Inference (Apache License 2.0): ``inference/agent/priority_scheduler.py`` (``priority_value``,
``parse_total_levels``, ``PrioritySnapshot``, the "remaining" tail table) and ``inference/agent/tool_agent.py``
(``_PriorityGate``, ``_maybe_handover``, ``_snapshot_priority`` with the tail fade, ``_note_history_evicted``) and
``inference/framework/solver.py`` (unstarted games acquire at 2,000,000 - queue position). The scoring function below
is a verbatim port of their ``priority_value`` without the empirical 80K tail table (their ``tail_lookup=True``) and
without the pace reference (``cost_multiplier`` stays 1): the user's scheduler rules allow no per-game information
from earlier runs, only in-run signals and generic constants. Their remaining constants are pooled generic ones.

The state machine lives in ``arena.scheduler.Scheduler``; this module holds the settings and the scoring.

Behaviour (their gate, at our stream count):

- ``slots`` = the focus streams (the scheduler's ``active_limit``). Every game is eligible; no opening, no level
  budgets, no retirement, no backlog ranking other than the priority below.
- Unstarted games come first, in roster order (their band 2,000,000 - queue position): every game gets a first window
  before any re-queued game returns. The release admits the first ``slots`` games (``first_window``).
- An admitted game keeps its slot until its next context eviction: a summary compaction (``GameRecord.note_compaction``)
  or a drain (``GameRecord.note_context_drain``). That is the one moment yielding the slot is free: the cached prefix is
  invalid anyway and the next request pays the prefill. There the game is handed over: re-scored and re-queued, and the
  best waiting game (re-scored now, ``refresh_queue``) takes the slot, which may be the same game (``handover``, kept).
- A finished game frees its slot (``finished``); the best waiting game takes it. Parked games keep their state.
- Score of a started game, at its current level L = levels cleared + 1 (1-based), with ``actions`` and ``tokens`` the
  actions and completion tokens spent on L, and N = the game's public level count clipped to 6..10 (10 if unknown):
    A = L x (h / (h + actions))^2 x 55 / (N (N + 1) / 2)   (normalize_score; h = human_actions)
    B = 8 / 7 / 5 / 0 for 3+ / 2 / 1 / 0 levels after L    (tail_lookup "remaining"), x the tail fade
    C = w x 0.5^((actions / action_scale)^2) + (1 - w) x 0.5^((tokens / token_scale)^2)   (w = action_weight)
    priority = max(1, int((A + B) x C x 100))   (``floor`` false: the float (A + B) x C x 100)
  ``tail_lookup`` "capped" is their capped tail min(tail_cap, tail_base + L - 1) instead.
- Level cap (``level_cap_tokens``, off by default): at an eviction where the game's tokens on its level have reached
  the cap, the game is capped: it ranks below every uncapped waiting game, so the slot goes to the best of those (the
  usual handover). Capped games are admitted only when no uncapped game waits (a slot that would otherwise idle),
  fewest tokens on the level first, each for ``level_cap_grant_tokens`` more; at the end of a grant the slot rotates to
  a waiting capped game with fewer tokens, else the game keeps it. A clear lifts the cap.
- Tail fade: over the last ``tail_fade_fraction`` of the run's wall (the real wall: the competition gateway shortens
  ``GlobalClock.seconds`` to its deadline), B is scaled by the remaining wall / that window, down to 0 at the end, so
  late slots go to games whose current level is worth finishing rather than to deep tails there is no time to reach.

Ties: the earlier arrival in the queue first (their heap's sequence number); the game being handed over arrives last.
"""

import math
from dataclasses import dataclass

# Heuristic continuation value by levels remaining AFTER the current level: 3 or more -> 8; 2 -> 7; 1 -> 5; 0 -> 0.
# These are final B values, so neither score normalization nor an efficiency discount applies. (dfranzen,
# ARC3-Inference, Apache-2.0: TAIL_LOOKUP_REMAINING.)
TAIL_LOOKUP_REMAINING = {n: (8.0,) * (n - 3) + (7.0, 5.0, 0.0) for n in range(6, 11)}

# Unstarted games: 2,000,000 - roster position, above any score (at most ~1,325).
UNSTARTED_BASE = 2_000_000

DEFAULTS = {
    # Their submitted settings (ARC3_PRIORITY_*): refresh queue on, pace off, tail fade over the final 20% of the wall,
    # the "remaining" tail, score normalization on and the pooled generic scales.
    "refresh_queue": True,
    "tail_fade": True,
    "tail_fade_fraction": 0.2,
    "tail_lookup": "remaining",
    "normalize_score": True,
    "human_actions": 25.0,
    "action_scale": 115.0,
    "token_scale": 62000.0,
    "action_weight": 0.25,
    # Only for tail_lookup "capped" (their environment defaults).
    "tail_base": 2.0,
    "tail_cap": 4.0,
    # Their integer floor (max(1, int(value))). False keeps the float: levels past ~300K
    # tokens no longer tie at 1 and rotate first-in-first-out; the least-spent level goes first.
    "floor": True,
    # Per-level token cap: a game whose gameplay tokens on its current level reach the cap
    # is handed over at its next context eviction and ranks below every uncapped waiting game until it clears the
    # level; capped games take slots that would otherwise idle, fewest tokens on the level first, in grants of
    # ``level_cap_grant_tokens`` (generic constants; no per-game information).
    "level_cap_tokens": 0,
    "level_cap_grant_tokens": 100000,
    # The same cap in requests: a loop guard, set far above any legitimate level.
    "level_cap_requests": 0,
    "level_cap_grant_requests": 100,
}
TAIL_LOOKUPS = ("remaining", "capped")
# Admission-log reasons (admission.jsonl ``reason``): admit = why the game got a slot (its first window, else what
# freed the slot); park = a handover to another game; grant = a handover the game won back.
REASONS = {
    "admit": ("first_window", "handover", "finished"),
    "park": ("handover",),
    "grant": ("handover",),
    "finish": ("finished", "wall_end"),
}
EVICTIONS = ("compaction", "drain")


def parse_total_levels(value):
    """Validate public level-count metadata, then clip to the table domain (dfranzen, ARC3-Inference, Apache-2.0)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        return None
    return min(10, max(6, int(number)))


@dataclass(frozen=True)
class PrioritySnapshot:
    level: int  # the level being played, 1-based (levels cleared + 1)
    actions: int  # actions on that level
    tokens: float  # completion tokens on that level
    cost_multiplier: float = 1.0  # their pace; always 1 here (no pace reference)
    total_levels: int | None = None


def priority_value(state, *, endgame=False, tail_fraction=None, human_actions=25.0, action_scale=115.0,
                   token_scale=62000.0, action_weight=0.25, tail_base=6.0, tail_cap=6.0, normalize_score=False,
                   tail_lookup=False, tail_efficiency=0.8, floor=True):
    """Port of dfranzen's ``priority_value`` (ARC3-Inference, Apache-2.0), argument for argument.

    ``tail_lookup`` "remaining" uses the fixed 8/7/5/0 table; False keeps the capped tail min(tail_cap, tail_base +
    level - 1). Their ``True`` (the empirical 80K table) is not ported: it is fitted on historical runs.
    ``tail_efficiency`` only applied to that table and is accepted for signature parity.
    """
    del tail_efficiency
    if tail_lookup not in ("remaining", False):
        raise ValueError("tail_lookup is 'remaining' or False (the 80K table is not ported)")
    level = max(1, state.level)
    total_levels = parse_total_levels(getattr(state, "total_levels", None)) or 10
    actions, tokens = max(0, state.actions), max(0.0, state.tokens)
    pace = min(2.0, max(0.5, state.cost_multiplier))
    efficiency = (human_actions / (human_actions + actions)) ** 2
    current_value = level * efficiency
    if normalize_score:
        # Normalize immediate reward A to a ten-level reference (sum 1..10). B is a separately chosen continuation
        # value, not normalized reward.
        current_value *= 55.0 / (total_levels * (total_levels + 1) / 2)
    if endgame and tail_fraction is None:
        value = (
            level * 0.25 + current_value if normalize_score
            else level * (0.25 + efficiency)
        ) / pace * 100
    else:
        if tail_lookup == "remaining":
            index = min(total_levels, level) - 1
            tail = TAIL_LOOKUP_REMAINING[total_levels][index]
        else:
            tail = min(tail_cap, tail_base + level - 1)
        if tail_fraction is not None:
            tail *= min(1.0, max(0.0, tail_fraction))
        chance = (
            action_weight * 0.5 ** ((actions / action_scale) ** 2)
            + (1.0 - action_weight) * 0.5 ** ((tokens / (token_scale * pace)) ** 2)
        )
        value = (current_value + tail) * chance / pace * 100
    return max(1, int(value)) if floor else value


def settings(raw=None):
    """Validated ``priority`` settings (``DEFAULTS`` overlaid with ``raw``)."""
    merged = {**DEFAULTS, **(raw or {})}
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown priority settings: {sorted(unknown)}")
    for key in ("refresh_queue", "tail_fade", "normalize_score", "floor"):
        if type(merged[key]) is not bool:
            raise ValueError(f"priority.{key} is true or false")
    if merged["tail_lookup"] not in TAIL_LOOKUPS:
        raise ValueError(f"priority.tail_lookup is one of {TAIL_LOOKUPS}")
    for key in ("tail_fade_fraction", "human_actions", "action_scale", "token_scale", "action_weight", "tail_base",
                "tail_cap"):
        value = merged[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"priority.{key} must be a finite number")
        merged[key] = float(value)
    if not 0.0 < merged["tail_fade_fraction"] <= 1.0:
        raise ValueError("priority.tail_fade_fraction must be > 0 and <= 1")
    for key in ("human_actions", "action_scale", "token_scale"):
        if merged[key] < 1.0:
            raise ValueError(f"priority.{key} must be at least 1")
    if not 0.0 <= merged["action_weight"] <= 1.0:
        raise ValueError("priority.action_weight must be in [0, 1]")
    if merged["tail_base"] < 0 or merged["tail_cap"] < 0:
        raise ValueError("priority.tail_base and tail_cap must be non-negative")
    for key, least in (("level_cap_tokens", 0), ("level_cap_grant_tokens", 1), ("level_cap_requests", 0), ("level_cap_grant_requests", 1)):
        if type(merged[key]) is not int or merged[key] < least:
            raise ValueError(f"priority.{key} must be an integer of at least {least}")
    return merged


def tail_fraction(cfg, wall_seconds, remaining_seconds):
    """The tail-fade factor (None when the fade is off): the remaining wall over the fade window, a ``tail_fade_fraction``
    share of the whole wall (their full shared run duration, not the shrinking time left); priority_value clips it."""
    if not cfg["tail_fade"]:
        return None
    window = cfg["tail_fade_fraction"] * max(0.0, float(wall_seconds))
    remaining = max(0.0, float(remaining_seconds))
    return remaining / window if window > 0.0 else 0.0


def snapshot(game):
    """The scoring inputs of a scheduler ``GameRecord``: the 1-based level being played, actions and completion tokens
    on it (summaries included, as their session generated-token count) and the public level count."""
    return PrioritySnapshot(level=game.levels + 1, actions=game.actions_on_level, tokens=float(game.tokens_on_level()),
                            total_levels=game.total_levels)


def score(game, cfg, wall_seconds, remaining_seconds):
    """``(priority, details)`` of a started game now."""
    state = snapshot(game)
    fade = tail_fraction(cfg, wall_seconds, remaining_seconds)
    value = priority_value(
        state, tail_fraction=fade, human_actions=cfg["human_actions"], action_scale=cfg["action_scale"],
        token_scale=cfg["token_scale"], action_weight=cfg["action_weight"], tail_base=cfg["tail_base"],
        tail_cap=cfg["tail_cap"], normalize_score=cfg["normalize_score"],
        tail_lookup="remaining" if cfg["tail_lookup"] == "remaining" else False, floor=cfg["floor"])
    details = {"level": state.level, "actions": state.actions, "tokens": int(state.tokens),
               "total_levels": parse_total_levels(state.total_levels) or 10,
               "tail_fraction": None if fade is None else round(min(1.0, fade), 4)}
    return value, details
