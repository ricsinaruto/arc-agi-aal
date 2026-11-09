"""The three-block scheduler.

The state machine lives in ``arena.scheduler.Scheduler`` (``budget_policy="blocks"``); this module holds the settings,
the block split and the tenure rule.

User decisions (29 September), for the full competition length:

- The play wall is ``wall_seconds`` = 30,600 s (the 32,400 s envelope minus setup and wind-down), split into
  ``blocks`` = 3 equal blocks of 10,200 s. A research run may stop earlier (``study.early_stop`` with ``always``); the
  scheduler still believes the 30,600 s wall.
- The roster, in roster order, is split into ``blocks`` contiguous groups (110 -> 37 / 37 / 36, the larger groups first);
  the largest group is the stream count (``slots``). Block 1's games start together at the release. No opening, no
  audition, no six-clear switch, no level budgets, no backlog ranking: games stop only by the rules below.
- Tenure: a game plays for one block length from its first admission (``tenure_unit`` "wall_seconds"; "tokens" counts
  ``tenure_tokens`` gameplay completion tokens from the first admission instead). At a block boundary the block's own
  games (``original``) that still play are parked (``block_end``); they may return later as old games. A game pulled
  forward early from a later block (``filler``) keeps playing across the boundary into its own block, but its tenure
  ends at its first admission + one block (``tenure_end``), before the next boundary.
- Level-0 budget: a game still on level 0 whose gameplay completion tokens (the token budget clock's count) reach
  ``level0_budget_tokens`` (90,000) is retired for good (``level0_retired``). Games with a level are not subject to it.
  A level that still arrives from the retired game's last in-flight reply lifts the retirement (``retired_level``): the
  game was not on level 0 when its budget ran out, so it becomes an old game.
- A finished game (all levels, or a worker that ended) frees its slot the same way (``finished``).
- Fill rule for a freed slot: the next never-run game in roster order (from a later block: ``filler``); if none is left,
  an old game that can still progress (parked, not retired, not finished) with the most levels, then the fewest gameplay
  completion tokens, then roster order (``old_refill``); an old refill plays until the wall end (or its retirement or
  finish). At a block boundary the next block's never-run games take the parked originals' slots in roster order
  (``block_start``), and any slot left over follows the fill rule.
- Wall end: every game's clock stops (``wall_end``).

Staggered starts (for the next build; ``stagger_seconds``, 0 = the first run's lockstep start):
block 1's games are admitted in roster order spread over ``stagger_seconds`` (one every stagger / slots seconds: 16.2 s
for 37 games over 600 s), so their contexts do not grow in lockstep into the gate cap and the compaction trigger. A
slot not yet started is not idle. Every tenure is still one block from the game's own first admission, so the block's
originals end staggered too: an original parks at first admission + one block (``block_end``; exactly the boundary when
it started at the boundary) and its slot goes to the next block's never-run games in roster order, so later blocks start
with the same spacing without a schedule of their own.

Stream-sized blocks (campaign arm ``blocks3-plans-prune25-40``): ``partition`` "streams" splits the
roster in roster order into contiguous groups of ``slots`` games, the last group the remainder (110 in 3 at 40 streams
-> 40 / 40 / 30), so a block still fills every stream at its start; the last block's spare slots follow the fill rule
(old refills). The default "even" is the split above; builds that do not name ``partition`` are unchanged.

Only in-run signals and generic constants: levels and tokens of this run's games, the roster order, the wall.
"""

import math

DEFAULTS = {
    "wall_seconds": 30600,
    "blocks": 3,
    "slots": 37,
    "roster_games": 110,
    "level0_budget_tokens": 90000,
    "level_token_allowance": None,  # opt-in independent base * (cleared levels + 1)
    "tenure_unit": "wall_seconds",
    "tenure_tokens": None,
    "stagger_seconds": 600,
    "partition": "even",
}
TENURE_UNITS = ("wall_seconds", "tokens")
PARTITIONS = ("even", "streams")
# Admission-log reasons (admission.jsonl ``reason``).
REASONS = {
    "admit": ("block_start", "filler", "old_refill", "retired_continue"),
    "park": ("block_end", "tenure_end", "level0_retired"),
    "finish": ("finished", "wall_end"),
    "progress": ("retired_level",),
}


def block_sizes(games, blocks, slots=None, partition="even"):
    """Contiguous groups in roster order, the larger groups first (110 in 3 -> [37, 37, 36]); ``partition`` "streams":
    groups of ``slots`` games, the last one the remainder (110 in 3 at 40 -> [40, 40, 30])."""
    if partition == "streams":
        last = int(games) - int(slots) * (int(blocks) - 1)
        if not 1 <= last <= int(slots):
            raise ValueError(f"{games} games do not split into {blocks} blocks of {slots} streams")
        return [int(slots)] * (int(blocks) - 1) + [last]
    base, extra = divmod(int(games), int(blocks))
    return [base + (1 if i < extra else 0) for i in range(int(blocks))]


def settings(raw=None, games=None):
    """Validated ``blocks`` settings with the derived ``block_seconds`` and ``block_sizes``. ``games``: the roster
    size the scheduler plays (must be ``roster_games``)."""
    merged = {**DEFAULTS, **(raw or {})}
    merged.pop("block_seconds", None)
    merged.pop("block_sizes", None)
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown blocks settings: {sorted(unknown)}")
    for key in ("blocks", "slots", "roster_games", "level0_budget_tokens"):
        if type(merged[key]) is not int or merged[key] < 1:
            raise ValueError(f"blocks.{key} must be a positive integer")
    allowance = merged["level_token_allowance"]
    if allowance is not None and (type(allowance) is not int or allowance < 1):
        raise ValueError("blocks.level_token_allowance must be a positive integer")
    wall = merged["wall_seconds"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("blocks.wall_seconds must be a positive number")
    stagger = merged["stagger_seconds"]
    if type(stagger) not in (int, float) or not math.isfinite(stagger) or not 0 <= stagger < wall / merged["blocks"]:
        raise ValueError("blocks.stagger_seconds must be a number in [0, block length)")
    if merged["tenure_unit"] not in TENURE_UNITS:
        raise ValueError(f"blocks.tenure_unit is one of {TENURE_UNITS}")
    tokens = merged["tenure_tokens"]
    if merged["tenure_unit"] == "tokens":
        if type(tokens) is not int or tokens < 1:
            raise ValueError("blocks.tenure_tokens must be a positive integer with tenure_unit 'tokens'")
    elif tokens is not None:
        raise ValueError("blocks.tenure_tokens applies to tenure_unit 'tokens' only")
    if games is not None and int(games) != merged["roster_games"]:
        raise ValueError(f"blocks settings are for {merged['roster_games']} games; the roster has {games}")
    if merged["blocks"] > merged["roster_games"]:
        raise ValueError("more blocks than games")
    if merged["partition"] not in PARTITIONS:
        raise ValueError(f"blocks.partition is one of {PARTITIONS}")
    sizes = block_sizes(merged["roster_games"], merged["blocks"], merged["slots"], merged["partition"])
    if merged["slots"] != max(sizes):
        raise ValueError(f"blocks.slots must be the largest block ({max(sizes)} for {merged['roster_games']} games in "
                         f"{merged['blocks']} blocks): a block is the stream count")
    return {**merged, "block_seconds": wall / merged["blocks"], "block_sizes": sizes}


def home_blocks(count, cfg):
    """The block of every roster position."""
    out = []
    for block, size in enumerate(cfg["block_sizes"]):
        out.extend([block] * size)
    if len(out) != count:
        raise ValueError(f"{count} games do not match the block sizes {cfg['block_sizes']}")
    return out


def release_cap(elapsed, cfg):
    """Slots open at scheduler wall time ``elapsed``: block 1's staggered start opens one slot every
    stagger / slots seconds (the first at the release); all slots afterwards."""
    stagger = float(cfg.get("stagger_seconds") or 0.0)
    if stagger <= 0 or elapsed >= stagger:
        return cfg["slots"]
    return min(cfg["slots"], 1 + int(max(0.0, float(elapsed)) // (stagger / cfg["slots"])))


def block_at(elapsed, cfg):
    """The block index at scheduler wall time ``elapsed`` (the last block from its start to the wall end)."""
    return min(cfg["blocks"] - 1, int(max(0.0, float(elapsed)) // cfg["block_seconds"]))


def composition(sources, cfg):
    """Per block, the source game of every roster position (``sources``: the roster's source games in roster order,
    e.g. ``[games[i % 20] for i in range(110)]`` for the cyclic c110 clones) and its counts."""
    out, start = [], 0
    for block, size in enumerate(cfg["block_sizes"]):
        names = list(sources[start:start + size])
        start += size
        counts = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        out.append({"block": block + 1, "games": size, "sources": names, "distinct_sources": len(counts),
                    "second_clones": sorted(n for n, c in counts.items() if c == 2),
                    "max_clones_of_one_source": max(counts.values()) if counts else 0})
    if start != len(sources):
        raise ValueError(f"{len(sources)} roster games do not match the block sizes {cfg['block_sizes']}")
    return out


def check_block1(sources, cfg):
    """User requirement (29 September): block 1 holds every distinct source game once plus one more clone each of
    distinct other sources (c110: 20 sources + 17 second clones = 37). Returns block 1's composition; raises
    otherwise."""
    first = composition(sources, cfg)[0]
    distinct = len(set(sources))
    extra = first["games"] - distinct
    if cfg["blocks"] == 2:
        counts = [first["sources"].count(s) for s in set(sources)]
        if max(counts) - min(counts) > 1:
            raise ValueError("two-block source counts must be balanced")
        return first
    if (first["distinct_sources"] != distinct or first["max_clones_of_one_source"] > 2
            or len(first["second_clones"]) != extra):
        raise ValueError(f"block 1 must hold all {distinct} source games plus {extra} distinct second clones; got "
                         f"{first['distinct_sources']} sources, second clones {first['second_clones']}")
    return first


def rule_text(cfg):
    unit = ("wall seconds from its first admission" if cfg["tenure_unit"] == "wall_seconds"
            else f"{cfg['tenure_tokens']:,} gameplay completion tokens from its first admission")
    stagger = (f"; block 1 starts staggered over {cfg['stagger_seconds']:g} s in roster order (one every "
               f"{cfg['stagger_seconds'] / cfg['slots']:.1f} s), and every original parks one block after its own first "
               f"admission" if cfg.get("stagger_seconds") else "")
    if cfg.get("level_token_allowance"):
        stagger += (f"; independent per-level gameplay allowance {cfg['level_token_allowance']:,} * "
                    "(cleared levels + 1), excluding summaries; only a new clear refreshes it, "
                    "not resets or readmission; retire exhausted levels, preserving late action results")
    spare = [cfg["slots"] - n for n in cfg["block_sizes"]]
    if cfg.get("partition") == "streams" and any(spare):
        # every block runs all the streams (40 / 40 / 40 at 110 games).
        stagger += (f"; every block runs all {cfg['slots']} streams ({' / '.join([str(cfg['slots'])] * cfg['blocks'])}): "
                    f"a block's spare slots ({' / '.join(str(s) for s in spare)}) take old refills from its start")
    return (f"{cfg['blocks']}-block schedule: a {cfg['wall_seconds']:,} s play wall in {cfg['blocks']} blocks of "
            f"{cfg['block_seconds']:,.0f} s; the roster in roster order split into blocks of "
            f"{' / '.join(str(s) for s in cfg['block_sizes'])} games ({cfg['slots']} streams); block 1 starts "
            f"{'staggered from' if cfg.get('stagger_seconds') else 'together at'} the release; no opening, no level budgets, no backlog ranking; tenure one block ({unit}); at a boundary "
            f"the block's own games park (block_end) and the next block's never-run games take their slots in roster "
            f"order; games pulled forward early (fillers) keep playing into their own block until their tenure ends "
            f"(tenure_end); a level-0 game retires for good at {cfg['level0_budget_tokens']:,} gameplay completion "
            f"tokens (level0_retired); a freed slot (retirement, finish, tenure end) takes the next never-run game in "
            f"roster order, else the parked old game with the most levels, then the fewest gameplay completion tokens "
            f"(old_refill), which then plays until the wall end" + stagger)
