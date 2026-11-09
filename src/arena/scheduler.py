"""The admission state machine: opening phase, the two-slot steady state, level budgets.

Rules:
- Opening: every game is active from the release; no level budgets. The moment the second
  distinct game reports level >= 1, every other game is parked and ``ACTIVE_LIMIT`` = 2 holds.
- Level budgets use the selected budget clock: T1 = budget time to level 1; level k -> k+1 gets (k+1) * T1
  from the moment level k was reached (the score weights levels 1, 2, 3, ...). An exhausted
  budget parks the game only while a backlog game exists.
- Re-admission grant after exhaustion, any level (level 0 included): max(1.5 * spent, spent + 30)
  (the 1.5 multiple is the study's ``regrant_multiple``; the 24 September variant uses 2)
  where ``spent`` is the budget seconds on the current level (level 0: all active seconds).
- Backlog priority: fewest budget seconds first, then roster order.
- Optional late-clear priority: a new level reached while parked earns one priority
  admission when a slot becomes available, without preempting active games.

Fixed-2x policy (2026; ``budget_policy="fixed-2x"``):
- The budget for playing level L (= levels cleared + 1) is 2 * L * T1, granted in full at
  every level clear and at every re-admission (never the remainder, never geometric).
- A game without a clear is re-admitted with the fixed amount it had spent when it was
  first parked (floored at the 30-second grant floor).
- Backlog order (``backlog_policy="readmissions"``): a clear that arrived while parked
  first, then fewest steady-state admissions, then most levels, then roster order.

Progress order (2026; ``backlog_policy="progress"``): the backlog is
ranked by an estimate of marginal score per busy second, highest first, and a slot always goes
to the top game (fixed-2x budgets unchanged):
  value = weight_next * rate * stall_penalty
  weight_next   = (levels + 1) / (N * (N + 1) / 2), N = the game's level count (8 if unknown):
                  the share of the game's score its next level carries;
  rate          = (levels + 0.5) / (busy seconds + 90): clears per busy second with a prior of
                  half a clear per level-0 grant, so untried games rank between stuck ones and
                  progressing ones;
  stall_penalty = 1 / (1 + actions_since_novel / 10).
A clear that arrived while parked still goes first; roster order breaks exact ties.

Audition pool (2026; ``audition_pool`` = games per free slot):
- In the steady state a free slot whose best backlog game has no clear does not audition one
  game at a time. Every game tied with it at the top of the order (same readmissions, same
  levels, no parked clear) is admitted at once, up to ``audition_pool`` per free slot, with
  no level budget; the KV gate bounds what actually runs.
- The first pool game to clear a level takes the slot as a focus game with its 2 * L * T1
  budget (``pool_clear``); pool games beyond the remaining pool capacity park (``pool_close``)
  with their readmission count unchanged, so they resume in the next pool.
- A pool game parks and drops one place in the order (its readmissions count) after
  ``pool_stall_actions`` consecutive actions without a board state new to the level
  (``pool_stall``) or ``pool_action_limit`` actions on the level (``pool_limit``).
- Budgets and T1 use the normalized clock (tokens at the two-stream rate plus tool time), so
  the pool's contention does not inflate T1 or shrink what a focus budget buys.

Token budgets (the default of every new build, ``budget_clock="tokens"``):
every rule above keeps its form, but the budget unit is completion tokens (``arena.clock.ActiveClock``:
gameplay replies, reasoning included) instead of busy seconds, so a budget buys the same model work at 6,
14 or 21 focus streams. T1 = tokens to the first clear; 2 * L * T1 tokens per level; the level-0
re-admission grant is the game's tokens at its first park. The time constants of the budget path get
token equivalents at one reference rate, ``TOKEN_REFERENCE_RATE`` = 64 completion tokens per busy second
(one focus game's phase-2 rate at six streams, kernels-v2 / hero-6x64k): the 30 s grant floor (the
level-0 floor, the opening-switch floor, the ``spent + 30`` regrant term, the T1-relative cap's floor) is
``TOKEN_FLOOR`` = 30 x 64 = 1,920 tokens, and the progress order's 90 s prior is 3 floors. The level-0
quantum and the other token settings were already tokens; the opening cap and the wall stay wall time.
Knobs defined in seconds (``max_grant_seconds``, ``t1_cap_seconds``, ``pool_time_limit_seconds``) are
refused on the token clock; their studies keep the clock they were run on.

Priority scheduling (``budget_policy="priority"``, ``arena.priority``; ported from dfranzen's
ARC3-Inference priority gate, Apache-2.0): no opening, no level budgets, no retirement. Unstarted games take the slots
in roster order; a game keeps its slot until its next context eviction (a summary compaction or a drain), where it is
re-scored and the best waiting game (unstarted first) takes the slot, which may be the same game.
"""

import math
import random
import statistics
from dataclasses import dataclass, field


def token_budget_settings():
    """The ``study["token_budget"]`` block of a token-clock study (build_arena and the hero-v2 factory write it; the
    trial reads ``purposes`` and ``floor_tokens``)."""
    from .clock import ActiveClock

    return {"unit": "completion_tokens", "purposes": list(ActiveClock.TOKEN_PURPOSES),
            "floor_tokens": int(Scheduler.TOKEN_FLOOR),
            "reference_tokens_per_busy_second": float(Scheduler.TOKEN_REFERENCE_RATE)}


@dataclass
class GameRecord:
    game_id: str
    index: int
    clock: object
    state: str = "backlog"  # backlog | active | parked | finished
    levels: int = 0
    total_levels: int | None = None  # the game's level count when the adapter knows it (progress order)
    t1: float | None = None
    level_started_active: float = 0.0
    level_started_budget: float = 0.0
    level_started_wall: float = 0.0
    time_to_level: list = field(default_factory=list)
    admissions: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    admission_attempts: list = field(default_factory=list)
    compactions: int = 0
    compaction_seconds: float = 0.0
    max_prompt_tokens: int = 0
    finished_reason: str | None = None
    error: str | None = None
    ready: object = None  # asyncio.Event set by the agent once its environment exists
    observer: object = None  # the Scheduler; ``report_level`` forwards to ``on_level``
    pending_parked_progress: bool = False
    readmissions: int = 0  # steady-state admissions (the opening-switch keep counts as one)
    level0_grant: float | None = None  # fixed-2x: budget seconds at the first park without a clear
    pool: bool = False  # active without a budget inside an audition pool
    pool_admissions: int = 0
    pool_clears: int = 0
    total_actions: int = 0
    actions_on_level: int = 0
    actions_since_novel: int = 0  # consecutive actions whose board the level had already shown
    frames_on_level: set = field(default_factory=set)
    stalls: int = 0
    admit_levels: int = 0  # levels held at the latest admission (Thompson backlog)
    # Hazard index (26 September): completion tokens at the current level's start and at the latest
    # admission, and the tokens each cleared level took.
    level_started_tokens: int = 0
    admit_tokens: int = 0
    level_tokens: list = field(default_factory=list)
    fruitless: int = 0  # admissions that ended in a park without a new clear (Thompson backlog)
    # Steady-state tenures on the current level that ended in a park without a clear (reset at every clear; the
    # progress-fruitless backlog order, hero-v2 run 2).
    fruitless_on_level: int = 0
    # Background lane (fixed-2x-index-lane, 26 September): a parked game outside the focus slots that may
    # still send requests on the capacity the focus games leave free (``arena.gate.BackgroundLane``).
    # Its clock stays parked; its requests and tokens are charged to it as usual.
    lane: bool = False
    latest_prompt_tokens: int = 0  # prompt tokens of the latest admitted request (the focus guard)
    lane_requests: int = 0
    lane_completion_tokens: int = 0  # settled at the game's next admission
    lane_open_tokens: int | None = None  # completion tokens when the latest lane request was admitted
    # Momentum commitment (fixed-2x-commit): demoted from a focus slot past its stall cap; only a clear
    # (which resets it) promotes it again.
    stalled_level: bool = False
    # Deep/explore (fixed-2x-deep-explore, 28 September): the slot the game holds ("deep" | "explore" | None), its
    # context window (the agent's compaction threshold follows it through ``window_listener``), the wall time of its
    # latest clear, whether it waits in the promotion queue, and completion tokens / inference seconds / requests per
    # slot kind, settled at every gate admission (a game's requests are sequential and a reply is charged before its
    # next admission, so what is charged between two admissions belongs to the earlier request's slot).
    slot: str | None = None
    window: int | None = None
    window_listener: object = None
    last_clear_wall: float | None = None
    deep_queued: bool = False
    slot_usage: dict = field(default_factory=dict)
    slot_open: tuple | None = None
    # Three-block schedule (hero-v2-blocks, 29 September; ``arena.blocks``): the game's home block (roster position),
    # its first admission (wall seconds and gameplay completion tokens), its current role ("original": admitted at
    # its own block's start; "filler": pulled forward before its block; "old": refilled after a park) and whether it
    # was retired on level 0 for good.
    home_block: int | None = None
    first_admit_wall: float | None = None
    first_admit_tokens: int | None = None
    role: str | None = None
    retired: bool = False
    # Retired-game continuation (the idling fix): a continued retired game's allowance counts from
    # its continuation (tokens already spent on the current level, or on level 0 for the level-0 budget), and how
    # many times it was continued (the rotation order).
    allowance_base: float = 0.0
    continuations: int = 0
    # Priority scheduling (1 October; ``arena.priority``): context drains reported by the agent, evictions seen by the
    # scheduler, and the queue arrival (sequence) and score of a game handed over.
    context_drains: int = 0
    context_evictions: int = 0
    priority_queued: int | None = None
    priority_queued_score: int | None = None
    # Level cap (9 October; ``priority.level_cap_tokens``): capped on the current level (reset by a clear), the tokens
    # on the level at which its present grant ends, and the run's gameplay tokens when it was last parked while capped.
    level_capped: bool = False
    level_cap_limit: float = 0.0
    level_cap_parked_total: float | None = None
    level_started_requests: int = 0
    level_cap_request_limit: int = 0

    def note_compaction(self):
        """Agents call this right after a summary compaction (``compactions`` already counted), before the next
        request: the cached prefix is gone, the one moment a priority handover is free. A no-op for other policies."""
        self._context_evicted("compaction")

    def note_context_drain(self, tokens_before=None, tokens_after=None):
        """Agents call this when they drain part of the context (e.g. drop the older half of the history) before the
        next request; the same handover point as a compaction. Safe without an observer and for every policy."""
        self.context_drains += 1
        self._context_evicted("drain", tokens_before=tokens_before, tokens_after=tokens_after)

    def _context_evicted(self, kind, **details):
        self.context_evictions += 1
        handler = getattr(self.observer, "on_context_eviction", None)
        if handler is not None:
            handler(self, kind, **details)

    def set_window(self, window):
        """Change the context window (the compaction threshold of the game's next request)."""
        window = None if window is None else int(window)
        if window == self.window:
            return
        self.window = window
        if self.window_listener is not None:
            self.window_listener(window)

    def note_slot_request(self):
        """The gate slot calls this at every admission (per-slot tokens and inference seconds)."""
        tokens, seconds = self.clock.completion_tokens(), self.clock.inference_seconds()
        self._settle_slot(tokens, seconds)
        self.slot_open = (self.slot or "none", tokens, seconds)

    def _settle_slot(self, tokens, seconds):
        if self.slot_open is None:
            return
        slot, tokens0, seconds0 = self.slot_open
        row = self.slot_usage.setdefault(slot, {"completion_tokens": 0, "inference_seconds": 0.0, "requests": 0})
        row["completion_tokens"] += tokens - tokens0
        row["inference_seconds"] += seconds - seconds0
        row["requests"] += 1

    def slot_usage_now(self):
        """``slot_usage`` including the request still open (charged so far)."""
        saved = ({k: dict(v) for k, v in self.slot_usage.items()}, self.slot_open)
        self._settle_slot(self.clock.completion_tokens(), self.clock.inference_seconds())
        result = self.slot_usage
        self.slot_usage, self.slot_open = saved
        return result

    def note_action(self, frame_key):
        """Adapters call this after every environment action with a key of the resulting board."""
        self.actions_on_level += 1
        self.total_actions += 1
        if frame_key in self.frames_on_level:
            self.actions_since_novel += 1
        else:
            self.frames_on_level.add(frame_key)
            self.actions_since_novel = 0
    on_park: object = None  # optional ``callback(reason)`` run when the scheduler parks the game

    def report_level(self, levels, wall_seconds, total_levels=None):
        """Adapters call this after every observation; new levels reach the scheduler."""
        if total_levels:
            self.total_levels = int(total_levels)
        before = self.levels
        self.note_level(levels, wall_seconds)
        if self.levels > before and self.observer is not None:
            self.observer.on_level(self)

    def note_level(self, levels, wall_seconds):
        """Record every level reached (the runner reports the count after each observation)."""
        levels = int(levels)
        if levels <= self.levels:
            return
        active = self.clock.active_seconds()
        budget_time = self.clock.budget_seconds()
        for level in range(self.levels + 1, levels + 1):
            self.time_to_level.append(
                {"level": level, "active_seconds": round(active, 3), "wall_seconds": round(wall_seconds, 3),
                 "budget_seconds": round(budget_time, 3)}
            )
        if self.t1 is None:
            self.t1 = max(budget_time, 1.0)
        tokens = self.clock.completion_tokens()
        self.level_tokens.append(tokens - self.level_started_tokens)
        self.level_tokens.extend([0] * (levels - self.levels - 1))
        self.level_started_tokens = tokens
        self.levels = levels
        self.fruitless_on_level = 0
        self.level_started_active = active
        self.level_started_budget = budget_time
        self.allowance_base = 0.0  # a new clear: a fresh allowance from the level's start
        self.level_started_requests = len(self.requests)
        self.level_started_wall = wall_seconds
        self.actions_on_level = 0
        self.actions_since_novel = 0
        self.frames_on_level = set()

    def note_request(self, lane):
        """The gate slot calls this at every admission with the lane (fixed-2x-index-lane). A game's
        requests are sequential and a reply is charged before the next admission, so the completion
        tokens charged between a lane admission and the next admission are that lane request's."""
        self.lane_completion_tokens = self.lane_tokens()
        self.lane_open_tokens = None
        if lane:
            self.lane_requests += 1
            self.lane_open_tokens = self.clock.completion_tokens()

    def lane_tokens(self):
        if self.lane_open_tokens is None:
            return self.lane_completion_tokens
        return self.lane_completion_tokens + self.clock.completion_tokens() - self.lane_open_tokens

    def note_finished(self, reason):
        if self.state != "finished":
            self.finished_reason = reason

    def spent_on_level(self):
        return self.clock.budget_seconds() - self.level_started_budget

    def requests_on_level(self):
        return len(self.requests) - self.level_started_requests

    def tokens_on_level(self):
        return self.clock.completion_tokens() - self.level_started_tokens


class Scheduler:
    OPENING_SWITCH_LEVEL1_GAMES = 2
    ACTIVE_LIMIT = 2
    REGRANT = 1.5
    REGRANT_FLOOR = 30.0
    # Token clock (29 September): the 30 s floor at the reference rate (one focus game at six streams, 64 completion
    # tokens per busy second; kernels-v2 61.5 and hero-6x64k 63.6 measured, docs/validation-20260927/token-budgets.md).
    TOKEN_REFERENCE_RATE = 64.0
    TOKEN_FLOOR = 1920

    def __init__(self, games, global_clock, log=None, *, active_limit=None, switch_games=None,
                 max_grant_seconds=None, prioritize_parked_progress=False, max_grant_t1_multiple=None,
                 regrant_multiple=None, budget_policy="incremental", backlog_policy="spent", progress_model="levels",
                 t1_cap_seconds=None, audition_pool=None, pool_stall_actions=None, pool_action_limit=None,
                 focus_stall_actions=None, opening_max_wall_seconds=None, level0_order="roster",
                 pool_time_limit_seconds=None, reopen_phase1=None, budget_multiple=None, thompson_seed=0,
                 t1_clip=None, index_settings=None, background_lane=None, commit=None, opening_min_tokens=None,
                 deep_explore=None, depthwise=None, rank=None, token_floor=None, backlog_key=None, blocks=None, level_token_allowance=None,
                 priority=None):
        self.games = list(games)
        self.global_clock = global_clock
        self._closed = False
        # The budget unit follows the games' clocks: completion tokens on the token clock, seconds otherwise.
        clocks = {getattr(g.clock, "budget_clock", None) for g in self.games}
        self.token_clock = "tokens" in clocks
        if self.token_clock and len(clocks) != 1:
            raise ValueError("Every game's clock must use the token budget clock or none may")
        if token_floor is not None and (not self.token_clock or type(token_floor) is not int or token_floor < 1):
            raise ValueError("token_floor needs the token budget clock and a positive integer")
        floor = self.TOKEN_FLOOR if token_floor is None else token_floor
        self.regrant_floor = float(floor) if self.token_clock else self.REGRANT_FLOOR
        # The progress order's prior spend (90 s = three 30 s floors) in the same unit.
        self.progress_prior = (self.PROGRESS_PRIOR_SECONDS / self.REGRANT_FLOOR * self.regrant_floor
                               if self.token_clock else self.PROGRESS_PRIOR_SECONDS)
        if self.token_clock and (max_grant_seconds is not None or t1_cap_seconds is not None
                                 or pool_time_limit_seconds is not None):
            raise ValueError("max_grant_seconds, t1_cap_seconds and pool_time_limit_seconds are seconds; "
                             "the token budget clock refuses them")
        if budget_multiple is not None and (
            budget_policy != "fixed-2x" or type(budget_multiple) not in (int, float)
            or not math.isfinite(budget_multiple) or budget_multiple <= 0
        ):
            raise ValueError("budget_multiple requires the fixed-2x policy and a positive finite number")
        # Timing variants (26 September): the level budget is multiple * L * T1 (control 2).
        self.budget_multiple = 2.0 if budget_multiple is None else float(budget_multiple)
        self.log = log
        self.active_limit = active_limit or self.ACTIVE_LIMIT
        self.switch_games = switch_games or self.OPENING_SWITCH_LEVEL1_GAMES
        if max_grant_seconds is not None and (
            type(max_grant_seconds) not in (int, float) or not math.isfinite(max_grant_seconds)
            or max_grant_seconds < self.REGRANT_FLOOR
        ):
            raise ValueError("max_grant_seconds must be finite and at least the 30-second grant floor")
        self.max_grant_seconds = max_grant_seconds
        if max_grant_t1_multiple is not None and (
            type(max_grant_t1_multiple) not in (int, float) or not math.isfinite(max_grant_t1_multiple)
            or max_grant_t1_multiple < 1
        ):
            raise ValueError("max_grant_t1_multiple must be a finite number of at least 1")
        # Renewable T1-relative cap (24 September): every steady-state allocation of a game
        # that has cleared a level is at most ``multiple * T1`` (never below the grant floor);
        # exhausted games rotate through the backlog and are re-admitted with the regrant rule.
        self.max_grant_t1_multiple = max_grant_t1_multiple
        if regrant_multiple is not None and (
            type(regrant_multiple) not in (int, float) or not math.isfinite(regrant_multiple)
            or regrant_multiple < 1
        ):
            raise ValueError("regrant_multiple must be a finite number of at least 1")
        # Re-admission multiple (24 September): a game whose budget ran out is re-admitted
        # with max(multiple * spent, spent + floor); the control multiple is 1.5.
        self.regrant_multiple = self.REGRANT if regrant_multiple is None else regrant_multiple
        if backlog_policy == "thompson" and budget_policy != "fixed-2x":
            raise ValueError("The Thompson backlog requires the fixed-2x policy")
        # Thompson backlog (26 September, paper notes 2.4): per waiting game a draw from
        # Beta(1 + levels cleared, 1 + admissions that ended without a clear), damped by the
        # actions since the level last showed a new board; seeded for reproducibility.
        self.rng = random.Random(thompson_seed)
        if backlog_policy == "llm" and budget_policy != "fixed-2x":
            raise ValueError("The LLM-ranked backlog requires the fixed-2x policy")
        # LLM-ranked backlog (26 September): game id -> position in the model's latest ranking.
        self.llm_rank = {}
        self.llm_rankings = 0
        # Hazard index (26 September): after the opening, slots go to the games with the highest
        # expected score per token under a hazard model estimated online from this run
        # (``arena.hazard_index``); no level budgets.
        self.index_cfg = None
        self.index_model = None
        self.index_history = []
        self.index_swaps = 0
        if budget_policy == "index":
            if backlog_policy != "readmissions":
                raise ValueError("The hazard index replaces the backlog order; keep the readmissions default")
            if (index_settings or {}).get("model") == "gittins":
                # Bayesian Gittins-style index (fixed-2x-gittins-broad, 27 September; ``arena.gittins``).
                from .gittins import GittinsModel
                from .gittins import settings as index_config

                self.index_model = GittinsModel
            else:
                from .hazard_index import RunModel
                from .hazard_index import settings as index_config

                self.index_model = RunModel
            self.index_cfg = index_config(index_settings)
        elif index_settings is not None:
            raise ValueError("index_settings require the index budget policy")
        # Background lane (26 September, fixed-2x-index-lane): after the opening every parked game is
        # in the lane (``GameRecord.lane``); the gate admits its requests on leftover capacity.
        self.background_lane = None
        self.lane_clears = []
        self.lane_promotions = []  # lane games that took a focus slot
        if background_lane is not None:
            from .gate import lane_settings

            if budget_policy != "index":
                raise ValueError("The background lane requires the index budget policy")
            self.background_lane = lane_settings(background_lane)
        # Momentum commitment (26 September, fixed-2x-commit; ``arena.commit``): after the opening, focus
        # slots go to games that clear levels and are kept up to a stall cap on tokens per level; every
        # other game plays in the preemptible lane (``GameRecord.lane``); no level budgets.
        self.commit = None
        self.commit_promotions = []
        self.commit_demotions = []
        self.commit_queue = []  # lane clears waiting for a slot (commit-queue: no replacement of unstalled focus games)
        if budget_policy == "commit":
            from .commit import commit_settings

            if backlog_policy != "readmissions" or background_lane is not None:
                raise ValueError("The commit policy keeps the readmissions default and has its own lane")
            self.commit = commit_settings(commit)
            self.active_limit = self.commit["focus_slots"]
        elif commit is not None:
            raise ValueError("commit settings require the commit budget policy")
        # Deep/explore (28 September, fixed-2x-deep-explore; ``arena.deep_explore``): no opening, no level budgets;
        # deep slots hold the games that clear, exploration slots rotate the rest on a token quantum.
        self.deep = None
        self.deep_decisions = []  # every promotion, demotion, queueing and rotation, with its evidence
        self.deep_tenures = []  # one row per deep-slot tenure
        self.deep_queue = []  # the promotion queue (kept sorted by ``_queue_key`` when read)
        self.deep_clears = {"deep": 0, "explore": 0, "none": 0}
        if budget_policy == "deep-explore":
            from .deep_explore import settings as deep_settings

            if backlog_policy != "readmissions" or background_lane is not None or commit is not None:
                raise ValueError("The deep-explore policy keeps the readmissions default and has no lane")
            self.deep = deep_settings(deep_explore)
            self.active_limit = self.deep["deep_slots"] + self.deep["explore_slots"]
            for game in self.games:
                game.set_window(self.deep["explore_window"])
        elif deep_explore is not None:
            raise ValueError("deep_explore settings require the deep-explore budget policy")
        # Depthwise (28 September, fixed-2x-depthwise; ``arena.depthwise``): no opening; level-0 games rotate through
        # any free slot at a small window on a token quantum; a game that clears keeps its slot at the full window
        # under fixed-2x budgets; freed slots go to parked depth games (fastest T1 first) before the rotation.
        self.depthwise = None
        self.dw_decisions = []  # every rotation, clear, park, keep and re-admission, with its evidence
        self.dw_tenures = []  # one row per depth tenure (a game holding a slot with >= 1 level)
        self.dw_clears = {"depth": 0, "level0": 0, "none": 0}
        self.dw_kept = set()  # depth games kept past an exhausted budget (nobody waiting) since their latest grant
        if budget_policy == "depthwise":
            from .depthwise import settings as depthwise_settings

            if (backlog_policy != "readmissions" or background_lane is not None or commit is not None
                    or deep_explore is not None or audition_pool is not None or reopen_phase1 is not None):
                raise ValueError("The depthwise policy keeps the readmissions default and has no lane or pool")
            self.depthwise = depthwise_settings(depthwise)
            self.active_limit = self.depthwise["slots"]
            for game in self.games:
                game.set_window(self.depthwise["level0_window"])
        elif depthwise is not None:
            raise ValueError("depthwise settings require the depthwise budget policy")
        # Continuous ranking (29 September, hero-v2-rank; ``arena.rank``): no opening, no level budgets; every
        # tick_seconds every game is re-scored by its recency-weighted clears per token x its next level's value, and
        # the focus slots refill from the ranking with a hysteresis and a token tenure.
        self.rank = None
        self.rank_last_tick = None
        self.rank_order = []
        self.rank_counts = {"ticks": 0, "fills": 0, "swaps": 0, "refills": 0}
        self.rank_history = []
        self.rank_swaps = []
        if budget_policy == "rank":
            from .rank import settings as rank_settings

            if (backlog_policy != "readmissions" or background_lane is not None or commit is not None
                    or deep_explore is not None or depthwise is not None or audition_pool is not None
                    or reopen_phase1 is not None):
                raise ValueError("The rank policy keeps the readmissions default and has no lane or pool")
            self.rank = rank_settings(rank)
            self.active_limit = self.rank["slots"]
        elif rank is not None:
            raise ValueError("rank settings require the rank budget policy")
        # Three-block schedule (29 September, hero-v2-blocks; ``arena.blocks``): no opening, no level budgets; blocks of
        # the roster play one block length each, level-0 games retire at a token budget, freed slots take the next
        # never-run game, else the best parked old game.
        self.level_token_allowance = level_token_allowance
        if level_token_allowance is not None:
            if type(level_token_allowance) is not int or level_token_allowance < 1 or budget_policy != "fixed-2x":
                raise ValueError("level_token_allowance needs fixed-2x and a positive integer")
            if any(g.clock.budget_clock != "tokens" for g in self.games):
                raise ValueError("independent level allowances require gameplay token clocks")
        self.blocks = None
        self.blocks_index = 0
        self.blocks_decisions = []
        self.blocks_counts = {"block_start": 0, "filler": 0, "old_refill": 0, "block_end": 0, "tenure_end": 0,
                              "level0_retired": 0, "level_retired": 0, "finished": 0, "retired_level": 0,
                              "retired_continue": 0}
        self.blocks_idle = {"slot_seconds": 0.0, "slot_seconds_with_eligible": 0.0, "last_tick": None, "free": 0,
                            "eligible": 0}
        if budget_policy == "blocks":
            from .blocks import home_blocks
            from .blocks import settings as blocks_settings

            if (backlog_policy != "readmissions" or background_lane is not None or commit is not None
                    or deep_explore is not None or depthwise is not None or rank is not None
                    or audition_pool is not None or reopen_phase1 is not None or backlog_key is not None):
                raise ValueError("The blocks policy keeps the readmissions default and has no lane, pool or ranking")
            if not self.token_clock:
                raise ValueError("The blocks policy counts its level-0 budget on the token budget clock")
            self.blocks = blocks_settings(blocks, games=len(self.games))
            self.active_limit = self.blocks["slots"]
            for game, block in zip(self.games, home_blocks(len(self.games), self.blocks), strict=True):
                game.home_block = block
        elif blocks is not None:
            raise ValueError("blocks settings require the blocks budget policy")
        # Priority scheduling (1 October; ``arena.priority``): slots are the focus streams; unstarted games first in
        # roster order, then handovers at context evictions to the best-scored waiting game; no level budgets.
        self.priority_cfg = None
        self.priority_seq = 0
        self.priority_decisions = []
        self.priority_counts = {"first_window": 0, "handover": 0, "handover_kept": 0, "finished": 0,
                                "compaction": 0, "drain": 0, "eviction_not_active": 0}
        # Level cap: levels capped, capped games parked for an uncapped one, returns to an otherwise idle slot, grants
        # extended in place (nobody better waited), rotations between capped games, caps lifted by a clear, and one
        # slot's share of the run's gameplay tokens while capped games sat parked (the tokens the cap freed).
        self.level_cap_counts = {"capped": 0, "parked": 0, "returns": 0, "kept": 0, "rotations": 0, "lifted": 0}
        self.level_cap_freed = 0.0
        self.priority_fade_wall = None
        if budget_policy == "priority":
            from .priority import settings as priority_settings

            if (backlog_policy != "readmissions" or background_lane is not None or commit is not None
                    or deep_explore is not None or depthwise is not None or rank is not None or blocks is not None
                    or audition_pool is not None or reopen_phase1 is not None or backlog_key is not None
                    or focus_stall_actions is not None or opening_max_wall_seconds is not None
                    or opening_min_tokens is not None):
                raise ValueError("The priority policy keeps the readmissions default and has no lane, pool, ranking, "
                                 "blocks, stall parking or opening")
            self.priority_cfg = priority_settings(priority)
            if self.priority_cfg["level_cap_tokens"] and not self.token_clock:
                raise ValueError("priority.level_cap_tokens needs the token budget clock")
        elif priority is not None:
            raise ValueError("priority settings require the priority budget policy")
        if budget_policy not in ("incremental", "fixed-2x", "index", "commit", "deep-explore", "depthwise",
                                 "rank", "blocks", "priority") or backlog_policy not in ("spent", "readmissions", "progress", "llm", "thompson"):
            raise ValueError("Unknown budget or backlog policy")
        if budget_policy == "fixed-2x" and (max_grant_seconds is not None or max_grant_t1_multiple is not None
                                            or regrant_multiple is not None):
            raise ValueError("The fixed-2x policy has no caps or regrant multiple")
        if t1_cap_seconds is not None and (
                budget_policy != "fixed-2x" or type(t1_cap_seconds) not in (int, float)
                or not math.isfinite(t1_cap_seconds) or t1_cap_seconds < 1):
            raise ValueError("t1_cap_seconds requires the fixed-2x policy and at least one second")
        self.t1_cap_seconds = t1_cap_seconds
        # Clipped budget unit (26 September budget audit): T1 enters the level budgets clipped to
        # [low, high] x the median T1 of the games that have cleared level 1, so an easy first level
        # no longer means short budgets later (sb26 29 s, ar25 74 s) and a first clear that took
        # several admissions no longer earns a runaway grant (sp80 662 s -> 2,648 s at level 2).
        if t1_clip is not None:
            low, high = (float(v) for v in t1_clip)
            if budget_policy != "fixed-2x" or not (0 < low <= 1 <= high) or not math.isfinite(high):
                raise ValueError("t1_clip requires the fixed-2x policy and 0 < low <= 1 <= high")
            t1_clip = (low, high)
        self.t1_clip = t1_clip
        for label, value in (("audition_pool", audition_pool), ("pool_stall_actions", pool_stall_actions),
                             ("pool_action_limit", pool_action_limit)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{label} must be a positive integer")
        if audition_pool is not None and (budget_policy != "fixed-2x" or backlog_policy != "readmissions"):
            raise ValueError("The audition pool requires the fixed-2x policy and the readmissions order")
        if audition_pool is None and (pool_stall_actions is not None or pool_action_limit is not None):
            raise ValueError("Pool stall rules require an audition pool")
        self.audition_pool = audition_pool
        self.pool_stall_actions = pool_stall_actions
        self.pool_action_limit = pool_action_limit
        # Pool time limit (25 September, after pool2-cap v1): an audition bounded only by actions
        # held a pool place for 540 s at xhigh without stalling; a budget on the game's clock
        # parks it (``pool_timeout``, one readmission) like a stalled audition.
        if pool_time_limit_seconds is not None and (
                audition_pool is None or type(pool_time_limit_seconds) not in (int, float)
                or not math.isfinite(pool_time_limit_seconds) or pool_time_limit_seconds <= 0):
            raise ValueError("pool_time_limit_seconds requires an audition pool and a positive number")
        self.pool_time_limit_seconds = pool_time_limit_seconds
        # Level-0 backlog order (25 September, after the stall-open360 repeats): within a tie on
        # readmissions and levels, ``novelty`` admits the level-0 game that has shown the most
        # distinct boards first (its opening actions changed the board), instead of roster order.
        # ``tokens`` (hero-v2 run 2, 29 September; docs/validation-20260927/hero-v2-next-runs.md): the level-0 game
        # with the most gameplay completion tokens on level 0 first (an in-run signal: this run's own level-0 clear
        # hazard rises with the tokens spent on the level up to ~20K; roster order breaks exact ties).
        if level0_order not in ("roster", "novelty", "tokens"):
            raise ValueError("level0_order must be 'roster', 'novelty' or 'tokens'")
        if level0_order != "roster" and backlog_policy != "readmissions":
            raise ValueError("level0_order applies to the readmissions backlog order")
        self.level0_order = level0_order
        # Progress-fruitless order: the readmissions order let a
        # deep game parked twice wait behind a level-0 game auditioned once, and let second auditions of failed
        # level-0 games pass deep games. ``backlog_key="progress-fruitless"`` orders the fixed-2x backlog by
        #   1. progress: a level cleared since the game's last admission (in its tenure or while parked);
        #   2. the fewest fruitless tenures on the current level (steady-state parks without a clear since the
        #      admission; reset at every clear), so stuck games sink as their count grows;
        #   3. more levels; 4. the level-0 tie-break (``level0_order``); 5. roster order.
        # A never-auditioned level-0 game (0 fruitless) ties with a deep game on its first fruitless-free return. The
        # opening-switch keep counts as the kept game's admission at its level. A re-admission in the same tick as the
        # game's park is logged "regrant_same_tick" (park + re-admission stay; the fruitless count is the brake).
        if backlog_key not in (None, "progress-fruitless") or (backlog_key is not None and (
                backlog_policy != "readmissions" or budget_policy != "fixed-2x" or audition_pool is not None
                or reopen_phase1 is not None)):
            raise ValueError("backlog_key 'progress-fruitless' needs the fixed-2x readmissions order (no audition "
                             "pool, no reopened phase 1)")
        self.backlog_key = backlog_key
        self._parked_this_tick = set()
        self.same_tick_regrants = 0
        # Stall parking and the opening cap (25 September variant): a focus game that makes
        # ``focus_stall_actions`` consecutive actions without a board its level has not shown parks
        # and drops one place; the opening ends at the switch clears or at ``opening_max_wall_seconds``.
        if focus_stall_actions is not None and (type(focus_stall_actions) is not int or focus_stall_actions < 1):
            raise ValueError("focus_stall_actions must be a positive integer")
        if opening_max_wall_seconds is not None and (
                type(opening_max_wall_seconds) not in (int, float) or not math.isfinite(opening_max_wall_seconds)
                or opening_max_wall_seconds <= 0):
            raise ValueError("opening_max_wall_seconds must be a positive number")
        self.focus_stall_actions = focus_stall_actions
        self.opening_max_wall_seconds = opening_max_wall_seconds
        # Broad opening (fixed-2x-gittins-broad, 27 September): besides the switch clears, every game still
        # playing must have spent ``opening_min_tokens`` completion tokens before the switch (the opening cap
        # still ends it). Short contexts decode at ~310 tok/s against ~230 in phase two, and at the two-clear
        # switch 21 games hold ~5K tokens, inside the warm-up band where a level rarely clears.
        if opening_min_tokens is not None and (type(opening_min_tokens) is not int or opening_min_tokens < 1):
            raise ValueError("opening_min_tokens must be a positive integer")
        self.opening_min_tokens = opening_min_tokens
        self.switch_reason = None
        self.budget_policy = budget_policy
        self.backlog_policy = backlog_policy
        # Reopen phase 1: when a focus slot frees, every backlog
        # game tied at the top of the order (readmissions, levels) plays at once with no budget, as in
        # the opening; the first of them to clear a level takes the slot as a focus game and the others
        # are parked with their order unchanged. "pause" holds the surviving focus game(s) during the
        # reopened phase and resumes them with the budget they had; "run" keeps them playing.
        if reopen_phase1 not in (None, "pause", "run"):
            raise ValueError("reopen_phase1 must be None, 'pause' or 'run'")
        if reopen_phase1 is not None and (audition_pool is not None or budget_policy != "fixed-2x"
                                         or backlog_policy != "readmissions"):
            raise ValueError("reopen_phase1 needs fixed-2x budgets, the readmissions order and no audition pool")
        self.reopen_phase1 = reopen_phase1
        self.reopens = 0
        if progress_model not in ("levels", "novelty"):
            raise ValueError("progress_model must be levels or novelty")
        self.progress_model = progress_model
        # The readmissions order always promotes a clear that arrived while parked.
        self.prioritize_parked_progress = prioritize_parked_progress or backlog_policy in ("readmissions", "progress", "llm", "thompson")
        self.phase = "opening"
        self.switch_wall_seconds = None
        self.events = []

    # --- queries -------------------------------------------------------------------------

    def active(self):
        return [g for g in self.games if g.state == "active"]

    def focus(self):
        return [g for g in self.games if g.state == "active" and not g.pool]

    def pool_games(self):
        return [g for g in self.games if g.state == "active" and g.pool]

    def held(self):
        return [g for g in self.games if g.state == "held"]

    def reopen_active(self):
        return self.reopen_phase1 is not None and bool(self.pool_games())

    def _order_key(self, game):
        return self.priority(game)[:3]  # the tie group ignores novelty and roster order

    def backlog(self):
        return sorted((g for g in self.games if g.state in ("backlog", "parked") and not g.retired), key=self.priority)

    PROGRESS_PRIOR_CLEARS = 0.5
    PROGRESS_PRIOR_SECONDS = 90.0
    PROGRESS_STALL_SCALE = 10.0
    PROGRESS_DEFAULT_LEVELS = 8
    PROGRESS_NOVELTY_PRIOR = 3  # pseudo-actions in the distinct-board fraction

    def progress_value(self, game):
        """Estimated marginal score per busy second (``backlog_policy="progress"``).

        ``progress_model="levels"`` (progress v1/v2, 25 September): the next level's share of the
        game's score times clears per busy second. On a level-0 backlog that share is the only
        differentiator, so six-level games precede eight-level ones and the fast eight-level games
        are reached last (progress v2 never re-admitted lp85). ``progress_model="novelty"`` drops
        the level share and adds the level's distinct-board fraction (distinct boards shown over
        actions taken plus a prior) to the clears in the rate: a game whose actions keep
        producing new states is the one making progress towards its first clear.
        """
        rate_clears = game.levels + self.PROGRESS_PRIOR_CLEARS
        if self.progress_model == "novelty":
            weight_next = 1.0
            rate_clears += len(game.frames_on_level) / (game.actions_on_level + self.PROGRESS_NOVELTY_PRIOR)
        else:
            total = game.total_levels or self.PROGRESS_DEFAULT_LEVELS
            weight_next = (game.levels + 1) / (total * (total + 1) / 2)
        rate = rate_clears / (game.clock.budget_seconds() + self.progress_prior)
        stall_penalty = 1.0 / (1.0 + game.actions_since_novel / self.PROGRESS_STALL_SCALE)
        return weight_next * rate * stall_penalty

    def priority(self, game):
        preferred = self.prioritize_parked_progress and game.pending_parked_progress
        if self.backlog_policy == "progress":
            return (not preferred, -self.progress_value(game), game.index)
        if self.backlog_policy == "thompson":
            draw = self.rng.betavariate(1 + game.levels, 1 + game.fruitless)
            draw /= 1.0 + game.actions_since_novel / self.PROGRESS_STALL_SCALE
            return (not preferred, -draw, game.index)
        if self.backlog_policy == "llm":
            # The model's latest order; unranked games (before the first ranking, or left out)
            # follow in the readmissions order.
            rank = self.llm_rank.get(game.game_id, len(self.games))
            return (not preferred, rank, game.readmissions, -game.levels, game.index)
        if self.backlog_policy == "readmissions":
            novelty = 0
            if game.levels == 0 and self.level0_order == "novelty":
                novelty = len(game.frames_on_level)
            elif game.levels == 0 and self.level0_order == "tokens":
                novelty = game.tokens_on_level()
            if self.backlog_key == "progress-fruitless":
                progress = preferred or game.pending_parked_progress or game.levels > game.admit_levels
                return (not progress, game.fruitless_on_level, -game.levels, -novelty, game.index)
            return (not preferred, game.readmissions, -game.levels, -novelty, game.index)
        return (not preferred, game.clock.budget_seconds(), game.index)

    def set_llm_ranking(self, game_ids):
        """Install a ranking from ``arena.llm_rank`` (waiting games, best first)."""
        self.llm_rank = {game_id: position for position, game_id in enumerate(game_ids)}
        self.llm_rankings += 1
        self.events.append({"game": None, "event": "llm_rank", "reason": "ranking",
                            "wall_seconds": round(self.global_clock.elapsed(), 3),
                            "ranking": [g[:4] for g in game_ids]})

    def level_budget(self, game):
        """Fixed-2x: the full budget for the level the game is playing now.

        With ``t1_cap_seconds`` (25 September variant) a long first clear does not earn a
        proportionally long budget: T1 enters the formula capped.
        """
        if self.level_token_allowance is not None:
            return self.level_token_allowance * (game.levels + 1)
        return self.budget_multiple * (game.levels + 1) * self.budget_unit(game)

    def budget_unit(self, game):
        """The T1 that enters the level budgets (capped and/or clipped to the median T1)."""
        t1 = game.t1 if self.t1_cap_seconds is None else min(game.t1, self.t1_cap_seconds)
        if self.t1_clip is not None:
            median = statistics.median(g.t1 for g in self.games if g.t1 is not None)
            low, high = self.t1_clip
            t1 = min(max(t1, low * median), high * median)
        return t1

    # --- transitions ---------------------------------------------------------------------

    def _record(self, game, event, reason, budget=None, extra=None):
        entry = {
            "event": event,
            "reason": reason,
            "wall_seconds": round(self.global_clock.elapsed(), 3),
            "active_seconds": round(game.clock.active_seconds(), 3),
            "budget_seconds": round(game.clock.budget_seconds(), 3),
            "levels": game.levels,
            "budget": None if budget is None else round(budget, 3),
        }
        if self.blocks is not None:
            # hero-v2-blocks: every event carries the scheduler's block, the game's home block and role.
            entry.update(block=self.blocks_index, home_block=game.home_block, role=game.role)
        if extra:
            entry.update(extra)
        game.admissions.append(entry)
        self.events.append({"game": game.game_id, **entry})
        if self.log:
            self.log({"game": game.game_id, **entry})

    def start_all(self):
        """Opening: every game active with no budget."""
        if self._closed:
            return
        if self.deep is not None:
            # No opening phase: the exploration slots run from the release; the deep slot fills at the first clear.
            self.phase = "steady"
            self.switch_reason = "deep_explore_start"
            self.switch_wall_seconds = round(self.global_clock.elapsed(), 3)
            for game in self.games:
                game.clock.pause()  # (the trial pauses every clock before the release already)
            self._explore_fill("explore_start")
            return
        if self.depthwise is not None:
            # No opening phase: every slot takes a level-0 game at the release (never played, roster order).
            self.phase = "steady"
            self.switch_reason = "depthwise_start"
            self.switch_wall_seconds = round(self.global_clock.elapsed(), 3)
            for game in self.games:
                game.clock.pause()
            self._dw_fill("level0_start")
            return
        if self.blocks is not None:
            # No opening phase: block 1's games start together at the release (roster order).
            self.phase = "steady"
            self.switch_reason = "blocks_start"
            self.switch_wall_seconds = round(self.global_clock.elapsed(), 3)
            for game in self.games:
                game.clock.pause()
            self.blocks_index = 0
            self._blocks_fill()
            return
        if self.priority_cfg is not None:
            # No opening phase: the slots take the first games in roster order (their first windows).
            self.phase = "steady"
            self.switch_reason = "priority_start"
            self.switch_wall_seconds = round(self.global_clock.elapsed(), 3)
            for game in self.games:
                game.clock.pause()
            self._priority_fill("release")
            return
        if self.rank is not None:
            # No opening phase: the focus slots take the best-ranked games at the release (all unplayed: the prior
            # rate times each game's next-level value, then roster order).
            self.phase = "steady"
            self.switch_reason = "rank_start"
            self.switch_wall_seconds = round(self.global_clock.elapsed(), 3)
            for game in self.games:
                game.clock.pause()
            self._rank_tick(fill_reason="rank_start")
            return
        for game in self.games:
            game.state = "active"
            game.clock.grant(None)
            game.clock.resume()
            self._record(game, "admit", "opening_start")

    def park(self, game, reason, extra=None):
        game.state = "parked"
        game.pool = False
        game.clock.pause()
        if (self.background_lane is not None or self.commit is not None) and self.phase == "steady":
            game.lane = True  # a game that loses (or never gets) a focus slot joins the lane
        if game.levels == 0 and game.level0_grant is None:
            game.level0_grant = max(game.clock.budget_seconds(), self.regrant_floor)
        if game.levels == game.admit_levels:
            game.fruitless += 1
            if self.backlog_key is not None and self.phase == "steady" and reason != "opening_switch":
                game.fruitless_on_level += 1
        if self.backlog_key is not None:
            self._parked_this_tick.add(game.game_id)
        self._record(game, "park", reason, game.clock.budget, extra)
        if game.on_park is not None:
            game.on_park(reason)

    def admit(self, game, reason, budget, extra=None):
        game.pending_parked_progress = False
        game.admit_levels = game.levels
        game.readmissions += 1
        game.state = "active"
        game.lane = False
        game.clock.grant(budget)
        game.clock.resume()
        self._record(game, "admit", reason, budget, extra)

    def cap_grant(self, seconds, game=None):
        """Optional experimental caps per allocation; the control is unchanged."""
        if self.max_grant_seconds is not None:
            seconds = min(seconds, self.max_grant_seconds)
        if self.max_grant_t1_multiple is not None and game is not None and game.t1 is not None:
            seconds = min(seconds, max(self.max_grant_t1_multiple * game.t1, self.regrant_floor))
        return seconds

    def finish(self, game, reason):
        game.pending_parked_progress = False
        was = game.state
        game.state = "finished"
        game.finished_reason = reason
        game.lane = False
        game.clock.pause()
        if self.blocks is not None:
            # hero-v2-blocks: "finished" frees the slot (the game's own reason, e.g. won, is kept beside it).
            self.blocks_counts["finished"] += 1
            self._blocks_note(game, "finished", finished_reason=reason, was=was)
            self._record(game, "finish", "finished", extra={"finished_reason": reason, "was": was})
            return
        if self.priority_cfg is not None:
            self._level_cap_settle(game)
            # Priority: "finished" frees the slot (the game's own reason is kept beside it); the tick refills it.
            self._record(game, "finish", "finished", extra={"finished_reason": reason, "was": was})
            return
        self._record(game, "finish", reason)

    KNOB_CAP = 64  # max_num_seqs of the served stacks; the KV gate still admits by blocks

    def _live_knobs(self):
        """Live admission knob (checklist item 5, 30 Oct): {"active_limit": N} in $ARENA_KNOBS, re-read on change."""
        import json
        import os

        path = os.environ.get("ARENA_KNOBS")
        if not path:
            return
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return
        if mtime == getattr(self, "_knob_mtime", None):
            return
        self._knob_mtime = mtime
        try:
            with open(path) as handle:
                want = int(json.load(handle)["active_limit"])
        except Exception as exc:  # noqa: BLE001 - a bad knob file never stops play
            if self.log:
                self.log({"event": "knobs", "error": repr(exc)[:200]})
            return
        new = max(1, min(want, self.KNOB_CAP))
        if new != self.active_limit:
            old, self.active_limit = self.active_limit, new
            if self.log:
                self.log({"event": "knobs", "active_limit": new, "was": old, "requested": want})

    def tick(self):
        """Apply the rules once; safe to call every second and after level or finish events."""
        if self._closed:
            return
        self._live_knobs()
        for game in self.games:
            if game.state != "finished" and game.finished_reason is not None:
                self.finish(game, game.finished_reason)
        if self.level_token_allowance is not None:
            for game in self.games:
                if (game.state in ("active", "parked") and not game.retired
                        and game.spent_on_level() >= self.level_budget(game)):
                    game.retired = True
                    if game.state == "active":
                        self.park(game, "level_retired")
                    else:
                        self._record(game, "progress", "level_retired")
        if self.phase == "opening":
            reached = [g for g in self.games if g.levels >= 1]
            capped = (self.opening_max_wall_seconds is not None
                      and self.global_clock.elapsed() >= self.opening_max_wall_seconds)
            broad = self.opening_min_tokens is None or all(
                g.clock.completion_tokens() >= self.opening_min_tokens for g in self.games if g.state != "finished")
            if (len(reached) >= self.switch_games and broad) or capped:
                self.phase = "steady"
                self.switch_reason = "clears" if len(reached) >= self.switch_games and broad else "opening_cap"
                self.switch_wall_seconds = round(self.global_clock.elapsed(), 3)
                eligible = [g for g in reached if g.state != "finished" and not g.retired]
                keep = sorted(eligible, key=lambda g: (g.time_to_level[0]["budget_seconds"], g.index))[: self.active_limit]
                if self.opening_min_tokens is not None and self.index_cfg is not None:
                    # After a broad opening every game has evidence: keep the highest indexes.
                    model = self.index_model(self.games, self.index_cfg)
                    remaining = self.global_clock.remaining()
                    live = [g for g in self.games if g.state == "active"]
                    keep = sorted(live, key=lambda g: (-model.index(g, remaining)[0], g.index))[: self.active_limit]
                for game in self.games:
                    if game.state == "active" and game not in keep:
                        self.park(game, "opening_switch")
                for game in keep:
                    if game.state == "active" and self.budget_policy in ("index", "commit"):
                        game.readmissions += 1
                        game.admit_tokens = game.clock.completion_tokens()
                        game.clock.grant(None)
                        self._record(game, "grant", "opening_switch")
                    elif game.state == "active":
                        if self.backlog_key is not None:
                            game.admit_levels = game.levels  # the keep is the kept game's admission at its level
                        # The focus games start their level budget from the level they hold.
                        full = self.level_budget(game) if self.budget_policy == "fixed-2x" else (game.levels + 1) * game.t1
                        budget = full - game.spent_on_level()
                        game.readmissions += 1
                        game.clock.grant(self.cap_grant(max(budget, self.regrant_floor), game))
                        self._record(game, "grant", "opening_switch", game.clock.budget)
            return
        if self.budget_policy == "index":
            self._index_tick()
            return
        if self.budget_policy == "commit":
            self._commit_tick()
            return
        if self.budget_policy == "deep-explore":
            self._deep_tick()
            return
        if self.budget_policy == "depthwise":
            self._dw_tick()
            return
        if self.budget_policy == "rank":
            self._rank_tick()
            return
        if self.budget_policy == "blocks":
            self._blocks_tick()
            return
        if self.budget_policy == "priority":
            self._priority_tick()
            return
        self._parked_this_tick = set()
        backlog = self.backlog()
        for game in self.focus():
            if game.clock.budget_exhausted() and backlog:
                self.park(game, "level_budget")
                backlog = self.backlog()
            elif (self.focus_stall_actions is not None and backlog
                  and game.actions_since_novel >= self.focus_stall_actions):
                game.stalls += 1
                game.readmissions += 1
                self.park(game, "stall")
                backlog = self.backlog()
        if self.audition_pool is not None:
            self._pool_stalls()
            self._settle_pool()
        if self.reopen_phase1 is not None:
            if self.reopen_active():
                return  # the reopened phase 1 runs until one of its games clears a level
            free = self.active_limit - len(self.focus())
            if free > 0 and backlog:
                group = [g for g in backlog if self._order_key(g) == self._order_key(backlog[0])]
                if len(group) > 1:
                    self._reopen(group)
                    return
        while len(self.focus()) < self.active_limit and backlog:
            game = backlog[0]
            if self.audition_pool is not None and game.levels == 0 and not game.pending_parked_progress:
                self._fill_pool(backlog)
                return
            backlog.pop(0)
            budget = self.grant_for(game)
            reason = "level_budget" if (game.levels >= 1 and game.spent_on_level() <= 0.0) else "regrant"
            if self.prioritize_parked_progress and game.pending_parked_progress:
                reason = "parked_progress"
            elif game.game_id in self._parked_this_tick:
                reason = "regrant_same_tick"  # parked by its level budget and re-admitted in the same tick
                self.same_tick_regrants += 1
            self.admit(game, reason, budget)
            if self.audition_pool is not None:
                self._settle_pool()  # a focus admission shrinks the pool's capacity

    # --- hazard index ----------------------------------------------------------------------

    def _index_tick(self):
        """Fill free slots with the highest index; then at most one swap per tick: the best waiting
        game displaces the lowest-index focus game that has used its quantum, if it beats it by the
        hysteresis margin. The hazard model is re-estimated from this run's records every tick."""
        cfg = self.index_cfg
        model = self.index_model(self.games, cfg)
        remaining = self.global_clock.remaining()
        scores = {g.game_id: model.index(g, remaining) for g in self.games if g.state != "finished"}
        elapsed = self.global_clock.elapsed()
        if not self.index_history or elapsed - self.index_history[-1]["wall_seconds"] >= 120:
            self.index_history.append({"wall_seconds": round(elapsed, 1), "model": model.summary(), "games": {
                g.game_id[:4]: {"state": g.state, "levels": g.levels, "index": round(scores[g.game_id][0], 5),
                                **scores[g.game_id][1]} for g in self.games if g.game_id in scores}})
        waiting = sorted((g for g in self.games if g.state in ("parked", "backlog")),
                         key=lambda g: (-scores[g.game_id][0], g.index))
        while len(self.focus()) < self.active_limit and waiting:
            self._index_admit(waiting.pop(0), scores, "index_fill")
        if not waiting:
            return
        best = waiting[0]
        eligible = [g for g in self.focus() if g.clock.completion_tokens() - g.admit_tokens >= model.quantum()]
        if not eligible:
            return
        worst = min(eligible, key=lambda g: (scores[g.game_id][0], -g.index))
        if scores[best.game_id][0] > (1 + cfg["hysteresis"]) * scores[worst.game_id][0]:
            self.index_swaps += 1
            if self.log:
                self.log({"event": "index_swap", "wall_seconds": round(elapsed, 3), "model": model.summary(),
                          "out": {"game": worst.game_id, "index": scores[worst.game_id][0], **scores[worst.game_id][1]},
                          "in": {"game": best.game_id, "index": scores[best.game_id][0], **scores[best.game_id][1]}})
            self.park(worst, "index")
            self._index_admit(best, scores, "index")

    def _index_admit(self, game, scores, reason):
        if game.lane:
            self.lane_promotions.append({"game": game.game_id, "reason": reason, "levels": game.levels,
                                         "wall_seconds": round(self.global_clock.elapsed(), 3),
                                         "after_lane_clear": game.pending_parked_progress})
        self.admit(game, reason, None)
        game.admit_tokens = game.clock.completion_tokens()

    # --- continuous ranking (hero-v2-rank) ---------------------------------------------------

    def _rank_tick(self, fill_reason="rank_fill"):
        """Every ``tick_seconds``: score every unfinished game, fill free slots from the top, then swap the worst
        focus game past its token tenure for the best waiting game while the waiting game scores more than
        ``1 + hysteresis`` times it. Between ticks only a slot freed by a finished game refills (latest order)."""
        from .rank import horizon, score

        cfg = self.rank
        elapsed = self.global_clock.elapsed()
        if self.rank_last_tick is not None and elapsed - self.rank_last_tick < cfg["tick_seconds"]:
            for game in self.rank_order:
                if len(self.focus()) >= self.active_limit:
                    break
                if game.state in ("parked", "backlog") and game.finished_reason is None:
                    self.rank_counts["refills"] += 1
                    self._rank_admit(game, "rank_fill")
            return
        self.rank_last_tick = elapsed
        self.rank_counts["ticks"] += 1
        h = horizon(self.games, cfg)
        scores = {g.game_id: score(g, h, cfg) for g in self.games if g.state != "finished"}

        def key(g):
            return (-scores[g.game_id][0], g.index)

        waiting = sorted((g for g in self.games if g.state in ("parked", "backlog")), key=key)
        while len(self.focus()) < self.active_limit and waiting:
            self.rank_counts["fills"] += 1
            self._rank_admit(waiting.pop(0), fill_reason)
        while waiting:
            eligible = [g for g in self.focus()
                        if g.clock.completion_tokens() - g.admit_tokens >= cfg["tenure_tokens"]]
            if not eligible:
                break
            worst = min(eligible, key=lambda g: (scores[g.game_id][0], -g.index))
            best = waiting[0]
            if scores[best.game_id][0] <= (1 + cfg["hysteresis"]) * scores[worst.game_id][0]:
                break
            waiting.pop(0)
            self.rank_counts["swaps"] += 1
            swap = {"wall_seconds": round(elapsed, 3), "horizon_tokens": round(h),
                    "out": {"game": worst.game_id, "score": scores[worst.game_id][0], **scores[worst.game_id][1]},
                    "in": {"game": best.game_id, "score": scores[best.game_id][0], **scores[best.game_id][1]}}
            self.rank_swaps.append(swap)
            if self.log:
                self.log({"event": "rank_swap", **swap})
            self.park(worst, "rank")
            self._rank_admit(best, "rank")
        self.rank_order = sorted((g for g in self.games if g.state != "finished"), key=key)
        if not self.rank_history or elapsed - self.rank_history[-1]["wall_seconds"] >= 120:
            self.rank_history.append({"wall_seconds": round(elapsed, 1), "horizon_tokens": round(h), "top": [
                {"game": g.game_id, "state": g.state, "score": scores[g.game_id][0], **scores[g.game_id][1]}
                for g in self.rank_order[: 2 * self.active_limit]]})

    def _rank_admit(self, game, reason):
        self.admit(game, reason, None)
        game.admit_tokens = game.clock.completion_tokens()

    def rank_summary(self):
        played = sum(1 for g in self.games if g.clock.completion_tokens() > 0)
        return {"settings": dict(self.rank), **self.rank_counts, "games_played": played,
                "swaps_detail": self.rank_swaps[-200:], "history": self.rank_history}

    # --- three-block schedule (hero-v2-blocks) ------------------------------------------------

    def _blocks_note(self, game, decision, **details):
        entry = {"decision": decision, "game": game.game_id, "levels": game.levels, "home_block": game.home_block,
                 "role": game.role, "tokens": game.clock.budget_tokens(), "block": self.blocks_index,
                 "wall_seconds": round(self.global_clock.elapsed(), 3), **details}
        self.blocks_decisions.append(entry)
        if self.log:
            self.log({"event": "blocks", **entry})

    def blocks_unrun(self):
        """Never-admitted games that can still play, in roster order."""
        return [g for g in self.games if g.state == "backlog" and g.first_admit_wall is None and not g.retired
                and g.finished_reason is None]

    def blocks_old(self):
        """Parked games that can still progress (not retired, not finished), best first: most levels, then fewest
        gameplay completion tokens, then roster order."""
        return sorted((g for g in self.games if g.state == "parked" and g.first_admit_wall is not None
                       and not g.retired and g.finished_reason is None),
                      key=lambda g: (-g.levels, g.clock.budget_tokens(), g.index))

    def blocks_retired(self):
        """Retired games that are not finished (the idling fix), in continuation order: fewest
        continuations first (every retired game gets one before any gets a second), then most levels cleared, then
        fewest gameplay completion tokens, then roster order."""
        return sorted((g for g in self.games if g.state == "parked" and g.first_admit_wall is not None
                       and g.retired and g.finished_reason is None),
                      key=lambda g: (g.continuations, -g.levels, g.clock.budget_tokens(), g.index))

    def _blocks_exhausted(self, game):
        allowance = self.blocks.get("level_token_allowance")
        if allowance:
            return game.spent_on_level() - game.allowance_base >= allowance * (game.levels + 1)
        return game.levels == 0 and game.clock.budget_tokens() - game.allowance_base >= self.blocks["level0_budget_tokens"]

    def _blocks_admit(self, game, reason, role, **details):
        first = game.first_admit_wall is None
        if first:
            game.first_admit_wall = self.global_clock.elapsed()
            game.first_admit_tokens = game.clock.budget_tokens()
        game.role = role
        self.admit(game, reason, None)
        game.admit_tokens = game.clock.completion_tokens()
        self.blocks_counts[reason] += 1
        self._blocks_note(game, reason, first_admission=first, **details)

    def blocks_open_slots(self):
        """Slots open now (block 1's staggered start opens them one by one; all afterwards)."""
        from .blocks import release_cap

        return release_cap(self.global_clock.elapsed(), self.blocks)

    def _blocks_fill(self, freed=None):
        """Free slots take the next never-run game in roster order (block_start inside the current block, filler from
        a later block), else the best old game (old_refill)."""
        while len(self.active()) < self.blocks_open_slots():
            unrun = self.blocks_unrun()
            if unrun:
                game = unrun[0]
                if game.home_block <= self.blocks_index:
                    self._blocks_admit(game, "block_start", "original", freed=freed)
                else:
                    self._blocks_admit(game, "filler", "filler", freed=freed,
                                       tenure_end_wall=self._blocks_tenure_wall(game, planned=True))
                continue
            old = self.blocks_old()
            if not old:
                # Idling fix: every game has started and no old game can return, so a free slot
                # continues a retired (not finished) game with a fresh allowance for its current level.
                retired = self.blocks_retired()
                if not retired:
                    return
                game = retired[0]
                game.retired = False
                game.continuations += 1
                game.allowance_base = (game.spent_on_level() if self.blocks.get("level_token_allowance")
                                       else game.clock.budget_tokens())
                self._blocks_admit(game, "retired_continue", "old", freed=freed, candidates=len(retired),
                                   continuation=game.continuations, allowance_base=round(game.allowance_base, 1))
                continue
            self._blocks_admit(old[0], "old_refill", "old", freed=freed,
                               candidates=len(old), next_candidate=old[1].game_id if len(old) > 1 else None)

    def _blocks_tenure_wall(self, game, planned=False):
        if self.blocks["tenure_unit"] != "wall_seconds":
            return None
        start = self.global_clock.elapsed() if planned and game.first_admit_wall is None else game.first_admit_wall
        return round(start + self.blocks["block_seconds"], 3)

    def _blocks_block_over(self, game):
        """An original's block ends one block length (wall) after its own first admission: the boundary itself when it
        started at its block's start, later when block 1 started staggered."""
        return self.global_clock.elapsed() - game.first_admit_wall >= self.blocks["block_seconds"] - 1e-6

    def _blocks_tenure_over(self, game):
        cfg = self.blocks
        if cfg["tenure_unit"] == "wall_seconds":
            return self.global_clock.elapsed() - game.first_admit_wall >= cfg["block_seconds"]
        return game.clock.budget_tokens() - game.first_admit_tokens >= cfg["tenure_tokens"]

    def _blocks_retire(self, game):
        tokens = game.clock.budget_tokens()
        reason = "level_retired" if self.blocks.get("level_token_allowance") else "level0_retired"
        game.retired = True
        self.blocks_counts[reason] += 1
        limit = ((game.levels + 1) * self.blocks["level_token_allowance"]
                 if self.blocks.get("level_token_allowance") else self.blocks["level0_budget_tokens"])
        self._blocks_note(game, reason, budget_tokens=limit, level_spent=game.spent_on_level(), was=game.state)
        if game.state == "active":
            self.park(game, reason)
        else:
            # Its last reply landed after an earlier park: it is out for good but no slot frees.
            self._record(game, "progress", reason, extra={"tokens": tokens})

    def _blocks_tick(self):
        from .blocks import block_at

        cfg = self.blocks
        now = self.global_clock.elapsed()
        idle = self.blocks_idle
        if idle["last_tick"] is not None and now > idle["last_tick"]:
            dt = now - idle["last_tick"]
            idle["slot_seconds"] += dt * idle["free"]
            idle["slot_seconds_with_eligible"] += dt * min(idle["free"], idle["eligible"])
        idle["last_tick"] = now
        # Block boundaries on the scheduler's own wall (a research run's earlier stop does not move them).
        target = block_at(now, cfg)
        while self.blocks_index < target:
            ending = self.blocks_index
            self.blocks_index += 1
            parked = []
            for game in self.active():
                if game.role == "original" and game.home_block == ending and self._blocks_block_over(game):
                    self.park(game, "block_end")
                    self.blocks_counts["block_end"] += 1
                    parked.append(game.game_id)
            entry = {"decision": "boundary", "ended_block": ending, "block": self.blocks_index,
                     "wall_seconds": round(now, 3), "parked": parked,
                     "unrun_in_block": [g.game_id for g in self.blocks_unrun() if g.home_block == self.blocks_index]}
            self.blocks_decisions.append(entry)
            if self.log:
                self.log({"event": "blocks", **entry})
            self._blocks_fill(freed="block_end")
        # Level-0 retirements (an active game frees its slot; a parked game's late reply retires it in place).
        for game in self.games:
            if not game.retired and game.state in ("active", "parked") and self._blocks_exhausted(game):
                self._blocks_retire(game)
        # Staggered originals (stagger_seconds): one block after their own first admission (block_end), after the
        # boundary; without a stagger they all parked at the boundary above.
        for game in self.active():
            if game.role == "original" and game.home_block < self.blocks_index and self._blocks_block_over(game):
                self.park(game, "block_end")
                self.blocks_counts["block_end"] += 1
                self._blocks_note(game, "block_end", first_admit_wall=round(game.first_admit_wall, 3))
        # Tenure ends (fillers; an original's tenure ends at its own block's end above).
        for game in self.active():
            if game.role == "filler" and self._blocks_tenure_over(game):
                self.park(game, "tenure_end")
                self.blocks_counts["tenure_end"] += 1
                self._blocks_note(game, "tenure_end", first_admit_wall=round(game.first_admit_wall, 3))
        self._blocks_fill(freed="slot")
        idle["free"] = max(0, self.blocks_open_slots() - len(self.active()))  # a slot not yet started is not idle
        idle["eligible"] = len(self.blocks_unrun()) + len(self.blocks_old()) + len(self.blocks_retired())

    def _blocks_level(self, game):
        """No level budgets. A level from a retired game's last in-flight reply lifts the retirement: it was not
        still on level 0 when its budget ran out, so it becomes an old game (``retired_level``)."""
        if game.retired and game.levels >= 1 and game.finished_reason is None and game.state != "finished":
            game.retired = False
            self.blocks_counts["retired_level"] += 1
            self._record(game, "progress", "retired_level")
            self._blocks_note(game, "retired_level")

    def blocks_progress(self):
        """Progress-row counts (trial ``progress``)."""
        active = self.active()
        return {"block": self.blocks_index + 1, "blocks": self.blocks["blocks"],
                "next_boundary_seconds": (round((self.blocks_index + 1) * self.blocks["block_seconds"], 1)
                                          if self.blocks_index + 1 < self.blocks["blocks"] else None),
                "active": len(active), "slots": self.active_limit,
                "active_by_role": {r: sum(g.role == r for g in active) for r in ("original", "filler", "old")},
                "started": sum(g.first_admit_wall is not None for g in self.games),
                "unrun": len(self.blocks_unrun()), "parked_eligible": len(self.blocks_old()),
                "retired": sum(g.retired for g in self.games),
                "finished": sum(g.state == "finished" for g in self.games),
                **{k: v for k, v in self.blocks_counts.items()
                   if k in ("filler", "old_refill", "block_end", "tenure_end", "level0_retired")},
                "idle_slot_seconds": round(self.blocks_idle["slot_seconds"], 1)}

    def blocks_summary(self):
        games = {}
        for g in self.games:
            games[g.game_id] = {
                "index": g.index, "home_block": g.home_block, "role": g.role, "state": g.state, "levels": g.levels,
                "retired": g.retired, "tokens": g.clock.budget_tokens(),
                "first_admit_wall": None if g.first_admit_wall is None else round(g.first_admit_wall, 3),
                "admissions": sum(e["event"] == "admit" for e in g.admissions),
                "finished_reason": g.finished_reason}
        return {"settings": dict(self.blocks), "block": self.blocks_index, "counts": dict(self.blocks_counts),
                "progress": self.blocks_progress(),
                "idle_slot_seconds": round(self.blocks_idle["slot_seconds"], 3),
                "idle_slot_seconds_with_eligible": round(self.blocks_idle["slot_seconds_with_eligible"], 3),
                "games_started": sum(g.first_admit_wall is not None for g in self.games),
                "decisions": list(self.blocks_decisions), "games": games}

    # --- priority scheduling (dfranzen's priority gate) ----------------------------------------

    def priority_waiting(self):
        """Games waiting for a slot (never started or handed over; finished games never wait)."""
        return [g for g in self.games if g.state in ("backlog", "parked") and g.finished_reason is None]

    def _priority_now(self):
        """``(wall seconds, remaining seconds)`` of the run's real wall (the gateway's deadline shortens it)."""
        return self.global_clock.seconds, self.global_clock.remaining()

    def priority_value(self, game, fresh=False):
        """``(value, details)``: an unstarted game's band (2,000,000 - roster position), else its score now (or, without
        ``refresh_queue``, the score it was queued with at its handover)."""
        from .priority import UNSTARTED_BASE, score

        if game.readmissions == 0:
            return UNSTARTED_BASE - game.index, {"unstarted": True}
        if not fresh and not self.priority_cfg["refresh_queue"] and game.priority_queued_score is not None:
            return game.priority_queued_score, {"queued_score": True}
        return score(game, self.priority_cfg, *self._priority_now())

    def _priority_rank(self, candidates, holder=None):
        """Candidates ``[(game, value, details, arrival)]`` best first: highest value, then the earlier arrival (their
        heap's sequence; unstarted games arrive in roster order). Level-capped games come after every uncapped one,
        fewest tokens on the level first; ``holder`` (a capped game inside its grant) leads the capped ones."""
        def key(c):
            game = c[0]
            if not game.level_capped:
                return (0, -c[1], c[3])
            return (1, -1.0 if game is holder else game.spent_on_level(), c[3])

        return sorted(candidates, key=key)

    # --- level cap (priority.level_cap_tokens) ---------------------------------------------

    def _gameplay_tokens(self):
        return sum(g.clock.budget_seconds() for g in self.games)

    def _level_cap_over(self, game):
        """At an eviction: cap ``game`` if its level reached a cap (tokens or requests); True when no grant is left."""
        cap, cap_requests = self.priority_cfg["level_cap_tokens"], self.priority_cfg["level_cap_requests"]
        if not cap and not cap_requests:
            return False
        spent, asked = game.spent_on_level(), game.requests_on_level()
        if not game.level_capped and ((cap and spent >= cap) or (cap_requests and asked >= cap_requests)):
            game.level_capped = True
            game.level_cap_limit = spent
            game.level_cap_request_limit = asked
            self.level_cap_counts["capped"] += 1
            self._priority_note("level_capped", game, tokens_on_level=int(spent), requests_on_level=asked, cap=cap,
                                cap_requests=cap_requests)
        return game.level_capped and bool((cap and spent >= game.level_cap_limit)
                                          or (cap_requests and asked >= game.level_cap_request_limit))

    def _level_cap_grant(self, game):
        game.level_cap_limit = game.spent_on_level() + self.priority_cfg["level_cap_grant_tokens"]
        game.level_cap_request_limit = game.requests_on_level() + self.priority_cfg["level_cap_grant_requests"]

    def _level_cap_settle(self, game):
        """A capped game leaves the queue (admitted or cap lifted): book the tokens its absence freed."""
        if game.level_cap_parked_total is not None:
            self.level_cap_freed += (self._gameplay_tokens() - game.level_cap_parked_total) / max(self.active_limit, 1)
            game.level_cap_parked_total = None

    def level_cap_progress(self):
        """Live counters (inside ``priority_progress``): ``capped`` levels capped so far, ``parked_now`` capped games
        waiting, ``returns`` admissions of a capped game to a slot nothing uncapped wanted, ``tokens_freed`` one slot's
        share of the run's gameplay tokens while capped games waited."""
        total = self._gameplay_tokens()
        waiting = [g for g in self.games if g.level_capped and g.state == "parked" and g.finished_reason is None]
        open_share = sum(total - g.level_cap_parked_total for g in waiting if g.level_cap_parked_total is not None)
        return {"cap": self.priority_cfg["level_cap_tokens"], "grant": self.priority_cfg["level_cap_grant_tokens"],
                "cap_requests": self.priority_cfg["level_cap_requests"],
                **self.level_cap_counts, "parked_now": len(waiting),
                "active_now": sum(g.level_capped and g.state == "active" for g in self.games),
                "tokens_freed": int(self.level_cap_freed + open_share / max(self.active_limit, 1))}

    def _priority_candidates(self):
        out = []
        for game in self.priority_waiting():
            value, details = self.priority_value(game)
            arrival = game.index - len(self.games) if game.priority_queued is None else game.priority_queued
            out.append((game, value, details, arrival))
        return out

    def _priority_note(self, decision, game=None, **details):
        entry = {"decision": decision, "wall_seconds": round(self.global_clock.elapsed(), 3),
                 **({} if game is None else {"game": game.game_id, "levels": game.levels}), **details}
        self.priority_decisions.append(entry)
        if self.log:
            self.log({"event": "priority", **entry})

    def _priority_admit(self, game, value, details, freed):
        """Admit ``game`` into a free slot: reason ``first_window`` for its first admission, else what freed the slot
        (``handover`` or ``finished``)."""
        reason = "first_window" if game.readmissions == 0 else freed
        game.priority_queued = None
        game.priority_queued_score = None
        extra = {"freed": freed, "priority": value}
        if game.level_capped:
            # Nothing uncapped waits: the slot would idle. A further grant on the capped level.
            self._level_cap_settle(game)
            self._level_cap_grant(game)
            self.level_cap_counts["returns"] += 1
            extra.update(level_cap_return=True, tokens_on_level=int(game.spent_on_level()))
        self.admit(game, reason, None, extra=extra)
        game.admit_tokens = game.clock.completion_tokens()
        self.priority_counts[reason] += 1
        return reason

    def _priority_fill(self, freed):
        """Free slots (the release, or slots of finished games) take the best waiting games."""
        while len(self.active()) < self.active_limit:
            ranked = self._priority_rank(self._priority_candidates())
            if not ranked:
                return
            game, value, details, _ = ranked[0]
            self._priority_admit(game, value, details, freed)

    def _priority_tick(self):
        """Finished games' slots refill (``finish`` ran at the top of ``tick``); the tail fade's start is logged once.
        Nothing is preempted here: handovers happen only at context evictions (``on_context_eviction``)."""
        from .priority import tail_fraction

        self._priority_fill("finished")
        if self.priority_fade_wall is None:
            fade = tail_fraction(self.priority_cfg, *self._priority_now())
            if fade is not None and fade <= 1.0:
                self.priority_fade_wall = round(self.global_clock.elapsed(), 3)
                self._priority_note("tail_fade", remaining_seconds=round(self.global_clock.remaining(), 3),
                                    window_seconds=round(self.priority_cfg["tail_fade_fraction"]
                                                         * self.global_clock.seconds, 3))

    def on_context_eviction(self, game, kind, **details):
        """``GameRecord.note_compaction`` / ``note_context_drain``: the game's cached prefix is gone. Priority: hand the
        slot over (re-score the game, re-queue it, and give the slot to the best waiting game, maybe itself). Every
        other policy ignores it."""
        if self._closed or self.priority_cfg is None:
            return
        from .priority import EVICTIONS

        if kind not in EVICTIONS:
            raise ValueError(f"Unknown context eviction {kind!r}")
        self.priority_counts[kind] += 1
        extra = {k: v for k, v in details.items() if v is not None}
        if self.phase != "steady" or game.state != "active" or game.finished_reason is not None:
            self.priority_counts["eviction_not_active"] += 1
            self._priority_note("eviction_not_active", game, eviction=kind, state=game.state, **extra)
            return
        self.priority_seq += 1
        value, info = self.priority_value(game, fresh=True)
        candidates = self._priority_candidates()
        over = self._level_cap_over(game)
        ranked = self._priority_rank(candidates + [(game, value, info, self.priority_seq)],
                                     holder=None if over else game)
        best, best_value, best_details, _ = ranked[0]
        top = [{"game": g.game_id, "priority": v} for g, v, _, _ in ranked[:3]]
        if game.level_capped:
            extra["level_cap"] = True
            extra["tokens_on_level"] = int(game.spent_on_level())
            if best is game and over:
                self._level_cap_grant(game)  # nobody better waits: a further grant in place
                self.level_cap_counts["kept"] += 1
            elif best is not game:
                self.level_cap_counts["rotations" if best.level_capped else "parked"] += 1
                game.level_cap_parked_total = self._gameplay_tokens()
        if best is game:
            self.priority_counts["handover_kept"] += 1
            self._record(game, "grant", "handover", extra={"kept": True, "eviction": kind, "priority": value})
            self._priority_note("handover_kept", game, eviction=kind, priority=value, **info, waiting=len(candidates),
                                top=top, **extra)
            return
        game.priority_queued = self.priority_seq
        game.priority_queued_score = value
        self.park(game, "handover", extra={"eviction": kind, "priority": value, "to": best.game_id,
                                           **({"level_cap": True} if game.level_capped else {})})
        reason = self._priority_admit(best, best_value, best_details, "handover")
        self._priority_note("handover", game, eviction=kind, priority=value, **info, to=best.game_id,
                            to_priority=best_value, to_reason=reason, waiting=len(candidates), top=top, **extra)

    def priority_progress(self):
        """Progress-row counts (trial ``progress``)."""
        waiting = self.priority_waiting()
        return {"active": len(self.active()), "slots": self.active_limit, "waiting": len(waiting),
                "unstarted": sum(g.readmissions == 0 for g in waiting), **self.priority_counts,
                "tail_fade_wall_seconds": self.priority_fade_wall,
                **({"level_cap": self.level_cap_progress()} if self.priority_cfg["level_cap_tokens"] or self.priority_cfg["level_cap_requests"] else {})}

    def priority_summary(self):
        games = {}
        for g in self.games:
            value, details = (None, {}) if g.state == "finished" else self.priority_value(g, fresh=True)
            games[g.game_id] = {
                "index": g.index, "state": g.state, "levels": g.levels, "total_levels": g.total_levels,
                "admissions": sum(e["event"] == "admit" for e in g.admissions),
                "handovers": sum(e["event"] == "park" and e["reason"] == "handover" for e in g.admissions),
                "kept": sum(e["event"] == "grant" and e["reason"] == "handover" for e in g.admissions),
                "compactions": g.compactions, "context_drains": g.context_drains,
                "context_evictions": g.context_evictions, "priority": value, **details,
                **({"level_capped": True, "tokens_on_level": int(g.spent_on_level())} if g.level_capped else {}),
                "finished_reason": g.finished_reason}
        ranked = self._priority_rank(self._priority_candidates())
        return {"settings": dict(self.priority_cfg), "counts": dict(self.priority_counts),
                "progress": self.priority_progress(), "tail_fade_wall_seconds": self.priority_fade_wall,
                "waiting_order": [{"game": g.game_id, "priority": v} for g, v, _, _ in ranked[:20]],
                "games_started": sum(g.readmissions > 0 for g in self.games),
                "decisions": list(self.priority_decisions), "games": games}

    # --- momentum commitment (fixed-2x-commit) ---------------------------------------------

    def stall_cap(self, game):
        """Tokens a focus game may spend on its current level: clamp(stall_multiple x the median tokens
        per level it has cleared (level 1 included), stall_min_tokens, stall_max_tokens); the minimum
        before its first clear. Zero entries (levels jumped in one observation) are not attempts."""
        cfg = self.commit
        cleared = [tokens for tokens in game.level_tokens if tokens > 0]
        cap = cfg["stall_multiple"] * statistics.median(cleared) if cleared else 0
        return int(min(max(cap, cfg["stall_min_tokens"]), cfg["stall_max_tokens"]))

    def _commit_tick(self):
        """Demote focus games past their stall cap to the lane; fill free slots."""
        for game in self.focus():
            if game.tokens_on_level() > self.stall_cap(game):
                self._commit_demote(game)
        # With no clear to promote, a free slot goes to the lane game with the most levels, then the
        # fewest tokens on its level. A demoted game comes back only through a clear, and a game already
        # past its stall cap (a level-0 game past the floor) is not filled only to be demoted at once.
        # Queued lane clears (commit-queue) take freed slots first, earliest clear first; a queued game that
        # has since run past its own stall cap in the lane, or finished, leaves the queue.
        while self.commit_queue and len(self.focus()) < self.active_limit:
            game = self.commit_queue.pop(0)
            if (game.state in ("parked", "backlog") and game.finished_reason is None and not game.stalled_level
                    and game.tokens_on_level() <= self.stall_cap(game)):
                self._commit_promote(game, "lane_clear_queued")
        while len(self.focus()) < self.active_limit:
            waiting = [g for g in self.games if g.state in ("parked", "backlog") and g.finished_reason is None
                       and not g.stalled_level and g.tokens_on_level() <= self.stall_cap(g)]
            if not waiting:
                return
            self._commit_promote(min(waiting, key=lambda g: (-g.levels, g.tokens_on_level(), g.index)), "fill")

    def _commit_demote(self, game):
        tokens, cap = game.tokens_on_level(), self.stall_cap(game)
        game.stalled_level = True
        self.commit_demotions.append({"game": game.game_id, "levels": game.levels, "tokens_on_level": tokens,
                                      "stall_cap": cap, "wall_seconds": round(self.global_clock.elapsed(), 3)})
        self.park(game, "stalled_level")

    def _commit_promote(self, game, reason, replaced=None):
        self.commit_promotions.append({
            "game": game.game_id, "reason": reason, "levels": game.levels,
            "tokens_on_level": game.tokens_on_level(), "wall_seconds": round(self.global_clock.elapsed(), 3),
            "replaced": None if replaced is None else replaced.game_id})
        self.admit(game, reason, None)
        game.admit_tokens = game.clock.completion_tokens()

    def _commit_lane_clear(self, game):
        """A clear in the lane takes a focus slot at once: a free one, else the focus game with the worst
        standing (one past its stall cap first, else the most tokens on its current level) goes to the lane."""
        replaced = None
        focus = self.focus()
        if len(focus) >= self.active_limit:
            stalled = [g for g in focus if g.tokens_on_level() > self.stall_cap(g)]
            if not stalled and not self.commit.get("replace_unstalled", True):
                # commit-queue (commit-v1: 13 promotions replaced ar25 and sb26 right after their level-1
                # clears): a focus game within its stall cap keeps its slot; the clear waits for one.
                if game not in self.commit_queue:
                    self.commit_queue.append(game)
                    self.commit_promotions.append({
                        "game": game.game_id, "reason": "queued", "levels": game.levels,
                        "tokens_on_level": game.tokens_on_level(),
                        "wall_seconds": round(self.global_clock.elapsed(), 3), "replaced": None})
                return
            replaced = max(stalled or focus, key=lambda g: (g.tokens_on_level(), -g.index))
            if stalled:
                self._commit_demote(replaced)
            else:
                self.park(replaced, "commit_replaced")
        self._commit_promote(game, "lane_clear", replaced)

    def commit_summary(self):
        floor = self.commit["level1_floor_tokens"]
        level1 = {}
        for g in self.games:
            tokens = g.level_tokens[0] if g.levels >= 1 and g.level_tokens else g.tokens_on_level()
            level1[g.game_id] = {"tokens": tokens, "cleared": g.levels >= 1}
        total = sum(g.clock.completion_tokens() for g in self.games)
        lane_tokens = sum(g.lane_tokens() for g in self.games)
        elapsed = min(self.global_clock.elapsed(), self.global_clock.seconds)
        reasons = {}
        for promotion in self.commit_promotions:
            reasons[promotion["reason"]] = reasons.get(promotion["reason"], 0) + 1
        return {
            "settings": dict(self.commit),
            "promotions": list(self.commit_promotions), "promotion_reasons": reasons,
            "demotions": list(self.commit_demotions),
            "level1_floor": {
                "floor_tokens": floor, "games": level1,
                "uncleared_below_floor": sorted(k for k, v in level1.items() if not v["cleared"] and v["tokens"] < floor),
                "uncleared_at_or_above_floor": sorted(k for k, v in level1.items()
                                                      if not v["cleared"] and v["tokens"] >= floor)},
            "lane": {"requests": sum(g.lane_requests for g in self.games), "completion_tokens": lane_tokens,
                     "clears": list(self.lane_clears),
                     "games": {g.game_id: {"requests": g.lane_requests, "completion_tokens": g.lane_tokens(),
                                           "clears": sum(c["game"] == g.game_id for c in self.lane_clears)}
                               for g in self.games if g.lane_requests}},
            "completion_tokens": total, "lane_completion_tokens": lane_tokens,
            "tokens_per_hour": round(total / elapsed * 3600, 1) if elapsed > 0 else None,
        }

    # --- deep/explore (fixed-2x-deep-explore) ---------------------------------------------

    def deep_games(self):
        return [g for g in self.games if g.state == "active" and g.slot == "deep"]

    def explore_games(self):
        return [g for g in self.games if g.state == "active" and g.slot == "explore"]

    def deep_budget(self, game):
        """Tokens a deep game may spend on its current level before it can be demoted: max(deep_min_tokens,
        deep_multiple x the median tokens of the levels it has cleared in this run). Zero entries (levels jumped in
        one observation) are not attempts."""
        cfg = self.deep
        cleared = [tokens for tokens in game.level_tokens if tokens > 0]
        scaled = cfg["deep_multiple"] * statistics.median(cleared) if cleared else 0
        return int(max(cfg["deep_min_tokens"], scaled))

    def _queue_key(self, game):
        """Most recent clear first; ties: fewer tokens on the cleared levels, then roster order."""
        return (-(game.last_clear_wall or 0.0), sum(game.level_tokens), game.index)

    def promotion_queue(self):
        """The valid queue in promotion order (finished games leave it)."""
        for game in [g for g in self.deep_queue if g.finished_reason is not None or g.state == "finished"]:
            self.deep_queue.remove(game)
            game.deep_queued = False
        return sorted(self.deep_queue, key=self._queue_key)

    def _explore_key(self, game):
        """Never played first (roster order), then fewest total completion tokens; a game with explore_max_tokens or
        more on level 1 and no clear goes last."""
        if game.readmissions == 0:
            return (0, 0, game.index)
        exhausted = game.levels == 0 and game.tokens_on_level() >= self.deep["explore_max_tokens"]
        return (2 if exhausted else 1, game.clock.completion_tokens(), game.index)

    def explore_waiting(self):
        return sorted((g for g in self.games if g.state in ("backlog", "parked") and g.finished_reason is None
                       and not g.deep_queued), key=self._explore_key)

    def _deep_note(self, game, decision, **details):
        entry = {"decision": decision, "game": game.game_id, "levels": game.levels,
                 "completion_tokens": game.clock.completion_tokens(), "tokens_on_level": game.tokens_on_level(),
                 "wall_seconds": round(self.global_clock.elapsed(), 3), **details}
        self.deep_decisions.append(entry)
        if self.log:
            self.log({"event": "deep_explore", **entry})

    def _set_slot(self, game, slot):
        game.slot = slot
        game.set_window(self.deep["deep_window"] if slot == "deep" else self.deep["explore_window"])

    def _deep_tick(self):
        """Quantum ends first (they fill the queue), then deep demotions, deep promotions and exploration fills."""
        for game in self.games:
            if game.state == "finished" and game.slot is not None:
                self._leave_slot(game, "finished")
        self._explore_quanta()
        queue = self.promotion_queue()
        for game in self.deep_games():
            budget = self.deep_budget(game)
            if game.tokens_on_level() > budget and queue:
                self._deep_demote(game, budget, queue[0])
                queue = self.promotion_queue()
        self._deep_fill()
        self._explore_fill("explore_rotate")

    def _explore_quanta(self):
        quantum = self.deep["explore_quantum_tokens"]
        for game in self.explore_games():
            used = game.clock.completion_tokens() - game.admit_tokens
            if used < quantum:
                continue
            if game.levels > game.admit_levels:
                # A clear in this turn: the slot was kept to the end of the quantum; now the game waits for a deep slot.
                self._leave_slot(game, "explore_queue")
                self.park(game, "explore_queue")
                game.deep_queued = True
                self.deep_queue.append(game)
                self._deep_note(game, "explore_queue", quantum_tokens=used,
                                queue=[g.game_id for g in self.promotion_queue()])
            elif self.explore_waiting():
                self._leave_slot(game, "explore_rotate")
                self.park(game, "explore_rotate")
                self._deep_note(game, "explore_rotate", quantum_tokens=used)
            else:
                # Nobody waits for an exploration slot: a new quantum instead of an idle slot.
                game.admit_tokens = game.clock.completion_tokens()
                game.admit_levels = game.levels
                self._record(game, "grant", "explore_renew")
                self._deep_note(game, "explore_renew", quantum_tokens=used)

    def _deep_fill(self):
        """A free deep slot takes the queue head, else an exploration game that cleared in its current turn (the most
        recent clear first); with neither, it stays free until the next clear."""
        while len(self.deep_games()) < self.deep["deep_slots"]:
            queue = self.promotion_queue()
            if queue:
                self._deep_promote(queue[0], "queue")
                continue
            fresh = [g for g in self.explore_games() if g.levels > g.admit_levels]
            if not fresh:
                return
            self._deep_promote(min(fresh, key=self._queue_key), "explore_clear")

    def _deep_promote(self, game, source, replaced=None):
        if game.deep_queued:
            game.deep_queued = False
            self.deep_queue.remove(game)
        was_active = game.state == "active"
        previous = game.slot
        if previous is not None:
            self._leave_slot(game, "deep_promote")
        self._set_slot(game, "deep")
        if was_active:
            # From an exploration slot: the game keeps running; the grant marks the move (no budget).
            self._record(game, "grant", "deep_promote")
        else:
            self.admit(game, "deep_promote", None)
        game.admit_tokens = game.clock.completion_tokens()
        game.admit_levels = game.levels
        self.deep_tenures.append({
            "game": game.game_id, "source": source, "start_wall_seconds": round(self.global_clock.elapsed(), 3),
            "start_levels": game.levels, "start_completion_tokens": game.clock.completion_tokens(),
            "start_inference_seconds": round(game.clock.inference_seconds(), 3), "replaced": replaced,
            "end_wall_seconds": None, "end_reason": None})
        self._deep_note(game, "deep_promote", source=source, from_slot=previous, replaced=replaced,
                        window=game.window)

    def _deep_demote(self, game, budget, candidate):
        self._leave_slot(game, "deep_demote")
        # Back in the exploration rotation with the exploration window: the existing compaction path shrinks its
        # context at its next request.
        game.set_window(self.deep["explore_window"])
        self.park(game, "deep_demote")
        self._deep_note(game, "deep_demote", deep_budget=budget, candidate=candidate.game_id, window=game.window)
        self._deep_promote(candidate, "queue", replaced=game.game_id)

    def _leave_slot(self, game, reason):
        if game.slot == "deep":
            for tenure in reversed(self.deep_tenures):
                if tenure["game"] == game.game_id and tenure["end_wall_seconds"] is None:
                    tenure.update(end_wall_seconds=round(self.global_clock.elapsed(), 3), end_reason=reason,
                                  end_levels=game.levels, end_completion_tokens=game.clock.completion_tokens(),
                                  end_inference_seconds=round(game.clock.inference_seconds(), 3))
                    break
        game.slot = None

    def _explore_fill(self, reason):
        while len(self.explore_games()) < self.deep["explore_slots"]:
            waiting = self.explore_waiting()
            if not waiting:
                return
            game = waiting[0]
            self._set_slot(game, "explore")
            self.admit(game, reason, None)
            game.admit_tokens = game.clock.completion_tokens()

    def _deep_level(self, game):
        """A clear: counted per slot; a clear in an exploration slot takes a free deep slot at once, otherwise the
        game keeps its exploration slot to the end of its quantum (``_explore_quanta`` queues it). A clear that
        arrives after the game left its slot (its last request finishing) queues it directly."""
        game.last_clear_wall = self.global_clock.elapsed()
        self.deep_clears[game.slot or "none"] += 1
        if self.phase != "steady" or game.finished_reason is not None or game.state == "finished":
            return
        if game.slot == "deep":
            self._deep_note(game, "deep_clear", deep_budget=self.deep_budget(game))
            return
        if game.slot == "explore":
            self._record(game, "progress", "explore_clear")
        free = len(self.deep_games()) < self.deep["deep_slots"]
        if free and (game.slot == "explore" or not self.promotion_queue()):
            self._deep_promote(game, "first_clear" if not self.deep_tenures else "free_slot")
            return
        if game.slot is None and not game.deep_queued:
            game.deep_queued = True
            self.deep_queue.append(game)
            self._deep_note(game, "explore_queue", late_clear=True, queue=[g.game_id for g in self.promotion_queue()])
            if free:
                self._deep_fill()
        elif game.slot == "explore":
            self._deep_note(game, "explore_clear", quantum_tokens=game.clock.completion_tokens() - game.admit_tokens)

    def slot_usage_totals(self):
        """Completion tokens, inference seconds, requests and per-stream tok/s summed per slot kind (each request is
        charged to the slot its game held when the gate admitted it)."""
        usage = {}
        for game in self.games:
            for slot, row in game.slot_usage_now().items():
                total = usage.setdefault(slot, {"completion_tokens": 0, "inference_seconds": 0.0, "requests": 0})
                for key in total:
                    total[key] += row[key]
        for row in usage.values():
            row["inference_seconds"] = round(row["inference_seconds"], 3)
            row["tokens_per_second_per_stream"] = (round(row["completion_tokens"] / row["inference_seconds"], 1)
                                                   if row["inference_seconds"] > 0 else None)
        return usage

    def deep_summary(self):
        usage = self.slot_usage_totals()
        tenures = []
        for tenure in self.deep_tenures:
            row = dict(tenure)
            if row["end_wall_seconds"] is None:
                game = next(g for g in self.games if g.game_id == row["game"])
                row.update(end_levels=game.levels, end_completion_tokens=game.clock.completion_tokens(),
                           end_inference_seconds=round(game.clock.inference_seconds(), 3), open=True)
            row["levels"] = row["end_levels"] - row["start_levels"]
            row["completion_tokens"] = row["end_completion_tokens"] - row["start_completion_tokens"]
            tenures.append(row)
        clears = sorted(g.time_to_level[0]["wall_seconds"] for g in self.games if g.time_to_level)
        explore_tokens = usage.get("explore", {}).get("completion_tokens", 0)
        counts = {}
        for decision in self.deep_decisions:
            counts[decision["decision"]] = counts.get(decision["decision"], 0) + 1
        return {
            "settings": dict(self.deep), "decision_counts": counts, "decisions": list(self.deep_decisions),
            "tenures": tenures, "clears_by_slot": dict(self.deep_clears),
            "first_clear_wall_seconds": clears[0] if clears else None,
            "explore_clears_per_10k_tokens": (round(self.deep_clears["explore"] / explore_tokens * 10_000, 4)
                                              if explore_tokens else None),
            "slot_usage": usage,
            "queue": [g.game_id for g in self.promotion_queue()],
            "deep": [g.game_id for g in self.deep_games()],
            "games_played": sum(g.readmissions > 0 for g in self.games),
        }

    # --- depthwise (fixed-2x-depthwise) -----------------------------------------------------

    def dw_slot_games(self, slot):
        """Active games in ``slot`` ("depth": at least one level; "level0": a rotation turn)."""
        return [g for g in self.games if g.state == "active" and g.slot == slot]

    def _dw_key(self, game):
        """Refill order. ``backlog="depth-first"``: parked clears, then parked depth games by fastest level-1 clear
        (smallest T1), then level-0 games never played (roster order), then level-0 games by fewest completion tokens;
        roster order breaks ties. ``backlog="fixed-2x"``: the fixed-2x readmissions order (parked clears, fewest
        admissions, most levels, roster order)."""
        if self.depthwise.get("backlog") == "fixed-2x":
            return (0 if game.pending_parked_progress else 1, game.readmissions, -game.levels, game.index)
        if game.levels >= 1:
            return (0 if game.pending_parked_progress else 1, game.t1 or 0.0, game.index)
        if game.readmissions == 0:
            return (2, 0, game.index)
        return (3, game.clock.completion_tokens(), game.index)

    def dw_waiting(self, exclude=()):
        """Games waiting for a slot, in refill order (finished games never wait)."""
        return sorted((g for g in self.games if g.state in ("backlog", "parked") and g.finished_reason is None
                       and g not in exclude), key=self._dw_key)

    def dw_grant(self, game):
        """Re-admission grant for a parked depth game. ``regrant="spent"``: the rest of the level's fixed-2x allowance
        if any is left (a parked clear starts a fresh level), else max(1.5 x spent, spent + 30) with spent = busy
        seconds on the current level; ``regrant="full"``: the full fixed-2x allowance again."""
        full = self.level_budget(game)
        if self.depthwise["regrant"] == "full":
            return full
        spent = game.spent_on_level()
        fresh = full - spent
        if fresh > 0:
            return fresh
        return max(self.regrant_multiple * spent, spent + self.regrant_floor)

    def _dw_note(self, game, decision, **details):
        entry = {"decision": decision, "game": game.game_id, "levels": game.levels, "slot": game.slot,
                 "completion_tokens": game.clock.completion_tokens(),
                 "budget_seconds": round(game.clock.budget_seconds(), 3),
                 "wall_seconds": round(self.global_clock.elapsed(), 3), **details}
        self.dw_decisions.append(entry)
        if self.log:
            self.log({"event": "depthwise", **entry})

    def _dw_set_slot(self, game, slot):
        game.slot = slot
        game.set_window(self.depthwise["deep_window"] if slot == "depth" else self.depthwise["level0_window"])

    def _dw_open_tenure(self, game, source, budget):
        self.dw_tenures.append({
            "game": game.game_id, "source": source, "start_wall_seconds": round(self.global_clock.elapsed(), 3),
            "start_levels": game.levels, "start_completion_tokens": game.clock.completion_tokens(),
            "start_inference_seconds": round(game.clock.inference_seconds(), 3),
            "t1_seconds": None if game.t1 is None else round(game.t1, 3), "budget": round(budget, 3),
            "end_wall_seconds": None, "end_reason": None})

    def _dw_leave(self, game, reason):
        if game.slot == "depth":
            for tenure in reversed(self.dw_tenures):
                if tenure["game"] == game.game_id and tenure["end_wall_seconds"] is None:
                    tenure.update(end_wall_seconds=round(self.global_clock.elapsed(), 3), end_reason=reason,
                                  end_levels=game.levels, end_completion_tokens=game.clock.completion_tokens(),
                                  end_inference_seconds=round(game.clock.inference_seconds(), 3))
                    break
        game.slot = None

    def _dw_tick(self):
        """Finished games free their slots; free slots fill; level-0 turns past the quantum rotate out and exhausted
        depth games park when someone waits (each freed slot refills at once, never with the game that left it)."""
        for game in self.games:
            if game.state == "finished" and game.slot is not None:
                self._dw_note(game, "depth_finished" if game.slot == "depth" else "level0_finished",
                              reason=game.finished_reason)
                self._dw_leave(game, "finished")
        self._dw_fill()
        quantum = self.depthwise["level0_quantum_tokens"]
        for game in self.dw_slot_games("level0"):
            used = game.clock.completion_tokens() - game.admit_tokens
            if used < quantum:
                continue
            waiting = self.dw_waiting()
            if waiting:
                self._dw_leave(game, "level0_rotate")
                self.park(game, "level0_rotate")
                self._dw_note(game, "level0_rotate", turn_tokens=used, next=waiting[0].game_id)
                self._dw_fill(exclude=(game,))
            else:
                # Nobody waits: a new turn instead of an idle slot.
                game.admit_tokens = game.clock.completion_tokens()
                self._record(game, "grant", "level0_renew")
                self._dw_note(game, "level0_renew", turn_tokens=used)
        for game in self.dw_slot_games("depth"):
            if not game.clock.budget_exhausted():
                continue
            waiting = self.dw_waiting()
            if waiting:
                budget, spent = game.clock.budget, game.clock.budget_spent()
                self._dw_leave(game, "depth_park")
                self.park(game, "depth_park")
                self._dw_note(game, "depth_park", budget=round(budget, 3), spent_since_grant=round(spent, 3),
                              spent_on_level=round(game.spent_on_level(), 3), t1_seconds=round(game.t1, 3),
                              next=waiting[0].game_id, next_levels=waiting[0].levels)
                self._dw_fill(exclude=(game,))
            elif game.game_id not in self.dw_kept:
                self.dw_kept.add(game.game_id)
                self._dw_note(game, "depth_keep", budget=round(game.clock.budget, 3),
                              spent_since_grant=round(game.clock.budget_spent(), 3))
        self._dw_fill()

    def _dw_fill(self, reason=None, exclude=()):
        while len(self.active()) < self.depthwise["slots"]:
            waiting = self.dw_waiting(exclude)
            if not waiting:
                return
            game = waiting[0]
            if game.levels >= 1:
                parked_clear = game.pending_parked_progress
                spent = game.spent_on_level()
                budget = self.dw_grant(game)
                reason_ = "depth_parked_clear" if parked_clear else "depth_readmit"
                self._dw_set_slot(game, "depth")
                self.admit(game, reason_, budget)
                self.dw_kept.discard(game.game_id)
                self._dw_open_tenure(game, reason_, budget)
                self._dw_note(game, reason_, budget=round(budget, 3), spent_on_level=round(spent, 3),
                              t1_seconds=round(game.t1, 3), window=game.window)
            else:
                self._dw_set_slot(game, "level0")
                self.admit(game, reason or "level0_rotate", None)
                game.admit_tokens = game.clock.completion_tokens()

    def _dw_level(self, game):
        """A clear. In a level-0 turn: the game keeps its slot as a depth game at the full window with the fixed-2x
        budget from its T1 (``level0_clear``). In a depth slot: the full fixed-2x budget for the new level
        (``depth_level``). While parked (its last request's actions landed after it left the slot): it goes to the
        front of the refill order (``parked_clear``)."""
        game.last_clear_wall = self.global_clock.elapsed()
        self.dw_clears[game.slot if game.slot in ("depth", "level0") else "none"] += 1
        self.dw_kept.discard(game.game_id)
        if self.phase != "steady" or game.finished_reason is not None or game.state == "finished":
            return
        if game.state == "active" and game.slot == "level0":
            turn = game.clock.completion_tokens() - game.admit_tokens
            self._dw_set_slot(game, "depth")
            budget = self.level_budget(game)
            game.clock.grant(budget)
            self._record(game, "grant", "level0_clear", budget)
            self._dw_open_tenure(game, "level0_clear", budget)
            self._dw_note(game, "level0_clear", budget=round(budget, 3), t1_seconds=round(game.t1, 3),
                          turn_tokens=turn, window=game.window)
        elif game.state == "active" and game.slot == "depth":
            budget = self.level_budget(game)
            game.clock.grant(budget)
            self._record(game, "grant", "depth_level", budget)
            self._dw_note(game, "depth_level", budget=round(budget, 3), level_tokens=game.level_tokens[-1])
        elif game.state in ("backlog", "parked"):
            game.pending_parked_progress = True
            self._record(game, "progress", "parked_level_reached")
            self._dw_note(game, "parked_clear", was_level0=game.admit_levels == 0,
                          queue_position=self.dw_waiting().index(game))

    def depthwise_summary(self):
        usage = self.slot_usage_totals()
        tenures = []
        for tenure in self.dw_tenures:
            row = dict(tenure)
            if row["end_wall_seconds"] is None:
                game = next(g for g in self.games if g.game_id == row["game"])
                row.update(end_levels=game.levels, end_completion_tokens=game.clock.completion_tokens(),
                           end_inference_seconds=round(game.clock.inference_seconds(), 3), open=True)
            row["levels"] = row["end_levels"] - row["start_levels"]
            row["completion_tokens"] = row["end_completion_tokens"] - row["start_completion_tokens"]
            tenures.append(row)
        counts, per_game = {}, {}
        for decision in self.dw_decisions:
            counts[decision["decision"]] = counts.get(decision["decision"], 0) + 1
            if decision["decision"] in ("depth_park", "depth_readmit", "depth_parked_clear", "level0_rotate",
                                        "level0_clear"):
                row = per_game.setdefault(decision["game"], {})
                row[decision["decision"]] = row.get(decision["decision"], 0) + 1
        level0_turns = {}
        for game in self.games:
            turns = sum(e["event"] == "admit" and e["reason"] in ("level0_start", "level0_rotate")
                        for e in game.admissions)
            if turns:
                level0_turns[game.game_id] = turns
        t1 = {g.game_id: {"t1_seconds": round(g.t1, 3), "level1_wall_seconds": g.time_to_level[0]["wall_seconds"],
                          "level1_tokens": g.level_tokens[0], "levels": g.levels}
              for g in sorted(self.games, key=lambda g: (g.t1 or 0.0, g.index)) if g.t1 is not None}
        clears = sorted(g.time_to_level[0]["wall_seconds"] for g in self.games if g.time_to_level)
        total = sum(row["completion_tokens"] for row in usage.values())
        return {
            "settings": dict(self.depthwise), "regrant_multiple": self.regrant_multiple,
            **({"regrant_floor_tokens": int(self.regrant_floor)} if self.token_clock
               else {"regrant_floor_seconds": self.REGRANT_FLOOR}), "budget_multiple": self.budget_multiple,
            "decision_counts": counts, "decisions": list(self.dw_decisions),
            "slot_usage": usage,
            "token_share": {slot: round(row["completion_tokens"] / total, 4) for slot, row in usage.items()}
            if total else {},
            "t1": t1, "tenures": tenures, "clears_by_slot": dict(self.dw_clears),
            "first_clear_wall_seconds": clears[0] if clears else None,
            "parks": counts.get("depth_park", 0),
            "readmissions": counts.get("depth_readmit", 0) + counts.get("depth_parked_clear", 0),
            "per_game": per_game, "level0_turns": level0_turns,
            "depth": [g.game_id for g in self.dw_slot_games("depth")],
            "level0": [g.game_id for g in self.dw_slot_games("level0")],
            "parked_depth": [g.game_id for g in self.dw_waiting() if g.levels >= 1],
            "games_played": sum(g.readmissions > 0 for g in self.games),
        }

    # --- audition pool -------------------------------------------------------------------

    def pool_capacity(self):
        return max(0, self.active_limit - len(self.focus())) * (self.audition_pool or 0)

    def _pool_stalls(self):
        for game in self.pool_games():
            if self.pool_stall_actions is not None and game.actions_since_novel >= self.pool_stall_actions:
                reason = "pool_stall"
            elif self.pool_action_limit is not None and game.actions_on_level >= self.pool_action_limit:
                reason = "pool_limit"
            elif self.pool_time_limit_seconds is not None and game.clock.budget_exhausted():
                reason = "pool_timeout"
            else:
                continue
            game.stalls += 1
            game.readmissions += 1  # an audition that stalled goes behind the untried games
            self.park(game, reason)

    def _settle_pool(self):
        """Park pool games beyond the capacity the focus games leave; lowest priority first."""
        excess = len(self.pool_games()) - self.pool_capacity()
        for game in sorted(self.pool_games(), key=self.priority, reverse=True)[:max(0, excess)]:
            self.park(game, "pool_close")

    def _fill_pool(self, backlog):
        top = backlog[0]
        room = self.pool_capacity() - len(self.pool_games())
        group = [g for g in backlog if self._order_key(g) == self._order_key(top)]
        for game in group[:max(0, room)]:
            game.pool = True
            game.pool_admissions += 1
            game.state = "active"
            game.clock.grant(self.pool_time_limit_seconds)
            game.clock.resume()
            self._record(game, "admit", "pool", self.pool_time_limit_seconds)

    def _reopen(self, group):
        self.reopens += 1
        if self.reopen_phase1 == "pause":
            for game in self.focus():
                game.state = "held"
                game.held_levels = game.levels
                game.clock.pause()
                self._record(game, "park", "reopen_hold", game.clock.budget)
        for game in group:
            game.pool = True
            game.pool_admissions += 1
            game.state = "active"
            game.clock.grant(None)
            game.clock.resume()
            self._record(game, "admit", "reopen", None)

    def _close_reopen(self, winner):
        for game in self.pool_games():
            if game is not winner:
                self.park(game, "reopen_close")  # order unchanged: no readmission counted
        for game in self.held():
            game.state = "active"
            game.clock.resume()
            self._record(game, "admit", "reopen_resume", game.clock.budget)
            if game.levels != getattr(game, "held_levels", game.levels) and game.t1 is not None:
                game.clock.grant(self.cap_grant(self.level_budget(game), game))
                self._record(game, "grant", "level_reached", game.clock.budget)

    def grant_for(self, game):
        """Budget for a (re)admitted game in the steady state."""
        if self.level_token_allowance is not None:
            return max(0, self.level_budget(game) - game.spent_on_level())
        if self.budget_policy == "fixed-2x":
            if game.levels >= 1 and game.t1 is not None:
                return self.level_budget(game)
            if game.level0_grant is None:  # never parked yet (a backlog game at the switch)
                game.level0_grant = max(game.clock.budget_seconds(), self.regrant_floor)
            return game.level0_grant
        if game.levels >= 1 and game.t1 is not None:
            fresh = (game.levels + 1) * game.t1 - game.spent_on_level()
            if fresh > 0:
                return self.cap_grant(fresh, game)
            spent = game.spent_on_level()
        else:
            spent = game.clock.budget_seconds()
        return self.cap_grant(max(self.regrant_multiple * spent, spent + self.regrant_floor), game)

    def on_level(self, game):
        """A level was reached: a fresh budget for the new level if the game is active."""
        if self._closed:
            return
        if self.level_token_allowance is not None:
            game.retired = False  # only actual progress lifts a retirement
        if self.blocks is not None:
            self._blocks_level(game)
            return
        if self.priority_cfg is not None:
            if game.level_capped:
                # A clear lifts the cap: the new level counts from zero.
                game.level_capped = False
                game.level_cap_limit = 0.0
                self._level_cap_settle(game)
                self.level_cap_counts["lifted"] += 1
                self._priority_note("level_cap_lifted", game)
            return  # no level budgets: the level, actions and tokens enter the game's score at its next handover
        if self.deep is not None:
            self._deep_level(game)
            return
        if self.depthwise is not None:
            self._dw_level(game)
            return
        if game.lane:
            # A background clear: the index re-scores the game at the next tick like any other clear,
            # so it can take a focus slot there.
            self.lane_clears.append({"game": game.game_id, "levels": game.levels,
                                     "wall_seconds": round(self.global_clock.elapsed(), 3),
                                     "level_tokens": game.level_tokens[-1], "lane_requests": game.lane_requests})
        if (self.prioritize_parked_progress and self.phase == "steady"
                and game.state == "parked" and game.finished_reason is None):
            game.pending_parked_progress = True
            self._record(game, "progress", "parked_level_reached")
        if self.phase == "steady" and game.state == "active" and game.pool and game.t1 is not None and self.reopen_phase1 is not None:
            # Reopened phase 1: the first clear takes the free slot; the rest of the group is parked
            # with its order unchanged and held focus games resume.
            game.pool = False
            game.pool_clears += 1
            game.readmissions += 1
            game.clock.grant(self.level_budget(game))
            self._record(game, "grant", "reopen_clear", game.clock.budget)
            self._close_reopen(game)
            return
        if self.phase == "steady" and game.state == "held":
            self._record(game, "progress", "held_level_reached")
            return
        if self.budget_policy == "commit":
            game.stalled_level = False
            if self.phase == "steady" and game.lane and game.state == "parked" and game.finished_reason is None:
                self._commit_lane_clear(game)
            return  # no level budgets; a focus game's level tokens restart with the clear
        if self.phase == "steady" and game.state == "active" and game.pool and game.t1 is not None:
            # The pool's first clear takes the slot as a focus game with the full level budget.
            game.pool = False
            game.pool_clears += 1
            game.readmissions += 1
            game.clock.grant(self.level_budget(game))
            self._record(game, "grant", "pool_clear", game.clock.budget)
            self._settle_pool()
            return
        if self.budget_policy in ("index", "rank"):
            return  # no level budgets: the index (or the ranking) re-scores the game at its next tick
        if self.phase == "steady" and game.state == "active" and game.t1 is not None:
            full = self.level_budget(game) if self.budget_policy == "fixed-2x" else (game.levels + 1) * game.t1
            game.clock.grant(self.cap_grant(full, game))
            self._record(game, "grant", "level_reached", game.clock.budget)

    def wall_end(self):
        if self._closed:
            return
        # Worker cleanup can still deliver a context-eviction callback on the event loop.
        self._closed = True
        for game in self.games:
            if game.state != "finished":
                game.clock.pause()
                self._record(game, "finish", "wall_end")

    def summary(self):
        return {
            "phase": self.phase,
            "budget_clock": self.games[0].clock.budget_clock,
            **({"token_budget": {"unit": "completion_tokens", "floor_tokens": int(self.regrant_floor),
                                 "reference_tokens_per_busy_second": self.TOKEN_REFERENCE_RATE,
                                 "purposes": list(self.games[0].clock.token_purposes)}} if self.token_clock else {}),
            "switch_wall_seconds": self.switch_wall_seconds,
            "active_limit": self.active_limit,
            "opening_switch_level1_games": self.switch_games,
            "max_grant_seconds": self.max_grant_seconds,
            "max_grant_t1_multiple": self.max_grant_t1_multiple,
            "regrant_multiple": self.regrant_multiple,
            "budget_policy": self.budget_policy,
            **({"level_token_allowance": self.level_token_allowance} if self.level_token_allowance else {}),
            "t1_cap_seconds": self.t1_cap_seconds,
            "t1_clip": None if self.t1_clip is None else list(self.t1_clip),
            "index_settings": self.index_cfg,
            "index_swaps": self.index_swaps,
            "index_history": self.index_history,
            "budget_multiple": self.budget_multiple,
            "focus_stall_actions": self.focus_stall_actions,
            "opening_max_wall_seconds": self.opening_max_wall_seconds,
            "opening_min_tokens": self.opening_min_tokens,
            "switch_reason": self.switch_reason,
            "audition_pool": self.audition_pool,
            "pool_stall_actions": self.pool_stall_actions,
            "pool_action_limit": self.pool_action_limit,
            "level0_order": self.level0_order,
            **({} if self.backlog_key is None else {"backlog_key": self.backlog_key,
                                                    "same_tick_regrants": self.same_tick_regrants}),
            "pool_time_limit_seconds": self.pool_time_limit_seconds,
            "reopen_phase1": self.reopen_phase1,
            "reopens": self.reopens,
            "normalization": None if self.games[0].clock.normalization is None else self.games[0].clock.normalization.summary(),
            "backlog_policy": self.backlog_policy,
            "progress_order": None if self.backlog_policy != "progress" else {
                "model": self.progress_model,
                "prior_clears": self.PROGRESS_PRIOR_CLEARS,
                **({"prior_tokens": int(self.progress_prior)} if self.token_clock
                   else {"prior_seconds": self.PROGRESS_PRIOR_SECONDS}),
                "stall_scale": self.PROGRESS_STALL_SCALE, "default_levels": self.PROGRESS_DEFAULT_LEVELS,
                "novelty_prior": self.PROGRESS_NOVELTY_PRIOR},
            "prioritize_parked_progress": self.prioritize_parked_progress,
            **({} if self.background_lane is None else {"background_lane": self.lane_summary()}),
            **({} if self.commit is None else {"commit": self.commit_summary()}),
            **({} if self.deep is None else {"deep_explore": self.deep_summary()}),
            **({} if self.depthwise is None else {"depthwise": self.depthwise_summary()}),
            **({} if self.rank is None else {"rank": self.rank_summary()}),
            **({} if self.blocks is None else {"blocks": self.blocks_summary()}),
            **({} if self.priority_cfg is None else {"priority": self.priority_summary()}),
            "events": list(self.events),
        }

    def lane_summary(self):
        games = {g.game_id: {"requests": g.lane_requests, "completion_tokens": g.lane_tokens(),
                             "clears": sum(c["game"] == g.game_id for c in self.lane_clears)}
                 for g in self.games if g.lane_requests}
        return {**self.background_lane, "requests": sum(g.lane_requests for g in self.games),
                "completion_tokens": sum(g.lane_tokens() for g in self.games),
                "clears": list(self.lane_clears), "promotions": list(self.lane_promotions), "games": games}
