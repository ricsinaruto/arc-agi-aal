"""Level bench: many short, independent level attempts (RL groups, held-out reads).

Each start point (``start_points``: a recorded action prefix, a restored recorded try (``restore``), a fresh level-1
``segment`` or a whole game played ``through``) is one Duck-transfer agent with the study's full harness (context drain,
dfr, hx, loop rule), which plays until the level clears, its completion tokens reach the budget (checked before each
request) or ``attempt_wall_seconds`` pass. ``concurrency`` attempts run at once through the KV gate (one shared pool
and gate per server under rl_seq: ``SHARED_SLOTS_BY_URL``); the plan is ``passes`` seeded permutations of the points
(``arms`` interleaved), so runs with the same settings pair on (point, pass). Attempts running at the bench wall are
censored and left out of every headline metric.
"""

import asyncio
import copy
import json
import math
import random
import statistics
import time
from pathlib import Path

import httpx

from arc_infra import KEEPALIVE_EXPIRY
from arc_infra.arcade import make_arcade
from arc_infra.config import Config as DuckConfig
from arc_infra.fleet import SessionClient
from duck_transfer.agent import GameplayRequestExpired

from .clock import ActiveClock, GlobalClock

from .duck import ArenaTransferDuckAgent
from .gate import GateSlot, KVGate
from .level_points import check_replay, level_score
from . import phase
from .scheduler import GameRecord

DEFAULTS = {
    "split": "dev",                 # dev | holdout | all
    "games": None,                  # optional list of 4-character game prefixes
    "concurrency": 16,              # attempts running at once (and gate slots)
    "window": 49152,                # agent context per attempt
    "token_budget": 40000,          # completion tokens per attempt (every purpose)
    "level_token_cap": None,        # removed; null only
    "token_budget_by_level": None,  # removed; null only
    "token_budget_by_point": None,  # {"<point id>": tokens}: the budget of that start
    "rollout": None,                # {"top_logprobs", "token_ids"}: every reply recorded (arena.rollout_record)
    "sampling": None,               # {temperature, top_p, ...}: sent on every request in place of the harness's sampler
    "seed_base": None,              # every request of pass k carries vLLM seed seed_base + k (paired reads)
    "attempt_wall_seconds": 1800,   # safety cap per attempt
    "passes": 3,                    # seeded permutations of the start points, in order
    "order_seed": 0,
    "progress_seconds": 30,         # a live progress line at least this often (at most 60)
    "launch_cutoff_seconds": 120,   # no new attempt when less bench wall remains
    "min_wall_seconds": 600,        # the bench refuses to start with less
    "start_points": "configs/level_bench_start_points.json",
    "capacity_tokens": None,        # external endpoint: gate capacity in tokens (no server log)
    "gate_completion_reserve": None,  # completion tokens the gate reserves per request (None: the completion cap)
    "serving": None,                # recorded label of the serving profile
    "arms": None,                   # [{"name", "study": {...}}]; None: one base arm
    "first_pass": 0,                # index of the first pass (a later run extends an earlier one's passes)
    "lead_token_budget": 80000,     # removed; default only
    "action_cap": None,             # removed; null only
}
GONE = {"lead_token_budget": 80000, "action_cap": None, "token_budget_by_level": None, "level_token_cap": None}
ARM_KEYS = ("name", "study")
SPLITS = ("dev", "holdout", "all")
UNSUPPORTED = ("cross_game_notes", "explorer", "background_lane", "commit", "llm_backlog", "audition_pool",
               "gate_overcommit")  # trial-level services and removed study keys
HEADLINE_EXCLUDED = ("censored", "error", "replay_mismatch", "recorded")
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"}


def settings(raw):
    """Validated ``study["level_bench"]`` merged over the defaults."""
    raw = dict(raw or {})
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown level-bench settings: {sorted(unknown)}")
    merged = {**DEFAULTS, **raw}
    gone = sorted(k for k, v in GONE.items() if merged[k] != v)
    if gone:
        raise ValueError(f"level_bench: removed settings {gone}")
    if merged["split"] not in SPLITS:
        raise ValueError(f"level_bench.split must be one of {SPLITS}")
    for key in ("concurrency", "window", "token_budget", "passes", "progress_seconds"):
        if type(merged[key]) is not int or merged[key] < 1:
            raise ValueError(f"level_bench.{key} must be a positive integer")
    if merged["progress_seconds"] > 60:
        raise ValueError("level_bench.progress_seconds must be at most 60 (a line every minute)")
    for key in ("attempt_wall_seconds", "launch_cutoff_seconds", "min_wall_seconds"):
        if type(merged[key]) not in (int, float) or merged[key] < 0:
            raise ValueError(f"level_bench.{key} must be a non-negative number")
    if type(merged["first_pass"]) is not int or merged["first_pass"] < 0:
        raise ValueError("level_bench.first_pass must be a non-negative integer")
    if merged["arms"] is not None:
        arms = merged["arms"]
        if (not isinstance(arms, list) or not arms or any(not isinstance(a, dict) or set(a) - set(ARM_KEYS)
                                                           or not a.get("name") for a in arms)
                or len({a["name"] for a in arms}) != len(arms)):
            raise ValueError("level_bench.arms is a list of {name, study} with distinct names")
    sampling = merged["sampling"]
    if sampling is not None and (not isinstance(sampling, dict) or not sampling or set(sampling) - SAMPLING_KEYS
                                 or any(type(v) not in (int, float) or isinstance(v, bool) for v in sampling.values())):
        raise ValueError(f"level_bench.sampling is {{key: number}} over {sorted(SAMPLING_KEYS)}")
    if merged["rollout"] is not None:
        from .rollout_record import settings as rollout_settings

        merged["rollout"] = rollout_settings(merged["rollout"])
    by_point = merged["token_budget_by_point"]
    if by_point is not None and (not isinstance(by_point, dict) or not by_point or any(
            type(v) is not int or v < 1 for v in by_point.values())):
        raise ValueError("level_bench.token_budget_by_point is {point id: positive integer} or null")
    if merged["seed_base"] is not None and (type(merged["seed_base"]) is not int or merged["seed_base"] < 0):
        raise ValueError("level_bench.seed_base must be a non-negative integer")
    if merged["games"] is not None and (not isinstance(merged["games"], list) or not merged["games"]):
        raise ValueError("level_bench.games must be a non-empty list of game prefixes")
    if merged["gate_completion_reserve"] is not None and (type(merged["gate_completion_reserve"]) is not int
                                                         or merged["gate_completion_reserve"] < 1):
        raise ValueError("level_bench.gate_completion_reserve is a positive integer or null")
    if merged["capacity_tokens"] is not None and (type(merged["capacity_tokens"]) is not int
                                                   or merged["capacity_tokens"] < merged["window"]):
        raise ValueError("level_bench.capacity_tokens must be an integer of at least one window")
    return merged


def load_points(path, cfg):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != 1:
        raise ValueError("Unknown start-point schema")
    points = [p for p in data["points"] if cfg["split"] == "all" or p["split"] == cfg["split"]]
    if cfg["games"]:
        wanted = {str(g)[:4] for g in cfg["games"]}
        points = [p for p in points if p["game"][:4] in wanted]
    if not points:
        raise ValueError("No start points selected")
    return sorted(points, key=lambda p: p["id"])


COLD = {"name": "base", "study": {}}
# removed point kinds (lead, warm, in-level reset, per-point level caps, priority order)
POINT_GONE = ("lead_levels", "warm", "level_start_actions", "level_caps", "priority")


def token_budget(cfg, point):
    return int((cfg.get("token_budget_by_point") or {}).get(point.get("id"), cfg["token_budget"]))


def plan(points, passes, seed, arms=None, first_pass=0):
    """``[(pass, point, arm)]``: each pass a seeded permutation of the points (sorted by id first); within a pass the
    arms rotate from point to point, so a run cut short stays balanced."""
    gone = sorted({k for point in points for k in POINT_GONE if point.get(k)})
    if gone:
        raise ValueError(f"level bench: removed point kinds {gone}")
    arms = [{**COLD, **a} for a in arms] if arms else [COLD]
    out = []
    for index in range(first_pass, first_pass + passes):
        order = list(points)
        random.Random(f"{seed}:{index}").shuffle(order)
        for turn in range(len(arms)):
            for place, point in enumerate(order):
                out.append((index, point, arms[(place + turn + index) % len(arms)]))
    return out


def bench_priority(cfg, study):
    """Slot priority of a bench in a shared pool (higher takes first): its largest token cap."""
    return max([int(v) for v in (cfg.get("token_budget_by_point") or {}).values()] or [0])


class ReplayMismatch(RuntimeError):
    """The replayed prefix did not reproduce the recorded level count and board."""


class PrefixArcade:
    """The arcade the attempt's agent sees: ``make`` replays the start point's prefix first."""

    def __init__(self, arc, point):
        self.arc, self.point, self.replay = arc, point, None

    def make(self, game_id, **kwargs):
        env = self.arc.make(game_id, **kwargs)
        if env is None:
            return None
        started = time.monotonic()
        ok, detail, _ = check_replay(env, self.point)
        self.replay = {"ok": ok, "detail": detail, "actions": len(self.point["prefix"]),
                       "seconds": round(time.monotonic() - started, 3)}
        if not ok:
            raise ReplayMismatch(f"{self.point['id']}: {detail}")
        return env


class LevelBenchAgent(ArenaTransferDuckAgent):
    """The arena's Duck-transfer agent, stopped at the first level clear or at the token budget."""

    bench_start = 0
    bench_budget = None
    bench_outcome = None  # cleared | budget | won | censored ...
    bench_clear = None
    bench_through = False       # play-through point: no stop at a level clear
    bench_levels = None         # play-through: [{"level", "completion_tokens", "steps"}] per cleared level
    bench_actions = None        # every action the attempt executed, in order (restore checks)
    bench_restore = None        # arena.restore: a recorded attempt replayed to a request, then live (F-small)
    bench_restore_state = None  # the state check at the first live request
    # in-run harvest (envs): a cleared segment sends one recorded 1-token request at the next board, the
    # restore point of a new start one level deeper; never trained (rollout row ``harvest``)
    bench_harvest = False
    bench_harvest_state = None  # None | "due" (cleared, request owed) | "sending" | "sent"
    bench_harvest_row = None    # {"request", "prompt_tokens", "step", "levels"} of the recorded request

    def bench_setup(self, *, start_level, budget, prefix=0, through=False):
        self.bench_start, self.bench_budget = int(start_level), int(budget)
        self.bench_through, self.bench_levels = bool(through), []
        if through and (prefix or start_level):
            raise ValueError("level bench: a play-through point is a fresh game")
        self.bench_outcome = self.bench_clear = None
        self.bench_actions = []
        return self

    def remaining(self):
        if self.bench_outcome is not None and self.bench_harvest_state not in ("due", "sending"):
            return 0.0
        return super().remaining()

    def bench_harvest_due(self):
        """At a clear (restore and target branches of ``step``): owe the harvest request, and mark the clear like a
        play-through level (``level_bench_level``) so restore.load / level_starts / unsafe_from read the snapshot as
        a bank start. Actions the clearing reply still holds run as usual (a replay runs them too)."""
        if self.bench_harvest and self.bench_outcome == "cleared" and self.bench_harvest_state is None:
            self.bench_harvest_state = "due"
            self.workspace.append("level_bench_level", step=self.steps, level=int(self.obs.levels),
                                  completion_tokens=self.game.clock.completion_tokens(), steps=self.steps)

    def bench_gate(self):
        """Before every request: an attempt past its budget (or finished) sends nothing more."""
        tokens = self.game.clock.completion_tokens()
        if self.bench_outcome is None and tokens - self.bench_base_tokens() >= self.bench_budget:
            self.bench_outcome = "budget"
            self.workspace.append("level_bench_budget", step=self.steps, completion_tokens=tokens)
        if self.bench_outcome is not None:
            raise GameplayRequestExpired(f"Level-bench attempt finished: {self.bench_outcome}")

    def bench_base_tokens(self):
        """Completion tokens that do not count against the budget: the restored prefix."""
        return self.bench_restore["tokens"] if self.bench_restore is not None else 0

    def bench_board(self):
        from arc_infra.perception import board_hash

        return board_hash(self.obs.board) if self.obs is not None and self.obs.board is not None else None

    def replaying(self):
        """A recorded prefix is being replayed (restore): its tool calls must reproduce the recording."""
        # the state check runs at the first live chat, after the last replayed reply's tools
        return self.bench_restore is not None and self.bench_restore_state is None

    def bench_restore_live(self):
        return self.bench_restore is not None and self.rollout.restore_at > self.bench_restore["request"]

    def bench_restore_check(self):
        """At the restored attempt's first live request: the game and the action list as the recording had them."""
        stop = self.bench_restore["stop"]
        now = {"steps": self.steps, "levels": int(self.obs.levels), "board_hash": self.bench_board()}
        problems = [f"{key} {now[key]} != {stop[key]}" for key in now if now[key] != stop[key]]
        if self.bench_actions != self.bench_restore["actions"]:
            problems.append(f"actions differ ({len(self.bench_actions)} vs {len(self.bench_restore['actions'])})")
        self.bench_restore_state = {**now, "ok": not problems, "detail": "; ".join(problems) or "ok"}
        self.workspace.append("level_bench_restore_state", **self.bench_restore_state)
        self.workspace.append("level_bench_level_start", step=self.steps, levels=int(self.obs.levels),
                              level_start_step=int(self.bench_restore["level_start_step"]))
        if problems:
            self.bench_outcome = "replay_mismatch"

    async def chat(self, messages, tools):
        if (self.bench_restore is not None and self.bench_restore_state is None
                and self.rollout.restore_at == self.bench_restore["request"]):
            self.bench_restore_check()
        if self.bench_harvest_state == "due":  # the harvest request: past the gate once, 1 token (RolloutHTTP)
            self.bench_harvest_state = "sending"
        else:
            self.bench_gate()
        return await super().chat(messages, tools)

    def bench_mark_clear(self, outcome, actions, tokens):
        self.bench_outcome = outcome
        self.bench_clear = {"actions": actions, "completion_tokens": tokens,
                            "wall_seconds": round(self.game.clock.global_clock.elapsed(), 3),
                            "requests": len(self.game.requests)}
        self.workspace.append("level_bench_clear", step=self.steps, **self.bench_clear)

    async def step(self, action):
        await super().step(action)
        if self.bench_actions is not None:
            self.bench_actions.append(action.name + (f" {action.x} {action.y}" if action.name == "ACTION6" else ""))
        if self.bench_restore is not None:  # a segment: the restored level, live, until it clears
            if (self.bench_outcome is None and self.bench_restore_live()
                    and self.obs.levels > self.bench_restore["stop"]["levels"]):
                self.bench_mark_clear("won" if self.obs.state == "WIN" else "cleared",
                                      self.steps - self.bench_restore["level_start_step"],
                                      self.game.clock.completion_tokens() - self.bench_restore["tokens"])
                self.bench_harvest_due()
            return
        if self.bench_through:
            if self.bench_outcome is None and self.obs.levels > len(self.bench_levels):
                mark = {"completion_tokens": self.game.clock.completion_tokens(), "steps": self.steps}
                for level in range(len(self.bench_levels) + 1, self.obs.levels + 1):
                    self.bench_levels.append({"level": level, **mark})
                    self.workspace.append("level_bench_level", step=self.steps, level=level, **mark)
                if getattr(self, "bench_on_level", None) is not None:  # the read's opening (ReadOpening)
                    self.bench_on_level(self.obs.levels)
                if self.obs.state == "WIN" or self.obs.levels >= self.obs.total_levels:
                    self.bench_mark_clear("won", self.steps, mark["completion_tokens"])
            return
        if self.bench_outcome is None and self.obs.levels > self.bench_start:
            self.bench_mark_clear("cleared", self.steps, self.game.clock.completion_tokens())
            if self.obs.state != "WIN":
                self.bench_harvest_due()


def request_actions(log_path):
    """``(gameplay requests, requests without an executed action before the next request)``."""
    requests = noaction = 0
    acted = True
    try:
        with open(log_path) as file:
            for line in file:
                if line.startswith('{"kind":"model"'):
                    rec = json.loads(line)
                    if rec.get("purpose", "gameplay") != "gameplay":
                        continue
                    requests += 1
                    noaction += not acted
                    acted = False
                elif line.startswith('{"kind":"observation"') and '"action":null' not in line[:200]:
                    acted = True
    except OSError:
        return 0, 0
    noaction += requests > 0 and not acted
    return requests, int(noaction)


def summarize(attempts, *, wall_seconds=None, budget=None):
    """Headline metrics over completed attempts (censored, errored and mismatched ones excluded)."""
    done = [a for a in attempts if a["outcome"] not in HEADLINE_EXCLUDED]
    clears = [a for a in done if a["cleared"]]
    tokens = sum(a["completion_tokens"] for a in done)
    all_tokens = sum(a["completion_tokens"] for a in attempts)
    return {
        "token_budget": budget,
        "attempts_completed": len(done),
        "attempts_censored": sum(a["outcome"] == "censored" for a in attempts),
        "attempts_failed": sum(a["outcome"] in ("error", "replay_mismatch") for a in attempts),
        "outcomes": {k: sum(a["outcome"] == k for a in attempts) for k in sorted({a["outcome"] for a in attempts})},
        "clears": len(clears),
        "clear_rate": round(len(clears) / len(done), 4) if done else None,
        "completion_tokens": tokens,
        "clears_per_10k_tokens": round(len(clears) * 1e4 / tokens, 4) if tokens else None,
        "median_tokens_to_clear": statistics.median(a["tokens_to_clear"] for a in clears) if clears else None,
        "mean_level_score": round(sum(a["level_score"] for a in done) / len(done), 3) if done else None,
        "all_completion_tokens": all_tokens,
        "censored_completion_tokens": sum(a["completion_tokens"] for a in attempts if a["outcome"] == "censored"),
        "tokens_per_second": round(all_tokens / wall_seconds, 1) if wall_seconds else None,
    }


class SharedSlots:
    """One pool of attempt slots for every bench on one play server (rl_seq / rl_drivers: ``SHARED_SLOTS_BY_URL``):
    an attempt starts only on a free slot, so the running attempts' contexts stay in the prefix cache; a bench with a
    larger token budget takes slots first, then first come. ``shared_gate`` (v8.5p admission): one exact KV gate for
    every bench on the server. ``capacity_share``: one of N driver processes takes 1/N of the server's KV capacity."""

    def __init__(self, slots, kv_fill=None, project=None, stagger_s=0.0, shared_gate=False, capacity_share=1.0):
        if kv_fill or project or stagger_s:
            raise ValueError("mg active_kv_fill / kv_project / admit_stagger_s are gone (rl-min)")
        self.slots, self.used, self.peak, self.pending = int(slots), 0, 0, {}
        self.kv_fill, self.shared_gate, self.capacity_share = None, bool(shared_gate), float(capacity_share)
        self.capacity, self.gate, self.budget = None, None, None  # the gate's cache budget: capacity in its units
        self.live, self.kv_waits, self.turn_waits = set(), 0, 0
        self.entered, self.enter_seq, self.sending = {}, 0, {}  # try -> entry order; try -> requests in flight

    def want(self, key, priority, count):
        self.pending[key] = (priority, int(count))

    def remaining(self, key, count):
        if key in self.pending:
            self.pending[key] = (self.pending[key][0], int(count))

    def _units(self, tokens):
        return self.budget.cost(tokens) if self.budget is not None else int(tokens)

    def load(self):
        """The live tries' next requests (prompt + cap) in gate units; a try before its first request as the median."""
        known = sorted(self._units(s.next_estimate + s.cap) for s in self.live if s.next_estimate > 0)
        new = known[len(known) // 2] if known else self._units(32000)
        return sum(self._units(s.next_estimate + s.cap) if s.next_estimate > 0 else new for s in self.live)

    def share(self, capacity):
        """``capacity`` x capacity_share (unchanged at the default 1.0)."""
        return capacity if self.capacity_share == 1.0 else int(capacity * self.capacity_share)

    def capacity_tokens(self):
        """The engine's whole KV pool in tokens (a block budget's blocks at its longest row's tokens per block, without
        the budget's margin), x capacity_share."""
        if self.budget is None:
            return self.capacity
        layout = self.budget.layout
        return self.share(int(layout["num_blocks"] * layout["costs"][-1]["tokens"] / layout["costs"][-1]["blocks"]))

    def take(self, key):
        """A slot for one of ``key``'s tries: free slots and no higher-priority bench waiting."""
        priority, count = self.pending.get(key, (0, 0))
        if count <= 0 or self.used >= self.slots:
            return False
        if any(p > priority and n > 0 for k, (p, n) in self.pending.items() if k != key):
            return False
        self.used += 1
        self.peak = max(self.peak, self.used)
        self.pending[key] = (priority, count - 1)
        return True

    def enter(self, slot, key=None):
        self.live.add(slot)
        self.enter_seq += 1
        self.entered[id(slot)] = self.enter_seq

    def give(self, slot=None):
        self.used -= 1
        self.live.discard(slot)
        self.entered.pop(id(slot), None)
        self.sending.pop(id(slot), None)

    async def turn(self, slot):
        """The gate's turn wait (no turn cap: every live try may send)."""

    def begin(self, slot):
        """``slot`` sends a request now: count it in flight until ``end``. True when counted (a live try of this pool)."""
        key = id(slot)
        if key not in self.entered:
            return False
        self.sending[key] = self.sending.get(key, 0) + 1
        return True

    def end(self, slot):
        """One of ``slot``'s requests is done (a try may have an aside out beside its main request)."""
        key = id(slot)
        n = self.sending.get(key, 0) - 1
        if n > 0:
            self.sending[key] = n
        else:
            self.sending.pop(key, None)

    def done(self, key):
        self.pending.pop(key, None)


class ReadOpening:
    """The submission's opening on a held-out read (scheduler: reached = games with levels >= 1, no
    reopen): the gate's continuation affinity (``opening_burst_requests``) holds until ``opening_switch_level1_games``
    distinct read tries have cleared level 1, then ends for good."""

    def __init__(self, switch, log):
        self.switch, self.log, self.games, self.open = int(switch), log, set(), int(switch) > 0

    def __call__(self):
        return self.open

    def reached(self, label, levels):
        if not self.open or int(levels) < 1 or label in self.games:
            return
        self.games.add(label)
        if len(self.games) >= self.switch:
            self.open = False
            self.log({"event": "opening_end", "reached": len(self.games), "switch": self.switch, "last": label})


def read_opening(study, gate, log):
    """Wire the opening into a held-out read's gate (study.hread with opening_burst_requests > 1); None otherwise
    (training benches keep burst 1, so their admissions are unchanged)."""
    burst = int(study.get("opening_burst_requests") or 1)
    if not study.get("hread") or burst <= 1:
        return None
    if not 1 <= burst <= 3:
        raise ValueError("opening_burst_requests must be between 1 and 3")
    opening = ReadOpening(study.get("opening_switch_level1_games") or 0, log)
    gate.opening_burst_requests, gate.opening = burst, opening
    log({"event": "opening_start", "burst": burst, "switch": opening.switch})
    return opening


PROGRESS = {"done": 0, "clears": 0}  # finished tries since the caller's last reset
# the RL loop's callable(bench name, finished row), called once per finished
# try beside the PROGRESS counters
PROGRESS_HOOK = None
RESTORE_CACHE = {}  # (attempt, request) -> restore.load(...): the event loop parsed each start's rollout once per try
SHARED_SLOTS_BY_URL = {}  # RL redesign (arena.rl_seq): one pool per play server, by the bench's base URL
PHASE_GATE = None  # RL redesign (arena.phase.PhaseGate): every bench client's transport over the N play servers
# G-long v5 N3 (27 Oct): every group's bench polled /metrics every 5 s (3 s timeout) and spawned nvidia-smi every
# 30 s; ~23 benches flooded the server and the guard read nothing for 270 s. rl_seq sets False: the bench keeps its
# start and end samples, the guard (rl_seq.server_guard) is the one periodic reader.
ENGINE_POLL = True


def duck_config(study, source, *, window, cap, wall, base_url):
    """The agent's config: configs/duck_controls.yaml with the study's window, reply cap, sampling and server."""
    cfg = DuckConfig.load(str(Path(source) / "configs/duck_controls.yaml"))
    cfg.max_model_len = int(window)
    cfg.max_output_tokens = int(cap)
    cfg.reasoning_effort = study["reasoning_effort"]
    cfg.enable_thinking = True
    cfg.reasoning_history = study["reasoning_history"]
    cfg.temperature = float(study["temperature"])
    cfg.gameplay_seconds = wall
    cfg.parallel_games = int(study["generation_slots"])
    cfg.max_num_seqs = max(cfg.max_num_seqs, cfg.parallel_games)
    cfg.base_url = base_url
    cfg.serving_backend = "vllm"
    cfg.model = study.get("served_model", cfg.model)
    cfg.speculative_tokens = study["server"]["mtp_tokens"]  # client-side lookahead: the MTP depth
    return cfg


def gate_capacity(study, cfg, server_dir, window):
    """``(capacity, budget, choice)`` for the KV gate on this backend."""
    if cfg["capacity_tokens"] is not None:
        return int(cfg["capacity_tokens"]), None, {"units": "tokens", "source": "level_bench.capacity_tokens"}
    if study.get("cache_admission") != "hybrid_blocks":
        raise ValueError("level bench: cache_admission is hybrid_blocks (rl-min)")
    from .cache import BlockBudget

    budget = BlockBudget.load(server_dir, float(study["kv_margin"]))
    if budget.cost(window) > budget.capacity:
        raise RuntimeError("One attempt window does not fit the served cache; no silent context downgrade")
    return budget.capacity, budget, {"units": getattr(budget, "units", "cache_blocks"), "source": "hybrid cache blocks",
                                     "full_window_cost": budget.cost(window), "layout_rule": budget.layout.get("rule")}


class LevelBench:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir, analysis_python=None,
                 base_url, deadline_epoch, log=print, arcade=None, transport=None):
        self.study, self.trial = study, trial
        self.cfg = settings(study.get("level_bench"))
        for key in UNSUPPORTED:
            if study.get(key):
                raise ValueError(f"The level bench does not support the study key {key!r}")
        self.source, self.games_dir, self.output = Path(source), games_dir, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.server_dir, self.base_url = server_dir, base_url
        self.analysis_python = analysis_python or "python3"
        self.deadline_epoch, self.log, self.arc = float(deadline_epoch), log, arcade
        self.transport = transport  # tests: an httpx transport in place of the server
        self.cap = int(study["completion_cap"])
        self.attempts, self.requests = [], []
        self.monitor = None
        self.read_metrics = None
        if study.get("hread"):
            from .hread import ReadMetrics

            self.read_metrics = ReadMetrics(self.output, study["hread"])
        self.policy_version = study.get("policy_version")
        self.admission_log = (self.output / "admission.jsonl").open("a")

    def gate_log(self, entry):
        self.admission_log.write(json.dumps({"epoch": time.time(), **entry}) + "\n")

    def wall_seconds(self):
        wanted = float(self.study.get("wall_seconds") or 0)
        available = self.deadline_epoch - time.time()
        wall = min(wanted, available) if wanted else available
        if wall < self.cfg["min_wall_seconds"]:
            raise TimeoutError(f"Only {available:.0f} s left for the level bench")
        return wall

    async def run(self):
        from arc_infra.metrics import GpuMonitor

        cfg = self.cfg
        window = cfg["window"]
        if window <= self.cap:
            raise ValueError("The attempt window must exceed the completion cap")
        capacity, budget, choice = gate_capacity(self.study, cfg, self.server_dir, window)
        request_capacity = capacity
        points = load_points(self.source / cfg["start_points"], cfg)
        work = plan(points, cfg["passes"], cfg["order_seed"], cfg["arms"], cfg["first_pass"])
        bench_wall = GlobalClock(self.wall_seconds())
        agent_cfg = duck_config(self.study, self.source, window=window, cap=self.cap,
                                wall=cfg["attempt_wall_seconds"], base_url=self.base_url)
        agent_cfg.max_actions = 10 ** 6  # replayed history counts as steps; the token budget ends the attempt
        agent_cfg.parallel_games = cfg["concurrency"]
        agent_cfg.max_num_seqs = max(agent_cfg.max_num_seqs, cfg["concurrency"])
        self.config_dict = agent_cfg.to_dict()
        pool = SHARED_SLOTS_BY_URL.get(self.base_url.rstrip("/"))  # None: this bench alone
        if pool is not None and pool.shared_gate:  # one gate for every bench on this server
            if pool.capacity is None:
                pool.capacity, pool.budget = pool.share(capacity), budget
            pool.gate = pool.gate or KVGate(pool.share(request_capacity), slots=max(cfg["concurrency"], int(self.study.get("generation_slots") or 0)),
                                            budget=budget)
            gate = pool.gate
        else:
            gate = KVGate(request_capacity, slots=cfg["concurrency"], log=self.gate_log, budget=budget)
        self.read_opening = read_opening(self.study, gate, self.gate_log)
        arc = self.arc or make_arcade("offline", self.games_dir, self.output / "recordings")
        card = arc.open_scorecard(tags=["level-bench", self.trial["name"]])
        self.checks = (arc, card)
        # idle connections retire before the server's 5 s keep-alive closes them (as the arena's trial client): at
        # httpx's default 5 s a reused connection raced the server's close (base ref: 58 ReadErrors in 4,236 requests)
        shared = httpx.AsyncClient(base_url=self.base_url.rstrip("/") + "/", trust_env=False,
                                   transport=self.transport or PHASE_GATE,
                                   limits=httpx.Limits(max_connections=256, max_keepalive_connections=128,
                                                       keepalive_expiry=KEEPALIVE_EXPIRY))
        for adapter in self.study.get("load_adapters") or []:  # a standalone read of trained adapters (RL, 25 Oct):
            # served under their own names before any try (arms pick one through study.served_model)
            if self.study.get("hread"):  # the held-out read: the adapter's training mask is the server's, or no read
                from .hread import mask_problem

                problem = mask_problem(adapter["path"], self.study)
                if problem:
                    raise RuntimeError(f"held-out read refuses adapter {adapter['name']}: {problem}")
            got = await shared.post("load_lora_adapter", json={"lora_name": adapter["name"], "lora_path": adapter["path"],
                                                                  "is_3d_lora_weight": True}, timeout=600)
            if got.status_code >= 400 and "already" not in got.text:
                raise RuntimeError(f"load_lora_adapter {adapter['name']}: {got.status_code} {got.text[:200]}")
            self.log(json.dumps({"event": "adapter_loaded", **adapter}), flush=True)
        monitor = self.monitor = GpuMonitor(self.output, self.base_url)
        header ={"event": "level_bench_start", "trial": self.trial["name"], "points": len(points),
                  "planned_attempts": len(work), "split": cfg["split"], "window": window, "cap": self.cap,
                  "token_budget": cfg["token_budget"], "concurrency": cfg["concurrency"],
                  "gate_capacity": capacity, "gate": choice, "wall_seconds": round(bench_wall.seconds, 1),
                  "serving": cfg["serving"], "server": self.study.get("server"),
                  "reasoning_effort": self.study.get("reasoning_effort")}
        self.log(json.dumps(header), flush=True)
        (self.output / "level-bench-plan.json").write_text(json.dumps(
            {**header, "plan": [{"pass": p, "point": pt["id"], "arm": arm["name"]} for p, pt, arm in work]}, indent=1))
        await monitor.sample()
        bench_wall.release()
        monitor.start = bench_wall.start
        sampling = asyncio.create_task(monitor.run() if ENGINE_POLL else asyncio.sleep(0))
        running = {}
        next_index = 0
        last_progress = -math.inf
        key = object()
        priority = bench_priority(cfg, self.study)
        if pool is not None:
            pool.want(key, priority, len(work))

        def take():
            if pool is None:
                return "none"
            if pool.take(key):
                pool.remaining(key, len(work) - next_index - 1)
                return pool
            return None

        try:
            while True:
                while (next_index < len(work) and len(running) < cfg["concurrency"]
                       and bench_wall.remaining() > cfg["launch_cutoff_seconds"]):
                    chosen = take()
                    if chosen is None:
                        break
                    index = next_index
                    next_index += 1
                    attempt = self.make_attempt(index, *work[index], gate=gate, arc=arc, card=card, shared=shared,
                                                agent_cfg=agent_cfg, window=window)
                    attempt["pool"] = chosen if chosen != "none" else None
                    # turn gate on the server's pool (v6k: server 1 at KV 0.89 with 73 senders; slots bound starts only)
                    attempt["slot"].turn_pool = attempt["pool"]
                    if attempt["pool"] is not None:
                        attempt["pool"].enter(attempt["slot"])
                    running[asyncio.create_task(self.play(attempt))] = attempt
                if not running:
                    if (pool is not None and next_index < len(work)
                            and bench_wall.remaining() > cfg["launch_cutoff_seconds"]):
                        await asyncio.sleep(0.5)  # waiting for a shared slot
                        continue
                    break
                done, _ = await asyncio.wait(list(running), timeout=min(1.0, max(0.05, bench_wall.remaining())),
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    finished = running.pop(task)
                    if finished.get("pool") is not None:
                        finished["pool"].give(finished["slot"])
                if bench_wall.elapsed() - last_progress >= cfg["progress_seconds"]:
                    last_progress = bench_wall.elapsed()
                    self.progress(bench_wall, running.values(), gate, len(work), next_index)
                if bench_wall.remaining() <= 0:
                    break
        finally:
            for attempt in running.values():
                # An attempt that already cleared or spent its budget is only winding down: not censored.
                if attempt["agent"].bench_outcome is None:
                    attempt["agent"].bench_outcome = "censored"
                    attempt["censored"] = True
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            if pool is not None:
                for attempt in running.values():
                    if attempt.get("pool") is not None:
                        attempt["pool"].give(attempt["slot"])
                pool.done(key)
            sampling.cancel()
            await asyncio.gather(sampling, return_exceptions=True)
            await monitor.sample()
            elapsed = bench_wall.elapsed()
            (self.output / "requests.json").write_text(json.dumps(self.requests))
            await shared.aclose()
            try:
                await asyncio.to_thread(arc.close_scorecard, card)
            except Exception as exc:  # noqa: BLE001 - the bench does not use the scorecard's score
                self.log(json.dumps({"event": "level_bench_scorecard_close_failed", "error": str(exc)[:300]}), flush=True)
            self.admission_log.close()
        self.progress(bench_wall, [], gate, len(work), next_index, final=True)
        rows = [a["row"] for a in sorted(self.attempts, key=lambda a: a["index"])]
        summary = summarize(rows, wall_seconds=elapsed, budget=cfg["token_budget"])
        engine = monitor.summary()
        errors = [f"{r['label']}: {r['error']}" for r in rows if r["outcome"] in ("error", "replay_mismatch")]
        status = "complete" if not errors and engine.get("valid") else "complete_with_errors"
        report = {"trial": self.trial, "status": status, "settings": cfg, "window": window, "completion_cap": self.cap,
                  "reasoning_effort": self.study.get("reasoning_effort"), "temperature": self.study.get("temperature"),
                  "server": self.study.get("server"), "served_model": self.study.get("served_model"),
                  "gate": {**gate.snapshot(), "choice": choice}, "engine": engine, "wall_seconds": round(elapsed, 1),
                  "planned_attempts": len(work), "launched_attempts": next_index, "points": len(points),
                  "summary": summary, "task_errors": errors, "config": self.config_dict, "attempts": rows}
        if cfg["rollout"]:
            from .rollout_record import totals

            report["rollout"] = totals(rows)
        (self.output / "level-bench.json").write_text(json.dumps(report, indent=1))
        self.log(json.dumps({"event": "level_bench_result", **{k: summary[k] for k in (
            "attempts_completed", "attempts_censored", "attempts_failed", "clears", "clear_rate",
            "clears_per_10k_tokens", "median_tokens_to_clear", "mean_level_score",
            "tokens_per_second")}, "preemptions": engine.get("preemptions")}), flush=True)
        return report

    def make_attempt(self, index, pass_index, point, arm, *, gate, arc, card, shared, agent_cfg, window):
        label = f"a{index:03d}-{point['id']}-p{pass_index}" + (f"-{arm['name']}" if self.cfg["arms"] else "")
        clock = GlobalClock(self.cfg["attempt_wall_seconds"], now=phase.monotonic)
        game = GameRecord(label, index, ActiveClock(clock, phase.monotonic,
                                                    budget_clock=self.study.get("budget_clock", "inference")))
        game.ready = asyncio.Event()
        slot = GateSlot(gate, game, cap=self.cap, window=window, reserve=self.cfg["gate_completion_reserve"])
        entry = {"game_id": point["game"], "replica": 0, "instance_id": label}
        model = (arm.get("study") or {}).get("served_model")
        if model and model != agent_cfg.model:  # arm A/B over served models (base vs an adapter) in one bench
            agent_cfg = copy.copy(agent_cfg)
            agent_cfg.model = model
        client = SessionClient(agent_cfg, shared, slot, entry, self.requests)
        if self.study.get("transport_backoff"):  # RL benches: more replays of a generation request (19 Oct)
            client.client.backoff = tuple(float(x) for x in self.study["transport_backoff"])
        prefix_arc = PrefixArcade(arc, point)
        agent = LevelBenchAgent(point["game"], prefix_arc, card, client, agent_cfg,
                                self.output / "attempts" / label, self.analysis_python,
                                phase.monotonic() + self.cfg["attempt_wall_seconds"] + 3600)
        study = {**self.study, **(arm.get("study") or {})}
        agent.attach(game, window=window, cap=self.cap, harness_fixes=study.get("harness_fixes") or None,
                     context_drain=study.get("drain_half") or None, dfr_harness=study.get("dfr_harness") or None,
                     dfr_loop=study.get("dfr_loop") or None, hx_harness=study.get("hx_harness") or None,
                     loop_rule=study.get("loop_rule") or None)

        if self.cfg["rollout"]:
            from .rollout_record import RolloutHTTP

            client.client = agent.rollout = RolloutHTTP(client.client, agent, self.cfg["rollout"])
        if point.get("restore"):
            from .restore import load as load_restore

            if getattr(agent, "rollout", None) is None:
                raise ValueError("level bench: a restored attempt needs the rollout record")
            spec = point["restore"]
            key = (str(spec["attempt"]), int(spec["request"]))
            if key not in RESTORE_CACHE:  # one read per start (the round's tries of a start share it; read-only rows)
                if len(RESTORE_CACHE) > 256:
                    RESTORE_CACHE.clear()
                RESTORE_CACHE[key] = load_restore(*key)
            agent.bench_restore = {**RESTORE_CACHE[key],
                                   "check_only": bool(spec.get("check_only"))}
            agent.rollout.restore = agent.bench_restore
        # mg.kv_project (SharedSlots.project_tokens): the restored prefix the first live request carries, and the
        # try's budget progress
        rows = (getattr(agent, "bench_restore", None) or {}).get("rows") or []
        slot.floor_tokens = int((rows[-1].get("usage") or {}).get("prompt_tokens") or 0) if rows else 0
        slot.prefix_key = repr(key) if rows else None  # kv_project group_prefix: tries of one start share it
        budget_now = token_budget(self.cfg, point)
        slot.progress = lambda: (game.clock.completion_tokens() - agent.bench_base_tokens(),
                                 getattr(agent, "bench_budget", None) or budget_now)
        agent.bench_harvest = bool(point.get("harvest")) and getattr(agent, "rollout", None) is not None
        agent.sampling_override = dict(self.cfg["sampling"] or {})
        if self.cfg["seed_base"] is not None:
            agent.sampling_override["seed"] = int(self.cfg["seed_base"]) + int(pass_index)
        agent.bench_setup(start_level=point["level_index"],
                          budget=int((point.get("restore") or {}).get("budget") or point.get("budget")
                                     or token_budget(self.cfg, point)),
                          through=bool(point.get("through")), prefix=len(point["prefix"]))
        if getattr(self, "read_opening", None) is not None and point.get("through"):
            agent.bench_on_level = lambda level, _label=label: self.read_opening.reached(_label, level)
        return {"index": index, "pass": pass_index, "point": point, "arm": arm["name"], "label": label, "game": game,
                "clock": clock, "agent": agent, "arcade": prefix_arc, "censored": False, "slot": slot}

    async def play(self, attempt):
        agent, game, clock = attempt["agent"], attempt["game"], attempt["clock"]
        clock.release()
        game.clock.release()
        self.attempts.append(attempt)
        started = time.monotonic()
        result, error = None, None
        try:
            result = await agent.run()
        except asyncio.CancelledError:
            error = None
        except Exception as exc:  # noqa: BLE001 - every attempt outcome is recorded
            error = f"{type(exc).__name__}: {exc}"
        finally:
            attempt["row"] = self.row(attempt, result, error, time.monotonic() - started)
            if self.policy_version is not None:
                attempt["row"]["policy_version"] = self.policy_version
            if not attempt["censored"]:  # live play counters (the RL loop's play_progress rows)
                PROGRESS["done"] += 1
                PROGRESS["clears"] += bool(attempt["row"].get("cleared"))
                if PROGRESS_HOOK is not None:
                    try:
                        PROGRESS_HOOK(self.output.name, attempt["row"])
                    except Exception as exc:  # noqa: BLE001 - live stats never fail a try
                        self.log(json.dumps({"event": "progress_hook_failed", "error": str(exc)[:200]}))
            if not attempt["censored"]:
                self.log(json.dumps({"event": "level_bench_attempt", **{k: attempt["row"][k] for k in (
                    "label", "outcome", "completion_tokens", "tokens_to_clear", "actions", "level_score",
                    "gameplay_requests", "wall_seconds")}}), flush=True)
                try:  # per finished try, so a stop or crash keeps what was played (level-bench.json comes at the end)
                    with open(self.output / "level-bench.partial.jsonl", "a") as file:
                        file.write(json.dumps(attempt["row"], default=str) + "\n")
                except Exception as exc:  # noqa: BLE001
                    self.log(json.dumps({"event": "level_bench_partial_failed", "error": str(exc)[:200]}), flush=True)

    def row(self, attempt, result, error, seconds):
        agent, point = attempt["agent"], attempt["point"]
        result = result or {}
        error = error or result.get("error")
        replay = attempt["arcade"].replay
        if attempt["censored"]:
            outcome = "censored"
        elif agent.bench_outcome in ("cleared", "budget", "won", "recorded", "replay_mismatch"):
            outcome = agent.bench_outcome
        elif replay is not None and not replay["ok"]:
            outcome = "replay_mismatch"
        elif error:
            outcome = "error"
        elif attempt["clock"].remaining() <= 0:
            outcome = "wall"
        else:
            outcome = result.get("status") or "stopped"
        cleared = outcome in ("cleared", "won")
        clear = agent.bench_clear or {}
        enter = {"completion_tokens": 0, "steps": 0}
        restore = agent.bench_restore
        if restore is not None:  # tokens from the first live request, actions from the level's first board
            enter = {"completion_tokens": restore["tokens"], "steps": restore["level_start_step"]}
        # an error before the restored prefix is replayed leaves the clock short of the enter mark (v17 29 Oct: -16681)
        actions = clear.get("actions", max(0, agent.steps - enter["steps"]))
        requests, noaction = request_actions(agent.workspace.log)
        harvested = agent.bench_harvest_row
        if harvested is not None:  # the harvest's extra request is not part of the try
            requests, noaction = requests - 1, max(0, noaction - 1)
        through = None
        if restore is not None or (point.get("segment") and not point.get("through")):
            # One level played (a restored segment, or a fresh level-1 segment, RL multi-game 16 Oct): credited as a
            # play-through level, nothing before it
            level = restore["level"] if restore is not None else int(point.get("level") or 1)
            base = point.get("baselines") or []
            rows_ = []
            if cleared:
                credit = level_score(base[level - 1] if level <= len(base) else None, clear["actions"], True)
                rows_ = [{"level": level, "tokens": clear["completion_tokens"], "at_tokens": clear["completion_tokens"],
                          "actions": clear["actions"], "score": round(credit, 3)}]
            total = int(point.get("total_levels") or len(base) or 1)
            through = {"levels": len(rows_), "total_levels": total, "level_rows": rows_,
                       "game_score": round(sum(min(100.0, r["score"]) for r in rows_) / total, 3), "enter_applied": []}
        elif point.get("through"):
            # Play-through: every cleared level with its own tokens, actions and credit; the game score is the mean
            # credit over all the game's levels (uncleared levels count 0).
            base, levels, before = point.get("baselines") or [], [], {"completion_tokens": 0, "steps": 0}
            for item in agent.bench_levels or []:
                used = item["steps"] - before["steps"]
                credit = level_score(base[item["level"] - 1] if item["level"] <= len(base) else None, used, True)
                levels.append({"level": item["level"], "tokens": item["completion_tokens"] - before["completion_tokens"],
                               "at_tokens": item["completion_tokens"], "actions": used, "score": round(credit, 3)})
                before = item
            total = int(point.get("total_levels") or len(base) or 1)
            through = {"levels": len(levels), "total_levels": total, "level_rows": levels,
                       "game_score": round(sum(min(100.0, r["score"]) for r in levels) / total, 3),
                       "enter_applied": []}
        row = self.row_fields(attempt, result, error, seconds, outcome=outcome, cleared=cleared, clear=clear,
                              enter=enter, actions=actions, requests=requests, noaction=noaction, replay=replay)
        if through is not None:
            row.update(through, level_score=through["game_score"])
        if getattr(agent, "rollout", None) is not None:
            from .rollout_record import clear_replay

            row["rollout"] = agent.rollout.stats
            # the attempt's own actions on a fresh game: exact for a start with no replayed prefix
            row["clear_replay"] = (clear_replay(*self.checks, agent, point) if cleared and not point["prefix"]
                                   else None)
        if restore is not None:
            result = dict(agent.rollout.restore_result or {})
            state = agent.bench_restore_state or {"ok": False, "detail": "first live request not reached"}
            row["restore"] = {"source": restore["source"], "request": restore["request"], "level": restore["level"],
                              "scripted": result.get("requests", 0), "diffs": result.get("diffs", [])[:20],
                              "diff_count": len(result.get("diffs", [])), "prompt_digest": result.get("prompt_digest"),
                              "state": state, "ok": bool(state.get("ok")) and not result.get("diffs"),
                              **{k: result[k] for k in ("seeded", "seed_wait_s", "seed_wait_timeout") if k in result}}
        if harvested is not None:
            row["harvest"] = dict(harvested)
            row["completion_tokens"] = max(0, row["completion_tokens"] - int(harvested.get("completion_tokens") or 0))
        return row

    def row_fields(self, attempt, result, error, seconds, *, outcome, cleared, clear, enter, actions, requests,
                   noaction, replay):
        agent, point, game = attempt["agent"], attempt["point"], attempt["game"]
        return {
            "label": attempt["label"], "index": attempt["index"], "pass": attempt["pass"], "point": point["id"],
            "arm": attempt["arm"], "server_index": attempt.get("server_index", 1),
            "game": point["game"], "split": point["split"], "level": point["level"],
            "level_index": point["level_index"], "outcome": outcome, "cleared": cleared,
            "completion_tokens": max(0, game.clock.completion_tokens() - enter["completion_tokens"]),
            "tokens_to_clear": clear.get("completion_tokens") if cleared else None,
            "actions": actions, "total_actions": agent.steps,
            "baseline_actions": point.get("baseline_actions"),
            "token_budget": agent.bench_budget,
            "level_score": round(level_score(point.get("baseline_actions"), actions, cleared), 3),
            "gameplay_requests": requests, "noaction_requests": noaction, "admitted_requests": len(game.requests),
            "prompt_tokens": agent.prompt_tokens, "compactions": game.compactions,
            "context_drain": dict(agent.drain_stats) if getattr(agent, "drain_stats", None) else None,
            "wall_seconds": round(seconds, 1), "clear_wall_seconds": clear.get("wall_seconds"),
            "inference_seconds": round(game.clock.inference_seconds(), 1),
            "tool_seconds": round(game.clock.tool_seconds(), 1),
            "gate_wait_seconds": round(sum(r["gate_wait_seconds"] for r in game.requests), 1),
            "max_prompt_tokens": game.max_prompt_tokens, "replay": replay,
            "agent_status": result.get("status"), "error": error if outcome in ("error", "replay_mismatch") else None,
        }

    def progress(self, wall, running, gate, planned, launched, *, final=False):
        rows = [a["row"] for a in self.attempts if "row" in a and not a["censored"]]
        live = [a for a in self.attempts if "row" not in a]
        tokens = sum(a["game"].clock.completion_tokens() for a in self.attempts)
        elapsed = max(1e-6, wall.elapsed())
        clears = sum(r["cleared"] for r in rows)
        done_tokens = sum(r["completion_tokens"] for r in rows)
        line = {"event": "level_bench_final" if final else "level_bench_progress",
                "elapsed_seconds": round(wall.elapsed(), 1), "remaining_seconds": round(wall.remaining(), 1),
                "launched": launched, "planned": planned, "running": len(live), "done": len(rows),
                "clears": clears, "budget_outs": sum(r["outcome"] == "budget" for r in rows),
                "failed": sum(r["outcome"] in ("error", "replay_mismatch") for r in rows),
                "completion_tokens": tokens, "tokens_per_second": round(tokens / elapsed, 1),
                "clears_per_10k_tokens": round(clears * 1e4 / done_tokens, 3) if done_tokens else None,
                "gate": {k: gate.snapshot()[k] for k in ("reserved", "inflight", "waiting", "capacity")}}
        with (self.output / "progress.jsonl").open("a") as file:
            file.write(json.dumps(line) + "\n")
        self.log(json.dumps(line), flush=True)
        if getattr(self, "read_metrics", None) is not None:  # held-out read (study.hread): the dashboard row; never a reason to stop
            try:
                samples = getattr(self.monitor, "samples", None) or []
                self.read_metrics.emit(line, rows, samples[-1]["metrics"] if samples else None)
            except Exception as exc:
                self.log(json.dumps({"event": "read_metrics_error", "error": repr(exc)[:300]}), flush=True)


async def run_level_bench(study, trial, **kwargs):
    return await LevelBench(study, trial, **kwargs).run()
