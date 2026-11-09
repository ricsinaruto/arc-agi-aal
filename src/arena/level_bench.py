"""Level bench (27 September): many short, independent level attempts instead of one 23-game hour.

A one-hour arena run is one sample (sd 1.4-2.4 points) for ~1.3 GPU-hours. The bench measures the
agent where the score comes from, one level at a time, and runs many attempts at once:

- Start points (``arena.level_points``, ``data/level_bench/start_points.json``): a recorded action
  prefix per (game, level) that reaches the level's first board. The attempt replays it on a fresh
  environment (the model never sees it) and checks the board; the agent then starts with a fresh
  context at that board, as at a new game.
- Each attempt is one Duck-transfer agent (the fixed-2x base's agent and config: study effort, cap,
  temperature; context ``window`` with the arena's append-only compaction) that plays until the level
  clears, its completion tokens (every purpose) reach ``token_budget`` (checked before each request:
  the reply that crosses the budget still executes its actions), or ``attempt_wall_seconds`` pass.
- ``concurrency`` attempts run at once through the arena's KV gate. The plan is ``passes`` seeded
  permutations of the selected start points (``split``: dev by default; holdout is a separate
  confirmation set). Pairing across runs uses ``(start point, pass)``, so two runs with the same
  settings attempt the same list in the same order.
- Attempts still running at the bench wall are cancelled and recorded as censored: they are excluded
  from every headline metric (their tokens are reported separately).

Stuck-level bench (hx4, 7 October): ``arms`` runs every start point once per arm and pass, interleaved, so arms are
paired inside one run on one server. An arm is ``{"name", "warm", "history", "reset", "study"}``: ``warm`` gives the attempt
what the recorded source agent carried into the level (the point's ``warm``: its cross-level notes and ``note()``
entries); ``history`` puts the replayed prefix into the sandbox's ``history`` / ``transitions`` (the earlier levels'
frames, as the source agent had them); ``reset`` starts a point recorded inside a level from that level's first
board; ``study`` overrides study keys for that arm's agents (harness blocks such as ``hx_harness``). Without ``arms`` there is one cold arm. Attempts run the study's full harness (context drain, dfr,
hx, loop rule), as the arena trial does.

Model-agnostic: the gate's capacity comes from the served layout (hybrid blocks on the Flash vLLM
profile, the SGLang pool, the vLLM ``GPU KV cache size`` line) or from ``capacity_tokens`` for an
external endpoint; the serving profile (MTP off or on) is a study/server setting, recorded here.
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
from arc_infra.fleet import SessionClient
from duck_transfer.agent import GameplayRequestExpired

from .clock import ActiveClock, GlobalClock
from .compaction import DEFAULT_COMPACTION_THINKING
from .duck import ArenaTransferDuckAgent
from .gate import GateSlot, KVGate
from .level_points import check_replay, level_score
from .scheduler import GameRecord
from .trim_bed import ScriptedResponse, record_settings, reply_record
from .trim_bed import settings as trim_settings

DEFAULTS = {
    "split": "dev",                 # dev | holdout | all
    "games": None,                  # optional list of 4-character game prefixes
    "concurrency": 16,              # attempts running at once (and gate slots)
    "window": 49152,                # agent context per attempt (compaction at window - cap)
    "token_budget": 40000,          # completion tokens per attempt (every purpose)
    "level_token_cap": None,        # play-through: an attempt ends after this many tokens on one level without a clear
    "token_budget_by_level": None,  # {"<level number>": tokens}: the budget of starts on that level (else token_budget)
    "token_budget_by_point": None,  # {"<point id>": tokens}: the budget of that start (before the level's)
    # {"top_logprobs", "token_ids"}: every live reply is stored in the attempt's rollout.jsonl with the request it
    # answered, its sampled token ids and the engine's log-probabilities (arena.rollout_record); a clear is replayed
    # from a fresh game
    "rollout": None,
    # {temperature, top_p, top_k, presence_penalty, ...}: sent on every request of every attempt in place of the
    # harness's own sampler (RL step 0 on Qwen3.6, the model card's recommended settings)
    "sampling": None,
    # RL step 1 evals (13 Oct): every request of the attempt of pass k carries vLLM seed seed_base + k, so eval
    # attempts are paired across rounds (common random numbers); None: unseeded
    "seed_base": None,
    "attempt_wall_seconds": 1800,   # safety cap per attempt
    "passes": 3,                    # seeded permutations of the start points, in order
    "order_seed": 0,
    "progress_seconds": 30,         # a live progress line at least this often (at most 60)
    "launch_cutoff_seconds": 120,   # no new attempt when less bench wall remains
    "min_wall_seconds": 600,        # the bench refuses to start with less
    "start_points": "configs/level_bench_start_points.json",
    "capacity_tokens": None,        # external endpoint: gate capacity in tokens (no server log)
    "gate_completion_reserve": None,  # completion tokens the gate reserves per request (None: the completion cap)
    "serving": None,                # recorded label of the serving profile (e.g. "mtp0")
    "arms": None,                   # [{"name", "warm": bool, "history": bool, "study": {...}}]; None: one cold arm
    "first_pass": 0,                # index of the first pass (a later run extends an earlier one's passes)
    "lead_token_budget": 80000,     # lead points: tokens to get through the lead-in level(s), else "lead_out"
    # {"multiple": m, "floor": f}: an attempt also ends ("action_cap") once it has executed max(f, ceil(m x the
    # level's human baseline)) actions of its own (checked before each request, as the token budget is); None: off
    "action_cap": None,
}
ARM_KEYS = ("name", "warm", "history", "reset", "study", "points", "games", "enter", "thinking", "levels",
            "passes", "seed_base")  # points: start | deep | lead |
# None; games: None or a list of 4-character game prefixes the arm is limited to; enter (lead points): what the harness
# takes away at the target level's first board, before the agent's first request on it (``ENTER``)
ENTER = (None, "clear", "restart", "both")  # clear: the conversation and the carried world model (python session
# kept); restart: the python session (interpreter restarted: the agent's variables, functions and registered tools are
# gone; the conversation stays); both: both. The agent's note() entries (hx memo) are a third store: every arm keeps
# them (a restarted interpreter starts with them). A play-through point applies the removal at every level entry.
ENTER_CLEAR_NOTE = (
    "[Context cleared by the harness] You have just completed level {done} and are at the first board of level "
    "{level}. The conversation before this point was removed by the harness (routine, not a failure). Your python "
    "session is unchanged: `history`, `transitions`, your variables, functions and registered tools are all still "
    "there."
)
ENTER_RESTART_NOTE = (
    "[Python session restarted by the harness] Your python interpreter was restarted (routine, not a failure): the "
    "variables, functions and registered tools you defined are gone. Your note() entries, `history`, `transitions` "
    "and `current_frame` are provided as before."
)
ENTER_BOTH_NOTE = (
    "[Context cleared and python session restarted by the harness] You have just completed level {done} and are at "
    "the first board of level {level}. The conversation before this point was removed and your python interpreter was "
    "restarted (routine, not a failure): the variables, functions and registered tools you defined are gone. Your "
    "note() entries, `history`, `transitions` and `current_frame` are provided as before."
)
HX_NOTES_EMPTY = '"sim_seen": 0, "notes": {}}'  # the pinned empty note store of the hx child block
SPLITS = ("dev", "holdout", "all")
UNSUPPORTED = ("cross_game_notes", "explorer", "background_lane", "commit", "llm_backlog", "audition_pool")
HEADLINE_EXCLUDED = ("censored", "error", "replay_mismatch", "lead_out", "recorded")


SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"}


def settings(raw):
    """Validated ``study["level_bench"]`` merged over the defaults."""
    raw = dict(raw or {})
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown level-bench settings: {sorted(unknown)}")
    merged = {**DEFAULTS, **raw}
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
                or len({a["name"] for a in arms}) != len(arms)
                or any(a.get("points") not in (None, "start", "deep", "lead", "game") for a in arms)
                or any(a.get("enter") not in ENTER or (a.get("enter") and a.get("points") not in ("lead", "game"))
                       for a in arms)):
            raise ValueError("level_bench.arms is a list of {name, warm, history, study} with distinct names "
                             "(enter only on lead-point or play-through arms)")
        if any(type(a.get("thinking", True)) is not bool
               or (a.get("levels") is not None and (not isinstance(a["levels"], list) or not a["levels"]))
               or (a.get("passes") is not None and (type(a["passes"]) is not int or a["passes"] < 1))
               or (a.get("seed_base") is not None and (type(a["seed_base"]) is not int or a["seed_base"] < 0))
               for a in arms):
            raise ValueError("level_bench arm: thinking is true/false, levels a list of level numbers, passes >= 1")
    sampling = merged["sampling"]
    if sampling is not None and (not isinstance(sampling, dict) or not sampling or set(sampling) - SAMPLING_KEYS
                                 or any(type(v) not in (int, float) or isinstance(v, bool) for v in sampling.values())):
        raise ValueError(f"level_bench.sampling is {{key: number}} over {sorted(SAMPLING_KEYS)}")
    cap = merged["level_token_cap"]
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("level_bench.level_token_cap is a positive integer or null")
    by_level = merged["token_budget_by_level"]
    if by_level is not None and (not isinstance(by_level, dict) or not by_level or any(
            not str(k).isdigit() or type(v) is not int or v < 1 for k, v in by_level.items())):
        raise ValueError("level_bench.token_budget_by_level is {level number: positive integer} or null")
    if merged["rollout"] is not None:
        from .rollout_record import settings as rollout_settings

        merged["rollout"] = rollout_settings(merged["rollout"])
    by_point = merged["token_budget_by_point"]
    if by_point is not None and (not isinstance(by_point, dict) or not by_point or any(
            type(v) is not int or v < 1 for v in by_point.values())):
        raise ValueError("level_bench.token_budget_by_point is {point id: positive integer} or null")
    cap = merged["action_cap"]
    if cap is not None and (not isinstance(cap, dict) or set(cap) != {"multiple", "floor"}
                            or type(cap["floor"]) is not int or cap["floor"] < 1
                            or type(cap["multiple"]) not in (int, float) or cap["multiple"] <= 0):
        raise ValueError("level_bench.action_cap is {multiple: > 0, floor: integer >= 1} or null")
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


# thinking False: the arm's gameplay requests go out with enable_thinking off (the template's empty think block);
# levels: None or the level numbers the arm plays; passes: None or the number of passes (from pass 0) the arm plays
COLD = {"name": "base", "warm": False, "history": False, "reset": False, "study": {}, "points": None, "games": None,
        "enter": None, "thinking": True, "levels": None, "passes": None, "seed_base": None}


def token_budget(cfg, point):
    by_point, by_level = cfg.get("token_budget_by_point") or {}, cfg.get("token_budget_by_level") or {}
    return int(by_point.get(point.get("id"), by_level.get(str(point.get("level")), cfg["token_budget"])))


def action_cap(cfg, point):
    """The attempt's own-action limit on this start point, or None."""
    cap = cfg.get("action_cap")
    if cap is None:
        return None
    return max(cap["floor"], math.ceil(cap["multiple"] * float(point.get("baseline_actions") or 0)))


def point_kind(point):
    """``lead``: the agent plays ``lead_levels`` levels itself before the target level (it arrives with its own tools
    and conversation); ``deep``: a board inside the level; ``start``: the level's first board; ``game``: a
    play-through point (``through``: a fresh game played for the whole token budget, every level counted)."""
    return ("game" if point.get("through") else "lead" if point.get("lead_levels") else
            "deep" if point.get("deep_tokens") else "start")


def plan(points, passes, seed, arms=None, first_pass=0):
    """``[(pass, point, arm)]``: each pass a seeded permutation of the points (sorted by id first); within a pass the
    arms rotate from point to point, so a run cut short stays balanced. An arm with ``points`` takes only level-start
    or only in-level (``deep_tokens``) points."""
    arms = [{**COLD, **a} for a in arms] if arms else [COLD]
    out = []
    for index in range(first_pass, first_pass + passes):
        order = list(points)
        random.Random(f"{seed}:{index}").shuffle(order)
        for turn in range(len(arms)):
            for place, point in enumerate(order):
                arm = arms[(place + turn + index) % len(arms)]
                if (arm["points"] in (None, point_kind(point))
                        and (not arm["games"] or str(point.get("game", ""))[:4] in arm["games"])
                        and (not arm["levels"] or point.get("level") in arm["levels"])
                        and (arm["passes"] is None or index < arm["passes"])):
                    out.append((index, point, arm))
    if any(point.get("priority") for _, point, _ in out):  # e.g. per-start budgets: the longest attempts start first
        out.sort(key=lambda work: -float(work[1].get("priority") or 0))
    return out


def bench_priority(cfg, study):
    """Slot priority of a bench in a shared pool (higher takes first): its largest token cap, or, for an RL v7
    training group (study ``admit_order``, v7.oldest_first), minus its group number: older groups first, every group
    under the held-out reads (cap >= 0) and inside the tail_fill/explore bands (+-10**9)."""
    if study.get("admit_order") is not None:
        return -1 - int(study["admit_order"])
    return max([int(v) for v in (cfg.get("token_budget_by_point") or {}).values()] or [0])


def tail_order(work, expected):
    """tail_fill launch order (indices into ``work``): the longest expected tries first. ``expected``: {point id:
    tokens}; a point without history counts as the median of the known ones; ties keep the plan order. No history:
    the plan order."""
    known = [expected[w[1]["id"]] for w in work if w[1]["id"] in (expected or {})]
    if not known:
        return list(range(len(work)))
    mid = statistics.median(known)
    return sorted(range(len(work)), key=lambda i: -float(expected.get(work[i][1]["id"], mid)))


class ReplayMismatch(RuntimeError):
    """The replayed prefix did not reproduce the recorded level count and board."""


class PrefixArcade:
    """The arcade the attempt's agent sees: ``make`` replays the start point's prefix first."""

    def __init__(self, arc, point, keep=False):
        self.arc, self.point, self.replay = arc, point, None
        self.trace = [] if keep else None  # (action text, raw) per replayed step, for the history arm

    def make(self, game_id, **kwargs):
        env = self.arc.make(game_id, **kwargs)
        if env is None:
            return None
        started = time.monotonic()
        ok, detail, _ = check_replay(env, self.point, self.trace)
        self.replay = {"ok": ok, "detail": detail, "actions": len(self.point["prefix"]),
                       "seconds": round(time.monotonic() - started, 3)}
        if not ok:
            raise ReplayMismatch(f"{self.point['id']}: {detail}")
        return env


class LevelBenchAgent(ArenaTransferDuckAgent):
    """The arena's Duck-transfer agent, stopped at the first level clear or at the token budget."""

    bench_start = 0
    bench_budget = None
    bench_outcome = None  # cleared | budget | action_cap | censored
    bench_action_cap = None  # own actions that end the attempt (level_bench.action_cap), or None
    bench_clear = None

    bench_warm = None     # {"cross_level_notes", "notes"}: what the source agent carried into the level
    bench_trace = None    # the PrefixArcade whose replayed steps become the sandbox history
    bench_offset = 0      # prefix actions counted in ``steps`` (history arm)
    bench_reset = False   # reset arm: a start point inside a level begins from the level's first board
    bench_lead = 0        # lead point: levels the agent plays before the target level
    bench_lead_budget = None
    bench_enter = None    # {"completion_tokens", "steps"} at the target level's first board (lead points)
    bench_enter_mode = None     # ``ENTER``: what is taken away at that board
    bench_enter_applied = None  # the workspace records of that (one per level entry)
    bench_enter_due = False     # a level entry whose removal is not applied yet
    bench_enter_note = None     # restart: the one-off line of the next user turn
    bench_through = False       # play-through point: no stop at a level clear
    bench_level_cap = None      # play-through: tokens on one level before the attempt ends (None: no cap)
    bench_levels = None         # play-through: [{"level", "completion_tokens", "steps"}] per cleared level
    # hx7 trim bed (arena.trim_bed)
    bench_record = None         # collector: {"stop_tokens"}; the gameplay replies go to script.jsonl
    bench_script = None         # the point's recorded replies, answered before any live request
    bench_script_at = 0
    bench_script_end = None     # {"completion_tokens", "steps", "levels", "ok", "detail"} at the first live request
    bench_actions = None        # every action the attempt executed, in order (record and replay)
    bench_restore = None        # arena.restore: a recorded attempt replayed to a request, then live (F-small)
    bench_restore_state = None  # the state check at the first live request
    # in-run harvest (envs): a cleared segment sends one recorded 1-token request at the next board, the
    # restore point of a new start one level deeper; never trained (rollout row ``harvest``)
    bench_harvest = False
    bench_harvest_state = None  # None | "due" (cleared, request owed) | "sending" | "sent"
    bench_harvest_row = None    # {"request", "prompt_tokens", "step", "levels"} of the recorded request

    def bench_setup(self, *, start_level, budget, warm=None, trace=None, reset=False, level_start=None, prefix=0,
                    lead=0, lead_budget=None, enter=None, through=False, script=None, record=None, action_cap=None,
                    level_cap=None):
        self.bench_start, self.bench_budget = int(start_level), int(budget)
        self.bench_level_cap = level_cap
        self.bench_action_cap = action_cap
        self.bench_lead, self.bench_lead_budget, self.bench_enter = int(lead or 0), lead_budget, None
        self.bench_through, self.bench_levels = bool(through), []
        if through and (self.bench_lead or prefix or start_level):
            raise ValueError("level bench: a play-through point is a fresh game")
        if enter and (not (self.bench_lead or through) or self.drain is None):
            raise ValueError("level bench: an enter arm needs a lead or play-through point and drain_half")
        self.bench_enter_mode, self.bench_enter_applied, self.bench_enter_note = enter, [], None
        self.bench_enter_due, self.bench_bootstrap = False, None
        self.bench_outcome = self.bench_clear = None
        self.bench_record, self.bench_script = record_settings(record), script or None
        self.bench_script_at, self.bench_script_end, self.bench_actions = 0, None, []
        # A trim before the first live request (its summary) counts against the budget: the base is the script's own.
        self.bench_script_tokens = sum(int(r["usage"].get("completion_tokens") or 0)
                                       for r in (script or {}).get("replies", []))
        if (self.bench_record or self.bench_script) and not through:
            raise ValueError("level bench: a recorded or scripted attempt is a play-through point")
        if self.bench_record and self.bench_script:
            raise ValueError("level bench: an attempt records or replays, not both")
        self.bench_warm, self.bench_trace, self.bench_reset = warm or None, trace, bool(reset)
        self.bench_level_start = level_start  # prefix actions before the level's first board (None: all of them)
        self.bench_prefix = int(prefix)
        notes = (warm or {}).get("notes")
        if notes and getattr(self, "hx", None) and self.hx["memo"]:
            empty = HX_NOTES_EMPTY
            if self.sandbox.bootstrap.count(empty) != 1:
                raise ValueError("level bench: pinned hx notes store changed")
            self.sandbox.bootstrap = self.sandbox.bootstrap.replace(
                empty, '"sim_seen": 0, "notes": %r}' % {str(k)[:40]: str(v)[:400] for k, v in list(notes.items())[:12]})
        return self

    def warm_text(self):
        warm = self.bench_warm or {}
        parts = [warm.get("cross_level_notes") or (warm.get("knowledge") or {}).get("cross_level_notes") or ""]
        if warm.get("notes"):
            parts.append("Your notes from the earlier levels: " + " | ".join(f"{k}: {v}" for k, v in warm["notes"].items()))
        return "\n".join(p for p in parts if p)

    def fresh_beliefs(self):
        """Before the first reply the abandoned attempt is the recorded source agent's: its carried beliefs."""
        if self.bench_warm and not self.gameplay_replies:
            warm = self.bench_warm
            knowledge = dict(warm.get("knowledge") or {})
            if warm.get("cross_level_notes") and not knowledge.get("cross_level_notes"):
                knowledge["cross_level_notes"] = warm["cross_level_notes"]
            return knowledge, dict(warm.get("notes") or {})
        return super().fresh_beliefs()

    async def before_first_turn(self):
        await super().before_first_turn()
        inside = self.bench_level_start is not None and self.bench_level_start != self.bench_prefix
        if inside and self.steps == self.level_start_step:
            self.level_start_step = -1  # no replayed history: the level did not start at this board
        if self.bench_reset and inside and self.obs.state == "NOT_FINISHED":
            # A start point inside a level: this arm begins from the level's first board.
            from arc_infra.plans import Action

            await self.step(Action("RESET"))
            self.write_duck_state("RESET")
            self.level_start_step = self.steps  # the reset counts as one of the attempt's actions
            self.bench_mark_level_start()

    async def before_turn(self):
        await super().before_turn()
        if (self.bench_enter_mode and self.bench_enter_due and self.bench_outcome is None
                and self.obs.state == "NOT_FINISHED"):
            self.bench_enter_due = False
            await self.bench_enter_apply()

    async def bench_enter_apply(self):
        """The enter arm's removal, between turns (event loop, worker idle), before the first user turn of the entered
        level is built: ``clear`` blanks the carried world model and leaves ``[system, note + python inventory,
        that turn's observation]`` at the next request (the fresh-context path of ``drain_context``); ``restart``
        discards the interpreter (the next call starts a new one, which receives the whole ``history`` again and
        starts with the note() entries of the old one)."""
        mode = self.bench_enter_mode
        level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
        notes = dict(((self.sandbox.hx_state or {}).get("notes")) or {})
        record = {"mode": mode, "level": level, "session_calls": self.sandbox.calls, "notes": len(notes)}
        if mode in ("restart", "both"):
            inventory = await asyncio.to_thread(self.sandbox.inventory)
            record.update(functions=len((inventory or {}).get("functions") or []),
                          values=len((inventory or {}).get("values") or []))
            with self.sandbox.lock:
                self.sandbox._discard("bench_enter")
                if self.bench_bootstrap is None:
                    self.bench_bootstrap = self.sandbox.bootstrap
                if self.bench_bootstrap.count(HX_NOTES_EMPTY) == 1:
                    self.sandbox.bootstrap = self.bench_bootstrap.replace(
                        HX_NOTES_EMPTY, '"sim_seen": 0, "notes": %r}' % notes)
                elif notes:
                    raise ValueError("level bench: pinned hx notes store changed")
            self.sandbox.hx_state = {"notes": notes} if notes else None
            self.inventory_restarts_seen = self.sandbox.restarts  # the note below says it; no second remark
        if mode in ("clear", "both"):
            analyzer = getattr(self, "_analyzer", None)
            knowledge = getattr(analyzer, "_summarized_knowledge", None) or {}
            record["world_model_chars"] = sum(len(str(v)) for v in knowledge.values())
            for key in list(knowledge):
                knowledge[key] = ""
            if self.fresh_stats is None:
                self.fresh_stats = {"attempts": 0, "resets": 0, "tokens_freed": 0}
            note = ENTER_BOTH_NOTE if mode == "both" else ENTER_CLEAR_NOTE
            self.fresh_pending = note.format(done=level - 1, level=level)
        else:
            self.bench_enter_note = ENTER_RESTART_NOTE
        self.bench_enter_applied.append(record)
        self.workspace.append("level_bench_enter_apply", step=self.steps, **record)

    def fresh_context(self, messages, tools, measure):
        """After a ``both`` removal the note must not say the interpreter was untouched."""
        from .context_drain import PYTHON_NOTE

        both = (self.fresh_pending or "").startswith(ENTER_BOTH_NOTE[:40])
        result = super().fresh_context(messages, tools, measure)
        if both and len(result) > 1 and isinstance(result[1].get("content"), str):
            text = result[1]["content"].replace(PYTHON_NOTE.strip(), "", 1)
            result[1] = {**result[1], "content": "\n\n".join(part.strip() for part in text.split("\n\n") if part.strip())}
        return result

    def turn_notice(self):
        notice = super().turn_notice()
        if self.bench_enter_note is not None:
            notice, self.bench_enter_note = "\n".join(filter(None, (self.bench_enter_note, notice))), None
        return notice

    def analyzer(self, loop):
        analyzer = super().analyzer(loop)
        text = self.warm_text()
        knowledge = dict((self.bench_warm or {}).get("knowledge") or {})
        if self.fresh is not None and self.fresh["at_start"]:
            text, knowledge = "", {}  # the fresh attempt's note carries the beliefs as suspects
        if text or knowledge:
            ensure = analyzer._ensure_session

            def ensure_session(state_path):
                fresh = analyzer._session_runtime_dir != state_path.parent
                ensure(state_path)
                if fresh:  # the session start clears the carried world model
                    for key, value in knowledge.items():
                        if key in analyzer._summarized_knowledge and key != "cross_level_notes":
                            analyzer._summarized_knowledge[key] = str(value)
                    if text:
                        analyzer._summarized_knowledge["cross_level_notes"] = text

            analyzer._ensure_session = ensure_session
        return analyzer

    def write_duck_state(self, action="", result=None):
        trace = self.bench_trace.trace if self.bench_trace is not None else None
        if trace and not self.duck_history and not action:
            # History arm: the replayed steps 0..n-1 as history entries, then the level's first board as step n.
            from arc_infra.perception import Observation
            from inference.agent.runtime_state import Frame, HistoryEntry

            from .level_points import decode_action

            labels, previous = [], None
            for index, (text, raw) in enumerate(trace):
                label = ""
                if text is not None:
                    name, x, y = decode_action(text)
                    label = f"MOUSE(row={y}, col={x})" if name == "ACTION6" else self.model_action(name)
                labels.append(label)
                previous = Observation.from_raw(raw, previous)
                if index < len(trace) - 1:
                    self.duck_history.append(HistoryEntry(action=label, frame=Frame(
                        grid=tuple(map(tuple, previous.board.tolist())), step=index,
                        level=max(1, min(previous.total_levels, previous.levels + 1)))))
            self.bench_offset = self.steps = len(trace) - 1
            self.level_start_step = self.steps if self.bench_level_start is None else int(self.bench_level_start)
            self.bench_mark_level_start()
            self.bench_trace.trace = None  # the raw observations are no longer needed
            if labels[-1]:
                return super().write_duck_state(labels[-1])
        return super().write_duck_state(action, result)

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
        if self.bench_outcome is None and self.bench_lead and self.bench_enter is None:
            if tokens >= self.bench_lead_budget:
                self.bench_outcome = "lead_out"
                self.workspace.append("level_bench_lead_out", step=self.steps, completion_tokens=tokens)
        elif self.bench_outcome is None and tokens - self.bench_base_tokens() >= self.bench_budget:
            self.bench_outcome = "budget"
            self.workspace.append("level_bench_budget", step=self.steps, completion_tokens=tokens)
        elif (self.bench_outcome is None and self.bench_through and self.bench_level_cap
              and tokens - (self.bench_levels[-1]["completion_tokens"] if self.bench_levels else 0)
              >= self.level_cap_now()):
            self.bench_outcome = "level_cap"
            self.workspace.append("level_bench_level_cap", step=self.steps, level=len(self.bench_levels) + 1,
                                  completion_tokens=tokens)
        elif (self.bench_outcome is None and self.bench_action_cap is not None
              and self.steps - self.bench_offset >= self.bench_action_cap):
            self.bench_outcome = "action_cap"
            self.workspace.append("level_bench_action_cap", step=self.steps, actions=self.steps - self.bench_offset,
                                  completion_tokens=tokens)
        if self.bench_outcome is not None:
            raise GameplayRequestExpired(f"Level-bench attempt finished: {self.bench_outcome}")

    def bench_base_tokens(self):
        """Completion tokens that do not count against the budget: the lead-in or the replayed script."""
        if self.bench_restore is not None:
            return self.bench_restore["tokens"]
        if self.bench_script is not None:
            return self.bench_script_tokens
        return (self.bench_enter or {}).get("completion_tokens", 0)

    def bench_dir(self):
        return Path(self.workspace.log).parent

    def bench_board(self):
        from arc_infra.perception import board_hash

        return board_hash(self.obs.board) if self.obs is not None and self.obs.board is not None else None

    def bench_script_finish(self):
        """The first live request of a scripted attempt: the replay must have reproduced the recording."""
        stop = self.bench_script["stop"]
        now = {"steps": self.steps, "levels": int(self.obs.levels), "board_hash": self.bench_board()}
        problems = [f"{key} {now[key]} != {stop[key]}" for key in now if now[key] != stop[key]]
        if self.bench_actions != self.bench_script["actions"]:
            problems.append("actions differ")
        self.bench_script_end = {"completion_tokens": self.bench_script_tokens, **now,
                                 "requests": self.bench_script_at, "ok": not problems,
                                 "detail": "; ".join(problems) or "ok", "compactions": self.game.compactions}
        self.workspace.append("level_bench_script_end", step=self.steps, **self.bench_script_end)
        if problems:
            self.bench_outcome = "replay_mismatch"

    def replaying(self):
        """A recorded prefix is being replayed (restore or script): its tool calls must reproduce the recording."""
        # the state check / script finish runs at the first live chat, after the last replayed reply's tools
        return ((self.bench_restore is not None and self.bench_restore_state is None)
                or (self.bench_script is not None and self.bench_script_end is None))

    def level_cap_now(self):
        """The play-through level cap of the level being played: one number, or a per-level list (envs explorer
        stuck stop, 22 Oct; the last entry holds for deeper levels)."""
        cap = self.bench_level_cap
        if isinstance(cap, (list, tuple)):
            return int(cap[min(len(self.bench_levels), len(cap) - 1)])
        return int(cap)

    def bench_mark_level_start(self):
        """The current level's first step, for restore.load of a cut from this attempt (a start past level 1 has no
        ``level_bench_level`` mark for the levels before it)."""
        if self.level_start_step >= 0:
            self.workspace.append("level_bench_level_start", step=self.steps, levels=int(self.obs.levels),
                                  level_start_step=int(self.level_start_step))

    def bench_restore_live(self):
        return self.bench_restore is not None and self.rollout.restore_at > self.bench_restore["request"]

    def bench_restore_check(self):
        """At the restored attempt's first live request: the game and the action list as the recording had them."""
        stop = self.bench_restore["stop"]
        now = {"steps": self.steps, "levels": int(self.obs.levels), "board_hash": self.bench_board()}
        problems = [f"{key} {now[key]} != {stop[key]}" for key in now if now[key] != stop[key]]
        if self.bench_actions != self.bench_restore["actions"]:
            problems.append(f"actions differ ({len(self.bench_actions)} vs {len(self.bench_restore['actions'])})")
        self.bench_restore_state = {**now, "ok": not problems, "detail": "; ".join(problems) or "ok",
                                    "pinned": {k: getattr(self, f"bench_pin_{k}", 0) for k in ("changed", "dropped", "filled")}}
        self.workspace.append("level_bench_restore_state", **self.bench_restore_state)
        self.workspace.append("level_bench_level_start", step=self.steps, levels=int(self.obs.levels),
                              level_start_step=int(self.bench_restore["level_start_step"]))
        if problems:
            self.bench_outcome = "replay_mismatch"

    async def chat(self, messages, tools):
        if (self.bench_restore is not None and self.bench_restore_state is None
                and self.rollout.restore_at == self.bench_restore["request"]):
            if self.bench_restore.get("pin_actions"):
                await self.bench_pin_fill()
            self.bench_restore_check()
        if (self.bench_restore is not None and self.bench_restore.get("check_only") and self.bench_outcome is None
                and self.bench_restore_live()):
            self.bench_outcome = "restored"
        if (self.bench_script is not None and self.bench_script_end is None
                and self.bench_script_at >= len(self.bench_script["replies"])):
            self.bench_script_finish()
        if self.bench_record is not None and self.bench_outcome is None:
            # The measure the arms' trims use (harness_fixes exact_trigger: the tokenizer's count).
            estimate = (await self.exact_gameplay_tokens(messages, tools) if self.fix("exact_trigger")
                        else self.estimate_prompt_tokens(messages, tools))
            if estimate >= self.bench_record["stop_tokens"]:
                stop = {"estimate": estimate, "requests": self.bench_script_at, "steps": self.steps,
                        "levels": int(self.obs.levels), "board_hash": self.bench_board(),
                        "completion_tokens": self.game.clock.completion_tokens(), "messages": len(messages)}
                (self.bench_dir() / "script_stop.json").write_text(json.dumps(
                    {"stop": stop, "actions": self.bench_actions}))
                self.bench_outcome = "recorded"
                self.workspace.append("level_bench_recorded", **stop)
        if self.bench_harvest_state == "due":  # the harvest request: past the gate once, 1 token (RolloutHTTP)
            self.bench_harvest_state = "sending"
            return await super().chat(messages, tools)
        self.bench_gate()
        return await super().chat(messages, tools)

    async def request_completion(self, payload, *args, **kwargs):
        gameplay = (self.request_purpose or ("summary" if not payload.get("tools") else "gameplay")) == "gameplay"
        script = self.bench_script
        if gameplay and script is not None and self.bench_script_at < len(script["replies"]):
            import copy

            # What the served request would have left behind for the summary request's shared prefix.
            self.last_gameplay_prefix = {"chat_template_kwargs": copy.deepcopy(payload.get("chat_template_kwargs") or {}),
                                         "tools": payload.get("tools")}
            self.bench_script_at += 1
            return ScriptedResponse(script["replies"][self.bench_script_at - 1])
        response = await super().request_completion(payload, *args, **kwargs)
        if gameplay and self.bench_record is not None and response.status_code == 200:
            with (self.bench_dir() / "script.jsonl").open("a") as file:
                file.write(json.dumps(reply_record(response.json()), separators=(",", ":")) + "\n")
            self.bench_script_at += 1
        return response

    async def aside_request(self, *args, **kwargs):
        if self.bench_harvest_state in ("due", "sending"):  # a summary before the harvest request: no start
            self.bench_harvest_state = "cancelled"
            self.workspace.append("level_bench_harvest_cancelled", step=self.steps)
        self.bench_gate()
        return await super().aside_request(*args, **kwargs)

    def bench_pinned(self, action):
        """Restore action pinning (17 Oct, gate v3: 16-19 of 20 deep restores diverged when re-executed tool code chose
        other actions): before the restored attempt goes live, every step takes the recording's next action; steps past
        the recorded list are dropped (None)."""
        from arc_infra.plans import Action

        recorded = self.bench_restore["actions"]
        index = len(self.bench_actions or [])
        if index >= len(recorded):
            self.bench_pin_dropped = getattr(self, "bench_pin_dropped", 0) + 1
            return None
        name, *xy = recorded[index].split()
        pinned = Action(name, int(xy[0]), int(xy[1])) if name == "ACTION6" else Action(name)
        if action is not None and recorded[index] != action.name + (f" {action.x} {action.y}" if action.name == "ACTION6" else ""):
            self.bench_pin_changed = getattr(self, "bench_pin_changed", 0) + 1
        return pinned

    async def bench_pin_fill(self):
        """Recorded actions the re-executed code never issued, played before the restore check."""
        recorded = self.bench_restore["actions"]
        while len(self.bench_actions) < len(recorded):
            self.bench_pin_filled = getattr(self, "bench_pin_filled", 0) + 1
            await self.step(self.bench_pinned(None))

    async def step(self, action):
        if (self.bench_restore is not None and self.bench_restore.get("pin_actions") and self.bench_actions is not None
                and not self.bench_restore_live() and self.bench_restore_state is None):
            if action is not None:
                action = self.bench_pinned(action)
            if action is None:
                return
        await super().step(action)
        if self.bench_actions is not None:
            self.bench_actions.append(action.name + (f" {action.x} {action.y}" if action.name == "ACTION6" else ""))
        if self.bench_restore is not None:  # a segment: the restored level, live, until it clears
            if (self.bench_outcome is None and self.bench_restore_live()
                    and self.obs.levels > self.bench_restore["stop"]["levels"]):
                self.bench_outcome = "won" if self.obs.state == "WIN" else "cleared"
                self.bench_clear = {"actions": self.steps - self.bench_restore["level_start_step"],
                                    "completion_tokens": self.game.clock.completion_tokens() - self.bench_restore["tokens"],
                                    "wall_seconds": round(self.game.clock.global_clock.elapsed(), 3),
                                    "requests": len(self.game.requests)}
                self.workspace.append("level_bench_clear", step=self.steps, **self.bench_clear)
                self.bench_harvest_due()
            return
        if self.bench_through:
            if self.bench_outcome is None and self.obs.levels > len(self.bench_levels):
                mark = {"completion_tokens": self.game.clock.completion_tokens(), "steps": self.steps}
                for level in range(len(self.bench_levels) + 1, self.obs.levels + 1):
                    self.bench_levels.append({"level": level, **mark})
                    self.workspace.append("level_bench_level", step=self.steps, level=level, **mark)
                if self.obs.state == "WIN" or self.obs.levels >= self.obs.total_levels:
                    self.bench_outcome = "won"
                    self.bench_clear = {"actions": self.steps, "completion_tokens": mark["completion_tokens"],
                                        "wall_seconds": round(self.game.clock.global_clock.elapsed(), 3),
                                        "requests": len(self.game.requests)}
                    self.workspace.append("level_bench_clear", step=self.steps, **self.bench_clear)
                else:
                    self.bench_enter_due = True
            return
        target = self.bench_start + self.bench_lead
        if self.bench_outcome is None and self.bench_lead and self.bench_enter is None and self.obs.levels >= target:
            self.bench_enter = {"completion_tokens": self.game.clock.completion_tokens(), "steps": self.steps}
            self.bench_enter_due = True
            self.workspace.append("level_bench_enter", step=self.steps, **self.bench_enter)
        if self.bench_outcome is None and self.obs.levels > target:
            enter = self.bench_enter or {"completion_tokens": 0, "steps": self.bench_offset}
            self.bench_outcome = "cleared"
            self.bench_clear = {"actions": self.steps - enter["steps"],
                                "completion_tokens": self.game.clock.completion_tokens() - enter["completion_tokens"],
                                "wall_seconds": round(self.game.clock.global_clock.elapsed(), 3),
                                "requests": len(self.game.requests)}
            self.workspace.append("level_bench_clear", step=self.steps, **self.bench_clear)
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


def wilson(successes, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def poisson_interval(events, alpha=0.05):
    """Exact (Garwood) interval of a Poisson count."""
    from scipy.stats import chi2

    lower = 0.0 if events == 0 else chi2.ppf(alpha / 2, 2 * events) / 2
    upper = chi2.ppf(1 - alpha / 2, 2 * events + 2) / 2
    return float(lower), float(upper)


def summarize(attempts, *, wall_seconds=None, budget=None):
    """Headline metrics over completed attempts (censored, errored and mismatched ones excluded)."""
    done = [a for a in attempts if a["outcome"] not in HEADLINE_EXCLUDED]
    clears = [a for a in done if a["cleared"]]
    tokens = sum(a["completion_tokens"] for a in done)
    rate = lower = upper = None
    if tokens:
        low, high = poisson_interval(len(clears))
        rate, lower, upper = (round(v * 1e4 / tokens, 4) for v in (len(clears), low, high))
    all_tokens = sum(a["completion_tokens"] for a in attempts)
    per_level = {}
    for a in done:
        row = per_level.setdefault(a["point"], {"game": a["game"], "level": a["level"], "split": a["split"],
                                                "attempts": 0, "clears": 0, "tokens": 0, "scores": []})
        row["attempts"] += 1
        row["clears"] += a["cleared"]
        row["tokens"] += a["completion_tokens"]
        row["scores"].append(a["level_score"])
    for row in per_level.values():
        scores = row.pop("scores")
        row["mean_level_score"] = round(sum(scores) / len(scores), 2)
    depth = {}
    for a in done:
        key = "L1" if a["level"] == 1 else "L2+"
        bucket = depth.setdefault(key, {"attempts": 0, "clears": 0, "tokens": 0})
        bucket["attempts"] += 1
        bucket["clears"] += a["cleared"]
        bucket["tokens"] += a["completion_tokens"]
    for bucket in depth.values():
        bucket["clears_per_10k_tokens"] = round(bucket["clears"] * 1e4 / bucket["tokens"], 4) if bucket["tokens"] else None
    arms = {}
    for a in done:
        row = arms.setdefault(a.get("arm") or "base", {"attempts": 0, "clears": 0, "tokens": 0, "points_cleared": set()})
        row["attempts"] += 1
        row["clears"] += a["cleared"]
        row["tokens"] += a["completion_tokens"]
        if a["cleared"]:
            row["points_cleared"].add(a["point"])
    for a in done:
        if "levels" in a:  # play-through attempts: levels and score per arm
            row = arms[a.get("arm") or "base"]
            row["levels"] = row.get("levels", 0) + a["levels"]
            row["game_score_sum"] = round(row.get("game_score_sum", 0.0) + a["game_score"], 3)
    for row in arms.values():
        row["points_cleared"] = sorted(row["points_cleared"])
        row["clear_rate"] = round(row["clears"] / row["attempts"], 4)
        row["clear_rate_95"] = wilson(row["clears"], row["attempts"])
        row["clears_per_mtok"] = round(row["clears"] * 1e6 / row["tokens"], 3) if row["tokens"] else None
    requests = sum(a["gameplay_requests"] for a in done)
    return {
        "token_budget": budget,
        "attempts_completed": len(done),
        "attempts_censored": sum(a["outcome"] == "censored" for a in attempts),
        "attempts_failed": sum(a["outcome"] in ("error", "replay_mismatch") for a in attempts),
        "outcomes": {k: sum(a["outcome"] == k for a in attempts) for k in sorted({a["outcome"] for a in attempts})},
        "clears": len(clears),
        "clear_rate": round(len(clears) / len(done), 4) if done else None,
        "clear_rate_95": wilson(len(clears), len(done)),
        "completion_tokens": tokens,
        "clears_per_10k_tokens": rate,
        "clears_per_10k_tokens_95": [lower, upper],
        "median_tokens_to_clear": statistics.median(a["tokens_to_clear"] for a in clears) if clears else None,
        "median_actions_to_clear": statistics.median(a["actions"] for a in clears) if clears else None,
        "mean_level_score": round(sum(a["level_score"] for a in done) / len(done), 3) if done else None,
        "mean_level_score_of_clears": round(sum(a["level_score"] for a in clears) / len(clears), 3) if clears else None,
        "noaction_share": round(sum(a["noaction_requests"] for a in done) / requests, 4) if requests else None,
        "all_completion_tokens": all_tokens,
        "censored_completion_tokens": sum(a["completion_tokens"] for a in attempts if a["outcome"] == "censored"),
        "tokens_per_second": round(all_tokens / wall_seconds, 1) if wall_seconds else None,
        "by_depth": depth,
        "by_arm": dict(sorted(arms.items())),
        "per_level": dict(sorted(per_level.items())),
    }


KV_PROJECT = {"growth": 1.8, "ctx_frac": 0.87, "scale": 0.74}  # fitted on ovf2/ovf3 R1 (sc25-L3 base play)


def turn_auto_step(pool, kv, spec, now=None):
    """mg.turn_auto (v6k 25 Oct: running tries grew server 1's KV 0.31 -> 0.89 in 9 min under a 70-slot cap; a slot cap
    bounds starts, not the growth of tries already in): the pool's turn cap follows the engine's own KV. Engine KV
    samples swing 0.12 <-> 0.90 within seconds (v6l 05:44, between turns): a sample above ``hi`` drops the cap by
    ``step`` (from the live count when off) at most once per ``cut_s``; it rises by ``step`` (up to ``max``, tries
    waiting) only when the max over ``window_s`` is below ``lo``, at most once per ``reopen_s``. Returns the new cap
    or None."""
    if kv is None or not spec:
        return None
    now = time.time() if now is None else now
    window = float(spec.get("window_s", 60.0))
    hist = [x for x in getattr(pool, "_turn_kv", []) if now - x[0] <= window] + [(now, float(kv))]
    pool._turn_kv = hist
    peak = max(v for _, v in hist)
    hi, lo = float(spec.get("hi", 0.80)), float(spec.get("lo", 0.65))
    step, floor = int(spec.get("step", 4)), int(spec.get("min", 12))
    top = int(spec.get("max") or pool.slots)
    cap = int(pool.turn_cap or 0) or len(pool.live)
    last = getattr(pool, "_turn_moved", -1e18)
    if kv > hi and now - last >= float(spec.get("cut_s", 20.0)):
        new = max(floor, min(cap, len(pool.live)) - step)
    elif peak < lo and pool.turn_cap and len(pool.live) > cap and now - last >= float(spec.get("reopen_s", 60.0)):
        new = min(top, cap + step)
    else:
        return None
    if pool.turn_cap and new == int(pool.turn_cap):
        return None
    pool.turn_cap, pool._turn_moved = new, now
    return new


def pool_state(pool):
    """One pool's admission state for the feed's ``pool`` row: slots, live tries, queue, projected fill, scale."""
    out = {"active": bool(getattr(pool, "active", True)), "slots": pool.slots, "used": pool.used,
           "live": len(pool.live), "kv_waits": pool.kv_waits, "paused": getattr(pool, "paused_why", None)}
    try:
        out["queued"] = pool.queued()
        out["room"] = pool.room()
    except Exception as error:  # noqa: BLE001 - a diagnostic row
        out["error"] = repr(error)[:120]
    if pool.project is not None:
        out["engine_scale"] = pool.project.get("engine_scale", 1.0)
        out["auto_off"] = bool(pool.project.get("auto_off"))
        if pool.capacity:
            out["projected"] = round(pool.load() / max(1, pool.capacity_tokens()), 3)
    out["kv_fill"] = pool.kv_fill
    out["turn_cap"], out["turn_waits"] = getattr(pool, "turn_cap", None), getattr(pool, "turn_waits", 0)
    out["sending"] = len(getattr(pool, "sending", ()))
    return out


class SharedSlots:
    """mg5f: one pool of attempt slots for every bench running at once on one server (each bench's own gate counts the
    whole KV pool as its own). An attempt starts only on a free slot, so the running attempts' contexts stay in the
    prefix cache; a bench with a larger token budget takes slots first, then first come.

    mg6 (``kv_fill``): one KV gate for all benches, and a new attempt starts only while the running attempts' live
    contexts (latest prompt + cap; an attempt before its first request counts as the median live context) plus one
    more stay under ``kv_fill`` x the gate capacity. Without it, long contexts fill the cache and every turn
    re-prefills an evicted prefix (screen mg6-4: KV 99%, 570 generated tok/s against 18K prompt tok/s)."""

    def __init__(self, slots, kv_fill=None, new_tokens=32000, project=None, stagger_s=0.0):
        self.slots, self.used, self.peak, self.pending = int(slots), 0, 0, {}
        # mg.admit_stagger_s: a bench's tries start at least this far apart, so the tries of
        # one group (one start) do not grow in lockstep; other benches take slots meanwhile
        self.stagger_s, self.last_take = float(stagger_s or 0.0), {}
        self.kv_fill, self.new_tokens = (float(kv_fill) if kv_fill else None), int(new_tokens)
        # mg.kv_project (MFU-2, 23 Oct, after ovf2 R1's thrash at 132 tries): the kv_fill gate counts each live try at
        # its projected end-of-try context instead of its next request; see project_tokens
        self.project = {**KV_PROJECT, **project} if isinstance(project, dict) else (dict(KV_PROJECT) if project else None)
        self.project_peak, self.scale_samples = 0.0, []
        self.capacity, self.gate, self.live, self.kv_waits, self.peak_load = None, None, set(), 0, 0
        self.budget = None  # the gate's cache budget: capacity and loads in its units (blocks), not tokens
        self.active, self.served = True, 0  # a second server's pool (time share) takes attempts only while active
        self.hold_count, self.hold_until, self.joined = 0, 0.0, 0
        self.paused_why, self.engine_waits = None, 0  # EngineAdmission: the engine's own KV/preemption/hit signals
        # explorers (mg.explore) play outside the pool; their KV (tokens; set by the explorer as plays start and
        # end) counts in the kv_fill gate (ovf3r3 R6 KV replay, 23 Oct: uncounted explorers put the peak at 1.6)
        self.reserved = 0.0
        # class shares (v6h 24 Oct): reads outrank every training group (oldest_first) and took the whole projected-KV
        # gate; a classed bench ("read") takes tries only while its class's projected load stays under share x fill
        self.classes, self.shares = {}, {}
        # turn_cap (card2_turn_cap, MFU-2 25 Oct, v6j3 server 2: 128 live tries ~10M tokens against a 4.87M-token KV pool,
        # every turn re-prefilled an evicted prefix, 88% of block time on prefill): while the pool is active only the
        # turn_cap earliest-entered live tries may send requests; the rest wait at their next request until one ends.
        # None/0 = off. The set is sticky (entry order), so the active tries' prefixes stay cached between turns.
        self.turn_cap, self.entered, self.turn_waits, self.enter_seq = None, {}, 0, 0
        # v6l 05:26-05:43 (server 2 at 0-1 running, 45 slots used, turn_waits 31): ranking every entered try let 32 idle
        # ones (restoring, in tool time) hold the places; the cap now counts requests in flight, and among tries waiting
        # to send the earliest-entered go first (their prefixes stay warm)
        self.sending, self.wanting = {}, {}  # try -> requests in flight (len = tries in flight); try -> entry order

    def classify(self, key, cls):
        if cls:
            self.classes[key] = cls

    def class_load(self, cls):
        """``load`` over the live tries of one class (group_prefix: a start's prefix once + growth, engine_scale)."""
        mine = [s for s in self.live if getattr(s, "pool_cls", None) == cls]
        if self.project is None:
            return sum(self._units(s.next_estimate + s.cap) for s in mine if s.next_estimate > 0)
        if not self.project.get("group_prefix"):
            return sum(self.project_tokens(s) for s in mine)
        prefix, growth = {}, 0.0
        for s in mine:
            floor = int(getattr(s, "floor_tokens", 0) or 0)
            k = getattr(s, "prefix_key", None) or id(s)
            prefix[k] = max(prefix.get(k, 0), floor)
            growth += max(0.0, self.project_end(s) - floor)
        return (sum(prefix.values()) + growth) * float(self.project.get("engine_scale", 1.0))

    def capped(self, key):
        """Is ``key``'s class at its share of the KV gate? A capped bench neither takes nor holds lower ones back."""
        share = self.shares.get(self.classes.get(key))
        if share is None or not (self.kv_fill and self.capacity):
            return False
        room = self.capacity_tokens() if self.project is not None else self.capacity
        return self.class_load(self.classes[key]) + self._allowance() > float(share) * self.kv_fill * room

    def hold(self, count, seconds=120):
        """tail_fill: no attempt starts until ``count`` benches have registered (or ``seconds`` pass), so the slot
        priorities (held-out last) also order the round's first wave, not only the first bench to register."""
        self.hold_count, self.hold_until, self.joined = int(count), time.time() + seconds, 0

    def want(self, key, priority, count, join=True):
        """``join`` False: a background bench (mg.explore) never counts towards a round's ``hold``."""
        if key not in self.pending and join:
            self.joined += 1
        self.pending[key] = (priority, int(count))

    def _units(self, tokens):
        return self.budget.cost(tokens) if self.budget is not None else int(tokens)

    def project_tokens(self, s):
        """kv_project: a live try's end-of-try context in engine tokens. Its context now (the latest prompt, or the
        restored prefix before its first live request) grows by ``growth`` tokens per completion token left in its
        budget, up to ``ctx_frac`` x (window - cap): a try at the window drops old turns and saw-tooths below it
        (ovf3 R1: prompts drop at ~114K and average ~0.87 of it). ``scale``: engine KV per client-side context token
        (prefix sharing between tries of a start; ovf3 R1 at 74 in flight: KV 0.851 for 1.152 pools of contexts)."""
        return self.project_end(s) * float(self.project["scale"])

    def project_end(self, s):
        """A live try's projected end-of-try context in client tokens (no prefix-sharing credit)."""
        p = self.project
        ctx = max(int(s.next_estimate), int(getattr(s, "floor_tokens", 0) or 0))
        progress = getattr(s, "progress", None)
        left = 0
        if progress is not None:
            generated, budget = progress()
            left = max(0, int(budget or 0) - int(generated or 0))
        return min(ctx + left * float(p["growth"]), float(p["ctx_frac"]) * max(1, int(s.window) - int(s.cap)))

    def _allowance(self):
        if self.project is not None:  # a try not yet started: the median live try's projection at its own start
            known = sorted(s.project_start for s in self.live if getattr(s, "project_start", None))
            return known[len(known) // 2] if known else self.new_tokens * float(self.project["scale"])
        known = sorted(self._units(s.next_estimate + s.cap) for s in self.live if s.next_estimate > 0)
        return known[len(known) // 2] if known else self._units(self.new_tokens)

    def load(self):
        new = self._allowance()
        if self.project is not None:
            if self.project.get("group_prefix"):  # KV replay 23 Oct: vLLM shares a start's restored prefix between the
                # tries restored from it (one copy), never their growth; ``scale`` (fitted where the prefix dominates)
                # overcredited long tries. A start's prefix once + every try's growth in full.
                prefix, growth = {}, 0.0
                for s in self.live:
                    floor = int(getattr(s, "floor_tokens", 0) or 0)
                    key = getattr(s, "prefix_key", None) or id(s)
                    prefix[key] = max(prefix.get(key, 0), floor)
                    growth += max(0.0, self.project_end(s) - floor)
                # engine_scale (RL 24 Oct): engine KV per client token, from scale_all samples (v6e: projected 0.886
                # in client tokens while the engine ran to 0.997); 1.0 keeps the old gate
                return (sum(prefix.values()) + growth) * float(self.project.get("engine_scale", 1.0)) + self.reserved
            return sum(self.project_tokens(s) for s in self.live) + self.reserved
        return (sum(self._units(s.next_estimate + s.cap) if s.next_estimate > 0 else new for s in self.live)
                + (self._units(int(self.reserved)) if self.reserved else 0))

    def scale_sample(self, kv):
        """kv_project: one engine KV reading against the in-flight tries' client-side contexts (the measured
        ``scale``; logged, not applied). None when fewer than half the live tries are in flight."""
        if self.project is None or not self.capacity or not self.live or kv is None:
            return None
        flying = [s for s in self.live if getattr(s, "_stack", None)]
        tokens = sum(max(int(s.next_estimate), int(getattr(s, "floor_tokens", 0) or 0)) for s in flying)
        if len(flying) * 2 < len(self.live) or tokens <= 0:
            return None
        ratio = float(kv) * self.capacity_tokens() / tokens
        self.scale_samples = (self.scale_samples + [round(ratio, 4)])[-200:]
        # the same reading over every live try (the engine also holds the cached prefixes of tries between requests):
        # the ratio engine_scale should take
        every = sum(max(int(s.next_estimate), int(getattr(s, "floor_tokens", 0) or 0)) for s in self.live)
        if every > 0:
            self.scale_all = (getattr(self, "scale_all", []) + [round(float(kv) * self.capacity_tokens() / every, 4)])[-200:]
        return ratio

    def calibrate(self, kv, now=None):
        """kv_project.auto_scale (v6h 24 Oct: projected 0.75 against engine KV 0.23): engine_scale follows the measured
        engine KV over the unscaled projection, its ``q`` quantile over the last ``window_s`` (young tries understate
        their end-of-try load: a high quantile, ``floor``; EngineAdmission still pauses on the engine's own KV).
        Returns the new scale when it moved by more than 0.05, else None."""
        auto = (self.project or {}).get("auto_scale")
        if not auto or self.project.get("auto_off") or not self.project.get("group_prefix") or kv is None \
                or not self.capacity or not self.live:
            return None
        now = time.time() if now is None else now
        es = float(self.project.get("engine_scale", 1.0))
        raw = (self.load() - self.reserved) / max(es, 1e-9)
        if raw <= 0:
            return None
        ratio = (float(kv) * self.capacity_tokens() - self.reserved) / raw
        window = float(auto.get("window_s", 1800))
        self.calib = [x for x in getattr(self, "calib", []) if now - x[0] <= window] + [(now, ratio)]
        if now - self.calib[0][0] < float(auto.get("min_s", 600)):
            return None
        xs = sorted(r for _, r in self.calib)
        q = xs[min(len(xs) - 1, int(float(auto.get("q", 0.8)) * len(xs)))]
        new = round(min(float(auto.get("ceil", 1.0)), max(float(auto.get("floor", 0.5)), q)), 3)
        if abs(new - es) <= 0.05:
            return None
        self.project["engine_scale"] = new
        return new

    def capacity_tokens(self):
        """kv_project works in tokens: the engine's whole KV pool (a block budget's blocks at its longest row's
        tokens per block, without the budget's margin)."""
        if self.budget is None:
            return self.capacity
        layout = self.budget.layout
        return int(layout["num_blocks"] * layout["costs"][-1]["tokens"] / layout["costs"][-1]["blocks"])

    def remaining(self, key, count):
        if key in self.pending:
            self.pending[key] = (self.pending[key][0], int(count))

    def held(self, key, now=None):
        """Is ``key``'s next try held back by its admission stagger?"""
        if not self.stagger_s:
            return False
        return (time.time() if now is None else now) - self.last_take.get(key, -1e18) < self.stagger_s

    def room(self):
        """Would a try waiting only on its turn be admitted now (slots, engine pause, KV gate)? No side effects."""
        if not self.active or self.used >= self.slots or (self.paused_why and self.used > 0):
            return False
        if self.kv_fill and self.capacity and self.live:
            room = self.capacity_tokens() if self.project is not None else self.capacity
            return self.load() + self._allowance() <= self.kv_fill * room
        return True

    def queued(self, now=None, floor=-10 ** 9):
        """Tries above ``floor`` priority that wait on the pool. A bench's tries held back only by its admission
        stagger are not a queue while the pool has room (hero-v6 24 Oct: counting them stopped group creation at 3
        groups with 2-4 live tries)."""
        now = time.time() if now is None else now
        room = self.room()
        n = 0
        for key, (pr, c) in self.pending.items():
            if pr <= floor or c <= 0:
                continue
            if room and self.held(key, now):
                continue
            if self.capped(key):  # a read at its share is no queue for training groups
                continue
            n += c
        return n

    def take(self, key):
        priority, count = self.pending.get(key, (0, 0))
        if not self.active or count <= 0 or self.used >= self.slots:
            return False
        if self.hold_count and self.joined < self.hold_count and time.time() < self.hold_until:
            return False
        now = time.time()
        # a higher-priority bench holds lower ones back only while it could take a try itself: one held back by its
        # own admission stagger does not (hero-v6/v6b 24 Oct: the highest-cap group and the held-out read, each
        # admitting one try per 150 s, blocked every other bench; 2-4 live tries against 38)
        if any(p > priority and n > 0 and not self.held(k, now) and not self.capped(k)
               for k, (p, n) in self.pending.items() if k != key):
            return False
        if self.held(key, now):
            return False
        if self.capped(key):
            self.kv_waits += 1
            return False
        if self.paused_why and self.used > 0:  # an idle pool always starts one: a pause never deadlocks the round
            self.engine_waits += 1
            return False
        if self.kv_fill and self.capacity and self.live:
            value = self.load() + self._allowance()
            room = self.capacity_tokens() if self.project is not None else self.capacity
            if value > self.kv_fill * room:
                self.kv_waits += 1
                return False
            if self.project is not None:
                self.project_peak = max(self.project_peak, value / room)
        self.used += 1
        self.peak = max(self.peak, self.used)
        self.pending[key] = (priority, count - 1)
        if self.stagger_s:
            self.last_take[key] = time.time()
        return True

    def enter(self, slot, key=None):
        if key is not None:
            slot.pool_cls = self.classes.get(key)
        if self.project is not None:
            slot.project_start = self.project_end(slot) if self.project.get("group_prefix") else self.project_tokens(slot)
        self.live.add(slot)
        self.enter_seq += 1
        self.entered[id(slot)] = self.enter_seq
        self.peak_load = max(self.peak_load, self.load())

    def give(self, slot=None):
        self.used -= 1
        self.live.discard(slot)
        self.entered.pop(id(slot), None)
        self.sending.pop(id(slot), None)
        self.wanting.pop(id(slot), None)

    def may_send(self, slot):
        """May ``slot`` send now (always, when the gate is off or the pool inactive)? Under the cap: fewer than turn_cap
        requests in flight, and ``slot`` among the earliest-entered of the tries waiting for the free places."""
        cap = int(self.turn_cap or 0)
        if cap <= 0 or not self.active or id(slot) not in self.entered or id(slot) in self.sending:
            return True
        free = cap - len(self.sending)
        if free <= 0:
            return False
        mine = self.entered[id(slot)]
        return sum(1 for k, r in self.wanting.items() if r < mine and k != id(slot)) < free

    async def turn(self, slot, poll_s=0.2):
        """Wait until ``slot`` may send (turn_cap / card2_turn_cap). It counts in flight only from ``begin`` (after the
        game's active wait and the KV gate): a parked or queued try holds no place (MFU-2 review, 25 Oct)."""
        key = id(slot)
        self.wanting[key] = self.entered.get(key, 0)
        try:
            if not self.may_send(slot):
                self.turn_waits += 1
                while not self.may_send(slot):
                    await asyncio.sleep(poll_s)
        finally:
            self.wanting.pop(key, None)

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
        self.classes.pop(key, None)


METRIC_KEYS = ("vllm:kv_cache_usage_perc", "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total",
               "vllm:prefix_cache_hits_total")


def engine_sample(text):
    """The admission signals of a vLLM /metrics page (label sets summed): kv, preempt, queries, hits; None if absent."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in METRIC_KEYS:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    if "vllm:kv_cache_usage_perc" not in out:
        return None
    return {"kv": out["vllm:kv_cache_usage_perc"], "preempt": out.get("vllm:num_preemptions_total", 0.0),
            "queries": out.get("vllm:prefix_cache_queries_total", 0.0), "hits": out.get("vllm:prefix_cache_hits_total", 0.0)}


class EngineAdmission:
    """mg.engine_admission (after R13's KV thrash under kv_fill 3.0): new tries start only while
    the engine's KV use is under ``kv_max``, no preemption happened in the last ``preempt_s`` seconds, and the prefix
    hit rate of the last ``hit_s`` seconds is no more than ``hit_drop`` under the median of the last ``hit_hist``
    such readings. Running tries go on; the static kv_fill stays a ceiling. Pure: ``update`` takes one sample."""

    def __init__(self, kv_max=0.85, preempt_s=60.0, hit_s=60.0, hit_drop=0.05, hit_hist=30, hit_kv=0.5):
        self.kv_max, self.preempt_s, self.hit_s, self.hit_drop = float(kv_max), float(preempt_s), float(hit_s), float(hit_drop)
        self.hit_kv = float(hit_kv)  # 23 Oct: the hit-rate rule only with KV >= hit_kv (r5 R13: 47 "hit" pauses at KV
        # 0.03-0.32; new tries' fresh prompts lower the hit rate, so the rule throttled the ramp-up itself)
        self.hit_hist, self.samples, self.rates, self.last_preempt = int(hit_hist), [], [], None

    def update(self, t, sample):
        """``(why or None)`` after this sample (``sample`` from engine_sample; None: no signal, no pause)."""
        if sample is None:
            return None
        if self.samples and sample["preempt"] > self.samples[-1][1]["preempt"]:
            self.last_preempt = t
        self.samples.append((t, sample))
        self.samples = [x for x in self.samples if x[0] >= t - max(self.hit_s, self.preempt_s) - 30]
        old = next((x for x in self.samples if x[0] >= t - self.hit_s), None)
        rate = None
        if old is not None and old[0] < t and sample["queries"] - old[1]["queries"] > 0:
            rate = (sample["hits"] - old[1]["hits"]) / (sample["queries"] - old[1]["queries"])
        why = None
        if sample["kv"] >= self.kv_max:
            why = "kv"
        elif self.last_preempt is not None and t - self.last_preempt < self.preempt_s:
            why = "preempt"
        elif rate is not None and len(self.rates) >= 5 and sample["kv"] >= self.hit_kv:
            median = sorted(self.rates)[len(self.rates) // 2]
            if rate < median - self.hit_drop:
                why = "hit"
        if rate is not None:
            self.rates = (self.rates + [rate])[-self.hit_hist:]
        return why


PROGRESS = {"done": 0, "clears": 0}  # finished tries since the caller's last reset
# the RL loop's callable(bench name, finished row), called once per finished
# try beside the PROGRESS counters
PROGRESS_HOOK = None
RESTORE_CACHE = {}  # (attempt, request) -> restore.load(...): the event loop parsed each start's rollout once per try
def balanced(*pools):
    """two_server_placement "balanced": pools to try for a new attempt, the least loaded (used / slots) first; ties keep
    the given order. Two servers
    then share every bench's tries, so with longest-first order (tail_order) the long tries spread over both cards and
    neither card idles through the tail (seed screen 22 Oct: server 1 took every try it could, card 2 sat at 25% for
    the last hour)."""
    live = [p for p in pools if p is not None]
    return sorted(live, key=lambda p: (p.used / max(1, p.slots)) if p.active else 2.0)


SHARED_SLOTS = None  # set by the caller (rl_loop mg.active_tries); None: each bench alone
SHARED_SLOTS2, SERVER2_URL = None, None  # time share: a second server's pool and URL (rl_loop mg.timeshare)
SERVER2_VERSIONS = None  # v7 Mode B: adapter versions server 2 holds; a bench on another version stays on server 1
TRANSPORT2 = None  # v7 Mode B (rl_async.Route2): server 2's requests, sent to server 1 while server 2 sleeps
# restart from finished tries: each finished try is copied to TRY_KEEP/<bench>/ as it ends;
# a relaunch with TRY_REUSE (earlier runs' keep roots) skips every planned try found there with the same plan entry.
# Benches named in TRY_REUSE_SKIP (their starts come from a recording replayed in this run) replay everything.
TRY_KEEP, TRY_REUSE, TRY_REUSE_SKIP = None, [], set()
BENCH_DONE = {}  # bench output name -> finished (or reused) tries: v7's in_flight counts a group's unfinished tries


def plan_digest(label, point, budget, cap):
    import hashlib

    text = json.dumps([label, point, budget, cap], sort_keys=True, default=str)
    # a packed preflight leg's restore paths are read from its unpacked /tmp copy: digest them as the volume paths
    # they stand for, so a resumed leg's kept tries match (pfB4, 22 Oct); real runs never hold /tmp/validation paths
    return hashlib.sha1(text.replace('"/tmp/validation/', '"/runs/validation/').encode()).hexdigest()


def kept_tries(name, roots):
    """{label: (row, attempt folder)} of finished tries kept by earlier runs (the latest root wins). A root may be
    ``(path, version)`` (mg.lookahead_below: a versioned keep root): its rows without their own ``policy_version``
    get that one."""
    out = {}
    for root in roots:
        version = None
        if isinstance(root, (tuple, list)):
            root, version = root
        try:
            lines = (Path(root) / name / "tries.jsonl").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a line cut by the stop
            folder = Path(root) / name / "attempts" / row["label"]
            if row.get("outcome") not in ("error", "replay_mismatch", "censored") and (folder / "rollout.jsonl").exists():
                if version is not None:
                    row.setdefault("policy_version", version)
                out[row["label"]] = (row, folder)
    return out


# Resume mid-try (RL 24 Oct, proposal b): every snap_s seconds each running try's rollout.jsonl and log.jsonl are
# appended (whole lines only) to <keep>/<bench>/live/<label>/ with meta.json; a finished try's folder is removed. A
# relaunch's replay of the bench restores a planned try found there (same plan digest) at its last restorable request:
# its earlier replies are answered from the copy (its own replies stay trainable), then it plays live.
SNAP_FILES = ("rollout.jsonl", "log.jsonl")


def live_tries(name, roots):
    """{label: (meta, snapshot folder)} of unfinished tries snapshotted by earlier runs (the latest root wins)."""
    out = {}
    for root in roots:
        if isinstance(root, (tuple, list)):
            root = root[0]
        for meta_path in sorted((Path(root) / name / "live").glob("*/meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                continue
            out[meta["label"]] = (meta, meta_path.parent)
    return out


def snap_copy(src, dest, offsets):
    """Append ``src``'s whole new lines to ``dest`` (append-only files); ``offsets`` {file: bytes copied}. Returns
    {file: complete lines copied so far}."""
    dest.mkdir(parents=True, exist_ok=True)
    lines = {}
    for name in SNAP_FILES:
        start = offsets.get(name, 0)
        try:
            with open(src / name, "rb") as file:
                file.seek(start)
                data = file.read()
        except OSError:
            data = b""
        cut = data.rfind(b"\n") + 1
        if cut:
            with open(dest / name, "ab" if start else "wb") as file:
                file.write(data[:cut])
            offsets[name] = start + cut
            offsets[name + ":lines"] = offsets.get(name + ":lines", 0) + data[:cut].count(b"\n")
        lines[name] = offsets.get(name + ":lines", 0)
    return lines


def resume_request(folder):
    """``(request, base tokens)`` to restore a snapshotted try at: its last gameplay request after the start's own
    scripted prefix whose step the log copy has, before any event that cannot replay (restore.unsafe_from); None when
    no own reply would be kept. Base tokens: the scripted prefix's completion tokens (they never counted against the
    budget, as in the original try)."""
    from .restore import load, log_rows, unsafe_from

    folder = Path(folder)
    try:
        rows = [json.loads(line) for line in open(folder / "rollout.jsonl")]
        seen = {int(o["step"]) for o in log_rows(folder / "log.jsonl", ("observation",)) if o.get("frames")}
        first_unsafe, _ = unsafe_from(folder)
    except (OSError, ValueError, KeyError):
        return None
    prefix = sum(1 for r in rows if r.get("scripted"))
    if any(r.get("scripted") for r in rows[prefix:]):  # a prefix that is not one leading run: not a restore point
        return None
    for k in range(len(rows) - 1, prefix, -1):
        r = rows[k]
        step = int(r.get("step") or 0)
        if not r.get("gameplay") or r.get("harvest") or step not in seen:
            continue
        if first_unsafe is not None and step >= first_unsafe:
            continue
        try:
            load(folder, k)
        except (OSError, ValueError, KeyError, IndexError):
            continue
        base = sum(int((x.get("usage") or {}).get("completion_tokens") or 0) for x in rows[:prefix])
        return k, base
    return None


def gate_capacity(study, cfg, server_dir, window):
    """``(capacity, budget, choice)`` for the KV gate on this backend."""
    if cfg["capacity_tokens"] is not None:
        return int(cfg["capacity_tokens"]), None, {"units": "tokens", "source": "level_bench.capacity_tokens"}
    if study.get("backend") == "sglang":
        from .sglang import TokenBudget

        budget = TokenBudget.load(server_dir, study)
        source = "sglang pool"
    elif study.get("cache_admission") == "hybrid_blocks":
        from .cache import BlockBudget

        budget = BlockBudget.load(server_dir, float(study["kv_margin"]))
        source = "hybrid cache blocks"
    else:
        from .capacity import kv_pool_tokens

        pool = kv_pool_tokens(server_dir)
        capacity = int(pool * float(study["kv_margin"]))
        if capacity < window:
            raise RuntimeError("One attempt window does not fit the KV pool")
        return capacity, None, {"units": "tokens", "source": "vLLM GPU KV cache size", "pool": pool}
    if budget.cost(window) > budget.capacity:
        raise RuntimeError("One attempt window does not fit the served cache; no silent context downgrade")
    return budget.capacity, budget, {"units": getattr(budget, "units", "cache_blocks"), "source": source,
                                     "full_window_cost": budget.cost(window), "layout_rule": budget.layout.get("rule")}


class LevelBench:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir, analysis_python=None,
                 base_url, deadline_epoch, log=print, arcade=None, transport=None):
        self.study, self.trial = study, trial
        self.cfg = settings(study.get("level_bench"))
        for key in UNSUPPORTED:
            if study.get(key):
                raise ValueError(f"The level bench does not support the trial-level service {key!r}")
        self.source, self.games_dir, self.output = Path(source), games_dir, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.server_dir, self.base_url = server_dir, base_url
        self.analysis_python = analysis_python or "python3"
        self.deadline_epoch, self.log, self.arc = float(deadline_epoch), log, arcade
        self.transport = transport  # tests: an httpx transport in place of the server
        self.transport2 = None  # tests: the second (time-share) server's transport
        self.cap = int(study["completion_cap"])
        self.attempts, self.requests = [], []
        self.reused = []  # finished tries taken from an earlier run (TRY_REUSE)
        # mg.lookahead_below (MFU-2, 22 Oct): a bench may carry its own keep/reuse roots (a look-ahead bench outlives the
        # round whose roots the globals hold), its adapter version (written into every try row) and server-1 only
        own = study.get("try_keep") or {}
        self.try_keep = own.get("keep") if "keep" in own else TRY_KEEP
        self.try_reuse = list(own.get("reuse") or []) if "reuse" in own else TRY_REUSE
        self.snap_s = float(own.get("snap_s") or 0) if self.try_keep is not None else 0.0  # resume mid-try (b)
        import threading

        self.snap_lock = threading.Lock()
        self.policy_version = study.get("policy_version")
        self.server1_only = bool(study.get("server1_only"))
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

        from .trial import duck_config

        cfg = self.cfg
        window = cfg["window"]
        if window <= self.cap:
            raise ValueError("The attempt window must exceed the completion cap")
        capacity, budget, choice = gate_capacity(self.study, cfg, self.server_dir, window)
        request_capacity = capacity
        if self.study.get("gate_overcommit"):  # RL screens/loops (19 Oct): the per-request gate (prompt + full reply cap
            # per request in flight) ran at 99% reserved with requests queued while the server's running requests held
            # 48% of the pool. Only that gate is scaled: the shared attempt pool (kv_fill) keeps the true capacity, so
            # every live attempt's context (idle ones' cached prefixes too) still fits and nothing is re-prefilled
            request_capacity = int(capacity * float(self.study["gate_overcommit"]))
            choice = {**choice, "overcommit": float(self.study["gate_overcommit"])}
        points = load_points(self.source / cfg["start_points"], cfg)
        work = plan(points, cfg["passes"], cfg["order_seed"], cfg["arms"], cfg["first_pass"])
        bench_wall = GlobalClock(self.wall_seconds())
        agent_cfg = duck_config(self.study, self.source, window=window, cap=self.cap,
                                wall=cfg["attempt_wall_seconds"], base_url=self.base_url)
        agent_cfg.max_actions = 10 ** 6  # replayed history counts as steps; the token budget ends the attempt
        agent_cfg.parallel_games = cfg["concurrency"]
        agent_cfg.max_num_seqs = max(agent_cfg.max_num_seqs, cfg["concurrency"])
        from duck_transfer import loader

        loader.configure(agent_cfg)
        self.config_dict = agent_cfg.to_dict()
        pool = SHARED_SLOTS
        if pool is not None and pool.kv_fill:  # one gate for every bench on this server
            if pool.capacity is None:
                pool.capacity, pool.budget = capacity, budget
            pool.gate = pool.gate or KVGate(request_capacity, slots=max(cfg["concurrency"], int(self.study.get("generation_slots") or 0)),
                                            budget=budget)
            gate = pool.gate
        else:
            gate = KVGate(request_capacity, slots=cfg["concurrency"], log=self.gate_log, budget=budget)
        arc = self.arc or make_arcade("offline", self.games_dir, self.output / "recordings")
        card = arc.open_scorecard(tags=["level-bench", self.trial["name"]])
        self.checks = (arc, card)
        # idle connections retire before the server's 5 s keep-alive closes them (as the arena's trial client): at
        # httpx's default 5 s a reused connection raced the server's close (base ref: 58 ReadErrors in 4,236 requests)
        shared = httpx.AsyncClient(base_url=self.base_url.rstrip("/") + "/", trust_env=False, transport=self.transport,
                                   limits=httpx.Limits(max_connections=256, max_keepalive_connections=128,
                                                       keepalive_expiry=KEEPALIVE_EXPIRY))
        for adapter in self.study.get("load_adapters") or []:  # a standalone read of trained adapters (RL, 25 Oct):
            # served under their own names before any try (arms pick one through study.served_model)
            got = await shared.post("load_lora_adapter", json={"lora_name": adapter["name"], "lora_path": adapter["path"],
                                                                  "is_3d_lora_weight": True}, timeout=600)
            if got.status_code >= 400 and "already" not in got.text:
                raise RuntimeError(f"load_lora_adapter {adapter['name']}: {got.status_code} {got.text[:200]}")
            self.log(json.dumps({"event": "adapter_loaded", **adapter}), flush=True)
        monitor = GpuMonitor(self.output, self.base_url)
        header = {"event": "level_bench_start", "trial": self.trial["name"], "points": len(points),
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
        sampling = asyncio.create_task(monitor.run())
        running = {}
        next_index = 0
        last_progress = -math.inf
        name = self.output.name
        kept = {} if name in TRY_REUSE_SKIP else kept_tries(name, self.try_reuse)
        snaps = {} if name in TRY_REUSE_SKIP else live_tries(name, self.try_reuse)

        def planned(i):
            pass_index, point, arm = work[i]
            label = f"a{i:03d}-{point['id']}-p{pass_index}" + (f"-{arm['name']}" if cfg["arms"] else "")
            return label, plan_digest(label, point, token_budget(cfg, point), self.cap)

        async def reuse(i):
            label, digest = planned(i)
            hit = kept.get(label)
            if hit is None or hit[0].get("plan") != digest:
                return False
            import shutil

            # off the event loop (r4, 22 Oct: ~170 copies from the runs volume held every bench for ~7 min)
            await asyncio.to_thread(shutil.copytree, hit[1], self.output / "attempts" / label, dirs_exist_ok=True)
            from arc_infra.fleet import ACTIVITY  # a reused try is progress for the stall watchdog (pf jra-v1 pf1r)
            ACTIVITY["last_reply"] = time.time()
            self.reused.append({"index": i, "row": hit[0], "censored": False})
            BENCH_DONE[self.output.name] = BENCH_DONE.get(self.output.name, 0) + 1
            PROGRESS["reused"] = PROGRESS.get("reused", 0) + 1
            return True

        def resume_of(i):
            """The snapshot spec to restore planned try ``i`` mid-try, or None (play it from its start)."""
            label, digest = planned(i)
            hit = snaps.get(label)
            point = work[i][1]
            # a restored start keeps its base point's ``through`` (rl1_points: every game point), yet plays as a segment
            # (a restore ends at its level clear): v6i restored none of v6h's ~66 snapshots on L2+ starts
            if (hit is None or hit[0].get("plan") != digest or (point.get("through") and not point.get("restore"))
                    or point.get("script")
                    or point.get("prefix") or point.get("lead_levels") or (point.get("restore") or {}).get("hint")
                    or (point.get("restore") or {}).get("check_only") or not cfg["rollout"]):
                return None
            found = resume_request(hit[1])
            if found is None:
                self.log(json.dumps({"event": "level_bench_resume_skipped", "label": label}), flush=True)
                return None
            return {"folder": str(hit[1]), "request": found[0], "base_tokens": found[1],
                    "policy_version": hit[0].get("policy_version")}

        async def snapper():
            while True:
                await asyncio.sleep(self.snap_s)
                try:
                    await asyncio.to_thread(self.snap_all, list(running.values()))
                except Exception as exc:  # noqa: BLE001 - a failed snapshot costs only the mid-try resume
                    self.log(json.dumps({"event": "level_bench_snap_failed", "error": str(exc)[:200]}), flush=True)

        snap_task = asyncio.create_task(snapper()) if self.snap_s > 0 else None
        key = object()
        priority = bench_priority(cfg, self.study)
        # tail_fill (RL loop, 22 Oct): held-out benches take slots after every other bench's tries have started; within
        # a bench the longest expected tries start first (the round's stragglers start early, not last)
        tail = self.study.get("tail_fill")
        # two_server_placement (envs 22 Oct): "first" (default) fills server 1 first; "balanced" sends each new try to
        # the less loaded server (screens: both cards end together; RL time share: server 2 while awake, r5 R14).
        placement = self.study.get("two_server_placement") or "first"
        if placement not in ("first", "balanced"):
            raise ValueError(f"two_server_placement {placement!r}: first | balanced")
        order = tail_order(work, tail.get("expected")) if tail else list(range(len(work)))
        if tail and tail.get("last"):
            priority -= 10 ** 9
        if tail and tail.get("first"):  # mg.heldout_first: the main read lands soon after the update
            priority += 10 ** 9
        floor = bool(tail and tail.get("floor"))  # mg.explore (envs 22 Oct): explorer plays take only slots no bench wants
        if floor:
            priority -= 2 * 10 ** 9
        if pool is not None:
            pool.classify(key, self.study.get("pool_class"))
            pool.want(key, priority, len(work), join=not floor)
        pool2, shared2 = None, None

        def join2():
            """The time-share pool, once the loop has booted the second server: a bench that started before it (round
            1 plays while the server boots; tscheck-2 served 0 tries) joins it at its next launch."""
            nonlocal pool2, shared2
            if pool2 is not None or pool is None or SHARED_SLOTS2 is None or not SERVER2_URL or self.server1_only:
                return
            pool2 = SHARED_SLOTS2
            if pool2.capacity is None:
                pool2.capacity, pool2.budget = capacity, budget
            pool2.gate = pool2.gate or KVGate(pool2.capacity, slots=max(cfg["concurrency"], int(self.study.get("generation_slots") or 0)),
                                              budget=pool2.budget)
            shared2 = httpx.AsyncClient(base_url=SERVER2_URL.rstrip("/") + "/", trust_env=False,
                                        transport=self.transport2 or TRANSPORT2,
                                        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128,
                                                            keepalive_expiry=KEEPALIVE_EXPIRY))
            pool2.classify(key, self.study.get("pool_class"))
            pool2.want(key, priority, len(work) - next_index, join=not floor)

        def take():
            if pool is None:
                return "none"
            join2()
            pools = balanced(pool, pool2) if placement == "balanced" else [x for x in (pool, pool2) if x is not None]
            if SERVER2_VERSIONS is not None and self.policy_version not in SERVER2_VERSIONS:
                pools = [x for x in pools if x is not pool2]
            for p_ in pools:
                if p_.take(key):
                    for q_ in (pool, pool2):  # both pools see the bench's true remaining count
                        if q_ is not None:
                            q_.remaining(key, len(work) - next_index - 1)
                    return p_
            return None

        try:
            while True:
                while (next_index < len(work) and len(running) < cfg["concurrency"]
                       and bench_wall.remaining() > cfg["launch_cutoff_seconds"]):
                    if kept and await reuse(order[next_index]):
                        next_index += 1
                        # a reused try is not pending: lower benches may go (v6h 24 Oct: tail-only, so 3 partly replayed
                        # groups kept phantom pending tries that outranked every new group and held group creation)
                        for q_ in (pool, pool2):
                            if q_ is not None:
                                q_.remaining(key, len(work) - next_index)
                        continue
                    chosen = take()
                    if chosen is None:
                        break
                    index = order[next_index]
                    next_index += 1
                    second = chosen is pool2 and pool2 is not None
                    resume = resume_of(index) if snaps else None
                    attempt = self.make_attempt(index, *work[index], gate=pool2.gate if second else gate, arc=arc, card=card,
                                                shared=shared2 if second else shared, agent_cfg=agent_cfg, window=window,
                                                resume=resume)
                    attempt["pool"] = chosen if chosen != "none" else None
                    attempt["server_index"] = 2 if second else 1
                    if second:
                        pool2.served += 1
                    # turn gate on either server's pool (v6k: server 1 at KV 0.89 with 73 senders; slots bound starts only)
                    attempt["slot"].turn_pool = pool2 if second else (chosen if chosen not in (None, "none") else None)
                    if attempt["pool"] is not None:
                        attempt["slot"].pool_cls = getattr(attempt["pool"], "classes", {}).get(key)  # class share
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
            if snap_task is not None:
                snap_task.cancel()
                await asyncio.gather(snap_task, return_exceptions=True)
                try:  # a stopped bench's unfinished tries: their last lines, for the relaunch's resume
                    await asyncio.to_thread(self.snap_all, [a for a in running.values() if a["agent"].bench_outcome is None])
                except Exception as exc:  # noqa: BLE001
                    self.log(json.dumps({"event": "level_bench_snap_failed", "error": str(exc)[:200]}), flush=True)
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
            if pool2 is not None:
                pool2.done(key)
            if shared2 is not None:
                await shared2.aclose()
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
        rows = [a["row"] for a in sorted(self.attempts + self.reused, key=lambda a: a["index"])]
        if self.reused:
            self.log(json.dumps({"event": "level_bench_reused", "tries": len(self.reused)}), flush=True)
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
            "clears_per_10k_tokens", "clears_per_10k_tokens_95", "median_tokens_to_clear", "mean_level_score",
            "tokens_per_second")}, "preemptions": engine.get("preemptions")}), flush=True)
        return report

    def make_attempt(self, index, pass_index, point, arm, *, gate, arc, card, shared, agent_cfg, window, resume=None):
        label = f"a{index:03d}-{point['id']}-p{pass_index}" + (f"-{arm['name']}" if self.cfg["arms"] else "")
        clock = GlobalClock(self.cfg["attempt_wall_seconds"])
        game = GameRecord(label, index, ActiveClock(clock, budget_clock=self.study.get("budget_clock", "inference")))
        game.ready = asyncio.Event()
        slot = GateSlot(gate, game, cap=self.cap, window=window, reserve=self.cfg["gate_completion_reserve"])
        entry = {"game_id": point["game"], "replica": 0, "instance_id": label}
        if not arm["thinking"]:
            agent_cfg = copy.copy(agent_cfg)
            agent_cfg.enable_thinking = False
        model = (arm.get("study") or {}).get("served_model")
        if model and model != agent_cfg.model:  # arm A/B over served models (base vs an adapter) in one bench
            agent_cfg = copy.copy(agent_cfg)
            agent_cfg.model = model
        client = SessionClient(agent_cfg, shared, slot, entry, self.requests)
        if self.study.get("transport_backoff"):  # RL benches: more replays of a generation request (19 Oct)
            client.client.backoff = tuple(float(x) for x in self.study["transport_backoff"])
        prefix_arc = PrefixArcade(arc, point, keep=bool(arm["history"]) and bool(point["prefix"]))
        agent = LevelBenchAgent(point["game"], prefix_arc, card, client, agent_cfg,
                                self.output / "attempts" / label, self.analysis_python,
                                time.monotonic() + self.cfg["attempt_wall_seconds"] + 3600)
        study = {**self.study, **(arm.get("study") or {})}
        agent.attach(game, window=window, cap=self.cap, memory=study.get("memory_block") or None,
                     escalation=study.get("model_escalation") or None, snapshot=study.get("game_snapshot") or None,
                     procedure=study.get("stuck_procedure") or None, critical_errors=bool(study.get("critical_errors")),
                     harness_v2=study.get("harness_v2") or None,
                     adaptive_thinking=study.get("adaptive_thinking") or None,
                     worldmodel_notice=bool(study.get("worldmodel_notice")),
                     harness_fixes=study.get("harness_fixes") or None,
                     parallel_samples=study.get("parallel_samples") or None,
                     action_plans=bool(study.get("action_plans")),
                     context_drain=study.get("drain_half") or None,
                     dfr_harness=study.get("dfr_harness") or None, dfr_loop=study.get("dfr_loop") or None,
                     hx_harness=study.get("hx_harness") or None, loop_rule=study.get("loop_rule") or None,
                     fresh_attempt=study.get("fresh_attempt") or None)
        agent.compaction_keep_observations = int(study.get("compaction_keep_observations", 1))
        agent.compaction_keep_fraction = study.get("compaction_keep_fraction")
        agent.compaction_summary_cap = study.get("compaction_summary_cap")
        agent.compaction_full_prefix = bool(study.get("compaction_full_prefix"))
        agent.compaction_thinking = study.get("compaction_thinking", DEFAULT_COMPACTION_THINKING)
        agent.trim = trim_settings(study.get("trim"))
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
                                   "check_only": bool(spec.get("check_only")), "hint": spec.get("hint"),
                                   "pin_actions": bool(spec.get("pin_actions"))}  # PTH (sft-heal); off by default
            agent.rollout.restore = agent.bench_restore
        if resume is not None:  # mid-try resume: the try's own snapshot replaces its start (which the copy begins with)
            from .restore import load as load_restore

            if getattr(agent, "rollout", None) is None:
                raise ValueError("level bench: a resumed attempt needs the rollout record")
            key = ("resume", resume["folder"], int(resume["request"]))
            agent.bench_restore = {**load_restore(resume["folder"], int(resume["request"])),
                                   "tokens": int(resume["base_tokens"]), "check_only": False, "hint": None,
                                   "resume": True}
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
        seed_base = arm["seed_base"] if arm.get("seed_base") is not None else self.cfg["seed_base"]
        if seed_base is not None:  # an arm's own seed_base (RL: seeded eval attempts inside a training bench)
            agent.sampling_override["seed"] = int(seed_base) + int(pass_index)
        agent.bench_setup(start_level=point["level_index"],
                          budget=int((point.get("restore") or {}).get("budget") or point.get("budget")
                                     or token_budget(self.cfg, point)),
                          warm=point.get("warm") if arm["warm"] else None,
                          trace=prefix_arc if prefix_arc.trace is not None else None,
                          reset=arm["reset"], level_start=point.get("level_start_actions"), enter=arm.get("enter"),
                          through=bool(point.get("through")), script=point.get("script"),
                          record=study.get("trim_record"),
                          prefix=len(point["prefix"]), lead=point.get("lead_levels", 0),
                          lead_budget=self.cfg["lead_token_budget"], action_cap=action_cap(self.cfg, point),
                          level_cap=(point.get("level_caps") or self.cfg["level_token_cap"]) if point.get("through") else None)
        return {"index": index, "pass": pass_index, "point": point, "arm": arm["name"], "label": label, "game": game,
                "clock": clock, "agent": agent, "arcade": prefix_arc, "censored": False, "slot": slot, "resume": resume}

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
            earlier = (attempt.get("resume") or {}).get("policy_version")
            if earlier is not None:  # a resumed try: its oldest replies' version (staleness, group_reused)
                attempt["row"]["policy_version"] = min(int(earlier), int(attempt["row"].get("policy_version", earlier)))
            if not attempt["censored"]:  # live play counters (the RL loop's play_progress rows)
                PROGRESS["done"] += 1
                PROGRESS["clears"] += bool(attempt["row"].get("cleared"))
                if PROGRESS_HOOK is not None:
                    try:
                        PROGRESS_HOOK(self.output.name, attempt["row"])
                    except Exception as exc:  # noqa: BLE001 - live stats never fail a try
                        self.log(json.dumps({"event": "progress_hook_failed", "error": str(exc)[:200]}))
            if self.try_keep is not None and not attempt["censored"] and attempt["row"]["outcome"] not in ("error", "censored"):
                try:
                    self.keep_try(attempt)
                except OSError as exc:  # a failed copy costs only the reuse
                    self.log(json.dumps({"event": "level_bench_keep_failed", "label": attempt["label"], "error": str(exc)[:200]}))
            if not attempt["censored"]:
                BENCH_DONE[self.output.name] = BENCH_DONE.get(self.output.name, 0) + 1
            if self.snap_s > 0 and not attempt["censored"]:
                self.snap_drop(attempt)
            if not attempt["censored"]:
                self.log(json.dumps({"event": "level_bench_attempt", **{k: attempt["row"][k] for k in (
                    "label", "outcome", "completion_tokens", "tokens_to_clear", "actions", "level_score",
                    "gameplay_requests", "wall_seconds")}}), flush=True)
                try:  # per finished try, so a stop or crash keeps what was played (level-bench.json comes at the end)
                    with open(self.output / "level-bench.partial.jsonl", "a") as file:
                        file.write(json.dumps(attempt["row"], default=str) + "\n")
                except Exception as exc:  # noqa: BLE001
                    self.log(json.dumps({"event": "level_bench_partial_failed", "error": str(exc)[:200]}), flush=True)

    def snap_dest(self, attempt):
        return Path(self.try_keep) / self.output.name / "live" / attempt["label"]

    def snap_all(self, attempts):
        """Each running try's new whole lines to its live snapshot, then its meta (thread: off the event loop)."""
        for attempt in attempts:
            with self.snap_lock:
                if attempt.get("snap_closed") or getattr(attempt["agent"], "rollout", None) is None:
                    continue
                dest = self.snap_dest(attempt)
                lines = snap_copy(attempt["agent"].bench_dir(), dest, attempt.setdefault("snap", {}))
                if not lines.get("rollout.jsonl"):
                    continue
                point, label = attempt["point"], attempt["label"]
                earlier = (attempt.get("resume") or {}).get("policy_version")
                version = self.policy_version if earlier is None else (
                    int(earlier) if self.policy_version is None else min(int(earlier), int(self.policy_version)))
                meta = {"label": label, "plan": plan_digest(label, point, token_budget(self.cfg, point), self.cap),
                        "policy_version": version, "rows": lines["rollout.jsonl"], "log_lines": lines["log.jsonl"],
                        "epoch": time.time()}
                (dest / "meta.json.tmp").write_text(json.dumps(meta))
                (dest / "meta.json.tmp").replace(dest / "meta.json")

    def snap_drop(self, attempt):
        """A finished try's snapshot is removed (a relaunch reuses its kept copy, or plays it again)."""
        import shutil

        with self.snap_lock:
            attempt["snap_closed"] = True
            shutil.rmtree(self.snap_dest(attempt), ignore_errors=True)

    def keep_try(self, attempt):
        """The finished try's folder, then its row (the row marks the copy complete)."""
        import shutil

        point, label = attempt["point"], attempt["label"]
        dest = Path(self.try_keep) / self.output.name
        shutil.copytree(self.output / "attempts" / label, dest / "attempts" / label, dirs_exist_ok=True)
        row = {**attempt["row"], "plan": plan_digest(label, point, token_budget(self.cfg, point), self.cap)}
        with (dest / "tries.jsonl").open("a") as file:
            file.write(json.dumps(row, default=str) + "\n")

    def row(self, attempt, result, error, seconds):
        agent, point, game = attempt["agent"], attempt["point"], attempt["game"]
        result = result or {}
        error = error or result.get("error")
        replay = attempt["arcade"].replay
        if attempt["censored"]:
            outcome = "censored"
        elif agent.bench_outcome in ("cleared", "budget", "level_cap", "action_cap", "lead_out", "won", "recorded",
                                     "replay_mismatch", "restored"):
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
        enter = agent.bench_enter or {"completion_tokens": 0, "steps": agent.bench_offset}
        if agent.bench_script is not None and agent.bench_script_end is not None:
            enter = agent.bench_script_end  # tokens and actions count from the first live request
        if outcome == "lead_out":
            enter = {"completion_tokens": game.clock.completion_tokens(), "steps": agent.steps}
        restore = agent.bench_restore
        if restore is not None:  # tokens from the first live request, actions from the level's first board
            enter = {"completion_tokens": restore["tokens"], "steps": restore["level_start_step"]}
        actions = clear.get("actions", agent.steps - enter["steps"])
        requests, noaction = request_actions(agent.workspace.log)
        harvested = agent.bench_harvest_row
        if harvested is not None:  # the harvest's extra request is not part of the try
            requests, noaction = requests - 1, max(0, noaction - 1)
        through = None
        if restore is not None:
            # A restored segment: the one level it plays (credit as a play-through level), nothing before it
            level = restore["level"]
            base = point.get("baselines") or []
            rows_ = []
            if cleared:
                credit = level_score(base[level - 1] if level <= len(base) else None, clear["actions"], True)
                rows_ = [{"level": level, "tokens": clear["completion_tokens"], "at_tokens": clear["completion_tokens"],
                          "actions": clear["actions"], "score": round(credit, 3)}]
            total = int(point.get("total_levels") or len(base) or 1)
            through = {"levels": len(rows_), "total_levels": total, "level_rows": rows_,
                       "game_score": round(sum(min(100.0, r["score"]) for r in rows_) / total, 3), "enter_applied": []}
        elif point.get("segment") and not point.get("through"):
            # A fresh level-1 segment (RL multi-game, 16 Oct): the one level it plays, credited as a restored segment
            level = int(point.get("level") or 1)
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
                       "enter_applied": list(agent.bench_enter_applied or [])}
        row = self.row_fields(attempt, result, error, seconds, outcome=outcome, cleared=cleared, clear=clear,
                              enter=enter, actions=actions, requests=requests, noaction=noaction, replay=replay)
        if through is not None:
            row.update(through, level_score=through["game_score"])
        elif agent.bench_enter_mode:
            row["enter_applied"] = list(agent.bench_enter_applied or [])
        if agent.sc_on():
            row["sim_check"] = dict(agent.sc_stats)
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
                              "state": state, "ok": bool(state.get("ok")) and not result.get("diffs")}
            if restore.get("resume"):  # lineage stays the group's start; the resume check is the row's restore check
                origin = point.get("restore")
                row["resumed"] = {"snapshot": restore["source"], "request": restore["request"],
                                  "ok": row["restore"]["ok"], "diff_count": row["restore"]["diff_count"]}
                if origin:
                    row["restore"].update(source=origin["attempt"], request=int(origin["request"]))
                else:
                    del row["restore"]
        if harvested is not None:
            row["harvest"] = dict(harvested)
            row["completion_tokens"] -= int(harvested.get("completion_tokens") or 0)
        if agent.bench_script is not None:
            row["script"] = dict(agent.bench_script_end or {"ok": False, "detail": "script not finished",
                                                            "requests": agent.bench_script_at})
        return row

    def row_fields(self, attempt, result, error, seconds, *, outcome, cleared, clear, enter, actions, requests,
                   noaction, replay):
        agent, point, game = attempt["agent"], attempt["point"], attempt["game"]
        return {
            "label": attempt["label"], "index": attempt["index"], "pass": attempt["pass"], "point": point["id"],
            "arm": attempt["arm"], "server_index": attempt.get("server_index", 1),
            "game": point["game"], "split": point["split"], "level": point["level"],
            "level_index": point["level_index"], "outcome": outcome, "cleared": cleared,
            "completion_tokens": game.clock.completion_tokens() - enter["completion_tokens"],
            "lead_tokens": enter["completion_tokens"] if point.get("lead_levels") else None,
            "lead_actions": enter["steps"] - agent.bench_offset if point.get("lead_levels") else None,
            "tokens_to_clear": clear.get("completion_tokens") if cleared else None,
            "actions": actions, "total_actions": agent.steps - agent.bench_offset,
            "baseline_actions": point.get("baseline_actions"), "action_cap": agent.bench_action_cap,
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


async def run_level_bench(study, trial, **kwargs):
    return await LevelBench(study, trial, **kwargs).run()
