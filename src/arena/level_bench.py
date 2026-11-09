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
import json
import math
import random
import statistics
import time
from pathlib import Path

import httpx

from arc_infra.arcade import make_arcade
from arc_infra.fleet import SessionClient
from duck_transfer.agent import GameplayRequestExpired

from .clock import ActiveClock, GlobalClock
from .compaction import DEFAULT_COMPACTION_THINKING
from .duck import ArenaTransferDuckAgent
from .gate import GateSlot, KVGate
from .level_points import check_replay, level_score
from .scheduler import GameRecord

DEFAULTS = {
    "split": "dev",                 # dev | holdout | all
    "games": None,                  # optional list of 4-character game prefixes
    "concurrency": 16,              # attempts running at once (and gate slots)
    "window": 49152,                # agent context per attempt (compaction at window - cap)
    "token_budget": 40000,          # completion tokens per attempt (every purpose)
    "attempt_wall_seconds": 1800,   # safety cap per attempt
    "passes": 3,                    # seeded permutations of the start points, in order
    "order_seed": 0,
    "progress_seconds": 30,         # a live progress line at least this often (at most 60)
    "launch_cutoff_seconds": 120,   # no new attempt when less bench wall remains
    "min_wall_seconds": 600,        # the bench refuses to start with less
    "start_points": "configs/level_bench_start_points.json",
    "capacity_tokens": None,        # external endpoint: gate capacity in tokens (no server log)
    "serving": None,                # recorded label of the serving profile (e.g. "mtp0")
    "arms": None,                   # [{"name", "warm": bool, "history": bool, "study": {...}}]; None: one cold arm
    "first_pass": 0,                # index of the first pass (a later run extends an earlier one's passes)
    "lead_token_budget": 80000,     # lead points: tokens to get through the lead-in level(s), else "lead_out"
    "reader_probe": None,           # rsi arm: {"file", "variants": [{"name", "thinking", "effort", "cap"}], "repeats"}:
                                    # recorded second-reader prompts answered before the attempts (reader-probe.json)
}
ARM_KEYS = ("name", "warm", "history", "reset", "study", "points", "games", "enter", "passes")  # points: start | deep | lead |
# None; games: None or a list of 4-character game prefixes the arm is limited to; enter (lead points): what the harness
# takes away at the target level's first board, before the agent's first request on it (``ENTER``)
ENTER = (None, "clear", "restart", "both")  # clear: the conversation and the carried world model (python session
# kept); restart: the python session (interpreter restarted: the agent's variables, functions and registered tools are
# gone; the conversation stays); both: both. The agent's note() entries (hx memo) are a third store: every arm keeps
# them (a restarted interpreter starts with them). A play-through point applies the removal at every level entry.
# passes: the arm runs in the first ``passes`` passes of the run only (a small sentinel arm); None: in all.
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
HEADLINE_EXCLUDED = ("censored", "error", "replay_mismatch", "lead_out")


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
    if merged["games"] is not None and (not isinstance(merged["games"], list) or not merged["games"]):
        raise ValueError("level_bench.games must be a non-empty list of game prefixes")
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


COLD = {"name": "base", "warm": False, "history": False, "reset": False, "study": {}, "points": None, "games": None,
        "enter": None, "passes": None}


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
                if arm["points"] in (None, point_kind(point)) and (
                        not arm["games"] or str(point.get("game", ""))[:4] in arm["games"]) and (
                        arm["passes"] is None or index - first_pass < arm["passes"]):
                    out.append((index, point, arm))
    return out


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
    bench_outcome = None  # cleared | budget | censored
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
    bench_levels = None         # play-through: [{"level", "completion_tokens", "steps"}] per cleared level

    def bench_setup(self, *, start_level, budget, warm=None, trace=None, reset=False, level_start=None, prefix=0,
                    lead=0, lead_budget=None, enter=None, through=False):
        self.bench_start, self.bench_budget = int(start_level), int(budget)
        self.bench_lead, self.bench_lead_budget, self.bench_enter = int(lead or 0), lead_budget, None
        self.bench_through, self.bench_levels = bool(through), []
        if through and (self.bench_lead or prefix or start_level):
            raise ValueError("level bench: a play-through point is a fresh game")
        if enter and (not (self.bench_lead or through) or self.drain is None):
            raise ValueError("level bench: an enter arm needs a lead or play-through point and drain_half")
        self.bench_enter_mode, self.bench_enter_applied, self.bench_enter_note = enter, [], None
        self.bench_enter_due, self.bench_bootstrap = False, None
        self.bench_outcome = self.bench_clear = None
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
            self.bench_trace.trace = None  # the raw observations are no longer needed
            if labels[-1]:
                return super().write_duck_state(labels[-1])
        return super().write_duck_state(action, result)

    def remaining(self):
        if self.bench_outcome is not None:
            return 0.0
        return super().remaining()

    def bench_gate(self):
        """Before every request: an attempt past its budget (or finished) sends nothing more."""
        tokens = self.game.clock.completion_tokens()
        if self.bench_outcome is None and self.bench_lead and self.bench_enter is None:
            if tokens >= self.bench_lead_budget:
                self.bench_outcome = "lead_out"
                self.workspace.append("level_bench_lead_out", step=self.steps, completion_tokens=tokens)
        elif self.bench_outcome is None and tokens - (self.bench_enter or {}).get("completion_tokens", 0) >= self.bench_budget:
            self.bench_outcome = "budget"
            self.workspace.append("level_bench_budget", step=self.steps, completion_tokens=tokens)
        if self.bench_outcome is not None:
            raise GameplayRequestExpired(f"Level-bench attempt finished: {self.bench_outcome}")

    async def chat(self, messages, tools):
        self.bench_gate()
        return await super().chat(messages, tools)

    async def aside_request(self, *args, **kwargs):
        self.bench_gate()
        return await super().aside_request(*args, **kwargs)

    async def step(self, action):
        await super().step(action)
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
        self.cap = int(study["completion_cap"])
        self.attempts, self.requests = [], []
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
        gate = KVGate(capacity, slots=cfg["concurrency"], log=self.gate_log, budget=budget)
        arc = self.arc or make_arcade("offline", self.games_dir, self.output / "recordings")
        card = arc.open_scorecard(tags=["level-bench", self.trial["name"]])
        shared = httpx.AsyncClient(base_url=self.base_url.rstrip("/") + "/", trust_env=False, transport=self.transport,
                                   limits=httpx.Limits(max_connections=256, max_keepalive_connections=128))
        monitor = GpuMonitor(self.output, self.base_url)
        if cfg["reader_probe"]:
            await self.reader_probe(shared, agent_cfg)
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
        try:
            while True:
                while (next_index < len(work) and len(running) < cfg["concurrency"]
                       and bench_wall.remaining() > cfg["launch_cutoff_seconds"]):
                    index = next_index
                    next_index += 1
                    attempt = self.make_attempt(index, *work[index], gate=gate, arc=arc, card=card,
                                                shared=shared, agent_cfg=agent_cfg, window=window)
                    running[asyncio.create_task(self.play(attempt))] = attempt
                if not running:
                    break
                done, _ = await asyncio.wait(list(running), timeout=min(1.0, max(0.05, bench_wall.remaining())),
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    running.pop(task)
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
        (self.output / "level-bench.json").write_text(json.dumps(report, indent=1))
        self.log(json.dumps({"event": "level_bench_result", **{k: summary[k] for k in (
            "attempts_completed", "attempts_censored", "attempts_failed", "clears", "clear_rate",
            "clears_per_10k_tokens", "clears_per_10k_tokens_95", "median_tokens_to_clear", "mean_level_score",
            "tokens_per_second")}, "preemptions": engine.get("preemptions")}), flush=True)
        return report

    async def reader_probe(self, shared, agent_cfg):
        """rsi arm, offline check of the second reader: every recorded prompt (``messages``: system + user) is
        answered ``repeats`` times per variant, eight at a time, outside the gate (nothing else is running)."""
        from duck_transfer import loader

        from .compaction import summary_payload, summary_text

        spec, module = self.cfg["reader_probe"], loader.load()
        cases = [json.loads(line) for line in (self.source / spec["file"]).read_text().splitlines() if line.strip()]
        limit, rows = asyncio.Semaphore(8), []

        async def one(case, variant, repeat):
            payload = summary_payload(
                build_chat_payload=module.build_chat_payload, model=agent_cfg.model, messages=case["messages"][:-1],
                prompt=case["messages"][-1]["content"], cap=int(variant["cap"]), temperature=agent_cfg.temperature,
                top_p=module._LOCAL_ANALYZER_TOP_P, top_k=module._LOCAL_ANALYZER_TOP_K,
                thinking=bool(variant.get("thinking")), reasoning_effort=variant.get("effort"))
            row = {"id": case["id"], "variant": variant["name"], "repeat": repeat}
            async with limit:
                started = time.monotonic()
                try:
                    response = await shared.post("chat/completions", json=payload, timeout=900)
                    response.raise_for_status()
                    data = response.json()
                    choice = data["choices"][0]
                    row.update(answer=summary_text(choice["message"]), finish_reason=choice.get("finish_reason"),
                               usage=data.get("usage"), reasoning=str(choice["message"].get("reasoning_content")
                                                                      or choice["message"].get("reasoning") or ""))
                except Exception as exc:  # noqa: BLE001 - a failed probe is a row, not a failed bench
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                row["seconds"] = round(time.monotonic() - started, 2)
            rows.append(row)

        await asyncio.gather(*(one(case, variant, repeat) for case in cases for variant in spec["variants"]
                               for repeat in range(int(spec.get("repeats", 1)))))
        rows.sort(key=lambda r: (r["id"], r["variant"], r["repeat"]))
        (self.output / "reader-probe.json").write_text(json.dumps(rows, indent=1))
        self.log(json.dumps({"event": "reader_probe", "cases": len(cases), "rows": len(rows),
                             "errors": sum(1 for r in rows if r.get("error")),
                             "empty": sum(1 for r in rows if not r.get("answer"))}), flush=True)

    def make_attempt(self, index, pass_index, point, arm, *, gate, arc, card, shared, agent_cfg, window):
        label = f"a{index:03d}-{point['id']}-p{pass_index}" + (f"-{arm['name']}" if self.cfg["arms"] else "")
        clock = GlobalClock(self.cfg["attempt_wall_seconds"])
        game = GameRecord(label, index, ActiveClock(clock, budget_clock=self.study.get("budget_clock", "inference")))
        game.ready = asyncio.Event()
        slot = GateSlot(gate, game, cap=self.cap, window=window)
        entry = {"game_id": point["game"], "replica": 0, "instance_id": label}
        client = SessionClient(agent_cfg, shared, slot, entry, self.requests)
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
                     fresh_attempt=study.get("fresh_attempt") or None, rsi_loop=study.get("rsi_loop") or None)
        agent.compaction_keep_observations = int(study.get("compaction_keep_observations", 1))
        agent.compaction_thinking = study.get("compaction_thinking", DEFAULT_COMPACTION_THINKING)
        agent.bench_setup(start_level=point["level_index"], budget=self.cfg["token_budget"],
                          warm=point.get("warm") if arm["warm"] else None,
                          trace=prefix_arc if prefix_arc.trace is not None else None,
                          reset=arm["reset"], level_start=point.get("level_start_actions"), enter=arm.get("enter"),
                          through=bool(point.get("through")),
                          prefix=len(point["prefix"]), lead=point.get("lead_levels", 0),
                          lead_budget=self.cfg["lead_token_budget"])
        return {"index": index, "pass": pass_index, "point": point, "arm": arm["name"], "label": label, "game": game,
                "clock": clock, "agent": agent, "arcade": prefix_arc, "censored": False}

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
            if not attempt["censored"]:
                self.log(json.dumps({"event": "level_bench_attempt", **{k: attempt["row"][k] for k in (
                    "label", "outcome", "completion_tokens", "tokens_to_clear", "actions", "level_score",
                    "gameplay_requests", "wall_seconds")}}), flush=True)

    def row(self, attempt, result, error, seconds):
        agent, point, game = attempt["agent"], attempt["point"], attempt["game"]
        result = result or {}
        error = error or result.get("error")
        replay = attempt["arcade"].replay
        if attempt["censored"]:
            outcome = "censored"
        elif agent.bench_outcome in ("cleared", "budget", "lead_out", "won"):
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
        if outcome == "lead_out":
            enter = {"completion_tokens": game.clock.completion_tokens(), "steps": agent.steps}
        actions = clear.get("actions", agent.steps - enter["steps"])
        requests, noaction = request_actions(agent.workspace.log)
        through = None
        if point.get("through"):
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
        if agent.rsi is not None:
            row["rsi_loop"] = agent.rsi_report()
        return row

    def row_fields(self, attempt, result, error, seconds, *, outcome, cleared, clear, enter, actions, requests,
                   noaction, replay):
        agent, point, game = attempt["agent"], attempt["point"], attempt["game"]
        return {
            "label": attempt["label"], "index": attempt["index"], "pass": attempt["pass"], "point": point["id"],
            "arm": attempt["arm"],
            "game": point["game"], "split": point["split"], "level": point["level"],
            "level_index": point["level_index"], "outcome": outcome, "cleared": cleared,
            "completion_tokens": game.clock.completion_tokens() - enter["completion_tokens"],
            "lead_tokens": enter["completion_tokens"] if point.get("lead_levels") else None,
            "lead_actions": enter["steps"] - agent.bench_offset if point.get("lead_levels") else None,
            "tokens_to_clear": clear.get("completion_tokens") if cleared else None,
            "actions": actions, "total_actions": agent.steps - agent.bench_offset,
            "baseline_actions": point.get("baseline_actions"),
            "level_score": round(level_score(point.get("baseline_actions"), actions, cleared), 3),
            "gameplay_requests": requests, "noaction_requests": noaction, "admitted_requests": len(game.requests),
            "prompt_tokens": agent.prompt_tokens, "compactions": game.compactions,
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
