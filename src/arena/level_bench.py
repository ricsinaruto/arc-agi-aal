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
}
SPLITS = ("dev", "holdout", "all")
UNSUPPORTED = ("cross_game_notes", "explorer", "background_lane", "commit", "llm_backlog", "audition_pool")
HEADLINE_EXCLUDED = ("censored", "error", "replay_mismatch")


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


def plan(points, passes, seed):
    """``[(pass, point)]``: each pass a seeded permutation of the points (sorted by id first)."""
    out = []
    for index in range(passes):
        order = list(points)
        random.Random(f"{seed}:{index}").shuffle(order)
        out.extend((index, point) for point in order)
    return out


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
    bench_outcome = None  # cleared | budget | censored
    bench_clear = None

    def bench_setup(self, *, start_level, budget):
        self.bench_start, self.bench_budget = int(start_level), int(budget)
        self.bench_outcome = self.bench_clear = None
        return self

    def remaining(self):
        if self.bench_outcome is not None:
            return 0.0
        return super().remaining()

    def bench_gate(self):
        """Before every request: an attempt past its budget (or finished) sends nothing more."""
        if self.bench_outcome is None and self.game.clock.completion_tokens() >= self.bench_budget:
            self.bench_outcome = "budget"
            self.workspace.append("level_bench_budget", step=self.steps,
                                  completion_tokens=self.game.clock.completion_tokens())
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
        if self.bench_outcome is None and self.obs.levels > self.bench_start:
            self.bench_outcome = "cleared"
            self.bench_clear = {"actions": self.steps, "completion_tokens": self.game.clock.completion_tokens(),
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
        work = plan(points, cfg["passes"], cfg["order_seed"])
        bench_wall = GlobalClock(self.wall_seconds())
        agent_cfg = duck_config(self.study, self.source, window=window, cap=self.cap,
                                wall=cfg["attempt_wall_seconds"], base_url=self.base_url)
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
        header = {"event": "level_bench_start", "trial": self.trial["name"], "points": len(points),
                  "planned_attempts": len(work), "split": cfg["split"], "window": window, "cap": self.cap,
                  "token_budget": cfg["token_budget"], "concurrency": cfg["concurrency"],
                  "gate_capacity": capacity, "gate": choice, "wall_seconds": round(bench_wall.seconds, 1),
                  "serving": cfg["serving"], "server": self.study.get("server"),
                  "reasoning_effort": self.study.get("reasoning_effort")}
        self.log(json.dumps(header), flush=True)
        (self.output / "level-bench-plan.json").write_text(json.dumps(
            {**header, "plan": [{"pass": p, "point": pt["id"]} for p, pt in work]}, indent=1))
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

    def make_attempt(self, index, pass_index, point, *, gate, arc, card, shared, agent_cfg, window):
        label = f"a{index:03d}-{point['id']}-p{pass_index}"
        clock = GlobalClock(self.cfg["attempt_wall_seconds"])
        game = GameRecord(label, index, ActiveClock(clock, budget_clock="inference"))
        game.ready = asyncio.Event()
        slot = GateSlot(gate, game, cap=self.cap, window=window)
        entry = {"game_id": point["game"], "replica": 0, "instance_id": label}
        client = SessionClient(agent_cfg, shared, slot, entry, self.requests)
        prefix_arc = PrefixArcade(arc, point)
        agent = LevelBenchAgent(point["game"], prefix_arc, card, client, agent_cfg,
                                self.output / "attempts" / label, self.analysis_python,
                                time.monotonic() + self.cfg["attempt_wall_seconds"] + 3600)
        study = self.study
        agent.attach(game, window=window, cap=self.cap, memory=study.get("memory_block") or None,
                     escalation=study.get("model_escalation") or None, snapshot=study.get("game_snapshot") or None,
                     procedure=study.get("stuck_procedure") or None, critical_errors=bool(study.get("critical_errors")),
                     harness_v2=study.get("harness_v2") or None,
                     adaptive_thinking=study.get("adaptive_thinking") or None,
                     worldmodel_notice=bool(study.get("worldmodel_notice")),
                     harness_fixes=study.get("harness_fixes") or None,
                     dfr_harness=study.get("dfr_harness") or None)
        agent.compaction_keep_observations = int(study.get("compaction_keep_observations", 1))
        agent.compaction_thinking = study.get("compaction_thinking", DEFAULT_COMPACTION_THINKING)
        agent.bench_setup(start_level=point["level_index"], budget=self.cfg["token_budget"])
        return {"index": index, "pass": pass_index, "point": point, "label": label, "game": game, "clock": clock,
                "agent": agent, "arcade": prefix_arc, "censored": False}

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
        elif agent.bench_outcome in ("cleared", "budget"):
            outcome = agent.bench_outcome
        elif replay is not None and not replay["ok"]:
            outcome = "replay_mismatch"
        elif error:
            outcome = "error"
        elif attempt["clock"].remaining() <= 0:
            outcome = "wall"
        else:
            outcome = result.get("status") or "stopped"
        cleared = outcome == "cleared"
        clear = agent.bench_clear or {}
        actions = clear.get("actions", agent.steps)
        requests, noaction = request_actions(agent.workspace.log)
        return {
            "label": attempt["label"], "index": attempt["index"], "pass": attempt["pass"], "point": point["id"],
            "game": point["game"], "split": point["split"], "level": point["level"],
            "level_index": point["level_index"], "outcome": outcome, "cleared": cleared,
            "completion_tokens": game.clock.completion_tokens(),
            "tokens_to_clear": clear.get("completion_tokens") if cleared else None,
            "actions": actions, "total_actions": agent.steps, "baseline_actions": point.get("baseline_actions"),
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
