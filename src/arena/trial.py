"""One arena trial: 25 games, one wall clock, the scheduler, one agent kind."""

import asyncio
import json
import time
from pathlib import Path

import httpx

from arc_infra import KEEPALIVE_EXPIRY
from arc_infra.arcade import make_arcade
from arc_infra.config import Config as DuckConfig
from arc_infra.fleet import SessionClient
from arc_infra.metrics import GpuMonitor

from .cache import BlockBudget
from .capacity import (
    choose_window,
    competition_focus_fit,
    focus_fit,
    kv_pool_tokens,
    phase2_capacity,
)
from .clock import ActiveClock, GlobalClock
from .compaction import DEFAULT_COMPACTION_THINKING
from .early_stop import EarlyStop, end_trial, stop_paths
from .early_stop import settings as early_stop_settings
from .gate import BackgroundLane, GateSlot, KVGate
from .live_progress import LiveProgress, serving_check
from .live_progress import settings as live_settings
from .report import game_row, trial_report
from .scheduler import GameRecord, Scheduler
from .trim_bed import settings as trim_settings

KINDS = ("duck-transfer", "duck-public")
READY_TIMEOUT = 600


def trial_settings(study, *, smoke=False):
    """Games, wall seconds and the smoke overrides."""
    games = list(study["games"])
    wall = float(study["wall_seconds"])
    if smoke:
        games = games[: int(study["smoke"]["games"])]
        wall = float(study["smoke"]["wall_seconds"])
    return games, wall


def duck_config(study, source, *, window, cap, wall, base_url):
    cfg = DuckConfig.load(str(Path(source) / "configs/duck_controls.yaml"))
    cfg.max_model_len = int(window)
    cfg.max_output_tokens = int(cap)
    cfg.reasoning_effort = study["reasoning_effort"]
    cfg.enable_thinking = bool(study.get("enable_thinking", True))
    # Old bundles carry no key: the public loop's normalized history (byte-identical prompts).
    cfg.reasoning_history = study.get("reasoning_history", "normalized")
    cfg.temperature = float(study["temperature"])
    cfg.gameplay_seconds = wall
    cfg.parallel_games = int(study["generation_slots"])
    cfg.max_num_seqs = max(cfg.max_num_seqs, cfg.parallel_games)
    cfg.base_url = base_url
    cfg.serving_backend = study.get("backend", "vllm")
    cfg.model = study.get("served_model", cfg.model)
    cfg.model_revision = study.get("model_revision", cfg.model_revision)
    # Client-side lookahead: the MTP depth, or the n-gram draft length when it replaces MTP.
    cfg.speculative_tokens = (study['server'].get('speculative') or {}).get(
        'num_speculative_tokens', study['server'].get('mtp_tokens', cfg.speculative_tokens))
    if study.get('competition'):
        cfg.max_actions = study['competition']['max_actions_per_game']
    return cfg


class Trial:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir,
                 analysis_python=None, base_url, smoke=False, log=print,
                 arcade=None, gameplay_deadline_epoch=None):
        self.study, self.trial = study, trial
        self.kind = trial["kind"]
        if self.kind not in KINDS:
            raise ValueError(f"Unknown trial kind: {self.kind}")
        self.source, self.games_dir = Path(source), games_dir
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.server_dir = server_dir
        self.analysis_python = analysis_python or "python3"
        self.base_url, self.smoke, self.log = base_url, smoke, log
        self.arcade = arcade
        self.gameplay_deadline_epoch = gameplay_deadline_epoch
        self.cap = int(study["completion_cap"])
        self.games_list, self.wall_seconds = trial_settings(study, smoke=smoke)
        self.admission_log = (self.output / "admission.jsonl").open("a")
        from .heap import FREEZE_SECONDS, HeapKeeper

        self.heap = HeapKeeper(study.get("heap_freeze_seconds", FREEZE_SECONDS))

    def gate_log(self, entry):
        clock = getattr(self, "wall", None)
        self.admission_log.write(json.dumps({
            "epoch": time.time(),
            "wall_seconds": clock.elapsed() if clock is not None else None,
            **entry,
        }) + "\n")

    async def run(self):
        study = self.study
        block_budget = None
        if study.get("backend") == "sglang":
            from .sglang import TokenBudget

            block_budget = TokenBudget.load(self.server_dir, study)
            window = int(study["windows"][0])
            if block_budget.cost(window) > block_budget.capacity:
                raise RuntimeError("Configured context cannot fit SGLang; no silent downgrade")
            choice = {"window": window, "capacity": block_budget.capacity, "units": "tokens",
                      "rule": block_budget.layout["rule"], "margin": study["kv_margin"]}
        elif study.get("backend") == "llamacpp":
            # llama-server with fixed per-slot windows: a plain slot gate, no KV-pool estimate.
            from .llamacpp import SlotCapacity

            choice = SlotCapacity.load(self.server_dir, study).choice()
            window = choice["window"]
        elif study.get("cache_admission") == "hybrid_blocks":
            if self.kind != "duck-transfer":
                raise ValueError("Block admission supports Duck-transfer only")
            block_budget = BlockBudget.load(self.server_dir, float(study["kv_margin"]))
            window = int(study["windows"][0])  # preserve context/compaction policy
            if block_budget.cost(window) > block_budget.capacity:
                raise RuntimeError("Configured context cannot fit; no silent context downgrade")
            choice = {"window": window, "capacity": block_budget.capacity, "units": "cache_blocks",
                      "rule": block_budget.layout["rule"], "margin": study["kv_margin"]}
            (self.output / "cache-layout.json").write_text(json.dumps(block_budget.layout, indent=2))
            if study.get("deep_explore") is not None:
                # fixed-2x-deep-explore: mixed windows; every deep and exploration window must fit at once.
                from .capacity import deep_explore_fit
                from .deep_explore import settings as deep_settings

                deep = deep_settings(study["deep_explore"])
                if window != deep["deep_window"] or study.get("active_limit") != deep["deep_slots"] + deep["explore_slots"]:
                    raise ValueError("deep_explore needs the deep window as the study window and deep + explore slots "
                                     "as the active limit")
                choice["deep_explore_fit"] = deep_explore_fit(block_budget, deep, required=True)
            elif isinstance(study.get("active_limit"), int):
                if study.get("depthwise") is not None:
                    # fixed-2x-depthwise: every slot may hold a depth game at the full window, so the six full
                    # windows must fit (require_focus_windows_fit); level-0 turns at the smaller window cost less.
                    from .depthwise import settings as depthwise_settings

                    depthwise = depthwise_settings(study["depthwise"])
                    if (window != depthwise["deep_window"] or study["active_limit"] != depthwise["slots"]
                            or not study.get("require_focus_windows_fit")):
                        raise ValueError("depthwise needs the depth window as the study window, its slots as the "
                                         "active limit and require_focus_windows_fit")
                    choice["depthwise_level0"] = focus_fit(block_budget, depthwise["level0_window"],
                                                           depthwise["slots"], required=False)
                if study.get("rank") is not None:
                    # hero-v2-rank: the ranking's slots are the focus games at the study window.
                    from .rank import settings as rank_settings

                    rank = rank_settings(study["rank"])
                    if window != rank["window"] or study["active_limit"] != rank["slots"]:
                        raise ValueError("rank needs its window as the study window and its slots as the active limit")
                # Recorded for every block-admitted trial; enforced when the study requires it.
                # Competition builds run with what fits instead of refusing (competition_focus_fit).
                choice["focus_fit"] = competition_focus_fit(block_budget, window, study)
            elif study.get("require_focus_windows_fit"):
                raise ValueError("require_focus_windows_fit needs a fixed integer active_limit")
        else:
            pool = kv_pool_tokens(self.server_dir)
            choice = choose_window(
                pool, active_limit=int(study["active_limit"]), margin=float(study["kv_margin"]),
                windows=tuple(study["windows"]),
            )
            window = choice["window"]
        phase2 = phase2_capacity(study, block_budget, window, len(self.games_list))
        choice['phase2'] = phase2
        if study.get("blocks") is not None:
            # hero-v2-blocks: a block is the stream count, so the served fit must keep every stream (no fallback to
            # fewer), and the scheduler's wall is the study's (the run may stop earlier: early_stop ``always``).
            from .blocks import settings as blocks_settings

            blocks = blocks_settings(study["blocks"], games=len(self.games_list))
            if phase2["active_limit"] != blocks["slots"] or study["active_limit"] != blocks["slots"]:
                raise RuntimeError(f"blocks need {blocks['slots']} streams; the served fit gives "
                                   f"{phase2['active_limit']} (focus fit {choice.get('focus_fit')})")
            if float(blocks["wall_seconds"]) != float(self.wall_seconds):
                raise ValueError("blocks.wall_seconds must be the study wall the agents see")
            floor = (study.get("blocks_plan") or {}).get("min_pool_blocks")
            if floor and block_budget is not None and int(block_budget.layout["num_blocks"]) < int(floor):
                raise RuntimeError(f"blocks need a served pool of at least {floor} blocks (the window replay's); "
                                   f"got {block_budget.layout['num_blocks']}")
            choice["blocks"] = {"slots": blocks["slots"], "block_sizes": blocks["block_sizes"],
                                "block_seconds": blocks["block_seconds"], "wall_seconds": blocks["wall_seconds"]}
            roster = self.output / "public-roster.json"
            if roster.is_file():
                # User requirement (29 September): block 1 = every source game once + distinct second clones.
                from .blocks import check_block1

                rows = json.loads(roster.read_text())
                if [r["game_id"] for r in rows] != list(self.games_list):
                    raise RuntimeError("public-roster.json does not match the trial's roster order")
                first = check_block1([r["source_game_id"] for r in rows], blocks)
                choice["blocks"]["block1"] = first
                self.gate_log({"event": "blocks_block1", **first})
        (self.output / 'phase2-capacity.json').write_text(json.dumps(phase2, indent=2))
        wall = GlobalClock(self.wall_seconds)
        self.wall = wall
        games = []
        # Token budgets (29 September): completion tokens are charged by the Duck request path only
        # (``ArenaDuckMixin.record_reply``); other agent kinds would never exhaust a token budget.
        token_budget = study.get("token_budget")
        if study.get("budget_clock") == "tokens" and not self.kind.startswith("duck"):
            raise ValueError("The token budget clock is wired for the Duck agents only")
        if (token_budget is not None) != (study.get("budget_clock") == "tokens"):
            raise ValueError("The token budget clock and token_budget go together")
        for index, game_id in enumerate(self.games_list):
            record = GameRecord(game_id, index, ActiveClock(
                wall, budget_clock=study.get("budget_clock", "inference"),
                normalization=study.get("cost_normalization"),
                token_purposes=None if token_budget is None else token_budget["purposes"]))
            record.clock.pause()  # nothing plays before the release
            record.ready = asyncio.Event()
            games.append(record)
        lane = None
        if study.get("background_lane") is not None:
            # fixed-2x-index-lane: parked games keep requesting in the gate's background lane, so every
            # consumer of ``clock.parked`` must be the Duck-transfer request path alone.
            if (self.kind != "duck-transfer" or study.get("explorer") or study.get("cross_game_notes")
                    or study.get("audition_pool") is not None or study.get("llm_backlog")):
                raise ValueError("The background lane is wired for plain Duck-transfer only")
            lane = BackgroundLane(games, window=window, cap=self.cap, **study["background_lane"])
        commit_server = attribution = None
        if study.get("commit") is not None:
            # fixed-2x-commit: the same plain Duck-transfer restriction as the lane above, on vLLM, and the
            # server must run the priority policy before any request carries a priority.
            from .commit import CommitLane, server_scheduling_policy

            if (self.kind != "duck-transfer" or study.get("explorer") or study.get("cross_game_notes")
                    or study.get("audition_pool") is not None or study.get("llm_backlog")
                    or study.get("background_lane") is not None or study.get("backend", "vllm") != "vllm"):
                raise ValueError("The commit lane is wired for plain Duck-transfer on vLLM only")
            policy, source = server_scheduling_policy(self.server_dir)
            if policy != "priority":
                raise RuntimeError(f"fixed-2x-commit needs a vLLM server with --scheduling-policy priority; "
                                   f"found {policy!r} ({source})")
            commit_server = {"scheduling_policy": policy, "source": source}
            self.gate_log({"event": "commit_server", **commit_server})
            pool_units = block_budget.layout["num_blocks"] if block_budget is not None else choice["pool"]
            lane = CommitLane(games, window=window, cap=self.cap, pool_units=pool_units, **study["commit"])
        if (study.get("deep_explore") is not None or study.get("depthwise") is not None) and (
                self.kind != "duck-transfer" or study.get("cache_admission") != "hybrid_blocks"
                or study.get("explorer") or study.get("cross_game_notes") or study.get("audition_pool") is not None
                or study.get("llm_backlog") or lane is not None):
            # The per-game window (compaction threshold) is wired for the Duck-transfer agent, and the capacity
            # check above needs the served block layout.
            raise ValueError("deep_explore and depthwise are wired for plain Duck-transfer with hybrid-block admission only")
        # parallel_samples: the siblings' sequences are the server's max_num_seqs above the generation slots (regular
        # requests keep the generation slots); 0 on every other study.
        background_slots = (int(study["server"]["max_num_seqs"]) - int(study["generation_slots"])
                            if study.get("parallel_samples") else 0)
        gate = KVGate(choice["capacity"], slots=int(study["generation_slots"]), log=self.gate_log,
                      budget=block_budget, opening_burst_requests=study.get("opening_burst_requests", 1),
                      lane=lane, background_slots=background_slots)
        # Prefix cache v3 (a): keep parked games' prefixes warm (arena.keep_alive); off unless the study sets it.
        from .keep_alive import settings as keep_alive_settings

        keep_spec = keep_alive_settings((study.get("cache_experiment") or {}).get("keep_alive"))
        if keep_spec is not None and (not self.kind.startswith("duck") or block_budget is None or lane is not None):
            raise ValueError("keep_alive is wired for Duck agents on the plain gate with hybrid-block admission")
        if study.get("parallel_samples") and (not self.kind.startswith("duck") or block_budget is None
                                              or lane is not None):
            # Siblings reserve their exact blocks through try_acquire (plain gate, served block layout).
            raise ValueError("parallel_samples is wired for Duck agents on the plain gate with hybrid-block admission")
        if study.get("action_plans") and self.kind != "duck-transfer":
            raise ValueError("action_plans extends the corrected Duck agent's persistent sandbox (duck-transfer)")
        if study.get("drain_half") and not self.kind.startswith("duck"):
            raise ValueError("drain_half replaces the Duck agents' summary compaction (duck-transfer, duck-public)")
        if study.get("dfr_harness") and self.kind != "duck-transfer":
            raise ValueError("dfr_harness extends the corrected Duck agent (duck-transfer)")
        if study.get("dfr_loop") and self.kind != "duck-transfer":
            raise ValueError("dfr_loop extends the corrected Duck agent (duck-transfer)")
        if study.get("hx_harness") and self.kind != "duck-transfer":
            raise ValueError("hx_harness extends the corrected Duck agent (duck-transfer)")
        scheduler = Scheduler(
            games, wall, log=self.gate_log, active_limit=phase2['active_limit'],
            switch_games=int(study["opening_switch_level1_games"]),
            max_grant_seconds=study.get("max_level_grant_seconds"),
            max_grant_t1_multiple=study.get("max_level_grant_t1_multiple"),
            regrant_multiple=study.get("regrant_multiple"),
            budget_policy=study.get("level_budget_policy", "incremental"),
            backlog_policy=study.get("backlog_policy", "spent"),
            progress_model=study.get("progress_model", "levels"),
            prioritize_parked_progress=study.get("prioritize_parked_progress", False),
            t1_cap_seconds=study.get("t1_cap_seconds"),
            t1_clip=study.get("t1_clip"),
            index_settings=study.get("index_scheduler"),
            audition_pool=study.get("audition_pool"),
            pool_stall_actions=study.get("pool_stall_actions"),
            pool_action_limit=study.get("pool_action_limit"),
            focus_stall_actions=study.get("focus_stall_actions"),
            opening_max_wall_seconds=study.get("opening_max_wall_seconds"),
            opening_min_tokens=study.get("opening_min_tokens"),
            level0_order=study.get("level0_order", "roster"),
            pool_time_limit_seconds=study.get("pool_time_limit_seconds"),
            reopen_phase1=study.get("reopen_phase1"),
            budget_multiple=study.get("level_budget_multiple"),
            thompson_seed=study.get("thompson_seed", 0),
            background_lane=study.get("background_lane"),
            commit=study.get("commit"),
            deep_explore=study.get("deep_explore"),
            depthwise=study.get("depthwise"),
            rank=study.get("rank"),
            token_floor=None if token_budget is None else token_budget["floor_tokens"],
            backlog_key=study.get("backlog_key"),
            blocks=study.get("blocks"),
            level_token_allowance=study.get("level_token_allowance"),
            priority=study.get("priority"),
        )
        if (study.get("budget_clock") == "normalized" or study.get("audition_pool") is not None) and self.kind != "duck-transfer":
            raise ValueError("The normalized clock and the audition pool are wired for Duck-transfer only")
        gate.opening = lambda: scheduler.phase == "opening" or scheduler.reopen_active()
        for game in games:
            game.observer = scheduler
        arc = self.arcade if self.arcade is not None else make_arcade("offline", self.games_dir, self.output / "recordings")
        available = {g.game_id for g in arc.get_environments()}
        missing = [g for g in self.games_list if g not in available]
        if missing:
            raise ValueError(f"Games not in the arcade: {missing}")
        card = arc.open_scorecard(tags=["arena", self.trial["name"]])
        monitor_type = GpuMonitor
        if study.get("backend") == "sglang":
            from .sglang import ArenaSglangMonitor

            monitor_type = ArenaSglangMonitor
        monitor = monitor_type(self.output, self.base_url)
        live = live_settings(study.get("live_progress"))
        # Live progress (hero-v2): score, levels, tokens and throughput in progress.json; monitoring only.
        self.live = None if live is None else LiveProgress(live, arc=arc, card=card, monitor=monitor, games=games)
        self.early_stop = self.early_stop_rule(study, arc, card, games)
        self.end_reason = None
        if commit_server is not None:
            from .commit import PreemptionAttribution

            attribution = PreemptionAttribution(gate, slack_seconds=lane.cfg["preemption_slack_seconds"])
            monitor.on_sample = attribution.observe
        self.compaction_preflight = None
        if (self.kind.startswith("duck") and study.get("compaction_thinking") == "prefilled-empty"
                and study.get("backend", "vllm") == "vllm"):
            # The prefilled-empty summary's token identity on the served template (arena.compaction_preflight);
            # "gameplay" when it fails.
            from .compaction_preflight import run_preflight

            async with httpx.AsyncClient(timeout=60, trust_env=False) as preflight_client:
                self.compaction_preflight = await run_preflight(
                    preflight_client, self.base_url, model=study.get("served_model") or "Qwen/Qwen3.8-Flash-Next-NVFP4",
                    effort=study.get("reasoning_effort"), enable_thinking=study.get("enable_thinking", True))
            (self.output / "compaction-preflight.json").write_text(json.dumps(self.compaction_preflight, indent=2))
            self.log(json.dumps({"event": "compaction_preflight", "ok": self.compaction_preflight["ok"],
                                 "mode": self.compaction_preflight["mode"],
                                 "reason": self.compaction_preflight["reason"]}), flush=True)
        workers, rows_fn, close = await self.duck_agents(games, gate, window, wall, card, arc)
        async def run_worker(worker, game):
            try:
                returned = await worker.run()
            except Exception as exc:
                game.error = f"{type(exc).__name__}: {exc}"
                game.note_finished("worker_error")
                self.gate_log({"event": "worker_error", "game": game.game_id, "error": game.error})
                raise
            else:
                # Duck also returns normally on action limits and handled errors.
                # A dead worker must not keep accumulating active time or hold a
                # focus slot. Preserve explicit outcomes (won, model_finished, ...).
                if wall.remaining() > 0:
                    if game.finished_reason is None:
                        game.note_finished("worker_returned")
                    self.gate_log({"event": "worker_returned", "game": game.game_id,
                                   "reported_status": returned.get("status") if isinstance(returned, dict) else None,
                                   "reported_seconds": returned.get("seconds") if isinstance(returned, dict) else None,
                                   "finished_reason": game.finished_reason})

        tasks = [asyncio.create_task(run_worker(w, g)) for w, g in zip(workers, games, strict=True)]
        # Barrier: every environment exists before the clocks release.
        try:
            async with asyncio.timeout(READY_TIMEOUT):
                await asyncio.gather(*(g.ready.wait() for g in games))
        except TimeoutError as exc:
            unready = [g.game_id for g in games if not g.ready.is_set()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise RuntimeError(f"Games not ready after {READY_TIMEOUT} s: {unready}") from exc
        await monitor.sample()
        if self.gameplay_deadline_epoch is not None:
            # Worker initialization counts against the whole notebook budget.
            # Recompute at the release barrier so setup cannot extend gameplay.
            wall.seconds = min(wall.seconds, max(0, self.gameplay_deadline_epoch - time.time()))
        wall.release()
        for game in games:
            game.clock.release()
        monitor.start = wall.start
        scheduler.start_all()
        sampling = asyncio.create_task(monitor.run())
        ranker = ranking = None
        if study.get("llm_backlog"):
            from .llm_rank import LLMRanker

            spec = study["llm_backlog"]
            rank_client = httpx.AsyncClient(base_url=self.base_url.rstrip("/") + "/", timeout=300, trust_env=False)
            ranker = LLMRanker(scheduler, gate, rank_client, model=study.get("served_model"), wall=wall,
                               interval_seconds=spec.get("interval_seconds", 180), cap_tokens=spec.get("cap_tokens", 768),
                               temperature=study.get("temperature", 0.7), log=self.gate_log)
            ranking = asyncio.create_task(ranker.run())
        explorer = exploring = None
        if study.get("explorer") and self.kind.startswith("duck"):
            from .explorer import ExplorerService

            explorer = ExplorerService(scheduler, workers, wall, study["explorer"], log=self.gate_log)
            exploring = asyncio.create_task(explorer.run())
        keeper = keeping = None
        if keep_spec is not None:
            from .keep_alive import KeepAlive

            keeper = KeepAlive(scheduler, gate, {g.game_id: w for g, w in zip(games, workers, strict=True)},
                               keep_spec, window=window + self.cap, log=self.gate_log)
            keeping = asyncio.create_task(keeper.run())
        last_progress = 0.0
        try:
            while wall.remaining() > 0 and any(not t.done() for t in tasks):
                await asyncio.wait([t for t in tasks if not t.done()],
                                   timeout=min(1.0, max(0.05, wall.remaining())),
                                   return_when=asyncio.FIRST_COMPLETED)
                scheduler.tick()
                if lane is not None:
                    gate.refresh()  # roles and focus guards may have changed
                stop = self.early_stop.check(wall.elapsed())
                if stop is not None and self.end_reason is None:
                    # A graceful early end (arena.early_stop): the wall ends now, everything after is a normal wall end.
                    self.end_reason = stop
                    end_trial(wall)
                    last_progress = wall.elapsed()
                    self.progress(wall, games, scheduler, gate)
                if wall.elapsed() - last_progress >= 15:
                    last_progress = wall.elapsed()
                    self.heap.tick(wall.elapsed())
                    self.progress(wall, games, scheduler, gate)
            if self.end_reason is None:
                self.end_reason = "wall" if wall.remaining() <= 0 else "all_workers_done"
        finally:
            scheduler.wall_end()
            if keeping is not None:
                keeping.cancel()
                await asyncio.gather(keeping, return_exceptions=True)
                await keeper.close()
            if exploring is not None:
                exploring.cancel()
                await asyncio.gather(exploring, return_exceptions=True)
            if ranking is not None:
                ranking.cancel()
                await asyncio.gather(ranking, return_exceptions=True)
                await ranker.client.aclose()
            sampling.cancel()
            await asyncio.gather(sampling, return_exceptions=True)
            await monitor.sample()
            elapsed = wall.elapsed()
            for task in tasks:
                if not task.done():
                    task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            try:
                # C/R rows read SQLite evidence. Drain workers first so their final
                # receipts are included, then snapshot metrics before closing archives.
                base_rows = rows_fn()
            finally:
                try:
                    await close()
                finally:
                    closed = await asyncio.to_thread(arc.close_scorecard, card)
        if closed is None:
            raise RuntimeError("Failed to close the arena scorecard")
        (self.output / "scorecard.json").write_text(
            json.dumps(closed.model_dump(mode="json", exclude={"api_key"}), indent=2)
        )
        scores = {env.id: env.score for env in closed.environments}
        if set(scores) != set(self.games_list):
            raise RuntimeError("Scorecard sessions do not match the game list")
        task_errors = [
            f"{game.game_id}: {type(o).__name__}: {o}"
            for game, o in zip(games, outcomes, strict=True)
            if isinstance(o, BaseException) and not isinstance(o, asyncio.CancelledError)
        ]
        rows = [game_row(game, {**base, "score": scores[game.game_id],
                               **({"status": "error", "error": game.error} if game.error else {})})
                for game, base in zip(games, base_rows, strict=True)]
        config = workers[0].arena_config() if hasattr(workers[0], "arena_config") else self.config_dict
        report = trial_report(
            self.trial, rows=rows, elapsed=elapsed, choice=choice, cap=self.cap, config=config,
            scheduler=scheduler.summary(), gate=gate.snapshot(), engine=monitor.summary(),
            task_errors=task_errors, preemption_policy=attribution,
        )
        if commit_server is not None:
            report["commit_server"] = commit_server
        if getattr(self, "compaction_preflight", None) is not None:
            report["compaction_preflight"] = self.compaction_preflight
        if getattr(self, "notes", None) is not None:
            report["cross_game_notes"] = self.notes.summary()
        if ranker is not None:
            report["llm_backlog"] = ranker.summary()
        if explorer is not None:
            report["explorer"] = explorer.summary()
        if keeper is not None:
            report["keep_alive"] = keeper.summary()
        report.update(self.feature_totals([base for base in base_rows]))
        if getattr(self, "harness", None) is not None:
            report["rlm_harness"] = self.harness.summary()
        report["end_reason"] = self.end_reason or "error"
        report["wall"] = {"configured_seconds": self.wall_seconds, "ended_seconds": round(wall.seconds, 3)}
        if self.early_stop.cfg is not None or self.early_stop.reason is not None:
            report["early_stop"] = self.early_stop.summary()
        (self.output / "arena.json").write_text(json.dumps(report, indent=2))
        self.admission_log.close()
        return report

    def progress(self, wall, games, scheduler, gate):
        progress = {
            "event": "progress",
            "elapsed_seconds": round(wall.elapsed(), 1),
            "phase": scheduler.phase,
            "active": [g.game_id for g in games if g.state == "active"],
            "levels": sum(g.levels for g in games),
            "parked": sum(g.state == "parked" for g in games),
            "finished": sum(g.state == "finished" for g in games),
            "gate": {k: gate.snapshot()[k] for k in ("reserved", "inflight", "waiting")},
        }
        if getattr(self, "heap", None) is not None:
            progress["heap"] = self.heap.summary()
        if getattr(scheduler, "deep", None) is not None:
            progress["deep"] = [g.game_id for g in scheduler.deep_games()]
            progress["deep_queue"] = [g.game_id for g in scheduler.promotion_queue()]
            progress["explore"] = len(scheduler.explore_games())
        if getattr(scheduler, "depthwise", None) is not None:
            progress["depth"] = [g.game_id for g in scheduler.dw_slot_games("depth")]
            progress["level0"] = len(scheduler.dw_slot_games("level0"))
            progress["parked_depth"] = [g.game_id for g in scheduler.dw_waiting() if g.levels >= 1]
        if getattr(self, "live", None) is not None:
            progress["live"] = self.live.fields(wall.elapsed())
            progress["live"]["serving"] = serving_check(progress.get("gate"), progress["live"].get("engine"))
        workers = getattr(self, "duck_workers", None)
        if workers and (self.study.get("parallel_samples") or self.study.get("action_plans")
                        or self.study.get("drain_half") or self.study.get("loop_rule")):
            rows = []
            for worker in workers:
                row = {}
                if worker.sampler is not None:
                    row["parallel_samples"] = worker.sampler.summary()
                if worker.action_plans:
                    row["action_plans"] = worker.plan_summary()
                if worker.drain is not None:
                    row["context_drain"] = dict(worker.drain_stats)
                if worker.lrule is not None:
                    row["loop_rule"] = dict(worker.lrule_stats)
                rows.append(row)
            progress.update(self.feature_totals(rows))
        if getattr(scheduler, "rank", None) is not None:
            progress["rank"] = {**scheduler.rank_counts, "top": [g.game_id for g in scheduler.rank_order[:10]]}
        if getattr(scheduler, "blocks", None) is not None:
            progress["blocks"] = scheduler.blocks_progress()
        if getattr(scheduler, "priority_cfg", None) is not None:
            progress["priority"] = scheduler.priority_progress()
        early = getattr(self, "early_stop", None)
        if early is not None and (early.cfg is not None or early.reason is not None):
            summary = early.summary()
            progress["early_stop"] = {k: summary[k] for k in ("decision", "reason", "score",
                                                               "decided_at_elapsed_seconds", "stop_request")}
            progress["early_stop"]["score_above"] = (early.cfg or {}).get("score_above")
        if gate.lane is not None:
            lane = gate.lane_snapshot()
            progress["lane"] = {k: lane[k] for k in ("lane_inflight", "lane_waiting", "lane_admitted",
                                                      "guard_deficit", "lane_estimate_units", "focus_wait_seconds")
                                if k in lane}
        (self.output / "progress.json").write_text(json.dumps(progress, indent=2))
        with (self.output / "progress.jsonl").open("a") as file:
            file.write(json.dumps(progress) + "\n")
        self.log(json.dumps(progress), flush=True)

    def feature_totals(self, rows):
        """``{"parallel_samples": totals, "action_plans": totals}`` over the games' rows, for the features on."""
        totals = {}
        if self.study.get("parallel_samples"):
            from .parallel_samples import aggregate

            totals["parallel_samples"] = aggregate([r.get("parallel_samples") for r in rows])
        if self.study.get("action_plans"):
            from .action_plans import aggregate as plans_aggregate

            totals["action_plans"] = plans_aggregate([r.get("action_plans") for r in rows])
        if self.study.get("drain_half"):
            from .context_drain import aggregate as drain_aggregate

            totals["context_drain"] = drain_aggregate([r.get("context_drain") for r in rows])
        if self.study.get("loop_rule"):
            from .loop_rule import aggregate as loop_rule_aggregate

            totals["loop_rule"] = loop_rule_aggregate([r.get("loop_rule") for r in rows])
        if self.study.get("dfr_loop"):
            from .dfr_loop import aggregate as loop_aggregate

            totals["dfr_loop"] = loop_aggregate([r.get("dfr_loop") for r in rows])
        return totals

    def early_stop_rule(self, study, arc, card, games):
        """The early-stop rule (``study.early_stop``) and the manual stop request (arena.early_stop)."""
        cfg = early_stop_settings(study.get("early_stop"))
        live = getattr(self, "live", None)

        def read(elapsed):
            if live is not None:
                return live.read_now(elapsed)  # the same read as live progress, which then shows it
            from .live_progress import read_score

            score = read_score(arc, card, len(games))
            return None if score is None else {**score, "at_elapsed_seconds": round(elapsed, 1)}

        return EarlyStop(cfg, read_score=read, paths=stop_paths(self.output), log=self.gate_log)

    # --- agent construction ---------------------------------------------------------------

    async def duck_agents(self, games, gate, window, wall, card, arc):
        cfg = duck_config(self.study, self.source, window=window, cap=self.cap, wall=self.wall_seconds, base_url=self.base_url)
        from duck_transfer import loader

        loader.configure(cfg, sampling=self.study.get("sampling"))
        from .duck import ArenaPublicDuckAgent, ArenaTransferDuckAgent
        from .harness import DuckHarness

        cls = ArenaTransferDuckAgent if self.kind == "duck-transfer" else ArenaPublicDuckAgent
        self.config_dict = cfg.to_dict()
        # 25 September variants: one notebook shared by the trial's games (fixed-2x-notes,
        # mirrored to cross-game-notes.md) and the memory-block settings (fixed-2x-memory).
        self.notes = None
        if self.study.get("cross_game_notes"):
            from .notes import CrossGameNotes

            self.notes = CrossGameNotes.from_settings(self.output / "cross-game-notes.md",
                                                      self.study["cross_game_notes"])
        memory = self.study.get("memory_block") or None
        # Offline harness evolution: the study names a harness directory (relative to the bundle
        # source or absolute); every Duck game loads it.
        self.harness_dir = None
        if self.study.get("duck_harness"):
            path = Path(self.study["duck_harness"])
            self.harness_dir = path if path.is_absolute() else self.source / path
            if self.study.get("harness_v2"):
                from .harness_v2 import HarnessV2Dir

                self.gate_log({"event": "duck_harness", **HarnessV2Dir(self.harness_dir, self.study["harness_v2"]).summary()})
            else:
                self.gate_log({"event": "duck_harness", **DuckHarness(self.harness_dir).summary()})
        shared = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/") + "/",
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128, keepalive_expiry=KEEPALIVE_EXPIRY),
            trust_env=False,
        )
        requests = []
        agents = []
        for game in games:
            entry = {"game_id": game.game_id, "replica": 0, "instance_id": f"r0-{game.game_id}"}
            slot = GateSlot(gate, game, cap=self.cap, window=window)
            client = SessionClient(cfg, shared, slot, entry, requests)
            agent = cls(
                game.game_id, arc, card, client, cfg, self.output / "games" / game.game_id,
                self.analysis_python, time.monotonic() + self.wall_seconds + 3600,
            )
            agent.attach(game, window=window, cap=self.cap, notes=self.notes, memory=memory,
                         escalation=self.study.get("model_escalation") or None,
                         snapshot=self.study.get("game_snapshot") or None,
                         harness=self.harness_dir, procedure=self.study.get("stuck_procedure") or None,
                         explorer=self.study.get("explorer") or None,
                         critical_errors=bool(self.study.get("critical_errors")),
                         harness_v2=self.study.get("harness_v2") or None,
                         adaptive_thinking=self.study.get("adaptive_thinking") or None,
                         worldmodel_notice=bool(self.study.get("worldmodel_notice")),
                         harness_fixes=self.study.get("harness_fixes") or None,
                         parallel_samples=self.study.get("parallel_samples") or None,
                         action_plans=bool(self.study.get("action_plans")),
                         context_drain=self.study.get("drain_half") or None,
                         dfr_harness=self.study.get("dfr_harness") or None,
                         dfr_loop=self.study.get("dfr_loop") or None,
                         hx_harness=self.study.get("hx_harness") or None,
                         loop_rule=self.study.get("loop_rule") or None)
            agent.compaction_keep_observations = int(self.study.get("compaction_keep_observations", 1))
            agent.compaction_keep_fraction = self.study.get("compaction_keep_fraction")
            agent.compaction_summary_cap = self.study.get("compaction_summary_cap")
            agent.compaction_full_prefix = bool(self.study.get("compaction_full_prefix"))
            agent.trim = trim_settings(self.study.get("trim"))
            stagger = self.study.get("first_trigger_stagger")
            if stagger is not None:
                # Game i's first drain / compaction triggers at min + (1 - min) x frac(i x golden ratio) of the budget.
                low = float(stagger["min_fraction"])
                if not 0.3 <= low < 1:
                    raise ValueError("first_trigger_stagger.min_fraction is in [0.3, 1)")
                agent.first_trigger_fraction = low + (1 - low) * ((len(agents) * 0.6180339887498949) % 1.0)
            agent.compaction_thinking = self.study.get("compaction_thinking", DEFAULT_COMPACTION_THINKING)
            if getattr(self, "compaction_preflight", None) is not None:
                agent.compaction_thinking = self.compaction_preflight["mode"]  # "gameplay" if the preflight failed
            agents.append(agent)
        self.duck_workers = agents  # parallel_samples / action_plans totals in progress.json

        def rows():
            return [agent.result() for agent in agents]

        async def close():
            (self.output / "requests.json").write_text(json.dumps(requests, indent=2))
            await shared.aclose()

        return agents, rows, close


async def run_trial(study, trial, **kwargs):
    return await Trial(study, trial, **kwargs).run()
