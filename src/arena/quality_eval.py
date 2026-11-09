"""Fixed-game quality evaluations; all generated purposes share a per-game allowance."""

import asyncio
import hashlib
import time

DEV10 = tuple("cd82 cn04 ft09 lp85 ls20 re86 sb26 sk48 vc33 wa30".split())
HELDOUT15 = tuple("ar25 bp35 dc22 g50t ka59 lf52 m0r0 r11l s5i5 sc25 sp80 su15 tn36 tr87 tu93".split())
LONG20 = tuple(sorted(set(DEV10 + HELDOUT15) - {"ft09", "lp85", "ar25", "vc33", "tu93"}))
SPLITS = {"dev10": DEV10, "heldout15": HELDOUT15, "long20": LONG20}


def settings(spec):
    if spec is None:
        return None
    if isinstance(spec, dict) and type(spec.get("version")) is int and spec["version"] == 2:
        return _settings_v2(spec)
    keys = {"version", "split", "per_game_tokens", "checkpoints", "seed"}
    if not isinstance(spec, dict) or set(spec) != keys or type(spec["version"]) is not int or spec["version"] != 1:
        raise ValueError("quality_eval requires version 1 and exactly " + str(sorted(keys)))
    if spec["split"] not in SPLITS:
        raise ValueError("Unknown quality_eval split")
    budget, points = spec["per_game_tokens"], spec["checkpoints"]
    if type(budget) is not int or budget < 1 or type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("quality_eval needs positive integer tokens and a nonnegative integer seed")
    if (not isinstance(points, list) or not points or any(type(p) is not int or not 0 < p <= budget for p in points)
            or points != sorted(set(points)) or points[-1] != budget):
        raise ValueError("quality_eval checkpoints must increase, end at per_game_tokens, and stay within it")
    return {**spec, "checkpoints": list(points)}


def _settings_v2(spec):
    """Replicated roster: ``replicas`` copies of every split game in one run; replica r is the single-roster
    (version 1) run with seed ``seeds[r]``: the same environment seed and per-request seed stream."""
    keys = {"version", "split", "per_game_tokens", "checkpoints", "replicas", "seeds"}
    if set(spec) - {"families"} != keys:
        raise ValueError("quality_eval version 2 requires exactly " + str(sorted(keys)) + " (and optional families)")
    if spec["split"] not in SPLITS:
        raise ValueError("Unknown quality_eval split")
    if "families" in spec:
        # A shard (7 October): every replica of some of the split's families, for servers whose KV gate cannot hold
        # the whole roster. Seeds and budgets are per worker, so shards add up to the full-roster run.
        shard = spec["families"]
        if (not isinstance(shard, list) or not shard or len(set(shard)) != len(shard)
                or not set(shard) < set(SPLITS[spec["split"]])):
            raise ValueError("quality_eval families must be a non-empty proper subset of the split, without repeats")
    budget, points, replicas, seeds = (spec[k] for k in ("per_game_tokens", "checkpoints", "replicas", "seeds"))
    if type(budget) is not int or budget < 1:
        raise ValueError("quality_eval needs positive integer tokens")
    if (not isinstance(points, list) or not points or any(type(p) is not int or not 0 < p <= budget for p in points)
            or points != sorted(set(points)) or points[-1] != budget):
        raise ValueError("quality_eval checkpoints must increase, end at per_game_tokens, and stay within it")
    if type(replicas) is not int or not 2 <= replicas <= 16:
        raise ValueError("quality_eval version 2 needs 2..16 replicas")
    if (not isinstance(seeds, list) or len(seeds) != replicas or len(set(seeds)) != replicas
            or any(type(v) is not int or v < 0 for v in seeds)):
        raise ValueError("quality_eval version 2 needs one distinct nonnegative integer seed per replica")
    return {**spec, "checkpoints": list(points), "seeds": list(seeds)}


def families_of(cfg):
    """The families a run plays: the split, or a version 2 shard's ``families``."""
    return tuple(cfg.get("families") or SPLITS[cfg["split"]])


def worker_count(cfg):
    return len(families_of(cfg)) * cfg.get("replicas", 1)


def gateway_runs(cfg):
    """``total_runs`` of the public gateway: None serves the source games themselves (version 1); version 2 clones
    them, exposed game i being source ``i % n`` (taaf.competition_arcade)."""
    return worker_count(cfg) if cfg.get("version") == 2 else None


def roster(cfg, sources, exposed):
    """Version 2 worker identities in gateway order: replica ``i // n`` of source ``i % n``. None for version 1."""
    if cfg.get("version") != 2:
        return None
    sources, exposed, n = list(sources), list(exposed), len(families_of(cfg))
    if len(sources) != n or len(exposed) != n * cfg["replicas"] or len(set(exposed)) != len(exposed):
        raise ValueError("quality_eval roster needs every source game once per replica under unique ids")
    return [{"game_id": game, "source_game_id": sources[i % n], "replica": i // n, "seed": cfg["seeds"][i // n]}
            for i, game in enumerate(exposed)]


def worker_identity(cfg, study, game_id):
    """(environment seed, request-seed game id, replica) of one worker. Version 1: the run's seed and the worker's own
    id. Version 2: its replica's seed and its *source* id, so replica r repeats the version 1 run with seed r."""
    if cfg.get("version") != 2:
        return {"seed": cfg["seed"], "source_game_id": game_id, "replica": None}
    row = next((r for r in study.get("quality_roster") or [] if r["game_id"] == game_id), None)
    if row is None:
        raise ValueError(f"quality_eval roster has no worker {game_id}")
    return {k: row[k] for k in ("seed", "source_game_id", "replica")}


_FORBIDDEN = ("priority", "blocks", "rank", "deep_explore", "depthwise", "background_lane", "commit", "explorer",
              "audition_pool", "parallel_samples", "llm_backlog", "cross_game_notes", "focus_fit_fallback",
              "focus_oversubscription", "reopen_phase1", "early_stop")


def _validate_v2(study, cfg):
    families = families_of(cfg)
    n, total = len(families), worker_count(cfg)
    rows = study.get("quality_roster")
    if rows is None:  # the bundle's study: the source games; the gateway clones them at run time
        sources = list(study["games"])
    else:  # the resolved study: the exposed clone ids with their identities
        sources = [r.get("source_game_id") for r in rows[:n]] if isinstance(rows, list) else []
        if len(sources) != n or rows != roster(cfg, sources, study["games"]):
            raise ValueError("quality_eval roster does not match the replicated games")
    if len(sources) != n or {str(g).split("-", 1)[0] for g in sources} != set(families):
        raise ValueError("quality_eval requires exactly one source game per split family in every replica")
    slots = study.get("generation_slots")
    if study.get("active_limit") != total or type(slots) is not int or not 1 <= slots <= total:
        raise ValueError("quality_eval version 2 needs one persistent worker per replica game and 1..workers "
                         "generation slots")
    if (any(study.get(k) for k in _FORBIDDEN) or (study.get("cache_experiment") or {}).get("keep_alive")
            or study.get("require_focus_windows_fit")):
        raise ValueError("quality_eval disallows scheduling, cross-game sharing, and unaccounted background generation")
    if study.get("phase2_admission", "fixed") != "fixed" or study.get("level_budget_policy") != "quality-fixed":
        raise ValueError("quality_eval requires the explicit quality-fixed policy")
    if study.get("budget_clock") != "tokens" or (study.get("token_budget") or {}).get("purposes") != ["*"]:
        raise ValueError("quality_eval must count every generated request purpose")
    if int(study["server"]["max_num_seqs"]) < slots:
        raise ValueError("quality_eval server cannot admit the generation slots")
    if (study.get("competition") or {}).get("preview_instances", total) != total:
        raise ValueError("quality_eval preview must contain every replica of the fixed roster exactly once")
    return cfg


def validate_study(study):
    cfg = settings(study.get("quality_eval"))
    if cfg is None:
        return None
    if cfg["version"] == 2:
        return _validate_v2(study, cfg)
    games = study["games"]
    families = [g.split("-", 1)[0] for g in games]
    n = len(SPLITS[cfg["split"]])
    if len(games) != n or len(set(families)) != n or set(families) != set(SPLITS[cfg["split"]]):
        raise ValueError("quality_eval requires exactly one source game per split family (no clones)")
    if study.get("generation_slots") != n or study.get("active_limit") != n:
        raise ValueError("quality_eval needs one persistent worker and generation slot per game")
    forbidden = ("priority", "blocks", "rank", "deep_explore", "depthwise", "background_lane", "commit", "explorer",
                 "audition_pool", "parallel_samples", "llm_backlog", "cross_game_notes", "focus_fit_fallback",
                 "focus_oversubscription", "reopen_phase1", "early_stop")
    if any(study.get(k) for k in forbidden) or (study.get("cache_experiment") or {}).get("keep_alive"):
        raise ValueError("quality_eval disallows scheduling, cross-game sharing, and unaccounted background generation")
    if study.get("phase2_admission", "fixed") != "fixed" or study.get("level_budget_policy") != "quality-fixed":
        raise ValueError("quality_eval requires the explicit quality-fixed policy")
    if study.get("budget_clock") != "tokens" or (study.get("token_budget") or {}).get("purposes") != ["*"]:
        raise ValueError("quality_eval must count every generated request purpose")
    if int(study["server"]["max_num_seqs"]) < n:
        raise ValueError("quality_eval server cannot admit the fixed worker roster")
    if (study.get("competition") or {}).get("preview_instances", n) != n:
        raise ValueError("quality_eval preview must contain the fixed roster exactly once")
    return cfg


class SeededArcade:
    def __init__(self, arcade, seed):
        self.arcade, self.seed = arcade, seed

    def __getattr__(self, key):
        return getattr(self.arcade, key)

    def make(self, *args, **kwargs):
        return self.arcade.make(*args, **{**kwargs, "seed": self.seed})


class FixedWorkers:
    """No ranking, grants, eviction handovers, retirement, or replacement workers."""

    phase = "quality-fixed"

    def __init__(self, games, wall, *, log, cfg):
        self.games, self.wall, self.log, self.cfg = games, wall, log, cfg

    def start_all(self):
        for game in self.games:
            game.state = "active"
            game.clock.grant(self.cfg["per_game_tokens"])
            game.clock.resume()
            row = {"event": "admit", "game": game.game_id, "wall_seconds": self.wall.elapsed(),
                   "reason": "quality_start", "budget_tokens": self.cfg["per_game_tokens"]}
            game.admissions.append(row)
            self.log(row)

    def reopen_active(self):
        return False

    def on_level(self, game):
        pass

    def on_context_eviction(self, game, kind, **details):
        pass

    def tick(self):
        for game in self.games:
            if game.finished_reason is not None and game.state != "finished":
                game.state = "finished"
                game.clock.pause()
                self.log({"event": "finished", "game": game.game_id, "reason": game.finished_reason})

    def wall_end(self):
        self.tick()
        for game in self.games:
            game.clock.pause()

    def summary(self):
        return {"budget_policy": self.phase, "budget_clock": "tokens", "active_limit": len(self.games),
                "quality_eval": self.cfg, "token_budget": {"purposes": ["*"], "floor_tokens": 1}}


class GameBudget:
    def __init__(self, cfg, game, *, log, score, identity=None):
        self.cfg, self.game, self.log, self.score = cfg, game, log, score
        # Version 2: the worker's replica seed and source game id (``identity``); version 1: the run's.
        self.identity = identity or {"seed": cfg["seed"], "source_game_id": game.game_id, "replica": None}
        self.tokens = 0
        self.calls = 0
        self.zero_token_streak = 0
        self.invalid = []
        self.stopped = False
        self.inflight = False
        self.purposes = {}
        self.score_points = [{"tokens": 0, "score": 0.0}]

    @property
    def remaining(self):
        return max(0, self.cfg["per_game_tokens"] - self.tokens)

    def observe_score(self):
        value = self.score()
        if value is None:
            self.invalidate("score unavailable")
            return
        point = {"tokens": self.tokens, "score": float(value)}
        if self.score_points[-1] != point:
            self.score_points.append(point)
            self.log("quality_score", **point)

    def stop_if_exhausted(self):
        from duck_transfer.agent import GameplayRequestExpired

        if self.invalid or self.remaining == 0:
            self.stopped = True
            self.game.note_finished("quality_accounting_invalid" if self.invalid else "quality_token_budget")
            raise GameplayRequestExpired(self.game.finished_reason)

    def invalidate(self, reason):
        self.invalid.append(reason)
        self.log("quality_accounting_invalid", reason=reason, accepted_tokens=self.tokens)

    def snapshot(self):
        # A paid response can perform multiple actions; score is sampled after each action. Never interpolate
        # across a response that crosses a checkpoint, or pretend an unfinished worker reached its budget.
        terminal = self.game.finished_reason == "won"
        checkpoints = []
        for cap in self.cfg["checkpoints"]:
            candidates = [p for p in self.score_points if p["tokens"] <= cap]
            point = candidates[-1]
            checkpoints.append({"per_game_tokens": cap, "tokens": min(self.tokens, cap),
                                "score": point["score"], "score_at_tokens": point["tokens"],
                                "reached": terminal or self.tokens >= cap})
        return {"accepted_tokens": self.tokens, "remaining_tokens": self.remaining, "calls": self.calls,
                "purposes": dict(self.purposes), "accounting_valid": not self.invalid,
                "invalid_reasons": list(self.invalid), "checkpoints": checkpoints,
                **({} if self.identity["replica"] is None else self.identity)}


class QualityHTTP:
    """Account at the actual HTTP boundary, including summaries, retries, and merged replies.

    Nonstreamed lost responses have no per-request accepted-token receipt. They invalidate the run;
    replaying them or imputing their max_tokens would falsely claim exact accepted-token accounting.
    """

    def __init__(self, inner, budget, purpose):
        self.inner, self.budget, self.purpose = inner, budget, purpose

    def __getattr__(self, key):
        return getattr(self.inner, key)

    async def post(self, url, **kwargs):
        if not str(url).rstrip("/").endswith("chat/completions"):
            return await self.inner.post(url, **kwargs)
        budget = self.budget
        budget.stop_if_exhausted()
        if budget.inflight:
            budget.invalidate("overlapping generation for one game")
            budget.stop_if_exhausted()
        payload = dict(kwargs["json"])
        cap = min(int(payload["max_tokens"]), budget.remaining)
        if cap < 1:
            raise ValueError("Generation max_tokens must be positive")
        payload["max_tokens"] = cap
        # Stable per source game and request index; unrelated games finishing never change this seed stream.
        key = f'{budget.identity["seed"]}:{budget.identity["source_game_id"]}:{budget.calls}'
        payload["seed"] = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7fffffff
        payload["cache_salt"] = f"arc-public-replica-{self.inner.replica}"
        kwargs = {**kwargs, "json": payload}
        extra = kwargs.pop("record_extra", None) or {}
        purpose = self.purpose(payload)
        record = {"instance_id": self.inner.instance_id, "started_monotonic": time.monotonic(),
                  "purpose": purpose, "max_tokens": cap, "seed": payload["seed"], **extra}
        self.inner.requests.append(record)
        budget.calls += 1
        budget.inflight = True
        try:
            response = await self.inner.shared.post(url, **kwargs)
            record["status"] = response.status_code
            if response.status_code == 400:
                # Explicit validation/context rejection occurs before generation.
                record["rejected_before_generation"] = True
                return response
            try:
                usage = response.json()["usage"]
                tokens = usage["completion_tokens"]
                if type(tokens) is not int or tokens < 0 or tokens > cap:
                    raise ValueError("invalid accepted-token receipt")
            except (KeyError, TypeError, ValueError) as exc:
                budget.invalidate(f"HTTP {response.status_code}: no valid accepted-token receipt")
                raise RuntimeError("Quality generation accounting unavailable") from exc
            budget.zero_token_streak = budget.zero_token_streak + 1 if tokens == 0 else 0
            if budget.zero_token_streak >= 3:
                budget.invalidate("Three consecutive zero-token generation receipts")
            budget.tokens += tokens
            budget.purposes[purpose] = budget.purposes.get(purpose, 0) + tokens
            budget.game.clock.charge(usage.get("prompt_tokens", 0), tokens, purpose)
            record["usage"] = usage
            budget.log("quality_generation", purpose=purpose, usage=usage, accepted_tokens=budget.tokens,
                       max_tokens=cap, request_index=budget.calls - 1, seed=payload["seed"])
            return response
        except BaseException as exc:
            record["error"] = type(exc).__name__
            if not budget.invalid:
                budget.invalidate(f"Unreceipted generation: {type(exc).__name__}")
            raise
        finally:
            budget.inflight = False
            record["seconds"] = time.monotonic() - record["started_monotonic"]


def audit(report):
    issues = []
    n = len(report["games"])
    if report["scheduler"]["active_limit"] != n:
        issues.append("fixed worker count changed")
    for game in report["games"]:
        admissions = game["admissions"]
        if len(admissions) != 1 or admissions[0].get("reason") != "quality_start":
            issues.append(f'{game["game_id"]}: worker restarted or never started')
        if game["readmissions"] or game["state"] == "parked":
            issues.append(f'{game["game_id"]}: worker rotated or parked')
        if any(a.get("parked_wait_seconds", 0) > .025 for a in game["admission_timing"]["attempts"]):
            issues.append(f'{game["game_id"]}: parked during admission')
    return {"valid": not issues, "issues": issues, "warnings": [], "checks": {"games": n}}


def checkpoint_summary(games, cfg):
    rows = []
    for index, cap in enumerate(cfg["checkpoints"]):
        points = [g["checkpoints"][index] for g in games.values()]
        score = sum(p["score"] for p in points) / len(points)
        tokens = sum(p["tokens"] for p in points)
        rows.append({"per_game_tokens": cap, "mean_score": score, "accepted_tokens": tokens,
                     "score_per_million_tokens": score * 1_000_000 / tokens if tokens else None,
                     "complete": all(p["reached"] for p in points) and all(g["accounting_valid"] for g in games.values())})
    return rows


def replica_summary(games, cfg, valid):
    """Version 2: one version-1-shaped block per replica (its seed, its games by source id, its checkpoint means), so
    each is comparable with a single-roster run of that seed. {} for version 1 (its report is unchanged)."""
    if cfg.get("version") != 2:
        return {}
    blocks = []
    for replica, seed in enumerate(cfg["seeds"]):
        own = {g["source_game_id"]: g for g in games.values() if g.get("replica") == replica}
        # The start-time roster check guarantees this; a report after a paid run must still never raise.
        whole = (len(own) == len(families_of(cfg)) == sum(g.get("replica") == replica for g in games.values())
                 and all(g["seed"] == seed for g in own.values()))
        blocks.append({"replica": replica, "seed": seed, "games": own, "roster_complete": whole,
                       "accepted_tokens": sum(g["accepted_tokens"] for g in own.values()),
                       "checkpoints": [{**row, "accounting_valid": valid,
                                        "complete": row["complete"] and valid and whole}
                                       for row in (checkpoint_summary(own, cfg) if own else [])]})
    return {"replicas": blocks, "pooling": "checkpoints: equal-weight mean over every replica game"}


async def reconcile_engine(monitor, budgets, *, attempts=11, delay=1, timeout=10):
    """After workers drain, require exact equality; no guessed tolerance for missing generation."""
    receipts = sum(b.tokens for b in budgets)
    result = {}
    deadline = time.monotonic() + timeout
    for attempt in range(attempts):
        before = len(monitor.samples)
        timed_out = False
        try:
            await asyncio.wait_for(monitor.sample(), timeout=max(.001, deadline - time.monotonic()))
        except TimeoutError:
            timed_out = True
        fresh = len(monitor.samples) > before and not timed_out
        engine = monitor.summary()
        tokens = engine.get("accepted_generation_tokens")
        result = {"valid": fresh and bool(engine.get("valid")) and tokens == receipts,
                  "fresh_sample": fresh,
                  "accepted_receipts": receipts, "engine_accepted_tokens": tokens,
                  "engine_minus_receipts": None if tokens is None else tokens - receipts,
                  "tolerance_tokens": 0, "samples": attempt + 1}
        if result["valid"] or time.monotonic() >= deadline:
            break
        if attempt + 1 < attempts:
            await asyncio.sleep(min(delay, max(0, deadline - time.monotonic())))
    if not result["valid"]:
        result["reason"] = "Drained engine counter does not exactly reconcile with per-game HTTP receipts"
    return result
