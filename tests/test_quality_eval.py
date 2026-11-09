"""Fixed roster invariants and HTTP receipts, independent of response parsing/harness purposes."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from arc_infra.fleet import SessionHTTP
from arena.clock import ActiveClock, GlobalClock
from arena.quality_eval import (
    DEV10,
    FixedWorkers,
    GameBudget,
    QualityHTTP,
    settings,
    validate_study,
)
from arena.scheduler import GameRecord
from duck_transfer.agent import GameplayRequestExpired


def cfg():
    return {"version": 1, "split": "dev10", "per_game_tokens": 100, "checkpoints": [40, 100], "seed": 17}


def study():
    return {"quality_eval": cfg(), "games": list(DEV10), "generation_slots": 10, "active_limit": 10,
            "level_budget_policy": "quality-fixed", "budget_clock": "tokens", "token_budget": {"purposes": ["*"]},
            "server": {"max_num_seqs": 14}, "competition": {"preview_instances": 10}}


@pytest.mark.parametrize("change", [{"version": True}, {"seed": True}, {"per_game_tokens": 0},
                                   {"checkpoints": [100, 40]}, {"checkpoints": [40]},
                                   {"checkpoints": [40, 40, 100]}, {"typo": 1}])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        settings({**cfg(), **change})


@pytest.mark.parametrize("change", [{"games": ["cd82"] * 10}, {"games": list(DEV10[:-1]) + ["ar25"]},
                                   {"generation_slots": 9}, {"active_limit": 8}, {"priority": {"on": True}},
                                   {"level_budget_policy": "priority"}, {"parallel_samples": {"k": 2}},
                                   {"token_budget": {"purposes": ["gameplay"]}}, {"phase2_admission": "cache_capacity"}])
def test_no_legacy_scheduler_or_roster_slips_through(change):
    assert validate_study(study()) == cfg()
    with pytest.raises(ValueError):
        validate_study({**study(), **change})


def budget_fixture(score=lambda: 0):
    wall = GlobalClock(1000)
    game = GameRecord("cd82", 0, ActiveClock(wall, budget_clock="tokens", token_purposes=["*"]))
    wall.release()
    game.clock.release()
    events = []
    budget = GameBudget(cfg(), game, log=lambda kind, **row: events.append({"kind": kind, **row}), score=score)
    return budget, events


def test_all_start_no_eviction_handover_and_no_reallocation():
    budget, _ = budget_fixture()
    wall = budget.game.clock.global_clock
    games = [budget.game, GameRecord("cn04", 1, ActiveClock(wall, budget_clock="tokens", token_purposes=["*"]))]
    events = []
    scheduler = FixedWorkers(games, wall, log=events.append, cfg=cfg())
    for g in games:
        g.clock.pause()
        g.observer = scheduler
    scheduler.start_all()
    assert all(g.state == "active" and not g.clock.parked for g in games)
    games[0].note_context_drain(50000, 25000)
    games[0].report_level(1, 1)
    scheduler.tick()
    assert all(g.state == "active" and not g.clock.parked for g in games)
    games[0].note_finished("won")
    scheduler.tick()
    assert games[0].state == "finished" and games[1].state == "active"
    assert games[1].clock.budget == 100 and len(games[1].admissions) == 1


async def test_every_http_receipt_charged_even_without_record_reply_and_cap_clipped():
    budget, events = budget_fixture()
    sent = []
    purpose = ["gameplay"]

    async def post(url, **kwargs):
        sent.append(kwargs["json"])
        tokens = [35, 45, 20][len(sent) - 1]
        return httpx.Response(200, json={"usage": {"prompt_tokens": 70, "completion_tokens": tokens}})

    inner = SessionHTTP(SimpleNamespace(post=post, base_url="http://test"), 0, "cd82", [])
    client = QualityHTTP(inner, budget, lambda p: purpose[0])
    for p in ("gameplay", "fresh", "summary"):
        purpose[0] = p
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    assert [p["max_tokens"] for p in sent] == [80, 65, 20]
    assert {p["cache_salt"] for p in sent} == {"arc-public-replica-0"}
    assert budget.tokens == budget.game.clock.budget_tokens() == budget.game.clock.completion_tokens() == 100
    assert budget.purposes == {"gameplay": 35, "fresh": 45, "summary": 20}
    assert len(events) == 3 and len({p["seed"] for p in sent}) == 3
    with pytest.raises(GameplayRequestExpired):
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    assert len(sent) == 3 and budget.stopped


@pytest.mark.parametrize("error", [httpx.ReadError("lost"), asyncio.CancelledError()])
async def test_unknown_generation_invalidates_without_ambiguous_replay(error):
    budget, _ = budget_fixture()
    calls = []

    async def post(url, **kwargs):
        calls.append(1)
        raise error

    inner = SessionHTTP(SimpleNamespace(post=post, base_url="http://test"), 0, "cd82", [])
    client = QualityHTTP(inner, budget, lambda p: "summary")
    with pytest.raises(type(error)):
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    assert len(calls) == 1 and budget.tokens == 0 and budget.invalid
    with pytest.raises(GameplayRequestExpired):
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    assert len(calls) == 1


@pytest.mark.parametrize("response", [httpx.Response(200, json={}), httpx.Response(503),
                                     httpx.Response(200, json={"usage": {"completion_tokens": 81}})])
async def test_missing_invalid_or_oversized_receipt_invalidates(response):
    budget, _ = budget_fixture()

    async def post(url, **kwargs):
        return response

    inner = SessionHTTP(SimpleNamespace(post=post, base_url="http://test"), 0, "cd82", [])
    with pytest.raises(RuntimeError):
        await QualityHTTP(inner, budget, lambda p: "gameplay").post(
            "chat/completions", json={"max_tokens": 80}, timeout=50)
    assert budget.invalid


def test_checkpoint_never_credits_later_score_and_solved_game_keeps_unused_budget():
    score = [0]
    budget, _ = budget_fixture(lambda: score[0])
    budget.tokens, score[0] = 35, 5
    budget.observe_score()
    budget.tokens, score[0] = 50, 20
    budget.observe_score()
    points = budget.snapshot()["checkpoints"]
    assert points[0] == {"per_game_tokens": 40, "tokens": 40, "score": 5.0, "score_at_tokens": 35, "reached": True}
    assert not points[1]["reached"]
    budget.game.note_finished("won")
    points = budget.snapshot()["checkpoints"]
    assert points[1]["reached"] and points[1]["tokens"] == 50
    assert budget.remaining == 50


async def test_trial_quality_runtime_all_workers_finish_without_rotation(tmp_path, monkeypatch):
    from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

    from arena import trial as trial_module
    from arena.trial import Trial

    config = {**STUDY, **study(), "wall_seconds": .5}
    class ReconciledMonitor(OfflineMonitor):
        samples = []

        async def sample(self):
            self.samples.append({})

        def summary(self):
            return {"valid": True, "preemptions": 0, "accepted_generation_tokens": 1000}

    monkeypatch.setattr(trial_module, "GpuMonitor", ReconciledMonitor)
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda *a: 10_000_000)
    workers = []

    class Worker(FakeWorker):
        async def run(self):
            self.game.ready.set()
            await self.game.clock.wait_active()
            assert all(g.game.state == "active" for g in workers)
            self.game.note_context_drain(60000, 30000)
            self.game.report_level(1, 0)
            self.quality_budget.tokens = 100
            self.game.clock.charge(40, 100, "fresh")
            try:
                self.quality_budget.stop_if_exhausted()
            except GameplayRequestExpired:
                pass
            return self.result()

    async def agents(self, records, gate, window, wall, card, arc):
        for game in records:
            worker = Worker(game)
            worker.game_id = game.game_id
            worker.quality_budget = GameBudget(cfg(), game, log=lambda *a, **k: None, score=lambda: 0)
            workers.append(worker)
        self.config_dict = {}

        async def close():
            pass

        return workers, lambda: [w.result() for w in workers], close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    result = await Trial(config, {"name": "quality-fixed", "kind": "duck-transfer"}, source=".", games_dir="games",
                         output=tmp_path, server_dir=tmp_path, base_url="http://test", arcade=FakeArcade(DEV10),
                         log=lambda *a, **k: None).run()
    assert result["status"] == "complete", result
    assert result["quality_eval"]["budget_complete"] and result["quality_eval"]["accounting_valid"]
    assert len(result["games"]) == 10 and all(g["state"] == "finished" for g in result["games"])
    assert all(len(g["admissions"]) == 1 for g in result["games"])
    assert result["scheduler_audit"]["valid"]


async def test_paid_final_reply_can_act_before_next_request_stops(tmp_path):
    from test_arena_duck import arena_agent, text_reply

    from arena.duck import ArenaTransferDuckAgent

    class Transport:
        async def post(self, url, **kwargs):
            return text_reply("ready")

    agent, _, game, _ = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    events = []
    agent.quality_budget = GameBudget(cfg(), game, log=lambda *a, **k: events.append(a), score=lambda: 0)
    agent.quality_budget.tokens = 100
    runtime = agent.runtime_message(record=False)["content"]
    assert "game_budget_remaining_tokens" in runtime and "level_budget" not in runtime
    # Exhaustion does not hide remaining wall time until the next model request / analyzer turn.
    assert agent.remaining() > 0
    with pytest.raises(GameplayRequestExpired):
        agent.quality_budget.stop_if_exhausted()
    assert agent.remaining() == 0 and game.finished_reason == "quality_token_budget"


async def test_final_paid_tool_call_executes_and_charges_once(tmp_path):
    from conftest import tool_reply
    from test_arena_duck import arena_agent

    from arena.duck import ArenaTransferDuckAgent

    class Transport:
        calls = 0
        base_url = "http://test"

        async def post(self, url, **kwargs):
            self.calls += 1
            assert kwargs["json"]["max_tokens"] == 100
            return tool_reply("action([{'action':'UP'}])", call_id="act",
                              usage={"completion_tokens": 100, "prompt_tokens": 7000})

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.quality_budget = GameBudget(cfg(), game, log=agent.workspace.append, score=lambda: 100)
    inner = SessionHTTP(transport, 0, game.game_id, [])
    agent.client.client = QualityHTTP(inner, agent.quality_budget, lambda p: "gameplay")
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and transport.calls == 1
    assert game.clock.completion_tokens() == 100
    assert agent.quality_budget.tokens == 100 and not agent.quality_budget.invalid


def test_real_gateway_preserves_source_ids_and_fixed_seed(tmp_path, monkeypatch):
    from pathlib import Path

    from arena.competition import open_arcade
    from arena.quality_eval import SeededArcade

    games_dir = Path("./environment_files")
    if not games_dir.is_dir():
        pytest.skip("public games unavailable")
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    config = study()
    config["competition"]["preview_live_score"] = True
    with open_arcade(config, games_dir, tmp_path) as (resolved, arc):
        assert set(g.split("-", 1)[0] for g in resolved["games"]) == set(DEV10)
        assert len(resolved["games"]) == 10 and validate_study(resolved)
        card = arc.open_scorecard(tags=["quality-cpu-test"])
        env = SeededArcade(arc, 17).make(resolved["games"][0], scorecard_id=card)
        assert env is not None
        assert arc.peek_score(card) is not None


async def test_engine_reconciliation_requires_exact_drained_receipts():
    from arena.quality_eval import reconcile_engine

    budget, _ = budget_fixture()
    budget.tokens = 100

    class Monitor:
        values = iter([90, 99, 100])
        samples = []

        async def sample(self):
            self.tokens = next(self.values)
            self.samples.append({})

        def summary(self):
            return {"valid": True, "accepted_generation_tokens": self.tokens}

    result = await reconcile_engine(Monitor(), [budget], attempts=3, delay=0)
    assert result["valid"] and result["samples"] == 3 and result["tolerance_tokens"] == 0
    monitor = Monitor()
    monitor.values = iter([101, 101])
    result = await reconcile_engine(monitor, [budget], attempts=2, delay=0)
    assert not result["valid"] and result["engine_minus_receipts"] == 1


async def test_zero_token_generation_cannot_loop_indefinitely():
    budget, _ = budget_fixture()
    calls = []

    async def post(url, **kwargs):
        calls.append(1)
        return httpx.Response(200, json={"usage": {"completion_tokens": 0}})

    inner = SessionHTTP(SimpleNamespace(post=post, base_url="http://test"), 0, "cd82", [])
    client = QualityHTTP(inner, budget, lambda p: "gameplay")
    for _ in range(3):
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    with pytest.raises(GameplayRequestExpired):
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    assert budget.tokens == 0 and budget.invalid and len(calls) == 3


async def test_matching_stale_engine_counter_is_not_reconciled():
    from arena.quality_eval import reconcile_engine

    budget, _ = budget_fixture()
    budget.tokens = 100

    class FailedMonitor:
        samples = [{}]

        async def sample(self):
            pass  # Monitor.sample records HTTP errors and retains its last successful counter.

        def summary(self):
            return {"valid": True, "accepted_generation_tokens": 100}

    result = await reconcile_engine(FailedMonitor(), [budget], attempts=1, delay=0)
    assert not result["valid"] and not result["fresh_sample"] and result["engine_minus_receipts"] == 0


def test_quality_server_capacity_passes_pinned_flash_guard(tmp_path):
    from arc_infra.serving import load_recipe

    for slots in (16, 25):
        with pytest.raises(FileNotFoundError, match="Expected one mounted dataset"):
            load_recipe(tmp_path / "working", input_root=tmp_path / "no-datasets", generation_slots=slots, mtp_tokens=1)


# ----------------------------------------------------------------------------- version 2: replicated roster
def cfg2(replicas=2):
    return {"version": 2, "split": "dev10", "per_game_tokens": 100, "checkpoints": [40, 100],
            "replicas": replicas, "seeds": list(range(replicas))}


def clones(replicas=2):
    return [f"c{i:03d}" for i in range(10 * replicas)]


def study2(replicas=2, resolved=False):
    from arena.quality_eval import roster

    base = {**study(), "quality_eval": cfg2(replicas), "active_limit": 10 * replicas, "generation_slots": 12,
            "competition": {"preview_instances": 10 * replicas}}
    if resolved:
        base.update(games=clones(replicas), quality_roster=roster(cfg2(replicas), DEV10, clones(replicas)))
    return base


@pytest.mark.parametrize("change", [{"replicas": 1, "seeds": [0]}, {"replicas": True, "seeds": [0]},
                                   {"seeds": [0, 0]}, {"seeds": [0]}, {"seeds": [0, -1]}, {"seeds": [0, True]},
                                   {"seed": 0}, {"checkpoints": [40]}, {"per_game_tokens": 0}, {"split": "all"}])
def test_version2_invalid_config(change):
    assert settings(cfg2()) == cfg2()
    with pytest.raises(ValueError):
        settings({**cfg2(), **change})


def test_version2_shard_plays_every_replica_of_some_families():
    from arena.quality_eval import SPLITS, families_of, replica_summary, roster, validate_study, worker_count

    shard = {**cfg2(), "split": "heldout15", "replicas": 4, "seeds": [0, 1, 2, 3], "families": ["ar25", "bp35", "dc22"]}
    assert settings(shard) == shard and families_of(shard) == ("ar25", "bp35", "dc22") and worker_count(shard) == 12
    assert families_of({**cfg2(), "split": "heldout15"}) == SPLITS["heldout15"]
    for bad in ([], ["ar25", "ar25"], ["cd82"], list(SPLITS["heldout15"]), "ar25"):
        with pytest.raises(ValueError):
            settings({**shard, "families": bad})
    sources = ["ar25-a", "bp35-b", "dc22-c"]
    rows = roster(shard, sources, [f"c{i:03d}" for i in range(12)])
    assert [(r["source_game_id"], r["replica"], r["seed"]) for r in rows[3:5]] == [("ar25-a", 1, 1), ("bp35-b", 1, 1)]
    study = {"games": sources, "quality_eval": shard, "active_limit": 12, "generation_slots": 12,
             "level_budget_policy": "quality-fixed", "budget_clock": "tokens", "token_budget": {"purposes": ["*"]},
             "server": {"max_num_seqs": 25}, "competition": {"preview_instances": 12}}
    assert validate_study(study) == shard
    for change in ({"games": sources[:2]}, {"games": ["ar25-a", "bp35-b", "g50t-d"]}, {"active_limit": 60}):
        with pytest.raises(ValueError):
            validate_study({**study, **change})
    games = {r["game_id"]: {**r, "accepted_tokens": 10, "accounting_valid": True,
                            "checkpoints": [{"per_game_tokens": p, "tokens": p, "score": 1.0, "score_at_tokens": 1,
                                             "reached": True} for p in shard["checkpoints"]]} for r in rows}
    assert all(b["roster_complete"] and len(b["games"]) == 3 for b in replica_summary(games, shard, True)["replicas"])


def test_version2_roster_follows_gateway_clone_order():
    from arena.quality_eval import gateway_runs, roster, worker_count, worker_identity

    assert gateway_runs(cfg()) is None and roster(cfg(), DEV10, DEV10) is None and worker_count(cfg()) == 10
    assert gateway_runs(cfg2(4)) == worker_count(cfg2(4)) == 40
    rows = roster({**cfg2(), "seeds": [5, 9]}, DEV10, clones())
    assert rows[0] == {"game_id": "c000", "source_game_id": "cd82", "replica": 0, "seed": 5}
    assert rows[13] == {"game_id": "c013", "source_game_id": DEV10[3], "replica": 1, "seed": 9}
    assert worker_identity(cfg2(), study2(resolved=True), "c013") == {
        "seed": 1, "source_game_id": DEV10[3], "replica": 1}
    assert worker_identity(cfg(), study(), "cd82") == {"seed": 17, "source_game_id": "cd82", "replica": None}
    with pytest.raises(ValueError):
        roster(cfg2(), DEV10, clones()[:-1])
    with pytest.raises(ValueError):
        worker_identity(cfg2(), study2(resolved=True), "c999")


@pytest.mark.parametrize("resolved", [False, True])
def test_version2_study_accepts_bundle_and_resolved_forms(resolved):
    assert validate_study(study2(resolved=resolved)) == cfg2()
    assert validate_study({**study2(resolved=resolved), "generation_slots": 20,
                           "server": {"max_num_seqs": 20}}) == cfg2()


@pytest.mark.parametrize("change", [{"active_limit": 10}, {"generation_slots": 0}, {"generation_slots": 21},
                                   {"generation_slots": 15}, {"competition": {"preview_instances": 10}},
                                   {"require_focus_windows_fit": True}, {"focus_oversubscription": {"focus_slots": 20}},
                                   {"priority": {"on": True}}, {"level_budget_policy": "priority"},
                                   {"token_budget": {"purposes": ["gameplay"]}},
                                   {"games": list(DEV10[:-1]) + ["ar25"]}, {"games": clones()}])
def test_version2_no_scheduler_capacity_or_roster_slips_through(change):
    with pytest.raises(ValueError):
        validate_study({**study2(), **change})  # server.max_num_seqs is 14: 15 slots cannot be admitted


def test_version2_resolved_roster_must_be_exact():
    good = study2(resolved=True)
    swapped = [dict(r) for r in good["quality_roster"]]
    swapped[0]["seed"], swapped[10]["seed"] = swapped[10]["seed"], swapped[0]["seed"]
    foreign = [dict(r) for r in good["quality_roster"]]
    foreign[3]["source_game_id"] = foreign[13]["source_game_id"] = "ar25"
    for rows in (swapped, foreign, good["quality_roster"][:-1], good["quality_roster"][::-1]):
        with pytest.raises(ValueError):
            validate_study({**good, "quality_roster": rows})
    with pytest.raises(ValueError):
        validate_study({**good, "games": clones()[::-1]})


async def request_seeds(config, game_id, identity, calls=3):
    wall = GlobalClock(1000)
    game = GameRecord(game_id, 0, ActiveClock(wall, budget_clock="tokens", token_purposes=["*"]))
    wall.release()
    game.clock.release()
    budget = GameBudget(config, game, log=lambda kind, **row: None, score=lambda: 0, identity=identity)
    sent = []

    async def post(url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(200, json={"usage": {"prompt_tokens": 7, "completion_tokens": 5}})

    client = QualityHTTP(SessionHTTP(SimpleNamespace(post=post, base_url="http://test"), 0, game_id, []), budget,
                         lambda p: "gameplay")
    for _ in range(calls):
        await client.post("chat/completions", json={"max_tokens": 80}, timeout=50)
    return [p["seed"] for p in sent]


async def test_version2_replica_repeats_the_single_roster_seed_stream():
    from arena.quality_eval import worker_identity

    resolved = study2(resolved=True)
    streams = {}
    for clone in ("c003", "c013"):
        who = worker_identity(cfg2(), resolved, clone)
        streams[who["replica"]] = await request_seeds(cfg2(), clone, who)
        # the version 1 run with this replica's seed, on the source game under its own id
        assert streams[who["replica"]] == await request_seeds({**cfg(), "seed": who["seed"]}, DEV10[3], None)
    assert not set(streams[0]) & set(streams[1])
    other_game = await request_seeds(cfg2(), "c004", worker_identity(cfg2(), resolved, "c004"))
    assert not set(streams[0]) & set(other_game)


def test_version2_reports_each_replica_and_the_pooled_mean():
    from arena.quality_eval import checkpoint_summary, replica_summary, roster

    rows = roster(cfg2(), DEV10, clones())
    games = {}
    for i, row in enumerate(rows):
        score = float(i)
        games[row["game_id"]] = {
            "accepted_tokens": 100, "accounting_valid": True, "source_game_id": row["source_game_id"],
            "replica": row["replica"], "seed": row["seed"],
            "checkpoints": [{"per_game_tokens": 40, "tokens": 40, "score": score / 2, "reached": True},
                            {"per_game_tokens": 100, "tokens": 100, "score": score, "reached": row["replica"] == 0}]}
    assert replica_summary({}, cfg(), True) == {}
    out = replica_summary(games, cfg2(), True)
    blocks = out["replicas"]
    assert [b["seed"] for b in blocks] == [0, 1] and [b["accepted_tokens"] for b in blocks] == [1000, 1000]
    assert list(blocks[1]["games"]) == list(DEV10) and blocks[1]["games"]["cd82"]["checkpoints"][1]["score"] == 10.0
    assert [b["checkpoints"][1]["mean_score"] for b in blocks] == [4.5, 14.5]
    assert [b["checkpoints"][1]["complete"] for b in blocks] == [True, False]
    pooled = checkpoint_summary(games, cfg2())
    assert pooled[1]["mean_score"] == 9.5 == sum(b["checkpoints"][1]["mean_score"] for b in blocks) / 2
    assert pooled[1]["accepted_tokens"] == 2000 and not pooled[1]["complete"]
    assert not replica_summary(games, cfg2(), False)["replicas"][0]["checkpoints"][0]["complete"]
    assert all(b["roster_complete"] for b in blocks)
    games.pop("c019")  # cannot happen after the start-time roster check; the report still must not raise
    short = replica_summary(games, cfg2(), True)["replicas"]
    assert short[0]["roster_complete"] and not short[1]["roster_complete"]
    assert short[0]["checkpoints"][0]["complete"] and not short[1]["checkpoints"][0]["complete"]


async def test_version2_trial_runs_every_replica_worker_without_rotation(tmp_path, monkeypatch):
    from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

    from arena import trial as trial_module
    from arena.quality_eval import worker_identity
    from arena.trial import Trial

    config = {**STUDY, **study2(resolved=True), "wall_seconds": .5}

    class ReconciledMonitor(OfflineMonitor):
        samples = []

        async def sample(self):
            self.samples.append({})

        def summary(self):
            return {"valid": True, "preemptions": 0, "accepted_generation_tokens": 2000}

    monkeypatch.setattr(trial_module, "GpuMonitor", ReconciledMonitor)
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda *a: 10_000_000)
    workers = []

    class Worker(FakeWorker):
        async def run(self):
            self.game.ready.set()
            await self.game.clock.wait_active()
            assert len(workers) == 20 and all(g.game.state == "active" for g in workers)
            self.quality_budget.tokens = 100
            self.game.clock.charge(40, 100, "fresh")
            try:
                self.quality_budget.stop_if_exhausted()
            except GameplayRequestExpired:
                pass
            return self.result()

    async def agents(self, records, gate, window, wall, card, arc):
        assert gate.slots == 12
        for game in records:
            worker = Worker(game)
            worker.game_id = game.game_id
            worker.quality_budget = GameBudget(self.quality, game, log=lambda *a, **k: None, score=lambda: 0,
                                               identity=worker_identity(self.quality, self.study, game.game_id))
            workers.append(worker)
        self.config_dict = {}

        async def close():
            pass

        return workers, lambda: [w.result() for w in workers], close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    result = await Trial(config, {"name": "quality-fixed", "kind": "duck-transfer"}, source=".", games_dir="games",
                         output=tmp_path, server_dir=tmp_path, base_url="http://test", arcade=FakeArcade(clones()),
                         log=lambda *a, **k: None).run()
    quality = result["quality_eval"]
    assert result["status"] == "complete", result
    assert quality["budget_complete"] and quality["accounting_valid"] and quality["accepted_tokens"] == 2000
    assert len(result["games"]) == 20 and all(len(g["admissions"]) == 1 for g in result["games"])
    assert result["scheduler_audit"]["valid"] and result["scheduler"]["active_limit"] == 20
    assert [b["seed"] for b in quality["replicas"]] == [0, 1]
    assert all(list(b["games"]) == list(DEV10) and b["accepted_tokens"] == 1000 for b in quality["replicas"])
    # FakeArcade scores clone i at 10 * i: replica means 45 / 145, pooled 95.
    assert [b["checkpoints"][-1]["mean_score"] for b in quality["replicas"]] == [45.0, 145.0]
    assert quality["checkpoints"][-1]["mean_score"] == 95.0 and quality["checkpoints"][-1]["complete"]
    assert quality["games"]["c013"]["source_game_id"] == DEV10[3] and quality["games"]["c013"]["seed"] == 1


async def test_version2_unresolved_study_cannot_start_a_trial(tmp_path):
    from test_arena_trial import STUDY, FakeArcade

    from arena.trial import Trial

    with pytest.raises((RuntimeError, ValueError)):
        await Trial({**STUDY, **study2(), "wall_seconds": .5}, {"name": "quality-fixed", "kind": "duck-transfer"},
                    source=".", games_dir="games", output=tmp_path, server_dir=tmp_path, base_url="http://test",
                    arcade=FakeArcade(DEV10), log=lambda *a, **k: None).run()


async def test_version2_real_agent_wiring_seeds_each_replica(tmp_path, monkeypatch):
    """The un-mocked ``Trial.duck_agents``: environment seed and request-seed identity per replica worker."""
    import importlib

    from test_arena_trial import STUDY

    from arena import trial as trial_module
    from arena.gate import KVGate
    from arena.quality_eval import SeededArcade
    from arena.trial import Trial

    # duck_agents imports these by name at call time: patch the sys.modules entries (another test may have re-imported
    # arena.duck, leaving the package attribute on an older module object).
    duck_module = importlib.import_module("arena.duck")
    loader = importlib.import_module("duck_transfer.loader")

    built = []

    class Agent:
        request_purpose = None

        def __init__(self, game_id, arcade, card, client, cfg, folder, python, deadline):
            self.game_id, self.arcade, self.client = game_id, arcade, client
            self.workspace = SimpleNamespace(append=lambda *a, **k: None)
            built.append(self)

        def attach(self, game, **kwargs):
            self.game = game

        def result(self):
            return {}

    monkeypatch.setattr(duck_module, "ArenaTransferDuckAgent", Agent)
    monkeypatch.setattr(loader, "configure", lambda *a, **k: None)
    monkeypatch.setattr(trial_module, "duck_config", lambda *a, **k: SimpleNamespace(
        base_url="http://test", to_dict=lambda: {}, request_timeout=5, model="m"))
    config = {**STUDY, **study2(resolved=True), "wall_seconds": .5}
    trial = Trial(config, {"name": "quality-fixed", "kind": "duck-transfer"}, source=".", games_dir="games",
                  output=tmp_path, server_dir=tmp_path, base_url="http://test", arcade=object(),
                  log=lambda *a, **k: None)
    trial.cap = 1000
    wall = GlobalClock(1000)
    records = [GameRecord(g, i, ActiveClock(wall, budget_clock="tokens", token_purposes=["*"]))
               for i, g in enumerate(clones())]
    arc = SimpleNamespace(make=lambda *a, **k: k)
    agents, _, close = await trial.duck_agents(records, KVGate(10_000_000, slots=12), 4096, wall, "card", arc)
    await close()
    assert [a.game_id for a in agents] == clones() and len(built) == 20
    assert all(isinstance(a.arcade, SeededArcade) for a in agents)
    assert [a.arcade.seed for a in agents] == [0] * 10 + [1] * 10
    assert agents[13].arcade.make("c013")["seed"] == 1
    assert agents[13].quality_budget.identity == {"seed": 1, "source_game_id": DEV10[3], "replica": 1}
    assert all(isinstance(a.client.client, QualityHTTP) for a in agents)


def test_version2_real_gateway_clones_each_source_per_replica(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from arena.competition import open_arcade
    from arena.quality_eval import SeededArcade, worker_identity

    games_dir = Path("./environment_files")
    if not games_dir.is_dir():
        pytest.skip("public games unavailable")
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    config = study2()
    config["competition"]["preview_live_score"] = True
    with open_arcade(config, games_dir, tmp_path) as (resolved, arc):
        assert len(resolved["games"]) == 20 and len(set(resolved["games"])) == 20
        assert validate_study(resolved) == cfg2()
        rows = resolved["quality_roster"]
        assert [r["source_game_id"] for r in rows] == list(DEV10) * 2
        assert [r["replica"] for r in rows] == [0] * 10 + [1] * 10
        # the controller's provenance file names the same sources
        public = json.loads((tmp_path / "public-roster.json").read_text())
        assert [(p["game_id"], p["source_game_id"]) for p in public] == [(r["game_id"], r["source_game_id"]) for r in rows]
        card = arc.open_scorecard(tags=["quality-cpu-test"])
        clone = rows[13]["game_id"]
        env = SeededArcade(arc, worker_identity(cfg2(), resolved, clone)["seed"]).make(clone, scorecard_id=card)
        assert env is not None
        assert {e.id for e in arc.peek_score(card).environments} >= {clone}
