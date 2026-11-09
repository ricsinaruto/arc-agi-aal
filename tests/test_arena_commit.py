"""Momentum commitment with a preemptible lane (fixed-2x-commit): focus slots on clears, stall-cap
demotion, promotion replacement, the level-1 floor lane order, the estimate-based lane admission,
request priorities, preemption attribution and the priority-policy server."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_arena_hazard_index import build, charged

from arena.commit import (
    COMMIT_DEFAULTS,
    COUNTER,
    CommitLane,
    PreemptionAttribution,
    commit_settings,
    server_scheduling_policy,
)
from arena.gate import BackgroundLane, GateSlot, KVGate
from arena.scheduler import Scheduler
from arena.variants import VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
WINDOW, CAP = 100_000, 16_000


def steady(count=6, **commit):
    """Commit scheduler past the opening: g0 and g1 cleared level 1 and hold the focus slots."""
    now, wall, games, sched = build("commit", count=count, commit=commit or {})
    for g in games:
        charged(now, g, 10, 3000)
    games[0].report_level(1, wall.elapsed())
    games[1].report_level(1, wall.elapsed())
    sched.tick()
    assert [g.state for g in games[:2]] == ["active", "active"] and not any(g.lane for g in games[:2])
    assert all(g.state == "parked" and g.lane for g in games[2:])
    return now, wall, games, sched


def commit_gate(games, now, *, capacity=10**7, pool=100_000, **settings):
    log = []
    lane = CommitLane(games, window=WINDOW, cap=CAP, pool_units=pool, **settings)
    return KVGate(capacity, slots=25, log=log.append, now=now, lane=lane), log


def test_a_clear_commits_the_game_and_a_free_slot_is_filled_by_levels_then_fewest_tokens():
    now, wall, games, sched = steady()
    g0, g1, g2, g3, g4, g5 = games
    assert all(e["budget"] is None for g in (g0, g1) for e in g.admissions if e["reason"] == "opening_switch")
    charged(now, g0, 20, 5000)  # g0: 5,000 tokens on level 2; g1: 2,000
    charged(now, g1, 20, 2000)
    charged(now, g3, 20, 4000)  # a lane request clears g3's level 1
    g3.report_level(1, wall.elapsed())
    # Promoted at once (not at the next tick), replacing the focus game with the most tokens on its level.
    assert g3.state == "active" and not g3.lane and g3.admissions[-1]["reason"] == "lane_clear"
    assert g0.state == "parked" and g0.lane and not g0.stalled_level and g0.admissions[-1]["reason"] == "commit_replaced"
    assert sched.commit_promotions[-1] == {"game": "g3", "reason": "lane_clear", "levels": 1, "tokens_on_level": 0,
                                           "wall_seconds": round(wall.elapsed(), 3), "replaced": "g0"}
    # A finished focus game frees a slot: no clear to promote, so the lane game with the most levels
    # (g0, displaced but not stalled) takes it before the level-0 games.
    g1.note_finished("won")
    sched.tick()
    assert g1.state == "finished" and g0.state == "active" and sched.commit_promotions[-1]["reason"] == "fill"
    # With a free slot, a lane clear takes it without displacing anyone.
    g3.note_finished("won")
    sched.tick()  # fill: level-0 games, fewest tokens on the level (g2, g4, g5 tie at 3,000: roster order)
    assert g2.state == "active" and sched.commit_promotions[-1]["reason"] == "fill"
    # A game already past its cap is not filled (it would be demoted at the next tick): the slot stays
    # empty while every other waiting game is past its cap too; the lane keeps serving them.
    g2.note_finished("won")
    charged(now, g4, 60, 30_000)
    charged(now, g5, 60, 30_000)
    sched.tick()
    assert (g4.state, g5.state) == ("parked", "parked") and len(sched.focus()) == 1
    summary = sched.summary()["commit"]
    assert summary["promotion_reasons"] == {"lane_clear": 1, "fill": 2}
    assert summary["level1_floor"]["uncleared_at_or_above_floor"] == ["g4", "g5"]
    assert summary["settings"] == COMMIT_DEFAULTS and sched.active_limit == 2


def test_stall_cap_is_the_clamped_multiple_of_the_median_cleared_level():
    now, wall, games, sched = steady()
    g0 = games[0]
    for level_tokens, cap in (([4000], 25_000), ([30_000], 60_000), ([12_000, 14_000], 32_500),
                              ([0, 12_000, 14_000], 32_500), ([], 25_000)):
        g0.level_tokens = level_tokens
        assert sched.stall_cap(g0) == cap
    custom = Scheduler(games, wall, budget_policy="commit", backlog_policy="readmissions",
                       commit={"stall_multiple": 3, "stall_min_tokens": 10_000, "stall_max_tokens": 20_000})
    g0.level_tokens = [5000]
    assert custom.stall_cap(g0) == 15_000


def test_a_focus_game_past_its_cap_moves_to_the_lane_and_returns_only_with_a_clear():
    now, wall, games, sched = steady()
    g0, g1, g2 = games[:3]
    assert g0.level_tokens == [3000] and sched.stall_cap(g0) == 25_000
    charged(now, g0, 60, 25_000)
    sched.tick()
    assert g0.state == "active"  # at the cap, not past it
    charged(now, g0, 1, 1)
    sched.tick()
    assert g0.state == "parked" and g0.lane and g0.stalled_level and g0.admissions[-1]["reason"] == "stalled_level"
    assert sched.commit_demotions == [{"game": "g0", "levels": 1, "tokens_on_level": 25_001, "stall_cap": 25_000,
                                       "wall_seconds": round(wall.elapsed(), 3)}]
    # The freed slot is filled in the same tick, never by the demoted game (it has the most levels).
    assert g2.state == "active" and sched.commit_promotions[-1] == {
        "game": "g2", "reason": "fill", "levels": 0, "tokens_on_level": 3000,
        "wall_seconds": round(wall.elapsed(), 3), "replaced": None}
    sched.tick()
    assert g0.state == "parked"
    # A clear in the lane resets the flag and the level tokens and promotes it again.
    charged(now, g0, 5, 2000)
    g0.report_level(2, wall.elapsed())
    assert g0.state == "active" and not g0.stalled_level and g0.tokens_on_level() == 0
    assert g0.level_tokens == [3000, 27_001]
    assert sched.stall_cap(g0) == 37_501  # 2.5 x median(3,000, 27,001), truncated


def test_a_lane_clear_replaces_a_focus_game_past_its_cap_before_the_one_with_most_tokens():
    now, wall, games, sched = steady()
    g0, g1, g2, g3 = games[:4]
    g1.level_tokens = [30_000]  # g1's cap: 60K
    charged(now, g0, 30, 26_000)  # past g0's 25K cap (the tick has not demoted it yet)
    charged(now, g1, 30, 40_000)  # more tokens, but within its cap
    charged(now, g3, 10, 1000)
    g3.report_level(1, wall.elapsed())
    assert g3.state == "active" and g1.state == "active"
    assert g0.state == "parked" and g0.stalled_level and g0.admissions[-1]["reason"] == "stalled_level"
    assert sched.commit_demotions[-1]["game"] == "g0" and sched.commit_promotions[-1]["replaced"] == "g0"
    # Without a stalled focus game, the one with the most tokens on its level goes (ties: later roster).
    charged(now, g2, 10, 1000)
    g2.report_level(1, wall.elapsed())
    assert g1.state == "parked" and not g1.stalled_level and g1.admissions[-1]["reason"] == "commit_replaced"
    assert {g.game_id for g in sched.focus()} == {"g2", "g3"}


def test_commit_queue_keeps_unstalled_focus_games_and_fills_freed_slots_with_queued_clears():
    now, wall, games, sched = steady(replace_unstalled=False)
    g0, g1, g2, g3 = games[:4]
    charged(now, g0, 10, 5000)
    charged(now, g1, 10, 5000)
    charged(now, g3, 10, 1000)
    g3.report_level(1, wall.elapsed())
    charged(now, g2, 10, 1000)
    g2.report_level(1, wall.elapsed())
    # Both focus games are within their caps: the clears queue (earliest first) and nobody is replaced.
    assert {g.game_id for g in sched.focus()} == {"g0", "g1"}
    assert [g.game_id for g in sched.commit_queue] == ["g3", "g2"]
    assert [p["reason"] for p in sched.commit_promotions[-2:]] == ["queued", "queued"]
    # g0 runs past its 25K cap: the tick demotes it and the earliest queued clear takes the slot.
    charged(now, g0, 30, 26_000)
    sched.tick()
    assert g0.state == "parked" and g0.stalled_level
    assert g3.state == "active" and sched.commit_promotions[-1]["reason"] == "lane_clear_queued"
    assert [g.game_id for g in sched.commit_queue] == ["g2"]
    # A stalled focus game is still replaced at once by a new lane clear.
    charged(now, g1, 30, 26_000)
    charged(now, games[4], 10, 1000)
    games[4].report_level(1, wall.elapsed())
    assert g1.state == "parked" and g1.stalled_level and games[4].state == "active"


def test_commit_queue_variant_differs_from_commit_only_in_the_setting():
    base, queue = apply_variant(CONTROL, "fixed-2x-commit"), apply_variant(CONTROL, "fixed-2x-commit-queue")
    assert "replace_unstalled" not in base["commit"] and queue["commit"]["replace_unstalled"] is False
    assert {k: v for k, v in queue["commit"].items() if k != "replace_unstalled"} == base["commit"]
    assert commit_settings(queue["commit"])["replace_unstalled"] is False
    with pytest.raises(ValueError, match="replace_unstalled"):
        commit_settings({"replace_unstalled": "no"})


async def test_lane_order_is_the_level1_floor_then_deep_games_then_level1_past_the_floor():
    now, wall, games, sched = steady(count=8)
    g2, g3, g4, g5, g6, g7 = games[2:]
    # tokens on the current level: g2 L0 30K (past the floor), g3 L0 5K, g4 L2 40K, g5 L1 10K,
    # g6 L0 24,999, g7 L0 25K (at the floor: past it)
    for g, tokens in ((g2, 27_000), (g3, 2000), (g6, 21_999), (g7, 22_000)):
        charged(now, g, 10, tokens)
    for g, extra, levels in ((g4, 1000, 2), (g5, 1000, 1)):
        charged(now, g, 10, extra)
        g.level_started_tokens, g.levels = g.clock.completion_tokens(), levels
    charged(now, g4, 10, 40_000)
    charged(now, g5, 10, 10_000)
    lane = CommitLane(games, window=WINDOW, cap=CAP, pool_units=100_000)
    ranked = sorted((g.game_id for g in games[2:]), key=lane.order)
    assert ranked == ["g3", "g6", "g5", "g4", "g7", "g2"]
    # The gate serves queued lane requests in that order (one at a time here).
    gate, log = commit_gate(games, now, pool=10**6, lane_max_inflight=1)
    first = await gate.acquire("g2", 1000, prompt=500)
    waiters = [asyncio.create_task(gate.acquire(g.game_id, 1000, prompt=500)) for g in (g7, g4, g5, g6, g3)]
    await asyncio.sleep(0)
    gate.release(first)
    admitted = []
    for _ in waiters:
        await asyncio.sleep(0)
        done = [w for w in waiters if w.done() and w not in admitted]
        assert len(done) == 1
        admitted.append(done[0])
        gate.release(done[0].result())
    assert [e["game"] for e in log if e["event"] == "admit"] == ["g2", "g3", "g6", "g5", "g4", "g7"]  # g2 first: alone


async def test_lane_admission_on_the_occupancy_estimate_and_focus_ignores_lane_reservations():
    now, wall, games, sched = steady()
    g0, g1, g2, g3, g4, g5 = games
    # Pool 100K tokens, lane budget 92K; capacity for focus requests 90K (the plain full-length rule).
    gate, log = commit_gate(games, now, capacity=90_000, pool=100_000, lane_max_inflight=2)
    lane = gate.lane
    assert lane.budget_units() == 92_000 and lane.headroom() == 6000
    focus = await gate.acquire("g0", 30_000 + CAP, prompt=30_000)
    g1.latest_prompt_tokens = 20_000  # g1 is between requests: its next prompt + headroom is reserved
    assert gate.lane_estimate() == 30_000 + 26_000
    # 56,000 + 30,000 + 6,000 = 92,000 <= 92,000: admitted.
    first = await asyncio.wait_for(gate.acquire("g2", 30_000 + CAP, prompt=30_000), 1)
    assert first.lane and gate.lane_reserved == 46_000
    # 86,000 + 1,000 + 6,000 > 92,000: waits.
    second = asyncio.create_task(gate.acquire("g3", 1000 + CAP, prompt=1000))
    await asyncio.sleep(0)
    assert not second.done()
    # Ten seconds on, each in-flight request holds 1,200 generated tokens more (120 tok/s).
    now.advance(10)
    assert gate.lane_estimate() == 31_200 + 31_200 + 26_000
    # The focus request ends; g0's next prompt shrank after a compaction: 31,200 + 26,000 + 16,000 + 7,000 fits.
    gate.release(focus)
    g0.latest_prompt_tokens = 10_000
    gate.refresh()
    second = await asyncio.wait_for(second, 1)
    assert second.lane and gate.lane_inflight == 2
    # A third lane request waits on lane_max_inflight alone.
    third = asyncio.create_task(gate.acquire("g4", 10 + CAP, prompt=10))
    await asyncio.sleep(0)
    assert not third.done()
    # Focus requests fit against the focus reservations only (lane reservations: 63,000): no wait.
    big = await asyncio.wait_for(gate.acquire("g0", 80_000, prompt=64_000), 1)
    assert not big.lane and gate.reserved == 80_000 + 46_000 + 17_000 > gate.capacity
    stats = gate.snapshot()["lane"]
    assert stats["focus_wait_seconds"] == 0 and stats["focus_over_estimate"] == 1
    # ... while the plain rule still binds focus requests among themselves.
    blocked = asyncio.create_task(gate.acquire("g1", 20_000, prompt=4000))
    await asyncio.sleep(0)
    assert not blocked.done()
    for r in (big, first, second):
        gate.release(r)
    gate.release(await asyncio.wait_for(blocked, 1))
    gate.release(await asyncio.wait_for(third, 1))
    assert gate.reserved == gate.inflight == gate.lane_reserved == 0 and not gate.flights


def test_headroom_is_the_run_quantile_of_observed_completions():
    now, wall, games, sched = steady()
    lane = CommitLane(games, window=WINDOW, cap=CAP, pool_units=100_000)
    lane.completions = [100, 200, 300, 400, 500, 600, 700]
    assert lane.headroom() == 6000  # the default until eight completions are in
    lane.completions.append(800)
    assert lane.headroom() == 600  # nearest-rank p75 of eight
    lane.completions = [40_000] * 8
    assert lane.headroom() == CAP
    # The gate observes a game's previous completion at its next admission.
    g0 = games[0]
    lane.completions = []
    lane.note_admission("g0")
    charged(now, g0, 5, 1234)
    lane.note_admission("g0")
    lane.note_admission("g0")  # a failed request charged nothing: no observation
    assert lane.completions == [1234]
    assert lane.generated(10) == 1200 and lane.generated(10_000) == CAP


async def test_requests_carry_priority_focus_0_lane_1_only_with_the_commit_lane(tmp_path):
    from test_arena_duck import arena_agent, text_reply

    from arena.duck import ArenaTransferDuckAgent

    sent = []

    class Transport:
        async def post(self, url, **kwargs):
            sent.append(kwargs["json"])
            return text_reply("ready")

    agent, _, game, plain = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    payload = {"tools": [{"type": "function"}], "messages": []}
    async with agent.client.slots:  # the plain gate: no priority field
        assert agent.client.slots.request_priority() is None
        await agent.request_completion(payload)
    other = NS(game_id="other", state="active", lane=False, latest_prompt_tokens=0, index=1)
    gate = KVGate(10**6, slots=8, lane=CommitLane([game, other], window=WINDOW, cap=CAP, pool_units=10**6))
    agent.client.slots = GateSlot(gate, game, cap=CAP, window=WINDOW)
    game.state = "active"
    async with agent.client.slots:
        assert agent.client.slots.request_priority() == 0
        await agent.request_completion(payload)
    game.state, game.lane = "parked", True
    game.clock.pause()
    async with agent.client.slots:  # a lane game does not wait for a slot; its request is a lane request
        assert agent.client.slots.request_priority() == 1
        await agent.request_completion(payload)
    assert agent.client.slots.request_priority() is None  # outside a request
    # The index-lane gate sends no priority either.
    agent.client.slots = GateSlot(KVGate(10**6, slots=8, lane=BackgroundLane([game], window=WINDOW, cap=CAP)),
                                  game, cap=CAP, window=WINDOW)
    async with agent.client.slots:
        await agent.request_completion(payload)
    assert ["priority" in p for p in sent] == [False, True, True, False]
    assert [p.get("priority") for p in sent[1:3]] == [0, 1]
    assert "priority" not in payload  # the caller's payload is not modified


async def test_preemptions_count_only_when_a_focus_request_ran_without_a_lane_request(monkeypatch):
    now, wall, games, sched = steady()
    gate, _ = commit_gate(games, now)
    attribution = PreemptionAttribution(gate, slack_seconds=2)

    def sample(value):
        attribution.observe({"elapsed_seconds": now(), "metrics": {COUNTER: float(value)}})

    sample(0)  # baseline
    focus = await gate.acquire("g0", 20_000, prompt=4000)
    now.advance(5)
    sample(1)  # focus alone: a focus request could have been the victim
    lane = await gate.acquire("g2", 20_000, prompt=4000)
    now.advance(5)
    sample(1)  # no increment; the window [3, 10] still held the focus-only mix
    now.advance(5)
    sample(3)  # [8, 15]: whenever the focus request ran, a lane request ran too
    now.advance(1)
    gate.release(focus)
    now.advance(9)
    sample(4)  # [13, 25]: lane only
    gate.release(lane)
    now.advance(15)
    sample(4)
    now.advance(10)
    sample(5)  # [38, 50]: nothing in flight
    attribution.observe(None)
    assert attribution.counts == {"lane_policy": 3, "focus_exposed": 1, "unattributed": 1}
    assert [e["attribution"] for e in attribution.events] == ["focus_exposed", "lane_policy", "lane_policy",
                                                              "unattributed"]
    assert attribution.missed_samples == 1
    # The monitor's total counts too: increments no sample attributed count against the trial.
    assert attribution.counted(5) == 2 and attribution.counted(6) == 3
    # The report: lane preemptions alone keep the trial complete and preemption-free.
    from arena import report

    monkeypatch.setattr(report, "fleet_summary", lambda rows, seconds: {
        "failed_sessions": 0, "mean_score_equal_weight_25_games": 0})
    monkeypatch.setattr(report, "audit_scheduler", lambda result: {"valid": True})

    def make(policy, total):
        return report.trial_report({}, rows=[], elapsed=10, choice={}, cap=CAP, config={}, scheduler={}, gate={},
                                   engine={"valid": True, "preemptions": total}, task_errors=[],
                                   preemption_policy=policy)

    lane_only = PreemptionAttribution(gate)
    lane_only.counts["lane_policy"] = 7
    result = make(lane_only, 7)
    assert result["status"] == "complete" and result["preemption_free"] and result["preemptions_lane_policy"] == 7
    assert result["preemptions_counted"] == 0 and result["engine"]["preemptions"] == 7
    assert "rule" in result["preemption_attribution"]
    result = make(attribution, 5)
    assert result["status"] == "complete_with_preemptions" and not result["preemption_free"]
    # Other studies: every preemption counts, and the report has no attribution fields.
    result = make(None, 7)
    assert result["status"] == "complete_with_preemptions" and "preemptions_lane_policy" not in result


def test_priority_policy_reaches_the_server_command_for_this_variant_only(tmp_path, monkeypatch):
    from test_qwen_serving import import_fork

    from arena import serving

    # Only fixed-2x-commit sets a scheduling policy.
    for name in VARIANTS:
        try:
            study = apply_variant(CONTROL, name)
        except ValueError:
            continue
        assert ("scheduling_policy" in study["server"]) == (name in ("fixed-2x-commit", "fixed-2x-commit-queue")), name
    study = apply_variant(CONTROL, "fixed-2x-commit")
    assert study["server"]["scheduling_policy"] == "priority" and study["commit"] == COMMIT_DEFAULTS
    assert study["level_budget_policy"] == "commit"
    assert set(study["arena_variant"]["reference_difference"]) == {"level_budget_policy", "commit",
                                                                    "server.scheduling_policy"}
    # The pinned serving script's actual command, with and without the policy.
    recipe = import_fork(tmp_path, monkeypatch)
    for key, value in {'TAAF_VLLM_MAX_NUM_SEQS': '25', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '1',
                       'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS': '4096', 'TAAF_VLLM_MTP_TOKENS': '3'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(recipe, 'verify_and_extract_runtime', lambda *a, **k: {})
    recipe.ANALYZER_CONTEXT = 262144
    original_command = recipe.server_command
    plain = original_command(tmp_path / 'model', tuning=recipe.resolve_vllm_tuning())
    receipt = serving.configure(recipe, tmp_path, reclaim_skipped_blocks=True, scheduling_policy="priority")
    argv = recipe.server_command(tmp_path / 'model', tuning=recipe.resolve_vllm_tuning())
    assert receipt["scheduling_policy"] == "priority" and "--scheduling-policy" not in plain
    assert argv == plain + ["--scheduling-policy", "priority"]
    recipe.server_command = original_command
    receipt = serving.configure(recipe, tmp_path, reclaim_skipped_blocks=True)
    assert "scheduling_policy" not in receipt
    assert recipe.server_command(tmp_path / 'model', tuning=recipe.resolve_vllm_tuning()) == plain
    with pytest.raises(ValueError, match="scheduling policy"):
        serving.configure(NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ['serve']), tmp_path,
                          scheduling_policy="fcfs")
    stub = NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ['serve', '--scheduling-policy', 'fcfs'])
    serving.configure(stub, tmp_path, scheduling_policy="priority")
    with pytest.raises(ValueError, match="already sets"):
        stub.server_command()
    # arena.serving's entry point passes the study's policy to the hook.
    hooks = []
    monkeypatch.setattr(serving, "start", lambda *a, **k: hooks.append(k["recipe_hook"]))
    monkeypatch.setattr(serving.BlockBudget, "load", lambda *a: NS(layout={}))
    path = tmp_path / "study.json"
    for spec, expected in ((study, ['serve', '--scheduling-policy', 'priority']),
                           (apply_variant(CONTROL, "reclaim-fixed-2x"), ['serve'])):
        path.write_text(json.dumps(spec))
        monkeypatch.setattr("sys.argv", ["serving", "--working", str(tmp_path), "--input-root", "x",
                                         "--study", str(path)])
        serving.main()
        stub = NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ['serve'])
        hooks[-1](stub, tmp_path)
        assert stub.server_command() == expected
    # The trial reads the running server's policy from its recorded argv (else the log).
    server = tmp_path / "server"
    server.mkdir()
    assert server_scheduling_policy(server) == (None, None)
    (server / "vllm-openai-server.log").write_text("INFO non-default args: {'model': 'x', 'scheduling_policy': 'priority'}\n")
    assert server_scheduling_policy(server) == ("priority", "server log non-default args")
    (server / "vllm-server-identity.json").write_text(json.dumps({"argv": plain}))
    assert server_scheduling_policy(server)[0] == "fcfs"
    (server / "vllm-server-identity.json").write_text(json.dumps({"argv": argv}))
    assert server_scheduling_policy(server) == ("priority", "vllm-server-identity.json argv")


def test_settings_validation_and_policy_exclusions():
    assert commit_settings() == COMMIT_DEFAULTS
    for bad in ({"focus_slots": 0}, {"stall_max_tokens": 1000}, {"lane_kv_fraction": 1.5},
                {"lane_max_inflight": 2.0}, {"guard_caps": 2}):
        with pytest.raises(ValueError):
            commit_settings(bad)
    now, wall, games, sched = build("fixed-2x", count=3)
    with pytest.raises(ValueError, match="commit budget policy"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", commit={})
    with pytest.raises(ValueError, match="lane"):
        Scheduler(games, wall, budget_policy="commit", backlog_policy="readmissions", background_lane={})
    assert "commit" not in sched.summary()
    for name in ("fixed-2x-index-lane", "reclaim-fixed-2x"):
        assert "commit" not in apply_variant(CONTROL, name)
    with pytest.raises(ValueError, match="hybrid-block"):
        apply_variant({**CONTROL, "cache_admission": None}, "fixed-2x-commit")


def test_variant_bundle(tmp_path):
    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, built = builder.build("commit-test", paired_30m=True, output=tmp_path / "c", arena_variant="fixed-2x-commit",
                             opening_burst_requests=3, exclude_games=["ft09", "lp85"], wall_seconds=3600)
    assert built["commit"] == COMMIT_DEFAULTS and built["server"]["scheduling_policy"] == "priority"
    assert built["level_budget_policy"] == "commit" and "vLLM priority scheduling" in built["level_budget_rule"]
    assert built["backlog_priority"] == "momentum commitment (arena.commit)"
    written = json.loads((tmp_path / "c/configs/comparison.json").read_text())
    assert written["commit"]["lane_max_inflight"] == 4 and written["active_limit"] == 2


async def test_trial_wires_the_commit_lane_priorities_and_attribution(tmp_path, monkeypatch):
    from test_arena_trial import STUDY, FakeArcade

    from arena import trial as trial_module
    from arena.trial import Trial

    games = STUDY["games"][:5]
    study = {**STUDY, "games": games, "wall_seconds": 1.5, "level_budget_policy": "commit",
             "backlog_policy": "readmissions", "commit": {"lane_max_inflight": 3}}
    monkeypatch.setattr(trial_module, "make_arcade", lambda mode, games_dir, rec: FakeArcade(games))
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)

    class CountingMonitor:
        """The engine monitor's contract: ``on_sample`` after every sample; the counter never moves."""

        def __init__(self, output, base_url):
            self.start, self.on_sample, self.samples = None, None, 0

        async def sample(self):
            self.samples += 1
            if self.on_sample is not None:
                self.on_sample({"elapsed_seconds": 0.0, "metrics": {COUNTER: 4.0}})

        async def run(self):
            while True:
                await asyncio.sleep(0.2)
                await self.sample()

        def summary(self):
            return {"valid": True, "preemptions": 0, "accepted_generation_tokens": 10}

    monkeypatch.setattr(trial_module, "GpuMonitor", CountingMonitor)
    priorities = []

    class RequestingWorker:
        def __init__(self, game, gate, window, cap):
            self.game, self.slot = game, GateSlot(gate, game, cap=cap, window=window)

        async def run(self):
            self.game.ready.set()
            while True:
                self.slot.reserve(2000)
                async with self.slot:
                    priorities.append((self.game.lane, self.slot.request_priority()))
                    with self.game.clock.inference():
                        await asyncio.sleep(0.02)
                    self.game.clock.charge(2000, 500)
                if self.game.index in (1, 3) and self.game.levels == 0:
                    self.game.report_level(1, self.game.clock.global_clock.elapsed())

        def result(self):
            return {"game_id": self.game.game_id, "status": "time_limit", "error": None, "levels": self.game.levels,
                    "total_levels": 6, "actions": 3, "model_calls": 2, "prompt_tokens": 100, "completion_tokens": 50}

    async def fake_duck_agents(self, records, gate, window, wall, card, arc):
        assert gate.commit and gate.lane.max_inflight == 3 and gate.lane.pool_units == 291_900
        workers = [RequestingWorker(r, gate, window, self.cap) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", fake_duck_agents)

    def trial(folder, server):
        return Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / folder,
                     server_dir=server, base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None)

    # No priority server: the trial stops before any request carries a priority.
    fcfs = tmp_path / "fcfs"
    fcfs.mkdir()
    (fcfs / "vllm-server-identity.json").write_text(json.dumps({"argv": ["vllm", "serve"]}))
    with pytest.raises(RuntimeError, match="scheduling-policy priority"):
        await trial("a", fcfs).run()
    server = tmp_path / "priority"
    server.mkdir()
    (server / "vllm-server-identity.json").write_text(json.dumps({"argv": ["vllm", "serve", "--scheduling-policy",
                                                                           "priority"]}))
    report = await trial("b", server).run()
    assert report["status"] == "complete" and report["scheduler_audit"]["valid"], report["scheduler_audit"]
    assert report["commit_server"] == {"scheduling_policy": "priority", "source": "vllm-server-identity.json argv"}
    assert report["preemptions_lane_policy"] == 0 and report["preemption_free"]
    commit, gate = report["scheduler"]["commit"], report["gate"]["lane"]
    assert report["scheduler"]["active_limit"] == 2 and commit["tokens_per_hour"] > 0
    assert gate["lane_admitted"] == commit["lane"]["requests"] > 0 and gate["focus_wait_seconds"] == 0
    assert set(commit["level1_floor"]["games"]) == set(games)
    # Lane requests went out as priority 1, focus and opening requests as 0.
    assert {(lane, priority) for lane, priority in priorities} <= {(False, 0), (True, 1)}
    assert (True, 1) in priorities and (False, 0) in priorities
