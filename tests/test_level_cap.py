"""Per-level token cap of the priority gate (``priority.level_cap_tokens``): capped at an eviction,
ranked below every uncapped waiting game, back only into a slot that would idle (fewest tokens on the level first, in
grants), lifted by a clear; an offline replay of a whole run with a low cap; the real trial loop; off by default."""

import asyncio
import json
import random

import pytest
from test_arena_priority import act, active, at, build, charge, events, report

from arena.priority import DEFAULTS, settings
from arena.scheduler_audit import audit_scheduler


def drain(game):
    game.note_context_drain(tokens_before=110_000, tokens_after=55_000)


def test_off_by_default_and_validated():
    assert DEFAULTS["level_cap_tokens"] == 0 and DEFAULTS["level_cap_grant_tokens"] == 100_000
    assert DEFAULTS["level_cap_requests"] == 0 and DEFAULTS["level_cap_grant_requests"] == 100
    for bad in ({"level_cap_tokens": -1}, {"level_cap_tokens": 1.5}, {"level_cap_grant_tokens": 0},
                {"level_cap_requests": -1}, {"level_cap_grant_requests": 0}):
        with pytest.raises(ValueError):
            settings(bad)
    now, wall, games, sched, log = build(count=3, slots=1)
    charge(now, games[0], 900_000)
    for g in games[1:]:  # everyone started, so scores decide
        drain(next(x for x in games if x.state == "active"))
    assert not any(g.level_capped for g in games) and "level_cap" not in sched.priority_progress()


def test_capped_at_the_eviction_parked_and_back_only_into_an_idle_slot():
    now, wall, games, sched, log = build(count=3, slots=1, level_cap_tokens=20_000, level_cap_grant_tokens=10_000)
    g0, g1, g2 = games
    charge(now, g0, 25_000)
    at(now, sched, 5)
    assert active(games) == ["g0"] and not g0.level_capped  # nothing happens between evictions
    drain(g0)
    assert g0.level_capped and g0.state == "parked" and active(games) == ["g1"]
    assert g0.admissions[-1]["reason"] == "handover" and g0.admissions[-1]["level_cap"] is True
    # g1 and g2 are uncapped: whatever their scores, g0 stays behind them.
    charge(now, g1, 15_000)
    act(g1, 400)
    drain(g1)
    assert active(games) == ["g2"]
    charge(now, g2, 15_000)
    act(g2, 400)
    drain(g2)
    assert active(games) == ["g1"] and g0.state == "parked"  # g1 (15K, uncapped) before g0
    # Both uncapped games finish: the slot would idle, so the capped game returns with a grant.
    g1.finished_reason = "won"
    at(now, sched, 20)
    assert active(games) == ["g2"]
    g2.finished_reason = "won"
    at(now, sched, 30)
    assert active(games) == ["g0"] and g0.admissions[-1]["level_cap_return"] is True
    assert g0.level_cap_limit == 35_000 and sched.level_cap_counts["returns"] == 1
    charge(now, g0, 12_000)
    drain(g0)  # grant used up, nobody waits: it keeps the slot with a further grant
    assert active(games) == ["g0"] and sched.level_cap_counts["kept"] == 1 and g0.level_cap_limit == 47_000
    g0.report_level(1, wall.elapsed())  # the clear lifts the cap
    assert not g0.level_capped and sched.level_cap_counts["lifted"] == 1
    progress = sched.priority_progress()["level_cap"]
    assert progress["capped"] == 1 and progress["parked"] == 1 and progress["tokens_freed"] == 30_000
    at(now, sched, 300)
    sched.wall_end()
    assert audit_scheduler(report(games, sched, wall))["valid"]


def test_capped_games_rotate_fewest_tokens_first_and_never_leave_a_slot_idle():
    now, wall, games, sched, log = build(count=3, slots=1, level_cap_tokens=20_000, level_cap_grant_tokens=10_000)
    for g, tokens in zip(games, (40_000, 30_000, 21_000)):
        charge(now, g, tokens)
        drain(g)
    # All three capped; the slot went g0 -> g1 -> g2 (unstarted first), then to the capped game with the fewest tokens.
    assert all(g.level_capped for g in games) and active(games) == ["g2"]
    assert sched.level_cap_counts == {"capped": 3, "parked": 2, "returns": 0, "kept": 1, "rotations": 0, "lifted": 0}
    charge(now, games[2], 10_000)  # 31K: its grant is over; g1 (30K) has fewer
    drain(games[2])
    assert active(games) == ["g1"] and sched.level_cap_counts["rotations"] == 1
    charge(now, games[1], 5_000)  # inside its grant: it keeps the slot although g2 (31K) has fewer than its 35K
    drain(games[1])
    assert active(games) == ["g1"]
    charge(now, games[1], 6_000)  # 41K, grant over: g2 (31K) before g0 (40K)
    drain(games[1])
    assert active(games) == ["g2"]


def replay(seed, count=14, slots=4, cap=20_000, grant=10_000, steps=4000):
    """A whole run offline: every active game generates tokens each step, drains every ~6K tokens, clears a level at
    its own rate (three games never do). Returns the scheduler and what the checks saw."""
    rng = random.Random(seed)
    now, wall, games, sched, log = build(count=count, slots=slots, wall_seconds=steps + 10, total_levels=3,
                                         level_cap_tokens=cap, level_cap_grant_tokens=grant)
    rate = {g.game_id: (0.0 if g.index % 5 == 0 else rng.choice((0.02, 0.05, 0.15))) for g in games}
    since = {g.game_id: 0 for g in games}
    seen = {"idle": 0, "capped_over_uncapped": 0, "capped_active_steps": 0, "order": 0, "returns": []}
    for step in range(1, steps + 1):
        for game in [g for g in games if g.state == "active"]:
            tokens = rng.randint(500, 2500)
            charge(now, game, tokens, seconds=0.01)
            act(game, rng.randint(0, 3))
            since[game.game_id] += tokens
            if rng.random() < rate[game.game_id]:
                game.report_level(game.levels + 1, wall.elapsed())
                if game.levels >= 3:
                    game.finished_reason = "won"
                    continue
            if since[game.game_id] >= 6000:
                since[game.game_id] = 0
                waiting = [g for g in sched.priority_waiting()]
                before = {g.game_id for g in games if g.state == "active"}
                drain(game)
                for g in games:
                    if g.state == "active" and g.game_id not in before and g.level_capped:
                        # A capped game came in: nothing uncapped waited, and it had the fewest tokens on its level.
                        others = [w for w in waiting if w is not g and w.level_capped]
                        seen["order"] += any(w.spent_on_level() < g.spent_on_level() - grant for w in others)
                        seen["capped_over_uncapped"] += any(not w.level_capped for w in waiting)
                        seen["returns"].append(g.game_id)
        at(now, sched, now.value + 0.01)
        playable = sum(g.state != "finished" for g in games)
        seen["idle"] += len(sched.active()) != min(slots, playable)
        waiting = sched.priority_waiting()
        capped_active = [g for g in games if g.state == "active" and g.level_capped]
        seen["capped_active_steps"] += bool(capped_active)
        seen["capped_over_uncapped"] += bool(capped_active) and any(not w.level_capped for w in waiting)
        if playable == 0:
            break
    now.value = steps + 10
    sched.wall_end()
    return games, sched, wall, seen


@pytest.mark.parametrize("seed", range(6))
def test_offline_replay_with_a_low_cap(seed):
    games, sched, wall, seen = replay(seed)
    counts = sched.priority_progress()["level_cap"]
    assert seen["idle"] == 0  # every slot busy whenever a game could play
    assert seen["capped_over_uncapped"] == 0  # a capped game never holds a slot an uncapped game waits for
    assert seen["order"] == 0
    assert counts["capped"] >= 3 and counts["parked"] >= 1 and counts["returns"] >= 1 and seen["capped_active_steps"]
    assert counts["tokens_freed"] > 0
    # The three games that never clear end capped; the others won; the run ended at the wall without a hang.
    assert [g.game_id for g in games if g.finished_reason == "won"] == [g.game_id for g in games if g.index % 5]
    assert all(g.level_capped for g in games if g.index % 5 == 0)
    assert all(events(g)[-1] == (("finish", "finished") if g.index % 5 else ("finish", "wall_end")) for g in games)
    result = audit_scheduler(report(games, sched, wall, target=4010))
    assert result["valid"], result["issues"]


def test_the_cap_off_replays_exactly_as_before():
    """Same seed, cap off: the decisions equal a scheduler that has never heard of the cap (no capped game, no counter
    in the progress row)."""
    games, sched, wall, seen = replay(3, cap=0, steps=600)
    assert not any(g.level_capped for g in games) and "level_cap" not in sched.priority_progress()
    assert seen["idle"] == 0 and all("level_cap" not in d for d in sched.priority_decisions)


async def test_the_trial_loop_caps_parks_and_returns(tmp_path, monkeypatch):
    import test_arena_early_stop as fake

    from arena import trial as trial_module
    from arena.scheduler import token_budget_settings
    from arena.trial import Trial

    games = fake.STUDY["games"][:3]
    study = {**fake.STUDY, "games": games, "wall_seconds": 3.0, "active_limit": 1, "level_budget_policy": "priority",
             "backlog_policy": "readmissions", "budget_clock": "tokens", "token_budget": token_budget_settings(),
             "priority": {"level_cap_tokens": 20_000, "level_cap_grant_tokens": 10_000}}
    study.pop("cost_normalization", None)
    arcade = fake.ScoreArcade(games)
    arcade.score = 0.0
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", fake.OfflineMonitor)

    class Spending(fake.Worker):
        async def run(self):
            self.game.ready.set()
            while True:
                await self.game.clock.wait_active()
                await asyncio.sleep(0.1)
                self.game.clock.charge(1000, 12_000)
                self.game.note_context_drain(tokens_before=110_000, tokens_after=55_000)

    async def agents(self, records, gate, window, wall_clock, card, arc):
        workers = [Spending(r) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7, "gameplay_seconds": 3.0}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t", server_dir=tmp_path,
                  base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None, arcade=arcade)
    report_ = await trial.run()
    summary = report_["scheduler"]["priority"]
    cap = summary["progress"]["level_cap"]
    assert summary["settings"]["level_cap_tokens"] == 20_000
    assert cap["capped"] == 3 and cap["returns"] >= 2 and cap["rotations"] >= 1 and cap["tokens_freed"] > 0
    log = [json.loads(line) for line in (tmp_path / "t/admission.jsonl").read_text().splitlines()]
    assert any(e.get("level_cap_return") for e in log) and any(e.get("level_cap") for e in log if e.get("event") == "park")
    assert audit_scheduler(report_)["valid"]


def asks(game, count):
    game.requests.extend({} for _ in range(count))


def test_the_request_cap_catches_a_cheap_loop_the_token_cap_misses():
    now, wall, games, sched, log = build(count=2, slots=1, level_cap_tokens=300_000, level_cap_grant_tokens=100_000,
                                         level_cap_requests=1_000, level_cap_grant_requests=100)
    g0, g1 = games
    charge(now, g0, 37_000)  # 1,000 replies of 37 tokens: far under the token cap
    asks(g0, 999)
    drain(g0)
    assert not g0.level_capped and active(games) == ["g1"]  # 999: under the cap, an ordinary hand-over
    charge(now, g1, 200_000)
    drain(g1)
    assert active(games) == ["g0"]
    asks(g0, 1)
    drain(g0)
    assert g0.level_capped and g0.state == "parked" and active(games) == ["g1"]
    assert sched.priority_progress()["level_cap"]["cap_requests"] == 1_000


def test_the_request_grant_ends_a_grant_before_the_token_grant_and_a_clear_resets_the_count():
    now, wall, games, sched, log = build(count=1, slots=1, level_cap_tokens=0, level_cap_requests=50,
                                         level_cap_grant_requests=10, level_cap_grant_tokens=1_000_000)
    g0 = games[0]
    asks(g0, 50)
    drain(g0)  # nobody waits: a further grant in place, parked never
    assert g0.level_capped and g0.state == "active" and sched.level_cap_counts["kept"] == 1
    assert g0.level_cap_request_limit == 60
    asks(g0, 9)
    drain(g0)
    assert sched.level_cap_counts["kept"] == 1  # inside the request grant
    asks(g0, 1)
    drain(g0)
    assert sched.level_cap_counts["kept"] == 2 and g0.level_cap_request_limit == 70
    g0.report_level(1, wall.elapsed())
    assert not g0.level_capped and g0.requests_on_level() == 0
    asks(g0, 49)
    drain(g0)
    assert not g0.level_capped


def test_either_limit_caps_and_a_token_only_cap_ignores_requests():
    now, wall, games, sched, log = build(count=2, slots=1, level_cap_tokens=20_000, level_cap_grant_tokens=10_000)
    g0 = games[0]
    charge(now, g0, 25_000)
    asks(g0, 100_000)
    drain(g0)
    assert g0.level_capped and g0.state == "parked"  # by tokens
    g1 = games[1]
    g1.finished_reason = "won"
    at(now, sched, 20)
    assert active(games) == ["g0"]
    asks(g0, 100_000)
    charge(now, g0, 1_000)
    drain(g0)
    assert sched.level_cap_counts["kept"] == 0  # a token-only cap: requests do not end the 10K grant

