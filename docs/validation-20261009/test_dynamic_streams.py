"""E1 dynamic streams on a built bundle's own code (ARC_DYN_BUNDLE=<bundle built with --dynamic-streams>):
the controller's up/down/hold rules, and the patched priority scheduler (a raised limit refills at the next tick; a
lowered limit releases a game at its next eviction without a replacement, then handovers resume)."""
import os
import sys
from pathlib import Path

import pytest

B = os.environ.get("ARC_DYN_BUNDLE")
if not B:
    pytest.skip("ARC_DYN_BUNDLE not set", allow_module_level=True)
sys.path.insert(0, str(Path(B) / "src"))

from arena.clock import ActiveClock, GlobalClock  # noqa: E402
from arena.dynamic_streams import DynamicStreams  # noqa: E402
from arena.scheduler import GameRecord, Scheduler  # noqa: E402

CFG = {"lo": 2, "hi": 4, "up_kv": 0.85, "down_kv": 0.92, "every_seconds": 30}


class Now:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class Gate:
    def __init__(self, waiting=0):
        self.queue = [None] * waiting


def build(count=8, slots=2):
    now = Now()
    wall = GlobalClock(10_000, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="tokens"), total_levels=8)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    log = []
    sched = Scheduler(games, wall, log=log.append, active_limit=slots, budget_policy="priority",
                      backlog_policy="readmissions", priority={"equal": True})
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, games, sched, log


def active(games):
    return [g.game_id for g in games if g.state == "active"]


def sample(t, kv):
    return [{"elapsed_seconds": t, "metrics": {"vllm:kv_cache_usage_perc": kv}}]


def test_controller_moves_one_step_per_interval_inside_bounds():
    now, games, sched, log = build()
    dyn, notes = DynamicStreams(CFG, log=None), []
    dyn.log = notes.append
    assert dyn.step(0, sample(0, 0.4), sched, Gate()) == 3 and sched.active_limit == 3  # start at lo, then up
    assert dyn.step(10, sample(10, 0.4), sched, Gate()) is None  # within the interval
    assert dyn.step(31, sample(31, 0.4), sched, Gate(waiting=1)) == 3  # something waits at the gate: hold
    assert dyn.step(62, sample(62, 0.4), sched, Gate()) == 4
    assert dyn.step(93, sample(93, 0.4), sched, Gate()) == 4  # at hi
    assert dyn.step(124, sample(124, 0.88), sched, Gate()) == 4  # between up and down: hold
    assert dyn.step(155, sample(155, 0.95), sched, Gate()) == 3
    assert dyn.step(186, sample(100, 0.99), sched, Gate()) is None  # stale engine sample: hold
    assert dyn.step(217, sample(217, 0.99), sched, Gate()) == 2 and dyn.step(248, sample(248, 0.99), sched, Gate()) == 2
    assert dyn.changes == {"up": 2, "down": 2} and dyn.max_limit == 4
    assert notes[0]["what"] == "start" and [n["what"] for n in notes[1:]] == ["up", "up", "down", "down"]


def test_controller_starts_at_lo_whatever_the_boot_limit():
    now, games, sched, log = build(slots=5)
    DynamicStreams(CFG).step(0, [], sched, Gate())
    assert sched.active_limit == 2


def test_scheduler_refills_on_raise_and_releases_at_eviction_on_lower():
    now, games, sched, log = build()
    assert active(games) == ["g0", "g1"]
    sched.active_limit = 4
    now.value = 1
    sched.tick()
    assert active(games) == ["g0", "g1", "g2", "g3"]
    sched.active_limit = 2
    now.value = 2
    sched.tick()
    assert len(active(games)) == 4  # lowering never stops a game
    games[0].note_compaction()  # eviction while over the limit: released, no replacement
    assert active(games) == ["g1", "g2", "g3"] and games[0].state == "parked"
    assert sched.priority_counts["dynamic_release"] == 1
    games[1].note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert active(games) == ["g2", "g3"] and sched.priority_counts["dynamic_release"] == 2
    games[2].note_compaction()  # at the limit: an ordinary handover (the slot goes to the best waiting game)
    assert len(active(games)) == 2 and sched.priority_counts["dynamic_release"] == 2
    decisions = [d["decision"] for d in sched.priority_decisions]
    assert decisions.count("dynamic_release") == 2 and decisions[-1] == "handover"
    assert games[2].state == "parked" and games[4].state == "active"  # g2's slot went to the first unstarted game


async def test_the_trial_loop_moves_the_limit_from_live_kv(tmp_path, monkeypatch):
    """The patched trial.py: the limit rises while KV is low (more games admitted), then falls when KV is high
    (games released at their evictions, no refill), all through the real Trial loop and admission log. The loop ticks at most every 1 s when nothing finishes, so the steps land at ~1 s marks."""
    import asyncio
    import json
    import time

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    import test_arena_early_stop as fake

    from arena import trial as trial_module
    from arena.scheduler import token_budget_settings
    from arena.trial import Trial

    games = fake.STUDY["games"][:5]
    study = {**fake.STUDY, "games": games, "wall_seconds": 6.0, "active_limit": 1, "level_budget_policy": "priority",
             "backlog_policy": "readmissions", "budget_clock": "tokens", "token_budget": token_budget_settings(),
             "priority": {}, "dynamic_streams": {"lo": 1, "hi": 3, "up_kv": 0.85, "down_kv": 0.92,
                                                  "every_seconds": 0.2}}
    study.pop("cost_normalization", None)
    arcade = fake.ScoreArcade(games)
    arcade.score = 0.0
    t0 = time.monotonic()

    class KvMonitor(fake.OfflineMonitor):
        @property
        def samples(self):
            t = time.monotonic() - t0
            return [{"elapsed_seconds": t, "metrics": {"vllm:kv_cache_usage_perc": 0.4 if t < 3.5 else 0.99}}]

        @samples.setter
        def samples(self, value):
            pass

    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", KvMonitor)

    class Compacting(fake.Worker):
        async def run(self):
            self.game.ready.set()
            while True:
                await self.game.clock.wait_active()
                await asyncio.sleep(0.15)
                self.game.compactions += 1
                self.game.note_compaction()

    async def agents(self, records, gate, window, wall_clock, card, arc):
        workers = [Compacting(r) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7, "gameplay_seconds": 6.0}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t", server_dir=tmp_path,
                  base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None, arcade=arcade)
    await trial.run()
    log = [json.loads(line) for line in (tmp_path / "t/admission.jsonl").read_text().splitlines()]
    dyn = [e for e in log if e.get("event") == "dynamic_streams"]
    assert dyn[0]["what"] == "start" and dyn[0]["limit"] == 1
    whats = [e["what"] for e in dyn[1:]]
    assert whats[:2] == ["up", "up"] and whats[2:] == ["down", "down"]
    assert max(e["limit"] for e in dyn) == 3 and dyn[-1]["limit"] == 1
    releases = [e for e in log if e.get("event") == "park" and (e.get("dynamic_release")
                                                                or (e.get("extra") or {}).get("dynamic_release"))]
    assert len(releases) >= 2
