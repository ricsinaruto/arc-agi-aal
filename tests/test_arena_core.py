"""Arena core: clocks, the KV gate, the window choice and the scheduler rules."""

import asyncio

import pytest

from arena.capacity import choose_window, parse_kv_pool
from arena.clock import ActiveClock, GlobalClock
from arena.gate import GateSlot, KVGate
from arena.scheduler import GameRecord, Scheduler


class Now:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def make_games(now, count=5, seconds=900):
    wall = GlobalClock(seconds, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now)) for i in range(count)]
    return wall, games


def test_active_clock_counts_only_active_seconds():
    now = Now()
    wall = GlobalClock(900, now=now)
    clock = ActiveClock(wall, now=now)
    assert clock.active_seconds() == 0 and clock.remaining() == 900
    wall.release()
    clock.release()
    now.advance(10)
    clock.pause()
    now.advance(50)
    assert clock.active_seconds() == 10 and clock.parked_seconds() == 50 and clock.elapsed() == 10
    assert clock.remaining() == 840  # the wall clock keeps running while parked
    clock.resume()
    now.advance(5)
    assert clock.active_seconds() == 15 and clock.parked_seconds() == 50
    clock.grant(20)
    now.advance(12)
    assert clock.budget_spent() == 12 and clock.budget_remaining() == 8 and not clock.budget_exhausted()
    assert clock.remaining() == 900 - 77  # the wall clock, never the level budget
    now.advance(8)
    assert clock.budget_exhausted() and clock.remaining() == 900 - 85
    clock.pause()
    assert clock.remaining() == 900 - 85
    feedback = clock.feedback()
    assert feedback["game_active_seconds"] == 35 and feedback["level_budget_seconds"] == 20
    assert feedback["solve_target_seconds"] == 900 and feedback["level_budget_remaining_seconds"] == 0
    with pytest.raises(RuntimeError):
        clock.release()


async def test_gate_is_fifo_and_bounded_by_slots_and_tokens():
    now = Now()
    log = []
    gate = KVGate(1000, slots=2, log=log.append, now=now)
    a = await gate.acquire("a", 400)
    b = await gate.acquire("b", 400)
    assert gate.inflight == 2 and gate.reserved == 800
    waiter_c = asyncio.ensure_future(gate.acquire("c", 100))  # slots full
    waiter_d = asyncio.ensure_future(gate.acquire("d", 100))
    await asyncio.sleep(0)
    assert gate.snapshot()["waiting"] == 2 and not waiter_c.done()
    gate.release(a)
    await asyncio.sleep(0)
    c = await waiter_c  # FIFO: c before d
    assert gate.inflight == 2 and gate.reserved == 500 and not waiter_d.done()
    gate.release(b)
    d = await waiter_d
    assert gate.inflight == 2 and gate.reserved == 200
    # Token capacity blocks even with a free slot, and the head blocks everything behind it.
    gate.release(c)
    big = asyncio.ensure_future(gate.acquire("e", 995))
    small = asyncio.ensure_future(gate.acquire("f", 10))
    await asyncio.sleep(0)
    assert not big.done() and not small.done()
    gate.release(d)
    e = await big
    await asyncio.sleep(0)
    assert not small.done() and gate.reserved == 995
    gate.release(e)
    f = await small
    gate.release(f)
    snap = gate.snapshot()
    assert snap["peak_inflight"] == 2 and snap["peak_reserved"] == 995 and snap["admitted"] == 6
    assert snap["reserved"] == 0 and snap["inflight"] == 0
    assert [e["event"] for e in log][:3] == ["admit", "admit", "release"]


async def test_gate_slot_waits_for_an_active_game_and_reserves_estimate_plus_cap():
    now = Now()
    wall, games = make_games(now, count=1)
    game = games[0]
    wall.release()
    game.clock.release()
    gate = KVGate(300_000, slots=8, now=now)
    slot = GateSlot(gate, game, cap=16384, window=131072)
    slot.reserve(120_000)
    assert slot.tokens() == 131072  # clamped to the window
    slot.reserve(1000)
    game.clock.pause()
    entered = asyncio.ensure_future(slot.__aenter__())
    await asyncio.sleep(0)
    assert not entered.done()
    now.advance(7)
    game.clock.resume()
    await entered
    assert gate.reserved == 17384 and game.requests[-1]["tokens"] == 17384
    assert game.requests[-1]["gate_wait_seconds"] == 7
    await slot.__aexit__(None, None, None)
    assert gate.reserved == 0


def test_choose_window_from_the_pool():
    assert choose_window(291_900)["window"] == 131072
    assert choose_window(291_900)["capacity"] == 262_710 and choose_window(291_900)["rejected"] == []
    assert choose_window(280_000)["window"] == 98304 and choose_window(280_000)["rejected"] == [131072]
    assert choose_window(150_000)["window"] == 65536
    with pytest.raises(RuntimeError, match="No window"):
        choose_window(100_000)
    assert choose_window(291_900, margin=0.85)["window"] == 98304
    assert parse_kv_pool(["GPU KV cache size: 291,900 tokens, Maximum concurrency", "other"]) == 291_900
    assert parse_kv_pool(["nothing"]) is None


def test_kv_pool_survives_throughput_lines(tmp_path):
    from arena.capacity import kv_pool_tokens

    lines = ["(EngineCore pid=1) INFO GPU KV cache size: 291,900 tokens, Maximum concurrency for 262,144 tokens per request: 1.11x"]
    lines += [f"(APIServer pid=2) INFO Engine 000: Avg prompt throughput: {i} tokens/s, GPU KV cache usage: 1.0%" for i in range(40)]
    (tmp_path / "vllm-openai-server.log").write_text("\n".join(lines))
    assert kv_pool_tokens(tmp_path) == 291_900


def test_scheduler_opening_switch_budgets_and_regrant():
    now = Now()
    wall, games = make_games(now, count=5)
    sched = Scheduler(games, wall)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    assert all(g.state == "active" for g in games) and sched.phase == "opening"
    now.advance(100)
    games[1].note_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "opening" and all(g.state == "active" for g in games)  # one game is not enough
    now.advance(50)
    games[3].note_level(1, wall.elapsed())
    sched.on_level(games[3])
    sched.tick()
    assert sched.phase == "steady" and sched.switch_wall_seconds == 150
    assert [g.state for g in games] == ["parked", "active", "parked", "active", "parked"]
    assert games[1].t1 == 100 and games[3].t1 == 150
    assert games[1].time_to_level == [{"level": 1, "active_seconds": 100, "wall_seconds": 100, "budget_seconds": 100}]
    # Focus budgets: 2*T1 measured from the level, minus what was already spent on it.
    assert games[1].clock.budget == 2 * 100 - 50 and games[3].clock.budget == 2 * 150
    assert games[0].admissions[-1]["reason"] == "opening_switch" and games[0].clock.parked
    # Game 1 exhausts its budget at active 250: parked; the lowest-active-seconds backlog game is admitted.
    now.advance(150)
    sched.tick()
    assert games[1].state == "parked" and games[1].admissions[-1]["reason"] == "level_budget"
    admitted = [g for g in games if g.state == "active"]
    assert games[3] in admitted and len(admitted) == 2
    newcomer = [g for g in admitted if g is not games[3]][0]
    assert newcomer is games[0]  # all backlog games have 150 active seconds; roster order breaks the tie
    # Level-0 regrant: 1.5 x the 150 s already played = 225 s.
    assert newcomer.clock.budget == 225 and newcomer.admissions[-1]["reason"] == "regrant"
    # Game 3 reaches level 2 in flight: fresh budget 3*T1.
    now.advance(10)
    games[3].note_level(2, wall.elapsed())
    sched.on_level(games[3])
    assert games[3].clock.budget == 3 * 150 and games[3].admissions[-1]["reason"] == "level_reached"
    # Game 0 exhausts its 225 s: parked; the next backlog game (game 2, 150 s) comes in.
    now.advance(215)
    sched.tick()
    assert games[0].state == "parked" and games[2].state == "active" and games[2].clock.budget == 225
    # A parked level-1 game re-enters with the remaining level budget if any, else the regrant floor.
    games[1].clock.budget = None
    now.advance(300)
    sched.tick()
    assert games[2].state == "parked"
    # Priority now: game 4 (150 s) before game 1 (250 s) and game 0 (375 s).
    assert games[4].state == "active"
    now.advance(500)
    sched.tick()  # wall clock over 900: nothing else changes structurally
    sched.wall_end()
    assert all(g.clock.parked for g in games) and games[3].admissions[-1]["reason"] == "wall_end"
    summary = sched.summary()
    assert summary["phase"] == "steady" and summary["events"][0]["reason"] == "opening_start"


def test_scheduler_regrant_floor_and_no_park_without_backlog():
    now = Now()
    wall, games = make_games(now, count=2)
    sched = Scheduler(games, wall)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.advance(5)
    games[0].note_level(1, 5)
    games[1].note_level(1, 5)
    sched.tick()
    assert sched.phase == "steady" and all(g.state == "active" for g in games)
    now.advance(100)
    sched.tick()  # budgets exhausted but the backlog is empty: keep playing
    assert all(g.state == "active" for g in games)
    # Regrant floor: a level-0 game with 4 s played gets 4 + 30, not 6.
    wall2, games2 = make_games(Now(), count=1)
    g = games2[0]
    g.clock._active_before = 4.0
    assert Scheduler(games2, wall2).grant_for(g) == 34.0


def test_scheduler_finished_games_leave_the_rotation():
    now = Now()
    wall, games = make_games(now, count=3)
    sched = Scheduler(games, wall)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    games[0].note_level(1, 0)
    games[1].note_level(1, 0)
    sched.tick()
    games[0].note_finished("won")
    sched.tick()
    assert games[0].state == "finished" and games[2].state == "active" and games[1].state == "active"


def test_already_won_game_still_counts_toward_the_two_clear_trigger():
    now = Now()
    wall, games = make_games(now, count=3)
    sched = Scheduler(games, wall)
    wall.release()
    for game in games:
        game.clock.release()
    sched.start_all()
    now.advance(10)
    games[0].note_level(8, wall.elapsed())
    games[0].note_finished("won")
    sched.tick()
    assert sched.phase == "opening" and games[0].state == "finished"
    now.advance(10)
    games[1].note_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "steady" and sched.switch_wall_seconds == 20
    sched.tick()  # Fill the remaining slot from eligible unfinished games.
    assert [g.index for g in sched.active()] == [1, 2]
    assert games[0].state == "finished" and games[0].clock.parked


async def test_cancelled_queued_request_does_not_release_another_reservation():
    gate = KVGate(1000, slots=2)
    holder = await gate.acquire("holder", 800)
    head = asyncio.create_task(gate.acquire("head", 900))
    follower = asyncio.create_task(gate.acquire("follower", 100))
    await asyncio.sleep(0)
    head.cancel()
    with pytest.raises(asyncio.CancelledError):
        await head
    admitted = await asyncio.wait_for(follower, 1)
    assert gate.reserved == 900 and gate.inflight == 2
    gate.release(holder)
    gate.release(admitted)
    assert gate.reserved == gate.inflight == 0


async def test_cancelled_after_admission_returns_exactly_its_reservation():
    gate = KVGate(1000, slots=1)
    holder = await gate.acquire("holder", 800)
    queued = asyncio.create_task(gate.acquire("queued", 900))
    await asyncio.sleep(0)
    gate.release(holder)  # Resolves the waiter before it gets to run.
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert gate.reserved == gate.inflight == 0 and not gate.queue


async def test_game_parked_while_queued_does_not_send_a_request():
    now = Now()
    wall, games = make_games(now, count=1)
    game = games[0]
    wall.release()
    game.clock.release()
    gate = KVGate(1000, slots=1)
    holder = await gate.acquire("holder", 500)
    slot = GateSlot(gate, game, cap=100, window=500)
    entering = asyncio.create_task(slot.__aenter__())
    await asyncio.sleep(0)
    game.clock.pause()
    gate.release(holder)
    await asyncio.sleep(0)
    assert not entering.done() and gate.reserved == gate.inflight == 0
    assert not game.requests
    game.clock.resume()
    await asyncio.wait_for(entering, 1)
    assert gate.inflight == 1 and gate.reserved == 100
    await slot.__aexit__(None, None, None)
    assert gate.inflight == gate.reserved == 0


def test_compaction_retention_keeps_the_last_observations_verbatim():
    from arena.compaction import CompactionMixin, split_at_latest_observation

    history = [{"role": "user", "content": "obs1"}, {"role": "assistant", "content": "a1"},
               {"role": "tool", "content": "t1"}, {"role": "user", "content": "obs2"},
               {"role": "assistant", "content": "a2"}, {"role": "user", "content": "obs3"},
               {"role": "assistant", "content": "a3"}]
    prefix, tail = split_at_latest_observation(history, keep=2)
    assert [m["content"] for m in tail] == ["obs2", "a2", "obs3", "a3"] and len(prefix) == 3
    assert split_at_latest_observation(history, keep=9)[1][0]["content"] == "obs1"
    assert split_at_latest_observation(history)[1][0]["content"] == "obs3"

    class Owner:
        compaction_keep_observations = 3
        limit = 4

        def estimate_prompt_tokens(self, messages, tools):
            return len(messages)

        def summarize(self, messages):
            return "summary"

    class Analyzer(CompactionMixin):
        _context_budget_tokens = 6

    analyzer = Analyzer()
    analyzer._owner = Owner()
    messages = [{"role": "system", "content": "sys"}, *history]
    out = analyzer._trim_messages_for_context(messages)
    assert [m["content"] for m in out][2:] == ["obs1", "a1", "t1", "obs2", "a2", "obs3", "a3"][-4:] or len(out) <= 6
    # Three observations do not fit a 6-message budget: fall back to the latest one.
    assert out[2]["content"] == "obs3" and len(out) == 4 and messages[1]["content"] == "obs1"
    analyzer._context_budget_tokens = 8
    out = analyzer._trim_messages_for_context(messages + [{"role": "user", "content": "obs4"}] * 3)
    assert [m["content"] for m in out[2:]] == ["obs4"] * 3 and len(out) == 5  # keep=3 from the end
