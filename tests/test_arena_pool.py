"""the normalized budget clock and the audition pool."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_arena_core import Now

from arena.clock import ActiveClock, GlobalClock, Normalization
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / 'configs/arena_30m.json').read_text())
NORMALIZATION = dict(rate_tokens_per_second=100.0, context_weight=0.5, context_reference=100000,
                     request_overhead_seconds=1.0)


def test_normalized_clock_charges_tokens_and_tool_time_not_latency():
    now = Now()
    wall = GlobalClock(900, now=now)
    clock = ActiveClock(wall, now=now, budget_clock='normalized', normalization=NORMALIZATION)
    wall.release()
    clock.release()
    with clock.inference():
        now.advance(300)  # a slow, contended request costs what it generated, not its latency
    # 1 s overhead + 1000 tokens / 100 tok/s * (1 + 0.5 * 50000 / 100000) = 1 + 12.5
    assert clock.charge(50000, 1000) == pytest.approx(13.5)
    with clock.tool():
        now.advance(4)
    assert clock.inference_seconds() == 300 and clock.budget_seconds() == pytest.approx(17.5)
    assert clock.normalized_seconds() == clock.budget_seconds()
    assert clock.charges == [dict(wall_seconds=300.0, observed_wall_seconds=300.0,
                                  purpose='gameplay', prompt_tokens=50000,
                                  completion_tokens=1000, cost_seconds=13.5)]
    clock.grant(20)
    clock.charge(0, 500, 'compaction')  # 1 + 5
    assert clock.budget_remaining() == pytest.approx(14) and clock.feedback()['game_normalized_seconds'] == pytest.approx(23.5)
    busy = ActiveClock(wall, now=now, budget_clock='busy')
    assert busy.charge(10, 10) is None and busy.charges[0]['cost_seconds'] is None and busy.normalization is None
    with pytest.raises(ValueError, match='normalized budget clock only'):
        ActiveClock(wall, now=now, budget_clock='busy', normalization=NORMALIZATION)
    for bad in (dict(rate_tokens_per_second=0), dict(context_weight=-1), dict(request_overhead_seconds=float('nan'))):
        with pytest.raises(ValueError):
            Normalization(**bad)
    defaults = Normalization()
    assert defaults.summary() == dict(rate_tokens_per_second=104.0, context_weight=0.25, context_reference=131072,
                                      request_overhead_seconds=1.65)
    # A 128K prompt costs a quarter more per token than an empty one.
    assert defaults.cost(131072, 1040) == pytest.approx(1.65 + 10 * 1.25)


def make_pool_games(now, count=8):
    wall = GlobalClock(1800, now=now)
    games = [GameRecord(f'g{i}', i, ActiveClock(wall, now=now, budget_clock='normalized', normalization=NORMALIZATION))
             for i in range(count)]
    return wall, games


def charge(game, tokens):
    game.clock.charge(0, tokens)  # 1 s + tokens / 100 tok/s


def setup(count=8, pool=4, **extra):
    now = Now()
    wall, games = make_pool_games(now, count)
    sched = Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', audition_pool=pool,
                      pool_stall_actions=12, pool_action_limit=60, **extra)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.advance(100)
    charge(games[0], 900)   # T1 = 10 normalized seconds
    games[0].note_level(1, wall.elapsed())
    now.advance(50)
    charge(games[1], 1900)  # T1 = 20
    games[1].note_level(1, wall.elapsed())
    sched.tick()
    for g in games:
        g.observer = sched
    return now, wall, games, sched


def test_pool_runs_the_tied_level_zero_games_in_parallel_until_one_clears():
    now, wall, games, sched = setup()
    g0, g1 = games[0], games[1]
    assert sched.phase == 'steady' and sched.focus() == [g0, g1] and sched.pool_games() == []
    assert g0.clock.budget == 40 and g1.clock.budget == 80  # 2*2*T1 on the normalized clock
    # g0 exhausts its budget: the freed slot opens a pool instead of auditioning one game.
    charge(g0, 3900)
    sched.tick()
    assert g0.state == 'parked'
    pool = sched.pool_games()
    assert [g.index for g in pool] == [2, 3, 4, 5] and all(g.pool and g.clock.budget is None for g in pool)
    assert all(g.readmissions == 0 and g.pool_admissions == 1 for g in pool)
    assert [e['reason'] for e in games[2].admissions][-1] == 'pool' and games[2].admissions[-1]['budget'] is None
    assert [g.index for g in sched.backlog()] == [6, 7, 0]
    # The pool is contention-invariant: g4 clears after 300 wall seconds having generated 2900 tokens.
    now.advance(300)
    charge(games[4], 2900)  # T1 = 30 (opening charge 0 + 1 + 29)
    games[4].report_level(1, wall.elapsed())  # forwards to Scheduler.on_level
    assert games[4].state == 'active' and not games[4].pool and games[4].clock.budget == 120  # 2*2*30
    assert games[4].admissions[-1]['reason'] == 'pool_clear' and games[4].readmissions == 1 and games[4].pool_clears == 1
    # Both slots are focus games now: the rest of the pool parks with its order unchanged.
    assert sched.focus() == [g1, games[4]] and sched.pool_games() == []
    assert all(g.state == 'parked' and g.admissions[-1]['reason'] == 'pool_close' and g.readmissions == 0
               for g in (games[2], games[3], games[5]))
    # They lead the next pool (with g6, g7 they form the tied group) once a slot frees again.
    charge(g1, 8000)
    sched.tick()
    assert g1.state == 'parked' and [g.index for g in sched.pool_games()] == [2, 3, 5, 6]
    summary = sched.summary()
    assert summary['audition_pool'] == 4 and summary['pool_stall_actions'] == 12 and summary['pool_action_limit'] == 60
    assert summary['normalization'] == NORMALIZATION


def test_pool_games_park_when_they_stall_and_drop_behind_the_untried_games():
    now, wall, games, sched = setup()
    charge(games[0], 3900)
    sched.tick()
    g2 = games[2]
    assert g2.pool
    for _ in range(11):
        g2.note_action('same board')
    sched.tick()
    assert g2.pool and g2.actions_since_novel == 10  # the first action showed a new board
    g2.note_action('same board')
    g2.note_action('same board')
    sched.tick()
    assert g2.state == 'parked' and g2.admissions[-1]['reason'] == 'pool_stall' and g2.readmissions == 1 and g2.stalls == 1
    assert [g.index for g in sched.pool_games()] == [3, 4, 5, 6]  # the next tied game took its place
    assert sched.backlog()[-1] is g2 or g2.readmissions > games[7].readmissions
    g3 = games[3]
    for step in range(60):
        g3.note_action(f'board {step}')  # always novel: the action limit still ends the audition
    sched.tick()
    assert g3.state == 'parked' and g3.admissions[-1]['reason'] == 'pool_limit' and g3.readmissions == 1
    # A clear resets the counters.
    g4 = games[4]
    for step in range(30):
        g4.note_action(f'b{step}')
    g4.note_level(1, wall.elapsed())
    assert g4.actions_on_level == 0 and g4.actions_since_novel == 0 and g4.frames_on_level == set()


def test_pool_uses_both_slots_when_no_focus_game_is_active_and_a_parked_clear_still_goes_first():
    now, wall, games, sched = setup(count=12)
    charge(games[0], 3900)
    charge(games[1], 7900)
    sched.tick()
    assert sched.focus() == [] and [g.index for g in sched.pool_games()] == [2, 3, 4, 5, 6, 7, 8, 9]
    # A clear that arrives on a parked game outranks the pool: it takes a slot as a focus game and
    # the pool shrinks to the capacity the remaining slot leaves.
    late = games[0]
    late.report_level(2, wall.elapsed())
    assert late.pending_parked_progress
    sched.tick()
    assert late.state == 'active' and not late.pool and late.admissions[-1]['reason'] == 'parked_progress'
    assert late.clock.budget == 60  # 2*3*10
    assert [g.index for g in sched.pool_games()] == [2, 3, 4, 5]
    assert all(g.admissions[-1]['reason'] == 'pool_close' for g in games[6:10])


def test_pool_and_normalized_clock_are_rejected_off_their_policies():
    now = Now()
    wall, games = make_pool_games(now, 3)
    with pytest.raises(ValueError, match='fixed-2x policy and the readmissions order'):
        Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='spent', audition_pool=4)
    with pytest.raises(ValueError, match='require an audition pool'):
        Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', pool_stall_actions=5)
    with pytest.raises(ValueError, match='positive integer'):
        Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', audition_pool=0)


def test_duck_adapter_charges_replies_and_reports_board_identity():
    from arena.duck import ArenaDuckMixin

    now = Now()
    wall = GlobalClock(900, now=now)
    clock = ActiveClock(wall, now=now, budget_clock='normalized', normalization=NORMALIZATION)
    wall.release()
    clock.release()
    game = GameRecord('g', 0, clock)
    boards = iter([np.zeros((64, 64), dtype=np.uint8), np.ones((64, 64), dtype=np.uint8),
                   np.ones((64, 64), dtype=np.uint8)])

    class Base:
        def __init__(self):
            self.obs = None
            self.last_prompt_tokens = 0

        async def step(self, action):
            self.obs = type('Obs', (), {'board': next(boards), 'levels': 0, 'state': 'NOT_FINISHED'})()

        def record_reply(self, reply, purpose='gameplay'):
            self.last_prompt_tokens = reply['usage']['prompt_tokens']

    class Agent(ArenaDuckMixin, Base):
        pass

    agent = Agent()
    agent.game = game
    agent.notes = None
    import asyncio
    for _ in range(3):
        asyncio.run(agent.step(None))
    assert game.actions_on_level == 3 and game.actions_since_novel == 1 and len(game.frames_on_level) == 2
    agent.record_reply({'usage': {'prompt_tokens': 50000, 'completion_tokens': 1000}}, 'compaction')
    assert clock.budget_seconds() == pytest.approx(13.5) and clock.charges[0]['purpose'] == 'compaction'
    assert game.max_prompt_tokens == 50000
    row = game_row(game, {'game_id': 'g', 'levels': 0})
    assert row['charges'] == clock.charges and row['normalized_seconds'] == pytest.approx(13.5)
    assert row['pool_admissions'] == 0 and row['stalls'] == 0


def test_audit_models_the_normalized_clock_and_pool_events():
    norm = Normalization(**NORMALIZATION)

    def ev(kind, reason, wall, active, level=0, budget=None, budget_seconds=0.0):
        return dict(event=kind, reason=reason, wall_seconds=wall, active_seconds=active, levels=level, budget=budget,
                    budget_seconds=budget_seconds)

    def ch(wall, prompt, completion):
        return dict(wall_seconds=wall, purpose='gameplay', prompt_tokens=prompt, completion_tokens=completion,
                    cost_seconds=round(norm.cost(prompt, completion), 3))

    # p: parked at the switch (10 s), pool admission at 40 s, one 900-token reply (T1 = 10) at 60 s,
    # promoted with 2*2*10 = 40; tool time 2 s at 62-64 s; wall end at 100.
    p = dict(game_id='p', active_seconds=70, t1_seconds=10, budget_seconds=12, inference_seconds=30, tool_seconds=2,
             inference_intervals=[dict(started_wall_seconds=0, ended_wall_seconds=10),
                                  dict(started_wall_seconds=40, ended_wall_seconds=60)],
             tool_intervals=[dict(started_wall_seconds=62, ended_wall_seconds=64)],
             charges=[ch(60, 0, 900)],
             time_to_level=[dict(level=1, wall_seconds=60, active_seconds=30, budget_seconds=10)],
             admissions=[ev('admit', 'opening_start', 0, 0), ev('park', 'opening_switch', 10, 10),
                         ev('admit', 'pool', 40, 10), ev('grant', 'pool_clear', 60, 30, 1, 40, 10),
                         ev('finish', 'wall_end', 100, 70, 1, budget_seconds=12)])
    # a: the opening focus game, T1 = 10 at 10 s from a 900-token reply, level-2 budget 40 at the switch.
    a = dict(game_id='a', active_seconds=100, t1_seconds=10, budget_seconds=10, inference_seconds=10, tool_seconds=0,
             inference_intervals=[dict(started_wall_seconds=0, ended_wall_seconds=10)], tool_intervals=[],
             charges=[ch(10, 0, 900)],
             time_to_level=[dict(level=1, wall_seconds=10, active_seconds=10, budget_seconds=10)],
             admissions=[ev('admit', 'opening_start', 0, 0), ev('grant', 'opening_switch', 10, 10, 1, 40, 10),
                         ev('finish', 'wall_end', 100, 100, 1, budget_seconds=10)])
    # q: pool game that stalled at 55 s and was closed out of the next pool at 80 s.
    q = dict(game_id='q', active_seconds=35, t1_seconds=None, budget_seconds=0, inference_seconds=0, tool_seconds=0,
             inference_intervals=[], tool_intervals=[], charges=[], time_to_level=[],
             admissions=[ev('admit', 'opening_start', 0, 0), ev('park', 'opening_switch', 10, 10),
                         ev('admit', 'pool', 40, 10), ev('park', 'pool_stall', 55, 25),
                         ev('admit', 'pool', 70, 25), ev('park', 'pool_close', 80, 35),
                         ev('finish', 'wall_end', 100, 35)])
    data = dict(wall_seconds=100, config={'gameplay_seconds': 100}, games=[a, p, q],
                scheduler=dict(active_limit=2, opening_switch_level1_games=1, switch_wall_seconds=10,
                               budget_clock='normalized', normalization=norm.summary(), audition_pool=4,
                               budget_policy='fixed-2x', backlog_policy='readmissions'))
    out = audit_scheduler(data)
    assert out['valid'], out['issues']
    assert out['checks']['grants'] == 2 and out['checks']['pool_admissions'] == 3
    broken = copy.deepcopy(data)
    broken['games'][1]['admissions'][3]['budget'] = 60
    assert any('pool_clear grant' in i for i in audit_scheduler(broken)['issues'])
    broken = copy.deepcopy(data)
    broken['games'][1]['charges'][0]['cost_seconds'] = 5
    assert any('charge at' in i for i in audit_scheduler(broken)['issues'])
    broken = copy.deepcopy(data)
    broken['games'][1]['budget_seconds'] = 30  # latency is not cost
    assert any('budget total' in i for i in audit_scheduler(broken)['issues'])
    broken = copy.deepcopy(data)
    del broken['scheduler']['normalization']
    assert 'normalization parameters' in audit_scheduler(broken)['issues'][0]
    broken = copy.deepcopy(data)
    del broken['scheduler']['audition_pool']
    assert not audit_scheduler(broken)['valid']


def test_pool_variant_composes_on_the_fixed_2x_base():
    base = apply_variant(CONTROL, 'reclaim-fixed-2x')
    pool = apply_variant(CONTROL, 'fixed-2x-pool')
    assert pool['budget_clock'] == 'normalized' and pool['audition_pool'] == 4
    assert pool['pool_stall_actions'] == 12 and pool['pool_action_limit'] == 60 and pool.get('t1_cap_seconds') is None
    assert pool['cost_normalization'] == Normalization().summary()
    assert pool['arena_variant']['reference_variant'] == 'reclaim-fixed-2x'
    assert set(pool['arena_variant']['reference_difference']) == {
        'budget_clock', 'cost_normalization', 'audition_pool', 'pool_stall_actions', 'pool_action_limit'}
    assert pool['arena_variant']['checkpoint_and_serving_unchanged'] is True
    assert pool['server'] == base['server'] and pool['windows'] == base['windows'] and pool['active_limit'] == 2
    three = apply_variant(CONTROL, 'fixed-2x-80k-three')
    assert three['windows'] == [81920] and three['active_limit'] == 3 and three['compaction_keep_observations'] == 3
    assert three['budget_clock'] == 'busy' and three['cache_experiment']['compaction_changed'] is True
    combo = apply_variant(CONTROL, 'fixed-2x-pool2-cap')
    assert combo['budget_clock'] == 'normalized' and combo['audition_pool'] == 2
    assert combo['pool_stall_actions'] == 12 and combo['pool_action_limit'] == 60
    assert combo['focus_stall_actions'] == 15 and combo['opening_max_wall_seconds'] == 360
    assert combo['cost_normalization'] == Normalization().summary() and combo.get('t1_cap_seconds') is None
    assert set(combo['arena_variant']['reference_difference']) == {
        'budget_clock', 'cost_normalization', 'audition_pool', 'pool_stall_actions', 'pool_action_limit',
        'focus_stall_actions', 'opening_max_wall_seconds'}
    assert combo['server'] == base['server'] and combo['windows'] == base['windows'] and combo['active_limit'] == 2
    novelty = apply_variant(CONTROL, 'fixed-2x-pool2-timed-novelty')
    assert novelty['level0_order'] == 'novelty' and combo.get('level0_order') is None
    assert novelty['pool_time_limit_seconds'] == 120 and combo.get('pool_time_limit_seconds') is None
    assert {k: v for k, v in novelty.items() if k not in ('level0_order', 'pool_time_limit_seconds', 'level_budget_rule', 'arena_variant')} == {
        k: v for k, v in combo.items() if k not in ('level_budget_rule', 'arena_variant')}
    assert set(novelty['arena_variant']['reference_difference']) == set(combo['arena_variant']['reference_difference']) | {'level0_order', 'pool_time_limit_seconds'}


def test_pool_time_limit_parks_a_long_audition_and_counts_one_readmission():
    now = Now()
    wall, games = make_pool_games(now, 5)
    with pytest.raises(ValueError, match='pool_time_limit_seconds'):
        Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', pool_time_limit_seconds=120)
    sched = Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', audition_pool=2,
                      pool_stall_actions=12, pool_action_limit=60, opening_max_wall_seconds=360, pool_time_limit_seconds=120)
    wall.release()
    for g in games:
        g.clock.release()
        g.observer = sched
    sched.start_all()
    now.advance(100)
    charge(games[0], 900)
    games[0].report_level(1, wall.elapsed())
    now.advance(260)
    sched.tick()
    sched.tick()
    assert [g.game_id for g in sched.pool_games()] == ['g1', 'g2']
    assert all(g.clock.budget == 120 and g.admissions[-1]['budget'] == 120 for g in sched.pool_games())
    charge(games[1], 11000)  # 111 normalized seconds: under the limit
    sched.tick()
    assert games[1].state == 'active'
    charge(games[1], 1000)   # 122 s: over it
    sched.tick()
    sched.tick()
    assert games[1].state == 'parked' and games[1].admissions[-1]['reason'] == 'pool_timeout'
    assert games[1].readmissions == 1 and games[1].stalls == 1
    assert [g.game_id for g in sched.pool_games()] == ['g2', 'g3']
    assert sched.summary()['pool_time_limit_seconds'] == 120


def test_novelty_order_admits_the_level_0_game_that_showed_the_most_boards_and_keeps_the_pool_tie_group():
    now = Now()
    wall, games = make_pool_games(now, 6)
    with pytest.raises(ValueError, match='level0_order'):
        Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', level0_order='random')
    sched = Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', audition_pool=2,
                      pool_stall_actions=12, pool_action_limit=60, opening_max_wall_seconds=360, level0_order='novelty')
    wall.release()
    for g in games:
        g.clock.release()
        g.observer = sched
    sched.start_all()
    now.advance(100)
    charge(games[0], 900)
    games[0].report_level(1, wall.elapsed())
    for key in ('a', 'b', 'c'):
        games[4].note_action(key)   # three distinct boards in the opening
    for key in ('a', 'b'):
        games[2].note_action(key)
    games[1].note_action('a')
    games[1].note_action('a')
    now.advance(260)
    sched.tick()
    sched.tick()
    # The pool takes the tie group (readmissions 0, level 0) in novelty order: games[4] (3 boards), games[2] (2).
    assert sched.focus() == [games[0]] and sorted(g.game_id for g in sched.pool_games()) == ['g2', 'g4']  # roster order would take g1, g2
    assert [g.game_id for g in sched.backlog()[:3]] == ['g1', 'g3', 'g5']  # one board, then roster order
    assert sched.summary()['level0_order'] == 'novelty'
    # A cleared game is ordered by levels, not by boards: novelty only ranks level-0 games.
    assert sched.priority(games[0])[3] == 0


def test_pool_of_two_fills_the_slot_the_opening_cap_frees_and_rotates_stalled_auditions():
    now = Now()
    wall, games = make_pool_games(now, 6)
    sched = Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions', audition_pool=2,
                      pool_stall_actions=12, pool_action_limit=60, focus_stall_actions=15, opening_max_wall_seconds=360)
    wall.release()
    for g in games:
        g.clock.release()
        g.observer = sched
    sched.start_all()
    now.advance(100)
    charge(games[0], 900)
    games[0].report_level(1, wall.elapsed())
    now.advance(260)
    sched.tick()
    assert sched.phase == 'steady' and sched.switch_reason == 'opening_cap'
    sched.tick()
    # One clearer keeps a focus slot; the other slot runs two tied level-0 games as a pool.
    assert sched.focus() == [games[0]] and sched.pool_games() == [games[1], games[2]]
    assert all(g.admissions[-1]['reason'] == 'pool' and g.clock.budget is None for g in sched.pool_games())
    assert sched.pool_capacity() == 2
    # A stalled audition rotates out (readmissions + 1) and the next tied game enters the pool.
    for _ in range(13):  # the first action shows the board; twelve more without a novel one stall the audition
        games[1].note_action('same')
    sched.tick()
    sched.tick()
    assert games[1].state == 'parked' and games[1].admissions[-1]['reason'] == 'pool_stall' and games[1].readmissions == 1
    assert sched.pool_games() == [games[2], games[3]]
    # A pool clear promotes the game to focus and closes the pool.
    charge(games[2], 500)
    games[2].report_level(1, wall.elapsed())
    sched.tick()
    assert games[2] in sched.focus() and games[2].admissions[-1]['reason'] == 'pool_clear'
    assert games[3].state == 'parked' and games[3].admissions[-1]['reason'] == 'pool_close' and games[3].readmissions == 0
    assert sched.summary()['audition_pool'] == 2 and sched.summary()['opening_max_wall_seconds'] == 360


def test_focus_stall_parks_and_the_opening_cap_switches_without_two_clears():
    now = Now()
    wall, games = make_pool_games(now, 6)
    sched = Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions',
                      focus_stall_actions=15, opening_max_wall_seconds=360)
    wall.release()
    for g in games:
        g.clock.release()
        g.observer = sched
    sched.start_all()
    now.advance(100)
    charge(games[0], 900)
    games[0].report_level(1, wall.elapsed())
    now.advance(259)
    sched.tick()
    assert sched.phase == 'opening'  # one clear, 359 s
    now.advance(1)
    sched.tick()
    assert sched.phase == 'steady' and sched.switch_reason == 'opening_cap' and sched.switch_wall_seconds == 360
    sched.tick()  # the switch tick parks; the next tick fills the free slot
    # The single clearer keeps its slot; the other slot goes to the backlog by the usual order.
    assert sched.focus()[0] is games[0] and len(sched.focus()) == 2 and sched.focus()[1] is games[1]
    assert games[1].admissions[-1]['reason'] == 'regrant' and games[1].clock.budget == games[1].level0_grant
    # Fifteen actions without a board new to the level park the focus game and drop it one place.
    g0 = games[0]
    g0.note_action('a')
    for _ in range(14):
        g0.note_action('a')
    sched.tick()
    assert g0.state == 'active'
    g0.note_action('a')
    sched.tick()
    assert g0.state == 'parked' and g0.admissions[-1]['reason'] == 'stall' and g0.readmissions == 2 and g0.stalls == 1
    assert sched.summary()['focus_stall_actions'] == 15 and sched.summary()['opening_max_wall_seconds'] == 360
    # A clear-triggered switch is still recorded as such, and the cap alone also switches with no clears.
    wall2, games2 = make_pool_games(Now(), 3)
    with pytest.raises(ValueError, match='positive'):
        Scheduler(games2, wall2, budget_policy='fixed-2x', backlog_policy='readmissions', focus_stall_actions=0)
    with pytest.raises(ValueError, match='positive'):
        Scheduler(games2, wall2, budget_policy='fixed-2x', backlog_policy='readmissions', opening_max_wall_seconds=-1)


def test_audit_accepts_a_capped_opening_switch_and_stall_parks():
    def ev(kind, reason, wall, active, level=0, budget=None, budget_seconds=0.0):
        return dict(event=kind, reason=reason, wall_seconds=wall, active_seconds=active, levels=level, budget=budget,
                    budget_seconds=budget_seconds)
    # Busy clock; one clear at 10 s; the opening cap at 40 s ends the opening without a second clear.
    a = dict(game_id='a', active_seconds=90, t1_seconds=10, budget_seconds=60, inference_seconds=60, tool_seconds=0,
             inference_intervals=[dict(started_wall_seconds=0, ended_wall_seconds=10),
                                  dict(started_wall_seconds=40, ended_wall_seconds=90)], tool_intervals=[],
             time_to_level=[dict(level=1, wall_seconds=10, active_seconds=10, budget_seconds=10)],
             admissions=[ev('admit', 'opening_start', 0, 0), ev('grant', 'opening_switch', 40, 40, 1, 40, 10),
                         ev('park', 'stall', 90, 90, 1, 40, 60), ev('finish', 'wall_end', 100, 90, 1, budget_seconds=60)])
    b = dict(game_id='b', active_seconds=100, t1_seconds=None, budget_seconds=30, inference_seconds=30, tool_seconds=0,
             inference_intervals=[dict(started_wall_seconds=0, ended_wall_seconds=30)], tool_intervals=[],
             time_to_level=[],
             admissions=[ev('admit', 'opening_start', 0, 0), ev('finish', 'wall_end', 100, 100, 0, budget_seconds=30)])
    data = dict(wall_seconds=100, config={'gameplay_seconds': 100}, games=[a, b],
                scheduler=dict(active_limit=2, opening_switch_level1_games=2, switch_wall_seconds=40,
                               opening_max_wall_seconds=40, budget_clock='busy', budget_policy='fixed-2x',
                               backlog_policy='readmissions'))
    out = audit_scheduler(data)
    assert out['valid'], out['issues']
    broken = copy.deepcopy(data)
    del broken['scheduler']['opening_max_wall_seconds']
    assert any('without enough distinct first clears' in i for i in audit_scheduler(broken)['issues'])
    broken = copy.deepcopy(data)
    broken['scheduler']['switch_wall_seconds'] = 60
    assert any('without enough' in i for i in audit_scheduler(broken)['issues'])
