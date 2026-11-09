"""Budget time follows actual admitted generations, not eligibility or queue time."""
import asyncio

import pytest
from test_arena_core import Now

from arena.clock import ActiveClock, GlobalClock
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler


def setup(count=3, seconds=900):
    now = Now()
    wall = GlobalClock(seconds, now=now)
    games = [GameRecord(str(i), i, ActiveClock(wall, now=now, budget_clock='inference'))
             for i in range(count)]
    scheduler = Scheduler(games, wall)
    wall.release()
    for game in games:
        game.clock.release()
        game.observer = scheduler
    scheduler.start_all()
    return now, wall, games, scheduler


def test_queue_and_tools_do_not_inflate_t1_or_level_budget_and_audit_catches_regression():
    now, wall, games, scheduler = setup()
    a, b, c = games
    now.advance(100)  # admission waiting, no generation
    with a.clock.inference():
        now.advance(10)
    now.advance(50)  # Python execution, no generation
    a.report_level(1, wall.elapsed())
    assert a.t1 == 10 and a.clock.active_seconds() == 160
    with b.clock.inference():
        now.advance(20)
    b.report_level(1, wall.elapsed())
    scheduler.tick()
    assert a.clock.budget == 30  # preserved minimum allocation floor
    assert b.clock.budget == 40
    now.advance(40)  # queue/tool time after switch cannot exhaust the budget
    scheduler.tick()
    assert a.state == b.state == 'active'
    with a.clock.inference('summary'):
        now.advance(30)
        assert a.clock.budget_exhausted()  # includes a request still in flight
        scheduler.tick()
        assert a.state == 'parked'
        now.advance(2)  # already issued inference still counts while parked
    assert a.spent_on_level() == 32
    scheduler.wall_end()
    report = dict(wall_seconds=wall.elapsed(), games=[game_row(g, {}) | {'game_id': g.game_id}
                                                     for g in games], scheduler=scheduler.summary())
    result = audit_scheduler(report)
    assert result['valid'], result['issues']
    report['games'][0]['t1_seconds'] = 160
    assert not audit_scheduler(report)['valid']


@pytest.mark.parametrize('exception', [ValueError, asyncio.CancelledError])
def test_failed_cancelled_requests_charged_and_deadline_clips_cleanup(exception):
    now, wall, games, _ = setup(seconds=20)
    clock = games[0].clock
    now.advance(8)
    with pytest.raises(exception), clock.inference():
        now.advance(5)
        raise exception()
    assert clock.inference_seconds() == 5
    clock.pause()
    now.advance(2)
    with clock.inference('summary'):
        now.advance(20)
    assert clock.inference_seconds() == 10
    assert clock.inference_intervals[-1]['ended_wall_seconds'] == 20
    assert clock.remaining() == 0


def test_overlapping_generations_rejected_without_losing_outer_accounting():
    now, _, games, _ = setup()
    with games[0].clock.inference():
        now.advance(3)
        with pytest.raises(RuntimeError), games[0].clock.inference():
            pass
    assert games[0].clock.inference_seconds() == 3


async def test_duck_request_hook_excludes_tokenizer_and_counts_summary_failure(tmp_path):
    from test_arena_duck import arena_agent, text_reply

    from arena.duck import ArenaTransferDuckAgent

    class Transport:
        async def post(self, url, **kwargs):
            now.advance(7)
            if not kwargs['json'].get('tools'):
                raise RuntimeError('summary failed')
            return text_reply('ready')

    agent, _, game, _ = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    now, wall, measured, _ = setup(count=1)
    game.clock = measured[0].clock
    now.advance(100)  # exact tokenization / admission happens outside request_completion
    await agent.request_completion({'tools': [{'type': 'function'}]})
    with pytest.raises(RuntimeError):
        await agent.request_completion({})
    assert game.clock.inference_seconds() == 14
    assert [r['purpose'] for r in game.clock.inference_intervals] == ['gameplay', 'summary']
    assert game.clock.inference_intervals[-1]['status'] == 'RuntimeError'
