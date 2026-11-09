"""evening: busy clock, fixed 2*L*T1 budgets, readmission order."""
import importlib.util
import json
from pathlib import Path

import pytest
from test_arena_core import Now, make_games

from arena.clock import ActiveClock, GlobalClock
from arena.scheduler import Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / 'configs/arena_30m.json').read_text())


def test_busy_clock_charges_tool_time_once_and_never_overlaps_inference():
    now = Now()
    wall = GlobalClock(900, now=now)
    clock = ActiveClock(wall, now=now, budget_clock='busy')
    wall.release()
    clock.release()
    with clock.inference():
        now.advance(10)
    with clock.tool():
        now.advance(4)
        with clock.tool():  # nested calls charge once
            now.advance(2)
    now.advance(30)  # waiting at the gate costs nothing
    assert clock.inference_seconds() == 10 and clock.tool_seconds() == 6
    assert clock.budget_seconds() == clock.busy_seconds() == 16
    assert [i['purpose'] for i in clock.tool_intervals] == ['python']
    assert clock.feedback()['game_tool_seconds'] == 6
    clock.grant(20)
    with clock.tool():
        now.advance(5)
    assert clock.budget_remaining() == 15 and not clock.budget_exhausted()
    inference_only = ActiveClock(wall, now=now, budget_clock='inference')
    inference_only.release()
    with inference_only.tool():
        now.advance(9)
    assert inference_only.budget_seconds() == 0
    with pytest.raises(ValueError):
        ActiveClock(wall, now=now, budget_clock='tools')


def setup(policy='fixed-2x', backlog='readmissions'):
    now = Now()
    wall, games = make_games(now, count=6)
    sched = Scheduler(games, wall, budget_policy=policy, backlog_policy=backlog)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.advance(100)
    games[0].note_level(1, wall.elapsed())   # T1 = 100
    now.advance(50)
    games[1].note_level(1, wall.elapsed())   # T1 = 150
    sched.tick()
    return now, games, sched


def spend(now, game, seconds):
    game.clock.resume()
    now.advance(seconds)
    game.clock.pause()


def test_fixed_2x_budgets_are_full_per_admission_and_level_zero_uses_the_first_park_spend():
    now, games, sched = setup()
    g0, g1 = games[0], games[1]
    # Opening switch: 2*2*T1 for level 2, minus what was already spent on it (50 s for g0, 0 for g1).
    assert g0.clock.budget == 350 and g1.clock.budget == 600
    assert g0.readmissions == g1.readmissions == 1
    # The four parked games remember their first-park spend (150 s of opening).
    assert all(g.level0_grant == 150 for g in games[2:])
    # A level clear grants the full 2*L*T1 (not the remainder).
    g0.note_level(2, now.value)
    sched.on_level(g0)
    assert g0.clock.budget == 600  # 2*3*100
    now.advance(600)
    sched.tick()
    assert g0.state == 'parked'
    # Re-admission at level 2 is again the fixed 600, whatever was spent before.
    assert sched.grant_for(g0) == 600
    # Level-0 re-admission is the fixed first-park spend, every time.
    g2 = games[2]
    assert sched.grant_for(g2) == 150
    spend(now, g2, 300)
    assert sched.grant_for(g2) == 150
    assert sched.summary()['budget_policy'] == 'fixed-2x' and sched.summary()['backlog_policy'] == 'readmissions'
    with pytest.raises(ValueError, match='no caps'):
        Scheduler(games, GlobalClock(900, now=now), budget_policy='fixed-2x', regrant_multiple=2)
    # T1 cap: a 100 s T1 capped at 60 makes the level-2 budget 2*2*60 = 240, level 3 = 360.
    capped = Scheduler(games, GlobalClock(900, now=now), budget_policy='fixed-2x', backlog_policy='readmissions',
                       t1_cap_seconds=60)
    assert capped.level_budget(g0) == 360 and capped.summary()['t1_cap_seconds'] == 60  # g0 holds 2 levels
    assert Scheduler(games, GlobalClock(900, now=now), budget_policy='fixed-2x', t1_cap_seconds=500).level_budget(g0) == 600
    for bad in (dict(budget_policy='incremental', t1_cap_seconds=120), dict(budget_policy='fixed-2x', t1_cap_seconds=0.5)):
        with pytest.raises(ValueError, match='t1_cap'):
            Scheduler(games, GlobalClock(900, now=now), **bad)
    with pytest.raises(ValueError, match='Unknown budget'):
        Scheduler(games, GlobalClock(900, now=now), budget_policy='double')
    # T1 clip: the unit is clipped to [0.5, 2] x the median T1 of the games with a clear.
    clipped = Scheduler(games, GlobalClock(900, now=now), budget_policy='fixed-2x', t1_clip=[0.5, 2])
    others = [g for g in games if g is not g0]
    others[0].t1, others[1].t1 = 400.0, 40.0  # median of (100, 400, 40) = 100
    assert clipped.budget_unit(g0) == 100 and clipped.budget_unit(others[0]) == 200
    assert clipped.budget_unit(others[1]) == 50 and clipped.summary()['t1_clip'] == [0.5, 2.0]
    assert clipped.level_budget(g0) == 2 * 3 * 100
    others[0].t1 = others[1].t1 = None
    for bad in (dict(budget_policy='incremental', t1_clip=[0.5, 2]), dict(budget_policy='fixed-2x', t1_clip=[1.5, 2])):
        with pytest.raises(ValueError, match='t1_clip'):
            Scheduler(games, GlobalClock(900, now=now), **bad)
    # Timing variants: multiple * L * T1 (control 2); g0 plays level 3 with T1 = 100.
    for multiple, budget in ((1, 300), (4, 1200)):
        timed = Scheduler(games, GlobalClock(900, now=now), budget_policy='fixed-2x', backlog_policy='readmissions',
                          budget_multiple=multiple)
        assert timed.level_budget(g0) == budget and timed.summary()['budget_multiple'] == multiple
    for bad in (dict(budget_policy='incremental', budget_multiple=4), dict(budget_policy='fixed-2x', budget_multiple=0)):
        with pytest.raises(ValueError, match='budget_multiple'):
            Scheduler(games, GlobalClock(900, now=now), **bad)


def test_readmission_order_prefers_parked_clears_then_fewest_admissions_then_most_levels():
    now, games, sched = setup()
    g0 = games[0]
    backlog = [g.index for g in sched.backlog()]
    assert backlog == [2, 3, 4, 5]  # never admitted in steady state, roster order
    # Park the level-1 focus game g0 by exhausting its budget: it has one admission and one level.
    now.advance(350)
    sched.tick()
    assert g0.state == 'parked' and games[2].state == 'active'
    # It waits behind every game that has not had its second try...
    assert [g.index for g in sched.backlog()] == [3, 4, 5, 0]
    # ...and once they have, its level puts it ahead of them.
    for _ in range(3):
        now.advance(150)
        sched.tick()
    order = [g.index for g in sched.backlog()]
    assert order[0] == 0 or g0.state == 'active'
    # A clear that arrives while parked always goes first, even with more admissions.
    late = next(g for g in games if g.state == 'parked' and g.readmissions >= 1 and g is not g0)
    late.readmissions = 5
    late.note_level(1, now.value)
    sched.on_level(late)
    assert late.pending_parked_progress and sched.backlog()[0] is late
    assert sched.prioritize_parked_progress is True


def test_audit_models_the_busy_clock_and_fixed_2x_grants():
    def ev(kind, reason, wall, active, level=0, budget=None):
        return dict(event=kind, reason=reason, wall_seconds=wall, active_seconds=active, levels=level, budget=budget)

    def stamp(game):
        for e in game['admissions']:
            w = e['wall_seconds']
            e['budget_seconds'] = sum(max(0, min(i['ended_wall_seconds'], w) - i['started_wall_seconds'])
                                      for i in game['inference_intervals'] + game['tool_intervals'])
        return game
    # a: level 1 at 10 s (8 s inference + 2 s tool -> T1 = 10 on the busy clock); level-2 budget 2*2*10 = 40,
    # exhausted after 28 s inference + 12 s tool; re-admitted at 70 s with the same fixed 40.
    a = stamp(dict(game_id='a', active_seconds=100, t1_seconds=10, budget_seconds=66,
                   inference_seconds=52, tool_seconds=14,
                   inference_intervals=[dict(started_wall_seconds=0, ended_wall_seconds=8),
                                        dict(started_wall_seconds=22, ended_wall_seconds=50),
                                        dict(started_wall_seconds=70, ended_wall_seconds=86)],
                   tool_intervals=[dict(started_wall_seconds=8, ended_wall_seconds=10),
                                   dict(started_wall_seconds=50, ended_wall_seconds=62)],
                   time_to_level=[dict(level=1, wall_seconds=10, active_seconds=10, budget_seconds=10)],
                   admissions=[ev('admit', 'opening_start', 0, 0),
                               ev('grant', 'opening_switch', 10, 10, 1, 40),
                               ev('park', 'level_budget', 70, 70, 1, 40),
                               ev('admit', 'level_budget', 70, 70, 1, 40),
                               ev('finish', 'wall_end', 100, 100, 1)]))
    # c: parked at the switch with 10 s spent -> level-0 re-admission grant is the 30 s floor.
    c = stamp(dict(game_id='c', active_seconds=60, t1_seconds=None, time_to_level=[], budget_seconds=30,
                   inference_seconds=30, tool_seconds=0, tool_intervals=[],
                   inference_intervals=[dict(started_wall_seconds=0, ended_wall_seconds=20),
                                        dict(started_wall_seconds=60, ended_wall_seconds=70)],
                   admissions=[ev('admit', 'opening_start', 0, 0),
                               ev('park', 'opening_switch', 10, 10),
                               ev('admit', 'regrant', 50, 10, 0, 20),
                               ev('finish', 'wall_end', 100, 60)]))
    data = dict(wall_seconds=100, config={'gameplay_seconds': 100}, games=[a, c],
                scheduler=dict(active_limit=2, opening_switch_level1_games=1, switch_wall_seconds=10,
                               budget_clock='busy', budget_policy='fixed-2x', backlog_policy='readmissions'))
    out = audit_scheduler(data)
    assert [i for i in out['issues'] if 'regrant grant' in i] and len(out['issues']) == 1, out['issues']
    c['admissions'][2]['budget'] = 30
    out = audit_scheduler(data)
    assert out['valid'], out['issues']
    import copy
    pristine = copy.deepcopy(data)
    assert out['checks']['grants'] == 3 and out['checks']['budget_parks'] == 1
    a['admissions'][3]['budget'] = 60   # the incremental rule would regrant 1.5x; fixed-2x must not
    assert any('level_budget grant' in i for i in audit_scheduler(data)['issues'])
    a['admissions'][3]['budget'] = 40
    a['tool_seconds'] = 12
    assert any('tool total' in i for i in audit_scheduler(data)['issues'])
    # A T1 cap below the fixture's T1 changes every fixed-2x expectation (regression: the audit once
    # read the cap into the renewable-multiple variable and silently ignored it).
    pristine['scheduler']['t1_cap_seconds'] = 5  # binds: fixture T1 is 10
    capped = audit_scheduler(pristine)
    assert not capped['valid'] and any('grant' in i for i in capped['issues'])
    pristine['scheduler']['t1_cap_seconds'] = 10 ** 6
    assert audit_scheduler(pristine)['valid']
    pristine['scheduler']['t1_cap_seconds'] = 0.5
    assert not audit_scheduler(pristine)['valid'] and 'invalid T1 cap' in audit_scheduler(pristine)['issues'][0]


def test_fixed_2x_variant_and_competition_build_with_sanity_trial(tmp_path):
    candidate = apply_variant(CONTROL, 'reclaim-fixed-2x')
    assert candidate['arena_variant']['reference_variant'] == 'cache-reclaim'
    assert candidate['cache_experiment']['reclaim_skipped_blocks'] is True
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    out, study = builder.build('fixed-2x-competition', output=tmp_path / 'c', paired_30m=True,
                               arena_variant='reclaim-fixed-2x', competition=True, opening_burst_requests=3)
    # Competition builds take the token clock (29 September); --budget-clock time rebuilds the busy clock.
    assert study['budget_clock'] == 'tokens' and study['level_budget_policy'] == 'fixed-2x'
    assert study['backlog_policy'] == 'readmissions' and study['opening_burst_requests'] == 3
    assert '2*L*T1' in study['level_budget_rule'] and 'completion tokens' in study['level_budget_rule']
    assert '(floor 1,920 tokens)' in study['level_budget_rule'] and study['token_budget']['floor_tokens'] == 1920
    _, timed = builder.build('fixed-2x-competition-time', output=tmp_path / 'ct', paired_30m=True,
                             arena_variant='reclaim-fixed-2x', competition=True, opening_burst_requests=3,
                             budget_clock='time')
    assert timed['budget_clock'] == 'busy' and 'token_budget' not in timed
    assert '2*L*T1' in timed['level_budget_rule'] and 'tool execution' in timed['level_budget_rule']
    assert '(floor 30 s)' in timed['level_budget_rule']
    assert study['backlog_priority'].startswith('parked clears first')
    assert [t['name'] for t in study['trials']] == ['sanity-25g', 'duck-transfer']
    assert study['trials'][0]['competition'] is False and study['trials'][0]['wall_seconds'] == 1800
    assert study['wall_seconds'] == 3600 and study['notebook_seconds'] == 7500
    notebook = builder.notebook(study, 'runtime.blob', 'a' * 64, smoke=False)
    assert 'EXPECTED_PREVIEW_TRIALS = 2' in notebook['cells'][2]['source']
    assert "'--max-seconds', '7500', '--notebook-start-epoch'" in notebook['cells'][1]['source']
    # The stall-open360 variant may be a competition candidate (v5, 25 September); the cap is scaled per roster at run time.
    _, stall = builder.build('stall-competition', output=tmp_path / 's', paired_30m=True,
                             arena_variant='fixed-2x-stall-open360', competition=True, opening_burst_requests=3)
    assert stall['opening_max_wall_seconds'] == 360 and stall['focus_stall_actions'] == 15 and stall['budget_clock'] == 'tokens'
    # Short preview (26 September): the lane candidate, 110 instances for 900 s, no sanity trial.
    _, lane = builder.build('lane-competition', output=tmp_path / 'l', paired_30m=True, arena_variant='fixed-2x-index-lane',
                            competition=True, opening_burst_requests=3, preview_seconds=900, sanity_trial=False)
    assert [t['name'] for t in lane['trials']] == ['duck-transfer'] and lane['competition']['preview_seconds'] == 900
    assert lane['competition']['submission_seconds'] == 32400 and lane['notebook_seconds'] == 3600
    assert lane['wall_seconds'] == 900  # the bootstrap's preview budget check uses the study wall
    assert lane['background_lane'] and lane['level_budget_policy'] == 'index'
    lane_notebook = builder.notebook(lane, 'runtime.blob', 'a' * 64, smoke=False)
    assert 'EXPECTED_PREVIEW_TRIALS = 1' in lane_notebook['cells'][2]['source'] and '900 s of gameplay' in lane_notebook['cells'][0]['source']
    _, short = builder.build('lane-rehearsal', output=tmp_path / 'r', paired_30m=True, arena_variant='fixed-2x-index-lane',
                             competition=True, rerun_rehearsal=True, opening_burst_requests=3, rehearsal_seconds=3000)
    assert short['competition']['rehearsal'] and short['competition']['submission_seconds'] == 3000 and short['kernel_timeout'] == 3600
    with pytest.raises(ValueError, match='Competition requires'):
        builder.build('pool-competition', output=tmp_path / 'x', paired_30m=True, arena_variant='fixed-2x-pool', competition=True)
    # The plain 25-game study keeps the incremental rule; its clock is tokens (inference seconds with
    # --budget-clock time).
    _, plain = builder.build('plain', output=tmp_path / 'p', paired_30m=True, arena_variant='cache-reclaim')
    assert plain['budget_clock'] == 'tokens' and 'level_budget_policy' not in plain
    _, plain_time = builder.build('plain-time', output=tmp_path / 'pt', paired_30m=True, arena_variant='cache-reclaim',
                                  budget_clock='time')
    assert plain_time['budget_clock'] == 'inference'
    assert plain['trials'] == [{'name': 'duck-transfer', 'kind': 'duck-transfer'}, {'name': 'pi', 'kind': 'pi'}] or plain['trials'][0]['name'] == 'duck-transfer'


def test_bootstrap_skips_the_sanity_trial_on_the_hidden_rerun():
    source = (ROOT / 'scripts/arena/bootstrap.py').read_text()
    assert 'skipped: hidden rerun' in source and "trial.get('competition', True)" in source
    # The skip must not turn into a non-zero exit (the 25 September rehearsal failed on exactly this).
    assert 'v != "complete" and v != "skipped: hidden rerun"' in source
    assert "trial_wall = float(trial.get('wall_seconds', wall))" in source
    # v3 preview (25 September): the trial process must never print to the notebook pipe.
    assert 'stdout=stream, stderr=subprocess.STDOUT' in source and 'process_log.open("ab")' in source


def test_bootstrap_trial_lifecycle_lines_come_from_the_process_log(tmp_path):
    import ast
    import types
    # The bundle-only ``bootstrap`` sibling import keeps the module from loading here; take the function.
    source = (ROOT / 'scripts/arena/bootstrap.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'trial_lifecycle_lines')
    module = types.SimpleNamespace()
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'bootstrap', 'exec'), namespace)
    module.trial_lifecycle_lines = namespace['trial_lifecycle_lines']
    log = tmp_path / 'duck-transfer-process.log'
    log.write_text('Could not determine version for k000, using k000\n'
                   '{"event": "starting_trial", "trial": {"name": "duck-transfer"}}\n'
                   'noise\n'
                   '{"event": "trial_result", "trial": "duck-transfer", "status": "complete", "error": null}\n')
    assert module.trial_lifecycle_lines(log) == [
        '{"event": "starting_trial", "trial": {"name": "duck-transfer"}}',
        '{"event": "trial_result", "trial": "duck-transfer", "status": "complete", "error": null}']
    assert module.trial_lifecycle_lines(tmp_path / 'missing.log') == []


def test_sanity_trial_runs_the_plain_arena_path_inside_a_competition_study(monkeypatch, tmp_path):
    import time
    from contextlib import contextmanager

    from test_arena_competition import study as competition_study

    from arena import competition
    from arena import run as runner

    cfg = competition_study()
    cfg['trials'] = [{'name': 'sanity-25g', 'kind': 'duck-transfer', 'competition': False, 'wall_seconds': 1800},
                     {'name': 'duck-transfer', 'kind': 'duck-transfer'}]
    cfg['isolate_trial_cache'] = False
    path = tmp_path / 'study.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '0')

    @contextmanager
    def lifecycle(*args):
        pytest.fail('the sanity trial must not open the competition arcade')
        yield

    seen = {}

    async def execute(study, trial, **kwargs):
        seen.update(wall=study['wall_seconds'], games=study['games'], trial=trial['name'], arcade='arcade' in kwargs)
        return {'status': 'complete', 'task_errors': [], 'mean_score_equal_weight_25_games': 1.0, 'levels': 1,
                'actions': 1, 'scheduler': {}, 'engine': {'preemptions': 0}, 'window_choice': {}, 'preemption_free': True}

    async def validate(*args):
        return {}

    monkeypatch.setattr(competition, 'open_arcade', lifecycle)
    monkeypatch.setattr(runner, 'run_trial', execute)
    monkeypatch.setattr(runner, 'validate_api', validate)
    monkeypatch.setattr(runner, 'wait_idle', validate)
    monkeypatch.setattr(runner, 'hardware', lambda: {})
    monkeypatch.setattr(runner.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(runner.signal, 'setitimer', lambda *a: None)
    runner.main(['--source', str(ROOT), '--study', str(path), '--trial', 'sanity-25g',
                 '--output', str(tmp_path / 'out'), '--server-dir', str(tmp_path),
                 '--games-dir', 'public-games', '--notebook-start-epoch', str(time.time() - 750),
                 '--notebook-seconds', '7500'])
    assert seen == dict(wall=1800.0, games=CONTROL['games'], trial='sanity-25g', arcade=False)


@pytest.mark.parametrize('name,keys', [
    ('fixed-2x-64k-four', {'windows', 'cache_experiment', 'compaction_keep_observations', 'active_limit'}),
    ('fixed-2x-effort-medium', {'reasoning_effort'}),
    ('fixed-2x-effort-low', {'reasoning_effort'}),
    ('fixed-2x-mtp4', {'server'}),
    ('fixed-2x-t1cap-120', {'t1_cap_seconds'}),
    ('fixed-2x-switch-3', {'opening_switch_level1_games'}),
    ('fixed-2x-80k-three', {'windows', 'cache_experiment', 'compaction_keep_observations', 'active_limit'}),
    ('fixed-2x-stall-open360', {'focus_stall_actions', 'opening_max_wall_seconds'}),
    ('fixed-2x-notes', {'cross_game_notes'}),
    ('fixed-2x-memory', {'memory_block'}),
    ('fixed-2x-escalate', {'model_escalation'}),
    ('fixed-2x-snapshot', {'game_snapshot'}),
    ('fixed-2x-level-probe', {'game_snapshot'}),
    ('fixed-2x-reopen-pause', {'reopen_phase1', 'level_budget_rule'}),
    ('fixed-2x-reopen-run', {'reopen_phase1', 'level_budget_rule'}),
    ('fixed-2x-slots-1', {'active_limit', 'opening_switch_level1_games'}),
    ('fixed-2x-slots-4', {'active_limit', 'opening_switch_level1_games', 'windows', 'cache_experiment'}),
    ('fixed-2x-slots-8', {'active_limit', 'opening_switch_level1_games', 'windows', 'cache_experiment'}),
    ('fixed-2x-mtp0-wide', {'server', 'active_limit', 'opening_switch_level1_games', 'windows', 'cache_experiment',
                            'require_focus_windows_fit'}),
    ('fixed-2x-budget-1x', {'level_budget_multiple'}),
    ('fixed-2x-budget-4x', {'level_budget_multiple'}),
    ('fixed-2x-llm-rank', {'backlog_policy', 'llm_backlog'}),
    ('fixed-2x-procedure', {'stuck_procedure'}),
    ('fixed-2x-thompson', {'backlog_policy', 'thompson_seed'}),
    ('fixed-2x-coverage', {'duck_harness'}),
    ('fixed-2x-explorer', {'explorer'}),
    ('fixed-2x-token-budget', {'budget_clock', 'cost_normalization', 't1_clip'}),
    ('fixed-2x-index', {'level_budget_policy', 'index_scheduler'}),
    ('fixed-2x-index-lane', {'level_budget_policy', 'index_scheduler', 'background_lane'}),
    ('fixed-2x-explorer-start', {'explorer'}),
    ('fixed-2x-explorer-start-silent', {'explorer'}),
    ('fixed-2x-critical', {'critical_errors'}),
    ('fixed-2x-play', {'duck_harness'}),
    ('fixed-2x-harness-v2', {'duck_harness', 'harness_v2'}),
])
def test_composed_variants_differ_from_the_fixed_2x_base_by_one_change(name, keys):
    base = apply_variant(CONTROL, 'reclaim-fixed-2x')
    candidate = apply_variant(CONTROL, name)
    meta = candidate.pop('arena_variant')
    base.pop('arena_variant')
    diff = {k for k in base.keys() | candidate.keys() if base.get(k) != candidate.get(k)}
    assert diff == keys
    depth = {'fixed-2x-mtp4': 4, 'fixed-2x-mtp0-wide': 0}.get(name)
    serving_same = depth is None
    assert meta['reference_variant'] == 'reclaim-fixed-2x' and meta['reference_checkpoint_and_serving_unchanged'] is serving_same
    assert candidate['budget_clock'] == ('normalized' if name == 'fixed-2x-token-budget' else 'busy')
    assert candidate['level_budget_policy'] == ('index' if name.startswith('fixed-2x-index') else 'fixed-2x')
    assert candidate['backlog_policy'] == {'fixed-2x-llm-rank': 'llm', 'fixed-2x-thompson': 'thompson'}.get(name, 'readmissions')
    assert candidate['server'] == (CONTROL['server'] if serving_same else {**CONTROL['server'], 'mtp_tokens': depth})
    if name == 'fixed-2x-64k-four':
        assert candidate['windows'] == [65536] and candidate['active_limit'] == 4 and candidate['compaction_keep_observations'] == 3
    elif name == 'fixed-2x-t1cap-120':
        assert candidate['t1_cap_seconds'] == 120
    elif name == 'fixed-2x-switch-3':
        assert candidate['opening_switch_level1_games'] == 3
    elif name == 'fixed-2x-stall-open360':
        assert candidate['focus_stall_actions'] == 15 and candidate['opening_max_wall_seconds'] == 360
    elif name == 'fixed-2x-80k-three':
        assert candidate['windows'] == [81920] and candidate['active_limit'] == 3 and candidate['compaction_keep_observations'] == 3
    elif name == 'fixed-2x-notes':
        assert candidate['cross_game_notes'] == {'lines_per_request': 10, 'request_cap_tokens': 1024, 'inject_cap_tokens': 2000}
    elif name == 'fixed-2x-memory':
        assert candidate['memory_block'] == {'every_actions': 6, 'request_cap_tokens': 2048}
    elif name == 'fixed-2x-escalate':
        assert candidate['model_escalation'] == {'after_notices': 2}
    elif name == 'fixed-2x-snapshot':
        assert candidate['game_snapshot'] == {'probe': True, 'objects': 20}
    elif name == 'fixed-2x-level-probe':
        assert candidate['game_snapshot'] == {'probe': True, 'objects': 20, 'every_level': True, 'clicks': 3}
    elif name.startswith('fixed-2x-reopen-'):
        assert candidate['reopen_phase1'] == name.rsplit('-', 1)[1]
    elif name.startswith('fixed-2x-slots-'):
        slots = int(name.rsplit('-', 1)[1])
        assert candidate['active_limit'] == candidate['opening_switch_level1_games'] == slots
        window = candidate['windows'][0]
        assert window == {1: 131072, 4: 65536, 8: 32768}[slots] and slots * window <= 0.9 * 291900
    elif name == 'fixed-2x-coverage':
        assert (ROOT / candidate['duck_harness'] / 'skills/coverage.py').is_file()
    elif name == 'fixed-2x-thompson':
        assert candidate['thompson_seed'] == 0
    elif name.startswith('fixed-2x-index'):
        assert candidate['index_scheduler']['shape'] == 'recorded' and candidate['index_scheduler']['horizon_scale'] == 0.5
        assert candidate.get('background_lane') == ({'max_inflight': 2, 'guard_caps': 2} if name.endswith('-lane') else None)
    elif name == 'fixed-2x-token-budget':
        assert candidate['t1_clip'] == [0.5, 2.0] and candidate['cost_normalization']['context_weight'] == 0.0
    elif name.startswith('fixed-2x-explorer-start'):
        assert candidate['explorer']['at_start'] and candidate['explorer']['report'] is (not name.endswith('silent'))
        assert candidate['explorer']['max_level'] == -1  # the phase-2 service never runs in these arms
    elif name == 'fixed-2x-explorer':
        assert candidate['explorer'] == {'max_level': 0, 'max_actions': 20000, 'max_seconds': 90,
                                         'min_remaining_seconds': 120}
    elif name == 'fixed-2x-play':
        assert (ROOT / candidate['duck_harness'] / 'skills/play.py').is_file()
    elif name == 'fixed-2x-critical':
        assert candidate['critical_errors'] is True
    elif name == 'fixed-2x-harness-v2':
        assert (ROOT / candidate['duck_harness'] / 'skills/check_model.py').is_file()
        assert all(candidate['harness_v2'][edit] for edit in ('effects', 'nudge', 'legend', 'new_level', 'handoff'))
    elif name == 'fixed-2x-procedure':
        assert candidate['stuck_procedure'] == {'collapse_window': 30, 'collapse_share': 0.8}
    elif name == 'fixed-2x-llm-rank':
        assert candidate['llm_backlog'] == {'interval_seconds': 180, 'cap_tokens': 768, 'thinking': False}
    elif name.startswith('fixed-2x-budget-'):
        assert candidate['level_budget_multiple'] == int(name.rsplit('-', 1)[1][:-1])
    elif name == 'fixed-2x-mtp0-wide':
        assert candidate['active_limit'] == candidate['opening_switch_level1_games'] == 6
        assert candidate['windows'] == [65536] and candidate['require_focus_windows_fit'] is True
        assert meta['reference_difference']['server.mtp_tokens'] == {'before': 3, 'after': 0}
    elif name == 'fixed-2x-mtp4':
        assert meta['reference_difference'] == {'server.mtp_tokens': {'before': 3, 'after': 4}}
        with pytest.raises(ValueError, match='MTP-3 control'):
            apply_variant({**CONTROL, 'server': {**CONTROL['server'], 'mtp_tokens': 1}}, 'fixed-2x-mtp4')
    else:
        assert candidate['reasoning_effort'] == name.rsplit('-', 1)[1]
    with pytest.raises(ValueError):
        apply_variant({**CONTROL, 'served_model': 'primitive-ai/Nex-N2.5-mini-NVFP4'}, 'fixed-2x-effort-low')


def test_both_duck_variants_charge_python_tool_time_to_the_busy_clock():
    from arena.duck import ArenaDuckMixin, ArenaPublicDuckAgent, ArenaTransferDuckAgent

    for cls in (ArenaTransferDuckAgent, ArenaPublicDuckAgent):
        assert cls.run_python_tool is ArenaDuckMixin.run_python_tool
    now = Now()
    wall = GlobalClock(900, now=now)
    clock = ActiveClock(wall, now=now, budget_clock='busy')
    wall.release()
    clock.release()

    class Base:
        def run_python_tool(self, execute, state_path, arguments):
            now.advance(7)
            return execute(state_path, arguments)

    class Agent(ArenaDuckMixin, Base):
        pass

    agent = Agent.__new__(Agent)
    agent.game = type('G', (), {'clock': clock})()
    assert agent.run_python_tool(lambda p, a: 'out', 'state', {'code': 'x'}) == 'out'
    assert clock.tool_seconds() == 7 and clock.budget_seconds() == 7
