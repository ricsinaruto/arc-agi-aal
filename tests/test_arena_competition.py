import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arena import competition
from arena.trial import duck_config

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / 'configs/arena_30m.json').read_text())
SETTINGS = {'preview_instances': 110, 'preview_seconds': 3600,
            'submission_seconds': 32400, 'cleanup_seconds': 600,
            'max_actions_per_game': 100000}


def study():
    return {**CONTROL, 'notebook_seconds': 5400, 'competition': dict(SETTINGS)}


def test_whole_notebook_deadline_includes_setup_and_cleanup(monkeypatch):
    monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    assert competition.runtime_seconds(study()) == 5400
    assert competition.gameplay_seconds(study(), 5400, now=750) == 3600
    with pytest.raises(TimeoutError, match='complete public rehearsal'):
        competition.gameplay_seconds(study(), 5400, now=1300)
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', 'true')
    assert competition.runtime_seconds(study()) == 32400
    assert competition.gameplay_seconds(study(), 32400, now=750) == 31050
    with pytest.raises(TimeoutError, match='No competition gameplay'):
        competition.gameplay_seconds(study(), 32400, now=31900)


def test_hidden_roster_comes_only_from_gateway(monkeypatch, tmp_path):
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '1')
    arc = NS(get_environments=lambda: [NS(game_id='hidden-b'), NS(game_id='hidden-a')])
    calls = []
    monkeypatch.setattr(competition, 'gateway_arcade', lambda *a: calls.append(a) or arc)
    from taaf.competition_arcade import CompetitionArcadeServer
    monkeypatch.setattr(CompetitionArcadeServer, 'start', lambda *_: pytest.fail('must not load public games'))
    with competition.open_arcade(study(), None, tmp_path) as (resolved, owner):
        assert resolved['games'] == ['hidden-b', 'hidden-a']
        assert resolved['competition']['execution_mode'] == 'hidden_gateway'
        assert owner.arcade is arc
    assert calls[0][0] == 'http://gateway:8001'
    assert study()['games'] == CONTROL['games']


def test_scorecard_closes_on_initialization_failure_without_replay(tmp_path):
    calls = []
    card = NS(model_dump=lambda **kw: {'card_id': 'one'})
    arc = NS(open_scorecard=lambda **kw: 'one',
             close_scorecard=lambda ident: calls.append(ident) or card)
    owner = competition.ScorecardOwner(arc, tmp_path)
    assert owner.open_scorecard(tags=['test']) == 'one'
    with pytest.raises(RuntimeError, match='one scorecard'):
        owner.open_scorecard()
    owner.finalize()
    owner.finalize()
    assert calls == ['one']
    assert json.loads((tmp_path / 'gateway-scorecard.json').read_text())['card_id'] == 'one'


def test_uncertain_scorecard_close_is_never_retried(tmp_path):
    calls = []

    def close(card):
        calls.append(card)
        raise TimeoutError('response lost')

    owner = competition.ScorecardOwner(NS(open_scorecard=lambda: 'one', close_scorecard=close), tmp_path)
    owner.open_scorecard()
    with pytest.raises(TimeoutError):
        owner.finalize()
    owner.finalize()
    assert calls == ['one']


def test_competition_action_ceiling_does_not_change_benchmark_control():
    args = dict(window=131072, cap=16384, wall=3600, base_url='http://localhost/v1')
    assert duck_config(CONTROL, ROOT, **args).max_actions == 1200
    assert duck_config(study(), ROOT, **args).max_actions == 100000


def test_competition_notebook_preserves_agent_and_selects_gateway_branch(tmp_path):
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    out, config = builder.build('submission-test', output=tmp_path / 'bundle', paired_30m=True,
                                arena_variant='cache-reclaim', competition=True)
    assert config['competition']['preview_instances'] == 110
    assert config['wall_seconds'] == 3600 and config['kernel_timeout'] == 32400
    assert config['games'] == CONTROL['games'] and config['active_limit'] == 2
    assert config['server'] == CONTROL['server'] and config['windows'] == CONTROL['windows']
    notebook = builder.notebook(config, 'runtime.blob', 'a' * 64, smoke=False)
    import ast
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    assert '--notebook-start-epoch' in notebook['cells'][1]['source']
    assert 'if not rerun:' in notebook['cells'][2]['source']
    assert 'EXPECTED_PREVIEW_TRIALS = 2' in notebook['cells'][2]['source']
    assert [t['name'] for t in config['trials']] == ['sanity-25g', 'duck-transfer']
    assert config['trials'][0] == {'name': 'sanity-25g', 'kind': 'duck-transfer', 'competition': False, 'wall_seconds': 1800}
    assert config['notebook_seconds'] == 7500
    assert (out / 'src/arena/competition.py').is_file()
    with pytest.raises(ValueError, match='fixed Flash'):
        builder.build('bad', output=tmp_path / 'bad', paired_30m=True, competition=True, capacity_phase2=True)
    assert not (tmp_path / 'bad').exists()


def test_worldmodel_competition_with_a_23_game_30_minute_preview(tmp_path):
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, config = builder.build('wm-test', output=tmp_path / 'bundle', paired_30m=True, arena_variant='fixed-2x-worldmodel',
                              competition=True, exclude_games=['ft09', 'lp85'], preview_seconds=1800,
                              preview_instances=23, sanity_trial=False)
    assert len(config['games']) == 23 and not {g[:4] for g in config['games']} & {'ft09', 'lp85'}
    assert config['competition']['preview_instances'] == 23 and config['competition']['preview_seconds'] == 1800
    assert config['competition']['submission_seconds'] == 32400 and config['kernel_timeout'] == 32400
    assert [t['name'] for t in config['trials']] == ['duck-transfer']
    assert config['wall_seconds'] == 1800 and config['notebook_seconds'] == 4500
    assert config['duck_harness'] == 'src/arena/harnesses/worldmodel' and config['worldmodel_notice'] is True
    assert (tmp_path / 'bundle/src/arena/harnesses/worldmodel/skills/worldmodel.py').is_file()
    notebook = builder.notebook(config, 'runtime.blob', 'a' * 64, smoke=False)
    assert '23 independent cyclic replicas of the 23 public games' in notebook['cells'][0]['source']
    assert 'EXPECTED_PREVIEW_TRIALS = 1' in notebook['cells'][2]['source']
    with pytest.raises(ValueError, match='1-110 instances'):
        builder.build('bad', output=tmp_path / 'bad', paired_30m=True, arena_variant='fixed-2x-worldmodel',
                      competition=True, preview_instances=111)


def test_research_clone_roster_of_110_instances_from_20_games(tmp_path):
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, config = builder.build('clones-test', output=tmp_path / 'bundle', paired_30m=True, arena_variant='reclaim-fixed-2x',
                              exclude_games=['ft09', 'lp85', 'ar25', 'vc33', 'tu93'], wall_seconds=3600,
                              clone_instances=110)
    assert len(config['games']) == 20 and len(config['games_excluded']) == 5
    clones = config['competition']
    assert clones['research_clones'] and clones['preview_instances'] == 110 and clones['preview_seconds'] == 3600
    assert config['trials'] == [{'name': 'duck-transfer', 'kind': 'duck-transfer'}] and config['wall_seconds'] == 3600
    with pytest.raises(ValueError, match='clone-instances'):
        builder.build('bad', output=tmp_path / 'bad', paired_30m=True, arena_variant='reclaim-fixed-2x',
                      competition=True, clone_instances=110)


@pytest.mark.skipif(not (ROOT / 'environment_files').is_dir(), reason='public environments unavailable')
def test_real_sdk_110_independent_cyclic_games_one_scorecard(monkeypatch, tmp_path):
    from arcengine import GameAction

    monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    with competition.open_arcade(study(), ROOT / 'environment_files', tmp_path) as (resolved, owner):
        assert resolved['games'] == [f'k{i:03d}' for i in range(110)]
        roster = json.loads((tmp_path / 'public-roster.json').read_text())
        assert [r['source_game_id'] for r in roster] == CONTROL['games'] * 4 + CONTROL['games'][:10]
        # The gateway strips privileged metadata; local server cloning is tested
        # separately by matching source observations across corresponding replicas.
        card = owner.open_scorecard(tags=['cpu-lifecycle-test'])
        envs = [owner.make(g, scorecard_id=card, save_recording=False) for g in resolved['games']]
        assert all(env is not None for env in envs)
        import numpy as np
        original = np.array(envs[25].observation_space.frame[-1], copy=True)
        assert np.array_equal(original, envs[0].observation_space.frame[-1])
        action = envs[0].observation_space.available_actions[0]
        data = {'x': 32, 'y': 32} if action == 6 else {}
        result = envs[0].step(GameAction.from_id(action), data=data)
        assert result is not None
        assert np.array_equal(original, envs[25].observation_space.frame[-1])
        closed = owner.close_scorecard(card)
        assert len(closed.environments) == 110
        assert {g.id for g in closed.environments} == set(resolved['games'])

    # The stall-open360 opening cap is 360 s per 25 games: the 110-clone preview gets 1,584 s.
    capped = {**study(), 'opening_max_wall_seconds': 360}
    with competition.open_arcade(capped, ROOT / 'environment_files', tmp_path / 'capped') as (resolved, owner):
        assert resolved['opening_max_wall_seconds'] == 360 * 110 / 25
        assert resolved['competition']['opening_cap_scaled'] == {'per_25_games_seconds': 360, 'roster': 110, 'seconds': 1584.0}
    with competition.open_arcade(study(), ROOT / 'environment_files', tmp_path / 'plain') as (resolved, owner):
        assert 'opening_max_wall_seconds' not in resolved and 'opening_cap_scaled' not in resolved['competition']


async def test_injected_arcade_uses_dynamic_roster_and_clips_at_barrier(monkeypatch, tmp_path):
    import time

    from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

    from arena import trial as module

    games = ['hidden-a', 'hidden-b']
    config = {**STUDY, 'games': games, 'wall_seconds': 30}
    monkeypatch.setattr(module, 'kv_pool_tokens', lambda *_: 291900)
    monkeypatch.setattr(module, 'GpuMonitor', OfflineMonitor)
    monkeypatch.setattr(module, 'make_arcade', lambda *_: pytest.fail('must use injected gateway'))

    async def workers(self, records, *args):
        instances = [FakeWorker(g) for g in records]
        async def close():
            await asyncio.gather(*(w.close() for w in instances))
        self.config_dict = {'reasoning_effort': 'xhigh', 'temperature': .7}
        return instances, lambda: [w.result() for w in instances], close

    monkeypatch.setattr(module.Trial, 'duck_agents', workers)
    trial = module.Trial(config, {'name':'duck-transfer','kind':'duck-transfer'},
                         source=ROOT, games_dir=None, output=tmp_path, server_dir=tmp_path,
                         base_url='http://unused', arcade=FakeArcade(games),
                         gameplay_deadline_epoch=time.time()+.3)
    report = await trial.run()
    assert report['score_denominator_games'] == 2
    assert report['wall_seconds'] < 1
    assert [g['game_id'] for g in report['games']] == games


@pytest.mark.parametrize('rerun,status,allowed,returncode', [
    (False, 'complete', True, 0), (False, 'complete_with_preemptions', False, 0), (False, 'complete', False, 1),
    (True, 'complete_with_preemptions', True, 0), (True, 'error', True, 0), (True, 'complete', True, 1),
])
def test_submission_artifact_is_placeholder_only_in_successful_public_preview(monkeypatch, tmp_path, rerun, status, allowed, returncode):
    import sys
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    preview = {**study(), 'trials': [{'name': 'sanity-25g', 'kind': 'duck-transfer', 'competition': False, 'wall_seconds': 1800},
                                     {'name': 'duck-transfer', 'kind': 'duck-transfer'}]}
    source = builder.notebook(preview, 'unused', 'a'*64, smoke=False)['cells'][2]['source']
    source = source.replace('/kaggle/working', str(tmp_path))
    path = tmp_path / 'arena/trials/study.json'
    path.parent.mkdir(parents=True)
    # Preview runs the 25-game sanity trial and then the 110-instance hour; the rerun only the gateway.
    trials = [{'status': status}] if rerun else [{'status': 'complete'}, {'status': status}]
    path.write_text(json.dumps({'trials': trials, 'comparison': []}))
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', 'true' if rerun else 'false')
    writes = []
    monkeypatch.setitem(sys.modules, 'pandas', NS(DataFrame=lambda *a, **kw: NS(
        to_parquet=lambda *args, **kwargs: writes.append((args,kwargs)))))
    namespace = {'result':NS(returncode=returncode)}  # a bootstrap exit code never fails the hidden rerun after gameplay
    if allowed:
        exec(compile(source,'submission-cell','exec'),namespace)
    else:
        with pytest.raises(RuntimeError, match='validation failed'):
            exec(compile(source,'submission-cell','exec'),namespace)
    assert len(writes) == int(allowed and not rerun)


async def test_initialization_failure_through_trial_still_finalizes_one_scorecard(monkeypatch, tmp_path):
    from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

    from arena import trial as module

    games = ['hidden-a', 'hidden-b']
    config = {**STUDY, 'games': games, 'wall_seconds': 30}
    arc = FakeArcade(games)
    owner = competition.ScorecardOwner(arc, tmp_path)
    monkeypatch.setattr(module, 'kv_pool_tokens', lambda *_: 291900)
    monkeypatch.setattr(module, 'GpuMonitor', OfflineMonitor)
    monkeypatch.setattr(module, 'READY_TIMEOUT', .03)
    starts, stops = [], []

    class Worker(FakeWorker):
        async def run(self):
            starts.append(self.game.game_id)
            try:
                if self.game.index == 0:
                    raise RuntimeError('Environment initialization failed before ready')
                self.game.ready.set()
                await asyncio.Event().wait()
            finally:
                stops.append(self.game.game_id)

    async def workers(self, records, *args):
        instances = [Worker(g) for g in records]
        async def close():
            await asyncio.gather(*(w.close() for w in instances))
        self.config_dict = {}
        return instances, lambda: [w.result() for w in instances], close

    monkeypatch.setattr(module.Trial, 'duck_agents', workers)
    trial = module.Trial(config, {'name':'duck-transfer','kind':'duck-transfer'},
                         source=ROOT, games_dir=None, output=tmp_path, server_dir=tmp_path,
                         base_url='http://unused', arcade=owner)
    try:
        with pytest.raises(RuntimeError, match='Games not ready'):
            await trial.run()
    finally:
        owner.finalize()
    assert starts == games and set(stops) == set(games)
    assert owner.close_attempted and arc.closed is not None
    assert (tmp_path / 'gateway-scorecard.json').is_file()


@pytest.mark.parametrize('rerun,status,success', [
    (True, 'complete_with_preemptions', True),
    (False, 'complete_with_preemptions', False),
    (True, 'error', True),
])
def test_competition_cli_routes_roster_deadline_and_failure_status(monkeypatch, tmp_path, rerun, status, success):
    import time
    from contextlib import contextmanager

    from arena import run as runner

    cfg = study()
    cfg['trials'] = [{'name':'duck-transfer','kind':'duck-transfer'}]
    cfg['isolate_trial_cache'] = False
    path = tmp_path / 'study.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '1' if rerun else '0')
    calls = []
    arcade = object()

    @contextmanager
    def lifecycle(study, games_dir, output):
        calls.append(('open', games_dir))
        try:
            yield {**study,'games':['dynamic-id']}, arcade
        finally:
            calls.append(('closed',None))

    async def execute(study, trial, **kwargs):
        assert study['games'] == ['dynamic-id'] and kwargs['arcade'] is arcade
        assert ('gameplay_deadline_epoch' in kwargs) is rerun
        if rerun:
            assert 31000 < study['wall_seconds'] < 31200
        else:
            assert study['wall_seconds'] == 3600
        return {'status':status,'task_errors':['worker failed'] if status=='error' else [],
                'mean_score_equal_weight_25_games':1.0,'levels':1,'actions':1,
                'scheduler':{},'engine':{'preemptions':2},'window_choice':{}}

    async def validate(*args):
        return {}

    monkeypatch.setattr(competition,'open_arcade',lifecycle)
    monkeypatch.setattr(runner,'run_trial',execute)
    monkeypatch.setattr(runner,'validate_api',validate)
    monkeypatch.setattr(runner,'wait_idle',validate)
    monkeypatch.setattr(runner,'hardware',lambda: {})
    monkeypatch.setattr(runner,'find_games',lambda *_: pytest.fail('explicit path or hidden gateway expected'))
    monkeypatch.setattr(runner.signal,'signal',lambda *a: None)
    monkeypatch.setattr(runner.signal,'setitimer',lambda *a: None)
    argv=['--source',str(ROOT),'--study',str(path),'--trial','duck-transfer',
          '--output',str(tmp_path/'out'),'--server-dir',str(tmp_path),
          '--games-dir','public-games','--notebook-start-epoch',str(time.time()-750),
          '--notebook-seconds',str(32400 if rerun else 5400)]
    if success:
        runner.main(argv)
    else:
        with pytest.raises(SystemExit) as exc:
            runner.main(argv)
        assert exc.value.code == 1
    assert calls == [('open',None if rerun else 'public-games'),('closed',None)]
    record=json.loads((tmp_path/'out/study.json').read_text())
    assert record['study']['games']==['dynamic-id']
    assert record['trials'][0]['status']==status


def test_rerun_rehearsal_build_forces_the_hidden_branch_on_the_public_gateway(monkeypatch, tmp_path):
    """A rehearsal kernel follows the rerun branch end to end without spending a submission."""
    import ast
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    with pytest.raises(ValueError, match='competition build'):
        builder.build('bad', output=tmp_path / 'bad', paired_30m=True, arena_variant='reclaim-fixed-2x',
                      rerun_rehearsal=True)
    _, config = builder.build('rehearsal-test', output=tmp_path / 'bundle', paired_30m=True,
                              arena_variant='reclaim-fixed-2x', competition=True, rerun_rehearsal=True,
                              prioritize_parked_progress=False)
    assert config['competition']['rehearsal'] is True
    assert config['competition']['submission_seconds'] == 6000 and config['kernel_timeout'] == 6600
    assert [t['name'] for t in config['trials']] == ['sanity-25g', 'duck-transfer']
    notebook = builder.notebook(config, 'runtime.blob', 'a' * 64, smoke=False)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    assert "os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'" in notebook['cells'][1]['source']
    assert 'rehearsal' in notebook['cells'][0]['source']
    # The submission build never sets the flag.
    _, real = builder.build('real-test', output=tmp_path / 'real', paired_30m=True,
                            arena_variant='reclaim-fixed-2x', competition=True)
    assert 'rehearsal' not in real['competition'] and real['competition']['submission_seconds'] == 32400
    assert 'KAGGLE_IS_COMPETITION_RERUN' not in builder.notebook(real, 'b', 'a' * 64, smoke=False)['cells'][1]['source']

    # Under the flag, the roster comes from the public cyclic server, everything else is the rerun path.
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '1')
    rehearsal = {**study(), 'competition': {**SETTINGS, 'rehearsal': True, 'submission_seconds': 6000}}
    assert competition.is_rehearsal(rehearsal) and not competition.is_rehearsal(study())
    assert competition.runtime_seconds(rehearsal) == 6000
    from taaf.competition_arcade import CompetitionArcadeServer
    started = []

    def start(self):
        started.append((self.game_ids, self.total_runs, self.environments_dir))
        return self
    monkeypatch.setattr(CompetitionArcadeServer, 'start', start)
    monkeypatch.setattr(CompetitionArcadeServer, 'base_url', property(lambda self: 'http://public-clones'))
    monkeypatch.setattr(CompetitionArcadeServer, 'stop', lambda self: started.append('stopped'))
    roster = [NS(game_id=f'clone-{i}') for i in range(110)]
    urls = []
    monkeypatch.setattr(competition, 'gateway_arcade', lambda url, rec: urls.append(url) or NS(get_environments=lambda: roster))
    with competition.open_arcade(rehearsal, 'public-games', tmp_path / 'out') as (resolved, owner):
        assert resolved['competition']['execution_mode'] == 'rehearsed_hidden_gateway'
        assert len(resolved['games']) == 110
    assert urls == ['http://public-clones'] and started[0][1] == 110 and started[0][2] == 'public-games'
    assert started[-1] == 'stopped' and (tmp_path / 'out/public-roster.json').is_file()
    # A short public roster is rejected as in the preview; the hidden rerun keeps no size check.
    roster[:] = roster[:3]
    with pytest.raises(RuntimeError, match='roster size'):
        with competition.open_arcade(rehearsal, 'public-games', tmp_path / 'out2'):
            pass


@pytest.mark.parametrize('rehearsal', [False, True])
def test_rehearsal_keeps_the_games_directory_on_the_rerun_branch(monkeypatch, tmp_path, rehearsal):
    import time
    from contextlib import contextmanager

    from arena import run as runner

    cfg = study()
    cfg['trials'] = [{'name': 'sanity-25g', 'kind': 'duck-transfer', 'competition': False, 'wall_seconds': 1800},
                     {'name': 'duck-transfer', 'kind': 'duck-transfer'}]
    cfg['isolate_trial_cache'] = False
    if rehearsal:
        cfg['competition'].update(rehearsal=True, submission_seconds=6000)
    path = tmp_path / 'study.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '1')
    seen = {}

    @contextmanager
    def lifecycle(study, games_dir, output):
        seen['games_dir'] = games_dir
        yield {**study, 'games': ['dynamic-id']}, object()

    async def execute(study, trial, **kwargs):
        seen['wall'] = study['wall_seconds']
        seen['deadline'] = 'gameplay_deadline_epoch' in kwargs
        return {'status': 'complete', 'task_errors': [], 'mean_score_equal_weight_25_games': 1.0, 'levels': 1,
                'actions': 1, 'scheduler': {}, 'engine': {'preemptions': 0}, 'window_choice': {}}

    async def validate(*args):
        return {}
    monkeypatch.setattr(competition, 'open_arcade', lifecycle)
    monkeypatch.setattr(runner, 'run_trial', execute)
    monkeypatch.setattr(runner, 'validate_api', validate)
    monkeypatch.setattr(runner, 'wait_idle', validate)
    monkeypatch.setattr(runner, 'hardware', lambda: {})
    monkeypatch.setattr(runner, 'find_games', lambda *_: pytest.fail('explicit path expected'))
    monkeypatch.setattr(runner.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(runner.signal, 'setitimer', lambda *a: None)
    seconds = 6000 if rehearsal else 32400
    runner.main(['--source', str(ROOT), '--study', str(path), '--trial', 'duck-transfer',
                 '--output', str(tmp_path / 'out'), '--server-dir', str(tmp_path),
                 '--games-dir', 'public-games', '--notebook-start-epoch', str(time.time() - 750),
                 '--notebook-seconds', str(seconds)])
    assert seen['games_dir'] == ('public-games' if rehearsal else None)
    assert seen['deadline'] is True
    assert seconds - 750 - 600 - 5 < seen['wall'] <= seconds - 750 - 600

    # The bootstrap skips the sanity trial on both and only resolves public games for the rehearsal.
    source = (ROOT / 'scripts/arena/bootstrap.py').read_text()
    assert "hidden = bool(competition) and rerun and not competition.get('rehearsal')" in source


def test_no_preview_build_skips_the_public_run_and_keeps_the_rerun_branch(monkeypatch, tmp_path):
    """out of Kaggle GPU quota, so the saved version plays nothing; the hidden rerun is unchanged."""
    import ast
    import subprocess
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    with pytest.raises(ValueError, match='competition build option'):
        builder.build('bad', output=tmp_path / 'bad', paired_30m=True, arena_variant='reclaim-fixed-2x',
                      no_preview=True)
    _, config = builder.build('no-preview-test', output=tmp_path / 'bundle', paired_30m=True,
                              arena_variant='reclaim-fixed-2x', competition=True, no_preview=True)
    assert config['competition']['no_preview'] is True and config['competition']['submission_seconds'] == 32400
    cells = builder.notebook(config, 'runtime.blob', 'a' * 64, smoke=False)['cells']
    for cell in cells:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    assert 'if RERUN:\n    result = subprocess.run(command, check=False)' in cells[1]['source']
    # The ordinary submission build keeps the preview.
    _, real = builder.build('real-test', output=tmp_path / 'real', paired_30m=True,
                            arena_variant='reclaim-fixed-2x', competition=True)
    assert 'RERUN' not in builder.notebook(real, 'b', 'a' * 64, smoke=False)['cells'][1]['source']

    working = tmp_path / 'working'
    (working / 'arena/trials').mkdir(parents=True)
    final = cells[2]['source'].replace('/kaggle/working', str(working))
    result = subprocess.CompletedProcess([], 0)
    import sys
    import types

    class Frame:
        def __init__(self, rows, columns):
            self.rows = rows

        def to_parquet(self, path, index):
            Path(path).write_text(json.dumps(self.rows))
    monkeypatch.setitem(sys.modules, 'pandas', types.SimpleNamespace(DataFrame=Frame))
    # Saved version: no report exists, only the placeholder is written.
    monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    exec(final, {'result': result})
    assert (working / 'submission.parquet').is_file()
    (working / 'submission.parquet').unlink()
    # Hidden rerun: the report checks still run (a missing report fails) and no placeholder is written.
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '1')
    with pytest.raises(RuntimeError, match='before final report'):
        exec(final, {'result': result})
    (working / 'arena/trials/study.json').write_text(json.dumps({'trials': [{'status': 'complete'}]}))
    exec(final, {'result': result})
    assert not (working / 'submission.parquet').exists()


def test_gpu_memory_utilization_build_option_sets_every_trial_server_env(tmp_path):
    """the dfr-combined bundle's vision encoder ran out of GPU memory at 0.96; the option lowers the
    profiled share on every trial's server and records it in the study's server block."""
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from arena.campaign import components

    hero = components("comp-prune25")
    hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=38, density_blocks=1600,
                graph_capture=48, backlog_fix=False, budget_policy="priority", dfr_harness=True, drain_half=True)
    kwargs = dict(paired_30m=True, arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero,
                  hero_v2_served_blocks=1647, exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"],
                  opening_burst_requests=3, competition=True, sanity_trial=False)
    _, study = builder.build('gpu-util-test', output=tmp_path / 'bundle', gpu_memory_utilization='0.92', **kwargs)
    assert study['server']['gpu_memory_utilization'] == '0.92'
    assert study['trials'] and all(t['server_env']['ARENA_GPU_MEMORY_UTILIZATION'] == '0.92' for t in study['trials'])
    _, plain = builder.build('gpu-util-off', output=tmp_path / 'plain', **kwargs)
    assert plain['server']['gpu_memory_utilization'] == '0.96'
    assert not any('ARENA_GPU_MEMORY_UTILIZATION' in (t.get('server_env') or {}) for t in plain['trials'])
    for bad in ('0.96', '0.99', 0.92, '1'):
        with pytest.raises(ValueError, match='gpu_memory_utilization'):
            builder.build('gpu-util-bad', output=tmp_path / 'bad', gpu_memory_utilization=bad, **kwargs)


@pytest.mark.parametrize('flag, rerun, gateway', [(False, False, False), (True, False, True), (True, True, False)])
def test_preview_live_score_peeks_the_local_gateway_only_outside_a_rerun(monkeypatch, tmp_path, flag, rerun,
                                                                         gateway):
    from arc_agi import scorecard as sc
    from taaf import competition_arcade

    from arena.live_progress import read_score

    class Server:
        base_url = 'http://local'

        def __init__(self, **kwargs):
            envs = [NS(id='clone-0', score=30.0, levels_completed=2), NS(id='clone-1', score=0.0, levels_completed=0)]
            self._arcade = NS(scorecard_manager=NS(scorecards={'c1': NS(environments=envs)}),
                              available_environments=[])

        def start(self):
            return self

        def stop(self):
            pass

    monkeypatch.setattr(competition_arcade, 'CompetitionArcadeServer', Server)
    monkeypatch.setattr(sc.EnvironmentScorecard, 'from_scorecard', classmethod(
        lambda cls, card, envs: NS(environments=card.environments, total_levels_completed=2)))
    roster = [NS(game_id=f'clone-{i}') for i in range(110)]
    monkeypatch.setattr(competition, 'gateway_arcade', lambda *a: NS(get_environments=lambda: roster,
                                                                     open_scorecard=lambda **kw: 'c1'))
    if rerun:  # the rerun rehearsal: a local server, but the rerun flag is set
        monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '1')
    else:
        monkeypatch.delenv('KAGGLE_IS_COMPETITION_RERUN', raising=False)
    settings = {**SETTINGS, **({'preview_live_score': True} if flag else {}),
                **({'rehearsal': True} if rerun else {})}
    with competition.open_arcade({**study(), 'competition': settings}, 'games', tmp_path) as (resolved, owner):
        card = owner.open_scorecard()
        assert (owner.local_gateway is not None) is gateway
        score = read_score(owner, card, 110)
        if gateway:
            assert score['score'] == pytest.approx(30 / 110) and score['levels'] == 2 and score['scored_games'] == 1
        else:
            assert score is None
        owner.card = None  # nothing to close on the fake arcade
