"""Historical and current arena trace formats must produce the same accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'hero_comparison_loader', Path(__file__).resolve().parents[1] / 'scripts/arena_analysis/abc_load.py')
loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loader)


def test_epoch_uses_gameplay_admission_not_prebarrier_wait(tmp_path):
    rows = [dict(epoch=90, wall_seconds=None),
            dict(epoch=95, wall_seconds=0, event='request_wait_started'),
            dict(epoch=100, wall_seconds=0, event='admit', reason='block_start')]
    (tmp_path / 'admission.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    run = loader.Run.__new__(loader.Run)
    run.dir = tmp_path
    assert run._epoch0() == 100


@pytest.mark.parametrize('paired', [False, True])
def test_compaction_usage_counted_once_and_clear_keeps_evidence(tmp_path, paired):
    usage = dict(completion_tokens=23, prompt_tokens=100)
    records = [dict(kind='model', purpose='gameplay', usage=dict(completion_tokens=10))]
    if paired:
        records.append(dict(kind='model', purpose='compaction', usage=usage))
    records += [dict(kind='compaction', usage=usage),
                dict(kind='observation', action='ACTION7', levels=1, time=110)]
    path = tmp_path / 'games/g/log.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join(map(json.dumps, records)))
    run = loader.Run.__new__(loader.Run)
    run.dir, run.epoch0 = tmp_path, 100
    game = loader.Game('g', 'source')
    run.games = {'g': game}
    run._load_logs()
    assert len(game.replies) == 2
    assert game.clears == [dict(level=1, wall=10, tokens=33, actions=1)]
    assert len(game.log_events['compaction']) == 1


def test_unstarted_game_may_lack_log_but_started_game_must_have_it(tmp_path):
    run = loader.Run.__new__(loader.Run)
    run.dir, run.epoch0 = tmp_path, 100
    game = loader.Game('g', 'source')
    game.final_actions = game.final_levels = 0
    run.games = {'g': game}
    run._load_logs()
    game.events = [dict(event='admit')]
    with pytest.raises(FileNotFoundError):
        run._load_logs()
