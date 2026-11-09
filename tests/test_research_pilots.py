import hashlib
import json
from types import SimpleNamespace

import pytest

from arena import research_checkpoint
from arena.serving import configure


def test_swift_checkpoint_verifies_every_payload_and_records_actual_identity(tmp_path, monkeypatch):
    module = tmp_path / 'source' / 'research_checkpoint.py'
    models = module.parent / 'models'
    models.mkdir(parents=True)
    monkeypatch.setattr(research_checkpoint, '__file__', str(module))
    root = tmp_path / 'weights'
    root.mkdir()
    (root / 'weights.safetensors').write_bytes(b'checkpoint')
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'weights.safetensors'}}))
    inventory = {'repo': research_checkpoint.REPO, 'revision': research_checkpoint.REVISION,
                 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in root.iterdir()}}
    (models / 'swift15-flash-nvfp4.json').write_text(json.dumps(inventory))
    recipe = SimpleNamespace(SERVED_MODEL_NAME='comparison-alias')
    research_checkpoint.configure(recipe, tmp_path, {**inventory, 'path': str(root)})
    assert recipe.resolve_model_dir() == root
    result = recipe.verify_model(root)
    assert result['repo'] == research_checkpoint.REPO
    assert result['all_file_hashes_verified']
    assert not result['base_expert_overlay']
    (root / 'weights.safetensors').write_bytes(b'checkpoinX')
    with pytest.raises(ValueError, match='hash mismatch'):
        recipe.verify_model(root)


@pytest.mark.parametrize('key', ['expert_overlay', 'expert_prune'])
def test_new_checkpoint_cannot_inherit_base_weights_or_mask(tmp_path, key):
    with pytest.raises(ValueError, match='base Flash'):
        configure(SimpleNamespace(), tmp_path, research_checkpoint={'repo': 'swift'}, **{key: {}})


def test_research_checkpoint_rejects_unreviewed_revision(tmp_path):
    with pytest.raises(ValueError, match='Unreviewed'):
        research_checkpoint.configure(SimpleNamespace(), tmp_path, {'repo': research_checkpoint.REPO, 'revision': 'main'})


def test_swift_pilot_preserves_scheduler_and_weights_but_requires_capacity_evidence():
    import copy
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    apply = runpy.run_path(str(root / 'scripts/arena/build_research_models.py'))['swift_gameplay']
    replay = json.loads((root / 'docs/validation-20260930/swift-37-capacity.json').read_text())
    study = dict(hero_v2={'original': True}, windows=[93184], blocks_plan={},
                 focus_oversubscription={}, wall_seconds=32400, completion_cap=16384,
                 active_limit=37, action_plans=True, blocks={'wall_seconds': 32400, 'slots': 37},
                 stall_dump_seconds=90, trials=[{'server_env': {'ARENA_STALL_DUMP': '90',
                                                              'ARENA_READY_GUARD': '180,900'}}])
    apply(study, replay)
    assert study['windows'] == [49152]
    assert study['wall_seconds'] == study['blocks']['wall_seconds'] == 32400
    assert study['active_limit'] == study['blocks']['slots'] == 37
    assert study['completion_cap'] == 16384 and study['action_plans']
    assert study['early_stop']['at_seconds'] == 1800
    assert study['blocks_plan']['min_pool_blocks'] == 880
    assert 'stall_dump_seconds' not in study
    assert study['trials'][0]['server_env'] == {'ARENA_READY_GUARD': '180,900'}
    assert int(int(880 * .96) // study['focus_oversubscription']['gate_blocks_per_stream']) >= 37
    bad = copy.deepcopy(replay)
    next(r for r in bad['rows'] if r['window'] == 49152)['passes'] = False
    with pytest.raises(ValueError, match='replay scenarios'):
        apply({}, bad)
