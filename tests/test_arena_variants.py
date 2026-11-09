"""Paired arena interventions preserve controls and have measurable scheduling effects."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_arena_core import Now, make_games

from arena.scheduler import Scheduler
from arena.trial import duck_config
from arena.variants import NGRAM8_SPECULATION, NGRAM_SPECULATION, VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / 'configs/arena_30m.json').read_text())


@pytest.mark.parametrize('name', VARIANTS)
def test_interventions_preserve_checkpoint_serving_sampling_and_both_agents(name):
    original = json.loads(json.dumps(CONTROL))
    result = apply_variant(CONTROL, name)
    assert CONTROL == original
    for key in ('games','wall_seconds','trials','temperature','reasoning_effort',
                'generation_slots','completion_cap','backlog_priority','opening_switch_level1_games'):
        if key == 'reasoning_effort' and name in ('reclaim-effort-medium', 'fixed-2x-effort-medium', 'fixed-2x-effort-low'):
            assert result[key] == name.rsplit('-', 1)[1] and CONTROL[key] == 'xhigh'
            continue
        if key == 'opening_switch_level1_games' and name == 'fixed-2x-switch-3':
            assert result[key] == 3 and CONTROL[key] == 2
            continue
        if key == 'opening_switch_level1_games' and name.startswith('fixed-2x-slots-'):
            assert result[key] == result['active_limit'] == int(name.rsplit('-', 1)[1]) and CONTROL[key] == 2
            continue
        if key == 'opening_switch_level1_games' and name in ('fixed-2x-mtp0-wide', 'fixed-2x-mtp0-wide-kernels',
                                                             'fixed-2x-mtp3-wide-kernels', 'fixed-2x-depthwise', 'fixed-2x-depthwise-mtp3',
                                                             'fixed-2x-mtp3-wide-tweaks', 'fixed-2x-depthwise-mtp3-tweaks',
                                                             'fixed-2x-mtp3-wide-tweaks-nople', 'fixed-2x-depthwise-mtp3-tweaks-nople',
                                                             'fixed-2x-mtp3-wide-stack', 'fixed-2x-depthwise-mtp3-stack',
                                                             'fixed-2x-depthwise-mtp3-fixed-order',
                                                             'fixed-2x-depthwise-mtp3-w4a16', 'fixed-2x-depthwise-mtp3-gptq-arc', 'fixed-2x-depthwise-mtp3-gptq-arc-w4a16'):
            assert result[key] == result['active_limit'] == 6 and CONTROL[key] == 2
            continue
        if key == 'opening_switch_level1_games' and name in ('fixed-2x-mtp0-wide-kernels-kv8',
                                                             'fixed-2x-mtp0-wide-kv8-stack',
                                                             'fixed-2x-mtp0-wide-kv8-hero',
                                                             'fixed-2x-mtp0-wide-kv8-hero-uva',
                                                             'fixed-2x-mtp0-wide-kv8-hero-prefix',
                                                             'fixed-2x-mtp0-wide-kv8-hero-p6',
                                                             'fixed-2x-mtp0-wide-kv8-hero-gptq-arc'):
            assert result[key] == 6 and result['active_limit'] == 14 and CONTROL[key] == 2
            continue
        if key == 'opening_switch_level1_games' and name == 'fixed-2x-mtp0-wide-kv8-hero-6x64k':
            assert result[key] == 6 and result['active_limit'] == 6 and result['windows'] == [65536]
            continue
        if key == 'opening_switch_level1_games' and name in ('fixed-2x-mtp0-wide-kv8-hero-prune12',
                                                             'fixed-2x-mtp0-wide-kv8-hero-prune25', 'hero-v2-rank',
                                                             'fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc',
                                                             'fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc'):
            assert result[key] == 6 and result['active_limit'] == (21 if 'prune12' in name else 25)
            continue
        if key == 'opening_switch_level1_games' and name == 'fixed-2x-mtp0-wide-kv8-hero-21x42k':
            assert result[key] == 6 and result['active_limit'] == 21 and result['windows'] == [42496]
            continue
        if key == 'opening_switch_level1_games' and name == 'fixed-2x-mtp3-wide-kernels-kv8':
            from arena.variants import KV8_FOCUS_SLOTS
            assert result[key] == result['active_limit'] == KV8_FOCUS_SLOTS and CONTROL[key] == 2
            continue
        assert result[key] == CONTROL[key]
    if name not in ('graph-128', 'reclaim-mtp1', 'reclaim-mtp4', 'fixed-2x-mtp4', 'reclaim-graph-128', 'reclaim-ngram', 'reclaim-ngram8',
                    'fixed-2x-commit', 'fixed-2x-commit-queue', 'fixed-2x-mtp0-wide',
                    'fixed-2x-gittins-c110-wide', 'fixed-2x-mtp0-wide-kernels', 'fixed-2x-deep-explore',
                    'fixed-2x-mtp3-wide-kernels', 'fixed-2x-depthwise', 'fixed-2x-depthwise-mtp3',
                    'fixed-2x-mtp3-wide-tweaks', 'fixed-2x-depthwise-mtp3-tweaks',
                                                             'fixed-2x-mtp3-wide-tweaks-nople', 'fixed-2x-depthwise-mtp3-tweaks-nople',
                    'fixed-2x-mtp3-wide-stack', 'fixed-2x-depthwise-mtp3-stack', 'fixed-2x-mtp0-wide-kv8-stack',
                    'fixed-2x-depthwise-mtp3-fixed-order',
                    'fixed-2x-mtp3-wide-kernels-kv8', 'fixed-2x-mtp0-wide-kernels-kv8',
                    'fixed-2x-depthwise-mtp3-w4a16', 'fixed-2x-mtp0-wide-kv8-hero',
                    'fixed-2x-mtp0-wide-kv8-hero-21x42k', 'fixed-2x-mtp0-wide-kv8-hero-6x64k',
                    'fixed-2x-mtp0-wide-kv8-hero-prune12', 'fixed-2x-mtp0-wide-kv8-hero-prune25',
                    'fixed-2x-mtp0-wide-kv8-hero-uva', 'fixed-2x-depthwise-mtp3-gptq-arc', 'fixed-2x-depthwise-mtp3-gptq-arc-w4a16',
                    'fixed-2x-mtp0-wide-kv8-hero-prefix', 'fixed-2x-mtp0-wide-kv8-hero-p6', 'hero-v2-rank',
                    'fixed-2x-mtp0-wide-kv8-hero-gptq-arc', 'fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc', 'fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc'):
        assert result['server'] == CONTROL['server']
    if name == 'control':
        result.pop('arena_variant')
        assert result == CONTROL
    elif name in ('phase2-four', 'reclaim-phase2-four'):
        assert result['active_limit'] == 4 and result['windows'] == [131072]
    elif name in ('context-64k', 'reclaim-context-64k'):
        kwargs = dict(window=result['windows'][0],cap=result['completion_cap'],wall=1800,base_url='http://model/v1')
        duck = duck_config(result,ROOT,**kwargs)
        assert duck.max_model_len == 65536
        assert result['server']['max_model_len'] == 262144 and result['active_limit'] == 2
        if name == 'reclaim-context-64k':
            assert result['cache_experiment']['reclaim_skipped_blocks'] is True
            assert result['cache_experiment']['compaction_changed'] is True
    elif name in ('grant-180s', 'reclaim-grant-180s'):
        assert result['max_level_grant_seconds'] == 180 and result['active_limit'] == 2
    elif name == 'reclaim-grant-2t1':
        assert result['max_level_grant_t1_multiple'] == 2 and result.get('max_level_grant_seconds') is None
        assert result['active_limit'] == 2 and result['windows'] == [131072]
    elif name == 'reclaim-regrant-2x':
        assert result['regrant_multiple'] == 2 and result.get('max_level_grant_t1_multiple') is None
        assert 'regrant max(2*spent, spent+30)' in result['level_budget_rule']
    elif name == 'reclaim-effort-medium':
        assert result['active_limit'] == 2 and result['windows'] == [131072]
        assert result['cache_experiment']['reclaim_skipped_blocks'] is True
    elif name.startswith('fixed-2x-'):
        clock = 'normalized' if name in ('fixed-2x-pool', 'fixed-2x-pool2-cap', 'fixed-2x-pool2-timed-novelty', 'fixed-2x-token-budget') else 'busy'
        assert result['budget_clock'] == clock and result['arena_variant']['reference_variant'] == 'reclaim-fixed-2x'
        if name in ('fixed-2x-mtp4', 'fixed-2x-mtp0-wide', 'fixed-2x-gittins-c110-wide'):
            depth = 4 if name == 'fixed-2x-mtp4' else 0
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': depth}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name in ('fixed-2x-mtp0-wide-kernels', 'fixed-2x-deep-explore', 'fixed-2x-depthwise'):
            from arc_infra.dense_kernels_patch import PHASE2
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 0, 'dense_kernels': PHASE2}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name in ('fixed-2x-mtp3-wide-kernels', 'fixed-2x-depthwise-mtp3', 'fixed-2x-depthwise-mtp3-fixed-order'):
            from arc_infra.dense_kernels_patch import PHASE2
            from arc_infra.serving_kernels_patch import PHASE4
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 3, 'dense_kernels': PHASE2,
                                        'serving_kernels': PHASE4}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name in ('fixed-2x-mtp3-wide-tweaks', 'fixed-2x-depthwise-mtp3-tweaks',
                    'fixed-2x-mtp3-wide-tweaks-nople', 'fixed-2x-depthwise-mtp3-tweaks-nople'):
            from arc_infra.dense_kernels_patch import PHASE2
            from arc_infra.serving_kernels_patch import TWEAKS, TWEAKS_NO_PLE
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 3, 'dense_kernels': PHASE2,
                                        'serving_kernels': TWEAKS_NO_PLE if name.endswith('-nople') else TWEAKS}
        elif name == 'fixed-2x-mtp0-wide-kv8-hero-6x64k':
            hero = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero')
            assert result['server'] == hero['server'] and not result['arena_variant']['checkpoint_and_serving_unchanged']
            assert {k: v for k, v in result.items() if k not in ('arena_variant', 'active_limit')} == \
                {k: v for k, v in hero.items() if k not in ('arena_variant', 'active_limit')}
        elif name == 'fixed-2x-mtp0-wide-kv8-hero-21x42k':
            hero = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero')
            assert result['server'] == hero['server'] and not result['arena_variant']['checkpoint_and_serving_unchanged']
            assert {k: v for k, v in result.items() if k not in ('arena_variant', 'active_limit', 'windows')} == \
                {k: v for k, v in hero.items() if k not in ('arena_variant', 'active_limit', 'windows')}
        elif name == 'fixed-2x-mtp0-wide-kv8-hero':
            from arc_infra.dense_kernels_patch import PHASE5 as DENSE5
            from arc_infra.stack_kernels_patch import PHASE5 as STACK5
            from arena.variants import MTP0_KV8_SERVING
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 0, 'dense_kernels': DENSE5,
                                        'serving_kernels': {**MTP0_KV8_SERVING, 'moe_backend': 'marlin'},
                                        'stack_kernels': STACK5}
            base = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kernels-kv8')
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant')} \
                == {k: v for k, v in base.items() if k not in ('server', 'arena_variant')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name == 'fixed-2x-mtp0-wide-kv8-stack':
            from arc_infra.dense_kernels_patch import PHASE5 as DENSE5
            from arc_infra.stack_kernels_patch import PHASE5 as STACK5
            from arena.variants import MTP0_KV8_SERVING
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 0, 'dense_kernels': DENSE5,
                                        'serving_kernels': MTP0_KV8_SERVING, 'stack_kernels': STACK5}
            base = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kernels-kv8')
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant')} \
                == {k: v for k, v in base.items() if k not in ('server', 'arena_variant')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name in ('fixed-2x-mtp3-wide-stack', 'fixed-2x-depthwise-mtp3-stack'):
            from arc_infra.dense_kernels_patch import PHASE5 as DENSE5
            from arc_infra.serving_kernels_patch import PHASE4
            from arc_infra.stack_kernels_patch import PHASE5 as STACK5
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 3, 'dense_kernels': DENSE5,
                                        'serving_kernels': PHASE4, 'stack_kernels': STACK5}
            base = apply_variant(CONTROL, {'fixed-2x-mtp3-wide-stack': 'fixed-2x-mtp3-wide-kernels',
                                           'fixed-2x-depthwise-mtp3-stack': 'fixed-2x-depthwise-mtp3'}[name])
            # Serving is the only difference from the phase-4 arm it extends.
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant', 'depthwise_serving_source')} \
                == {k: v for k, v in base.items() if k not in ('server', 'arena_variant', 'depthwise_serving_source')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name == 'fixed-2x-mtp0-wide-kernels-kv8':
            from arc_infra.dense_kernels_patch import PHASE2
            from arena.variants import MTP0_KV8_SERVING
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 0, 'dense_kernels': PHASE2,
                                        'serving_kernels': MTP0_KV8_SERVING}
            assert MTP0_KV8_SERVING['kv_cache_8bit'] == 'int8' and MTP0_KV8_SERVING['topk_sampler'] is True
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name == 'fixed-2x-mtp3-wide-kernels-kv8':
            from arc_infra.dense_kernels_patch import PHASE2
            from arc_infra.serving_kernels_patch import PHASE4
            assert result['server'] == {**CONTROL['server'], 'mtp_tokens': 3, 'dense_kernels': PHASE2,
                                        'serving_kernels': {**PHASE4, 'kv_cache_8bit': 'int8'},
                                        'max_cudagraph_capture_size': 48}
        elif name in ('fixed-2x-depthwise-mtp3-w4a16', 'fixed-2x-depthwise-mtp3-gptq-arc',
                      'fixed-2x-depthwise-mtp3-gptq-arc-w4a16'):
            from arc_infra.dense_kernels_patch import PHASE2
            from arc_infra.serving_kernels_patch import PHASE4
            from arena.variants import EXPERT_OVERLAY_GPTQ_ARC
            server = {**CONTROL['server'], 'mtp_tokens': 3, 'dense_kernels': PHASE2,
                      'serving_kernels': {**PHASE4, 'moe_backend': 'marlin' if name.endswith('-w4a16') else None}}
            if '-gptq-arc' in name:
                server['expert_overlay'] = EXPERT_OVERLAY_GPTQ_ARC
            assert result['server'] == server
            source = apply_variant(CONTROL, 'fixed-2x-depthwise-mtp3')
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant', 'expert_quant_serving_source')} \
                == {k: v for k, v in source.items() if k not in ('server', 'arena_variant')}
        elif name in ('fixed-2x-mtp0-wide-kv8-hero-gptq-arc', 'fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc', 'fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc'):
            from arena.variants import EXPERT_OVERLAY_GPTQ_ARC, EXPERT_QUANT_SOURCE
            source = apply_variant(CONTROL, EXPERT_QUANT_SOURCE[name])
            assert result['server'] == {**source['server'], 'expert_overlay': EXPERT_OVERLAY_GPTQ_ARC}
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant', 'expert_quant_serving_source')} \
                == {k: v for k, v in source.items() if k not in ('server', 'arena_variant')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name in ('fixed-2x-mtp0-wide-kv8-hero-prune12', 'fixed-2x-mtp0-wide-kv8-hero-prune25'):
            from arena.variants import EXPERT_PRUNE_VARIANTS
            hero = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero')
            spec = EXPERT_PRUNE_VARIANTS[name]
            assert result['server'] == {**hero['server'], 'expert_prune': {'mask': spec['mask'],
                                                                           'sha256': spec['sha256']}}
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant', 'active_limit')} \
                == {k: v for k, v in hero.items() if k not in ('server', 'arena_variant', 'active_limit')}
            assert result['windows'] == [65536] and result['server']['mtp_tokens'] == 0
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name == 'fixed-2x-mtp0-wide-kv8-hero-p6':
            from arc_infra.dense_kernels_patch import PHASE6_FUSED
            uva = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero-uva')
            assert result['server'] == {**uva['server'], 'dense_kernels': PHASE6_FUSED}
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant')} \
                == {k: v for k, v in uva.items() if k not in ('server', 'arena_variant')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name == 'fixed-2x-mtp0-wide-kv8-hero-prefix':
            from arena.variants import HERO_PREFIX_CACHE
            hero = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero')
            assert result['cache_experiment'] == {**hero['cache_experiment'], 'prefix_cache': HERO_PREFIX_CACHE}
            assert HERO_PREFIX_CACHE == {'mamba_align_fix': True, 'mamba_keep_state': True,
                                         'mamba_demote_superseded': True}
            assert {k: v for k, v in result.items() if k not in ('cache_experiment', 'arena_variant')} \
                == {k: v for k, v in hero.items() if k not in ('cache_experiment', 'arena_variant')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
            assert result['arena_variant']['changed']['cache_experiment.prefix_cache']['after'] == HERO_PREFIX_CACHE
        elif name == 'fixed-2x-mtp0-wide-kv8-hero-uva':
            hero = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero')
            assert result['server'] == {**hero['server'], 'serving_kernels': {
                **hero['server']['serving_kernels'], 'ple_uva': True, 'embed_host': True}}
            assert {k: v for k, v in result.items() if k not in ('server', 'arena_variant')} \
                == {k: v for k, v in hero.items() if k not in ('server', 'arena_variant')}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        elif name in ('fixed-2x-commit', 'fixed-2x-commit-queue'):
            assert result['server'] == {**CONTROL['server'], 'scheduling_policy': 'priority'}
            assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        else:
            assert result['server'] == CONTROL['server']
    elif name == 'hero-v2-rank':
        # The rank scheduler on the 25%-pruned hero serving (tests/test_arena_rank.py checks the rest).
        source = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero-prune25')
        assert result['server'] == source['server'] and result['level_budget_policy'] == 'rank'
        assert result['arena_variant']['reference_variant'] == 'reclaim-fixed-2x'
        assert not result['arena_variant']['checkpoint_and_serving_unchanged']
    elif name == 'reclaim-fixed-2x':
        assert result['budget_clock'] == 'busy' and result['level_budget_policy'] == 'fixed-2x'
        assert result['backlog_policy'] == 'readmissions' and result['prioritize_parked_progress'] is True
        assert result['active_limit'] == 2 and result['windows'] == [131072] and result['server'] == CONTROL['server']
    elif name == 'reclaim-64k-keep3':
        assert result['windows'] == [65536] and result['compaction_keep_observations'] == 3
        assert result['cache_experiment']['compaction_changed'] is True and result['active_limit'] == 2
    elif name in ('cache-reclaim', 'reclaim-mtp1', 'reclaim-mtp4', 'reclaim-ngram', 'reclaim-ngram8'):
        if name == 'reclaim-ngram':
            assert result['server'].pop('speculative') == NGRAM_SPECULATION
        if name == 'reclaim-ngram8':
            assert result['server'].pop('speculative') == NGRAM8_SPECULATION
        if name != 'cache-reclaim':
            assert result['server']['mtp_tokens'] == {'reclaim-mtp1': 1, 'reclaim-mtp4': 4, 'reclaim-ngram': 0, 'reclaim-ngram8': 0}[name]
            result['server']['mtp_tokens'] = 3
        assert result['cache_experiment'].pop('reclaim_skipped_blocks') is True
        assert not result.pop('arena_variant')['checkpoint_and_serving_unchanged']
        assert result == CONTROL
    else:
        assert result['server'] == {**CONTROL['server'], 'max_cudagraph_capture_size': 128}
        assert result['active_limit'] == 2 and result['windows'] == [131072]
        assert not result['arena_variant']['checkpoint_and_serving_unchanged']
        if name == 'reclaim-graph-128':
            assert result['cache_experiment']['reclaim_skipped_blocks'] is True


def setup_scheduler(cap=None, active=2, t1_multiple=None):
    now = Now()
    wall, games = make_games(now, count=6)
    sched = Scheduler(games,wall,max_grant_seconds=cap,active_limit=active,max_grant_t1_multiple=t1_multiple)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.advance(100)
    games[0].note_level(1,wall.elapsed())
    now.advance(50)
    games[1].note_level(1,wall.elapsed())
    sched.tick()
    return now,games,sched


def test_cap_covers_opening_focus_level_completion_and_readmission():
    now,games,sched = setup_scheduler(cap=180)
    assert games[0].clock.budget == 150  # Deduct time already spent on the level.
    assert games[1].clock.budget == 180  # Control grants 300 here.
    games[1].note_level(2,150)
    sched.on_level(games[1])
    assert games[1].clock.budget == 180  # Control grants 450 here.
    now.advance(180)
    sched.tick()
    assert games[0].state == games[1].state == 'parked'
    assert len(sched.active()) == 2 and all(g.clock.budget <= 180 for g in sched.active())
    assert sched.grant_for(games[0]) <= 180
    assert sched.summary()['max_grant_seconds'] == 180


def test_t1_relative_cap_is_renewable_and_scales_with_each_games_own_t1():
    now,games,sched = setup_scheduler(t1_multiple=2)
    # g0: T1=100, 50 s already spent on level 2 -> control 150, cap 200 -> 150.
    # g1: T1=150, fresh level 2 -> control 300, cap 300 -> 300.
    assert games[0].clock.budget == 150 and games[1].clock.budget == 300
    games[0].note_level(2, now.value)
    sched.on_level(games[0])
    assert games[0].clock.budget == 200  # control would grant 3*100 = 300
    now.advance(200)
    sched.tick()
    assert games[0].state == 'parked' and games[1].state == 'active'
    # The remaining (k+1)*T1 allowance (300 - 200) comes first; once that is spent the
    # regrant rule (1.5x spent = 450) is capped at 2*T1 = 200: renewable, never starving.
    assert sched.grant_for(games[0]) == 100
    games[0].clock.grant(100)
    games[0].clock.resume()
    now.advance(100)
    games[0].clock.pause()
    assert sched.grant_for(games[0]) == 200
    assert sched.summary()['max_grant_t1_multiple'] == 2
    with pytest.raises(ValueError, match='max_grant_t1_multiple'):
        Scheduler(games, Now(), max_grant_t1_multiple=0.5)


def test_regrant_multiple_scales_readmission_grants_only():
    now,games,sched = setup_scheduler()
    assert sched.regrant_multiple == 1.5
    _now2,games2,sched2 = setup_scheduler()
    sched2.regrant_multiple = 2
    # Level budgets are untouched: the same (k+1)*T1 allowances as the control.
    assert [g.clock.budget for g in games2[:2]] == [g.clock.budget for g in games[:2]]
    # A parked level-0 game spent the 150 s opening: control 225 s, 2x variant 300 s.
    assert games[2].clock.budget_seconds() == 150
    assert sched.grant_for(games[2]) == 225 and sched2.grant_for(games2[2]) == 300
    # The +30 floor still binds for tiny spends under both rules.
    fresh_now = Now()
    _wall, fresh = make_games(fresh_now, count=2)
    fresh_sched = Scheduler(fresh, _wall, regrant_multiple=2)
    assert fresh_sched.grant_for(fresh[0]) == 30 == Scheduler(fresh, _wall).grant_for(fresh[0])
    assert fresh_sched.summary()['regrant_multiple'] == 2
    with pytest.raises(ValueError, match='regrant_multiple'):
        Scheduler(games, Now(), regrant_multiple=0.5)


def test_four_active_games_fill_after_same_two_clear_opening_trigger():
    _now,games,sched = setup_scheduler(active=4)
    assert sched.switch_wall_seconds == 150
    sched.tick()
    assert [g.index for g in sched.active()] == [0,1,2,3]
    assert [g.index for g in sched.backlog()] == [4,5]


@pytest.mark.parametrize('cap', [0,29,True,'180',float('inf'),float('nan')])
def test_invalid_allocation_caps_rejected(cap):
    now = Now()
    wall,games = make_games(now,count=2)
    with pytest.raises(ValueError,match='max_grant_seconds'):
        Scheduler(games,wall,max_grant_seconds=cap)


def test_capped_allocation_does_not_idle_when_no_backlog_exists():
    now,games,sched = setup_scheduler(cap=180)
    for game in games[2:]:
        sched.finish(game,'won')
    now.advance(500)
    sched.tick()
    assert len(sched.active())==2 and not sched.backlog()


def test_combined_or_nonpaired_interventions_rejected():
    with pytest.raises(ValueError,match='unmodified'):
        apply_variant(apply_variant(CONTROL,'phase2-four'),'context-64k')
    with pytest.raises(ValueError,match='paired Duck control'):
        apply_variant({**CONTROL,'trials':[{'kind':'pi'}]},'control')
    with pytest.raises(ValueError,match='requires vLLM'):
        apply_variant({**CONTROL,'backend':'sglang'},'graph-128')
    with pytest.raises(ValueError,match='32-token'):
        apply_variant({**CONTROL,'server':{**CONTROL['server'],'max_cudagraph_capture_size':64}},'graph-128')
    with pytest.raises(ValueError,match='requires vLLM'):
        apply_variant({**CONTROL,'backend':'sglang'},'cache-reclaim')
    with pytest.raises(ValueError,match='unpatched control'):
        apply_variant({**CONTROL,'cache_experiment':{'reclaim_skipped_blocks':True}},'cache-reclaim')


@pytest.mark.parametrize('name', VARIANTS)
def test_variant_bundle_records_exact_intervention_and_keeps_model_identity(tmp_path,name):
    spec = importlib.util.spec_from_file_location('variant_builder',ROOT/'scripts/build_arena.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    candidate = None
    output,study=builder.build('variant-'+name,output=tmp_path/name,paired_30m=True,
                               candidate_profile=candidate,arena_variant=name)
    assert study.get('candidate_profile') == candidate
    assert [t['kind'] for t in study['trials']]==['duck-transfer']
    saved=json.loads((output/'configs/comparison.json').read_text())
    assert saved==study and saved['arena_variant']['name']==name
    assert (output/'src/arena/variants.py').is_file()
    notebook = builder.notebook(study, 'runtime.blob', 'fake-digest', smoke=False)
    intro = notebook['cells'][0]['source']
    assert f"{study['windows'][0]:,}-token agent context" in intro
    if name == 'reclaim-context-64k':
        assert '128K' not in intro


def test_mtp_candidate_is_only_one_serving_change_from_reclaimed_flash_control():
    reference = apply_variant(CONTROL, 'cache-reclaim')
    candidate = apply_variant(CONTROL, 'reclaim-mtp1')
    assert candidate['arena_variant']['reference_variant'] == 'cache-reclaim'
    for doc in (reference, candidate):
        doc.pop('arena_variant')
    candidate['server']['mtp_tokens'] = 3
    assert candidate == reference
    with pytest.raises(ValueError, match='Flash vLLM'):
        apply_variant({**CONTROL, 'candidate_profile': 'ui-mate-9b'}, 'reclaim-mtp1')


def test_ngram_variant_differs_from_reclaimed_control_only_by_the_speculative_config():
    reference = apply_variant(CONTROL, 'cache-reclaim')
    candidate = apply_variant(CONTROL, 'reclaim-ngram')
    metadata = candidate.pop('arena_variant')
    reference.pop('arena_variant')
    # MTP off, n-gram on; the draft length equals the MTP-3 lookahead so admission is unchanged.
    assert candidate['server'].pop('speculative') == NGRAM_SPECULATION
    assert NGRAM_SPECULATION == {'method': 'ngram_gpu', 'num_speculative_tokens': 3,
                                 'prompt_lookup_max': 4, 'prompt_lookup_min': 2}
    assert candidate['server'].pop('mtp_tokens') == 0 and reference['server'].pop('mtp_tokens') == 3
    assert candidate == reference
    assert metadata['reference_variant'] == 'cache-reclaim'
    assert metadata['reference_difference'] == {
        'server.mtp_tokens': {'before': 3, 'after': 0},
        'server.speculative': {'before': None, 'after': NGRAM_SPECULATION}}
    assert metadata['changed'] == {'cache_experiment.reclaim_skipped_blocks': {'before': False, 'after': True},
                                   **metadata['reference_difference']}
    assert not metadata['checkpoint_and_serving_unchanged']
    assert not metadata['reference_checkpoint_and_serving_unchanged']
    # The agent's client-side lookahead follows the n-gram draft length, not the MTP depth of 0.
    cfg = duck_config(apply_variant(CONTROL, 'reclaim-ngram'), ROOT, window=131072, cap=16384, wall=1800,
                      base_url='http://model/v1')
    assert cfg.speculative_tokens == 3
    for extra in ({'candidate_profile': 'ui-mate-9b'}, {'backend': 'sglang'}):
        with pytest.raises(ValueError, match='Flash vLLM'):
            apply_variant({**CONTROL, **extra}, 'reclaim-ngram')
    for server in ({**CONTROL['server'], 'mtp_tokens': 1},
                   {**CONTROL['server'], 'speculative': NGRAM_SPECULATION}):
        with pytest.raises(ValueError, match='MTP-3 control'):
            apply_variant({**CONTROL, 'server': server}, 'reclaim-ngram')


def test_short_context_retains_reclamation_and_only_changes_context_policy():
    reference = apply_variant(CONTROL, 'cache-reclaim')
    candidate = apply_variant(CONTROL, 'reclaim-context-64k')
    assert candidate['arena_variant']['reference_variant'] == 'cache-reclaim'
    assert candidate['arena_variant']['reference_checkpoint_and_serving_unchanged']
    for doc in (reference, candidate):
        doc.pop('arena_variant')
    candidate['windows'] = [131072]
    candidate['cache_experiment']['compaction_changed'] = False
    assert candidate == reference
    for extra in ({'candidate_profile': 'ui-mate-9b'}, {'backend': 'sglang'}):
        with pytest.raises(ValueError, match='Flash vLLM'):
            apply_variant({**CONTROL, **extra}, 'reclaim-context-64k')


def test_graph_variant_keeps_cache_fix_and_changes_only_capture_ceiling():
    reference = apply_variant(CONTROL, 'cache-reclaim')
    candidate = apply_variant(CONTROL, 'reclaim-graph-128')
    metadata = candidate.pop('arena_variant')
    reference.pop('arena_variant')
    assert candidate['server'].pop('max_cudagraph_capture_size') == 128
    assert candidate == reference
    assert metadata['reference_difference'] == {
        'server.max_cudagraph_capture_size': {'before': 32, 'after': 128}}
    assert not metadata['reference_checkpoint_and_serving_unchanged']
    for extra in ({'candidate_profile': 'qwen38-27b-fp8'}, {'backend': 'sglang'}):
        with pytest.raises(ValueError, match='Flash vLLM'):
            apply_variant({**CONTROL, **extra}, 'reclaim-graph-128')


@pytest.mark.parametrize('name,changed_keys', [
    ('reclaim-phase2-four', {'active_limit'}),
    ('reclaim-grant-180s', {'max_level_grant_seconds', 'level_budget_rule'}),
    ('reclaim-grant-2t1', {'max_level_grant_t1_multiple', 'level_budget_rule'}),
    ('reclaim-effort-medium', {'reasoning_effort'}),
    ('reclaim-64k-keep3', {'windows', 'cache_experiment', 'compaction_keep_observations'}),
    ('reclaim-regrant-2x', {'regrant_multiple', 'level_budget_rule'}),
    ('reclaim-fixed-2x', {'budget_clock', 'level_budget_policy', 'backlog_policy', 'prioritize_parked_progress'}),
])
def test_scheduling_variants_retain_cache_fix_and_have_exact_reference_diff(name, changed_keys):
    reference = apply_variant(CONTROL, 'cache-reclaim')
    candidate = apply_variant(CONTROL, name)
    metadata = candidate.pop('arena_variant')
    reference.pop('arena_variant')
    actual = {k: {'before': reference.get(k), 'after': candidate.get(k)}
              for k in reference.keys() | candidate.keys()
              if reference.get(k) != candidate.get(k)}
    assert set(candidate) ^ set(reference) <= changed_keys
    assert set(actual) == changed_keys
    if name == 'reclaim-64k-keep3':
        assert metadata['reference_difference']['compaction_keep_observations'] == {'before': None, 'after': 3}
        assert metadata['reference_difference']['cache_experiment.compaction_changed'] == {'before': False, 'after': True}
        actual = {k: v for k, v in actual.items() if k != 'cache_experiment'}
        actual['cache_experiment.compaction_changed'] = {'before': False, 'after': True}
    assert metadata['reference_variant'] == 'cache-reclaim'
    assert metadata['reference_checkpoint_and_serving_unchanged'] is True
    assert metadata['reference_difference'] == actual
    assert candidate['cache_experiment']['reclaim_skipped_blocks'] is True
    # Exercise the real scheduler with the generated configuration.
    _, _, sched = setup_scheduler(
        cap=candidate.get('max_level_grant_seconds'), active=candidate['active_limit'],
        t1_multiple=candidate.get('max_level_grant_t1_multiple'))
    sched.tick()
    assert len(sched.active()) == candidate['active_limit']
    if name == 'reclaim-grant-180s':
        assert all(g.clock.budget <= 180 for g in sched.active())
    if name == 'reclaim-grant-2t1':
        assert all(g.clock.budget <= 2 * g.t1 for g in sched.active())
    if name == 'reclaim-effort-medium':
        assert duck_config(candidate, ROOT, window=131072, cap=16384, wall=1800,
                           base_url='http://model/v1').reasoning_effort == 'medium'
        # The Flash template rejects 'high' (effort-high-v1 failed at /tokenize); the
        # variant only ever emits an effort the served template accepts.
        from arc_infra.model_templates import allowed_efforts
        assert candidate['reasoning_effort'] in allowed_efforts('Qwen/Qwen3.8-Flash-Next-NVFP4')
        with pytest.raises(ValueError, match='does not accept'):
            apply_variant({**CONTROL, 'served_model': 'primitive-ai/Nex-N2.5-mini-NVFP4'}, name)
    for extra in ({'candidate_profile': 'ui-mate-9b'}, {'backend': 'sglang'}):
        with pytest.raises(ValueError, match='Flash vLLM'):
            apply_variant({**CONTROL, **extra}, name)


def test_capture_sizes_up_to_1024():
    """MFU-2 tput6 (23 Oct): explicit capture lists to 1024 pass; 1025 and an unsorted list do not."""
    import pytest
    from arena.serving import validate_graph_capture_sizes as v

    sizes = [1, 2, 4, 8, 16, 152, 168, 184, 200, 224, 256, 288, 320, 384, 448, 512, 576, 640, 768, 896, 1024]
    assert v(sizes, 1024) == sizes
    with pytest.raises(ValueError):
        v(sizes + [1025], 1025)
    with pytest.raises(ValueError):
        v([1, 512, 256], 256)
