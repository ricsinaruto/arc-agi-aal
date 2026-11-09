"""The arena bundle builder, notebook and kernel metadata; the Pi launcher variant."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_bundle_notebook_and_kernel_files(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build("arena-test", smoke=True, output=tmp_path / "bundle")
    digest = builder.kaggle_files(output, study, smoke=True)
    for package in builder.PACKAGES:
        assert (output / "src" / package / "__init__.py").is_file()
    assert not (output / "src/duck_controls").exists() and not (output / "src/sglang_bench").exists()
    assert (output / "pi/agent.mjs").is_file() and (output / "pi-arena/agent.mjs").is_file()
    assert (output / "pi/transport.mjs").is_file()
    assert (output / "bootstrap.py").is_file() and (output / "bootstrap_arena.py").is_file()
    assert (output / "configs/duck_controls.yaml").is_file() and (output / "configs/duck_public_notebook.json").is_file()
    comparison = json.loads((output / "configs/comparison.json").read_text())
    assert comparison["name"] == "arena-test" and comparison["smoke_active"] and len(comparison["games"]) == 25
    assert comparison["smoke"] == {"games": 3, "wall_seconds": 120, "notebook_seconds": 4080, "kernel_timeout": 5400}
    assert comparison["notebook_seconds"] == 7200 and comparison["kernel_timeout"] == 9000
    identity = json.loads((output / "MANIFEST.json").read_text())
    assert identity["dependencies_prepared"] is False and "requirements.txt" in identity["files"]
    notebook = json.loads((output / "experiment.ipynb").read_text())
    assert [c["cell_type"] for c in notebook["cells"]] == ["markdown", "code", "code"]
    for c in notebook["cells"][1:]:
        ast.parse(c["source"])
    assert digest in notebook["cells"][1]["source"] and "'--smoke'" in notebook["cells"][1]["source"]
    assert "--max-seconds', '4080'" in notebook["cells"][1]["source"]
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "richardcsaky/arc-arena-test" and metadata["dataset_sources"][0] == builder.DATASET
    assert metadata["machine_shape"] == "NvidiaRtxPro6000" and metadata["enable_internet"] is False
    upload = json.loads((output / "dataset-upload/dataset-metadata.json").read_text())
    assert upload["id"] == builder.DATASET and digest in upload["description"]
    assert (output / "dataset-upload/arena-test.tar.gz.blob").is_file()
    full_output, full_study = builder.build("arena-full", output=tmp_path / "full")
    builder.kaggle_files(full_output, full_study, smoke=False)
    source = json.loads((full_output / "experiment.ipynb").read_text())["cells"][1]["source"]
    assert "'--smoke'" not in source and "--max-seconds', '7200'" in source


def test_pi_arena_launcher_has_only_reviewed_timeout_and_modality_changes():
    original = (ROOT / 'pi/agent.mjs').read_text()
    arena = (ROOT / 'pi-arena/agent.mjs').read_text()
    replacements = {
        '    httpIdleTimeoutMs: 0,\n': '',
        "input: cfg.textOnly ? ['text'] : ['text', 'image']": "input: ['text', 'image']",
        'contextWindow: cfg.contextWindow ?? 262144, maxTokens: cfg.contextWindow ?? 262144':
            'contextWindow: 262144, maxTokens: 262144',
        "readFile: async p => Buffer.from((await rpc(\n"
        "      cfg.textOnly && p.toLowerCase().endsWith('.png') ? 'read_grid' : 'read',\n"
        "      {path: p})).output.trim(), 'base64')":
            "readFile: async p => Buffer.from((await rpc('read', {path: p})).output.trim(), 'base64')",
        "!cfg.textOnly && p.toLowerCase().endsWith('.png')": "p.toLowerCase().endsWith('.png')",
    }
    for changed, historical in replacements.items():
        assert arena.count(changed) == 1
        arena = arena.replace(changed, historical)
    assert arena == original


def test_separate_runtime_dataset_matches_notebook_upload_and_attachment(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build("context-test", paired_30m=True, output=tmp_path / "bundle")
    dataset = "richardcsaky/arc-arena-context-runtime"
    digest = builder.kaggle_files(output, study, smoke=False, runtime_dataset=dataset)
    notebook = json.loads((output / "experiment.ipynb").read_text())
    code = notebook['cells'][1]['source']
    assert '/kaggle/input/datasets/' + dataset in code
    assert '/kaggle/input/arc-arena-context-runtime' in code
    assert builder.DATASET not in code and digest in code
    for cell in notebook['cells'][1:]:
        ast.parse(cell['source'])
    assert json.loads((output / "kernel-metadata.json").read_text())["dataset_sources"] == [
        dataset, *builder.KAGGLE_INPUTS[1:]]
    assert json.loads((output / "dataset-upload/dataset-metadata.json").read_text())["id"] == dataset
    assert json.loads((output / "build-manifest.json").read_text())["runtime_dataset"] == dataset
    for value in ["no-owner", "a/b/c", "a/../b", "a/b'", "a/b\n"]:
        with pytest.raises(ValueError, match="owner/slug"):
            builder.notebook(study, 'x.blob', digest, smoke=False, runtime_dataset=value)


def test_bootstrap_arena_parses_and_names_its_steps():
    source = (ROOT / "scripts/arena/bootstrap.py").read_text()
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"main", "stage_pi"} <= names
    assert "serve-native" in source and "arena.run" in source and "--smoke" in source


def test_cache_variants_preserve_scheduler_context_and_compaction(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    original = json.loads((ROOT / 'configs/arena.json').read_text())
    for variant in ('blocks',):
        _, study = builder.build('arena-' + variant, output=tmp_path / variant, cache_variant=variant)
        for key in ('windows', 'active_limit', 'opening_switch_level1_games', 'completion_cap',
                    'reasoning_effort', 'temperature', 'wall_seconds', 'games'):
            assert study[key] == original[key]
        assert study['generation_slots'] == study['server']['max_num_seqs'] == 25
        assert study['server']['max_num_batched_tokens'] == 4096
        assert study['cache_experiment']['compaction_changed'] is False
        assert study['trials'] == [{'name': 'duck-transfer', 'kind': 'duck-transfer'}]
        assert study['cache_experiment']['kv_cache_dtype'] == ('fp8' if variant == 'fp8' else 'auto')


def test_recorded_budget_rule_follows_the_scheduling_variant(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    expected = {
        'reclaim-regrant-2x': ('regrant max(2*spent, spent+30)', 'regrant_multiple', 2),
        'reclaim-grant-2t1': ('2*T1 (renewable)', 'max_level_grant_t1_multiple', 2),
        'cache-reclaim': ('regrant max(1.5*spent, spent+30)', 'regrant_multiple', None),
    }
    for variant, (text, key, value) in expected.items():
        # The time clock these variants were run on (reproduction builds, --budget-clock time).
        _, study = builder.build('arena-' + variant, output=tmp_path / variant, arena_variant=variant, paired_30m=True,
                                 budget_clock='time')
        assert text in study['level_budget_rule'], study['level_budget_rule']
        assert study.get(key) == value
        assert study['budget_clock'] == 'inference' and 'token_budget' not in study
        # New builds (29 September): the same rule on the token clock, the 30 s floor as 1,920 tokens.
        _, tokens = builder.build('arena-' + variant, output=tmp_path / (variant + '-t'), arena_variant=variant,
                                  paired_30m=True)
        assert text.replace('spent+30', 'spent+1920') in tokens['level_budget_rule'], tokens['level_budget_rule']
        assert tokens.get(key) == value and tokens['budget_clock'] == 'tokens'
        assert tokens['token_budget']['floor_tokens'] == 1920 and tokens['token_budget']['purposes'] == ['gameplay']
        assert tokens['token_budget']['rule'] == tokens['level_budget_rule']
        assert {k: v for k, v in tokens.items() if k not in ('budget_clock', 'token_budget', 'level_budget_rule',
                                                             'backlog_priority', 'name')} == {
            k: v for k, v in study.items() if k not in ('budget_clock', 'level_budget_rule', 'backlog_priority', 'name')}
    # Regression (pool-v1 kernel, 25 September): the builder reset the normalized clock to inference
    # while keeping cost_normalization, and the trial refused the pair at the first game record.
    _, study = builder.build('arena-pool', output=tmp_path / 'pool', arena_variant='fixed-2x-pool', paired_30m=True)
    assert study['budget_clock'] == 'normalized' and study['audition_pool'] == 4
    assert 'completion tokens' in study['level_budget_rule']
    from arena.clock import ActiveClock, GlobalClock
    ActiveClock(GlobalClock(1), budget_clock=study['budget_clock'], normalization=study.get('cost_normalization'))


def test_known_unsupported_cache_builds_fail_before_creating_output(tmp_path):
    import importlib.util

    import pytest
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for variant in ('cpu', 'fp8'):
        output = tmp_path / variant
        with pytest.raises(NotImplementedError, match='backend'):
            builder.build('unsupported', output=output, cache_variant=variant, dependencies=True)
        assert not output.exists()


def test_paired_30m_bundle_is_independent_and_includes_both_agents(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    historical = (ROOT / 'configs/arena.json').read_bytes()
    output, study = builder.build('paired-test', output=tmp_path / 'paired', paired_30m=True, include_pi=True)
    builder.kaggle_files(output, study, smoke=False)
    assert study['wall_seconds'] == 1800 and len(study['games']) == 25
    assert [t['kind'] for t in study['trials']] == ['duck-transfer', 'pi']
    assert study['windows'] == [131072] and study['isolate_trial_cache']
    assert study['cache_admission'] == 'hybrid_blocks'
    assert study['generation_slots'] == 25 and study['server']['max_num_batched_tokens'] == 4096
    assert study['notebook_seconds'] >= 2 * (1800 + 600) + 1800
    notebook = json.loads((output / 'experiment.ipynb').read_text())
    assert '1800-second' in notebook['cells'][0]['source']
    assert 'qwen-c' in notebook['cells'][0]['source']
    for cell in notebook['cells'][1:]:
        ast.parse(cell['source'])
    assert (ROOT / 'configs/arena.json').read_bytes() == historical
    for kwargs in ({'smoke': True}, {'cache_variant': 'blocks'}):
        with pytest.raises(ValueError, match='separate full study'):
            builder.build('bad', output=tmp_path / 'bad', paired_30m=True, **kwargs)
    assert not (tmp_path / 'bad').exists()


def test_new_model_runs_default_to_duck_with_pi_explicitly_opted_in(tmp_path):
    import importlib.util

    import pytest
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for name, kwargs in [('flash', {}), ('nex', {'nex': True}),
                         ('sglang', {'sglang': True}), ('ui9', {'candidate_profile': 'ui-mate-9b'})]:
        _, duck = builder.build(name, paired_30m=True, output=tmp_path/name, **kwargs)
        _, both = builder.build(name+'-pi', paired_30m=True, include_pi=True,
                                output=tmp_path/(name+'-pi'), **kwargs)
        assert [t['kind'] for t in duck['trials']] == ['duck-transfer']
        assert [t['kind'] for t in both['trials']] == ['duck-transfer', 'pi']
        assert duck['wall_seconds'] == both['wall_seconds'] == 1800
        assert len(duck['games']) == len(both['games']) == 25
        assert duck['server'] == both['server'] and duck['windows'] == both['windows']
        assert duck['notebook_seconds'] == both['notebook_seconds'] - 2400
        assert duck['kernel_timeout'] == both['kernel_timeout'] - 2400
    with pytest.raises(ValueError, match='Pi opt-in'):
        builder.build('invalid', output=tmp_path/'invalid', paired_30m=True, include_pi=True,
                      candidate_profile='ui-mate-9b', ui_mate_native=True)
    assert not (tmp_path/'invalid').exists()


def test_exclude_games_drops_named_games_from_the_30m_roster_only(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build("roster-test", paired_30m=True, output=tmp_path / "bundle",
                                  arena_variant="reclaim-fixed-2x", opening_burst_requests=3,
                                  exclude_games=["ft09", "lp85-305b61c3"])
    assert len(study["games"]) == 23 and not any(g.startswith(("ft09", "lp85")) for g in study["games"])
    assert [g[:4] for g in study["games_excluded"]] == ["lp85", "ft09"] or sorted(g[:4] for g in study["games_excluded"]) == ["ft09", "lp85"]
    assert study["roster"].startswith("23 public games (excluded: ")
    saved = json.loads((output / "configs/comparison.json").read_text())
    assert saved["games"] == study["games"] and saved["arena_variant"]["name"] == "reclaim-fixed-2x"
    with pytest.raises(ValueError, match="Unknown games to exclude"):
        builder.build("roster-bad", paired_30m=True, output=tmp_path / "bad", exclude_games=["zz99"])
    with pytest.raises(ValueError, match="30-minute arena only"):
        builder.build("roster-smoke", smoke=True, output=tmp_path / "smoke", exclude_games=["ft09"])


def test_wall_seconds_extends_the_duck_arena_and_its_limits(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, base = builder.build("hour-base", paired_30m=True, output=tmp_path / "base",
                            arena_variant="fixed-2x-budget-4x", opening_burst_requests=3, exclude_games=["ft09", "lp85"])
    _, study = builder.build("hour-test", paired_30m=True, output=tmp_path / "hour",
                             arena_variant="fixed-2x-budget-4x", opening_burst_requests=3, exclude_games=["ft09", "lp85"],
                             wall_seconds=3600)
    assert base["wall_seconds"] == 1800 and study["wall_seconds"] == 3600
    assert study["notebook_seconds"] - base["notebook_seconds"] == 1800
    assert study["kernel_timeout"] - base["kernel_timeout"] == 1800
    assert study["level_budget_rule"].startswith("4*L*T1 for playing level L") and len(study["games"]) == 23
    with pytest.raises(ValueError, match="Duck-transfer arena only"):
        builder.build("hour-smoke", smoke=True, output=tmp_path / "smoke", wall_seconds=3600)
    _, evo = builder.build("hour-evolve", paired_30m=True, output=tmp_path / "evolve", arena_variant="reclaim-fixed-2x",
                           opening_burst_requests=3, exclude_games=["ft09", "lp85"], evolve=True, wall_seconds=3600)
    assert evo["evolution"]["round_wall_seconds"] == 3600 and evo["notebook_seconds"] == 41400
    assert evo["kernel_timeout"] <= 43200
    _, rehearsal = builder.build("evolve-rehearsal", paired_30m=True, output=tmp_path / "rehearsal",
                                 arena_variant="reclaim-fixed-2x", opening_burst_requests=3,
                                 exclude_games=["ft09", "lp85"], evolve=True, wall_seconds=600, evolve_rounds=1)
    assert rehearsal["evolution"]["rounds"] == 1 and rehearsal["evolution"]["round_wall_seconds"] == 600
    assert evo["evolution"]["rounds"] == 5
    # The Kaggle notebook runs the evolution driver (evolve-v3 ran a plain trial without the flag).
    source = "".join(builder.notebook(evo, "b.tar.gz.blob", "0" * 64, smoke=False)["cells"][1]["source"])
    assert "'--evolve'," in source and "'--evolve'," not in "".join(
        builder.notebook(study, "b.tar.gz.blob", "0" * 64, smoke=False)["cells"][1]["source"])
    # Short probe (26 September): only the named games, concurrently, never switching, 900 s.
    _, probe = builder.build("probe-test", paired_30m=True, output=tmp_path / "probe", arena_variant="reclaim-fixed-2x",
                             opening_burst_requests=3, wall_seconds=900, probe_games=["ar25", "tu93", "vc33"])
    assert sorted(g[:4] for g in probe["games"]) == ["ar25", "tu93", "vc33"] and len(probe["games_excluded"]) == 22
    assert probe["opening_switch_level1_games"] == 4 and probe["probe"]["games"] == ["ar25", "tu93", "vc33"]
    assert probe["wall_seconds"] == 900 and probe["notebook_seconds"] == base["notebook_seconds"] - 900
    for bad in (dict(probe_games=["zz99"], wall_seconds=900), dict(probe_games=["ar25"]),
                dict(probe_games=["ar25"], wall_seconds=900, exclude_games=["ft09"])):
        with pytest.raises(ValueError, match="probe"):
            builder.build("probe-bad", paired_30m=True, output=tmp_path / "bad", arena_variant="reclaim-fixed-2x", **bad)
    # Teacher runs (26 September): Flash plays the generated games shipped in the bundle, without their notes.
    out, teacher = builder.build("teacher-test", paired_30m=True, output=tmp_path / "teacher",
                                 arena_variant="reclaim-fixed-2x", opening_burst_requests=3, wall_seconds=1800,
                                 generated_games=ROOT / "data/posttrain/games")
    assert teacher["games"] and all(g.startswith("q") for g in teacher["games"]) and not teacher["games_excluded"]
    assert teacher["opening_switch_level1_games"] == len(teacher["games"]) + 1
    shipped = out / teacher["generated_games"]["dir"]
    assert sorted(p.parent.parent.name for p in shipped.glob("*/*/metadata.json")) == [g[:4] for g in teacher["games"]]
    assert not (out / "generated_games/notes").exists()
    with pytest.raises(ValueError, match="generated-games"):
        builder.build("teacher-bad", paired_30m=True, output=tmp_path / "bad", arena_variant="reclaim-fixed-2x",
                      wall_seconds=900, generated_games=ROOT / "data/posttrain/games", probe_games=["ar25"])


def test_serving_ab_bench_alternates_control_and_candidates_in_one_container(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    transcript = tmp_path / "t.txt"
    transcript.write_text("line\n" * 100)
    _, study = builder.build("ab-test", paired_30m=True, output=tmp_path / "ab", arena_variant="cache-reclaim",
                             throughput_bench=True, bench_transcript=transcript, serving_ab=True)
    names = [t["name"] for t in study["trials"]]
    assert names[:3] == ["bench-control-1", "bench-index-share-1", "bench-kv-11gib-1"]
    assert len(names) == 6 and {t["kind"] for t in study["trials"]} == {"serving-bench"}
    assert study["trials"][1]["server_env"] == {"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1"}
    assert study["trials"][0]["server_env"] == {} and study["notebook_seconds"] == 6600
    with pytest.raises(ValueError, match="--serving-ab needs --throughput-bench"):
        builder.build("ab-bad", paired_30m=True, output=tmp_path / "bad", arena_variant="cache-reclaim", serving_ab=True)
    source = (ROOT / "scripts/arena/bootstrap.py").read_text()
    assert "if server_env(trial) != current_env:" in source and '"event": "server_restart"' in source
    assert 'probe.rename(probe.with_name(f"cache-probe-' in source


def test_new_builds_default_the_duck_compaction_summary_to_the_prefilled_empty_think_block():
    """compaction reuses the gameplay prompt cache without reasoning in every new run;
    ``--compaction-thinking gameplay`` rebuilds an older study byte-identically (no key)."""
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("build_arena_compaction", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    duck = {"trials": [{"name": "duck-transfer", "kind": "duck-transfer"}, {"name": "pi", "kind": "pi"}]}
    assert builder.with_compaction_thinking(dict(duck))["compaction_thinking"] == "prefilled-empty"
    assert builder.with_compaction_thinking(dict(duck), "off")["compaction_thinking"] == "off"
    assert "compaction_thinking" not in builder.with_compaction_thinking(dict(duck), "gameplay")
    assert "compaction_thinking" not in builder.with_compaction_thinking({**duck, "compaction_thinking": "gameplay"})
    chosen = {**duck, "compaction_thinking": "off"}  # a component's choice stands
    assert builder.with_compaction_thinking(dict(chosen))["compaction_thinking"] == "off"
    with pytest.raises(ValueError, match="contradicts"):
        builder.with_compaction_thinking(dict(chosen), "prefilled-empty")
    bench = {"trials": [{"name": "bench", "kind": "serving-bench"}]}
    assert "compaction_thinking" not in builder.with_compaction_thinking(dict(bench))
    with pytest.raises(ValueError, match="no Duck trial"):
        builder.with_compaction_thinking(dict(bench), "off")
