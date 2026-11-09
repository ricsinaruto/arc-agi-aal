"""Duck controls: study scope, analyzer environment, the public benchmark and the bundle."""

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from duck_controls import public_trial
from duck_controls.run import (
    SELECTED_CLOCK_SESSIONS,
    analyzer_environment,
    public_command,
    scope,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ("ft09-0d8bbf25", "lp85-305b61c3", "ar25-0c556536")
CONFIG = json.loads((ROOT / "configs/duck_public_notebook.json").read_text())


def test_study_scope_is_selected_games_or_a_full_fleet():
    assert scope({"games": list(PUBLIC)}) == (list(PUBLIC), None)
    assert scope({"sessions": 110}) == (None, 110)
    with pytest.raises(ValueError, match="either games"):
        scope({"games": list(PUBLIC), "sessions": 25})
    with pytest.raises(ValueError, match="25 or 110"):
        scope({"sessions": 30})
    assert SELECTED_CLOCK_SESSIONS == 25  # selected games keep the 110/25 planning clock


def test_analyzer_environment_keeps_only_the_notebook_agent_settings(tmp_path):
    (tmp_path / "serving-environment.json").write_text(
        json.dumps(
            {
                "PYTHONPATH": "/runtime/site",
                "LD_LIBRARY_PATH": "/cuda",
                "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1234/v1",
                "LOCAL_ANALYZER_TEMPERATURE": "0.6",
                "LOCAL_ANALYZER_YIELD_SECONDS": "60",
                "OPENAI_API_KEY": "offline-kaggle-local-server",
                "MULTIMODAL_UPSCALE": "4",
                "INFERENCE_ANALYZER_MODEL": "Qwen/Qwen3.8-Flash-Next-NVFP4",
            }
        )
    )
    env = analyzer_environment(tmp_path)
    assert "PYTHONPATH" not in env and "LD_LIBRARY_PATH" not in env
    assert "LOCAL_ANALYZER_CONTEXT_WINDOW" not in env
    pinned = analyzer_environment(tmp_path, context_window=32768)
    assert pinned["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768"
    assert env["LOCAL_ANALYZER_TEMPERATURE"] == "0.6" and env["MULTIMODAL_UPSCALE"] == "4"
    # v5c: the control samples at the corrected agent's temperature.
    assert analyzer_environment(tmp_path, temperature=0.7)["LOCAL_ANALYZER_TEMPERATURE"] == "0.7"
    assert env["LOCAL_ANALYZER_YIELD_SECONDS"] == "60"
    (tmp_path / "serving-environment.json").write_text("{}")
    with pytest.raises(RuntimeError, match="analyzer environment"):
        analyzer_environment(tmp_path)
    command = public_command("/bundle", Path("/out"), "/games", 600, list(PUBLIC), 1)
    assert command[command.index("--config") + 1] == "/bundle/configs/duck_public_notebook.json"
    assert command.count("--game") == 3 and command[command.index("--passes") + 1] == "1"


def test_public_benchmark_is_rebuilt_from_the_notebook_fields(tmp_path):
    pytest.importorskip("imageio")
    from types import SimpleNamespace

    games = [SimpleNamespace(env_name=g) for g in PUBLIC]
    bm = public_trial.build_benchmark(CONFIG, games, tmp_path, 600, passes=2)
    solver = bm.solver
    assert (
        solver.label,
        solver.model,
        solver.max_runtime_s_per_game,
        solver.analyzer_timeout,
        solver.concurrency,
        solver.max_actions_per_game,
        solver.save_request_logs,
        solver.start_local_server,
        solver.cancel_drain_timeout_s,
    ) == ("duck-harness", "local", 600.0, 900.0, 28, None, False, False, 120.0)
    assert bm.label == "duck-harness-kaggle" and bm.n_passes == 2 and bm.job_dir == tmp_path
    assert bm.periodic_save_interval_s == 600.0 and bm.game_weights is None
    target = public_trial.build_target(CONFIG)
    assert target.kernel_slug == "taaf-duck-harness-kaggle-share"
    assert target.accelerator == "NvidiaRtxPro6000" and not target.actual_run_as_submission
    assert not target.run_as_submission and target.make_share_version
    assert len(CONFIG["public_game_ids"]) == 25
    order = [CONFIG["public_game_ids"].index(g) for g in ("lp85-305b61c3", "ar25-0c556536", "ft09-0d8bbf25")]
    assert order == sorted(order)  # the notebook's order is kept for the selected games
    assert CONFIG["git_status"].startswith("git status:")
    assert sys.version_info >= (3, 12)


def test_bundle_packages_source_configs_and_kaggle_files(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_duck_controls as builder

    output, digest = builder.build("fixture", games=PUBLIC, gameplay_seconds=600, output=tmp_path / "fixture")
    manifest = json.loads((output / "build-manifest.json").read_text())
    study = json.loads((output / "study.json").read_text())
    assert study["games"] == list(PUBLIC) and study["gameplay_seconds"] == 600
    assert "sessions" not in study
    assert [t["kind"] for t in study["trials"]] == ["corrected", "public"]
    config = (output / "configs/experiment.yaml").read_text()
    assert "gameplay_seconds: 600\n" in config and "gameplay_seconds: 900" not in config
    assert "max_model_len: 65536\n" in config and "reasoning_effort: xhigh\n" in config
    assert "max_output_tokens: 8192\n" in config
    # v5b: one server per trial; the corrected one profiled on the fork, the control pinned.
    assert "server" not in study
    # v5c: the fork serves the native 262K (full profiled pool); the agent keeps 64K.
    assert study["servers"] == {
        "duck-transfer": {"max_model_len": 262144, "kv_cache_gib": 0, "prefix_caching": True, "gpu_memory_utilization": "0.96", "python": "agent"},
        "duck-public-notebook": {"max_model_len": 65536, "kv_cache_gib": 5, "prefix_caching": True},
    }
    assert study["corrected"]["context_window"] == 65536 and study["corrected"]["server_context"] == 262144
    assert "256K server context, 64K agent limit" in study["corrected"]["server"]
    with pytest.raises(ValueError, match="fork server context"):
        builder.build("bad", games=list(PUBLIC), gameplay_seconds=600, output=tmp_path / "bad", fork_max_model_len=32768)
    assert [t["server_dir"] for t in study["trials"]] == ["server-duck-transfer", "server-duck-public-notebook"]
    assert study["notebook_seconds"] == 1800 + 600 + 2 * (600 + 200)
    assert study["corrected"]["python_state"] == "persistent"
    assert study["corrected"]["max_output_tokens"] == 8192
    assert {"prompt_layout", "feedback", "game_over", "plumbing", "server", "token_estimate"} <= set(study["corrected"])
    assert "64K context" in study["policy"] and "32K analyzer window" in study["policy"]
    assert "temperature (0.7" in study["policy"]
    assert "prefix caching on" in study["policy"] and "utilization 0.96" in study["policy"]
    assert "One server per trial" in study["policy"] and "5 GiB KV" in study["policy"]
    pinned, _ = builder.build("pinned", games=PUBLIC, gameplay_seconds=600, output=tmp_path / "pinned", gpu_memory_utilization=None)
    pinned_study = json.loads((pinned / "study.json").read_text())
    assert pinned_study["servers"]["duck-transfer"] == pinned_study["servers"]["duck-public-notebook"]
    assert "gpu_memory_utilization" not in pinned_study["servers"]["duck-transfer"]
    for name in (
        "src/arc_infra/fleet.py",
        "src/arc_infra/serving_setup.py",
        "src/arc_infra/serving_fork.py",
        "src/inference/agent/tool_agent.py",
        "src/taaf/benchmark.py",
        "src/duck_transfer/trimming.py",
        "src/duck_controls/public_trial.py",
        "configs/duck_public_notebook.json",
        "kaggle/flash_bootstrap.py",
        "kaggle/bootstrap.py",
        "kaggle/runtime-pins/manifest.json",
    ):
        assert (output / name).is_file(), name
    assert (output / "src/inference/agent/tool_agent.py").read_bytes() == (
        ROOT / "src/inference/agent/tool_agent.py"
    ).read_bytes()
    assert not list(output.rglob("*.pkl")) and not list(output.rglob("__pycache__"))
    assert list((output / "wheels").glob("imageio-*.whl"))
    kernel = builder.kaggle_files(output, digest, "-r2")
    assert kernel["id"] == "richardcsaky/arc-fixture-r2" and kernel["enable_gpu"]
    book = json.loads((output / "experiment.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
    assert digest in code and "duck_controls.run" in code and "fixture.zip.blob" in code
    assert "start_server(" in code and 'stop_server(OUTPUT, working)' in code and '"--trial", trial["name"]' in code
    assert "server-duck-transfer" in json.dumps(study) and code.count("stop_server(") == 2

    assert (output / "dataset-upload/fixture.zip.blob").read_bytes() == (
        output / "source.zip"
    ).read_bytes()
    with zipfile.ZipFile(io.BytesIO((output / "source.zip").read_bytes())) as bundle:
        assert set(bundle.namelist()) == set(manifest["files"]) | {"build-manifest.json"}
    with pytest.raises(ValueError, match="pinned 32K"):
        builder.build("fleet", sessions=110, gameplay_seconds=3600, output=tmp_path / "fleet")
    fleet_output, _ = builder.build(
        "fleet", sessions=110, gameplay_seconds=3600, output=tmp_path / "fleet",
        max_model_len=32768, kv_cache_gib=5,
    )
    fleet_study = json.loads((fleet_output / "study.json").read_text())
    assert fleet_study["sessions"] == 110 and "games" not in fleet_study
    assert fleet_study["notebook_seconds"] > 2 * 3600


def test_run_merges_per_trial_invocations(monkeypatch, tmp_path):
    """v5b: the notebook runs duck_controls.run once per trial; study.json accumulates."""

    from duck_controls import run

    study = {
        "name": "fixture", "gameplay_seconds": 600, "notebook_seconds": 4000, "games": list(PUBLIC),
        "trials": [
            {"name": "duck-transfer", "kind": "corrected", "server_dir": "server-duck-transfer"},
            {"name": "duck-public-notebook", "kind": "public", "server_dir": "server-duck-public-notebook"},
        ],
        "servers": {
            "duck-transfer": {"max_model_len": 262144, "kv_cache_gib": 0, "prefix_caching": True, "gpu_memory_utilization": "0.96"},
            "duck-public-notebook": {"max_model_len": 65536, "kv_cache_gib": 5, "prefix_caching": True},
        },
    }
    source = tmp_path / "source"
    (source / "configs").mkdir(parents=True)
    import yaml

    config = yaml.safe_load((ROOT / "configs/duck_controls.yaml").read_text())
    config.update(gameplay_seconds=600, max_model_len=65536)
    (source / "configs/experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_path / "study.json").write_text(json.dumps(study))
    output = tmp_path / "out"
    calls = []

    async def fake_corrected(cfg, games_dir, folder, interpreter, games=None, sessions=None):
        calls.append(("corrected", games))
        return {
            "games": [], "mean_score": 1.0, "actions": 2, "levels": 1, "failed_sessions": [],
            "task_errors": [], "engine": {"valid": True}, "gameplay_seconds": 600,
            "completed_replies": 3, "completed_reply_tokens": 4,
        }

    def fake_public(source, output, folder, games_dir, seconds, env, games=None, passes=1, server_dir=None, temperature=None):
        assert temperature == 0.7  # v5c: the control samples at the config temperature
        calls.append(("public", str(server_dir)))
        return {"status": "complete", "mean_final_score": 2.0}

    async def ok(cfg):
        return {"ok": True}

    monkeypatch.setattr(run, "corrected_trial", fake_corrected)
    monkeypatch.setattr(run, "public_trial", fake_public)
    monkeypatch.setattr(run, "validate_api", ok)
    monkeypatch.setattr(run, "wait_idle", ok)
    monkeypatch.setattr(run, "hardware", lambda: {"gpu": "fake"})
    monkeypatch.setattr(run, "find_games", lambda root: tmp_path / "games")
    monkeypatch.setattr(run.signal, "setitimer", lambda *a: None)
    base = ["--source", str(source), "--study", str(tmp_path / "study.json"), "--output", str(output),
            "--analysis-python", "/py", "--notebook-start-epoch", str(run.time.time())]
    monkeypatch.setattr(sys, "argv", ["run", *base, "--trial", "duck-transfer", "--server-dir", str(output / "server-duck-transfer")])
    run.main()
    record = json.loads((output / "study.json").read_text())
    assert [t["trial"]["name"] for t in record["trials"]] == ["duck-transfer"]
    assert "utilization 0.96" in record["serving_profiles"]["duck-transfer"]
    assert "server context 262144, agent limit 65536" in record["serving_profiles"]["duck-transfer"]
    monkeypatch.setattr(sys, "argv", ["run", *base, "--trial", "duck-public-notebook", "--server-dir", str(output / "server-duck-public-notebook")])
    run.main()
    record = json.loads((output / "study.json").read_text())
    assert [t["trial"]["name"] for t in record["trials"]] == ["duck-transfer", "duck-public-notebook"]
    assert record["trials"][1]["server"]["kv_cache_gib"] == 5 and record["trials"][1]["status"] == "complete"
    assert "explicit 5 GiB KV" in record["serving_profiles"]["duck-public-notebook"]
    assert calls == [("corrected", list(PUBLIC)), ("public", str(output / "server-duck-public-notebook"))]
    assert len(record["api_validations"]) == 1 and record["hardware"] == {"gpu": "fake"}
    monkeypatch.setattr(sys, "argv", ["run", *base, "--trial", "nope"])
    with pytest.raises(ValueError, match="Unknown trial"):
        run.main()


@pytest.mark.parametrize('note,expected,source', [
    ('tokens=19794', 19794, 'analyzer_final_note'),
    (None, 4724, 'action_history'),
    ('error: timeout tokens=19794', 4724, 'action_history'),
    ('tokens=3', 4724, 'action_history'),
])
def test_public_token_accounting_includes_inspection_tail(note, expected, source):
    from types import SimpleNamespace

    run = SimpleNamespace(game_id='ft09', state='gave_up', final_score=0,
                          levels_completed=0, number_of_levels=6,
                          history=[SimpleNamespace(generated_tokens=4700)],
                          final_generated_tokens=24, actions_per_level=[1],
                          base_actions_per_level=[43], solver_note=note)
    row = public_trial.run_rows([run])[0]
    assert row['generated_tokens'] == expected
    assert row['action_accounted_generated_tokens'] == 4724
    assert row['generated_tokens_source'] == source
