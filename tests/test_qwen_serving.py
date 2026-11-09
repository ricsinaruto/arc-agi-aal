"""Forked serving script, staged fork bundle, loader overrides and KV log capture."""

import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from arc_infra import serving as flash_serving
from qwen_baselines import serving

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/public-notebook-review/duck-bundle/serving_setup.py"
PARENT_LINES = {
    "# Keep the model author's validated single-GPU setting. Raising this to 0.98",
    "# would trade the known-safe GPU reserve for KV cache that vLLM can size itself.",
    'GPU_MEMORY_UTILIZATION = "0.92"',
    '        "num_speculative_tokens": 3,',
}


def import_fork(tmp_path, monkeypatch, utilization=None):
    for key in ("TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_WORKING_DIR", "TAAF_KAGGLE_SETUP_ENV"):
        monkeypatch.setenv(key, str(tmp_path))
    if utilization is None:
        monkeypatch.delenv("TAAF_VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    else:
        monkeypatch.setenv("TAAF_VLLM_GPU_MEMORY_UTILIZATION", utilization)
    spec = importlib.util.spec_from_file_location("arc_fork_serving", serving.FORK_SETUP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not PARENT.is_file(), reason="Pinned parent script review copy not present")
def test_fork_differs_from_reviewed_parent_only_in_utilization_and_mtp_reporting():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == flash_serving.SETUP_SHA256
    parent, fork = PARENT.read_text().splitlines(), serving.FORK_SETUP.read_text().splitlines()
    removed = {line[2:] for line in difflib.ndiff(parent, fork) if line.startswith("- ")}
    added = [line[2:] for line in difflib.ndiff(parent, fork) if line.startswith("+ ")]
    assert removed == PARENT_LINES
    assert 10 <= len(added) <= 25
    assert not any(line.startswith(("def ", "import ", "from ", "class ")) for line in added)
    assert any("GPU_MEMORY_UTILIZATION_ENV" in line for line in added)
    assert any(flash_serving.SETUP_SHA256 in line for line in added)
    assert '        "num_speculative_tokens": resolve_vllm_tuning()["mtp_speculative_tokens"],' in added


def test_fork_defaults_to_max_utilization_and_validates_override(tmp_path, monkeypatch):
    assert import_fork(tmp_path, monkeypatch).GPU_MEMORY_UTILIZATION == "0.96"
    assert serving.GPU_MEMORY_UTILIZATION == "0.96"
    assert import_fork(tmp_path, monkeypatch, "0.95").GPU_MEMORY_UTILIZATION == "0.95"
    for bad in ("1.0", "0.4", "98", "0.995"):
        with pytest.raises(RuntimeError, match="0.5 through 0.99"):
            import_fork(tmp_path, monkeypatch, bad)


@pytest.mark.parametrize('mtp_tokens', [1, 3])
def test_fork_command_uses_utilization_only_without_explicit_kv_bytes(tmp_path, monkeypatch, mtp_tokens):
    module = import_fork(tmp_path, monkeypatch, "0.97")
    monkeypatch.setenv("TAAF_VLLM_MTP_TOKENS", str(mtp_tokens))
    monkeypatch.setenv("TAAF_VLLM_KV_CACHE_MEMORY_BYTES", "0")
    monkeypatch.setenv("TAAF_VLLM_MAX_NUM_SEQS", "8")
    command = module.server_command(tmp_path / "model")
    assert json.loads(command[command.index('--speculative-config') + 1])['num_speculative_tokens'] == mtp_tokens
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.97"
    assert "--kv-cache-memory-bytes" not in command
    assert command[command.index("--max-num-seqs") + 1] == "8"
    monkeypatch.setenv("TAAF_VLLM_KV_CACHE_MEMORY_BYTES", "5368709120")
    explicit = module.server_command(tmp_path / "model")
    assert "--gpu-memory-utilization" not in explicit
    assert explicit[explicit.index("--kv-cache-memory-bytes") + 1] == "5368709120"


def fake_source(tmp_path):
    source = tmp_path / "datasets" / flash_serving.SOURCE_DATASET
    (source / "vllm-patches").mkdir(parents=True)
    (source / "vllm-patches/README.md").write_text("patch")
    (source / "serving_setup.py").write_text("parent")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/serving_setup.cpython-312.pyc").write_bytes(b"stale")
    (source / "serving_teardown.py").write_text("teardown")
    (source / "SOURCE_IDENTITY.json").write_text(
        json.dumps({"serving_setup_sha256": flash_serving.SETUP_SHA256, "schema": 1})
    )
    (tmp_path / "datasets" / flash_serving.RUNTIME_DATASET).mkdir(parents=True)
    return source


def test_stage_fork_bundle_rewrites_identity_and_keeps_regular_patch_dir(tmp_path):
    fake_source(tmp_path)
    staging = serving.stage_fork_bundle(tmp_path, tmp_path / "work/fork-bundle")
    fork_hash = hashlib.sha256(serving.FORK_SETUP.read_bytes()).hexdigest()
    assert hashlib.sha256((staging / "serving_setup.py").read_bytes()).hexdigest() == fork_hash
    identity = json.loads((staging / "SOURCE_IDENTITY.json").read_text())
    assert identity["serving_setup_sha256"] == fork_hash
    assert identity["arc_fork"]["parent_sha256"] == flash_serving.SETUP_SHA256
    assert identity["arc_fork"]["gpu_memory_utilization"] == "0.96"
    assert identity["schema"] == 1
    patches = staging / "vllm-patches"
    assert patches.is_dir() and not patches.is_symlink() and (patches / "README.md").is_file()
    assert (staging / "serving_teardown.py").read_text() == "teardown"
    assert not (staging / "__pycache__").exists()
    # Restaging replaces stale content instead of failing.
    serving.stage_fork_bundle(tmp_path, staging)
    (tmp_path / "datasets" / flash_serving.SOURCE_DATASET / "SOURCE_IDENTITY.json").write_text(
        json.dumps({"serving_setup_sha256": "other"})
    )
    with pytest.raises(RuntimeError, match="reviewed parent"):
        serving.stage_fork_bundle(tmp_path, tmp_path / "work/again")


def test_load_recipe_accepts_fork_bundle_and_removes_explicit_kv(tmp_path, monkeypatch):
    fake_source(tmp_path)
    bundle = tmp_path / "fork"
    bundle.mkdir()
    script = bundle / "serving_setup.py"
    script.write_text(
        "import os\nKV = os.environ['TAAF_VLLM_KV_CACHE_MEMORY_BYTES']\n"
        "BUNDLE = os.environ['TAAF_KAGGLE_BUNDLE_DIR']\n"
        "def source_identity():\n    return {'kv': KV, 'bundle': BUNDLE}\n"
    )
    for key in (
        *flash_serving.SERVER_ENV,
        "TAAF_KAGGLE_BUNDLE_DIR",
        "TAAF_KAGGLE_WORKING_DIR",
        "TAAF_KAGGLE_SETUP_ENV",
        "TAAF_KAGGLE_INPUT_PATHS",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="differs from reviewed source"):
        flash_serving.load_recipe(tmp_path / "w", tmp_path, bundle_dir=bundle)
    recipe = flash_serving.load_recipe(
        tmp_path / "w",
        tmp_path,
        bundle_dir=bundle,
        setup_sha256=digest,
        kv_cache_memory_bytes=0,
        prefix_caching=True,
        max_num_batched_tokens=4096,
    )
    assert recipe.source_identity() == {"kv": "0", "bundle": str(bundle)}
    assert os.environ["TAAF_VLLM_ENABLE_PREFIX_CACHING"] == "1"
    assert os.environ["TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"
    with pytest.raises(ValueError, match="prefix_caching"):
        flash_serving.load_recipe(
            tmp_path / "w", tmp_path, bundle_dir=bundle, setup_sha256=digest, prefix_caching=1
        )
    inputs = json.loads(os.environ["TAAF_KAGGLE_INPUT_PATHS"])
    assert inputs[flash_serving.SOURCE_DATASET].endswith(flash_serving.SOURCE_DATASET)
    with pytest.raises(ValueError, match="non-negative integer"):
        flash_serving.load_recipe(
            tmp_path / "w",
            tmp_path,
            bundle_dir=bundle,
            setup_sha256=digest,
            kv_cache_memory_bytes=-1,
        )
    # Duck comparisons keep the pinned dataset script and explicit 5 GiB default.
    (tmp_path / "datasets" / flash_serving.SOURCE_DATASET / "serving_setup.py").write_text(
        script.read_text()
    )
    monkeypatch.setattr(flash_serving, "SETUP_SHA256", digest)
    assert (
        flash_serving.load_recipe(tmp_path / "d", tmp_path).source_identity()["kv"] == "5368709120"
    )
    assert os.environ["TAAF_VLLM_ENABLE_PREFIX_CACHING"] == "0"
    assert os.environ["TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"] == "8192"


def test_kv_admission_summary_extracts_memory_lines(tmp_path):
    (tmp_path / "vllm-openai-server.log").write_text(
        "(Worker pid=1) INFO Model loading took 79.42 GiB memory and 115 seconds\n"
        "(Worker pid=1) INFO some unrelated line\n"
        "(Worker pid=1) INFO Available KV cache memory: 9.12 GiB\n"
        "(EngineCore pid=2) INFO GPU KV cache size: 320,000 tokens\n"
        "(EngineCore pid=2) INFO Maximum concurrency for 262,144 tokens per request: 1.22x\n"
    )
    assert serving.kv_admission_summary(tmp_path) == [
        "INFO Model loading took 79.42 GiB memory and 115 seconds",
        "INFO Available KV cache memory: 9.12 GiB",
        "INFO GPU KV cache size: 320,000 tokens",
        "INFO Maximum concurrency for 262,144 tokens per request: 1.22x",
    ]
    assert serving.kv_admission_summary(tmp_path / "missing") == []


async def test_preflight_uses_models_listing_and_checks_served_model(monkeypatch):
    calls = []

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Client:
        base_url = "http://127.0.0.1:1234/v1/"

        async def get(self, path):
            calls.append(path)
            return Response({"data": [{"id": served, "max_model_len": 262144}]})

    class Layer:
        def __init__(self, config, clock):
            self.client = Client()

        async def preflight(self):
            return 17

        async def close(self):
            calls.append("closed")

    monkeypatch.setattr(serving, "verify_hardware", lambda: "gpu")
    monkeypatch.setattr(serving, "RequestLayer", Layer)
    served = serving.Config().model
    result = await serving.preflight()
    assert calls == ["models", "closed"] and result["native_tokenizer_count"] == 17
    assert result["served_models"] == [served] and result["max_model_len"] == [262144]
    assert result["serving_setup"]["gpu_memory_utilization"] == "0.96"
    served = "other/model"
    with pytest.raises(RuntimeError, match="do not include"):
        await serving.preflight()


def test_launched_sequence_capacity_is_checked_not_just_profile(tmp_path):
    from qwen_baselines.serving import verify_sequence_capacity
    path = tmp_path / 'vllm-server-identity.json'
    path.write_text(json.dumps({'argv': ['serve', '--max-num-seqs', '26']}))
    with pytest.raises(RuntimeError, match='differs from requested 48'):
        verify_sequence_capacity(tmp_path, 48)
    path.write_text(json.dumps({'argv': ['serve', '--max-num-seqs', '48']}))
    assert verify_sequence_capacity(tmp_path, 48)['verified']
