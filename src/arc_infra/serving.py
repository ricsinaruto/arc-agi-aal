"""Load the reviewed offline serving recipe, independent of the Duck game agent."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

MODEL = "Qwen/Qwen3.8-Flash-Next-NVFP4"
REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
SOURCE_DATASET = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
SETUP_SHA256 = "037c041c9bd9dcffa9084b32f47af9cf1bf35849d5eaf2e1a3422daac098b2e2"  # the PARENT script in the source dataset; never the fork's (serving_fork hashes that live)
BASE_URL = "http://127.0.0.1:1234/v1"
SERVER_ENV = {
    "TAAF_KAGGLE_FAST_START": "1",
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "TAAF_VLLM_MAX_NUM_SEQS": "8",
    "TAAF_VLLM_MTP_TOKENS": "3",
    "TAAF_VLLM_OMP_THREADS": "1",
}


def dataset_mount(ref, input_root=Path("/kaggle/input")):
    root = Path(input_root)
    matches = [p for p in (root / "datasets" / ref, root / ref.split("/")[1]) if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one mounted dataset for {ref}; found {matches}")
    return matches[0]


def load_recipe(
    working,
    input_root=Path("/kaggle/input"),
    *,
    mtp_tokens=3,
    generation_slots=8,
    bundle_dir=None,
    setup_sha256=None,
    kv_cache_memory_bytes=None,
    prefix_caching=None,
    kv_cache_dtype="auto",
    max_num_batched_tokens=8192,
):
    """Import the pinned serving script, or a staged fork given by ``bundle_dir``.

    ``kv_cache_memory_bytes`` of 0 removes the explicit KV allocation, so the script
    passes its GPU-memory-utilization fraction and vLLM sizes the cache after profiling.
    """
    if type(mtp_tokens) is not int or mtp_tokens not in (0, 1, 2, 3, 4):
        raise ValueError("Flash serving comparisons support MTP depth 1-4, or 0 (no MTP draft)")
    # up to 384 (27 Oct, real-v5b): 344 tries, 352 seqs, graphs to 1,152
    if type(generation_slots) is not int or not (generation_slots in (8, 12, 16) or 25 <= generation_slots <= 384):
        raise ValueError("Flash server capacity comparisons support 8, 12, 16 or 25..384 slots")
    if kv_cache_dtype not in ("auto", "fp8"):
        raise ValueError("Cache dtype must be auto or fp8")
    if type(max_num_batched_tokens) is not int or max_num_batched_tokens not in (4096, 8192):
        raise ValueError("Prefill token batch must be 4096 or 8192")
    if setup_sha256 is None:
        setup_sha256 = SETUP_SHA256
    if kv_cache_memory_bytes is None:
        kv_cache_memory_bytes = int(SERVER_ENV["TAAF_VLLM_KV_CACHE_MEMORY_BYTES"])
    if prefix_caching is None:
        prefix_caching = SERVER_ENV["TAAF_VLLM_ENABLE_PREFIX_CACHING"] == "1"
    if type(prefix_caching) is not bool:
        raise ValueError("prefix_caching must be a bool")
    if type(kv_cache_memory_bytes) is not int or kv_cache_memory_bytes < 0:
        raise ValueError("Explicit KV cache bytes must be a non-negative integer")
    source = dataset_mount(SOURCE_DATASET, input_root)
    runtime = dataset_mount(RUNTIME_DATASET, input_root)
    bundle = source if bundle_dir is None else Path(bundle_dir)
    path = bundle / "serving_setup.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != setup_sha256:
        raise RuntimeError("Attached Flash-Next serving script differs from reviewed source")
    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    env_path = working / "serving-environment.json"
    if not env_path.exists():
        env_path.write_text("{}\n")
    os.environ.update(SERVER_ENV)
    os.environ["TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"] = str(max_num_batched_tokens)
    os.environ["TAAF_VLLM_KV_CACHE_DTYPE"] = kv_cache_dtype
    os.environ["TAAF_VLLM_MTP_TOKENS"] = str(mtp_tokens)
    os.environ["TAAF_VLLM_MAX_NUM_SEQS"] = str(generation_slots)
    os.environ["TAAF_VLLM_KV_CACHE_MEMORY_BYTES"] = str(kv_cache_memory_bytes)
    os.environ["TAAF_VLLM_ENABLE_PREFIX_CACHING"] = "1" if prefix_caching else "0"
    os.environ.update(
        TAAF_KAGGLE_BUNDLE_DIR=str(bundle),
        TAAF_KAGGLE_WORKING_DIR=str(working),
        TAAF_KAGGLE_SETUP_ENV=str(env_path),
        TAAF_KAGGLE_INPUT_PATHS=json.dumps(
            {SOURCE_DATASET: str(source), RUNTIME_DATASET: str(runtime)}
        ),
    )
    spec = importlib.util.spec_from_file_location("arc_pinned_flash_serving", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.source_identity()
    return module
