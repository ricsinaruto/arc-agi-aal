"""Prepare the pinned Qwen3.8-Flash-Next vLLM server (ARC fork of the Duck Kaggle setup script).

Parent: keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 serving_setup.py, sha256
037c041c9bd9dcffa9084b32f47af9cf1bf35849d5eaf2e1a3422daac098b2e2. arc_infra.serving.load_recipe imports this file from
the staged bundle (arc_infra.serving_fork) and calls main(): identity checks of the pinned runtime and model
(vLLM's load is the payload-integrity test), extraction, the PLE patch and the ARC boot patches (BOOT_PATCHES), then
the server.
GPU_MEMORY_UTILIZATION defaults to 0.96 (environment-overridable) so vLLM sizes the KV cache after profiling.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_DATASET = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
MODEL_HF_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_HF_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_MANIFEST_SHA256 = "a09bdad3fe3240c73332c0f99f4388a547205cb488c127f9b9059c9267dd9a5b"
MODEL_CONFIG_SHA256 = "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"
MODEL_QUANT_CONFIG_SHA256 = "7e69ef4b94302ae5b6f453b913621f698d5631a1d023d8b3e9e3b829721b98e8"
MODEL_FILE_COUNT = 419
MODEL_TOTAL_BYTES = 135_253_622_894
MODEL_KAGGLE_PATH = Path("/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1")

VLLM_IMAGE = "vllm/vllm-openai:qwen38-flash-next"
VLLM_INDEX_DIGEST = "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
VLLM_AMD64_MANIFEST_DIGEST = "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
VLLM_CONFIG_DIGEST = "sha256:bd995759b5b8ac51062e04c9e4d7c91c382d1ba377bb787e24dca2ccb39925e9"
VLLM_RUNTIME_MANIFEST_SHA256 = "e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad"
VLLM_VERSION = "0.1.dev20073+g8e685d198"
VLLM_PLE_STOCK_SHA256 = "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
VLLM_PLE_PATCHED_SHA256 = "a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed"
PLE_PATCH_DIRNAME = "vllm-patches"
PLE_PATCH_APPLICATOR = "apply_radixark_nvfp4_ple_fp8_patch.py"
PLE_PATCH_DIFF = "radixark_nvfp4_ple_fp8.patch"
PLE_PATCH_IDENTITY = "PATCH_IDENTITY.json"
PLE_PATCH_FILES = {  # the PLE patch bundle: name -> (bytes, sha256)
    PLE_PATCH_APPLICATOR: (8_230, "a9f88c019d04d7c8804ca6532e1c689d2ea97cf085cfec37dd4a779cb2c2c3fd"),
    PLE_PATCH_IDENTITY: (1_349, "f0bc8a948567e551d1bdf63f1163503ded897c3eba070d149d44c0c08fccbd35"),
    PLE_PATCH_DIFF: (3_940, "e3b83f9a65e436d21d3e757ddc8dd2699f3c59de9f73967107b3b20c101d0a9f"),
    "README.md": (1_581, "18e73020660bfe46b66c0ec1dfc3e702e848b51bc3d4773f58f6ae06ef126f9d"),
}
PLE_EXACT_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
PLE_EXCLUDED_PREFIX = "model.language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
PLE_TARGET = "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
CUTLASS_PTH_SHA256 = "cc710b99ddb3d76361a70461461d274212a979d00c381d821fce5bb3ad9cf246"
CUTLASS_PTH_TEXT = ("import sys, os, nvidia_cutlass_dsl; sys.path.insert(0, "
                    "os.path.join(nvidia_cutlass_dsl.__path__[0], 'dsl_packages'))")
TORCH_SHM_MANAGER_CMDLINE_SHA256 = "fd3a41c6faf7553a76d60949dea5632c423de414c9c0e9494171ea942543f428"
EXPECTED_RUNTIME_LAYERS = {  # index -> (sha256, bytes)
    8: ("f4f325a97c3e4739640b0e472a818bb299e7ce8e585bdbfe5586e9d3b5a2c475", 1_374_796_713),
    17: ("a5881cf32fc46f24318a6fc7015ccae863706075d4434d186b8b1b4e5bafb36e", 4_634_843_373),
    18: ("692ea0d28a87e9cea6db254c830f650f29804522045c9201f8ea63f30471b33b", 1_511_319_261),
    21: ("651e9c6638c77d15bf671d9264badcba80ea551f84109de486427b87d2ddbf75", 61_571_140),
    24: ("10dce885652a53a2e644a75d256b90830663330988b8599c3c0e56cd6c8a4675", 309_686_893),
    27: ("49a63ac9d116f92e5a8b006ae5665b940fcbc3d22d53aa7c01a3753ef341c53e", 14_479_724),
}
FAST_RUNTIME_WHITEOUTS = {
    8: ("usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88",),
    17: ("tmp/.wh.requirements-cuda.txt", "tmp/.wh.common.txt"),
}

SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"
HOST = "127.0.0.1"
PORT = 1234
BASE_URL = f"http://{HOST}:{PORT}/v1"
ANALYZER_CONTEXT = 32_768
MAX_NUM_SEQS = 28
GPU_MEMORY_UTILIZATION_ENV = "TAAF_VLLM_GPU_MEMORY_UTILIZATION"
GPU_MEMORY_UTILIZATION = os.environ.get(GPU_MEMORY_UTILIZATION_ENV, "0.96").strip()
if re.fullmatch(r"0\.\d{1,3}", GPU_MEMORY_UTILIZATION) is None or not 0.5 <= float(GPU_MEMORY_UTILIZATION) <= 0.99:
    raise RuntimeError(f"{GPU_MEMORY_UTILIZATION_ENV} must be a decimal from 0.5 through 0.99, "
                       f"found {GPU_MEMORY_UTILIZATION!r}.")
MAX_NUM_BATCHED_TOKENS = 8_192
SERVER_READY_TIMEOUT = 1_500  # arena.serving raises both (ARENA_SETUP_TIMEOUTS) before main()
SETUP_TOTAL_TIMEOUT = 1_800
MAX_NUM_SEQS_ENV = "TAAF_VLLM_MAX_NUM_SEQS"
MAX_NUM_BATCHED_TOKENS_ENV = "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"
OMP_THREADS_ENV = "TAAF_VLLM_OMP_THREADS"
MTP_TOKENS_ENV = "TAAF_VLLM_MTP_TOKENS"
KV_CACHE_MEMORY_BYTES_ENV = "TAAF_VLLM_KV_CACHE_MEMORY_BYTES"
KV_CACHE_DTYPE_ENV = "TAAF_VLLM_KV_CACHE_DTYPE"
ENABLE_PREFIX_CACHING_ENV = "TAAF_VLLM_ENABLE_PREFIX_CACHING"
MAX_CUDAGRAPH_CAPTURE_SIZE_ENV = "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"
# never read (vLLM's defaults); their tuning fields stay so the persisted document's schema is unchanged
MOE_BACKEND_ENV = "TAAF_VLLM_MOE_BACKEND"
MTP_INDEX_SHARE_FOR_ITERATION_ENV = "TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION"
MTP_DYNAMIC_BATCH_SCHEDULE_ENV = "TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE"
MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES = 192 * 1024**3
MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES = 64 * 1024**2
# the CUDA arch list follows the device (SM103 runs the SM100 build)
SUPPORTED_GPUS = {"rtx pro 6000": "12.0", "b200": "10.0", "b300": "10.0"}
VLLM_TUNING_FIELDS = {
    "schema_version", "enable_chunked_prefill", "enable_prefix_caching", "kv_cache_dtype", "kv_cache_memory_bytes",
    "max_cudagraph_capture_size", "max_num_batched_tokens", "max_num_batched_tokens_requested", "max_num_seqs",
    "moe_backend", "mtp_dynamic_batch_schedule", "mtp_index_share_for_iteration", "mtp_speculative_tokens",
    "omp_num_threads", "overridden"}
VLLM_TUNING_OVERRIDE_FIELDS = {
    MAX_NUM_SEQS_ENV, MAX_NUM_BATCHED_TOKENS_ENV, OMP_THREADS_ENV, MTP_TOKENS_ENV, KV_CACHE_MEMORY_BYTES_ENV,
    KV_CACHE_DTYPE_ENV, ENABLE_PREFIX_CACHING_ENV, MAX_CUDAGRAPH_CAPTURE_SIZE_ENV, MOE_BACKEND_ENV,
    MTP_INDEX_SHARE_FOR_ITERATION_ENV, MTP_DYNAMIC_BATCH_SCHEDULE_ENV}
# field -> (minimum, maximum); 384 seqs / 1,152 graphs: real-v5b (344 tries, 352 seqs x MTP 3)
TUNING_BOUNDS = {"kv_cache_memory_bytes": (0, MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES),
                 "max_cudagraph_capture_size": (0, 1152), "max_num_batched_tokens": (1, 1_048_576),
                 "max_num_batched_tokens_requested": (1, 1_048_576), "max_num_seqs": (1, 384),
                 "mtp_speculative_tokens": (0, 4), "omp_num_threads": (1, 64)}
MIN_HOST_AVAILABLE_BYTES = 64 * 1024**3
MIN_RUNTIME_FILESYSTEM_FREE_BYTES = 36_507_222_016

BUNDLE_DIR = Path(os.environ["TAAF_KAGGLE_BUNDLE_DIR"])
WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
SETUP_ENV_PATH = Path(os.environ["TAAF_KAGGLE_SETUP_ENV"])
RUNTIME_ROOT = Path("/tmp/qwen38-flash-next-vllm-runtime")
CACHE_ROOT = Path("/tmp/qwen38-flash-next-vllm-cache")
COMPILE_CACHE_ROOT = Path("/tmp/qwen38-flash-next-vllm-compile-cache")
TMP_ROOT = Path("/tmp/qwen38-flash-next-vllm-tmp")
SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
SERVER_IDENTITY = WORKING_DIR / "vllm-server-identity.json"
SETUP_PROVENANCE = WORKING_DIR / "vllm-setup-provenance.json"
PREFLIGHT_PATH = WORKING_DIR / "vllm-server-preflight.json"  # modal_rl1 waits for it (server up)
SITE_PACKAGES = "usr/local/lib/python3.12/dist-packages"
# ARC boot patches on the extracted tree, in this order, when their flag is "1":
# (flag, result key, arc_infra module, function, takes the working dir)
BOOT_PATCHES = (
    # determinism fixes: the edits read ARC_STABLE_ALIGN / ARC_STABLE_QSA / ARC_QSA_TIES at import in each server
    # process, so the shared tree is edited with both off and a second server can turn them on alone
    ("ARC_STABLE_PATCH", "stable_align", "stable_align", "apply", True),
    ("ARC_STABLE_PATCH", "stable_qsa", "stable_align", "apply_qsa", False),
    ("ARC_LORA_GRAPH_ONE", "lora_graph", "lora_graph_patch", "apply", True),  # LoRA cudagraph case only
    ("ARC_MARLIN_BATCH", "marlin_batch", "marlin_batch_patch", "apply", True),  # after expert_mask (extract wrapper)
    ("ARC_PINNED_LOAD", "pinned_load", "pinned_load_patch", "apply", True),  # weights read into pinned chunks
    ("ARC_LORA_NO_PIN", "lora_pin", "lora_pin_patch", "apply", True),  # adapter host copy not pinned per tensor
    ("ARC_DRAFT_LORA", "draft_lora", "draft_lora_patch", "apply", True),  # trained MTP draft LoRA merged at load
    ("ARC_MM_WARMUP_SKIP", "mm_warmup", "mm_warmup_patch", "apply", True),  # no start-up MM processor warmup
    ("ARC_TOKEN_SEGMENT_CACHE", "token_segment", "token_segment_patch", "apply", True),  # API prompt pieces cached
)


def cuda_arch():
    """TORCH_CUDA_ARCH_LIST for the one visible GPU (SUPPORTED_GPUS)."""
    global _CUDA_ARCH
    if _CUDA_ARCH is None:
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip().lower()
        arch = [a for k, a in SUPPORTED_GPUS.items() if k in name]
        if len(arch) != 1:
            raise RuntimeError(f"Expected one RTX PRO 6000 or B200, found: {name!r}")
        _CUDA_ARCH = arch[0]
    return _CUDA_ARCH


_CUDA_ARCH = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded_integer_environment(name: str, *, default: int, minimum: int, maximum: int) -> tuple[int, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    if re.fullmatch(r"\d+", raw.strip()) is None or not minimum <= int(raw.strip()) <= maximum:
        raise RuntimeError(f"{name} must be an integer from {minimum} through {maximum}, found {raw!r}.")
    return int(raw.strip()), True


def resolve_vllm_tuning() -> dict[str, Any]:
    """The serving surface the launcher sets (load_recipe / arena.serving), validated. Chunked prefill is always on; MoE
    backend, MTP index sharing and the MTP per-batch schedule stay at vLLM's defaults."""
    max_num_seqs, seqs_set = _bounded_integer_environment(MAX_NUM_SEQS_ENV, default=MAX_NUM_SEQS, minimum=1,
                                                          maximum=384)
    batch_tokens, batch_set = _bounded_integer_environment(MAX_NUM_BATCHED_TOKENS_ENV,
                                                           default=MAX_NUM_BATCHED_TOKENS, minimum=1, maximum=1_048_576)
    omp_threads, omp_set = _bounded_integer_environment(OMP_THREADS_ENV, default=4, minimum=1, maximum=64)
    mtp_tokens, mtp_set = _bounded_integer_environment(MTP_TOKENS_ENV, default=3, minimum=0, maximum=4)
    kv_bytes, kv_set = _bounded_integer_environment(KV_CACHE_MEMORY_BYTES_ENV, default=0, minimum=0,
                                                    maximum=MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES)
    if 0 < kv_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        raise RuntimeError(f"{KV_CACHE_MEMORY_BYTES_ENV} must be 0 or at least {MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES}, "
                           f"found {kv_bytes}.")
    dtype_raw = os.environ.get(KV_CACHE_DTYPE_ENV)
    kv_cache_dtype = "auto" if dtype_raw is None else dtype_raw.strip().lower()
    if kv_cache_dtype not in {"auto", "fp8"}:
        raise RuntimeError(f"{KV_CACHE_DTYPE_ENV} must be 'auto' or 'fp8', found {dtype_raw!r}.")
    prefix_raw = os.environ.get(ENABLE_PREFIX_CACHING_ENV)
    if prefix_raw is not None and prefix_raw.strip() not in {"0", "1"}:
        raise RuntimeError(f"{ENABLE_PREFIX_CACHING_ENV} must be 0 or 1, found {prefix_raw!r}.")
    graphs, graphs_set = _bounded_integer_environment(MAX_CUDAGRAPH_CAPTURE_SIZE_ENV, default=0, minimum=0,
                                                      maximum=1152)
    return {
        "schema_version": 1, "enable_chunked_prefill": True,
        "enable_prefix_caching": prefix_raw is not None and prefix_raw.strip() == "1",
        "kv_cache_dtype": kv_cache_dtype, "kv_cache_memory_bytes": kv_bytes, "max_cudagraph_capture_size": graphs,
        "max_num_batched_tokens": batch_tokens, "max_num_batched_tokens_requested": batch_tokens,
        "max_num_seqs": max_num_seqs, "moe_backend": None, "mtp_dynamic_batch_schedule": None,
        "mtp_index_share_for_iteration": False, "mtp_speculative_tokens": mtp_tokens, "omp_num_threads": omp_threads,
        "overridden": {
            MAX_NUM_SEQS_ENV: seqs_set, MAX_NUM_BATCHED_TOKENS_ENV: batch_set, OMP_THREADS_ENV: omp_set,
            MTP_TOKENS_ENV: mtp_set, KV_CACHE_MEMORY_BYTES_ENV: kv_set, KV_CACHE_DTYPE_ENV: dtype_raw is not None,
            ENABLE_PREFIX_CACHING_ENV: prefix_raw is not None, MAX_CUDAGRAPH_CAPTURE_SIZE_ENV: graphs_set,
            MOE_BACKEND_ENV: False, MTP_INDEX_SHARE_FOR_ITERATION_ENV: False, MTP_DYNAMIC_BATCH_SCHEDULE_ENV: False},
    }


def validate_vllm_tuning_document(value: Any) -> list[str]:
    """Errors in a persisted tuning document (the server identity's), without trusting it for process ownership."""
    if not isinstance(value, dict):
        return ["vllm_tuning_is_not_an_object"]
    missing, unknown = sorted(VLLM_TUNING_FIELDS - set(value)), sorted(set(value) - VLLM_TUNING_FIELDS)
    errors = ([f"vllm_tuning_missing_fields:{','.join(missing)}"] if missing else []) + (
        [f"vllm_tuning_unknown_fields:{','.join(unknown)}"] if unknown else [])
    if errors:
        return errors
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        errors.append("vllm_tuning_wrong_schema_version")
    for field in ("enable_chunked_prefill", "enable_prefix_caching", "mtp_index_share_for_iteration"):
        if type(value[field]) is not bool:
            errors.append(f"vllm_tuning_invalid_{field}")
    if not isinstance(value["kv_cache_dtype"], str) or value["kv_cache_dtype"] not in {"auto", "fp8"}:
        errors.append("vllm_tuning_invalid_kv_cache_dtype")
    if value["moe_backend"] not in {None, "flashinfer_b12x"}:
        errors.append("vllm_tuning_invalid_moe_backend")
    for field, (minimum, maximum) in TUNING_BOUNDS.items():
        if type(value[field]) is not int or not minimum <= value[field] <= maximum:
            errors.append(f"vllm_tuning_invalid_{field}")
    kv_bytes = value["kv_cache_memory_bytes"]
    if type(kv_bytes) is int and 0 < kv_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        errors.append("vllm_tuning_invalid_kv_cache_memory_bytes")
    if value["mtp_dynamic_batch_schedule"] is not None:
        errors.append("vllm_tuning_invalid_mtp_dynamic_batch_schedule")
    if value["enable_chunked_prefill"] is not True:
        errors.append("vllm_tuning_invalid_enable_chunked_prefill")
    if value["max_num_batched_tokens"] != value["max_num_batched_tokens_requested"]:
        errors.append("vllm_tuning_chunked_batch_tokens_mismatch")
    overridden = value["overridden"]
    if not isinstance(overridden, dict):
        return errors + ["vllm_tuning_overridden_is_not_an_object"]
    missing = sorted(VLLM_TUNING_OVERRIDE_FIELDS - set(overridden))
    unknown = sorted(set(overridden) - VLLM_TUNING_OVERRIDE_FIELDS)
    if missing:
        errors.append("vllm_tuning_overridden_missing_fields:" + ",".join(missing))
    if unknown:
        errors.append("vllm_tuning_overridden_unknown_fields:" + ",".join(unknown))
    if any(type(item) is not bool for item in overridden.values()):
        errors.append("vllm_tuning_overridden_values_are_not_boolean")
    return errors


def input_paths() -> dict[str, Path]:
    raw = os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.")
    return {str(key): Path(str(path)) for key, path in value.items()}


def source_identity() -> dict[str, Any]:
    path = BUNDLE_DIR / "SOURCE_IDENTITY.json"
    value = read_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid source identity: {path}")
    expected_setup, actual_setup = str(value.get("serving_setup_sha256", "")), sha256_file(Path(__file__).resolve())
    if not expected_setup or expected_setup != actual_setup:
        raise RuntimeError(f"Serving setup identity mismatch: {actual_setup} != {expected_setup or '<missing>'}")
    if value.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        raise RuntimeError("Source identity has the wrong model manifest hash.")
    if value.get("runtime_manifest_sha256") != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("Source identity has the wrong vLLM runtime manifest hash.")
    model, runtime = value.get("model") or {}, value.get("runtime") or {}
    if model.get("hf_repo") != MODEL_HF_REPO or model.get("hf_revision") != MODEL_HF_REVISION:
        raise RuntimeError("Source identity has the wrong model revision.")
    if runtime.get("image") != VLLM_IMAGE or runtime.get("kaggle_dataset") != RUNTIME_DATASET:
        raise RuntimeError("Source identity has the wrong vLLM runtime.")
    return value


def resolve_runtime_dir() -> Path:
    candidates = [input_paths().get(RUNTIME_DATASET), Path("/kaggle/input") / RUNTIME_DATASET.split("/", 1)[1],
                  Path("/kaggle/input/datasets") / RUNTIME_DATASET]
    for candidate in candidates:
        if candidate is not None and (candidate / "runtime-manifest.json").is_file():
            return candidate
    matches = []
    for path in Path("/kaggle/input").rglob("runtime-manifest.json"):
        try:
            manifest = read_json(path)
        except Exception:
            continue
        if manifest.get("index_digest") == VLLM_INDEX_DIGEST:
            matches.append(path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve one pinned {RUNTIME_DATASET}; matches={matches}")
    return matches[0]


def resolve_model_dir() -> Path:
    manifest_path = MODEL_KAGGLE_PATH / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected mounted {MODEL_HF_REPO}@{MODEL_HF_REVISION} at {MODEL_KAGGLE_PATH}.")
    value = read_json(manifest_path)
    if not isinstance(value, dict) or value.get("repo") != MODEL_HF_REPO or value.get("revision") != MODEL_HF_REVISION:
        raise RuntimeError(f"Mounted model identity is wrong at {MODEL_KAGGLE_PATH}.")
    return MODEL_KAGGLE_PATH


def verify_model(model_dir: Path) -> dict[str, Any]:
    """Identity only (manifest, sizes, configs); vLLM's load reads the payload."""
    actual_manifest_sha = sha256_file(model_dir / "MODEL_MANIFEST.json")
    if actual_manifest_sha != MODEL_MANIFEST_SHA256:
        raise RuntimeError(f"Model manifest hash mismatch: {actual_manifest_sha} != {MODEL_MANIFEST_SHA256}")
    manifest = read_json(model_dir / "MODEL_MANIFEST.json")
    if manifest.get("repo") != MODEL_HF_REPO or manifest.get("revision") != MODEL_HF_REVISION:
        raise RuntimeError("Mounted model provenance does not match the pinned checkpoint.")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != MODEL_FILE_COUNT:
        raise RuntimeError(f"Expected {MODEL_FILE_COUNT} model files, found {len(files or [])}.")
    try:
        total = sum(int(row["size"]) for row in files)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Model manifest file metadata is invalid.") from exc
    if total != MODEL_TOTAL_BYTES or total != int(manifest.get("total_bytes", -1)):
        raise RuntimeError(f"Model byte total mismatch: {total} != {MODEL_TOTAL_BYTES}")
    started = time.monotonic()
    print(f"MODEL_IDENTITY_ONLY files={len(files)} bytes={total} payload_check=vllm_load", flush=True)
    if sha256_file(model_dir / "config.json") != MODEL_CONFIG_SHA256:
        raise RuntimeError("Pinned model config hash does not match.")
    if sha256_file(model_dir / "hf_quant_config.json") != MODEL_QUANT_CONFIG_SHA256:
        raise RuntimeError("Pinned ModelOpt quantization config hash does not match.")
    config, quant = read_json(model_dir / "config.json"), read_json(model_dir / "hf_quant_config.json")
    text_config, quantization = config.get("text_config") or {}, config.get("quantization_config") or {}
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise RuntimeError(f"Wrong model architecture: {config.get('architectures')}")
    if config.get("model_type") != "qwen4_exp":
        raise RuntimeError(f"Wrong model type: {config.get('model_type')}")
    if (quantization.get("quant_method") != "modelopt" or quantization.get("quant_algo") != "NVFP4"
            or (quantization.get("producer") or {}).get("version") != "0.46.0"):
        raise RuntimeError(f"Wrong in-model quantization identity: {quantization}")
    quant_details = quant.get("quantization") or {}
    if quant_details.get("quant_algo") != "NVFP4" or int(quant_details.get("group_size", -1)) != 16:
        raise RuntimeError(f"Wrong ModelOpt NVFP4 details: {quant_details}")
    if (text_config.get("ple_embedding_dtype") != "float8_e4m3fn" or text_config.get("ple_layer_ids") != [2]
            or int(text_config.get("mtp_num_hidden_layers", -1)) != 1
            or int((text_config.get("mtp") or {}).get("num_hidden_layers", -1)) != 1):
        raise RuntimeError("The pinned FP8 PLE or native MTP identity does not match.")
    for name in ("model.safetensors.index.json", "chat_template.jinja"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"Incomplete checkpoint; missing {name}")
    return {"manifest_sha256": actual_manifest_sha, "file_count": len(files), "total_bytes": total,
            "architecture": config["architectures"][0], "quant_algo": quantization["quant_algo"],
            "ple_embedding_dtype": text_config["ple_embedding_dtype"],
            "mtp_num_hidden_layers": text_config["mtp_num_hidden_layers"], "verification_mode": "identity-only",
            "verify_seconds": time.monotonic() - started}


def _apply_whiteouts(expected_markers: tuple[str, ...]) -> list[str]:
    whiteouts = [RUNTIME_ROOT / relative for relative in expected_markers]
    missing = [str(path) for path in whiteouts if not os.path.lexists(path)]
    if missing:
        raise RuntimeError(f"Expected pinned runtime whiteouts are missing: {missing}")
    applied = []
    for marker in whiteouts:
        if marker.name == ".wh..wh..opq":
            raise RuntimeError(f"Unsupported opaque runtime whiteout: {marker}")
        target = marker.with_name(marker.name[4:])
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target)
        applied.append(str(marker.relative_to(RUNTIME_ROOT)))
        marker.unlink(missing_ok=True)
    return applied


def validate_runtime_storage() -> dict[str, Any]:
    """The runtime and cache roots share one filesystem with MIN_RUNTIME_FILESYSTEM_FREE_BYTES free."""
    rows, devices = [], set()
    for root in (RUNTIME_ROOT, CACHE_ROOT, COMPILE_CACHE_ROOT, TMP_ROOT):
        if root.is_symlink():
            raise RuntimeError(f"A runtime or cache root must not be a symlink: {root}")
        probe = root.parent
        if not probe.is_dir() or probe.is_symlink():
            raise RuntimeError(f"The runtime or cache parent is not a regular directory: {probe}")
        device = probe.stat().st_dev
        if root.exists() and root.stat().st_dev != device:
            raise RuntimeError(f"A runtime or cache root crossed its audited filesystem: {root}")
        filesystem = os.statvfs(probe)
        free_bytes = int(filesystem.f_bavail * filesystem.f_frsize)
        devices.add(int(device))
        rows.append({"root": str(root), "device": int(device), "free_bytes": free_bytes})
        if free_bytes < MIN_RUNTIME_FILESYSTEM_FREE_BYTES:
            raise RuntimeError(f"The filesystem for {root} has too little free space: "
                               f"{free_bytes} < {MIN_RUNTIME_FILESYSTEM_FREE_BYTES}.")
    if len(devices) != 1:
        raise RuntimeError(f"The pinned runtime and JIT/cache roots do not share one audited filesystem: {rows}")
    return {"roots": rows, "minimum_free_bytes": MIN_RUNTIME_FILESYSTEM_FREE_BYTES}


def verify_and_extract_runtime(runtime_dir: Path, *, full_layer_hashes: bool = False,
                               scan_extracted_caches: bool = False) -> dict[str, Any]:
    """Fast check (manifest identity and layer sizes; vLLM's load reads the payload) and extraction. The full-hash and
    cache-scan modes were retired; their keywords stay for arena.runtime_cache, which passes them (False)."""
    if full_layer_hashes or scan_extracted_caches:
        raise ValueError("full runtime hashing and the cache scan were retired")
    actual_sha = sha256_file(runtime_dir / "runtime-manifest.json")
    if actual_sha != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(f"Runtime manifest hash mismatch: {actual_sha} != {VLLM_RUNTIME_MANIFEST_SHA256}")
    manifest = read_json(runtime_dir / "runtime-manifest.json")
    expected_header = {"image": VLLM_IMAGE, "index_digest": VLLM_INDEX_DIGEST,
                       "amd64_manifest_digest": VLLM_AMD64_MANIFEST_DIGEST, "config_digest": VLLM_CONFIG_DIGEST,
                       "selected_compressed_bytes": 7_906_697_104}
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Wrong vLLM runtime {key}: {manifest.get(key)!r} != {expected!r}")
    layers = manifest.get("selected_layers")
    if not isinstance(layers, list) or len(layers) != len(EXPECTED_RUNTIME_LAYERS):
        raise RuntimeError("Runtime manifest has the wrong selected layer count.")
    layer_map = {int(row["index"]): row for row in layers}
    if set(layer_map) != set(EXPECTED_RUNTIME_LAYERS):
        raise RuntimeError(f"Wrong runtime layer indexes: {sorted(layer_map)}")
    shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
    RUNTIME_ROOT.mkdir(parents=True)
    started, verified, whiteouts = time.monotonic(), [], []
    for index in sorted(EXPECTED_RUNTIME_LAYERS):
        expected_hash, expected_size = EXPECTED_RUNTIME_LAYERS[index]
        row = layer_map[index]
        if (row.get("sha256") != expected_hash or row.get("digest") != f"sha256:{expected_hash}"
                or int(row.get("size", -1)) != expected_size):
            raise RuntimeError(f"Runtime layer {index} identity does not match: {row}")
        layer_path = runtime_dir / str(row["file"])
        if not layer_path.is_file():
            layer_path = runtime_dir / f"{row['file']}.blob"
        if not layer_path.is_file() or layer_path.stat().st_size != expected_size:
            raise RuntimeError(f"Missing or wrong-sized runtime layer: {layer_path}")
        print(f"RUNTIME_LAYER_FAST_CHECK index={index} bytes={expected_size} manifest_sha256={expected_hash}",
              flush=True)
        subprocess.run(["tar", "-xzf", str(layer_path), "-C", str(RUNTIME_ROOT), "--no-same-owner",
                        "--exclude=*/__pycache__", "--exclude=*/__pycache__/*", "--exclude=*.pyc", "--exclude=*.pyo"],
                       check=True)
        whiteouts.extend(_apply_whiteouts(FAST_RUNTIME_WHITEOUTS.get(index, ())))
        verified.append({"index": index, "file": str(row["file"]), "stored_file": layer_path.name,
                         "sha256": expected_hash, "size": expected_size})
    return {"manifest_sha256": actual_sha, "image": VLLM_IMAGE, "index_digest": VLLM_INDEX_DIGEST,
            "layers": verified, "whiteouts_applied": whiteouts, "verification_mode": "fast",
            "extract_seconds": time.monotonic() - started}


def patch_ple_layer() -> dict[str, Any]:
    """The pinned RadixArk FP8 PLE patch (BUNDLE_DIR/vllm-patches, every file size- and hash-checked) on the stock
    ple_layer.py; the written file's hash is checked."""
    site = RUNTIME_ROOT / SITE_PACKAGES
    path, patch_dir = site / PLE_TARGET, BUNDLE_DIR / PLE_PATCH_DIRNAME
    if not patch_dir.is_dir() or patch_dir.is_symlink():
        raise RuntimeError(f"Missing regular PLE patch directory: {patch_dir}")
    actual_names = {entry.name for entry in patch_dir.iterdir()}
    if actual_names != set(PLE_PATCH_FILES):
        raise RuntimeError(f"PLE patch bundle allowlist mismatch: actual={sorted(actual_names)} "
                           f"expected={sorted(PLE_PATCH_FILES)}")
    for name, (expected_size, expected_sha) in PLE_PATCH_FILES.items():
        artifact = patch_dir / name
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"PLE patch payload is not a regular file: {artifact}")
        actual_size, actual_sha = artifact.stat().st_size, sha256_file(artifact)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(f"PLE patch payload mismatch for {name}: "
                               f"size={actual_size}/{expected_size} sha256={actual_sha}/{expected_sha}")
    required_identity = {
        "artifact": PLE_PATCH_DIFF, "artifact_bytes": PLE_PATCH_FILES[PLE_PATCH_DIFF][0],
        "artifact_sha256": PLE_PATCH_FILES[PLE_PATCH_DIFF][1],
        "exact_model_config": "model/RadixArk-Qwen3.8-Flash-Next-NVFP4/config.json", "exact_model_config_bytes": 4_983,
        "exact_model_config_producer": {"name": "modelopt", "version": "0.46.0"},
        "exact_model_config_sha256": MODEL_CONFIG_SHA256,
        "modelopt_internal_gate": {"checkpoint_excluded_prefix": PLE_EXCLUDED_PREFIX, "group_size": 16,
                                   "is_checkpoint_nvfp4_serialized": True, "is_layer_excluded_checkpoint_prefix": True,
                                   "quant_method": "NVFP4", "runtime_prefix": PLE_EXACT_PREFIX},
        "patched_target_sha256": VLLM_PLE_PATCHED_SHA256, "stock_target_sha256": VLLM_PLE_STOCK_SHA256,
        "target": PLE_TARGET, "vllm_image_amd64_manifest": VLLM_AMD64_MANIFEST_DIGEST,
        "vllm_image_index_digest": VLLM_INDEX_DIGEST, "vllm_version": VLLM_VERSION}
    if read_json(patch_dir / PLE_PATCH_IDENTITY) != required_identity:
        raise RuntimeError("PATCH_IDENTITY.json did not match the exact approved patch identity.")
    before_sha = sha256_file(path)
    if before_sha != VLLM_PLE_STOCK_SHA256:
        raise RuntimeError(f"Stock vLLM PLE source hash mismatch: {before_sha}")
    completed = subprocess.run([sys.executable, "-B", str(patch_dir / PLE_PATCH_APPLICATOR), str(site)],
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True,
                               timeout=60)
    if completed.returncode != 0 or completed.stdout.strip() != "patched":
        raise RuntimeError(f"Pinned PLE patch applicator failed: returncode={completed.returncode} "
                           f"stdout={completed.stdout!r} stderr={completed.stderr!r}")
    after_sha = sha256_file(path)
    if after_sha != VLLM_PLE_PATCHED_SHA256:
        raise RuntimeError(f"Written vLLM PLE patch hash mismatch: {after_sha}")
    cache_artifacts = [child for root in (patch_dir, site / "vllm" / "models" / "qwen3_8_flash_next")
                       for child in root.rglob("*") if child.name == "__pycache__" or child.suffix in {".pyc", ".pyo"}]
    if cache_artifacts:
        raise RuntimeError(f"PLE patching produced forbidden Python cache files: {cache_artifacts}")
    return {"path": str(path), "stock_sha256": before_sha, "patched_sha256": after_sha}


def _find_real_cuda_driver() -> Path:
    candidates: list[Path] = []
    for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True).stdout.splitlines():
        if "libcuda.so.1" not in line or "=>" not in line:
            continue
        candidate = Path(line.split("=>", 1)[1].strip())
        if candidate.is_file() and "stubs" not in str(candidate).lower():
            candidates.append(candidate.resolve())
    search_dirs = [Path(entry) for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    for directory in search_dirs + [Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")]:
        candidate = directory / "libcuda.so.1"
        if candidate.is_file() and "stubs" not in str(candidate.resolve()).lower():
            candidates.append(candidate.resolve())
    if not candidates:
        raise RuntimeError("No real non-stub libcuda.so.1 was found.")
    return max(set(candidates), key=lambda path: path.stat().st_size)


def pyc_cache_env(env, environ=None):
    """ARC_PYC_CACHE=1: the server's processes write and read bytecode under CACHE_ROOT/pyc (the pinned runtime ships
    no .pyc: each server process compiled ~7.5K modules from source, 27-31 s of import vs 15 s with bytecode). The boot
    cache carries CACHE_ROOT. Bytecode is invalidated by source mtime/size, so patched files recompile."""
    if (os.environ if environ is None else environ).get("ARC_PYC_CACHE") != "1":
        return env
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env["PYTHONPYCACHEPREFIX"] = str(CACHE_ROOT / "pyc")
    return env


def check_cache_env(env, environ=None):
    """ARC_PLE_CHECK_SKIP=1: the PLE UVA load-time self-check (serving_kernels_runtime.cached_check) runs once per
    identity; its pass marker lives under CACHE_ROOT/arc-checks, which the boot cache carries to the next boot."""
    if (os.environ if environ is None else environ).get("ARC_PLE_CHECK_SKIP") != "1":
        return env
    env["ARC_SK_CHECK_CACHE"] = str(CACHE_ROOT / "arc-checks")
    return env


def aot_root_env(env, environ=None):
    """ARC_BOOT_CACHE_CARRY=1: vLLM's cache root (AOT graphs) per server flag fingerprint (boot_cache.env_fingerprint),
    so a carried or restored graph traced under other ARC_* flags never loads."""
    environ = os.environ if environ is None else environ
    if environ.get("ARC_BOOT_CACHE_CARRY") != "1":
        return env
    from arc_infra.boot_cache import vllm_cache_root

    env["VLLM_CACHE_ROOT"] = str(vllm_cache_root(CACHE_ROOT, environ))
    return env


def runtime_environment(*, tuning: dict[str, Any] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """The serving environment over the extracted runtime, and its fast checks."""
    resolved_tuning = tuning or resolve_vllm_tuning()
    allocator_input = {key: os.environ.get(key) for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")}
    expandable_true = re.compile(r"\bexpandable_segments\s*:\s*(?:true|1|yes)\b", re.IGNORECASE)
    bad_allocator = {k: v for k, v in allocator_input.items() if v is not None and expandable_true.search(v)}
    if bad_allocator:
        raise RuntimeError(f"Expandable CUDA allocator segments are forbidden for PLE CPU-offload IPC: {bad_allocator}")
    site = RUNTIME_ROOT / SITE_PACKAGES
    cutlass_pth, cutlass_nested = site / "nvidia_cutlass_dsl_packages.pth", site / "nvidia_cutlass_dsl" / "dsl_packages"
    cuda_home = RUNTIME_ROOT / "usr" / "local" / "cuda-13.0"
    cuda_lib, nvcc = cuda_home / "targets" / "x86_64-linux" / "lib", cuda_home / "bin" / "nvcc"
    ple_path = site / PLE_TARGET
    required = [site / "torch" / "__init__.py", site / "vllm" / "__init__.py", site / "flashinfer" / "__init__.py",
                ple_path, nvcc, cuda_lib / "libcudart.so", cutlass_pth, cutlass_nested / "cutlass" / "__init__.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete pinned vLLM runtime: {missing}")
    if sha256_file(ple_path) != VLLM_PLE_PATCHED_SHA256:
        raise RuntimeError("The patched PLE source hash changed before import.")
    if sha256_file(cutlass_pth) != CUTLASS_PTH_SHA256:
        raise RuntimeError("The pinned nvidia-cutlass-dsl .pth hash changed.")
    if cutlass_pth.read_text(encoding="utf-8").strip() != CUTLASS_PTH_TEXT:
        raise RuntimeError("The pinned nvidia-cutlass-dsl .pth instruction changed.")
    real_driver = _find_real_cuda_driver()
    library_dirs = [cuda_lib, cuda_home / "lib64", site / "torch" / "lib",
                    *sorted(path for path in (site / "nvidia").glob("*/lib") if path.is_dir()), real_driver.parent,
                    Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    for root in (CACHE_ROOT, COMPILE_CACHE_ROOT, TMP_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    cache_paths = {"HF_HOME": CACHE_ROOT / "huggingface", "XDG_CACHE_HOME": CACHE_ROOT,
                   "TORCH_HOME": CACHE_ROOT / "torch",
                   "TORCHINDUCTOR_CACHE_DIR": COMPILE_CACHE_ROOT / "torchinductor",
                   "TRITON_CACHE_DIR": COMPILE_CACHE_ROOT / "triton", "CUDA_CACHE_PATH": COMPILE_CACHE_ROOT / "cuda",
                   "FLASHINFER_WORKSPACE_BASE": COMPILE_CACHE_ROOT / "flashinfer"}
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["PYTHONPATH"] = os.pathsep.join([str(cutlass_nested), str(site)]
                                        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PATH"] = os.pathsep.join([str(cuda_home / "bin"), str(RUNTIME_ROOT / "usr" / "local" / "bin"),
                                   env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(path) for path in library_dirs if path.is_dir()]
                                             + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env.update({key: str(path) for key, path in cache_paths.items()})
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": str(cuda_home),
        "CUDACXX": str(nvcc), "TMPDIR": str(TMP_ROOT), "VLLM_PLE_CPU_OFFLOAD": "1",
        "VLLM_PLE_OFFLOAD_READY_TIMEOUT": str(SERVER_READY_TIMEOUT), "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
        "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1", "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TORCH_CUDA_ARCH_LIST": cuda_arch(),
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(resolved_tuning["omp_num_threads"]),
        "MKL_NUM_THREADS": str(resolved_tuning["omp_num_threads"])})
    pyc_cache_env(env)
    check_cache_env(env)
    aot_root_env(env)
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"The pinned vLLM runtime requires CPython 3.12, found {sys.implementation.name} "
                           f"{sys.version_info.major}.{sys.version_info.minor}.")
    glibc_raw = os.confstr("CS_GNU_LIBC_VERSION")
    glibc_match = re.fullmatch(r"glibc\s+(\d+)\.(\d+)", glibc_raw or "")
    if glibc_match is None or tuple(map(int, glibc_match.groups())) < (2, 28):
        raise RuntimeError(f"The vLLM runtime requires glibc >=2.28, found {glibc_raw!r}.")
    tools = {name: shutil.which(name, path=env["PATH"]) for name in ("gcc", "g++", "ld", "ninja", "nvcc")}
    missing_tools = [name for name, path in tools.items() if not path]
    if missing_tools:
        raise RuntimeError(f"Required vLLM tools are absent: {missing_tools}")
    return env, {"allocator_input": allocator_input, "torch_cuda_arch_list": cuda_arch(), "python": sys.version,
                 "glibc": glibc_raw, "real_cuda_driver": str(real_driver), "required_tools": tools}


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition(":")
        match = re.search(r"(\d+)", raw)
        if match:
            values[key] = int(match.group(1)) * 1024
    return values


def gpu_inventory() -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,"
                            "compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           check=True, timeout=30)
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or not any(k in rows[0].lower() for k in SUPPORTED_GPUS):
        raise RuntimeError(f"Expected one RTX PRO 6000 or B200, found: {rows}")
    available = int(_meminfo().get("MemAvailable", 0))
    if available < MIN_HOST_AVAILABLE_BYTES:
        raise RuntimeError(f"Host memory is too small for FP8 PLE CPU offload: {available} < "
                           f"{MIN_HOST_AVAILABLE_BYTES}")
    shm = os.statvfs("/dev/shm")
    probe = Path("/dev/shm") / f"duck-vllm-ipc-{os.getpid()}"
    try:
        probe.write_bytes(b"duck-vllm-ipc")
        if probe.read_bytes() != b"duck-vllm-ipc":
            raise RuntimeError("/dev/shm readback did not match.")
    finally:
        probe.unlink(missing_ok=True)
    disk = subprocess.run(["df", "-B1", str(WORKING_DIR), "/tmp", "/dev/shm"], capture_output=True, text=True,
                          check=True, timeout=30)
    return {"gpu_rows": rows, "host_mem_available_bytes": available,
            "shm_free_bytes": int(shm.f_bavail * shm.f_frsize), "disk": disk.stdout}


# ---- process ownership: the server runs in its own session; vllm-server-identity.json records its root and workers
WORKER_KEYS = ("pid", "start_ticks", "ppid", "pgid", "sid", "cmdline_sha256")
IDENTITY_KEYS = {"schema_version", "backend", "boot_id", "host", "port", "pid", "start_ticks", "pgid", "sid", "argv",
                 "argv_sha256", "vllm_tuning", "started_epoch", "phase", "snapshot_epoch", "ready_epoch", "workers"}


def argv_sha256(argv) -> str:
    return hashlib.sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def proc_record(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2:].split()
        cmdline_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [part.decode("utf-8", errors="replace") for part in cmdline_bytes.split(b"\x00") if part]
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]), "sid": int(fields[3]),
                "start_ticks": int(fields[19]),
                "cmdline": cmdline_bytes.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip(),
                "cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(), "argv_sha256": argv_sha256(argv)}
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def process_table() -> dict[int, dict[str, Any]]:
    records = {int(item.name): proc_record(int(item.name)) for item in Path("/proc").iterdir() if item.name.isdigit()}
    return {pid: record for pid, record in records.items() if record is not None}


def descendant_ids(root_pid: int, records: dict[int, dict[str, Any]]) -> set[int]:
    found, changed = {root_pid}, True
    while changed:
        changed = False
        for pid, record in records.items():
            if record["ppid"] in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def allowed_detached_worker(record, workers_by_pid, root_pgid: int, root_sid: int) -> bool:
    """torch's shared-memory manager: its own session, parented by a process of the server's session."""
    pid = int(record.get("pid", -1))
    parent = workers_by_pid.get(int(record.get("ppid", -1)))
    return (pid > 1 and parent is not None and int(parent.get("pgid", -1)) == root_pgid
            and int(parent.get("sid", -1)) == root_sid and int(record.get("pgid", -1)) == pid
            and int(record.get("sid", -1)) == pid and record.get("cmdline_sha256") == TORCH_SHM_MANAGER_CMDLINE_SHA256)


def same_process(live, record) -> bool:
    return all(int(live[k]) == int(record.get(k, -1)) for k in ("start_ticks", "pgid", "sid")) and (
        live["cmdline_sha256"] == record.get("cmdline_sha256"))


def worker_row(record) -> dict[str, Any]:
    return {k: (str if k == "cmdline_sha256" else int)(record[k]) for k in WORKER_KEYS}


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def tail_log(lines: int = 200) -> str:
    if not SERVER_LOG.exists():
        return ""
    return "\n".join(SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def alive(pid: int, ticks: int | None = None) -> bool:
    record = proc_record(pid)
    return record is not None and record["state"] != "Z" and (ticks is None or int(record["start_ticks"]) == ticks)


def identity_problems(identity, numbers, current_root, table) -> list[str]:
    """Why a recorded server identity does not describe a server this setup owns (empty: it does). ``numbers``: its
    (pid, start_ticks, pgid, sid), or None when they are not integers."""
    reasons = [] if set(identity) == IDENTITY_KEYS else ["identity key set mismatch"]
    reasons += validate_vllm_tuning_document(identity.get("vllm_tuning"))
    if numbers is None:
        reasons.append("invalid root numeric identity")
    pid, ticks, pgid, sid = numbers or (-1, -1, -1, -1)
    argv = identity.get("argv")
    expected = argv_sha256(argv) if isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv) else None
    if expected is None or identity.get("argv_sha256") != expected:
        reasons.append("canonical argv hash mismatch")
    if (identity.get("schema_version") != 1 or identity.get("backend") != "vllm" or identity.get("boot_id") != boot_id()
            or identity.get("host") != HOST or identity.get("port") != PORT or pid <= 1 or pgid != pid or sid != pid):
        reasons.append("root ownership header mismatch")
    phase, workers = identity.get("phase"), identity.get("workers")
    started, snapshot = identity.get("started_epoch"), identity.get("snapshot_epoch")
    ready = identity.get("ready_epoch")
    if (phase not in {"starting", "ready"} or not isinstance(started, (int, float))
            or not isinstance(snapshot, (int, float)) or snapshot < started or not isinstance(workers, list)
            or not workers):
        reasons.append("invalid phase or snapshot semantics")
    elif phase == "starting" and (ready is not None or len(workers) != 1):
        reasons.append("invalid starting-phase snapshot")
    elif phase == "ready" and (not isinstance(ready, (int, float)) or ready < started or ready > snapshot):
        reasons.append("invalid ready-phase snapshot")
    root_numbers = (("start_ticks", ticks), ("pgid", pgid), ("sid", sid))
    if isinstance(workers, list):
        by_pid = {int(w["pid"]): w for w in workers if isinstance(w, dict) and type(w.get("pid")) is int}
        for worker in workers:
            if not isinstance(worker, dict) or set(worker) != set(WORKER_KEYS):
                reasons.append("worker key set mismatch")
                break
            if int(worker.get("sid", -1)) != sid and not allowed_detached_worker(worker, by_pid, pgid, sid):
                reasons.append(f"unrecognized detached worker for pid {worker.get('pid')}")
                break
        roots = [w for w in workers if isinstance(w, dict) and w.get("pid") == pid]
        if len(roots) != 1 or any(int(roots[0].get(k, -1)) != v for k, v in root_numbers):
            reasons.append("root worker snapshot mismatch")
    if current_root is not None and (any(int(current_root[k]) != v for k, v in root_numbers)
                                     or current_root["argv_sha256"] != identity.get("argv_sha256")):
        reasons.append("live root fingerprint mismatch")
    if current_root is not None and isinstance(workers, list):
        root_worker = next((w for w in workers if isinstance(w, dict) and w.get("pid") == pid), None)
        if root_worker is None or current_root["cmdline_sha256"] != root_worker.get("cmdline_sha256"):
            reasons.append("live root cmdline fingerprint mismatch")
    group = [record for record in table.values() if int(record["pgid"]) == pgid]
    if current_root is not None and (not group or any(int(record["sid"]) != sid for record in group)):
        reasons.append("live process group is not wholly inside the owned session")
    for worker in workers if isinstance(workers, list) else ():
        live = table.get(int(worker.get("pid", -1))) if isinstance(worker, dict) else None
        if live is not None and not same_process(live, worker):
            reasons.append(f"live worker fingerprint mismatch for pid {worker.get('pid')}")
            break
    return reasons


def stop_owned_server() -> dict[str, Any]:
    """Stop the server vllm-server-identity.json names, only when the identity proves this setup owns it: SIGTERM to
    its process group and detached workers, SIGKILL after 120 s. ``stopped``: nothing owned survives and the port
    is free."""
    result: dict[str, Any] = {"identity_exists": SERVER_IDENTITY.exists(), "stopped": True}
    if not SERVER_IDENTITY.exists():
        return result
    identity = read_json(SERVER_IDENTITY)
    if not isinstance(identity, dict):
        result.update({"identity_matches": False, "identity_rejection_reasons": ["identity is not a JSON object"],
                       "survivors": None, "port_open": port_open(), "stopped": False})
        return result
    try:
        numbers = tuple(int(identity[k]) for k in ("pid", "start_ticks", "pgid", "sid"))
    except (KeyError, TypeError, ValueError):
        numbers = None
    pid, ticks, pgid, sid = numbers or (-1, -1, -1, -1)
    workers = identity.get("workers")
    current_root = proc_record(pid) if pid > 1 else None
    reasons = identity_problems(identity, numbers, current_root, process_table())
    matches = current_root is not None and not reasons
    result.update({"pid": pid, "start_ticks": ticks, "pgid": pgid, "sid": sid, "identity_matches": matches,
                   "identity_rejection_reasons": reasons})

    def live_owned_records() -> list[dict[str, Any]]:
        if sid <= 1:
            return []
        table = process_table()
        owned = {int(r["pid"]): r for r in table.values() if int(r["sid"]) == sid and r["state"] != "Z"}
        if isinstance(workers, list):
            by_pid = {int(w["pid"]): w for w in workers if isinstance(w, dict) and type(w.get("pid")) is int}
            for worker in workers:
                if (not isinstance(worker, dict) or int(worker.get("sid", -1)) == sid
                        or not allowed_detached_worker(worker, by_pid, pgid, sid)):
                    continue
                live = table.get(int(worker["pid"]))
                if live is not None and same_process(live, worker):
                    owned[int(live["pid"])] = live
        return list(owned.values())

    def signal_owned(sig, records) -> tuple[bool, list[int]]:
        """``sig`` to the process group (True if it was delivered) and to each record outside it (pids signalled);
        SIGKILL only to a record still the same process."""
        try:
            os.killpg(pgid, sig)
            group = True
        except ProcessLookupError:
            group = False
        sent = []
        for record in records:
            if int(record["pgid"]) == pgid:
                continue
            current = proc_record(int(record["pid"]))
            if sig == signal.SIGKILL and (current is None or not same_process(current, record)):
                continue
            try:
                os.kill(int(record["pid"]), sig)
                sent.append(int(record["pid"]))
            except ProcessLookupError:
                pass
        return group, sent

    if matches:
        group, sent = signal_owned(signal.SIGTERM, live_owned_records())
        if group:
            result["sigterm_sent"] = True
        if sent:
            result["separate_sigterm_pids"] = sent
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and live_owned_records():
            time.sleep(1)
        survivors = live_owned_records()
        if survivors:
            _, sent = signal_owned(signal.SIGKILL, survivors)
            result["sigkill_sent"] = True
            if sent:
                result["separate_sigkill_pids"] = sent
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and live_owned_records():
                time.sleep(0.5)
    survivors = live_owned_records()
    result["survivors"] = [worker_row(r) for r in sorted(survivors, key=lambda item: int(item["pid"]))]
    result["port_open"] = port_open()
    result["stopped"] = not survivors and not result["port_open"]
    return result


def port_open(timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout):
            return True
    except OSError:
        return False


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                              "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body[:4000]}") from exc
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object from {url}, found {type(value).__name__}.")
    return value


def server_command(model_dir: Path, *, tuning: dict[str, Any] | None = None) -> list[str]:
    t = tuning or resolve_vllm_tuning()
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir), "--served-model-name",
               SERVED_MODEL_NAME, "--host", HOST, "--port", str(PORT), "--load-format", "safetensors", "--dtype",
               "bfloat16", "--quantization", "modelopt_fp4", "--tensor-parallel-size", "1",
               "--distributed-executor-backend", "mp"]
    if int(t["kv_cache_memory_bytes"]) > 0:
        command += ["--kv-cache-memory-bytes", str(t["kv_cache_memory_bytes"])]
    else:
        command += ["--gpu-memory-utilization", GPU_MEMORY_UTILIZATION]
    command += ["--max-model-len", str(ANALYZER_CONTEXT), "--max-num-seqs", str(t["max_num_seqs"]),
                "--max-num-batched-tokens", str(t["max_num_batched_tokens"]), "--async-scheduling",
                "--enable-chunked-prefill"]
    if t["kv_cache_dtype"] != "auto":
        command += ["--kv-cache-dtype", str(t["kv_cache_dtype"])]
    if int(t["max_cudagraph_capture_size"]) > 0:
        command += ["--max-cudagraph-capture-size", str(t["max_cudagraph_capture_size"])]
    command += ["--enable-prefix-caching" if t["enable_prefix_caching"] else "--no-enable-prefix-caching",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
                "--chat-template", str(model_dir / "chat_template.jinja")]
    if int(t["mtp_speculative_tokens"]) > 0:
        command += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens":
                                                        t["mtp_speculative_tokens"]}, separators=(",", ":"))]
    return command + ["--no-enable-log-requests", "--disable-uvicorn-access-log", "--uvicorn-log-level", "info"]


def start_server(model_dir: Path, env: dict[str, str], *, tuning: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start vLLM in its own session and record its identity (phase "starting")."""
    cleanup = stop_owned_server()
    if not cleanup.get("stopped", False):
        raise RuntimeError(f"A prior owned vLLM server did not stop: {cleanup}")
    if port_open():
        raise RuntimeError(f"Port {HOST}:{PORT} is already in use by an unowned process.")
    SERVER_LOG.unlink(missing_ok=True)
    resolved_tuning = tuning or resolve_vllm_tuning()
    command = server_command(model_dir, tuning=resolved_tuning)
    print("VLLM_START_COMMAND", json.dumps(command), flush=True)
    from arc_infra.heap_census import install_for

    install_for(command, env)  # HEAP_CENSUS_DIR in its env: the vLLM processes' heap census (real-v7 growth)
    with SERVER_LOG.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
    argv_hash = argv_sha256(command)
    deadline, root_record = time.monotonic() + 10, None
    while time.monotonic() < deadline:  # setsid runs before exec: wait for the exec'd argv
        root_record = proc_record(process.pid)
        if (root_record is not None and root_record["pgid"] == process.pid and root_record["sid"] == process.pid
                and root_record["argv_sha256"] == argv_hash):
            break
        time.sleep(0.05)
    if root_record is None or root_record["pgid"] != process.pid or root_record["sid"] != process.pid:
        process.terminate()
        raise RuntimeError(f"vLLM root did not enter its isolated process session: {root_record}")
    if root_record["argv_sha256"] != argv_hash:
        process.terminate()
        raise RuntimeError("The live vLLM root argv did not match the exact launched command: "
                           f"{root_record['argv_sha256']} != {argv_hash}; "
                           f"live cmdline {root_record['cmdline'][:400]!r}")
    started_epoch = time.time()
    identity = {"schema_version": 1, "backend": "vllm", "boot_id": boot_id(), "host": HOST, "port": PORT,
                "pid": process.pid, "start_ticks": int(root_record["start_ticks"]), "pgid": int(root_record["pgid"]),
                "sid": int(root_record["sid"]), "argv": command, "argv_sha256": argv_hash,
                "vllm_tuning": resolved_tuning,
                "started_epoch": started_epoch, "phase": "starting", "snapshot_epoch": started_epoch,
                "ready_epoch": None, "workers": [worker_row(root_record)]}
    write_json(SERVER_IDENTITY, identity)
    return identity


def update_ready_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """The identity after readiness (phase "ready"): every process of the server's session as its workers."""
    if identity.get("boot_id") != boot_id():
        raise RuntimeError("vLLM identity boot ID changed before worker snapshot.")
    root = proc_record(int(identity["pid"]))
    if root is None or any(int(root[k]) != int(identity[k]) for k in ("start_ticks", "pgid", "sid")):
        raise RuntimeError(f"vLLM root identity changed before worker snapshot: {root}")
    table = process_table()
    pgid, sid = int(identity["pgid"]), int(identity["sid"])
    outside = [table[pid] for pid in sorted(descendant_ids(int(identity["pid"]), table))
               if pid in table and int(table[pid]["sid"]) != sid]
    escaped = [r for r in outside if not allowed_detached_worker(r, table, pgid, sid)]
    if escaped:
        raise RuntimeError(f"Owned vLLM descendants escaped the isolated session: {escaped}")
    owned = {int(r["pid"]): r for r in table.values() if int(r["sid"]) == sid}
    owned.update({int(r["pid"]): r for r in outside})
    if int(identity["pid"]) not in owned:
        raise RuntimeError("The vLLM root was absent from its post-readiness session snapshot.")
    now = time.time()
    updated = {**identity, "phase": "ready", "snapshot_epoch": now, "ready_epoch": identity.get("ready_epoch") or now,
               "workers": [worker_row(owned[pid]) for pid in sorted(owned)]}
    write_json(SERVER_IDENTITY, updated)
    return updated


def wait_for_server(identity: dict[str, Any], setup_deadline: float) -> dict[str, Any]:
    deadline = min(time.monotonic() + SERVER_READY_TIMEOUT, setup_deadline)
    started, last_notice = time.monotonic(), -1
    while time.monotonic() < deadline:
        if not alive(int(identity["pid"]), int(identity["start_ticks"])):
            raise RuntimeError(f"vLLM exited before readiness.\n{tail_log(300)}")
        try:
            remaining = setup_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"The {SETUP_TOTAL_TIMEOUT}s setup/preflight ceiling expired.")
            models = request_json(f"{BASE_URL}/models", timeout=max(1, min(10, math.ceil(remaining))))
            ids = [row.get("id") for row in models.get("data") or [] if isinstance(row, dict)]
            if ids != [SERVED_MODEL_NAME]:
                raise RuntimeError(f"vLLM served the wrong model identity: {ids}")
            return {"models": models, "ready_seconds": time.monotonic() - started}
        except Exception:
            elapsed = int(time.monotonic() - started)
            if elapsed // 60 != last_notice:
                last_notice = elapsed // 60
                print(f"VLLM_WAIT elapsed_s={elapsed}\n{tail_log(30)}", flush=True)
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for vLLM after {SERVER_READY_TIMEOUT}s.\n{tail_log(300)}")


def persist_analyzer_environment(env: dict[str, str]) -> dict[str, str]:
    """The Duck analyzer's environment, merged into TAAF_KAGGLE_SETUP_ENV (load_recipe's serving-environment.json)."""
    setup_env = read_json(SETUP_ENV_PATH) if SETUP_ENV_PATH.exists() else {}
    if not isinstance(setup_env, dict):
        raise RuntimeError("TAAF_KAGGLE_SETUP_ENV must contain a JSON object.")
    persisted = {
        **{key: env[key] for key in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDACXX")},
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_NO_USAGE_STATS": "1", "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False", "TORCH_CUDA_ARCH_LIST": cuda_arch(),
        "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1", "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
        "LOCAL_ANALYZER_BASE_URL": BASE_URL, "OPENAI_BASE_URL": BASE_URL, "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm", "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
        "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME, "OPENAI_API_KEY": "offline-kaggle-local-server",
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness", "LOCAL_ANALYZER_CONTEXT_WINDOW": str(ANALYZER_CONTEXT),
        "LOCAL_ANALYZER_MAX_OUTPUT": "0", "LOCAL_ANALYZER_TOOL_STEPS": "0", "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024", "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6", "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true", "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "4"}
    write_json(SETUP_ENV_PATH, {**setup_env, **persisted})
    return persisted


def patch_check_exit(checked, runtime_check, working_dir=None, emit=print):
    """ARC_SERVING_PATCH_CHECK=1: the boot patches ran on this exact runtime tree; a skipped one (None) is fatal.
    Prints ARC_PATCH_CHECK OK|FAIL and exits 0 / 3 before any GPU work."""
    folder = Path(working_dir or WORKING_DIR)
    row = {"patches": {k: v is not None and v is not False for k, v in checked.items()},
           "markers": sorted(p.name for p in folder.glob("*.json") if "patch" in p.name or "skip" in p.name),
           "runtime": {k: runtime_check.get(k) for k in ("mode", "cache_key") if isinstance(runtime_check, dict)}}
    ok = all(row["patches"].values())
    emit(f"ARC_PATCH_CHECK {'OK' if ok else 'FAIL'} " + json.dumps(row, default=str), flush=True)
    raise SystemExit(0 if ok else 3)


def main() -> None:
    setup_started, setup_started_epoch = time.monotonic(), time.time()
    setup_deadline = setup_started + SETUP_TOTAL_TIMEOUT
    tuning = resolve_vllm_tuning()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    print("VLLM_SETUP_MODE fast", flush=True)
    source_identity()
    patch_check = os.environ.get("ARC_SERVING_PATCH_CHECK") == "1"  # CPU gate: every boot patch, then exit
    try:
        inventory = gpu_inventory()
    except Exception as error:  # noqa: BLE001 - no GPU in the gate's container
        if not patch_check:
            raise
        inventory = {"skipped": repr(error)[:120]}
    runtime_dir, model_dir = resolve_runtime_dir(), resolve_model_dir()
    print(f"PINNED_VLLM_RUNTIME_PATH {runtime_dir}", flush=True)
    print(f"PINNED_MODEL_PATH {model_dir}", flush=True)
    storage_check = validate_runtime_storage()
    runtime_check = verify_and_extract_runtime(runtime_dir, full_layer_hashes=False, scan_extracted_caches=False)
    ple_patch = patch_ple_layer()
    checked = {"ple": bool(ple_patch)}
    site = RUNTIME_ROOT / SITE_PACKAGES
    for flag, key, module, function, takes_working in BOOT_PATCHES:
        if os.environ.get(flag) == "1":
            apply = getattr(importlib.import_module(f"arc_infra.{module}"), function)
            checked[key] = apply(site, WORKING_DIR) if takes_working else apply(site)
    if patch_check:
        patch_check_exit(checked, runtime_check)
    env, environment_check = runtime_environment(tuning=tuning)
    model_check = verify_model(model_dir)
    from arc_infra.boot_cache import key as boot_cache_key
    from arc_infra.boot_cache import restore_from_environment, save_from_environment

    cache_key = None
    if os.environ.get("ARC_BOOT_CACHE_IN") or os.environ.get("ARC_BOOT_CACHE_SAVE") == "1":
        cache_key = boot_cache_key(server_command(model_dir, tuning=tuning), WORKING_DIR)
    # ARC_BOOT_CACHE_IN: the compile caches of an earlier boot with the same argv and overlays
    restore_from_environment((CACHE_ROOT, COMPILE_CACHE_ROOT), cache_key)
    server_identity = start_server(model_dir, env, tuning=tuning)
    readiness = wait_for_server(server_identity, setup_deadline)
    save_from_environment((CACHE_ROOT, COMPILE_CACHE_ROOT), WORKING_DIR, cache_key)  # ARC_BOOT_CACHE_SAVE=1
    server_identity = update_ready_identity(server_identity)
    write_json(PREFLIGHT_PATH, {"mode": "fast", "served_model": SERVED_MODEL_NAME,
                                "models_endpoint": readiness["models"]})
    persisted = persist_analyzer_environment(env)
    write_json(SETUP_PROVENANCE, {
        "source_dataset": SOURCE_DATASET, "source_identity_sha256": sha256_file(BUNDLE_DIR / "SOURCE_IDENTITY.json"),
        "runtime_dataset": RUNTIME_DATASET, "model_hf_repo": MODEL_HF_REPO, "model_hf_revision": MODEL_HF_REVISION,
        "vllm_image": VLLM_IMAGE, "vllm_version": VLLM_VERSION, "vllm_tuning": tuning,
        "setup_started_epoch": setup_started_epoch, "setup_total_timeout_seconds": SETUP_TOTAL_TIMEOUT,
        "server_ready_timeout_seconds": SERVER_READY_TIMEOUT, "inventory": inventory, "runtime_storage": storage_check,
        "runtime": runtime_check, "ple_patch": ple_patch, "model": model_check, "environment": environment_check,
        "server_identity": server_identity, "readiness": readiness, "persisted_analyzer_environment": persisted,
        "server_log_tail": tail_log(300), "setup_elapsed_seconds": time.monotonic() - setup_started,
        "completed_epoch": time.time()})
    summary = {"mode": "fast", "ready_seconds": readiness["ready_seconds"], "pid": server_identity["pid"]}
    print("VLLM_SETUP_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
