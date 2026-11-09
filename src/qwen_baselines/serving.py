"""Explicit native-context profile; historical 32K profiles remain untouched."""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from arc_infra.serving import load_recipe
from arc_infra.serving_fork import (  # noqa: F401 - re-exported for the baselines and tests
    FORK_SETUP,
    GPU_MEMORY_UTILIZATION,
    GPU_MEMORY_UTILIZATION_ENV,
    fork_identity,
    stage_fork_bundle,
)

from . import segmentation
from .clock import Clock
from .config import Config
from .jail import Jail, needed
from .model import RequestLayer

PROFILE = {
    "name": "qwen-minimal-native",
    "max_model_len": 262144,
    "kv_cache_memory_bytes": 0,
    "kv_cache_allocation": (
        "vLLM-profiled remainder at gpu_memory_utilization "
        f"{GPU_MEMORY_UTILIZATION}; no explicit byte count"
    ),
    "kv_cache_dtype": "bfloat16",
    "max_num_seqs": 8,
    "speculative_tokens": 3,
    # User decision 2026-09-15: reuse identical conversation prefixes across turns.
    # Without it every request re-prefilled the full history (A: ~2.3M prompt tokens
    # per game in 600 s). The Duck controls keep it off.
    "prefix_caching": True,
    "gpu": "RTX PRO 6000 Blackwell",
    "gpu_memory_gib": 96,
    "serving_setup": "forked parent script; see fork_identity()",
}
KV_LOG_PATTERNS = (
    "prefix caching",
    "Prefix cach",
    "Initial free memory",
    "Available KV cache memory",
    "GPU KV cache size",
    "Maximum concurrency",
    "Model loading took",
    "KV cache is needed",
)


def kv_admission_summary(working):
    """Extract vLLM's memory/KV admission lines from the owned server log, if present."""
    lines = []
    for log in sorted(Path(working).glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if any(pattern in line for pattern in KV_LOG_PATTERNS):
                lines.append(re.sub(r"^\([^)]*\)\s*", "", line).strip())
    return lines[-12:]


def verify_hardware():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    cards = [line for line in output.splitlines() if line.strip()]
    if (
        len(cards) != 1
        or "RTX PRO 6000" not in cards[0]
        or "Blackwell" not in cards[0]
        or int(cards[0].rsplit(",", 1)[1]) < 95000
    ):
        raise RuntimeError(f"Require exactly one RTX PRO 6000 Blackwell 96 GB; found {cards}")
    return cards[0]


async def preflight(config=Config()):
    hardware = verify_hardware()
    layer = RequestLayer(config, Clock(config.gameplay_seconds))
    try:
        async with asyncio.timeout(60):
            count = await layer.preflight()
            # /server_info is a vLLM dev-mode endpoint (404 on the pinned build); the
            # models listing is the same readiness endpoint the pinned setup polls.
            response = await layer.client.get("models")
            response.raise_for_status()
            models = response.json()
        served = [m.get("id") for m in models.get("data", [])]
        if config.model not in served:
            raise RuntimeError(f"Served models {served} do not include {config.model}")
        return {
            "hardware": hardware,
            "native_tokenizer_count": count,
            "served_models": served,
            "max_model_len": [m.get("max_model_len") for m in models.get("data", [])],
            "profile": PROFILE,
            "serving_setup": fork_identity(),
            "capacity_note": "Startup must pass vLLM's full-length KV admission check; no context/cache fallback permitted.",
        }
    finally:
        await layer.close()


def worker_probe(working, jail=None):
    """Start B's persistent worker once, exactly as Kernel.start does, and close it.

    The worker needs the staged Duck segmentation helper; kernel abc-3g-v2-r1 failed this
    preflight when the probe omitted ARC_SEGMENTATION_SOURCE.
    """
    module = Path(__file__).parent
    helper = segmentation.write(Path(working) / "segmentation.py")
    return subprocess.run(
        [sys.executable, "-I", "-u", str(module / "worker.py")],
        input='{"type":"close"}\n',
        text=True,
        capture_output=True,
        timeout=30,
        env={
            **({"ARC_CHROOT": str(jail.root)} if jail else {}),
            "PATH": os.defpath,
            "ARC_SANDBOX_MODULE": str(module / "sandbox.py"),
            "ARC_SEGMENTATION_SOURCE": str(helper),
            "OPENBLAS_NUM_THREADS": "1",
        },
    )


def verify_sequence_capacity(working, expected):
    """Check the launched command, not just the intended profile, before gameplay."""
    identity = json.loads((Path(working) / 'vllm-server-identity.json').read_text())
    argv = identity.get('argv', [])
    values = [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == '--max-num-seqs']
    if values != [str(expected)]:
        raise RuntimeError(f'Launched vLLM max-num-seqs {values!r} differs from requested {expected}')
    return {'expected_max_num_seqs': expected, 'actual_max_num_seqs': int(values[0]), 'verified': True}


def start(working, input_root, *, kv_cache_memory_bytes=0, generation_slots=8,
          kv_cache_dtype="auto", recipe_hook=None, max_num_batched_tokens=8192,
          mtp_tokens=3, prefix_caching=None, gpu_memory_utilization=None):
    if type(kv_cache_memory_bytes) is not int or kv_cache_memory_bytes < 0:
        raise ValueError("Explicit KV cache bytes must be a non-negative integer")
    # None keeps the fork's 0.96. A lower share leaves GPU memory outside the profiled KV pool (the
    # dfr-combined build's 640 px images ran the vision encoder out of memory at 0.96).
    utilization = GPU_MEMORY_UTILIZATION if gpu_memory_utilization is None else gpu_memory_utilization
    if (type(utilization) is not str or re.fullmatch(r"0\.\d{1,3}", utilization) is None
            or not 0.5 <= float(utilization) <= 0.99):
        raise ValueError("gpu_memory_utilization is a decimal string from 0.5 through 0.99")
    if type(mtp_tokens) is not int or mtp_tokens not in (0, 1, 2, 3, 4):
        raise ValueError("Flash serving comparisons support MTP depth 1-4, or 0 (no MTP draft)")
    # None keeps the profile's prefix caching; False is a serving-lab diagnostic only (27 September).
    prefix_caching = PROFILE["prefix_caching"] if prefix_caching is None else prefix_caching
    if type(prefix_caching) is not bool:
        raise ValueError("prefix_caching must be a bool")
    profile = {**PROFILE, "max_num_seqs": generation_slots, "prefix_caching": prefix_caching,
               "speculative_tokens": mtp_tokens,
               "max_num_batched_tokens": max_num_batched_tokens,
               "kv_cache_dtype": "bfloat16" if kv_cache_dtype == "auto" else kv_cache_dtype}
    if gpu_memory_utilization is not None:
        profile["gpu_memory_utilization"] = utilization
    if kv_cache_memory_bytes:
        profile.update(kv_cache_memory_bytes=kv_cache_memory_bytes,
                       kv_cache_allocation="explicit byte limit; native context admission remains mandatory")
    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    recipe = None
    stage = "hardware"
    try:
        verify_hardware()
        stage = "sandbox"
        jail = Jail() if needed() else None
        probe = worker_probe(working, jail)
        if probe.returncode or '"type": "ready"' not in probe.stdout:
            raise RuntimeError(
                "Linux sandbox preflight failed before GPU model startup: " + probe.stderr
            )
        sandbox_result = json.loads(probe.stdout.splitlines()[0])
        stage = "server_startup"
        os.environ[GPU_MEMORY_UTILIZATION_ENV] = utilization
        bundle = stage_fork_bundle(input_root, working / "fork-bundle")
        recipe = load_recipe(
            working,
            input_root,
            mtp_tokens=mtp_tokens,
            generation_slots=generation_slots,
            kv_cache_dtype=kv_cache_dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            bundle_dir=bundle,
            setup_sha256=fork_identity()["fork_sha256"],
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            prefix_caching=prefix_caching,
        )
        if recipe_hook is not None:
            profile["arena_cache"] = recipe_hook(recipe, working)
        recipe.ANALYZER_CONTEXT = 262144
        recipe.main()
        stage = "sequence_capacity"
        sequence_capacity = verify_sequence_capacity(working, generation_slots)
        stage = "native_context"
        result = asyncio.run(preflight())
        result["profile"] = profile
        result["sequence_capacity"] = sequence_capacity
        result["sandbox_preflight"] = sandbox_result
        result["kv_admission_log"] = kv_admission_summary(working)
        (working / "native-preflight.json").write_text(json.dumps(result, indent=2))
    except BaseException as exc:
        if recipe is not None:
            recipe.stop_owned_server()
        (working / "native-preflight-failure.json").write_text(
            json.dumps(
                {
                    "error": str(exc),
                    "stage": stage,
                    "profile": profile,
                    "kv_admission_log": kv_admission_summary(working),
                    "diagnostic": (
                        f"Native 262144-token context with {profile['kv_cache_dtype']} cache "
                        + (f"limited to {kv_cache_memory_bytes} bytes" if kv_cache_memory_bytes
                           else f"profiled at utilization {utilization}")
                        + " is mandatory. Check startup "
                        "KV-capacity diagnostics; do not silently lower context or change hardware."
                    ),
                },
                indent=2,
            )
        )
        raise
