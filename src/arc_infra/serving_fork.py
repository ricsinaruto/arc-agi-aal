"""The ARC fork of the pinned serving script and its staged bundle.

The fork uses GPU_MEMORY_UTILIZATION=0.96 by default so vLLM profiles activations
and gives remaining GPU memory to the KV cache. Bundles pin the fork's hash.
"""

import hashlib
import json
import shutil
from pathlib import Path

from .serving import SETUP_SHA256, SOURCE_DATASET, dataset_mount

FORK_SETUP = Path(__file__).with_name("serving_setup.py")
GPU_MEMORY_UTILIZATION = "0.96"
GPU_MEMORY_UTILIZATION_ENV = "TAAF_VLLM_GPU_MEMORY_UTILIZATION"


def fork_identity():
    return {
        "parent_sha256": SETUP_SHA256,
        "fork_sha256": hashlib.sha256(FORK_SETUP.read_bytes()).hexdigest(),
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }


def stage_fork_bundle(input_root, staging):
    """Copy the pinned source bundle with the forked serving script and matching identity.

    The script verifies its own hash against SOURCE_IDENTITY.json in its bundle directory
    and requires the PLE patch directory to be a regular directory of regular files.
    """
    source = dataset_mount(SOURCE_DATASET, input_root)
    staging = Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(
        source, staging, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    identity = json.loads((staging / "SOURCE_IDENTITY.json").read_text())
    if identity.get("serving_setup_sha256") != SETUP_SHA256:
        raise RuntimeError("Source bundle identity does not name the reviewed parent script")
    fork = fork_identity()
    shutil.copyfile(FORK_SETUP, staging / "serving_setup.py")
    identity["serving_setup_sha256"] = fork["fork_sha256"]
    identity["arc_fork"] = {**fork, "parent_source_dataset": SOURCE_DATASET}
    (staging / "SOURCE_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
    return staging
