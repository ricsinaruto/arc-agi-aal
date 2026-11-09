"""The tensor-parallel all-reduce over PCIe peer access on four L4s (l4 arm, 27 Oct).

Kaggle's 4x L4 (probe richardcsaky/l4-p2p-probe v2): peer access on all 12 pairs (PHB, PCIe gen3 x16, ~10 GB/s a
copy), yet NCCL 2.27 runs every channel through host shared memory ("isAllDirectP2p 0", "via SHM/direct"): 23 / 59
/ 106 / 154 us a call in a graph at 1 / 16 / 32 / 64 rows, ~12 ms of a ~60 ms decode step at the play's ~40 rows.
vLLM's own custom all-reduce (one-shot / two-shot over peer pointers, CUDA-graph safe) is refused for more than two
PCIe GPUs only because ``is_fully_connected`` asks NVML for NVLink. ``custom_ar`` makes that answer True; vLLM still
runs its own P2P test (``_can_p2p``) and disables itself where peer access fails, so a host without it serves on
NCCL as before. Every rank sums the ranks in the same order: the ranks get the same bits, as with NCCL (the order
differs from NCCL's ring, so values are not bitwise NCCL's).

``ARC2_L4_CUSTOM_AR=1`` (option ``custom_ar``) turns it on; a vLLM general plugin, as ``arc2.l4_logits``.
"""

import json
import os

ENV = "ARC2_L4_CUSTOM_AR"


def install(module=None):
    """Patch the CUDA platform classes' ``is_fully_connected`` to True; returns the classes patched."""
    if module is None:
        from vllm.platforms import cuda as module
    patched = []
    for name in ("CudaPlatformBase", "NvmlCudaPlatform", "NonNvmlCudaPlatform"):
        cls = getattr(module, name, None)
        if cls is None or cls.__dict__.get("_arc2_custom_ar"):
            continue
        cls.is_fully_connected = classmethod(lambda cls, device_ids: True)
        cls._arc2_custom_ar = True
        patched.append(name)
    return patched


def register():
    """vLLM general plugin: in every process before the communicators are built."""
    if os.environ.get(ENV) != "1":
        return
    patched = install()
    print("ARC2_L4_CUSTOM_AR " + json.dumps({"patched": patched, "pid": os.getpid()}), flush=True)
