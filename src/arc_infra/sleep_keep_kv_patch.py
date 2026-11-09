"""ARC_SLEEP_KEEP_KV=1 (MFU-2, 25 Oct): vLLM's level-1 sleep offloads the KV pool to pinned host memory with the
weights, instead of discarding it, and keeps the prefix cache. After /wake_up the held tries' prefixes hit the cache
instead of re-prefilling (v6j3 server 2: 88% of block time on prefill, ~10M tokens per wake against a 4.87M-token pool).

Edits, all reading the env at call time (the patched runtime behaves as stock vLLM with ARC_SLEEP_KEEP_KV unset):
- device_allocator/sleep_mode_backend.py, CuMemBackend.suspend: level 1 offloads tags ("weights", "kv_cache").
  CuMemAllocator restores each allocation at its original address, so block tables and hashes stay valid.
- v1/engine/core.py, EngineCore.sleep: level 1 pauses the scheduler without clearing the prefix cache (requests in
  flight are still aborted; their blocks stay cached and evictable).
- device_allocator/cumem.py, CuMemAllocator.sleep: under the env, each offloaded allocation backs up into an exact-size
  host buffer registered once (cudaHostRegister) and reused across sleeps. torch's pinned-host cache rounds each block
  up to a power of two: +199 GiB host for the hero's 120 GiB KV pool (sleepkv2). ARC_SLEEP_EXACT_PIN=0 keeps torch's
  pinned path; a failed register falls back to it (ARC_SLEEP_EXACT_PIN_FALLBACK).
- v1/worker/gpu_worker.py, compile_or_warm_up_model: after CUDA graph capture a background thread registers every
  host buffer once (ARC_SLEEP_PREPIN=0 turns it off; log ARC_SLEEP_PREPIN done): the first suspend skips pinning.
  ARC_SLEEP_VERIFY=1 (checks only): each offloaded allocation is hashed on the GPU twice before suspend (A/A) and
  once after its restore; the last restore of a wake prints ARC_SLEEP_VERIFY checked= gib= bad= aa_bad=.
Prefix-cache keys carry the LoRA name, so a reused name with new weights would hit stale KV: callers serve each
policy version under its own name (rl mg_name: policy-rNN). Host cost: one pinned copy of the KV pool (B300 hero:
~120 GiB), held for the server's life after the first sleep. Marker sleep-keep-kv-patch.json (boot-cache key)."""

import hashlib
import json
from pathlib import Path

ENV = "ARC_SLEEP_KEEP_KV"
TAG = "ARC sleep_keep_kv"
BACKEND = "vllm/device_allocator/sleep_mode_backend.py"
CORE = "vllm/v1/engine/core.py"
CUMEM = "vllm/device_allocator/cumem.py"
WORKER = "vllm/v1/worker/gpu_worker.py"

HELPER = '''from vllm.utils.torch_utils import PIN_MEMORY
import threading as _arc_threading
import time as _arc_time

_ARC_HOST_LOCK = _arc_threading.Lock()  # the boot prepin thread and sleep() register the same buffers
_ARC_HOST_BUFS: dict = {}  # ARC sleep_keep_kv: exact-size registered host backups, one per allocation, kept


def _arc_host_buf(ptr, size):
    if (os.environ.get("ARC_SLEEP_KEEP_KV") != "1" or os.environ.get("ARC_SLEEP_EXACT_PIN", "1") != "1"
            or not PIN_MEMORY):
        return torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=PIN_MEMORY)
    with _ARC_HOST_LOCK:
        buf = _ARC_HOST_BUFS.get(ptr)
        if buf is not None and buf.numel() == size:
            return buf
        cudart = torch.cuda.cudart()
        if buf is not None:
            cudart.cudaHostUnregister(buf.data_ptr())
            del _ARC_HOST_BUFS[ptr]
        buf = torch.empty(size, dtype=torch.uint8, device="cpu")
        try:  # 1 = cudaHostRegisterPortable: pinned for every context (the prepin thread registers off-thread)
            torch.cuda.check_error(cudart.cudaHostRegister(buf.data_ptr(), size, 1))
        except Exception as error:  # noqa: BLE001 - torch's pinned path still works, only larger
            print("ARC_SLEEP_EXACT_PIN_FALLBACK " + str(error)[:200], flush=True)
            return torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=PIN_MEMORY)
        if not _ARC_HOST_BUFS:
            print("ARC_SLEEP_EXACT_PIN registered", flush=True)
        _ARC_HOST_BUFS[ptr] = buf
        return buf


def _arc_prepin():
    """Register every offloadable allocation's host buffer once, in the background, after CUDA graph capture: the
    first suspend then costs seconds instead of 1-2.5 min of pinning (sleepkv3 149 s). Holds the host copy from boot."""
    if (os.environ.get("ARC_SLEEP_KEEP_KV") != "1" or os.environ.get("ARC_SLEEP_PREPIN", "1") != "1"
            or os.environ.get("ARC_SLEEP_EXACT_PIN", "1") != "1" or not PIN_MEMORY):
        return None
    alloc = CuMemAllocator.instance
    if alloc is None:
        return None
    jobs = [(ptr, d.handle[1]) for ptr, d in list(alloc.pointer_to_data.items())
            if d.tag in ("weights", "kv_cache") and not d.is_asleep]
    device = torch.cuda.current_device()

    def run():
        torch.cuda.set_device(device)
        start = _arc_time.time()
        for ptr, size in jobs:
            _arc_host_buf(ptr, size)
        print("ARC_SLEEP_PREPIN done n=%d gib=%.2f s=%.1f"
              % (len(jobs), sum(size for _, size in jobs) / 1024**3, _arc_time.time() - start), flush=True)

    thread = _arc_threading.Thread(target=run, name="arc-sleep-prepin", daemon=True)
    thread.start()
    return thread


_ARC_SUMS: dict = {}
_ARC_VERIFY = {"pending": set(), "checked": 0, "bad": 0, "aa_bad": 0, "bytes": 0}


def _arc_view(ptr, size):
    class _Mem:
        __cuda_array_interface__ = {"shape": (size // 4,), "typestr": "<i4", "data": (ptr, False), "version": 3}
    return torch.as_tensor(_Mem(), device="cuda")


def _arc_digest(t, step=1 << 24):
    out, w = [], None
    for i in range(0, t.numel(), step):
        c = t[i:i + step].to(torch.int64)
        if w is None or w.numel() != c.numel():
            w = torch.arange(c.numel(), device=c.device, dtype=torch.int64) % 65521 + 1
        out.append(torch.stack([c.sum(), (c * w).sum()]))
    return torch.stack(out).cpu() if out else torch.zeros((0, 2), dtype=torch.int64)


def _arc_verify(ptr, size, when):
    if os.environ.get("ARC_SLEEP_VERIFY") != "1":
        return
    v = _ARC_VERIFY
    if when == "sleep":
        if not v["pending"]:
            v.update(checked=0, bad=0, aa_bad=0, bytes=0)
        digest = _arc_digest(_arc_view(ptr, size))
        if not torch.equal(digest, _arc_digest(_arc_view(ptr, size))):
            v["aa_bad"] += 1
        _ARC_SUMS[ptr] = digest
        v["pending"].add(ptr)
        return
    if ptr not in _ARC_SUMS:
        return
    v["checked"] += 1
    v["bytes"] += size
    if not torch.equal(_arc_digest(_arc_view(ptr, size)), _ARC_SUMS.pop(ptr)):
        v["bad"] += 1
    v["pending"].discard(ptr)
    if not v["pending"]:
        print("ARC_SLEEP_VERIFY checked=%d gib=%.2f bad=%d aa_bad=%d"
              % (v["checked"], v["bytes"] / 1024**3, v["bad"], v["aa_bad"]), flush=True)
'''
KEEP = '__import__("os").environ.get("ARC_SLEEP_KEEP_KV") == "1"'
EDITS = {
    BACKEND: (
        ('        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())\n',
         '        keep = ' + KEEP + '  # ARC sleep_keep_kv\n'
         '        allocator.sleep(offload_tags=(("weights", "kv_cache") if keep else ("weights",)) if level == 1 else tuple())\n'
         '        if keep and level == 1:\n'
         '            print("ARC_SLEEP_KEEP_KV suspend", flush=True)\n'),
    ),
    CUMEM: (
        ('from vllm.utils.torch_utils import PIN_MEMORY\n', HELPER),
        ('                cpu_backup_tensor = torch.empty(\n'
         '                    size_in_bytes,\n'
         '                    dtype=torch.uint8,\n'
         '                    device="cpu",\n'
         '                    pin_memory=PIN_MEMORY,\n'
         '                )\n',
         '                cpu_backup_tensor = _arc_host_buf(ptr, size_in_bytes)\n'),
        ('                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)\n',
         '                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)\n'
         '                _arc_verify(ptr, size_in_bytes, "sleep")\n'),
        ('                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)\n',
         '                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)\n'
         '                        _arc_verify(ptr, size_in_bytes, "wake")\n'),
    ),
    WORKER: (
        ('            cuda_graph_memory_bytes = self.model_runner.capture_model()\n',
         '            cuda_graph_memory_bytes = self.model_runner.capture_model()\n'
         '        if __import__("os").environ.get("ARC_SLEEP_KEEP_KV") == "1":  # ARC sleep_keep_kv: prepin host copy\n'
         '            from vllm.device_allocator.cumem import _arc_prepin\n'
         '            _arc_prepin()\n'),
    ),
    CORE: (
        ('        clear_prefix_cache = level >= 1\n',
         '        clear_prefix_cache = level >= 1 and not (level == 1 and ' + KEEP + ')  # ARC sleep_keep_kv\n'),
    ),
}


def patched_source(relative, original):
    text = original.decode()
    if TAG in text:
        raise RuntimeError(f"sleep_keep_kv patch: {relative} already carries the edit")
    for old, new in EDITS[relative]:
        if text.count(old) != 1:
            raise RuntimeError(f"sleep_keep_kv anchor is not unique in {relative}: {old.strip()[:70]}")
        text = text.replace(old, new)
    compile(text, relative, "exec")
    return text.encode()


def apply(site, output):
    """All files or none. Never raises: an unrecognized build keeps stock sleep (ARC_SLEEP_KEEP_KV_PATCH_SKIPPED)."""
    site, marker = Path(site), Path(output) / "sleep-keep-kv-patch.json"
    originals = {rel: (site / rel).read_bytes() for rel in EDITS}
    if all(TAG.encode() in data for data in originals.values()):
        return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
    try:
        patched = {rel: patched_source(rel, data) for rel, data in originals.items()}
    except Exception as error:  # noqa: BLE001 - a speed path, never a reason to fail a boot
        print("ARC_SLEEP_KEEP_KV_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    for rel, text in patched.items():
        (site / rel).write_bytes(text)
    identity = {rel: {"original_sha256": hashlib.sha256(originals[rel]).hexdigest(),
                      "patched_sha256": hashlib.sha256(text).hexdigest()} for rel, text in patched.items()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_SLEEP_KEEP_KV_PATCH " + json.dumps(identity), flush=True)
    return identity
