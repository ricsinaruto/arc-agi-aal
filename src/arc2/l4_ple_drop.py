"""Four L4s: the PLE offload process drops each PLE shard from the page cache once it has copied it (``ple_drop``).

v41 (19 Oct, Kaggle inputs on GCS FUSE): the notebook read the whole checkpoint (206 shards, 94.3 GiB) into the page
cache, which the container's cgroup charges (vLLM's cgroup headroom at load start: 20.3 GiB). The PLE offload
process then copies the FP8 PLE table (``model-plefp8-*``, 47.7 GiB) into its own memory; past the headroom the
kernel reclaimed the shards no loader had touched yet, and they came back through FUSE mmap faults (45-110 MB/s):
"Loading weights took 563 s" against 44 s for the 180 shards read before. With each PLE shard dropped one file
after it was read (its tensors are copied and released by then), cache plus copy stay near the checkpoint's size.

Standard library only. ``apply`` edits the extracted runtime's ``weight_utils.py`` (disposable), once, by exact
anchors; the edit does nothing unless ``ENV`` names a shard pattern and the process is the PLE offload process.
"""

import hashlib
import json
from pathlib import Path

RELATIVE = "vllm/model_executor/model_loader/weight_utils.py"
ENV = "ARC2_PLE_DROP"
PATTERN = "model-plefp8-"  # the table's shards in the pinned checkpoint; no other loader reads them
MARK = "ARC2 L4 ple_drop"
HELPER = '''
def _arc_ple_drop(pending, path):
    # ARC2 L4 ple_drop: in the PLE offload process, shards whose name holds $ARC2_PLE_DROP leave the page cache one
    # file after they were read (their tensors are copied into the table and released by then); the end of the
    # stream (path None) drops the last. The table's copy otherwise pushed the cgroup over its limit and the kernel
    # evicted the GPU ranks' unread shards (v41: 563 s over GCS FUSE).
    pattern = os.environ.get("ARC2_PLE_DROP")
    if not pattern or not is_offload_process():
        return
    ready = list(pending)
    pending.clear()
    if path is not None and pattern in os.path.basename(path):
        pending.append(path)
    for done in ready:
        try:
            size = os.path.getsize(done)
            fd = os.open(done, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            _ARC_PLE_DROPPED[0] += 1
            _ARC_PLE_DROPPED[1] += size
        except (AttributeError, OSError):
            pass
    if path is None:
        logger.info("ARC2_PLE_DROP %s", json.dumps({"files": _ARC_PLE_DROPPED[0],
                                                    "gib": round(_ARC_PLE_DROPPED[1] / 2**30, 2)}))


_ARC_PLE_DROPPED = [0, 0]

'''
EDITS = (
    ("\ndef safetensors_weights_iterator(\n", "\n" + HELPER + "\ndef safetensors_weights_iterator(\n"),
    ("    leftover_state_dict: dict[str, torch.Tensor] = {}\n    for st_file in tqdm(\n",
     "    leftover_state_dict: dict[str, torch.Tensor] = {}\n    _arc_pending: list[str] = []\n"
     "    for st_file in tqdm(\n"),
    ('''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
''', '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
            param = None  # ARC2 L4 ple_drop: the file's last tensor is the consumer's alone
        _arc_ple_drop(_arc_pending, st_file)
    _arc_ple_drop(_arc_pending, None)
'''),
)


def patched(text):
    """The file's text with the drop in the safetensors stream (unchanged if it is already there)."""
    if MARK in text:
        return text
    if "_arc_ple_drop" in text or any(text.count(anchor) != 1 for anchor, _ in EDITS):
        raise RuntimeError("ple_drop: the pinned safetensors stream is not where it was; re-read arc2.l4_ple_drop")
    for anchor, replacement in EDITS:
        text = text.replace(anchor, replacement)
    return text


def apply(site_packages, working):
    """Edit the runtime's weight loader and leave a record in ``working``."""
    path = Path(site_packages) / RELATIVE
    before = path.read_text()
    after = patched(before)
    compile(after, RELATIVE, "exec")
    if after != before:
        path.write_text(after)
    record = {"relative": RELATIVE, "pattern": PATTERN, "changed": after != before,
              "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
              "patched_sha256": hashlib.sha256(after.encode()).hexdigest()}
    (Path(working) / "ple-drop-patch.json").write_text(json.dumps(record, indent=1))
    return record
