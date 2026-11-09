"""Runtime helper of the draft-head overlay (copied to ``vllm/arc_draft_head.py``; see draft_head_patch.py).

Dumps (eager execution only): while ``<dump_dir>/ACTIVE`` exists, every target forward appends its
positions (int32) and pre-final-mixer multi-stream rows (BF16 bits) to ``current.target.pos`` /
``current.target.bin``; while ``ACTIVE_DRAFT`` exists, the draft's step-1 sampled rows go to
``current.draft.*``. The client (``arena.draft_head``) sends one prompt at a time and moves the files.

Overlay: ``overlay_weights`` wraps the MTP weight iterator and substitutes the fine-tuned tensors.
"""

import json
import os
from pathlib import Path

_SETTINGS = json.loads(Path(__file__).with_name("arc_draft_head.json").read_text())
_DIR = _SETTINGS.get("dump_dir")


def _write(kind, positions, rows):
    import torch

    n = rows.shape[0]
    if positions.ndim == 2:
        positions = positions[0]
    pos = positions[:n].to(torch.int32).cpu().numpy()
    data = rows[:n].contiguous().view(torch.int16).cpu().numpy()
    with open(os.path.join(_DIR, f"current.{kind}.pos"), "ab") as handle:
        handle.write(pos.tobytes())
    with open(os.path.join(_DIR, f"current.{kind}.bin"), "ab") as handle:
        handle.write(data.tobytes())


def dump_target(positions, multi_hidden):
    if _DIR and os.path.exists(os.path.join(_DIR, "ACTIVE")):
        _write("target", positions, multi_hidden)


def dump_draft(positions, sample_hidden, spec_step_idx):
    if _DIR and spec_step_idx == 0 and os.path.exists(os.path.join(_DIR, "ACTIVE_DRAFT")):
        _write("draft", positions, sample_hidden)


def overlay_weights(weights):
    overlay = _SETTINGS.get("overlay")
    if not overlay:
        yield from weights
        return
    from safetensors.torch import load_file

    manifest = json.loads((Path(overlay["path"]) / "overlay-manifest.json").read_text())
    tensors = load_file(str(Path(overlay["path"]) / manifest["overlay"]["file"]))
    used = set()
    for name, weight in weights:
        replacement = tensors.get(name)
        if replacement is not None:
            if replacement.shape != weight.shape or replacement.dtype != weight.dtype:
                raise RuntimeError(f"draft-head overlay {name}: {replacement.dtype} {tuple(replacement.shape)} "
                                   f"differs from the checkpoint's {weight.dtype} {tuple(weight.shape)}")
            used.add(name)
            weight = replacement
        yield name, weight
    missing = sorted(set(tensors) - used)
    if missing:
        raise RuntimeError(f"draft-head overlay tensors never loaded: {missing[:5]} ({len(missing)})")
    print(f"ARC_DRAFT_HEAD_OVERLAY applied {len(used)} tensors from {overlay['path']}", flush=True)
