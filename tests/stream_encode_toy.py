"""Spawn-importable toy encode for the ProcessEncoder test (test_heal_stream.fake_encode's rule)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)


def init(tag):
    assert tag == "toy"


def encode(capture):
    import torch

    import heal_stream

    img = 49
    ids = [int(t) for m in capture["payload"]["messages"] for t in m["content"].split()]
    start = len(ids)
    ids += [int(t) for t in capture["reply"].split()]
    tensor = torch.tensor([ids])
    inputs = {"input_ids": tensor, "mm_token_type_ids": (tensor == img).int()}
    runs = heal_stream.image_runs(tensor[0], img)
    if runs:
        inputs["image_grid_thw"] = torch.tensor([[1, 2, 2 * (b - a)] for a, b in runs])
        rows = int(inputs["image_grid_thw"].prod(-1).sum())
        inputs["pixel_values"] = torch.arange(rows, dtype=torch.float32)[:, None]
    return inputs, start
