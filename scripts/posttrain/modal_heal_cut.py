"""Modal CPU job: the healing student's BF16 checkpoint from the teacher's BF16 shards (heal_kd.cut_checkpoint), on the
assets volume. Default cut: prune75 (excl5 masks) + drop24 + top-k 4, the speed map's deepest unhealed cut.

    modal run --detach scripts/posttrain/modal_heal_cut.py --run-id sft-heal-cut-p75d24k4-v1-modal-20261016
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
DROP24 = [4, 11, 17, 25, 29, 30, 31, 32, 33, 34, 35, 36, 39, 41, 42, 44, 8, 9, 12, 13, 16, 27, 28, 43]
DROPS = {"24": DROP24, "16": DROP24[:16], "12": [29, 30, 31, 32, 33, 34, 35, 36, 39, 41, 42, 44], "": []}  # smg/smh sets
# single-layer-drop KL order on excl5 (sma-v3 ref; layers 0-2 and 47 unscored, kept); drop N = the first N
ORDER = [44, 32, 39, 42, 41, 35, 33, 36, 29, 30, 34, 31, 25, 17, 4, 11, 16, 43, 13, 28, 8, 12, 27, 9, 14, 23, 19, 38, 10,
         20, 15, 3, 18, 5, 26, 21, 6, 22, 45, 46, 7, 24, 40, 37]
DROPS.update({str(n): sorted(ORDER[:n]) for n in (4, 8, 20, 32, 36, 40)})
assert DROPS["12"] == sorted(ORDER[:12]) and sorted(DROPS["16"]) == sorted(ORDER[:16])
# greedy layer choice on p50 (sft-layers-p50-v1, 17 Oct): "c" = KL per decode cost (full attention 2x GDN), "k" = KL
DROPS.update({"g12c": [8, 9, 12, 13, 16, 21, 22, 29, 32, 35, 40, 44],
              "g16c": [8, 9, 12, 13, 16, 19, 20, 21, 22, 23, 29, 32, 34, 35, 40, 44],
              "g16k": [2, 4, 7, 12, 13, 16, 17, 19, 20, 22, 28, 29, 32, 37, 42, 44],
              "g8k62": [9, 10, 11, 17, 19, 20, 32, 44],  # greedy KL on p62 (sft-layers-p62-v1)
              "g24c3": [4, 5, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 31, 32, 37, 39, 42, 44]})  # p50 KL per cost, full 3x (sft-layers-p50d24-v1)
app = modal.App("sft-heal-cut", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")
         .add_local_file(str(HERE / "heal_kd.py"), "/opt/heal/heal_kd.py", copy=True)
         .add_local_file(str(HERE / "heal_prune-t75-excl5-v1.json"), "/opt/heal/prune.json", copy=True)
         .add_local_file(str(HERE / "heal_prune-t50-excl5-v1.json"), "/opt/heal/prune-t50.json", copy=True)
         .add_local_file(str(HERE / "heal_prune-t62-excl5-v1.json"), "/opt/heal/prune-t62.json", copy=True)
         .add_local_file(str(HERE / "heal_prune-t87-excl5-v1.json"), "/opt/heal/prune-t87.json", copy=True)
         .add_local_file(__file__, "/root/modal_heal_cut.py", copy=True))


def width_keep(energy_path, width, block=128):
    """Hidden channels kept: whole ``block``s with the most residual-stream energy on the excl5 usage set, each
    layer's mean squares normalised to sum 1 and averaged over layers (``channel_energy``: decoder outputs + mixer)."""
    import torch
    import heal_kd

    ms = torch.load(energy_path, weights_only=True)["mean_square"].double()
    score = (ms / ms.sum(-1, keepdim=True)).mean(0)
    return heal_kd.width_blocks(score, block=block, keep_blocks=width // block)


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/assets": assets, "/runs": runs})
def cut(name: str, top_k: int, width: int = 0, energy: str = "", stack: bool = True, prune: str = "t75",
        drop: str = "24"):
    import sys
    sys.path.insert(0, "/opt/heal")
    import heal_kd

    dst = Path("/assets/sft-heal") / name
    dst.mkdir(parents=True, exist_ok=True)
    keep = width_keep(energy, width) if width else None
    prune_file = {"t75": "/opt/heal/prune.json", "t50": "/opt/heal/prune-t50.json", "t62": "/opt/heal/prune-t62.json",
                  "t87": "/opt/heal/prune-t87.json", "": None}[prune]
    result = heal_kd.cut_checkpoint(f"/assets/expert-bf16/{REV}", dst, drop=DROPS[drop] if stack else [],
                                    prune_file=prune_file if stack else None,
                                    top_k=(top_k or None) if stack else None, log=str(dst), width_keep=keep)
    assets.commit()
    return {**(result or {}), "width": len(keep) if keep else None,
            "blocks": sorted({c // 128 for c in keep}) if keep else None}


@app.local_entrypoint()
def main(run_id: str = "sft-heal-cut-p75d24k4-v1", name: str = "student-p75d24k4", top_k: int = 4, width: int = 0,
         energy: str = "", stack: bool = True, prune: str = "t75", drop: str = "24"):
    print(json.dumps(cut.remote(name, top_k, width, energy, stack, prune, drop)))
