"""Modal CPU entry for scripts/sft/export.py:

    modal run --detach scripts/sft/modal_export.py --student student-p50g24c3 --out sft-heal/heal-overlay/NAME --prune t75
"""
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
AR_INDEX = "/assets/sft-heal/ar-check/model.safetensors.index.json"  # Intel AutoRound W4A16, what AV2C serves
ROUTER_SRC = f"/assets/expert-bf16/{REV}"
NVFP4_BASE = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"  # v8.5p's
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
# per-run volumes (28 Oct): a student in a per-run training volume is read at /in/<vol>/... (ARC_SFT_INPUT_VOLUMES)
INPUT_VOLUMES = [v for v in (os.environ.get("ARC_SFT_INPUT_VOLUMES") or "").split(",") if v]
app = modal.App("sft-export", include_source=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.14.0", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")
for _f in sorted((HERE / "prune").glob("*.json")) if modal.is_local() else []:
    image = image.add_local_file(str(_f), f"/opt/sft/prune/{_f.name}", copy=True)
image = (image.add_local_file(str(HERE / "export.py"), "/root/sft_export.py", copy=True)
         .add_local_file(__file__, "/root/modal_export.py", copy=True))
if INPUT_VOLUMES:  # the container re-imports this file: same mounts as the client's
    image = image.env({"ARC_SFT_INPUT_VOLUMES": ",".join(INPUT_VOLUMES)})


@app.function(image=image, timeout=7200, retries=0, volumes={"/assets": assets, "/runs": runs.read_only(),
                                                              **{f"/in/{v}": modal.Volume.from_name(v).read_only()
                                                                 for v in INPUT_VOLUMES}})
def run(student: str, out: str, prune: str = "t75", search: int = 20, nvfp4: bool = False):
    """``student``: /runs/... (arc-prolong-runs), /in/<vol>/... (a per-run volume, ARC_SFT_INPUT_VOLUMES) or a name
    under /assets/sft-heal; ``out``: a folder under /assets. A student on a per-run volume: set
    ARC_SFT_INPUT_VOLUMES=<vol> in the calling shell and pass --student /in/<vol>/... (the mounts are fixed at import,
    before the arguments are read). ``nvfp4``: the trained non-expert tensors as replacement shards of the
    NVFP4 checkpoint (export_nvfp4_dense, for the v8.5p serving path) instead of the AutoRound overlay."""
    import sft_export

    root = student if student.startswith("/") else f"/assets/sft-heal/{student}"
    # prune: a name (t75 -> heal_prune-t75-excl5-v1.json) or a mask file name in scripts/sft/prune
    prune_file = (f"/opt/sft/prune/{prune}" if prune.endswith(".json") else f"/opt/sft/prune/heal_prune-{prune}-excl5-v1.json") \
        if prune else None
    if nvfp4:
        counts = sft_export.export_nvfp4_dense(root, NVFP4_BASE, f"/assets/{out}", prune_file=prune_file)
    else:
        counts = sft_export.export(root, f"/assets/{out}", served_index=AR_INDEX, prune_file=prune_file,
                                   router_src=ROUTER_SRC, search=search)
    assets.commit()
    return counts


@app.local_entrypoint()
def main(student: str = "student-p50g24c3", out: str = "sft-heal/heal-overlay/p50g24c3-step0", prune: str = "t75",
         search: int = 20, nvfp4: bool = False):
    print(json.dumps(run.remote(student, out, prune, search, nvfp4)))
