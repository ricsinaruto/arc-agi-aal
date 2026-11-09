"""mmrtx expert pruning (Modal CPU) on the shared core (dots3 a01c1d064 scripts/rtxfit/prune_core.py, fetched unmodified
into prune_core.py): keep the top-K experts per MoE layer by routed mass summed over the reference run's FIT tags only
(fit3:* = excl5 ARC-3 games, fit2:* = ARC-2 training tasks). Experts stay in the shipped NVFP4 (byte copies).

usage: modal run --detach prune.py --ref REFRUN --k 24 --out minimax-m31f-p24-v1"""
import json
from pathlib import Path

import modal

import sys as _sys
_sys.path[:0] = [str(__import__('pathlib').Path(__file__).resolve().parent), '/root']
from image import IMAGE  # noqa: E402

app = modal.App("mmrtx-prune")
models = modal.Volume.from_name("mmrtx-models")
PRUNE_IMAGE = IMAGE.add_local_file(str(Path(__file__).resolve().parent / "prune_core.py"), "/root/prune_core.py") \
    if modal.is_local() else IMAGE


def fit_mass(route_pt):
    import torch
    stats = torch.load(route_pt, weights_only=False)
    tags = [t for t in stats if t.split(":")[0] in ("fit3", "fit2") and stats[t][0] is not None]
    assert tags and not any(t.startswith("held") for t in tags)
    return sum(stats[t][0]["mass"] for t in tags), tags


@app.function(image=PRUNE_IMAGE, timeout=3 * 3600, cpu=(0.125, 8), volumes={"/m": models})
def build(ref: str, k: int, out: str):
    import prune_core as pc
    mass, tags = fit_mass(f"/m/ref/{ref}/route.pt")
    keep, info = pc.fit_mask(mass, pc.MINIMAX_M3.layers, k)
    dst = Path("/m") / out
    total = pc.prune("/m/minimax-m31f", dst, keep, pc.MINIMAX_M3, k)
    assert (dst / "dspark").is_dir(), "DSpark draft not copied"
    info.update(k=k, bytes=total, ref=ref, fit_tags=len(tags))
    (dst / "mmrtx-mask.json").write_text(json.dumps({"info": info, "keep": {str(a): v for a, v in keep.items()}}))
    models.commit()
    print(json.dumps(info))
    return info


@app.local_entrypoint()
def main(ref: str, k: int, out: str):
    call = build.spawn(ref, k, out)
    print(json.dumps({"call": call.object_id, "app_id": app.app_id}))
