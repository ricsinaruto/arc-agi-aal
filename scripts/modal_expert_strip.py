"""Modal CPU job: strip a pruning mask's experts from the int3 g16 overlay (throughput arm phase 5, 5 October).

    ARC_VALIDATION_ROOT=. ARC_STRIP_MASK=data/expert_prune/prune-t50.json \
        uv run --extra modal modal run --detach scripts/modal_expert_strip.py --run-id tp-strip-t50-v1-modal-20261005

Writes /assets/flash-input/datasets/richardcsaky/arc-expert-int3-g16-<mask stem>-v1 (served with that mask only).
"""

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or HERE).resolve()
MASK = os.environ.get("ARC_STRIP_MASK", "data/expert_prune/prune-t50.json")
SRC = "/assets/flash-input/datasets/richardcsaky/arc-expert-int3-g16-v1"
DST = f"/assets/flash-input/datasets/richardcsaky/arc-expert-int3-g16-{Path(MASK).stem}-v1"
TIMEOUT, CPU, RAM_GIB = 3600, 4, 16
STORAGE_USD = 0.5

app = modal.App("arc-tp-expert-strip", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .add_local_dir(str(HERE / "src/arena"), "/opt/src/arena", copy=True, ignore=["**/__pycache__/**"])
         .add_local_file(str(HERE / MASK), "/opt/mask.json", copy=True)
         .add_local_file(__file__, "/root/modal_expert_strip.py", copy=True))


@app.function(image=image, cpu=CPU, memory=RAM_GIB * 1024, timeout=TIMEOUT, retries=0,
              volumes={"/assets": assets})
def strip(dst: str):
    import sys
    import time

    sys.path.insert(0, "/opt/src")
    from arena.expert_strip import strip_overlay

    started = time.time()
    totals = strip_overlay(SRC, dst, "/opt/mask.json")
    (Path(dst) / "strip-record.json").write_text(json.dumps({**totals, "seconds": round(time.time() - started)}) + "\n")
    assets.commit()
    return {**totals, "seconds": round(time.time() - started), "dst": dst}


@app.local_entrypoint()
def main(run_id: str):
    import fcntl

    if not run_id.startswith("tp-"):
        raise SystemExit("throughput-arm run ids start with tp-")
    rate = CPU * 0.0000131 + RAM_GIB * 0.00000222
    bound = round((TIMEOUT + 60) * rate * 1.15 + STORAGE_USD, 3)
    path = ROOT / "docs/spend.json"
    with path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        if any(r["run_id"] == run_id for r in d["runs"]):
            raise SystemExit("Already reserved")
        d["runs"].append({"run_id": run_id, "state": "reserved", "kind": "research-preparation",
                          "charged_bound_usd": bound, "max_seconds": TIMEOUT, "gpu": None, "cpu_ceiling": CPU,
                          "memory_ceiling_gib": RAM_GIB,
                          "authorization": "throughput research arm, in total (run ids tp-*)",
                          "research_kind": f"expert-strip {Path(MASK).stem}"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()
    call = strip.spawn(DST)
    record = {"run_id": run_id, "state": "submitted", "function_call_id": call.object_id, "mask": MASK, "dst": DST,
              "charged_bound_usd": bound}
    (HERE / "docs/validation-20261003" / f"modal-{run_id}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))
