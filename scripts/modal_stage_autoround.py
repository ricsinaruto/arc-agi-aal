"""Modal CPU job: copy dfranzen's AutoRound W4A16 checkpoint (arc-dfr-exact) into the assets volume, hash-checked.

    ARC_VALIDATION_ROOT=. uv run --extra modal modal run --detach \
        scripts/modal_stage_autoround.py --run-id tp-stage-autoround-v1-20261005

Destination /assets/research-20260930/intel-autoround-w4a16 (served by arena.research_checkpoint, inventory
src/arena/models/intel-autoround-w4a16.json).
"""

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or HERE).resolve()
SRC = "/dfr/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
DST = "/assets/research-20260930/intel-autoround-w4a16"
INVENTORY = "src/arena/models/intel-autoround-w4a16.json"
TIMEOUT, CPU, RAM_GIB = 5400, 8, 16
STORAGE_USD = 1.0

app = modal.App("arc-tp-stage-autoround", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
dfr = modal.Volume.from_name("arc-dfr-exact")
image = (modal.Image.debian_slim(python_version="3.12")
         .add_local_file(str(HERE / INVENTORY), "/opt/inventory.json", copy=True)
         .add_local_file(__file__, "/root/modal_stage_autoround.py", copy=True))


@app.function(image=image, cpu=CPU, memory=RAM_GIB * 1024, timeout=TIMEOUT, retries=0,
              volumes={"/assets": assets, "/dfr": dfr})
def stage():
    import hashlib
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor

    started = time.time()
    files = json.loads(Path("/opt/inventory.json").read_text())["files"]
    os.makedirs(DST, exist_ok=True)

    def one(name):
        entry = files[name]
        src, dst = Path(SRC) / name, Path(DST) / name
        digest = hashlib.sha256()
        with src.open("rb") as f, open(f"{dst}.part", "wb") as out:
            while chunk := f.read(64 << 20):
                digest.update(chunk)
                out.write(chunk)
        if os.path.getsize(f"{dst}.part") != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"hash or size mismatch: {name}")
        shutil.move(f"{dst}.part", dst)
        print(json.dumps({"file": name, "bytes": entry["bytes"], "seconds": round(time.time() - started)}), flush=True)
        return name

    with ThreadPoolExecutor(8) as pool:
        done = list(pool.map(one, sorted(files, key=lambda n: -files[n]["bytes"])))
    record = {"files": len(done), "bytes": sum(files[n]["bytes"] for n in done), "all_sha256_verified": True,
              "seconds": round(time.time() - started)}
    (Path(DST) / "stage-record.json").write_text(json.dumps(record) + "\n")
    assets.commit()
    return record


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
                          "authorization": "Throughput arm (run ids tp-*): stage the AutoRound W4A16 checkpoint "
                                           "for the held-out matrix (row 1)"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()
    call = stage.spawn()
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id, "charged_bound_usd": bound}))
