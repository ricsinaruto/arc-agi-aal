"""Scratch (arc2, 13 October): masked GPTQ W4 overlay on Modal CPU, one container per layer, from a persisted capture.

Same program and settings as the GPU calibrate phase (arena.expert_quant_gptq, prior_rows 4096, act_order,
omit_pruned); each layer writes into <dst>/.parts/Lnn, then `merge` moves the shards up and writes one manifest.

    modal run --detach modal_gptq_cpu.py --run-id arc2-gptq-c240-k256-v1 --capture-run <nebius run> \
        --mask masks-c240/prune-arc2-k256-v1.json --name arc-expert-gptq-arc2c240-k256-v1
"""
import json
import os
from pathlib import Path

import modal

ARM = Path("./.claude/worktrees/throughput-arm")
LEDGER = Path("./docs/spend.json")
LAYER_TIMEOUT = int(os.environ.get("ARC_GPTQ_LAYER_TIMEOUT", "3600"))
CPU, RAM_GIB, PARALLEL = 8, 48, 24
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
BF16_DIR = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
OVERLAY_ROOT = "/assets/flash-input/datasets/richardcsaky"
MASK = os.environ.get("ARC_GPTQ_MASK", "masks-c240/prune-arc2-k256-v1.json")

app = modal.App("arc2-gptq-cpu-20261013", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                      extra_index_url="https://pypi.org/simple")
         .add_local_dir(str(ARM / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_file(str(Path(__file__).parent / MASK), "/opt/mask.json", copy=True)
         .add_local_file(__file__, "/root/modal_gptq_cpu.py", copy=True))


@app.function(image=image, cpu=CPU, memory=RAM_GIB * 1024, timeout=LAYER_TIMEOUT, retries=0, max_containers=PARALLEL,
              volumes={"/assets": assets, "/runs": runs})
def fit_layer(layer: int, run_id: str, capture_run: str, name: str):
    import subprocess
    import time

    started = time.time()
    cap = f"/runs/validation/{capture_run}/expert-capture"
    part = f"{OVERLAY_ROOT}/{name}/.parts/L{layer:02d}"
    rec = Path("/runs/research") / run_id
    rec.mkdir(parents=True, exist_ok=True)
    out = rec / f"gptq-L{layer:02d}.json"
    args = ["--calib", f"{cap}/calib", "--capture", cap, "--model-dir", MODEL_DIR, "--bf16-dir", BF16_DIR,
            "--overlay-out", part, "--out", str(out), "--deadline-seconds", str(LAYER_TIMEOUT - 120),
            "--layers", str(layer), "--percdamp", "0.01", "--min-rows", "64", "--batch", "128",
            "--scale-search", "1.0", "--eval-rows", "4096", "--prior-rows", "4096", "--act-order",
            "--prune-mask", "/opt/mask.json", "--omit-pruned", "--device", "cpu"]
    env = {**os.environ, "PYTHONPATH": "/opt/src", "OMP_NUM_THREADS": str(CPU), "MKL_NUM_THREADS": str(CPU)}
    proc = subprocess.run(["python", "-m", "arena.expert_quant_gptq", *args], env=env, capture_output=True, text=True)
    (rec / f"gptq-L{layer:02d}.log").write_text(proc.stdout[-200000:] + "\n--- stderr\n" + proc.stderr[-50000:])
    res = json.loads(out.read_text()) if out.is_file() else {}
    assets.commit()
    runs.commit()
    row = res.get("layers", {}).get(str(layer), {})
    return {"layer": layer, "rc": proc.returncode, "errors": res.get("errors"), "shards": len(res.get("shards") or {}),
            "seconds": round(time.time() - started),
            "error": {k: row.get(k) for k in row if "error" in k or k in ("served", "new")} if row else None}


@app.function(image=image, cpu=2, memory=4096, timeout=1200, retries=0, volumes={"/assets": assets, "/runs": runs})
def merge(name: str, layers: int = 48):
    import hashlib
    import time

    assets.reload()
    dst = Path(OVERLAY_ROOT) / name
    shards, done, first = {}, [], None
    for layer in range(layers):
        part = dst / ".parts" / f"L{layer:02d}"
        man = json.loads((part / "overlay-manifest.json").read_text())
        if layer not in man["layers"]:
            raise RuntimeError(f"L{layer} has no shards")
        first = first or man
        for shard, meta in man["shards"].items():
            if shard in shards:
                raise RuntimeError(f"shard {shard} written twice")
            src = part / shard
            if src.stat().st_size != meta["bytes"]:
                raise RuntimeError(f"{shard} size differs")
            os.replace(src, dst / shard)
            shards[shard] = meta
        done.append(layer)
    manifest = {**{k: first[k] for k in first if k not in ("layers", "shards", "complete", "utc")},
                "layers": done, "shards": shards, "complete": len(done) == 48,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    raw = json.dumps(manifest, indent=1, sort_keys=True)
    (dst / "overlay-manifest.json").write_text(raw)
    assets.commit()
    return {"shards": len(shards), "layers": len(done), "bytes": sum(m["bytes"] for m in shards.values()),
            "manifest_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def reserve(run_id, bound, seconds):
    import fcntl

    with LEDGER.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        if any(r["run_id"] == run_id for r in d["runs"]):
            raise ValueError("Already reserved")
        d["runs"].append({"run_id": run_id, "state": "reserved", "kind": "research-preparation",
                          "charged_bound_usd": bound, "max_seconds": seconds, "gpu": None, "cpu_ceiling": CPU * PARALLEL,
                          "memory_ceiling_gib": RAM_GIB * PARALLEL,
                          "authorization": "redo the expert pruning, 3-bit requant and refit on "
                                           "all 120 public ARC-2 eval tasks (ARC-AGI-2 arm, run ids arc2-)",
                          "research_kind": "gptq-w4-cpu"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()


@app.local_entrypoint()
def main(run_id: str, capture_run: str, name: str, layers: str = "", do_merge: bool = False):
    if do_merge:
        print(json.dumps(merge.remote(name)))
        return
    todo = [int(x) for x in layers.split(",")] if layers else list(range(48))
    rate = CPU * 0.0000131 + RAM_GIB * 0.00000222
    bound = round(len(todo) * (LAYER_TIMEOUT + 60) * rate * 1.15, 2)
    reserve(run_id, bound, LAYER_TIMEOUT * -(-len(todo) // PARALLEL))
    calls = [fit_layer.spawn(layer, run_id, capture_run, name) for layer in todo]
    record = {"run_id": run_id, "capture_run": capture_run, "name": name, "layers": todo,
              "function_call_ids": [c.object_id for c in calls], "charged_bound_usd": bound, "state": "submitted"}
    Path(__file__).with_name(f"modal-{run_id}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "function_call_ids"}))
