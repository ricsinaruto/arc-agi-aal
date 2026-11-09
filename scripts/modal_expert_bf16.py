"""CPU-only Modal jobs for the ARC expert re-quantization (28 September; docs/validation-20260927/expert-quant.md):
the routed experts of the BF16 source model, Qwen/Qwen3.8-Flash-Next at a pinned revision, per layer, into the
assets volume.

    # download the BF16 expert shards of some layers (and config/index), verified against the HF LFS sha256
    uv run --extra modal modal run --detach scripts/modal_expert_bf16.py::fetch_main --run-id <id> --layers 0,10,20
    # delete them again (after the evaluation)
    uv run --extra modal modal run scripts/modal_expert_bf16.py::delete_main --run-id <id>

The BF16 checkpoint keeps each layer's routed experts fused: ``layers.N.mlp.experts.gate_up_proj`` [512, 1280, 2560]
(3.36 GB, a shard of its own) and ``layers.N.mlp.experts.down_proj`` [512, 2560, 640] (in the next shard, with the
layer's router, shared expert and hyper-connection weights). Files land in ``/assets/expert-bf16/<revision>/``.

Ledger: the main checkout's docs/spend.json (ARC_VALIDATION_ROOT), reserved before the spawn with the procedure of
scripts/modal_validation.py: (timeout + 60 s) x (CPU + memory rate) x 1.15, plus a week of volume storage.
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
RECORDS = "docs/validation-20260927"
BF16_REPO = "Qwen/Qwen3.8-Flash-Next"
BF16_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TARGET = f"/assets/expert-bf16/{BF16_REVISION}"
CPU, MEMORY_GIB, TIMEOUT = 8.0, 16, 3600
STORAGE_DAYS = int(os.environ.get("ARC_BF16_STORAGE_DAYS", "1"))  # the shards are deleted after the job that needs them
AUTHORIZATION = os.environ.get("ARC_BF16_AUTHORIZATION") or ("ARC-tuned routed-expert re-quantization (stage 1 microbenchmark <= "
                 "stage 2 GPTQ <=); CPU download of BF16 expert shards (no GPU lock)")

app = modal.App("arc-expert-bf16", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub>=0.34", "hf_xet", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
         .add_local_file(__file__, "/root/modal_expert_bf16.py", copy=True))


# --- pure helpers (tested) -------------------------------------------------------------------------


def layer_files(weight_map, layers):
    """The BF16 shards holding ``layers``' routed experts (gate_up_proj and down_proj), sorted."""
    files = set()
    for layer in layers:
        for proj in ("gate_up_proj", "down_proj"):
            files.add(weight_map[f"model.language_model.layers.{int(layer)}.mlp.experts.{proj}"])
    return sorted(files)


def file_specs(tree, names):
    """``{name: {"bytes", "sha256"}}`` from the HF tree listing (LFS sha256 for weights, size for small files)."""
    by_path = {row["path"]: row for row in tree}
    out = {}
    for name in names:
        row = by_path[name]
        lfs = row.get("lfs") or {}
        out[name] = {"bytes": int(row["size"]), "sha256": lfs.get("oid")}
    return out


def storage_gib(specs):
    return sum(s["bytes"] for s in specs.values()) / 2**30


def bound_usd(ledger, gib):
    rates = ledger.get("rates_per_second") or {}
    rate = CPU * rates.get("physical_cpu_core", 1.31e-05) + MEMORY_GIB * rates.get("memory_gib", 2.22e-06)
    storage = gib * ledger.get("volume_rate_usd_per_gib_month", 0.09) * STORAGE_DAYS / 30
    return (TIMEOUT + 60) * rate * 1.15 + storage, storage


def reserve(run_id, gib, extra):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound, storage = bound_usd(data, gib)
        committed = data["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": "expert-bf16-fetch", "charged_bound_usd": round(bound, 4),
            "max_seconds": TIMEOUT, "gpu": "none", "cpu_ceiling": CPU, "memory_ceiling_gib": MEMORY_GIB,
            "storage_bound_usd": round(storage, 4),
            "storage_note": f"{gib:.1f} GiB of BF16 expert shards in arc-prolong-assets{TARGET[7:]} for "
                            f"{STORAGE_DAYS} days; deleted after the evaluation",
            "authorization": AUTHORIZATION, **extra})
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


# --- remote ----------------------------------------------------------------------------------------


@app.function(image=image, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/assets": assets})
def fetch(specs: dict):
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    target = Path(TARGET)
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def one(name):
        spec = specs[name]
        local = target / name
        t0 = time.time()
        if not (local.is_file() and local.stat().st_size == spec["bytes"]):
            hf_hub_download(BF16_REPO, name, revision=BF16_REVISION, local_dir=str(target))
        size = local.stat().st_size
        digest = None
        if spec.get("sha256"):
            with local.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        ok = size == spec["bytes"] and (spec.get("sha256") is None or digest == spec["sha256"])
        row = {"file": name, "bytes": size, "sha256": digest, "ok": ok, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, sorted(specs)))
    manifest = {"repo": BF16_REPO, "revision": BF16_REVISION, "files": rows, "seconds": round(time.time() - started, 1),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (target / f"fetch-{int(started)}.json").write_text(json.dumps(manifest, indent=1))
    assets.commit()
    bad = [r["file"] for r in rows if not r["ok"]]
    if bad:
        raise RuntimeError(f"size/sha256 mismatch: {bad}")
    return manifest


@app.function(image=image, cpu=0.25, memory=512, timeout=600, volumes={"/assets": assets})
def delete_weights():
    import shutil

    target = Path(TARGET)
    existed = target.exists()
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if existed else 0
    if existed:
        shutil.rmtree(target)
    assets.commit()
    return {"deleted": TARGET, "existed": existed, "bytes": size}


@app.local_entrypoint()
def fetch_main(run_id: str, layers: str):
    import urllib.request

    base = f"https://huggingface.co/api/models/{BF16_REPO}"
    tree = json.loads(urllib.request.urlopen(f"{base}/tree/{BF16_REVISION}?recursive=1", timeout=60).read())
    index = json.loads(urllib.request.urlopen(
        f"https://huggingface.co/{BF16_REPO}/resolve/{BF16_REVISION}/model.safetensors.index.json", timeout=60).read())
    if layers == "all":  # throughput arm (6 Oct): the full BF16 model, the KL reference of the quantization matrix
        wanted = "all"
        names = sorted(row["path"] for row in tree if row.get("type") == "file")
    else:
        wanted = [int(x) for x in layers.split(",") if x.strip()]
        names = layer_files(index["weight_map"], wanted) + ["config.json", "model.safetensors.index.json"]
    specs = file_specs(tree, names)
    gib = storage_gib(specs)
    bound = reserve(run_id, gib, {"layers": wanted, "files": len(specs)})
    record = REPO / RECORDS / f"modal-{run_id}.json"
    data = {"state": "submission_intent", "run_id": run_id, "kind": "expert-bf16-fetch", "charged_bound_usd": bound,
            "repo": BF16_REPO, "revision": BF16_REVISION, "layers": wanted, "gib": round(gib, 2),
            "target": f"arc-prolong-assets:{TARGET[7:]}", "ledger": str(ROOT / "docs/spend.json")}
    record.write_text(json.dumps(data, indent=2))
    call = fetch.spawn(specs)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


@app.local_entrypoint()
def delete_main(run_id: str):
    result = delete_weights.remote()
    record = REPO / RECORDS / f"modal-{run_id}.json"
    data = json.loads(record.read_text()) if record.exists() else {"run_id": run_id}
    data["deleted"] = result
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(result))
