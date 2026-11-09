"""Modal app for the Ornith-1.5-35B-A3B LoRA test (``docs/posttrain/lora-test-plan.md``, "How to run").

Modes (``ARC_LORA_MODE``; the job code is ``scripts/posttrain/lora_job.py``):

- ``fetch``  (CPU, network): the BF16 and NVFP4 checkpoints at their pinned revisions into the assets
  volume, every file verified against the pinned inventories;
- ``smoke``  (one H200, <= 15 min): load, kernel checks, 5 steps at 32K and 16K, held-out loss on 10
  packs, a projection of the training time;
- ``train``  (one H200): M0, held-out loss before and every half epoch, adapters to the runs volume;
  ``ARC_LORA_SECONDS`` sets the function timeout (default 3 h; the trainer stops early enough to
  evaluate and save);
- ``merge``  (CPU): merge + re-quantize + zero-adapter check + overlay, from a train run's adapter.

Every launch reserves its bound in ``docs/spend.json`` (GPU, CPU and memory rates x (timeout + 60 s)
x 1.15; the fetch also reserves a week of volume storage) and writes
``docs/validation-20260926/modal-<run id>.json`` with the function call id before returning.

Launch (always detached; see the memory note on Modal launches):
    ARC_LORA_MODE=smoke uv run --extra modal modal run --detach scripts/modal_lora.py --run-id lora-m0-smoke-v1
    ARC_LORA_MODE=merge uv run --extra modal modal run --detach scripts/modal_lora.py --run-id lora-m0-merge-v1 \\
        --adapter-run lora-m0-train-v1
Local checks without Modal calls or ledger changes:
    uv run python scripts/modal_lora.py --dry-run --mode train [--seconds 7200] [--adapter-run RUN]
Poll: ``uv run --extra modal python scripts/posttrain/poll_modal_lora.py RUN_ID``.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = "docs/validation-20260926"
M0_DIR = "data/posttrain/m0"
NVFP4_INVENTORY = "src/arena/models/ornith15-35b-a3b-nvfp4.json"
BF16_INVENTORY = "src/posttrain/inventories/ornith15-35b-a3b-bf16.json"
# Modal list prices (modal.com/pricing, checked 26 September 2026); CPU and memory rates are read from
# docs/spend.json (rates_per_second) when present.
GPU_RATES = {"H200": 0.001261, "B200": 0.001736, "RTX-PRO-6000": 0.000842}
CPU_RATE, MEMORY_RATE = 1.31e-05, 2.22e-06
MARGIN = 1.15
STORAGE_GIB = 96  # BF16 71.9 GB + NVFP4 23.4 GB in the assets volume
STORAGE_DAYS = 7
MODES = {
    "fetch": {"gpu": None, "cpu": 4.0, "memory_gib": 16, "seconds": 3600, "network": True},
    "smoke": {"gpu": "H200", "cpu": 8.0, "memory_gib": 64, "seconds": 900, "network": False},
    "train": {"gpu": "H200", "cpu": 8.0, "memory_gib": 64, "seconds": 10800, "network": False},
    "merge": {"gpu": None, "cpu": 8.0, "memory_gib": 32, "seconds": 3600, "network": False},
}
MAX_TRAIN_SECONDS = 4 * 3600
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
TORCH_PINS = ("torch==2.13.0", "torchvision==0.28.0")
PIP_PINS = ("transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0", "safetensors==0.8.0",
            "flash-linear-attention==0.5.2", "fla-core==0.5.2", "huggingface_hub>=1.5,<2", "tokenizers>=0.23.1,<0.24",
            "pillow", "numpy", "einops")
IMAGE_FILES = {  # repo path -> image path (the job imports only these)
    "scripts/posttrain/lora_job.py": "/root/lora/lora_job.py",
    "src/posttrain/__init__.py": "/root/lora/src/posttrain/__init__.py",
    "src/posttrain/requant.py": "/root/lora/src/posttrain/requant.py",
    "src/arena/__init__.py": "/root/lora/src/arena/__init__.py",
    "src/arena/weights_overlay.py": "/root/lora/src/arena/weights_overlay.py",
    NVFP4_INVENTORY: "/root/lora/inventories/nvfp4.json",
    BF16_INVENTORY: "/root/lora/inventories/bf16.json",
}
AUTHORIZATION = ("evening: + for the Ornith LoRA test training run on Modal (smoke, "
                 "training, merge/re-quantize); training on Modal, evaluations on Kaggle")


# --- pure helpers (tested) -------------------------------------------------------------------------


def mode_spec(mode, seconds=None):
    """Resources of ``mode``; ``seconds`` overrides the train timeout only (at most 4 h)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    spec = dict(MODES[mode])
    if seconds is not None:
        seconds = int(seconds)
        if mode != "train":
            raise ValueError("only the train timeout can be changed")
        if not 1800 <= seconds <= MAX_TRAIN_SECONDS:
            raise ValueError(f"train seconds must be 1800-{MAX_TRAIN_SECONDS}")
        spec["seconds"] = seconds
    return spec


def ledger_rates(ledger):
    rates = ledger.get("rates_per_second") or {}
    return rates.get("physical_cpu_core", CPU_RATE), rates.get("memory_gib", MEMORY_RATE)


def rate_per_second(spec, cpu_rate=CPU_RATE, memory_rate=MEMORY_RATE):
    return (GPU_RATES[spec["gpu"]] if spec["gpu"] else 0.0) + spec["cpu"] * cpu_rate + spec["memory_gib"] * memory_rate


def storage_usd(ledger, gib=STORAGE_GIB, days=STORAGE_DAYS):
    return gib * ledger.get("volume_rate_usd_per_gib_month", 0.09) * days / 30


def bound_usd(mode, spec, ledger):
    """The reservation: (timeout + 60 s) x rate x 1.15, plus a week of weight storage for the fetch."""
    cpu_rate, memory_rate = ledger_rates(ledger)
    bound = (spec["seconds"] + 60) * rate_per_second(spec, cpu_rate, memory_rate) * MARGIN
    return bound + (storage_usd(ledger) if mode == "fetch" else 0.0)


def committed_usd(ledger):
    from arc_infra.runops.records import row_cost
    return ledger["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in ledger["runs"])


def reserve_lora(root, run_id, mode, spec, *, extra=None):
    """Append the run to ``docs/spend.json`` under a file lock; refuse duplicates and the cap."""
    import fcntl

    path = Path(root) / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound = bound_usd(mode, spec, data)
        committed = committed_usd(data)
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required > ${data['cap_usd']}")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        entry = {"run_id": run_id, "state": "reserved", "kind": f"lora-{mode}", "charged_bound_usd": round(bound, 4),
                 "max_seconds": spec["seconds"], "gpu": spec["gpu"] or "none", "cpu_ceiling": spec["cpu"],
                 "memory_ceiling_gib": spec["memory_gib"], "authorization": AUTHORIZATION}
        if mode == "fetch":
            entry["storage_bound_usd"] = round(storage_usd(data), 4)
            entry["storage_note"] = (f"{STORAGE_GIB} GiB of weights in arc-prolong-assets for {STORAGE_DAYS} days; "
                                     "delete lora-weights/ after the merge")
        entry.update(extra or {})
        data["runs"].append(entry)
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


def record_path(root, run_id):
    return Path(root) / RECORDS / f"modal-{run_id}.json"


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_inputs(root, mode, *, adapter_run=None, adapter="adapter"):
    """Local checks before any Modal call: pinned inventories, the M0 files against their manifest,
    for a merge the train run's record. Returns a summary."""
    root = Path(root)
    summary = {"mode": mode}
    for label, rel in (("nvfp4", NVFP4_INVENTORY), ("bf16", BF16_INVENTORY)):
        inventory = json.loads((root / rel).read_text())
        shards = [n for n in inventory["files"] if n.endswith(".safetensors")]
        if not shards or any(not inventory["files"][n].get("sha256") for n in shards):
            raise ValueError(f"{rel}: every shard needs a pinned sha256")
        summary[label] = {"repo": inventory["repo"], "revision": inventory["revision"], "shards": len(shards)}
    if mode in ("smoke", "train"):
        m0 = root / M0_DIR
        manifest = json.loads((m0 / "manifest.json").read_text())
        for name, spec in manifest["files"].items():
            if sha256_file(m0 / name) != spec["sha256"]:
                raise ValueError(f"{m0 / name}: differs from the M0 manifest; rebuild with build_m0.py")
        tokenizer = json.loads((root / NVFP4_INVENTORY).read_text())["files"]["tokenizer.json"]["sha256"]
        if manifest["tokenizer_sha256"] != tokenizer:
            raise ValueError("M0 was tokenized with another tokenizer than the served checkpoint's")
        summary["m0"] = {n: s["sequences"] for n, s in manifest["files"].items()}
    if mode == "merge":
        if not adapter_run:
            raise ValueError("merge needs --adapter-run (the train run id)")
        if not (adapter == "adapter" or (adapter.startswith("adapter-step") and adapter[12:].isdigit())):
            raise ValueError("--adapter must be 'adapter' or 'adapter-step<N>'")
        record = record_path(root, adapter_run)
        if not record.is_file():
            raise ValueError(f"no launch record {record} for the adapter run")
        data = json.loads(record.read_text())
        if data.get("mode") != "train":
            raise ValueError(f"{adapter_run} is not a train run")
        summary["adapter"] = f"/runs/lora/{adapter_run}/{adapter}"
    return summary


def image_spec():
    return {"base": "debian-slim python 3.12 + build-essential (Triton compiles launcher stubs)",
            "torch": {"index": TORCH_INDEX, "pins": list(TORCH_PINS)}, "pip": list(PIP_PINS),
            "files": dict(IMAGE_FILES), "data": {M0_DIR: "/root/lora/m0"}}


def dry_run(argv=None):
    parser = argparse.ArgumentParser(description="Validate a LoRA launch locally (no Modal call, no ledger change)")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--adapter-run", default=None)
    parser.add_argument("--adapter", default="adapter")
    args = parser.parse_args(argv)
    spec = mode_spec(args.mode, args.seconds)
    ledger = json.loads((ROOT / "docs/spend.json").read_text())
    for rel in IMAGE_FILES:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(rel)
    summary = validate_inputs(ROOT, args.mode, adapter_run=args.adapter_run, adapter=args.adapter)
    bound = bound_usd(args.mode, spec, ledger)
    committed = committed_usd(ledger)
    report = {"spec": spec, "image": image_spec(), "inputs": summary,
              "reservation_usd": round(bound, 2), "committed_usd": round(committed, 2), "cap_usd": ledger["cap_usd"],
              "headroom_after_usd": round(ledger["cap_usd"] - committed - bound, 2),
              "fits": committed + bound <= ledger["cap_usd"]}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__" and "--dry-run" in sys.argv:
    dry_run(sys.argv[1:])
    raise SystemExit(0)


# --- Modal app -------------------------------------------------------------------------------------

import modal  # noqa: E402

MODE = os.environ.get("ARC_LORA_MODE", "smoke")
SECONDS = os.environ.get("ARC_LORA_SECONDS")
SPEC = mode_spec(MODE, SECONDS if MODE == "train" and SECONDS else None)
app = modal.App("arc-lora", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential", "git")
         .pip_install(*TORCH_PINS, index_url=TORCH_INDEX)
         .pip_install(*PIP_PINS)
         .env({"ARC_LORA_MODE": MODE, "ARC_LORA_SECONDS": SECONDS or "", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
for _rel, _dest in IMAGE_FILES.items():
    image = image.add_local_file(str(ROOT / _rel) if modal.is_local() else _dest, _dest, copy=True)
image = (image.add_local_dir(str(ROOT / M0_DIR) if modal.is_local() else "/root/lora/m0", "/root/lora/m0", copy=True)
         .add_local_file(__file__, "/root/modal_lora.py", copy=True))
RESOURCES = {"retries": 0, "max_containers": 1, "scaledown_window": 2, "image": image}


def _gpu_kwargs(mode):
    spec = mode_spec(mode, SECONDS if mode == "train" and SECONDS else None)
    kwargs = {"cpu": spec["cpu"], "memory": int(spec["memory_gib"] * 1024), "timeout": spec["seconds"],
              "block_network": not spec["network"]}
    if spec["gpu"]:
        kwargs["gpu"] = spec["gpu"]
    return kwargs


@app.function(volumes={"/assets": assets, "/runs": runs}, **RESOURCES, **_gpu_kwargs("fetch"))
def fetch_weights(run_id: str):
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, "/root/lora")
    import lora_job

    out = _Path("/runs/lora") / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = lora_job.Log(out / "fetch-log.jsonl")
    inventories = {k: _json.loads(_Path(f"/root/lora/inventories/{k}.json").read_text()) for k in ("bf16", "nvfp4")}
    try:
        result = lora_job.fetch("/assets/lora-weights", bf16_inventory=inventories["bf16"],
                                nvfp4_inventory=inventories["nvfp4"], log=log)
    finally:
        assets.commit()
        runs.commit()
    (out / "fetch.json").write_text(_json.dumps(result, indent=1))
    runs.commit()
    return result


@app.function(volumes={"/assets": assets, "/runs": runs}, **RESOURCES, **_gpu_kwargs("smoke"))
def smoke(run_id: str):
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    _os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    _sys.path.insert(0, "/root/lora")
    import lora_job

    out = _Path("/runs/lora") / run_id
    out.mkdir(parents=True, exist_ok=True)
    model_dir = lora_job.checkpoint_dir("/assets/lora-weights", "bf16")
    if not (model_dir / ".verified.json").is_file():
        raise RuntimeError(f"{model_dir} is not fetched and verified; run the fetch mode first")
    try:
        result = lora_job.smoke(model_dir, "/root/lora/m0", out, commit=runs.commit, max_seconds=MODES["smoke"]["seconds"])
    finally:
        runs.commit()
    return {"passed": result["passed"], "projection": result["projection"]}


@app.function(volumes={"/assets": assets, "/runs": runs}, **RESOURCES, **_gpu_kwargs("train"))
def train(run_id: str, max_seconds: int):
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    _os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    _sys.path.insert(0, "/root/lora")
    import lora_job

    out = _Path("/runs/lora") / run_id
    out.mkdir(parents=True, exist_ok=True)
    model_dir = lora_job.checkpoint_dir("/assets/lora-weights", "bf16")
    if not (model_dir / ".verified.json").is_file():
        raise RuntimeError(f"{model_dir} is not fetched and verified; run the fetch mode first")
    try:
        result = lora_job.train(model_dir, "/root/lora/m0", out, max_seconds=max_seconds, commit=runs.commit)
    finally:
        runs.commit()
    return result["summary"]


@app.function(volumes={"/assets": assets, "/runs": runs}, **RESOURCES, **_gpu_kwargs("merge"))
def merge(run_id: str, adapter_path: str, zero_adapter: bool):
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, "/root/lora")
    import lora_job

    out = _Path("/runs/lora") / run_id
    out.mkdir(parents=True, exist_ok=True)
    runs.reload()
    inventory = _json.loads(_Path("/root/lora/inventories/nvfp4.json").read_text())
    try:
        result = lora_job.merge(adapter_path, lora_job.checkpoint_dir("/assets/lora-weights", "bf16"),
                                lora_job.checkpoint_dir("/assets/lora-weights", "nvfp4"), out,
                                nvfp4_inventory=inventory, zero_adapter=zero_adapter, commit=runs.commit)
    finally:
        runs.commit()
    return result


@app.local_entrypoint()
def main(run_id: str, adapter_run: str = "", adapter: str = "adapter", zero_adapter: bool = False):
    summary = validate_inputs(ROOT, MODE, adapter_run=adapter_run or None, adapter=adapter)
    bound = reserve_lora(ROOT, run_id, MODE, SPEC)
    record = record_path(ROOT, run_id)
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "submission_intent", "run_id": run_id, "kind": "lora", "mode": MODE, "spec": SPEC,
            "charged_bound_usd": round(bound, 4), "platform_policy": "modal (user: LoRA training on Modal, "
            "evaluations on Kaggle)", "inputs": summary, "image": image_spec(),
            "outputs": f"arc-prolong-runs:/lora/{run_id}"}
    record.write_text(json.dumps(data, indent=2))
    if MODE == "fetch":
        call = fetch_weights.spawn(run_id)
    elif MODE == "smoke":
        call = smoke.spawn(run_id)
    elif MODE == "train":
        call = train.spawn(run_id, SPEC["seconds"])
    else:
        call = merge.spawn(run_id, summary["adapter"], zero_adapter)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
