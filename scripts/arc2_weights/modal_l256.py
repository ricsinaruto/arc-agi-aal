"""ARC-2 l256 layer-aware expert mask: masked W4 GPTQ + INT3 requant on Modal CPU.
Derived from the throughput arm's t50 script: routing recomputed on CPU under the mask from the c240 capture
(b2-arc2-quant-capture-c240-k256-v1, served under k256), arena.expert_quant_gptq with the 13 Oct k256 settings.
No cpu=/memory= reservations; SIZES only bound the ledger.

  modal run --detach modal_l256.py --kind fit --run-id arc2-gptq-c240-l256-v1 --mask prune-arc2-l256-v1.json --out arc-expert-gptq-arc2c240-l256-v1
  modal run --detach modal_l256.py --kind requant --run-id arc2-int3-requant-c240-l256-v1 --mask arc-expert-gptq-arc2c240-l256-v1 --out arc-expert-int3-requant-arc2c240-l256-v1
"""

import json
import os
import time
from pathlib import Path

import modal

DFR = Path("./.claude/worktrees/dfr-exact")
MAIN = Path(".")
HERE = Path(__file__).resolve().parent
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
BF16_DIR = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
CAPTURE = "/runs/validation/b2-arc2-quant-capture-c240-k256-v1-nebius-20261013/expert-capture"
OUT_ROOT = "/assets/flash-input/datasets/richardcsaky"
E, H, TOPK = 512, 2560, 10
SETTINGS = {"percdamp": 0.01, "min_rows": 64, "blocksize": 128, "batch": 128, "scale_search": [1.0],
            "eval_rows": 4096, "calib_rows": 0, "prior_rows": 4096.0, "act_order": True, "omit_pruned": True}
SIZES = {"route": (8, 32, 1200), "fit": (16, 96, 2400), "requant": (8, 32, 1500)}  # cpu, GiB, seconds per layer

app = modal.App("arc2-l256-20261020", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                      extra_index_url="https://pypi.org/simple")
         .add_local_dir(str(DFR / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_dir(str(HERE / "masks"), "/opt/masks", copy=True)
         .add_local_file(__file__, "/root/modal_l256.py", copy=True))
TPA = Path("./.claude/worktrees/throughput-arm")
rq_image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
            .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                         extra_index_url="https://pypi.org/simple")
            .add_local_dir(str(TPA / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
            .add_local_dir(str(HERE / "masks"), "/opt/masks", copy=True)
            .add_local_file(__file__, "/root/modal_l256.py", copy=True))
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"


def rclone_env():
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")


def _rq_worker(job):
    import sys

    sys.path.insert(0, "/opt/src")
    import torch

    torch.set_num_threads(2)
    from arena import expert_3bit_overlay as ov

    source, out, group = job
    st = ov.convert_shard(torch, source, out, group, "cpu")
    rel = ov.verify_shard(torch, source, out, 0.35)
    return os.path.basename(out), st, rel


def kept_of(mask_path):
    mask = json.load(open(mask_path))
    return {int(l): sorted(set(range(E)) - {int(e) for e in ids}) for l, ids in mask["pruned"].items()}


def load_split(split, layer, torch):
    import numpy as np

    root = os.path.join(CAPTURE, split)
    meta = json.load(open(os.path.join(root, "meta.json")))
    rows = int(meta["rows"][str(layer)])
    ids = np.fromfile(os.path.join(root, f"L{layer:02d}.ids"), dtype=np.int16).reshape(rows, TOPK)
    w = np.fromfile(os.path.join(root, f"L{layer:02d}.w"), dtype=np.float32).reshape(rows, TOPK)
    if meta.get("format") == "fp8":
        q = np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.uint8).reshape(rows, H)
        s = np.fromfile(os.path.join(root, f"L{layer:02d}.s"), dtype=np.float32)
        x = (torch.from_numpy(q).view(torch.float8_e4m3fn).float() * torch.from_numpy(s).unsqueeze(1)).to(torch.bfloat16)
    else:
        x = torch.from_numpy(np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.int16).reshape(rows, H)).view(
            torch.bfloat16)
    return x, torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(w)


def router(layer, torch):
    from safetensors import safe_open

    name = f"model.language_model.layers.{layer}.mlp.gate.weight"
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(MODEL_DIR, index[name]), framework="pt", device="cpu") as fh:
        return fh.get_tensor(name).float()


def route_rows(x, gate, kept, torch, chunk=32768):
    """Top-10 over ``kept`` (None: all 512), renormalised; ids in kept ranks."""
    ids, wts = [], []
    g = gate if kept is None else gate[kept]
    for s in range(0, x.shape[0], chunk):
        logits = x[s:s + chunk].float() @ g.t()
        top, idx = logits.topk(TOPK, dim=-1)
        w = torch.softmax(top, dim=-1)
        ids.append(idx)
        wts.append(w)
    return torch.cat(ids), torch.cat(wts)


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, timeout=SIZES["fit"][2])
def fit(layer: int, mask_name: str, out_name: str, run_id: str):
    import sys

    import torch

    torch.set_num_threads(SIZES["fit"][0])
    sys.path.insert(0, "/opt/src")
    import types

    src = open("/opt/src/arena/expert_quant_gptq.py").read()
    # CPU torch has no boolean-mask index ops on float8: do the fallback copies on uint8 views.
    for a, b in (('p13[fb], f13[fb] = served["w13"][fb], served["w13_sf"][fb]',
                  'p13[fb] = served["w13"][fb]; f13.view(self.torch.uint8)[fb] = served["w13_sf"].view(self.torch.uint8)[fb]'),
                 ('p2[fb], f2[fb] = served["w2"][fb], served["w2_sf"][fb]',
                  'p2[fb] = served["w2"][fb]; f2.view(self.torch.uint8)[fb] = served["w2_sf"].view(self.torch.uint8)[fb]')):
        if src.count(a) != 1:
            raise RuntimeError(f"patch anchor missing: {a}")
        src = src.replace(a, b)
    mod = types.ModuleType("arena.expert_quant_gptq")
    mod.__file__ = "/opt/src/arena/expert_quant_gptq.py"
    sys.modules["arena.expert_quant_gptq"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mask_path = f"/opt/masks/{mask_name}"
    kept = kept_of(mask_path)[layer]
    gate = router(layer, torch)
    rerouted = {}

    def reroute(split):
        if split not in rerouted:
            x, _, _ = load_split(split, layer, torch)
            ids, w = route_rows(x, gate, kept, torch)
            rerouted[split] = (x, ids.to(torch.int32), w.float())
        return rerouted[split]

    class Fit(mod.Gptq):
        def load_calib(self, layer_):
            return reroute("calib")

        def load_capture(self, split, layer_):
            return reroute(split)

    overlay_out = f"{OUT_ROOT}/{out_name}"
    work = f"/runs/research/{run_id}"
    os.makedirs(overlay_out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    s = SETTINGS
    argv = ["--calib", CAPTURE + "/calib", "--capture", CAPTURE, "--model-dir", MODEL_DIR, "--bf16-dir", BF16_DIR,
            "--overlay-out", overlay_out, "--out", f"{work}/L{layer:02d}.json",
            "--deadline-seconds", str(SIZES["fit"][2] - 120), "--layers", str(layer), "--percdamp", str(s["percdamp"]),
            "--min-rows", str(s["min_rows"]), "--blocksize", str(s["blocksize"]), "--batch", str(s["batch"]),
            "--scale-search", *map(str, s["scale_search"]), "--eval-rows", str(s["eval_rows"]),
            "--prior-rows", str(s["prior_rows"]), "--act-order", "--prune-mask", mask_path, "--omit-pruned",
            "--device", "cpu"]
    import argparse  # noqa: F401

    mod.Gptq = Fit
    t0 = time.time()
    try:
        mod.main(argv)
    finally:
        assets.commit()
        runs.commit()
    res = json.load(open(f"{work}/L{layer:02d}.json"))
    lay = res["layers"].get(str(layer), {})
    return {"layer": layer, "seconds": round(time.time() - t0, 1), "errors": res["errors"],
            "moe_errors": lay.get("errors"), "experts_gptq": lay.get("experts_gptq"),
            "rows_per_expert": lay.get("rows_per_expert"), "shards": lay.get("shards"),
            "settings": res.get("settings")}


@app.function(image=rq_image, volumes={"/assets": assets, "/runs": runs}, timeout=SIZES["requant"][2], secrets=[modal.Secret.from_name("nebius-s3")])
def requant(layer: int, src_name: str, out_name: str, run_id: str):
    """INT3 g16 requant (rtn-search, the 7 Oct recipe) of one layer's W4 shards, verified, then pushed to Nebius."""
    import hashlib
    import multiprocessing as mp
    import subprocess

    t0 = time.time()
    src_dir, out_dir = Path(OUT_ROOT) / src_name, Path(OUT_ROOT) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(p.name for p in src_dir.glob(f"layer-{layer:05d}-experts-*.safetensors"))
    if len(shards) != 4:
        raise RuntimeError(f"L{layer}: {len(shards)} W4 shards")
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(_rq_worker, [(str(src_dir / s), str(out_dir / s), 16) for s in shards])
    assets.commit()
    info = {}
    for name, st, rel in rows:
        path = out_dir / name
        size = path.stat().st_size
        if size != (src_dir / name).stat().st_size:
            raise RuntimeError(f"{name}: size differs from the W4 source")
        with open(path, "rb") as fh:
            info[name] = {"bytes": size, "sha256": hashlib.file_digest(fh, "sha256").hexdigest(), "base_bytes": size,
                          "stats": st, "verify": rel}
    rclone_env()
    prefix = f"flash-input/datasets/richardcsaky/{out_name}"
    for name in shards:
        subprocess.run(["rclone", "copyto", str(out_dir / name), f"arc:{BUCKET}/assets/{prefix}/{name}",
                        "--s3-upload-concurrency", "8", "--s3-chunk-size", "64M", "--retries", "5"], check=True)
    return {"layer": layer, "shards": info, "seconds": round(time.time() - t0, 1)}


@app.function(image=rq_image, volumes={"/assets": assets, "/runs": runs}, timeout=4800,
              secrets=[modal.Secret.from_name("nebius-s3")])
def driver(kind: str, run_id: str, mask_name: str = "", out_name: str = "", layers: str = ""):
    work = Path(f"/runs/research/{run_id}")
    work.mkdir(parents=True, exist_ok=True)
    todo = [int(v) for v in layers.split(",")] if layers else list(range(48))
    results, errors = [], []
    if kind == "requant":
        calls = requant.starmap([(l, mask_name, out_name, run_id) for l in todo], return_exceptions=True)
    else:
        calls = fit.starmap([(l, mask_name, out_name, run_id) for l in todo], return_exceptions=True)
    for layer, r in zip(todo, calls):
        if isinstance(r, Exception):
            errors.append(f"L{layer}: {type(r).__name__}: {str(r)[:400]}")
        else:
            results.append(r)
            if kind == "fit" and r["errors"]:
                errors.append(f"L{layer}: {r['errors']}")
    record = {"run_id": run_id, "kind": kind, "mask": mask_name, "out": out_name, "layers": todo,
              "errors": errors, "results": results, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if kind == "fit" and not errors and len(results) == 48:
        import hashlib

        assets.reload()

        shards = {}
        for r in results:
            shards.update(r["shards"])
        raw = open(f"/opt/masks/{mask_name}", "rb").read()
        manifest = {"format": "arc-expert-overlay-v1", "base_model_dir": MODEL_DIR, "layers": list(range(48)),
                    "shards": shards, "settings": results[0]["settings"], "complete": len(shards) == 192,
                    "expert_prune_sha256": hashlib.sha256(raw).hexdigest(), "omits_pruned": True,
                    "utc": record["utc"], "capture": CAPTURE, "routing": "recomputed on CPU under the mask"}
        Path(f"{OUT_ROOT}/{out_name}/overlay-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        record["manifest_shards"] = len(shards)
        assets.commit()
    if kind == "requant" and not errors and len(results) == 48:
        import hashlib
        import subprocess

        assets.reload()  # the workers' commits are not visible to a container mounted before them

        src_manifest = json.loads((Path(OUT_ROOT) / mask_name / "overlay-manifest.json").read_text())
        shards, stats, verify = {}, {}, {}
        for r in results:
            for name, v in r["shards"].items():
                shards[name] = {k: v[k] for k in ("bytes", "sha256", "base_bytes")}
                stats[name], verify[name] = v["stats"], v["verify"]
        manifest = {"format": "arc-expert-overlay-v1", "base_model_dir": MODEL_DIR, "layers": list(range(48)),
                    "shards": shards, "complete": len(shards) == 192,
                    "settings": {"grid": "int3", "group": 16, "method": "rtn-search",
                                 "search": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75],
                                 "source_dir": f"{OUT_ROOT}/{mask_name}"},
                    "utc": record["utc"], "omits_pruned": True,
                    "expert_prune_sha256": src_manifest["expert_prune_sha256"]}
        raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
        out_dir = Path(OUT_ROOT) / out_name
        (out_dir / "build-stats.json").write_text(json.dumps(stats, indent=1))
        (out_dir / "verify.json").write_text(json.dumps(verify, indent=1, sort_keys=True))
        (out_dir / "overlay-manifest.json").write_bytes(raw)
        assets.commit()
        record["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        rclone_env()
        prefix = f"flash-input/datasets/richardcsaky/{out_name}"
        for name in ("build-stats.json", "verify.json", "overlay-manifest.json"):
            subprocess.run(["rclone", "copyto", str(out_dir / name), f"arc:{BUCKET}/assets/{prefix}/{name}"], check=True)
    (work / "result.json").write_text(json.dumps(record, indent=1))
    runs.commit()
    return {"errors": errors, "n": len(results)}


def reserve(run_id, kind, n_layers):
    import fcntl

    cpu, gib, secs = SIZES[kind]
    bound = round(n_layers * secs * (cpu * 0.0000131 + gib * 0.00000222) * 1.15 + 0.2, 4)
    path = MAIN / "docs/spend.json"
    with path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        if any(r["run_id"] == run_id for r in d["runs"]):
            raise ValueError("Already reserved")
        d["runs"].append({"run_id": run_id, "state": "reserved", "kind": "research-preparation",
                          "charged_bound_usd": bound, "max_seconds": secs, "gpu": None, "cpu_ceiling": None,
                          "memory_ceiling_gib": None,
                          "authorization": "new ARC-2 kernel tonight, "
                                           "layer-aware mean-256 mask fitted on c240 (ARC-AGI-2 arm, run ids arc2-)"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()
    return bound


@app.local_entrypoint()
def main(kind: str, run_id: str, mask: str = "", out: str = "", layers: str = ""):
    n = len(layers.split(",")) if layers else 48
    bound = reserve(run_id, kind, n)
    call = driver.spawn(kind, run_id, mask, out, layers)
    rec = {"run_id": run_id, "kind": kind, "function_call_id": call.object_id, "charged_bound_usd": bound,
           "mask": mask, "out": out, "layers": layers or "0-47", "state": "submitted"}
    (HERE / f"{run_id}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
