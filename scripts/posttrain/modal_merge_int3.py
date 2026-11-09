"""Merge an RL LoRA adapter into the v8.5p INT3 build (plan (i)-(iv)). Modal CPU, on the t50 chain
(throughput arm, modal_t50.py) with v8.5p's own recipe: t37 all25 mask, masked W4 GPTQ (step-4 settings) on the all25
capture, INT3 g16 rtn-search requant.

  fit      per layer: W4 GPTQ shards of the BF16 experts plus the adapter's expert delta (``--zero``: without it)
  requant  per layer: INT3 g16 shards of a W4 overlay (``--push``: also to the Nebius bucket)
  survive  per layer: does the expert delta survive? MoE outputs on the eval capture (12,288 rows): BF16 W vs W + dW
           against INT3 zero vs INT3 merged (same pipeline) and against v8.5p's INT3
  dense    the adapter's dense LoRA merged into the checkpoint's BF16 shards + an INT3 expert overlay -> one serving
           overlay folder (arc-expert-overlay-v1), pushed to Nebius

PEFT 0.21.1 ParamWrapper (the trainer's): lora_A [r E, in] expert-major rows, lora_B [out, r E] expert-minor columns;
dW_e = scale * B[:, :, e] @ A[e] in the parameter's [E, out, in] layout (= the BF16 experts' gate_up [E, 2I, H] and
down [E, H, I]).

  modal run scripts/posttrain/modal_merge_int3.py --kind fit --run-id ID --out NAME [--zero] [--layers 2,13]

Read the result with merge_arm.py (bundles), modal_overlay_check.py (CPU weight check), merge_watch.py and merge_read.py.
Code from the dfr-exact (arena.expert_quant_gptq, expert_quant_math) and throughput-arm (arena.expert_3bit_overlay)
worktrees (ARC_MERGE_DFR / ARC_MERGE_TPA); launch records in docs/validation-rl/merge-<run id>-<kind>.json.
"""

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or ".").resolve()
DFR = Path(os.environ.get("ARC_MERGE_DFR") or ROOT / ".claude/worktrees/dfr-exact")
TPA = Path(os.environ.get("ARC_MERGE_TPA") or ROOT / ".claude/worktrees/throughput-arm")
MASK_FILE = HERE.parent.parent / "data/expert_prune/prune-t37-int3-requant-all25v71b-v1.json"  # v8.5p's (sha 43dd9b2f)
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
BF16_DIR = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
CAPTURE = "/runs/validation/b2-tp-quant-capture-all25v71b-48l-v1-nebius-20261010/expert-capture"
MASK = "/opt/masks/prune-t37-int3-requant-all25v71b-v1.json"
V85P = "/assets/flash-input/datasets/richardcsaky/arc-expert-int3-requant-t37-all25v71b-v1"
OUT_ROOT = "/assets/flash-input/datasets/richardcsaky"
E, H, INTER, TOPK, R = 512, 2560, 640, 10, 16
SETTINGS = {"percdamp": 0.01, "min_rows": 64, "blocksize": 128, "batch": 128, "scale_search": [1.0],
            "eval_rows": 4096, "calib_rows": 0, "prior_rows": 4096.0, "act_order": True, "omit_pruned": True}
SIZES = {"fit": (16, 96, 2400), "requant": (8, 32, 1500), "survive": (16, 96, 2400), "dense": (8, 64, 3600)}

app = modal.App("arc-rl-merge-int3", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
base_pkgs = ("torch==2.9.1", "safetensors", "numpy")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
         .pip_install(*base_pkgs, index_url="https://download.pytorch.org/whl/cpu", extra_index_url="https://pypi.org/simple")
         .add_local_dir(str(DFR / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_dir(str(TPA / "src" / "arena"), "/opt/tpa/arena", copy=True, ignore=["**/__pycache__/**"])
         .add_local_file(str(MASK_FILE), MASK, copy=True)
         .add_local_file(str(HERE / "merge_lora.py"), "/root/merge_lora.py", copy=True)
         .add_local_file(__file__, "/root/modal_merge_int3.py", copy=True))
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"


def rclone_env():
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")


def push(local, name):
    import subprocess

    rclone_env()
    subprocess.run(["rclone", "copyto", str(local), f"arc:{BUCKET}/assets/flash-input/datasets/richardcsaky/{name}",
                    "--s3-upload-concurrency", "8", "--s3-chunk-size", "64M", "--retries", "5"], check=True)


def kept_of(mask_path):
    from merge_lora import kept_of as kept

    return kept(json.load(open(mask_path)), E)


def expert_delta(adapter, layer, torch):
    from merge_lora import expert_delta as delta

    return delta(adapter, layer, torch)


def gptq_module():
    """arena.expert_quant_gptq with t50's CPU patch (no boolean-mask index ops on float8 in CPU torch)."""
    import sys
    import types

    sys.path.insert(0, "/opt/src")
    src = open("/opt/src/arena/expert_quant_gptq.py").read()
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
    return mod


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, timeout=SIZES["fit"][2])
def fit(layer: int, out_name: str, run_id: str, adapter: str):
    import torch

    torch.set_num_threads(SIZES["fit"][0])
    mod = gptq_module()
    info = {}

    class Fit(mod.Gptq):
        def bf16_layer(self, layer_, thread):
            gate_up, down = super().bf16_layer(layer_, thread)
            if adapter:
                d13, d2, scale = expert_delta(adapter, layer_, torch)
                w13, w2 = gate_up.float(), down.float()
                info.update(scale=scale, d13_over_w=float(d13.norm() / w13.norm()), d2_over_w=float(d2.norm() / w2.norm()))
                # GPTQ's target in float32: a BF16 cast would drop most of a small delta (round to nearest)
                info.update(bf16_cast_kept13=float(((w13 + d13).to(torch.bfloat16).float() - w13).norm() / d13.norm()),
                            bf16_cast_kept2=float(((w2 + d2).to(torch.bfloat16).float() - w2).norm() / d2.norm()))
                gate_up, down = w13 + d13, w2 + d2
                del d13, d2, w13, w2
            return gate_up, down

    overlay_out, work = f"{OUT_ROOT}/{out_name}", f"/runs/research/{run_id}"
    os.makedirs(overlay_out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    s = SETTINGS
    argv = ["--calib", CAPTURE + "/calib", "--capture", CAPTURE, "--model-dir", MODEL_DIR, "--bf16-dir", BF16_DIR,
            "--overlay-out", overlay_out, "--out", f"{work}/L{layer:02d}.json",
            "--deadline-seconds", str(SIZES["fit"][2] - 120), "--layers", str(layer), "--percdamp", str(s["percdamp"]),
            "--min-rows", str(s["min_rows"]), "--blocksize", str(s["blocksize"]), "--batch", str(s["batch"]),
            "--scale-search", *map(str, s["scale_search"]), "--eval-rows", str(s["eval_rows"]),
            "--prior-rows", str(s["prior_rows"]), "--act-order", "--prune-mask", MASK, "--omit-pruned", "--device", "cpu"]
    mod.Gptq = Fit
    t0 = time.time()
    try:
        mod.main(argv)
    finally:
        assets.commit()
        runs.commit()
    res = json.load(open(f"{work}/L{layer:02d}.json"))
    lay = res["layers"].get(str(layer), {})
    return {"layer": layer, "seconds": round(time.time() - t0, 1), "errors": res["errors"], "delta": info,
            "moe_errors": lay.get("errors"), "experts_gptq": lay.get("experts_gptq"), "shards": lay.get("shards"),
            "settings": res.get("settings")}


def _rq_worker(job):
    import sys

    sys.path.insert(0, "/opt/tpa")
    sys.path.insert(1, "/opt/src")
    import torch

    torch.set_num_threads(2)
    from arena import expert_3bit_overlay as ov

    source, out, group = job
    st = ov.convert_shard(torch, source, out, group, "cpu")
    rel = ov.verify_shard(torch, source, out, 0.35)
    return os.path.basename(out), st, rel


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, timeout=SIZES["requant"][2], secrets=[modal.Secret.from_name("nebius-s3")])
def requant(layer: int, src_name: str, out_name: str, do_push: bool):
    import hashlib
    import multiprocessing as mp

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
        with open(path, "rb") as fh:
            info[name] = {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(fh, "sha256").hexdigest(),
                          "base_bytes": path.stat().st_size, "stats": st, "verify": rel}
        if do_push:
            push(path, f"{out_name}/{name}")
    return {"layer": layer, "shards": info, "seconds": round(time.time() - t0, 1)}


def int3_experts(folder, layer, kept, torch):
    """{expert: (gate_up [2I, H], down [H, I]) float32} of one layer's overlay shards (kept experts only)."""
    import sys

    sys.path.insert(0, "/opt/src")
    from safetensors import safe_open

    from arena import expert_quant_math as qm

    fp8 = torch.float8_e4m3fn
    out = {}
    for path in sorted(Path(folder).glob(f"layer-{layer:05d}-experts-*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as fh:
            names = set(fh.keys())
            for e in kept:
                base = f"model.language_model.layers.{layer}.mlp.experts.{e}."
                if base + "gate_proj.weight" not in names:
                    continue

                def deq(p):
                    w = fh.get_tensor(base + p + ".weight")
                    sc = fh.get_tensor(base + p + ".weight_scale")
                    s2 = fh.get_tensor(base + p + ".weight_scale_2").float().reshape(1)
                    return qm.dequantize(w.view(torch.uint8).unsqueeze(0), sc.view(fp8).unsqueeze(0), s2)[0]
                out[e] = (torch.cat([deq("gate_proj"), deq("up_proj")]), deq("down_proj"))
    return out


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, timeout=SIZES["survive"][2])
def survive(layer: int, zero_name: str, merged_name: str, adapter: str, rows: int):
    """Expert-delta survival on the eval capture: the MoE effect of dW in BF16 (y(W+dW) - y(W)) against its INT3
    image (y(Q merged) - y(Q zero), the same pipeline) and against v8.5p (y(Q merged) - y(v8.5p))."""
    import sys

    import numpy as np
    import torch

    torch.set_num_threads(SIZES["survive"][0])
    sys.path.insert(0, "/opt/src")
    from safetensors import safe_open

    from arena import expert_quant_math as qm

    t0 = time.time()
    kept = kept_of(MASK)[layer]
    root = os.path.join(CAPTURE, "eval")
    n = min(rows, int(json.load(open(os.path.join(root, "meta.json")))["rows"][str(layer)]))
    x = torch.from_numpy(np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.int16, count=n * H).reshape(n, H)).view(torch.bfloat16).float()
    ids = torch.from_numpy(np.fromfile(os.path.join(root, f"L{layer:02d}.ids"), dtype=np.int16, count=n * TOPK).reshape(n, TOPK).astype(np.int64))
    wts = torch.from_numpy(np.fromfile(os.path.join(root, f"L{layer:02d}.w"), dtype=np.float32, count=n * TOPK).reshape(n, TOPK))
    ids = qm.remap_pruned_ids(ids, kept).long()
    index = json.load(open(os.path.join(BF16_DIR, "model.safetensors.index.json")))["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    bf = {}
    for p in ("gate_up_proj", "down_proj"):
        with safe_open(os.path.join(BF16_DIR, index[prefix + p]), framework="pt", device="cpu") as fh:
            bf[p] = fh.get_tensor(prefix + p).float()
    d13, d2, _ = expert_delta(adapter, layer, torch)
    models = {"zero": int3_experts(Path(OUT_ROOT) / zero_name, layer, kept, torch),
              "merged": int3_experts(Path(OUT_ROOT) / merged_name, layer, kept, torch),
              "v85p": int3_experts(V85P, layer, kept, torch)}
    used = sorted(set(ids.reshape(-1).tolist()))
    missing = {k: [e for e in used if e not in m] for k, m in models.items()}
    if any(missing.values()):
        raise RuntimeError(f"L{layer}: routed experts missing from an overlay: { {k: v[:5] for k, v in missing.items()} }")

    def moe(expert):
        out = torch.zeros(n, H)
        for e in used:
            rows_, slot = (ids == e).nonzero(as_tuple=True)
            gu, dn = expert(e)
            y = x[rows_] @ gu.t()
            h = torch.nn.functional.silu(y[:, :INTER]) * y[:, INTER:]
            out.index_add_(0, rows_, (h @ dn.t()) * wts[rows_, slot].unsqueeze(1))
        return out

    y_w = moe(lambda e: (bf["gate_up_proj"][e], bf["down_proj"][e]))
    y_wd = moe(lambda e: (bf["gate_up_proj"][e] + d13[e], bf["down_proj"][e] + d2[e]))
    y = {k: moe(lambda e, m=m: m[e]) for k, m in models.items()}

    def cos(a, b):
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))

    true = y_wd - y_w
    q_same, q_v85 = y["merged"] - y["zero"], y["merged"] - y["v85p"]
    return {"layer": layer, "rows": n, "experts_routed": len(used), "seconds": round(time.time() - t0, 1),
            "delta_rel": float(true.norm() / y_w.norm()),
            "quant_noise_rel": {k: float((y[k] - (y_wd if k == "merged" else y_w)).norm() / y_w.norm()) for k in y},
            "survival_same_pipeline": {"cos": cos(q_same, true), "norm_ratio": float(q_same.norm() / true.norm())},
            "survival_vs_v85p": {"cos": cos(q_v85, true), "norm_ratio": float(q_v85.norm() / true.norm())},
            "zero_vs_v85p_rel": float((y["zero"] - y["v85p"]).norm() / y_w.norm()),
            "delta_weight_rel": {"gate_up": float(d13.norm() / bf["gate_up_proj"].norm()),
                                 "down": float(d2.norm() / bf["down_proj"].norm())}}


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, timeout=SIZES["dense"][2], secrets=[modal.Secret.from_name("nebius-s3")])
def dense(adapter: str, experts_name: str, out_name: str):
    """Dense LoRA (attention, linear attention, shared experts) merged into the checkpoint's BF16 shards in fp32, the
    expert overlay's INT3 shards beside them, one manifest; every merged tensor checked BF16 and named in the index."""
    import hashlib
    import re
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    t0 = time.time()
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    pairs = {}
    for key in lora:
        if ".lora_A." in key and ".mlp.experts." not in key:
            name = key.split("base_model.model.", 1)[1].replace(".lora_A.weight", ".weight")
            pairs[name] = (lora[key].float(), lora[key.replace(".lora_A.", ".lora_B.")].float())
    del lora
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    unknown = [n for n in pairs if n not in index]
    if unknown or len(pairs) != 372:
        raise RuntimeError(f"{len(pairs)} dense pairs, not in the checkpoint: {unknown[:3]}")
    out_dir = Path(OUT_ROOT) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, shards = {}, {}
    for shard in sorted({index[n] for n in pairs}):
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as fh:
            metadata, tensors = fh.metadata(), {k: fh.get_tensor(k) for k in fh.keys()}
        for name in [n for n in tensors if n in pairs]:
            w = tensors[name]
            if w.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} is {w.dtype}")
            a, b = pairs[name]
            delta = scale * (b @ a)
            merged = (w.float() + delta).to(torch.bfloat16)
            group = re.sub(r"layers\.\d+\.", "", name).split(".weight")[0].split("language_model.")[-1]
            g = stats.setdefault(group, {"tensors": 0, "d2": 0.0, "w2": 0.0, "r2": 0.0, "dot": 0.0})
            real = merged.float() - w.float()
            g["tensors"] += 1
            g["d2"] += float(delta.pow(2).sum())
            g["w2"] += float(w.float().pow(2).sum())
            g["r2"] += float(real.pow(2).sum())
            g["dot"] += float((real * delta).sum())
            tensors[name] = merged
        save_file(tensors, str(out_dir / shard), metadata=metadata)
        with open(out_dir / shard, "rb") as fh:
            shards[shard] = {"bytes": (out_dir / shard).stat().st_size, "sha256": hashlib.file_digest(fh, "sha256").hexdigest(),
                             "base_bytes": os.path.getsize(os.path.join(MODEL_DIR, shard))}
        del tensors
    merge = {k: {"tensors": g["tensors"], "delta_over_weight": round((g["d2"] / g["w2"]) ** 0.5, 6),
                 "realized_over_delta": round((g["r2"] / max(g["d2"], 1e-30)) ** 0.5, 4),
                 "realized_cosine": round(g["dot"] / max((g["r2"] * g["d2"]) ** 0.5, 1e-30), 4)} for k, g in stats.items()}
    experts = json.loads((Path(OUT_ROOT) / experts_name / "overlay-manifest.json").read_text())
    for name in experts["shards"]:
        shutil.copyfile(Path(OUT_ROOT) / experts_name / name, out_dir / name)
    manifest = {**experts, "shards": {**experts["shards"], **shards},
                "adapter": {"path": adapter, "scale": scale, "dense_merge": merge, "experts_from": experts_name}}
    raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
    (out_dir / "overlay-manifest.json").write_bytes(raw)
    assets.commit()
    for name in sorted(manifest["shards"]) + ["overlay-manifest.json"]:
        push(out_dir / name, f"{out_name}/{name}")
    return {"dense_shards": len(shards), "expert_shards": len(experts["shards"]), "merge": merge,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "seconds": round(time.time() - t0, 1)}


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, timeout=6000,
              secrets=[modal.Secret.from_name("nebius-s3")])
def driver(kind: str, run_id: str, out_name: str, layers: str, adapter: str, src_name: str, do_push: bool,
           zero_name: str, rows: int):
    import hashlib

    work = Path(f"/runs/research/{run_id}")
    work.mkdir(parents=True, exist_ok=True)
    todo = [int(v) for v in layers.split(",")] if layers else list(range(48))
    if kind == "fit":
        calls = fit.starmap([(l, out_name, run_id, adapter) for l in todo], return_exceptions=True)
    elif kind == "requant":
        calls = requant.starmap([(l, src_name, out_name, do_push) for l in todo], return_exceptions=True)
    elif kind == "survive":
        calls = survive.starmap([(l, zero_name, src_name, adapter, rows) for l in todo], return_exceptions=True)
    else:
        calls = [dense.remote(adapter, src_name, out_name)]
        todo = [-1]
    results, errors = [], []
    for layer, r in zip(todo, calls):
        if isinstance(r, Exception):
            errors.append(f"L{layer}: {type(r).__name__}: {str(r)[:500]}")
        else:
            results.append(r)
            if kind == "fit" and r.get("errors"):
                errors.append(f"L{layer}: {r['errors']}")
    record = {"run_id": run_id, "kind": kind, "out": out_name, "src": src_name, "adapter": adapter, "layers": todo,
              "errors": errors, "results": results, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if kind in ("fit", "requant") and not errors and len(results) == 48:
        assets.reload()
        shards = {}
        for r in results:
            for name, v in (r["shards"] or {}).items():
                shards[name] = {k: v[k] for k in ("bytes", "sha256", "base_bytes")} if kind == "requant" else v
        raw_mask = open(MASK, "rb").read()
        manifest = {"format": "arc-expert-overlay-v1", "base_model_dir": MODEL_DIR, "layers": list(range(48)),
                    "shards": shards, "complete": len(shards) == 192, "omits_pruned": True,
                    "expert_prune_sha256": hashlib.sha256(raw_mask).hexdigest(), "utc": record["utc"],
                    "adapter": adapter or None}
        if kind == "fit":
            manifest.update(settings=results[0]["settings"], capture=CAPTURE)
        else:
            manifest["settings"] = {"grid": "int3", "group": 16, "method": "rtn-search",
                                    "search": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75], "source_dir": f"{OUT_ROOT}/{src_name}"}
        raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
        (Path(OUT_ROOT) / out_name / "overlay-manifest.json").write_bytes(raw)
        assets.commit()
        record["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        if kind == "requant" and do_push:
            push(Path(OUT_ROOT) / out_name / "overlay-manifest.json", f"{out_name}/overlay-manifest.json")
    (work / f"{kind}-result.json").write_text(json.dumps(record, indent=1))
    runs.commit()
    return {"errors": errors, "n": len(results)}


@app.local_entrypoint()
def main(kind: str, run_id: str, out: str = "", layers: str = "", adapter: str = "/runs/validation/rl1-xfer-l4/adapter-r09",
         zero: bool = False, src: str = "", push_out: bool = False, zero_name: str = "", rows: int = 12288):
    call = driver.spawn(kind, run_id, out, layers, "" if zero else adapter, src, push_out, zero_name, rows)
    rec = {"run_id": run_id, "kind": kind, "function_call_id": call.object_id, "out": out, "src": src,
           "layers": layers or "0-47", "adapter": None if zero else adapter, "state": "submitted"}
    (ROOT / "docs/validation-rl" / f"merge-{run_id}-{kind}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
