"""Held-out kf runs: INT3 GPTQ-ARC requant refit on the 15 RL training games.
pull: the Nebius capture (expert-quant capture_only + persist) -> this run's Modal volume; fit: masked W4 GPTQ per
layer (arena.expert_quant_gptq, the 13 Oct / l256 settings; routing: the capture's own, or recomputed under the mask); requant: INT3
g16 rtn-search of the W4 shards (arena.expert_3bit_overlay), verified, pushed to the Nebius asset bucket.
Derived from scripts/arc2_weights/modal_l256.py (arc2). No cpu=/memory= reservations. One run id, one volume.

  HK_RUN=<run> modal run --detach modal_heldout_refit.py --capture-run <nebius capture run> --mask <mask file>
      --w4 <W4 overlay name> --out <INT3 overlay name> [--stage all|pull|fit|requant] [--layers 0,1] [--routing native|reroute]
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DFR = HERE.parents[2] if len(HERE.parents) > 2 else HERE  # /root in the container (images already built)
TPA_SRC = os.environ.get("HK_TPA_SRC", "")  # throughput-arm src/ (expert_3bit_overlay as used by every INT3 build)
RUN = os.environ.get("HK_RUN", "unset")
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
BF16_DIR = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
OUT_ROOT = "/assets/flash-input/datasets/richardcsaky"
E, H, TOPK, INTER = 512, 2560, 10, 640
SETTINGS = {"percdamp": 0.01, "min_rows": 64, "blocksize": 128, "batch": 128, "scale_search": [1.0],
            "eval_rows": 4096, "calib_rows": 0, "prior_rows": 4096.0, "act_order": True, "omit_pruned": True}
THREADS = {"fit": 16, "requant": 8}
TIMEOUT = {"fit": 2400, "requant": 3000}  # requant: 1,500 s timed out one layer (28 Oct, b-v1 L7)
MAX_CONTAINERS = 48  # one per layer; workspace ceiling 80 checked at launch
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"

app = modal.App(f"tp-heldout-{RUN}", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name(f"arc-run-{RUN}", create_if_missing=True, version=2)
prolong = modal.Volume.from_name("arc-prolong-runs")
base = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
        .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                     extra_index_url="https://pypi.org/simple")
        .env({"HK_RUN": RUN})
        .add_local_dir(str(HERE / "masks"), "/opt/masks", copy=True))
image = (base.add_local_dir(str(DFR / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_file(__file__, "/root/modal_heldout_refit.py", copy=True))
rq_image = (base.add_local_dir(TPA_SRC or str(DFR / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
            .add_local_file(__file__, "/root/modal_heldout_refit.py", copy=True))
secrets = [modal.Secret.from_name("nebius-s3")]


def rclone_env():
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")


def capture_dir(capture_run):
    """The capture on the shared runs volume (Nebius runs synced there) if present, else this run's pulled copy."""
    synced = f"/prolong/validation/{capture_run}/expert-capture"
    return synced if os.path.isdir(synced) else f"/runs/capture/{capture_run}/expert-capture"


def kept_of(mask_path):
    mask = json.load(open(mask_path))
    return {int(l): sorted(set(range(E)) - {int(e) for e in ids}) for l, ids in mask["pruned"].items()}


def load_split(root, split, layer, torch):
    import numpy as np

    root = os.path.join(root, split)
    meta = json.load(open(os.path.join(root, "meta.json")))
    rows = int(meta["rows"][str(layer)])
    if meta.get("format") == "fp8":
        q = np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.uint8).reshape(rows, H)
        s = np.fromfile(os.path.join(root, f"L{layer:02d}.s"), dtype=np.float32)
        x = (torch.from_numpy(q).view(torch.float8_e4m3fn).float() * torch.from_numpy(s).unsqueeze(1)).to(torch.bfloat16)
    else:
        x = torch.from_numpy(np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.int16).reshape(rows, H)).view(
            torch.bfloat16)
    return x


def router(layer, torch):
    from safetensors import safe_open

    name = f"model.language_model.layers.{layer}.mlp.gate.weight"
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(MODEL_DIR, index[name]), framework="pt", device="cpu") as fh:
        return fh.get_tensor(name).float()


def route_rows(x, gate, kept, torch, chunk=32768):
    """Top-10 over the kept experts, renormalised; ids in kept ranks (as a load-time-pruned server routes)."""
    ids, wts = [], []
    g = gate[kept]
    for s in range(0, x.shape[0], chunk):
        top, idx = (x[s:s + chunk].float() @ g.t()).topk(TOPK, dim=-1)
        ids.append(idx)
        wts.append(torch.softmax(top, dim=-1))
    return torch.cat(ids), torch.cat(wts)


@app.function(image=image, volumes={"/assets": assets, "/runs": runs, "/prolong": prolong}, timeout=TIMEOUT["fit"],
              max_containers=MAX_CONTAINERS, cpu=(0.125, 16), memory=(1024, 131072))
def fit(layer: int, mask_name: str, w4_name: str, capture_run: str, routing: str = "native", masters: str = "",
        mode: str = "gptq"):
    """masters: a BF16 student checkpoint dir (p75: fused [kept, 2I, H] / [kept, H, I] experts, kept ascending)
    quantized instead of the base experts, each kept expert with its own ModelOpt weight_scale_2; mode rtn: RTN
    of the masters (no capture read)."""
    import sys
    import types

    import torch

    torch.set_num_threads(THREADS["fit"])
    sys.path.insert(0, "/opt/src")
    src = open("/opt/src/arena/expert_quant_gptq.py").read()
    # CPU torch has no boolean-mask index ops on float8: the fallback copies go through uint8 views (arc2 l256).
    for a, b in (('p13[fb], f13[fb] = served["w13"][fb], served["w13_sf"][fb]',
                  'p13[fb] = served["w13"][fb]; f13.view(self.torch.uint8)[fb] = served["w13_sf"].view(self.torch.uint8)[fb]'),
                 ('p2[fb], f2[fb] = served["w2"][fb], served["w2_sf"][fb]',
                  'p2[fb] = served["w2"][fb]; f2.view(self.torch.uint8)[fb] = served["w2_sf"].view(self.torch.uint8)[fb]')):
        if src.count(a) != 1:
            raise RuntimeError(f"patch anchor missing: {a}")
        src = src.replace(a, b)
    ws2_old = ("                    else:  # weight_scale_2, input_scale: unchanged\n"
               "                        value = original\n")
    ws2_new = ("                    elif kind == \"weight_scale_2\" and \"masters_scale2\" in new:\n"
               "                        value = self.torch.tensor(float({\"gate_proj\": new[\"s2_gate\"], \"up_proj\": "
               "new[\"s2_up\"], \"down_proj\": new[\"s2_down\"]}[proj][e]), dtype=original.dtype).reshape(original.shape)\n"
               + ws2_old)
    if src.count(ws2_old) != 1:
        raise RuntimeError("patch anchor missing: weight_scale_2")
    src = src.replace(ws2_old, ws2_new)
    mod = types.ModuleType("arena.expert_quant_gptq")
    mod.__file__ = "/opt/src/arena/expert_quant_gptq.py"
    sys.modules["arena.expert_quant_gptq"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mask_path = f"/opt/masks/{mask_name}"
    kept = kept_of(mask_path)[layer]
    cap = capture_dir(capture_run)
    rerouted = {}

    def reroute(split):
        if split not in rerouted:
            x = load_split(cap, split, layer, torch)
            ids, w = route_rows(x, router(layer, torch), kept, torch)
            rerouted[split] = (x, ids.to(torch.int32), w.float())
        return rerouted[split]

    class Fit(mod.Gptq):  # native: the capture server already routed under this mask (its ids, kept ranks)
        if routing == "reroute":
            def load_calib(self, layer_):
                return reroute("calib")

            def load_capture(self, split, layer_):
                return reroute(split)
        if masters:
            def prefetch_bf16(self, layer_):
                return None

            def bf16_layer(self, layer_, thread):
                return student[0].to(self.dev), student[1].to(self.dev)

            def load_served(self, layer_):
                served = super().load_served(layer_)
                k = torch.tensor(kept)
                s13 = self.qm.scale2_of(student[0][:, :INTER], student[0][:, INTER:])
                s2d = self.qm.scale2_of(student[1])
                for key, val in (("s2_gate", s13), ("s2_up", s13), ("s2_down", s2d)):
                    served[key] = served[key].clone()
                    # the base scale_2 where it covers the student's amax (base gate_up sits ~2.7x above the
                    # ModelOpt rule, 28 Oct preflight), else the student's own ModelOpt rule: no clipping
                    served[key][k] = torch.maximum(served[key][k].float(), val[k].float()).to(served[key].dtype)
                served["s2_up"] = served["s2_gate"].clone()  # GPTQ quantizes gate and up with s2_gate
                served["masters_scale2"] = torch.ones(1)
                return served
        if mode == "rtn":
            def layer(self, layer_, thread):
                qm, u8 = self.qm, torch.uint8
                t0_ = time.time()
                served = self.load_served(layer_)
                gu, dn = self.bf16_layer(layer_, thread)
                k = torch.tensor(kept)
                new = dict(served)
                for wkey, skey, s2key, w in (("w13", "w13_sf", "s2_gate", gu), ("w2", "w2_sf", "s2_down", dn)):
                    q, sc = qm.rtn(w[k], served[s2key][k])
                    packed, scales = served[wkey].clone(), served[skey].clone()
                    packed[k] = q
                    scales.view(u8)[k] = sc.view(u8)
                    new[wkey], new[skey] = packed, scales
                res = {"layer": layer_, "experts_rtn": len(kept), "errors": None}
                self.results["layers"][str(layer_)] = res
                res["shards"] = self.write_shards(layer_, new)
                res["seconds"] = round(time.time() - t0_, 1)

    student = None
    if masters:
        from safetensors import safe_open

        index = json.load(open(f"{masters}/model.safetensors.index.json"))["weight_map"]
        pre = f"model.language_model.layers.{layer}.mlp.experts."
        got = {}
        if pre + "gate_up_proj" in index:
            for proj in ("gate_up_proj", "down_proj"):
                with safe_open(f"{masters}/{index[pre + proj]}", framework="pt", device="cpu") as fh:
                    got[proj] = fh.get_tensor(pre + proj)
            gu, dn = got["gate_up_proj"], got["down_proj"]
        else:  # trainer checkpoints split by rank: experts.<kept position>.{gate,up,down}_proj.weight, gate first in w13
            def one(name):
                with safe_open(f"{masters}/{index[pre + name]}", framework="pt", device="cpu") as fh:
                    return fh.get_tensor(pre + name)

            gu = torch.stack([torch.cat([one(f"{e}.gate_proj.weight"), one(f"{e}.up_proj.weight")], 0)
                              for e in range(len(kept))])
            dn = torch.stack([one(f"{e}.down_proj.weight") for e in range(len(kept))])
        if tuple(gu.shape) != (len(kept), 2 * INTER, H) or tuple(dn.shape) != (len(kept), H, INTER):
            raise RuntimeError(f"L{layer} masters: {tuple(gu.shape)} {tuple(dn.shape)} for {len(kept)} kept")
        k = torch.tensor(kept)
        student = (torch.zeros((E, 2 * INTER, H), dtype=gu.dtype), torch.zeros((E, H, INTER), dtype=dn.dtype))
        student[0][k], student[1][k] = gu, dn
        del got, gu, dn

    overlay_out = f"{OUT_ROOT}/{w4_name}"
    work = f"/runs/work/fit"
    os.makedirs(overlay_out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    s = SETTINGS
    argv = ["--calib", cap + "/calib", "--capture", cap, "--model-dir", MODEL_DIR, "--bf16-dir", BF16_DIR,
            "--overlay-out", overlay_out, "--out", f"{work}/L{layer:02d}.json",
            "--deadline-seconds", str(TIMEOUT["fit"] - 120), "--layers", str(layer), "--percdamp", str(s["percdamp"]),
            "--min-rows", str(s["min_rows"]), "--blocksize", str(s["blocksize"]), "--batch", str(s["batch"]),
            "--scale-search", *map(str, s["scale_search"]), "--eval-rows", str(s["eval_rows"]),
            "--prior-rows", str(s["prior_rows"]), "--act-order", "--prune-mask", mask_path, "--omit-pruned",
            "--device", "cpu"]
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
            "rows_per_expert": lay.get("rows_per_expert"), "shards": lay.get("shards"), "settings": res.get("settings")}


def _rq_worker(job):
    import sys

    sys.path.insert(0, "/opt/src")
    import torch

    torch.set_num_threads(2)
    from arena import expert_3bit_overlay as ov

    source, out, group = job
    st = None
    if not (os.path.exists(out) and os.path.getsize(out) == os.path.getsize(source)):
        st = ov.convert_shard(torch, source, out, group, "cpu")  # else a finished shard of an earlier attempt
    rel = ov.verify_shard(torch, source, out, 0.35)
    return os.path.basename(out), st, rel


@app.function(image=rq_image, volumes={"/assets": assets, "/runs": runs, "/prolong": prolong}, timeout=TIMEOUT["requant"],
              secrets=secrets, max_containers=MAX_CONTAINERS, cpu=(0.125, 8), memory=(1024, 65536))
def requant(layer: int, w4_name: str, out_name: str):
    """INT3 g16 requant (rtn-search, the 7 Oct recipe) of one layer's W4 shards, verified, pushed to Nebius."""
    import hashlib
    import multiprocessing as mp

    t0 = time.time()
    src_dir, out_dir = Path(OUT_ROOT) / w4_name, Path(OUT_ROOT) / out_name
    done = Path(f"/runs/work/rq-{out_name}/L{layer:02d}.json")  # a rerun redoes only the layers not finished
    if done.exists():
        return json.loads(done.read_text())
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
    res = {"layer": layer, "shards": info, "seconds": round(time.time() - t0, 1)}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(res))
    runs.commit()
    return res


def write_manifest(out_name, manifest, dense):
    """overlay-manifest.json of out_name; dense: an /assets dir of whole non-expert replacement shards (A's p75 dense
    shards) copied in and listed beside the expert shards. Returns (manifest sha256, files {name: size, sha256})."""
    import hashlib
    import shutil

    out_dir = Path(OUT_ROOT) / out_name
    files = {n: {"size": v["bytes"], "sha256": v["sha256"]} for n, v in manifest["shards"].items()}
    if dense:
        src = Path("/assets") / dense
        names = sorted(q.name for q in src.glob("*.safetensors"))
        if not names or any("-experts-" in n for n in names):
            raise RuntimeError(f"dense dir {src}: {names[:5]}")
        for n in names:
            shutil.copyfile(src / n, out_dir / n)
            with open(out_dir / n, "rb") as fh:
                digest = hashlib.file_digest(fh, "sha256").hexdigest()
            size = (out_dir / n).stat().st_size
            manifest["shards"][n] = {"bytes": size, "sha256": digest, "base_bytes": size}
            files[n] = {"size": size, "sha256": digest}
        manifest["dense"] = {"source": str(src), "shards": names}
    raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
    (out_dir / "overlay-manifest.json").write_bytes(raw)
    assets.commit()
    return hashlib.sha256(raw).hexdigest(), files


def publish(out_name, files):
    """The whole overlay dir to the Nebius bucket (size-only skip of shards already pushed) + the asset manifest the
    Nebius runner fetches by (modal_int3_fit mirror format)."""
    import hashlib

    out_dir = Path(OUT_ROOT) / out_name
    for q in out_dir.glob("*.json"):
        files[q.name] = {"size": q.stat().st_size, "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
    on_disk = {q.name for q in out_dir.iterdir()}
    if set(files) != on_disk:
        raise RuntimeError(f"{out_name}: files vs disk differ: {sorted(set(files) ^ on_disk)[:5]}")
    rclone_env()
    prefix = f"flash-input/datasets/richardcsaky/{out_name}"
    subprocess.run(["rclone", "copy", str(out_dir), f"arc:{BUCKET}/assets/{prefix}", "--size-only", "--transfers", "16",
                    "--s3-chunk-size", "64M", "--retries", "5"], check=True)
    for q in out_dir.glob("*.json"):  # rewritten manifests may keep their size: always copied
        subprocess.run(["rclone", "copyto", str(q), f"arc:{BUCKET}/assets/{prefix}/{q.name}"], check=True)
    slug = prefix.replace("/", "--") + ".json"
    Path(f"/tmp/{slug}").write_text(json.dumps({"prefix": prefix, "files": files}))
    subprocess.run(["rclone", "copyto", f"/tmp/{slug}", f"arc:{BUCKET}/assets-manifests/{slug}"], check=True)
    return slug


@app.function(image=rq_image, volumes={"/assets": assets, "/runs": runs, "/prolong": prolong}, timeout=7200, secrets=secrets)
def driver(stage: str, capture_run: str, mask_name: str, w4_name: str, out_name: str, layers: str = "",
           routing: str = "native", masters: str = "", dense: str = ""):
    import hashlib

    work = Path("/runs/work")
    work.mkdir(parents=True, exist_ok=True)
    todo = [int(v) for v in layers.split(",")] if layers else list(range(48))
    record = {"run_id": RUN, "stage": stage, "capture_run": capture_run, "mask": mask_name, "w4": w4_name,
              "out": out_name, "layers": todo, "routing": routing, "masters": masters, "dense": dense, "errors": [], "steps": {}}

    def save():
        (work / f"result-{stage}.json").write_text(json.dumps(record, indent=1))
        runs.commit()

    if stage in ("all", "pull"):
        t0 = time.time()
        rclone_env()
        dst = capture_dir(capture_run)
        if not dst.startswith("/prolong/"):
            os.makedirs(dst, exist_ok=True)
            subprocess.run(["rclone", "copy", f"arc:{BUCKET}/runs/validation/{capture_run}/expert-capture", dst,
                            "--transfers", "16", "--s3-chunk-size", "64M", "--retries", "5"], check=True)
            runs.commit()
        record["capture_dir"] = dst
        metas = {s: json.load(open(f"{dst}/{s}/meta.json")) for s in os.listdir(dst) if os.path.exists(f"{dst}/{s}/meta.json")}
        if "calib" not in metas:
            raise RuntimeError(f"no calib split in {dst}: {sorted(metas)}")
        record["steps"]["pull"] = {"seconds": round(time.time() - t0, 1),
                                   "rows": {s: m["rows"].get("0") for s, m in metas.items()}}
        save()
    if stage in ("all", "fit", "rtn"):
        t0 = time.time()
        mode = "rtn" if stage == "rtn" else "gptq"
        out = list(fit.starmap([(l, mask_name, w4_name, capture_run, routing, masters, mode) for l in todo],
                               return_exceptions=True))
        res = [r for r in out if not isinstance(r, Exception)]
        record["errors"] += [f"fit L{l}: {type(r).__name__}: {str(r)[:300]}" for l, r in zip(todo, out)
                             if isinstance(r, Exception)] + [f"fit L{r['layer']}: {r['errors']}" for r in res if r["errors"]]
        record["steps"]["fit"] = {"seconds": round(time.time() - t0, 1), "layers": len(res),
                                  "moe_errors": {r["layer"]: r["moe_errors"] for r in res}}
        if not record["errors"] and len(res) == len(todo):  # a partial run (preflight) writes a partial manifest
            assets.reload()
            shards = {}
            for r in res:
                shards.update(r["shards"])
            raw = open(f"/opt/masks/{mask_name}", "rb").read()
            manifest = {"format": "arc-expert-overlay-v1", "base_model_dir": MODEL_DIR, "layers": sorted(todo),
                        "shards": shards, "settings": res[0]["settings"], "complete": len(shards) == 192,
                        "expert_prune_sha256": hashlib.sha256(raw).hexdigest(), "omits_pruned": True,
                        "capture": None if mode == "rtn" else capture_dir(capture_run), "routing": routing,
                        "masters": masters or None, "mode": mode}
            if mode == "rtn":  # a served NVFP4 stand-in (A's capture): dense shards merged, published
                record["w4_manifest_sha256"], files = write_manifest(w4_name, manifest, dense)
                record["w4_assets_manifest"] = publish(w4_name, files)
            else:
                Path(f"{OUT_ROOT}/{w4_name}/overlay-manifest.json").write_text(
                    json.dumps(manifest, indent=1, sort_keys=True))
                assets.commit()
        save()
        if record["errors"]:
            return record
    if stage == "merge":  # dense shards into an existing overlay (w4 name), then publish
        assets.reload()
        manifest = json.loads((Path(OUT_ROOT) / w4_name / "overlay-manifest.json").read_text())
        if manifest.get("dense"):
            raise RuntimeError(f"{w4_name} already has dense shards")
        record["manifest_sha256"], files = write_manifest(w4_name, manifest, dense)
        record["assets_manifest"] = publish(w4_name, files)
        save()
        return record
    if stage in ("all", "requant"):
        t0 = time.time()
        out = list(requant.starmap([(l, w4_name, out_name) for l in todo], return_exceptions=True))
        res = [r for r in out if not isinstance(r, Exception)]
        record["errors"] += [f"requant L{l}: {type(r).__name__}: {str(r)[:300]}" for l, r in zip(todo, out)
                             if isinstance(r, Exception)]
        record["steps"]["requant"] = {"seconds": round(time.time() - t0, 1), "layers": len(res)}
        if not record["errors"] and len(res) == len(todo):  # a partial run (preflight) writes a partial manifest
            assets.reload()
            src_manifest = json.loads((Path(OUT_ROOT) / w4_name / "overlay-manifest.json").read_text())
            shards, stats, verify = {}, {}, {}
            for r in res:
                for name, v in r["shards"].items():
                    shards[name] = {k: v[k] for k in ("bytes", "sha256", "base_bytes")}
                    stats[name], verify[name] = v["stats"], v["verify"]
            manifest = {"format": "arc-expert-overlay-v1", "base_model_dir": MODEL_DIR, "layers": sorted(todo),
                        "shards": shards, "complete": len(shards) == 192,
                        "settings": {"grid": "int3", "group": 16, "method": "rtn-search",
                                     "search": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75], "source_dir": f"{OUT_ROOT}/{w4_name}"},
                        "omits_pruned": True, "expert_prune_sha256": src_manifest["expert_prune_sha256"]}
            if src_manifest.get("masters"):
                manifest["masters"] = src_manifest["masters"]
            out_dir = Path(OUT_ROOT) / out_name
            (out_dir / "build-stats.json").write_text(json.dumps(stats, indent=1))
            (out_dir / "verify.json").write_text(json.dumps(verify, indent=1, sort_keys=True))
            record["manifest_sha256"], files = write_manifest(out_name, manifest, dense)
            record["assets_manifest"] = publish(out_name, files)
        save()
    return record


@app.local_entrypoint()
def main(capture_run: str, mask: str, w4: str, out: str, stage: str = "all", layers: str = "", routing: str = "native",
         masters: str = "", dense: str = ""):
    call = driver.spawn(stage, capture_run, mask, w4, out, layers, routing, masters, dense)
    rec = {"run_id": RUN, "stage": stage, "function_call_id": call.object_id, "capture_run": capture_run, "mask": mask,
           "w4": w4, "out": out, "layers": layers or "0-47", "volume": f"arc-run-{RUN}", "masters": masters,
           "dense": dense}
    print(json.dumps(rec))
