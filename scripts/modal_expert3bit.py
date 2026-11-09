"""3-bit expert research pilots on Modal (30 September; docs/validation-20260930/expert-3bit.md).

``bench``: the kernel micro-benchmark (``arena.expert_3bit_bench``) in the pinned serving runtime, extracted from the
assets volume without a server (fast layer check, the runtime's own whiteouts), on one RTX PRO 6000.

The ledger is the main checkout's (``ARC_VALIDATION_ROOT``); launch records go to this checkout's
``docs/validation-20260930``. Research pilots count against the four research-GPU reservations.

    ARC_VALIDATION_ROOT=. modal run --detach scripts/modal_expert3bit.py --kind bench \
        --run-id research-expert3bit-bench-v1-20260930
"""

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or HERE).resolve()
RECORDS = HERE / "docs/validation-20260930"
TIMEOUTS = {"bench": 1200,  # v1 (2,400 s) was stopped for the 9 h runs; v2 is trimmed
            "splitbench": 1500,  # 7 October: arena.expert_split_bench (hot uint3 + cold uint2), no local capture
            "overlay": 3600,  # CPU only: the int3 overlay of 192 shards, 4 worker processes
            "cpustudy": 3600,
            "moeeval": 3600,  # CPU only: MoE output error of finished overlays on persisted eval captures
            "overlaydiff": 900}  # CPU only: rebuild one present overlay shard twice and diff it (determinism)  # CPU only: the 3-bit GPTQ-ARC quality screen, one container per layer
KIND = os.environ.get("ARC_EXPERT3BIT_KIND", "bench")
if os.environ.get("ARC_EXPERT3BIT_TIMEOUT"):  # a shorter limit (and bound) for a job of known length
    TIMEOUTS[KIND] = min(TIMEOUTS[KIND], int(os.environ["ARC_EXPERT3BIT_TIMEOUT"]))
TIMEOUT = TIMEOUTS[KIND]
GPU = KIND not in ("overlay", "cpustudy", "overlaydiff", "moeeval")
STUDY_LAYERS = (20, 40)
CPU, RAM_GIB = (8, 96) if GPU else (32, 64)
OVERLAY_ROOT = "/assets/flash-input/datasets/richardcsaky"
GPTQ_ARC_V1 = f"{OVERLAY_ROOT}/arc-expert-gptq-arc-v1"
STORAGE_USD = 0.5  # 63.3 GiB on the assets volume for about a week
RUNTIME_DIR = "/assets/flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
RT = "/tmp/qwen38-flash-next-vllm-runtime"
WHITEOUTS = {8: ("usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88",),
             17: ("tmp/.wh.requirements-cuda.txt", "tmp/.wh.common.txt")}

app = modal.App("arc-research-expert3bit-20260930", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .add_local_dir(str(HERE / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_dir(str(HERE / "runs/expert-3bit/capture") if KIND == "bench" else str(HERE / "src/arena"), "/opt/capture",
                        copy=True)  # the capture is local to the bench's worktree; CPU kinds do not build this image
         .add_local_file(__file__, "/root/modal_expert3bit.py", copy=True))
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                          extra_index_url="https://pypi.org/simple")
             .add_local_dir(str(HERE / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
             .add_local_dir(str(HERE / "scripts"), "/opt/scripts", copy=True, ignore=["**/__pycache__/**"])
             .add_local_file(__file__, "/root/modal_expert3bit.py", copy=True))


def reserve(run_id, kind, storage_usd=STORAGE_USD, authorization=None):
    import fcntl

    path = ROOT / "docs/spend.json"
    rate = (0.000842 if GPU else 0) + CPU * 0.0000131 + RAM_GIB * 0.00000222
    bound = (TIMEOUTS[kind] + 60) * rate * 1.15 * (len(STUDY_LAYERS) if kind == "cpustudy" else 1) \
        + (storage_usd if kind == "overlay" else 0)
    with path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        if any(r["run_id"] == run_id for r in d["runs"]):
            raise ValueError("Already reserved")
        active = sum(r.get("kind") == "research-pilot" and r.get("state") == "reserved" for r in d["runs"])
        if GPU and active >= 3:  # at most three pilot GPUs; the 9 h runs have priority
            raise RuntimeError("Three research pilot reservations active; reconcile before another")
        used = d["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in d["runs"])
        if used + bound > d["cap_usd"]:
            raise RuntimeError(f"Cap: {used}+{bound}>{d['cap_usd']}")
        d["runs"].append({"run_id": run_id, "state": "reserved",
                          "kind": "research-pilot" if GPU else "research-preparation",
                          "charged_bound_usd": bound, "max_seconds": TIMEOUTS[kind],
                          "gpu": "RTX-PRO-6000" if GPU else None,
                          "cpu_ceiling": CPU, "memory_ceiling_gib": RAM_GIB,
                          "authorization": "3-bit expert investigation;, <=3 pilot "
                                           "GPUs, each pilot <=1 hour including setup" if GPU else
                                           (authorization or
                                            "3-bit expert investigation; CPU-only overlay build"),
                          "research_kind": f"expert3bit-{kind}"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()
    return bound


def extract_runtime():
    import shutil
    import subprocess
    import time

    started = time.time()
    root = Path(RT)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    manifest = json.loads((Path(RUNTIME_DIR) / "runtime-manifest.json").read_text())
    for row in sorted(manifest["selected_layers"], key=lambda r: int(r["index"])):
        layer = Path(RUNTIME_DIR) / row["file"]
        if not layer.is_file():
            layer = Path(RUNTIME_DIR) / f"{row['file']}.blob"
        if layer.stat().st_size != int(row["size"]):
            raise RuntimeError(f"wrong-sized runtime layer {layer}")
        subprocess.run(["tar", "-xzf", str(layer), "-C", str(root), "--no-same-owner", "--exclude=*/__pycache__",
                        "--exclude=*/__pycache__/*", "--exclude=*.pyc"], check=True)
        for rel in WHITEOUTS.get(int(row["index"]), ()):
            marker = root / rel
            target = marker.with_name(marker.name[4:])
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            marker.unlink(missing_ok=True)
    return time.time() - started


def runtime_env():
    import subprocess

    site = Path(RT) / "usr/local/lib/python3.12/dist-packages"
    cuda_home = Path(RT) / "usr/local/cuda-13.0"
    driver = None
    out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "libcuda.so.1" in line and "=>" in line and "stub" not in line:
            driver = Path(line.split("=>", 1)[1].strip()).resolve().parent
    libs = [cuda_home / "targets/x86_64-linux/lib", cuda_home / "lib64", site / "torch/lib",
            *sorted(p for p in (site / "nvidia").glob("*/lib") if p.is_dir()),
            *([driver] if driver else []), Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    env = os.environ.copy()
    for d in ("/tmp/rtcache", "/tmp/rtcache/triton", "/tmp/rtcache/humming", "/tmp/rttmp"):
        Path(d).mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(site / "nvidia_cutlass_dsl/dsl_packages"), str(site), "/opt/src"]),
        "PATH": os.pathsep.join([str(cuda_home / "bin"), str(Path(RT) / "usr/local/bin"), env.get("PATH", "")]),
        "LD_LIBRARY_PATH": os.pathsep.join(str(p) for p in libs if p.is_dir()),
        "CUDA_HOME": str(cuda_home), "CUDA_PATH": str(cuda_home), "CUDACXX": str(cuda_home / "bin/nvcc"),
        "TORCH_CUDA_ARCH_LIST": "12.0", "TRITON_CACHE_DIR": "/tmp/rtcache/triton", "XDG_CACHE_HOME": "/tmp/rtcache",
        "HUMMING_CACHE_DIR": "/tmp/rtcache/humming", "TMPDIR": "/tmp/rttmp", "VLLM_NO_USAGE_STATS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False", "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONDONTWRITEBYTECODE": "1"})
    return env


@app.function(image=image, gpu="RTX-PRO-6000", cpu=(0.125, CPU), memory=(128, RAM_GIB * 1024), timeout=TIMEOUT,
              retries=0, max_containers=1, volumes={"/assets": assets, "/runs": runs})
def bench(run_id: str, argv: list, module: str = "arena.expert_3bit_bench"):
    import subprocess
    import sys
    import time
    import traceback

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "kind": "expert3bit-bench"}
    try:
        record["extract_seconds"] = extract_runtime()
        env = runtime_env()
        deadline = TIMEOUT - (time.time() - started) - 180
        cmd = [sys.executable, "-u", "-m", module, "--model-dir", MODEL_DIR,
               "--capture", "/opt/capture", "--out", str(out / "bench.json"), "--deadline-seconds", str(deadline),
               *argv]
        with (out / "bench.log").open("w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=deadline + 120)
        record["returncode"] = proc.returncode
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - started
    (out / "execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    return record


def _overlay_worker(job):
    import sys

    sys.path.insert(0, "/opt/src")
    import torch

    from arena import expert_3bit_overlay as ov

    source, out, group, int2 = job
    torch.set_num_threads(8)
    return os.path.basename(out), ov.convert_shard(torch, source, out, group, "cpu", int2=int2)


def _verify_worker(job):
    """(shard, relative errors or None, error text or None) of ``arena.expert_3bit_overlay.verify_shard``."""
    import sys

    sys.path.insert(0, "/opt/src")
    import torch

    from arena import expert_3bit_overlay as ov

    source, out, max_rel = job
    torch.set_num_threads(4)  # verification runs 8 processes
    try:
        return os.path.basename(out), ov.verify_shard(torch, source, out, max_rel), None
    except Exception as exc:  # noqa: BLE001
        return os.path.basename(out), None, repr(exc)


@app.function(image=cpu_image, cpu=CPU, memory=RAM_GIB * 1024, timeout=TIMEOUTS["cpustudy"], retries=0,
              max_containers=len(STUDY_LAYERS), volumes={"/assets": assets, "/runs": runs})
def cpustudy(run_id: str, layer: int, methods: str):
    """scripts/expert_3bit_cpu.py on one layer, with the volume copies of its inputs."""
    import subprocess
    import sys
    import time

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "/opt/scripts/expert_3bit_cpu.py", "--layer", str(layer),
           "--out", str(out / f"L{layer}.json"), "--methods", methods,
           "--bf16-dir", "/assets/research-20260930/bf16", "--model-dir", MODEL_DIR, "--v1-dir", GPTQ_ARC_V1,
           "--calib-root", "/runs/validation/arena-expert-quant-bench-v1-modal-20260928/expert-capture",
           "--eval-dir", "/runs/validation/arena-expert-quant-calibrate-v2-modal-20260928/expert-capture/eval"]
    env = {**os.environ, "PYTHONPATH": "/opt/src"}
    with (out / f"L{layer}.log").open("w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {"layer": layer, "returncode": proc.returncode, "elapsed_seconds": time.time() - started}
    (out / f"L{layer}.execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    return record


@app.function(image=cpu_image, cpu=CPU, memory=RAM_GIB * 1024, timeout=TIMEOUTS["overlay"], retries=0, max_containers=1,
              volumes={"/assets": assets, "/runs": runs})
def overlay(run_id: str, name: str, group: int, source: str, int2: dict = None):
    """int3 overlay (NVFP4 container) of every routed-expert shard of ``source`` into OVERLAY_ROOT/name.

    Resumable (v2, 30 September: v1 was stopped at 80 of 192 shards): shards already present are first checked
    (every present shard must pass ``verify_shard``; the ``overlaydiff`` kind checks determinism); a failing shard is renamed ``*.bad`` and rebuilt, and more than 8 failures abort (a verifier
    fault, not bad shards). Then the missing shards are built, verified the same way, and the manifest (sizes and
    sha256 of all 192) is written last, so an incomplete overlay never carries one."""
    import hashlib
    import multiprocessing as mp
    import sys
    import time
    import traceback

    sys.path.insert(0, "/opt/src")
    from arena import expert_3bit_overlay as ov

    started = time.time()
    out_dir = Path(OVERLAY_ROOT) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = Path("/runs/research") / run_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "kind": "expert3bit-overlay", "name": name, "group": group, "source": source}
    # int2: {layer: expert ids} on the 2-bit grid (7 October screen); their re-rounding error is larger.
    max_rel = 0.7 if int2 else 0.35
    if int2:
        record["int2_experts"] = sum(len(v) for v in int2.values())

    def log(**row):
        print(json.dumps({"seconds": round(time.time() - started), **row}), flush=True)

    def sha(path):
        with open(path, "rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()

    try:
        if (out_dir / "overlay-manifest.json").exists():
            raise RuntimeError("the overlay already has a manifest; refusing to rebuild over it")
        index = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
        # the routed-expert shards of the 48 layers; the MTP draft's fused BF16 experts (mtp.layers.0.mlp.experts.*,
        # model-bf16-00011) are not in the NVFP4 container and stay as they are
        shards = sorted({f for k, f in index.items() if ".mlp.experts." in k and not k.startswith("mtp.")})
        stray = sorted(p.name for p in out_dir.iterdir() if p.name not in shards)
        record["stray_files"] = stray
        for p in stray:
            if p.endswith(".tmp"):
                (out_dir / p).unlink()
        present = [s for s in shards if (out_dir / s).exists()]
        record["present_at_start"] = len(present)
        log(event="start", shards=len(shards), present=len(present), stray=stray)
        with mp.get_context("spawn").Pool(4) as pool, mp.get_context("spawn").Pool(8) as vpool:
            # Determinism: research-expert3bit-overlaydiff-v1 rebuilt a v1 shard twice: every code, block scale and
            # scale_2 identical to the volume copy (the file bytes differ, the tensors do not).
            verify = {}
            bad = []
            for shard, rel, err in vpool.imap_unordered(
                    _verify_worker, [(str(Path(source) / s), str(out_dir / s), max_rel) for s in present]):
                verify[shard] = rel
                if err:
                    bad.append({"shard": shard, "error": err})
            record["present_failed"] = bad
            log(event="verified-present", ok=len(present) - len(bad), failed=bad[:3])
            if len(bad) > 8:
                raise RuntimeError(f"{len(bad)} present shards failed verification: suspect the verifier")
            for row in bad:
                (out_dir / row["shard"]).rename(out_dir / (row["shard"] + ".bad"))
            jobs = [(str(Path(source) / s), str(out_dir / s), group, int2) for s in shards
                    if not (out_dir / s).exists()]
            stats = {}
            for n, (shard, st) in enumerate(pool.imap_unordered(_overlay_worker, jobs)):
                stats[shard] = st
                if n % 16 == 0:
                    log(event="built", done=n + 1, of=len(jobs))
            record["built"] = len(jobs)
            new_bad = []
            for shard, rel, err in vpool.imap_unordered(
                    _verify_worker, [(str(Path(source) / s), str(out_dir / s), max_rel) for s in sorted(stats)]):
                verify[shard] = rel
                if err:
                    new_bad.append({"shard": shard, "error": err})
            if new_bad:
                raise RuntimeError(f"built shards failed verification: {new_bad[:3]}")
            log(event="verified-built", ok=len(stats))
        missing = [s for s in shards if s not in verify or verify[s] is None]
        if missing:
            raise RuntimeError(f"unverified shards: {missing[:5]}")
        manifest_shards = {}
        # A source that omits a mask's pruned experts (tp phase 5) keeps its shards' sizes, not the base's.
        src_manifest = json.loads((Path(source) / "overlay-manifest.json").read_text())
        omits = bool(src_manifest.get("omits_pruned"))
        for s in shards:
            path = out_dir / s
            size = path.stat().st_size
            base = (Path(source) / s if omits else Path(MODEL_DIR) / s).stat().st_size
            if size != base:
                raise RuntimeError(f"{s}: {size} bytes != {'source' if omits else 'base'} {base}")
            manifest_shards[s] = {"bytes": size, "sha256": sha(path), "base_bytes": base}
        log(event="hashed", shards=len(manifest_shards))
        manifest = {"format": ov.FORMAT, "base_model_dir": MODEL_DIR, "layers": list(range(48)),
                    "shards": manifest_shards, "complete": True,
                    "settings": {"grid": "int3", "group": group, "method": "rtn-search", "search": list(ov.SEARCH),
                                 "source_dir": source},
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if omits:
            manifest.update(omits_pruned=True, expert_prune_sha256=src_manifest["expert_prune_sha256"])
        if int2:
            manifest["settings"]["int2_experts"] = {str(k): sorted(v) for k, v in sorted(int2.items(), key=lambda kv: int(kv[0]))}
        raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
        (out_dir / "build-stats.json").write_text(json.dumps(stats, indent=1))
        (out_dir / "verify.json").write_text(json.dumps(verify, indent=1, sort_keys=True))
        (out_dir / "overlay-manifest.json").write_bytes(raw)
        rel13 = [v["w13"] for v in verify.values()]
        rel2 = [v["w2"] for v in verify.values()]
        record.update(manifest_sha256=hashlib.sha256(raw).hexdigest(), shards=len(shards),
                      overlay_bytes=sum(v["bytes"] for v in manifest_shards.values()),
                      verify_rel_error={"w13_mean": sum(rel13) / len(rel13), "w13_max": max(rel13),
                                        "w2_mean": sum(rel2) / len(rel2), "w2_max": max(rel2)})
        assets.commit()
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        try:
            assets.commit()
        except Exception:  # noqa: BLE001
            pass
    record["elapsed_seconds"] = time.time() - started
    (rec_dir / "execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    return record


def _moe_eval_worker(job):
    """One layer: the routed-expert MoE output of each overlay against the true BF16 experts (dequantized weights,
    BF16 activations, float32: the W4A16 / W3A16 arm of ``arena.expert_quant_gptq.Gptq.evaluate``, same row pick), on
    each level's own eval capture. Only the routed experts are read, so overlays that omit pruned experts work."""
    import sys

    sys.path.insert(0, "/opt/src")
    import numpy as np
    import torch
    from safetensors import safe_open

    from arena import expert_quant_bench as B  # noqa: N812
    from arena import expert_quant_math as qm

    layer, spec = job
    torch.set_num_threads(spec["threads"])
    fp8 = torch.float8_e4m3fn
    index = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    bf16_index = json.loads((Path(spec["bf16_dir"]) / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    ref = {}
    for proj in ("gate_up_proj", "down_proj"):
        with safe_open(str(Path(spec["bf16_dir"]) / bf16_index[prefix + proj]), framework="pt", device="cpu") as fh:
            ref[proj] = fh.get_tensor(prefix + proj)
    out = {"layer": layer, "levels": {}}
    for level in spec["levels"]:
        root = Path(level["eval_dir"])
        rows = int(json.loads((root / "meta.json").read_text())["rows"][str(layer)])
        x = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.x", dtype=np.int16).reshape(rows, B.H)).view(
            torch.bfloat16)
        ids = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.ids", dtype=np.int16).reshape(rows, B.TOPK)
                               .astype(np.int32))
        w = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.w", dtype=np.float32).reshape(rows, B.TOPK))
        if level.get("pruned"):  # a pruned server routes in the kept experts' ranks
            ids = qm.remap_pruned_ids(ids, sorted(set(range(B.E)) - set(level["pruned"][str(layer)])))
        n = min(spec["eval_rows"], rows)
        pick = torch.linspace(0, rows - 1, n).round().long()
        xf, ids, w = x[pick].float(), ids[pick].contiguous(), w[pick].contiguous()
        token, weight, bounds = qm.routed_rows(ids, w, range(B.E), B.E)
        routed = [e for e in range(B.E) if bounds[e] != bounds[e + 1]]

        def moe(expert, xf=xf, token=token, weight=weight, bounds=bounds, routed=routed, n=n):
            y = torch.zeros((n, B.H), dtype=torch.float32)
            for e in routed:
                a, b = bounds[e], bounds[e + 1]
                gate_up, down = expert(e)
                h = xf[token[a:b]] @ gate_up.t()
                h = torch.nn.functional.silu(h[:, :B.INTER]) * h[:, B.INTER:]
                y.index_add_(0, token[a:b], (h @ down.t()) * weight[a:b].unsqueeze(1))
            return y

        base = moe(lambda e: (ref["gate_up_proj"][e].float(), ref["down_proj"][e].float()))
        res = {"rows": n, "routed_experts": len(routed)}
        for arm, folder in level["overlays"].items():
            handles = {}

            def tensor(name, folder=folder, handles=handles):
                shard = index[name]
                if shard not in handles:
                    handles[shard] = safe_open(str(Path(folder) / shard), framework="pt", device="cpu").__enter__()
                return handles[shard].get_tensor(name)

            def expert(e, tensor=tensor):
                def part(proj, kind):
                    return tensor(f"{prefix}{e}.{proj}.{kind}").view(torch.uint8)

                gate_up = qm.dequantize(
                    torch.cat([part("gate_proj", "weight"), part("up_proj", "weight")]).unsqueeze(0),
                    torch.cat([part("gate_proj", "weight_scale"), part("up_proj", "weight_scale")]).view(fp8).unsqueeze(0),
                    tensor(f"{prefix}{e}.gate_proj.weight_scale_2").reshape(1).float())[0]
                down = qm.dequantize(part("down_proj", "weight").unsqueeze(0),
                                     part("down_proj", "weight_scale").view(fp8).unsqueeze(0),
                                     tensor(f"{prefix}{e}.down_proj.weight_scale_2").reshape(1).float())[0]
                return gate_up, down

            y = moe(expert)
            res[arm] = {"vs_bf16": qm.relative_error(y, base), "cos_bf16": qm.cosine(y, base)}
            for fh in handles.values():
                fh.__exit__(None, None, None)
        out["levels"][level["name"]] = res
    return out


@app.function(image=cpu_image, cpu=CPU, memory=RAM_GIB * 1024, timeout=TIMEOUTS["moeeval"], retries=0,
              max_containers=1, volumes={"/assets": assets, "/runs": runs})
def moeeval(run_id: str, spec: dict):
    """MoE output error vs BF16 of finished overlays (7 October: the CPU requant INT3 has no such number from its fit).
    ``spec``: bf16_dir, eval_rows, layers, workers, threads and levels [{name, eval_dir, pruned (mask) or None,
    overlays {arm: dir}}]. Writes research/<run id>/moe-eval.json (per layer and the mean over layers)."""
    import multiprocessing as mp
    import time
    import traceback

    started = time.time()
    rec_dir = Path("/runs/research") / run_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "kind": "expert3bit-moeeval",
              "levels": [{k: v for k, v in level.items() if k != "pruned"} for level in spec["levels"]]}
    try:
        layers = {}
        # A fresh process per layer: the four-level 48-layer job was interrupted twice after ~40 layers (7 October;
        # cause not established, worker memory suspected) and Modal re-ran the input each time.
        with mp.get_context("spawn").Pool(spec["workers"], maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(_moe_eval_worker, [(layer, spec) for layer in spec["layers"]]):
                layers[row["layer"]] = row["levels"]
                print(json.dumps({"seconds": round(time.time() - started), "layer": row["layer"],
                                  "done": len(layers)}), flush=True)
                (rec_dir / "moe-eval.json").write_text(json.dumps({"layers": layers}, indent=1, sort_keys=True))
        mean = {}
        for level in spec["levels"]:
            for arm in level["overlays"]:
                values = [layers[i][level["name"]][arm]["vs_bf16"] for i in spec["layers"]]
                mean.setdefault(level["name"], {})[arm] = {"mean_vs_bf16": sum(values) / len(values),
                                                           "layers": len(values)}
        record["mean"] = mean
        (rec_dir / "moe-eval.json").write_text(json.dumps({"mean": mean, "layers": layers, "spec": record["levels"]},
                                                          indent=1, sort_keys=True))
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - started
    (rec_dir / "execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    return record


@app.function(image=cpu_image, cpu=CPU, memory=RAM_GIB * 1024, timeout=TIMEOUTS["overlaydiff"], retries=0,
              max_containers=1, volumes={"/assets": assets, "/runs": runs})
def overlaydiff(run_id: str, name: str, group: int, source: str, shard: str):
    """Rebuild one shard of OVERLAY_ROOT/name twice in /tmp and count what differs from the volume copy (and between
    the two rebuilds): packed weight bytes, block scales, scale_2, and each copy's re-rounding error."""
    import sys
    import time

    sys.path.insert(0, "/opt/src")
    import torch

    from safetensors import safe_open

    from arena import expert_3bit_overlay as ov

    started = time.time()
    torch.set_num_threads(8)
    rec_dir = Path("/runs/research") / run_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    present = Path(OVERLAY_ROOT) / name / shard
    src = str(Path(source) / shard)
    copies = {"present": str(present)}
    for tag in ("rebuild_a", "rebuild_b"):
        copies[tag] = f"/tmp/{tag}-{shard}"
        ov.convert_shard(torch, src, copies[tag], group, "cpu")
    record = {"run_id": run_id, "shard": shard, "cpu": open("/proc/cpuinfo").read().split("model name")[1]
              .split("\n")[0], "verify": {}, "diff": {}}
    for tag, path in copies.items():
        record["verify"][tag] = ov.verify_shard(torch, src, path)
    fp8 = torch.float8_e4m3fn

    def load(path):
        with safe_open(path, framework="pt", device="cpu") as fh:
            return {k: fh.get_tensor(k) for k in fh.keys()}

    tensors = {tag: load(path) for tag, path in copies.items()}
    for pair in (("present", "rebuild_a"), ("rebuild_a", "rebuild_b")):
        a, b = tensors[pair[0]], tensors[pair[1]]
        row = {}
        for kind in ("weight", "weight_scale", "weight_scale_2"):
            n = diff = 0
            for k in a:
                if k.rsplit(".", 1)[1] != kind:
                    continue
                x = a[k].view(torch.uint8) if a[k].dtype == fp8 else a[k]
                y = b[k].view(torch.uint8) if b[k].dtype == fp8 else b[k]
                if kind == "weight_scale_2":
                    x, y = x.float().reshape(-1), y.float().reshape(-1)
                n += x.numel()
                diff += int((x != y).sum())
            row[kind] = {"elements": n, "differ": diff}
        record["diff"]["-vs-".join(pair)] = row
    record["elapsed_seconds"] = time.time() - started
    (rec_dir / "execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    print(json.dumps(record), flush=True)
    return record


@app.local_entrypoint()
def main(kind: str, run_id: str, args: str = "", name: str = "arc-expert-int3-g16-v1", group: int = 16,
         source: str = GPTQ_ARC_V1, storage_usd: float = STORAGE_USD, authorization: str = ""):
    if kind != KIND:
        raise ValueError(f"set ARC_EXPERT3BIT_KIND={kind}")
    bound = reserve(run_id, kind, storage_usd, authorization or None)
    if kind == "overlay":
        call = overlay.spawn(run_id, name, group, source,
                             json.loads(Path(args).read_text())["int2"] if args else None)  # args: an int2 spec file
    elif kind == "overlaydiff":
        call = overlaydiff.spawn(run_id, name, group, source, args)
    elif kind == "moeeval":
        call = moeeval.spawn(run_id, json.loads(Path(args).read_text()))  # args: a local spec file
    elif kind == "cpustudy":
        calls = [cpustudy.spawn(run_id, layer, args) for layer in STUDY_LAYERS]
        call = calls[0]
        extra_calls = [c.object_id for c in calls]
    elif kind == "splitbench":
        call = bench.spawn(run_id, args.split(), "arena.expert_split_bench")
    else:
        call = bench.spawn(run_id, args.split() if args else [])
    record = {"run_id": run_id, "kind": f"expert3bit-{kind}", "function_call_id": call.object_id,
              "function_call_ids": locals().get("extra_calls"),
              "charged_bound_usd": bound, "timeout_seconds": TIMEOUTS[kind], "state": "submitted", "args": args}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
