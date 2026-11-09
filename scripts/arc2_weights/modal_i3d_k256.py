"""arc2 copy (21 Oct) of dfr-exact scripts/dfr_exact/int3/modal_int3_fit.py with the arc2k256 variant.

Item 3: direct GPTQ-INT3 experts from BF16 for x5, on one Modal B200.

x5 serves the t37 excl5 mask with a requant INT3 overlay (GPTQ-W4, then RTN re-rounding to INT3), which has ~47% more
MoE error than direct GPTQ-INT3 from BF16 (7 Oct, t0). This fits the direct INT3 grid (and the W4 side output) with
arena.expert_quant_gptq (throughput-arm src, --int3-out) on today's excl5 capture (b2-tp-cap-v85p-x5-dense-v2: MoE
inputs of all 48 layers, routed by the x5 server), same settings as the x5 capture bundle's expert_quant block
(percdamp .01, min rows 64, batch 128, prior rows 4,096, act order, INT3 group 16, t37 excl5 mask, omit pruned).
Output: /assets/flash-input/datasets/richardcsaky/arc-expert-int3-direct-t37-excl5-v1 (+ the fit's gptq.json and log
under /runs/throughput/<run id>/), then the mirror step copies it to the Nebius bucket with an assets manifest.

    modal run scripts/dfr_exact/int3/modal_int3_fit.py --run-id tp-int3-direct-x5-pre-v1 --preflight   # CPU, paths+imports
    modal run --detach scripts/dfr_exact/int3/modal_int3_fit.py --run-id tp-int3-direct-x5-v1-20261020    # B200
    modal run scripts/dfr_exact/int3/modal_int3_fit.py --run-id tp-int3-direct-x5-v1-20261020 --mirror  # to the bucket
A rerun with --layers continues missing layers into the same folders (each layer's shards are final when written).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal
import modal.exception
try:  # grpclib's stream end on a dropped connection (pf11, 26 Oct)
    from grpclib.exceptions import StreamTerminatedError as _StreamTerminated
except ImportError:  # pragma: no cover
    _StreamTerminated = OSError

HERE = Path(__file__).resolve().parent
ROOT = Path(".")
TP = ROOT / ".claude/worktrees/throughput-arm/src/arena"
SETUP = ROOT / "src/arc_infra/serving_setup.py"
# ARC_I3D_VARIANT (baked into the image): excl5 = x5 (20 Oct); all25 = v8.5p's all25v71b mask on the 21 Oct all25
# capture (tonight's submission).
VARIANT = os.environ.get("ARC_I3D_VARIANT", "arc2k256")
VARIANTS = {
    "excl5": ("prune-t37-int3-requant-excl5-v1.json", "b2-tp-cap-v85p-x5-dense-v2-nebius-20261020",
              "arc-expert-int3-direct-t37-excl5-v1"),
    "all25": ("prune-t37-int3-requant-all25v71b-v1.json", "b2-tp-cap-v85p-all25-dense-v1-nebius-20261021",
              "arc-expert-int3-direct-t37-all25v71b-v1"),
    # ARC-AGI-2 submission: v37's k256 mask on v37's refit capture (c240, routed by the k256 server)
    "arc2k256": ("prune-arc2-k256-v1.json", "b2-arc2-quant-capture-c240-k256-v1-nebius-20261013",
                 "arc-expert-int3-direct-arc2c240-k256-v1"),
}
MASK = (ROOT / ".claude/worktrees/arc2/scripts/arc2_weights/masks" if VARIANT == "arc2k256"
        else ROOT / ".claude/worktrees/dfr-exact/data/expert_prune") / VARIANTS[VARIANT][0]
CAPTURE = f"/runs/validation/{VARIANTS[VARIANT][1]}/expert-capture"
BASE = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
BF16 = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
RUNTIME = "/assets/flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
OUT = f"flash-input/datasets/richardcsaky/{VARIANTS[VARIANT][2]}"
TIMEOUT = 2700
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"

app = modal.App("arc-int3-direct-fit-" + VARIANT)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build", "rclone")
         .add_local_file(str(SETUP) if modal.is_local() else "/opt/arc/serving_setup.py", "/opt/arc/serving_setup.py",
                         copy=True)
         .add_local_file(str(MASK) if modal.is_local() else "/opt/arc/mask.json", "/opt/arc/mask.json", copy=True)
         .add_local_dir(str(TP) if modal.is_local() else "/opt/arcsrc/arena", "/opt/arcsrc/arena", copy=True)
         .add_local_file(__file__, "/root/modal_int3_fit.py", copy=True)
         .env({"ARC_I3D_VARIANT": VARIANT}))


def load_setup():
    import importlib.util

    for key, value in (("TAAF_KAGGLE_BUNDLE_DIR", "/tmp/fit-bundle"), ("TAAF_KAGGLE_WORKING_DIR", "/tmp/fit-work"),
                       ("TAAF_KAGGLE_SETUP_ENV", "/tmp/fit-work/setup-env.json")):
        os.environ.setdefault(key, value)
    for name in ("TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_WORKING_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    spec = importlib.util.spec_from_file_location("arc_serving_setup", "/opt/arc/serving_setup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["arc_serving_setup"] = module
    spec.loader.exec_module(module)
    return module


def body(run_id, gpu, layers):
    started = time.time()
    out = Path("/runs/throughput") / run_id
    out.mkdir(parents=True, exist_ok=True)
    state = {"run_id": run_id, "gpu": gpu, "started_epoch": started}

    def save():
        (out / "state.json").write_text(json.dumps(state, indent=1, default=str))
        runs.commit()

    cap = Path(CAPTURE)
    state["inputs"] = {"calib_meta": (cap / "calib/meta.json").is_file(), "eval_meta": (cap / "eval/meta.json").is_file(),
                       "calib_layers": len(list((cap / "calib").glob("L*.x"))), "bf16_shards": len(list(Path(BF16).glob("*.safetensors"))),
                       "base_index": (Path(BASE) / "model.safetensors.index.json").is_file()}
    save()
    setup = load_setup()
    extracted = setup.verify_and_extract_runtime(Path(RUNTIME), full_layer_hashes=False, scan_extracted_caches=False)
    site = setup.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages"
    if gpu:
        setup.VLLM_PLE_PATCHED_SHA256 = setup.VLLM_PLE_STOCK_SHA256  # no server here: the stock PLE layer is fine
        env, _ = setup.runtime_environment(deep_preload_validation=False)
    else:
        env = {**os.environ, "PYTHONPATH": f"{site / 'nvidia_cutlass_dsl/dsl_packages'}:{site}"}
    env["PYTHONPATH"] = "/opt/arcsrc:" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    state["runtime_extract_seconds"] = round(extracted["extract_seconds"], 1)
    save()
    int3_dir = Path("/assets") / OUT
    int3_dir.mkdir(parents=True, exist_ok=True)
    left = TIMEOUT - (time.time() - started) - 240
    args = ["--calib", str(cap / "calib"), "--capture", str(cap), "--model-dir", BASE, "--bf16-dir", BF16,
            "--overlay-out", "/tmp/w4-overlay", "--int3-out", str(int3_dir), "--out", str(out / "gptq.json"),
            "--deadline-seconds", str(int(left if gpu else 40)), "--eval-rows", "4096", "--percdamp", "0.01",
            "--min-rows", "64", "--batch", "128", "--scale-search", "1.0", "--prior-rows", "4096", "--act-order",
            "--prune-mask", "/opt/arc/mask.json", "--omit-pruned", "--int3-group", "16",
            "--device", "cuda" if gpu else "cpu"]
    if layers:
        args += ["--layers", layers]
    state["command"] = args
    save()
    log = out / "gptq.log"
    with open(log, "ab") as stream:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "arena.expert_quant_gptq", *args], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last = time.time()
        for line in iter(proc.stdout.readline, b""):
            stream.write(line)
            if b'"layer"' in line or b"error" in line.lower() or b'"done"' in line:
                print(line.decode(errors="replace").rstrip()[:400], flush=True)
            if time.time() - last > 60:
                stream.flush(); assets.commit(); runs.commit(); last = time.time()
        code = proc.wait()
    state["returncode"] = code
    state["elapsed_seconds"] = round(time.time() - started, 1)
    if (out / "gptq.json").is_file():
        result = json.loads((out / "gptq.json").read_text())
        state["errors"] = result.get("errors")
        state["int3_complete"] = bool((result.get("manifest_int3") or {}).get("complete"))
        state["layers_int3"] = len((result.get("manifest_int3") or {}).get("layers") or [])
    assets.commit()
    save()
    print("RESULT " + json.dumps({k: state.get(k) for k in ("returncode", "int3_complete", "layers_int3", "errors",
                                                             "elapsed_seconds", "inputs")}, default=str), flush=True)
    return state


@app.function(image=image, gpu="B200", timeout=TIMEOUT, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs})
def fit(run_id: str, layers: str = ""):
    return body(run_id, True, layers)


@app.function(image=image, timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs})
def preflight(run_id: str):
    return body(run_id, False, "0")


@app.function(image=image, timeout=3600, retries=0, max_containers=1, volumes={"/assets": assets},
              secrets=[modal.Secret.from_name("nebius-s3")])
def mirror():
    import hashlib

    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    src = Path("/assets") / OUT
    man = json.loads((src / "overlay-manifest.json").read_text())
    if not man.get("complete"):
        raise RuntimeError("overlay incomplete")
    subprocess.run(["rclone", "copy", str(src), f"arc:{BUCKET}/assets/{OUT}", "--transfers", "16"], check=True)
    files = {}
    for f in sorted(src.iterdir()):
        with open(f, "rb") as h:
            files[f.name] = {"size": f.stat().st_size, "sha256": hashlib.file_digest(h, "sha256").hexdigest()}
    slug = OUT.replace("/", "--") + ".json"
    Path(f"/tmp/{slug}").write_text(json.dumps({"prefix": OUT, "files": files}))
    subprocess.run(["rclone", "copyto", f"/tmp/{slug}", f"arc:{BUCKET}/assets-manifests/{slug}"], check=True)
    return {"files": len(files), "bytes": sum(v["size"] for v in files.values()),
            "manifest_sha256": files["overlay-manifest.json"]["sha256"]}

def wait_spawned(fn, *args):
    """spawn + wait, re-attaching by call id: a client holding .remote() cancels the call when its connection drops
    (r7/r8 26 Oct, pf-temp05b 27 Oct)."""
    import time as _time
    call = fn.spawn(*args)
    print("CALL " + call.object_id, flush=True)
    misses = 0
    while True:
        try:
            return call.get(timeout=60)
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except (OSError, modal.exception.ConnectionError, _StreamTerminated) as exc:  # a dropped connection: re-attach by id
            misses += 1
            if misses > 30:
                raise
            print("reattach after " + type(exc).__name__, flush=True)
            _time.sleep(20)
            call = modal.FunctionCall.from_id(call.object_id)


@app.local_entrypoint()
def main(run_id: str, preflight: bool = False, mirror: bool = False, layers: str = ""):
    if mirror:
        print(json.dumps(wait_spawned(globals()["mirror"])))
        return
    if preflight:
        print(json.dumps(wait_spawned(globals()["preflight"], run_id), default=str)[:3000])
        return
    call = fit.spawn(run_id, layers)
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id, "app_id": app.app_id,
                      "start_epoch": round(time.time()), "timeout": TIMEOUT}))
