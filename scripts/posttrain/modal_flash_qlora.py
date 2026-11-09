"""Modal launcher for the Flash QLoRA trainer jobs (``flash_qlora_job.py``; post-training pilot, 7 October).

    ARC_QLORA_MODE=check|parity|smoke ARC_VALIDATION_ROOT=<main checkout> \
        uv run --extra modal modal run --detach scripts/posttrain/modal_flash_qlora.py --run-id sft-qlora-<mode>-<n>

- ``check``  (one L4, <= 20 min, feasibility stage): imports, volume read speed, the smoke procedure on a tiny model;
- ``parity`` (CPU, <= 30 min, feasibility stage): adapted tensors, served checkpoint against the training checkpoint;
- ``smoke`` (one B200, 45 min, training stage): the real model.

The job runs with the Axolotl image's own Python as a child process; outputs land in
``arc-prolong-runs:/posttrain/<run id>``. The reservation goes into ``posttrain_pilot_v1`` of the main checkout's
``docs/spend.json`` before the spawn; the receipt is ``docs/validation-20261007/modal-<run id>.json``.
"""

import json
import os
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or REPO).resolve()
RECORDS = "docs/validation-20261007"
# axolotlai/axolotl:main-py3.12-cu130-2.13.0 of 7 October 2026 (axolotl 2768e6d0, transformers 5.18.0, peft 0.21.1)
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
BF16 = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"  # Qwen/Qwen3.8-Flash-Next, all 131 shards
SERVED = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
MODES = {  # rate: dollars per GPU second (modal.com/pricing)
    "check": {"gpu": "L4", "rate": 0.000222, "cpu": 4.0, "memory_gib": 24, "seconds": 1200, "stage": "feasibility"},
    "parity": {"gpu": None, "rate": 0.0, "cpu": 4.0, "memory_gib": 24, "seconds": 1800, "stage": "feasibility"},
    "smoke": {"gpu": "B200", "rate": 0.001736, "cpu": 8.0, "memory_gib": 200, "seconds": 2700, "stage": "training"},
    # G2 pre-check: a measurement adapter (never served for play), then its overlay folder on the assets volume
    "adapter": {"gpu": "B200", "rate": 0.001736, "cpu": 8.0, "memory_gib": 200, "seconds": 1200, "stage": "training"},
    "export": {"gpu": None, "rate": 0.0, "cpu": 8.0, "memory_gib": 64, "seconds": 2400, "stage": "feasibility"},
    "overlay": {"gpu": None, "rate": 0.0, "cpu": 4.0, "memory_gib": 48, "seconds": 1200, "stage": "feasibility"},
    # several GPUs: memory and gradient agreement on captured requests (ARC_QLORA_ARGS: "--plan ..."); multicheck is
    # the same code on the tiny model
    "multi": {"gpu": "B200:2", "rate": 2 * 0.001736, "cpu": 8.0, "memory_gib": 440, "seconds": 1200,
              "stage": "training"},
    "multicheck": {"gpu": "A10G:2", "rate": 2 * 0.000306, "cpu": 4.0, "memory_gib": 32, "seconds": 1200,
                   "stage": "feasibility"},
    # captured served requests (ARC_QLORA_TEXT: one JSON per line) as training examples
    "contexts": {"gpu": "L4", "rate": 0.000222, "cpu": 4.0, "memory_gib": 32, "seconds": 1200, "stage": "feasibility"},
    # the first adapter (ARC_QLORA_ARGS: "<corpus run id> [trainer arguments]"): one B200, 9 h; Modal bills a B200
    # container 435 GiB of host memory, so the bound counts that. traincheck: the same loop on the tiny model
    "train": {"gpu": "B200", "rate": 0.001736, "cpu": 8.0, "memory_gib": 200, "billed_memory_gib": 435,
              "seconds": 32400, "stage": "training"},
    "traincheck": {"gpu": "L4", "rate": 0.000222, "cpu": 4.0, "memory_gib": 32, "seconds": 1200,
                   "stage": "feasibility"},
    # training on the served NVFP4 checkpoint, one B300 (ARC_QLORA_ARGS as train, e.g. "<corpus> --phases
    # 16384,32768,49152 --rank 64 --rslora --lr 1e-4"; ARC_QLORA_PIP as the nvfp4 job; ARC_QLORA_SECONDS: the budget)
    "trainnv": {"gpu": "B300", "rate": 0.001972, "cpu": 8.0, "memory_gib": 300, "billed_memory_gib": 435,
                "seconds": int(os.environ.get("ARC_QLORA_SECONDS") or 7200), "stage": "training"},
    # the trainer on the served NVFP4 checkpoint (ARC_QLORA_ARGS: "<corpus run id> [--expert-lora]";
    # ARC_QLORA_TEXT: the served-NLL prompt file): load, span NLL, step cost on one and two B200s
    "nvfp4": {"gpu": "B200:2", "rate": 2 * 0.001736, "cpu": 8.0, "memory_gib": 440, "seconds": 3300,
              "stage": "training", "authorization": (
                  "'we should Train against the served NVFP4 experts'; "
                  "NVFP4 trainer feasibility job on two B200s, <= <= 1 h")},
    # RL arm (rl_job.py; run ids rl-<mode>-..., ledger rl_proof_v1). rlexperts: quantisation-aware adapters on the
    # routed experts, cost and merge (ARC_QLORA_ARGS: "<corpus run id> [rl_job arguments]"; ARC_QLORA_TEXT: the
    # served-NLL prompt file)
    "rlexperts": {"gpu": "B300", "rate": 0.001972, "cpu": 8.0, "memory_gib": 300, "billed_memory_gib": 435,
                  "seconds": 4200, "stage": "rl"},
    # rltrain: the RL update on the served checkpoint (ARC_QLORA_TEXT: one arm's training set, rl_rollouts.training_set
    # as JSON; ARC_QLORA_ARGS: rl_job train arguments; ARC_QLORA_SECONDS: the budget). rlcheck: the same loop on the
    # tiny model
    "rltrain": {"gpu": "B300", "rate": 0.001972, "cpu": 8.0, "memory_gib": 300, "billed_memory_gib": 435,
                "seconds": int(os.environ.get("ARC_QLORA_SECONDS") or 7200), "stage": "rl"},
    "rlcheck": {"gpu": "L4", "rate": 0.000222, "cpu": 4.0, "memory_gib": 32, "seconds": 1200, "stage": "rl"},
    # rlexport: an RL adapter merged into the served BF16 shards as an overlay folder over the served checkpoint's
    # own experts (ARC_QLORA_ARGS: "<rl-train run id> <overlay name>"); CPU
    "rlexport": {"gpu": None, "rate": 0.0, "cpu": 8.0, "memory_gib": 64, "seconds": 2400, "stage": "rl"},
    # rloverlay: the overlay check on an RL overlay (ARC_QLORA_ARGS: "<overlay name>"); CPU
    "rloverlay": {"gpu": None, "rate": 0.0, "cpu": 4.0, "memory_gib": 48, "seconds": 1200, "stage": "rl"},
}
RL_PIP = "quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2 ;; --no-deps triton==3.6.0"
EXPERTS = "/assets/flash-input/datasets/richardcsaky/arc-expert-int3-requant-t37-dev10-v2"  # locked v7.1p experts
OVERLAYS = "/assets/flash-input/datasets/posttrain"  # <name>: merged BF16 shards beside links to EXPERTS
AUTHORIZATION = {
    "feasibility": "post-training feasibility pilot; QLoRA trainer preflight (small GPU or CPU)",
    "training": ("'for post-training start with qlorea feasibility'; one B200, "
                 "45-minute trainer smoke; B200 feasibility measurements <= 20 min and <= "
                 "each inside a, G2 pre-check adapter (a measurement, not served for play); "
                 "'we can run the post-training arm on multiple B200's if needed', multi-GPU "
                 "measurements <= 20 min and <=; 'i allow the training run, but "
                 "monitor closely and stop if it's not going as expected or there are issues' (one run, one B200, "
                 "<= 10.5 h, <=)"),
}
MODE = os.environ.get("ARC_QLORA_MODE", "check")
SPEC = dict(MODES[MODE])
if os.environ.get("ARC_QLORA_GPU"):  # the same mode on another GPU type: "<type>[:<count>]=<dollars per GPU second>"
    kind, rate = os.environ["ARC_QLORA_GPU"].split("=")
    SPEC.update(gpu=kind, rate=float(rate) * int((kind.split(":") + ["1"])[1]))
if os.environ.get("ARC_QLORA_MEMORY_GIB"):  # replicas: ~130 GiB of host memory each after the load
    SPEC.update(memory_gib=int(os.environ["ARC_QLORA_MEMORY_GIB"]),
                billed_memory_gib=max(int(os.environ["ARC_QLORA_MEMORY_GIB"]), SPEC.get("billed_memory_gib", 0)))
if os.environ.get("ARC_QLORA_SECONDS"):  # a shorter run than the mode's ceiling
    SPEC["seconds"] = min(SPEC["seconds"], int(os.environ["ARC_QLORA_SECONDS"]))

app = modal.App("arc-flash-qlora", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .env({"ARC_QLORA_MODE": MODE, "ARC_QLORA_ARGS": os.environ.get("ARC_QLORA_ARGS", ""),
               "ARC_QLORA_SECONDS": os.environ.get("ARC_QLORA_SECONDS", ""),
               # --replicas: the GPU of each replica ("0,0": two replicas on one GPU, the tiny-model check)
               "ARC_REPLICA_DEVICES": os.environ.get("ARC_REPLICA_DEVICES", ""),
               "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
               # W4A16 like the serving kernels (fp4_cute also quantizes activations)
               "AXOLOTL_SONICMOE_NVFP4_BACKEND": os.environ.get("AXOLOTL_SONICMOE_NVFP4_BACKEND", "dequant")})
         .add_local_file(str(Path(__file__).with_name("flash_qlora_job.py")) if modal.is_local()
                         else "/root/flash_qlora_job.py", "/root/flash_qlora_job.py", copy=True)
         .add_local_file(str(Path(__file__).with_name("rl_job.py")) if modal.is_local()
                         else "/root/rl_job.py", "/root/rl_job.py", copy=True)
         .add_local_file(str(Path(__file__).resolve().parents[2] / "src/posttrain/rl_rollouts.py") if modal.is_local()
                         else "/root/rl_rollouts.py", "/root/rl_rollouts.py", copy=True)
         .add_local_file(__file__, "/root/modal_flash_qlora.py", copy=True))
# Extra installs into Axolotl's environment, " ;; " between commands. A B300 needs Triton 3.6.x for SonicMoE while
# the image's torch 2.13 asks for 3.7.1 and the hub kernels exist for torch 2.13 only (8 October), hence
# "quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2 ;; --no-deps triton==3.6.0".
for command in filter(None, (part.strip() for part in (os.environ.get("ARC_QLORA_PIP") or (
        RL_PIP if SPEC["gpu"] == "B300" and MODE.startswith("rl") else "")).split(";;"))):
    image = image.run_commands("uv pip install --python /workspace/axolotl-venv/bin/python "
                               + " ".join(f"'{item}'" for item in command.split()))
if os.environ.get("ARC_QLORA_TEXT"):  # a DEV-game context for --text /root/context.txt
    image = image.env({"ARC_QLORA_TEXT": "/root/context.txt"}).add_local_file(
        os.environ["ARC_QLORA_TEXT"], "/root/context.txt", copy=True)


def bound(spec):
    memory = spec.get("billed_memory_gib", spec["memory_gib"])
    return (spec["seconds"] + 60) * (spec["rate"] + spec["cpu"] * 1.31e-05 + memory * 2.22e-06) * 1.15


def job_command(mode, out, seconds):
    command = ["/workspace/axolotl-venv/bin/python", "-u", "/root/flash_qlora_job.py",
               {"rlexport": "export", "rloverlay": "overlay"}.get(mode, mode), "--model", BF16,
               "--out", out, "--seconds", str(seconds)]
    if mode == "parity":
        command += ["--served", SERVED]
    if mode == "contexts":
        command += ["--data", "/root/context.txt", "--served", SERVED]
    if mode in ("multi", "multicheck"):
        command[3] = "multi"
        command += ["--data", "/root/context.txt", "--served", SERVED] + os.environ["ARC_QLORA_ARGS"].split() + (
            ["--tiny"] if mode == "multicheck" else [])
    if mode in ("train", "traincheck", "trainnv"):
        command[3] = "train"
        corpus, *extra = os.environ["ARC_QLORA_ARGS"].split()
        if mode == "trainnv":
            command[command.index("--model") + 1] = SERVED
        command += ["--corpus", f"/runs/posttrain/{corpus}", "--served", SERVED, "--tail", "16384", "--lr", "1e-4",
                    "--nvfp4" if mode == "trainnv" else "--bf16-linear"] + extra + (
                        ["--tiny"] if mode == "traincheck" else [])
    if mode == "nvfp4":
        corpus, *extra = os.environ["ARC_QLORA_ARGS"].split()
        command[command.index("--model") + 1] = SERVED
        command += ["--corpus", f"/runs/posttrain/{corpus}", "--data", "/root/context.txt", "--gpus",
                    (str(SPEC["gpu"]).split(":") + ["1"])[1]] + extra
    if mode == "rlexperts":
        corpus, *extra = os.environ["ARC_QLORA_ARGS"].split()
        command = ["/workspace/axolotl-venv/bin/python", "-u", "/root/rl_job.py", "experts", "--model", SERVED,
                   "--out", out, "--seconds", str(seconds), "--corpus", f"/runs/posttrain/{corpus}",
                   *(["--data", "/root/context.txt"] if os.environ.get("ARC_QLORA_TEXT") else []), *extra]
    if mode in ("rltrain", "rlcheck"):
        command = ["/workspace/axolotl-venv/bin/python", "-u", "/root/rl_job.py", "train", "--model",
                   SERVED if mode == "rltrain" else BF16, "--served", SERVED, "--out", out, "--seconds", str(seconds),
                   "--data", "/root/context.txt", *os.environ.get("ARC_QLORA_ARGS", "").split(),
                   *(["--tiny"] if mode == "rlcheck" else [])]
    if mode in ("overlay", "rloverlay"):  # ARC_QLORA_ARGS: "<overlay name>"
        command += ["--served", SERVED, "--overlay", f"{OVERLAYS}/{os.environ['ARC_QLORA_ARGS'].strip()}"]
    if mode == "adapter" or (mode == "check" and os.environ.get("ARC_QLORA_TEXT")):
        command += ["--data", "/root/context.txt"] + os.environ.get("ARC_QLORA_ARGS", "").split()
    if mode == "export":  # ARC_QLORA_ARGS: "<adapter run id> <overlay name> [none]" (none: no expert shards beside)
        source, name, *plain = os.environ["ARC_QLORA_ARGS"].split()
        command += ["--served", SERVED, "--adapter", f"/runs/posttrain/{source}/adapter", "--experts",
                    "none" if plain else EXPERTS, "--overlay", f"{OVERLAYS}/{name}"]
    if mode == "rlexport":
        source, name = os.environ["ARC_QLORA_ARGS"].split()
        command += ["--served", SERVED, "--adapter", f"/runs/posttrain/{source}/adapter", "--experts", "none",
                    "--overlay", f"{OVERLAYS}/{name}"]
    if mode == "smoke" and os.environ.get("ARC_QLORA_ARGS"):  # e.g. "--lengths 8192 --windows 65536:16384"
        command += os.environ["ARC_QLORA_ARGS"].split()
    return command


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, retries=0, max_containers=1,
              scaledown_window=2, cpu=SPEC["cpu"], memory=int(SPEC["memory_gib"] * 1024), timeout=SPEC["seconds"],
              **({"gpu": SPEC["gpu"]} if SPEC["gpu"] else {}))
def job(run_id: str):
    import shlex
    import subprocess
    import time

    out = f"/runs/posttrain/{run_id}"
    Path(out).mkdir(parents=True, exist_ok=True)
    # the job stops itself before the container's timeout (multi checks the clock before every step itself)
    seconds = SPEC["seconds"] - (90 if MODE == "multi" else 240)
    environment = {**os.environ, "PATH": "/workspace/axolotl-venv/bin:" + os.environ["PATH"],
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "AXOLOTL_DO_NOT_TRACK": "1",
                   "WANDB_MODE": "disabled", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "HF_HOME": "/tmp/hf", "TMPDIR": "/tmp"}
    if MODE in ("nvfp4", "trainnv") or MODE.startswith("rl"):  # Axolotl's kernels plugin fetches 'kernels-community/activation' from the Hub
        # one tensor at a time: on a fast host the loader threads put the experts on the GPU as BF16 faster than
        # the NVFP4 converter packs them (B300 v6, 8 October: out of memory at 102 GiB and rising)
        environment.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", HF_DEACTIVATE_ASYNC_LOAD="1")
    for name in ("PYTHONPATH", "PYTHONHOME"):  # the child is the image's interpreter, not Modal's
        environment.pop(name, None)
    script = ("if [ -f /workspace/axolotl/scripts/cuda13_env.sh ]; then source /workspace/axolotl/scripts/cuda13_env.sh;"
              " fi; exec " + shlex.join(job_command(MODE, out, seconds)))
    began = time.time()
    with open(f"{out}/job.log", "a") as log:  # a restarted container continues the same folder
        child = subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=environment, cwd="/tmp")
        last = time.time()
        for line in child.stdout:
            log.write(line)
            log.flush()
            if line.startswith("{") or "Error" in line or "Traceback" in line:
                print(line.rstrip()[:2000], flush=True)
            if time.time() - last > 60:
                runs.commit()
                last = time.time()
        code = child.wait()
    result = json.loads(Path(f"{out}/result.json").read_text()) if Path(f"{out}/result.json").is_file() else {}
    tail = Path(f"{out}/job.log").read_text()[-3000:] if code else ""
    summary = {"run_id": run_id, "mode": MODE, "exit": code, "seconds": round(time.time() - began, 1),
               "passed": result.get("passed", False), "result": result, "log_tail": tail}
    Path(f"{out}/summary.json").write_text(json.dumps(summary, indent=1))
    runs.commit()
    if MODE in ("export", "rlexport"):
        assets.commit()
    return summary


@app.local_entrypoint()
def main(run_id: str):
    sys.path.insert(0, str(REPO / "src"))
    if MODE.startswith("rl"):
        if not run_id.startswith(f"rl-{MODE[2:]}-"):
            raise ValueError(f"{MODE} runs use rl-{MODE[2:]}- run ids")
        from posttrain import rl_spend

        charged = rl_spend.reserve(ROOT / "docs/spend.json", run_id, seconds=SPEC["seconds"], cpu=SPEC["cpu"],
                                   gpu_rate=SPEC["rate"], note=f"{MODE} on {SPEC['gpu']}",
                                   memory_gib=SPEC.get("billed_memory_gib", SPEC["memory_gib"]))
        record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
        record.parent.mkdir(exist_ok=True)
    else:
        if not run_id.startswith(f"sft-qlora-{MODE}-"):
            raise ValueError(f"{MODE} runs use sft-qlora-{MODE}- run ids")
        from posttrain.pilot_spend import reserve

        charged = reserve(ROOT / "docs/spend.json", run_id, seconds=SPEC["seconds"], teacher=False, cpu=SPEC["cpu"],
                          stage=SPEC["stage"], authorization=SPEC.get("authorization") or AUTHORIZATION[SPEC["stage"]],
                          gpu_rate=SPEC["rate"],
                          memory_gib=SPEC.get("billed_memory_gib", SPEC["memory_gib"]))
        record = ROOT / RECORDS / f"modal-{run_id}.json"
    data = {"state": "submission_intent", "run_id": run_id, "kind": "flash-qlora", "test": MODE,
            "charged_bound_usd": charged, "gpu": SPEC["gpu"], "timeout_seconds": SPEC["seconds"],
            "image": AXOLOTL_IMAGE, "model": SERVED if MODE.startswith("rl") else BF16,
            "output": f"arc-prolong-runs:/posttrain/{run_id}"}
    record.write_text(json.dumps(data, indent=2))
    call = job.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
