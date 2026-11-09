"""BF16 next-token reference for the quantization matrix (throughput arm, 6 October).

The full BF16 checkpoint (Qwen/Qwen3.8-Flash-Next @ de4b8e4d, 360 GB, /assets/expert-bf16/<rev>) on 4 RTX PRO 6000
with tensor parallelism, in the pinned vLLM fork runtime (the recipe's runtime extraction, PLE patch and environment;
no arena kernels, no quantization, eager mode). Teacher forcing with ``prompt_logprobs`` = 20 on the kernel-quality
bundle's token ids (``arena.kernel_gate.run_teacher``), at concurrency 1 and a repeat at concurrency 2 (BF16's own
noise). Output: arc-prolong-runs:/validation/<run_id>/kernel-quality.json in the kernel-quality format.

    ARC_VALIDATION_ROOT=. ARC_VALIDATION_BUNDLE=$PWD/dist/tp-kl-quant-v1 \
        uv run --extra modal modal run --detach scripts/modal_bf16_teacher.py --run-id <id>
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
RECORDS = "docs/validation-20261003"
BF16_DIR = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
GPUS, CPU, MEMORY_GIB, TIMEOUT = 4, 16.0, 256, 5400
# (tag, model dir, repeat pass): BF16, then BF16 with the PLE n-gram table round-tripped through FP8 at the NVFP4
# checkpoint's scale (the PLE-precision point; scratchpad ple_fp8_farm.py builds the farm)
MODELS = [("bf16", BF16_DIR, True), ("bf16-ple-fp8", "/assets/research-20261006/bf16-ple-fp8", False)]
SETTINGS = {"utilization": "0.93", "max_num_batched_tokens": 1024, "ready_seconds": 2400, "repeat_concurrency": 2, "nccl_p2p_disable": True, "nccl_cumem_host": False}
AUTHORIZATION = os.environ.get("ARC_BF16_AUTHORIZATION") or (
    "KL vs BF16 of the quantization matrix on dev contexts; throughput arm "
    " up to 4 GPUs in parallel")
BUNDLE = (Path(os.environ.get("ARC_VALIDATION_BUNDLE", "dist/tp-kl-quant-v1")).resolve()
          if modal.is_local() else Path("/opt/arc-validation"))

app = modal.App("arc-bf16-teacher", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .pip_install("httpx==0.28.1", "jinja2==3.1.6")
         .add_local_dir(str(BUNDLE), "/opt/arc-validation", copy=True,
                        ignore=["dataset-upload/**", "*.ipynb", "source.zip", "wheels/**", "node-*", "pi/**",
                                "pi-arena/**"])
         .add_local_file(__file__, "/root/modal_bf16_teacher.py", copy=True))


def bound_usd(rates, seconds=TIMEOUT):
    rate = GPUS * rates["RTX-PRO-6000"] + CPU * rates["physical_cpu_core"] + MEMORY_GIB * rates["memory_gib"]
    return (seconds + 60) * rate * 1.15


def server_argv(command, model_dir, settings):
    """The recipe's command for the BF16 checkpoint: no quantization flag, TP 4, eager, small prefill chunks, no
    multimodal inputs."""
    command = list(command)
    if "--quantization" in command:
        i = command.index("--quantization")
        del command[i:i + 2]
    if "--moe-backend" in command:
        i = command.index("--moe-backend")
        del command[i:i + 2]
    command[command.index("--tensor-parallel-size") + 1] = str(GPUS)
    command[command.index("--max-num-batched-tokens") + 1] = str(settings["max_num_batched_tokens"])
    command[command.index("--gpu-memory-utilization") + 1] = settings["utilization"]
    for flag in ("--async-scheduling",):
        if flag in command:
            command.remove(flag)
    return command + ["--enforce-eager", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]


def complete_teacher_artifact(result, expected_rows, repeat):
    """Only resume past complete, successful measurements, including the requested noise repeat."""
    if expected_rows <= 0:
        return False
    keys = ("teacher_forced", "teacher_forced_repeat") if repeat else ("teacher_forced",)
    indices = None
    for key in keys:
        rows = result.get(key, {}).get("rows", [])
        current = {row.get("index") for row in rows}
        if (len(rows) != expected_rows or len(current) != expected_rows
                or any(row.get("error") or not row.get("positions") for row in rows)):
            return False
        if indices is not None and current != indices:
            return False
        indices = current
    return True


async def run_continuations(client, prompts, settings, save):
    """Fresh BF16 assistant tokens on saved dev prefixes; these are not gameplay episodes."""
    from arena.kernel_gate import bounded, token_id

    async def work(prompt):
        reply = await client.post(client.base_url + "/completions", {
            "model": client.model, "prompt": prompt["ids"], "temperature": 0.7,
            "top_p": 0.95, "top_k": 20, "max_tokens": settings["continuation_tokens"],
            "seed": prompt["index"], "logprobs": 0, "return_tokens_as_token_ids": True})
        choice = reply["choices"][0]
        tokens = [token_id(t) for t in (choice.get("logprobs") or {}).get("tokens", [])]
        if not tokens or len(tokens) != reply["usage"]["completion_tokens"]:
            raise ValueError("Missing or incomplete generated-token IDs")
        row = {**prompt, "generated_ids": tokens, "finish_reason": choice["finish_reason"],
               "usage": reply["usage"], "seconds_completed": time.time()}
        save(row)
        return row

    return await bounded(prompts, settings["continuation_concurrency"], work)


def complete_continuations(result, expected_rows, repeat=False):
    rows = result.get("continuations", [])
    return (expected_rows > 0 and len(rows) == expected_rows
            and len({r.get("index") for r in rows}) == expected_rows
            and all(not r.get("error") and r.get("generated_ids") for r in rows))


@app.function(image=image, gpu=f"RTX-PRO-6000:{GPUS}", cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT,
              retries=0, max_containers=1, block_network=True, volumes={"/assets": assets, "/runs": runs})
def teacher(run_id: str, settings: dict):
    import asyncio
    import subprocess
    import sys
    import threading
    import urllib.request

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=True)
    deadline = settings.get("absolute_deadline", time.time() + TIMEOUT)
    if time.time() >= deadline:
        raise TimeoutError("BF16 job deadline exhausted after restart")
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TAAF_VLLM_GPU_MEMORY_UTILIZATION=settings["utilization"])
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    sys.path[:0] = [str(source / "src")]
    from arc_infra.serving import load_recipe
    from arc_infra.serving_fork import fork_identity, stage_fork_bundle

    record_path = output / "bf16-record.json"
    record = (json.loads(record_path.read_text()) if record_path.exists() else
              {"run_id": run_id, "settings": settings, "bf16_dir": BF16_DIR, "events": []})

    def note(event, **extra):
        row = {"event": event, "t": round(time.time(), 1), **extra}
        record["events"].append(row)
        print(json.dumps(row), flush=True)
        (output / "bf16-record.json").write_text(json.dumps(record, indent=1))
        runs.commit()

    working = Path("/tmp/bf16-work")
    bundle = stage_fork_bundle(Path("/kaggle/input"), working / "fork-bundle")
    recipe = load_recipe(working, Path("/kaggle/input"), mtp_tokens=0, generation_slots=8, max_num_batched_tokens=4096,
                         bundle_dir=bundle, setup_sha256=fork_identity()["fork_sha256"], kv_cache_memory_bytes=0,
                         prefix_caching=False)
    tuning = recipe.resolve_vllm_tuning()
    runtime_dir = recipe.resolve_runtime_dir()
    recipe.verify_and_extract_runtime(runtime_dir, full_layer_hashes=False, scan_extracted_caches=False)
    recipe.patch_ple_layer()
    env, _ = recipe.runtime_environment(deep_preload_validation=False, tuning=tuning)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(GPUS))
    env["NCCL_DEBUG"] = "INFO"
    if settings.get("nccl_p2p_disable"):
        env["NCCL_P2P_DISABLE"] = "1"
    if not settings.get("nccl_cumem_host", True):
        env["NCCL_CUMEM_HOST_ENABLE"] = "0"
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            runs.commit()

    threading.Thread(target=committer, daemon=True).start()
    base = f"http://{recipe.HOST}:{recipe.PORT}"
    from arena.kernel_gate import Client, run_teacher

    prompts = json.loads((source / "configs/kernel-quality/prompts.json").read_text())
    traffic = settings.get("traffic", False)
    texts = prompts["continuations" if traffic else "teacher_forced"]
    models = MODELS[:1] if traffic else MODELS
    complete = complete_continuations if traffic else complete_teacher_artifact
    rows = {}
    try:
        for tag, model, repeat in models:
            artifact = output / f"kernel-quality-{tag}.json"
            if artifact.exists() and complete(json.loads(artifact.read_text()), len(texts), repeat):
                rows[tag] = len(texts)
                note("resume_completed", tag=tag, rows=len(texts))
                continue
            if time.time() >= deadline - 60:
                raise TimeoutError("BF16 job deadline exhausted before next model")
            model_dir = Path(model)
            if not (model_dir / "model.safetensors.index.json").is_file():
                note("model_missing", tag=tag, path=model)
                raise FileNotFoundError(model_dir / "model.safetensors.index.json")
            command = server_argv(recipe.server_command(model_dir, tuning=tuning), model_dir, settings)
            note("start", tag=tag, argv=command)
            log_path = output / f"server-{tag}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            started = time.time()
            ready = False
            while not ready:
                if process.poll() is not None:
                    note("server_exit", tag=tag, code=process.returncode, tail=log_path.read_text()[-4000:])
                    break
                try:
                    urllib.request.urlopen(base + "/v1/models", timeout=5).read()
                    ready = True
                except Exception:  # noqa: BLE001 - not ready yet
                    if time.time() - started > settings["ready_seconds"] or time.time() >= deadline - 60:
                        note("ready_timeout", tag=tag, tail=log_path.read_text()[-4000:])
                        break
                    time.sleep(5)
            if not ready:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError(f"{tag}: server failed to become ready; see {log_path}")
            note("ready", tag=tag, seconds=round(time.time() - started, 1))
            result = {"role": "bf16", "trial": tag, "model_dir": model, "teacher_settings": {"prompt_logprobs": 20},
                      "errors": []}

            async def measure():
                client = Client(base + "/v1", recipe.SERVED_MODEL_NAME, "xhigh")
                try:
                    if traffic:
                        def save(row):
                            (output / f"continuation-{row['index']:03d}.json").write_text(json.dumps(row))
                            runs.commit()
                        result["continuations"] = await run_continuations(client, texts, settings, save)
                        note("continuations", tag=tag, rows=len(result["continuations"]),
                             tokens=sum(len(r.get("generated_ids", [])) for r in result["continuations"]))
                        return
                    result["teacher_forced"] = await run_teacher(client, texts, 1, [t["ids"] for t in texts])
                    note("teacher_forced", tag=tag, seconds=result["teacher_forced"]["seconds"],
                         failed=sum(1 for r in result["teacher_forced"]["rows"] if "error" in r))
                    (output / f"kernel-quality-{tag}.json").write_text(json.dumps(result))
                    if repeat:
                        result["teacher_forced_repeat"] = await run_teacher(
                            client, texts, settings["repeat_concurrency"], [t["ids"] for t in texts])
                        note("teacher_forced_repeat", tag=tag, seconds=result["teacher_forced_repeat"]["seconds"],
                             failed=sum(1 for r in result["teacher_forced_repeat"]["rows"] if "error" in r))
                finally:
                    await client.close()

            try:
                async def bounded_measure():
                    await asyncio.wait_for(measure(), timeout=max(1, deadline - time.time() - 60))

                asyncio.run(bounded_measure())
            finally:
                (output / f"kernel-quality-{tag}.json").write_text(json.dumps(result))
                rows[tag] = len(result.get("continuations", []) if traffic else result.get("teacher_forced", {}).get("rows", []))
                runs.commit()
                process.terminate()
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                time.sleep(20)  # TP workers release the GPUs
            if not complete(result, len(texts), repeat):
                raise RuntimeError(f"{tag}: incomplete or failed teacher measurements")
    finally:
        stop.set()
        note("done", rows=rows)
    if set(rows) != {tag for tag, _, _ in models} or any(count == 0 for count in rows.values()):
        raise RuntimeError(f"Incomplete BF16 measurements: {rows}")
    return {"run_id": run_id, "rows": rows}


def reserve(run_id):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound = bound_usd(data["rates_per_second"])
        committed = data["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in data["runs"])
        arm_committed = sum(r["charged_bound_usd"] for r in data["runs"] if r["run_id"].startswith("tp-"))
        if arm_committed + bound > 1000:
            raise RuntimeError(f"Throughput cap: ${arm_committed:.2f} + ${bound:.2f} required")
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({"run_id": run_id, "state": "reserved", "kind": "bf16-teacher",
                             "charged_bound_usd": round(bound, 4), "max_seconds": TIMEOUT,
                             "gpu": f"RTX-PRO-6000:{GPUS}", "cpu_ceiling": CPU, "memory_ceiling_gib": MEMORY_GIB,
                             "authorization": AUTHORIZATION})
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


@app.local_entrypoint()
def main(run_id: str, traffic: bool = False):
    if not run_id.startswith("tp-"):
        raise ValueError("throughput-arm run ids start with tp-")
    SETTINGS.update(absolute_deadline=time.time() + TIMEOUT, traffic=traffic, continuation_tokens=2048,
                    continuation_concurrency=4)
    bound = reserve(run_id)
    record = REPO / RECORDS / f"modal-{run_id}.json"
    data = {"state": "submission_intent", "run_id": run_id, "kind": "bf16-teacher", "charged_bound_usd": bound,
            "bundle": str(BUNDLE), "bf16_dir": BF16_DIR, "settings": SETTINGS, "ledger": str(ROOT / "docs/spend.json")}
    record.write_text(json.dumps(data, indent=2))
    call = teacher.spawn(run_id, SETTINGS)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
