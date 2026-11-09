"""ARC-tuned routed-expert quantization trials (28 September; docs/validation-20260927/expert-quant.md).

An ``expert-quant`` trial runs on a server started with ``server.expert_capture`` (``arc_infra.expert_capture_patch``):
the MoE input rows and the served routing of selected layers are written to disk while fixed recorded arena
sequences (``configs/expert-quant``, ``scripts/expert_quant_inputs.py``) are prefilled through ``/v1/completions``
(token ids, ``max_tokens`` 1). Then the trial stops the server (the GPU is needed) and runs a GPU program in the
pinned serving runtime, with the environment of the stopped server process:

- phase ``bench`` (stage 1): ``arena.expert_quant_bench`` on the captured rows of a few layers;
- phase ``calibrate`` (stage 2): ``arena.expert_quant_gptq`` over all 48 layers.

The captured rows are also copied to the Modal run volume (``ARC_VALIDATION_OUTPUT``) so a failed GPU program can be
rerun without a new capture.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from arc_infra.serving import MODEL

INPUTS = "configs/expert-quant"
STOP_PROGRAM = ("from arc_infra.serving import load_recipe; import sys; "
                "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")
RUNTIME_KEYS_DROP = ("VLLM_PORT", "VLLM_HOST_IP")


class Client:
    def __init__(self, base_url, model, transport=None):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url.removesuffix("/v1")
        self.model = model
        self.http = httpx.AsyncClient(timeout=1800, trust_env=False, transport=transport,
                                      limits=httpx.Limits(max_connections=16))

    async def close(self):
        await self.http.aclose()

    async def rpc(self, timeout=600, **command):
        response = await self.http.post(self.root + "/collective_rpc", json={
            "method": "arc_expert_capture", "args": [json.dumps(command)], "timeout": timeout})
        if response.status_code != 200:
            raise RuntimeError(f"collective_rpc {response.status_code}: {response.text[:500]}")
        reply = response.json()["results"][0]
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(f"expert capture {command.get('op')}: {str(reply)[:500]}")
        return reply

    async def prefill(self, ids):
        response = await self.http.post(self.base_url + "/completions", json={
            "model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0.0})
        if response.status_code != 200:
            raise RuntimeError(f"completions {response.status_code}: {response.text[:300]}")
        return response.json().get("usage") or {}


async def capture(client, rows, *, directory, layers, stride, fmt, concurrency, max_rows=None, deadline=None):
    """Prefill ``rows`` with the capture on; returns the worker's summary plus request accounting."""
    await client.rpc(op="start", dir=str(directory), layers=layers, stride=stride, format=fmt, max_rows=max_rows)
    started = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)
    done, failed, prompt_tokens = [], [], 0

    async def one(row):
        nonlocal prompt_tokens
        async with semaphore:
            if deadline is not None and time.time() > deadline:
                failed.append({"index": row["index"], "error": "deadline"})
                return
            try:
                usage = await client.prefill(row["ids"])
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                done.append(row["index"])
            except Exception as exc:  # noqa: BLE001 - one failed request is recorded
                failed.append({"index": row["index"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    try:
        await asyncio.gather(*(one(row) for row in rows))
    finally:
        summary = await client.rpc(op="stop", timeout=1700, drain_seconds=1500)
    summary.update(requests=len(done), failed=failed, prompt_tokens=prompt_tokens,
                   wall_seconds=round(time.monotonic() - started, 1))
    return summary


def server_process_env(server_dir):
    """The environment of the running vLLM server process (the pinned runtime's paths, caches and flags)."""
    identity = json.loads((Path(server_dir) / "vllm-server-identity.json").read_text())
    raw = Path(f"/proc/{int(identity['pid'])}/environ").read_bytes()
    env = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, _, value = item.partition(b"=")
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    for key in RUNTIME_KEYS_DROP:
        env.pop(key, None)
    return env, identity


def model_dir_of(identity):
    argv = identity["argv"]
    return argv[argv.index("serve") + 1]


def stop_server(server_dir, input_root="/kaggle/input"):
    started = time.monotonic()
    completed = subprocess.run([sys.executable, "-c", STOP_PROGRAM, str(server_dir), input_root],
                               check=False, timeout=180, capture_output=True, text=True)
    return {"returncode": completed.returncode, "seconds": round(time.monotonic() - started, 1),
            "stderr": completed.stderr[-500:]}


def persist_async(source, name):
    """Copy ``source`` to ``$ARC_VALIDATION_OUTPUT/<name>`` in a thread (None when not on Modal)."""
    root = os.environ.get("ARC_VALIDATION_OUTPUT")
    if not root:
        return None, None
    target = Path(root) / name
    state = {}

    def work():
        started = time.monotonic()
        try:
            shutil.copytree(source, target, dirs_exist_ok=True)
            state.update(target=str(target), seconds=round(time.monotonic() - started, 1))
        except Exception as exc:  # noqa: BLE001
            state.update(error=f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread, state


def run_program(module, arguments, *, env, source, log_path, timeout, echo_prefix='{"event": "expert_quant'):
    """Run ``python -m module`` in the runtime environment, echoing its progress lines."""
    env = dict(env)
    env["PYTHONPATH"] = os.pathsep.join([p for p in (env.get("PYTHONPATH"), str(Path(source) / "src")) if p])
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-u", "-m", module, *arguments]
    started = time.monotonic()
    with open(log_path, "ab") as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def pump():
            for line in iter(process.stdout.readline, b""):
                stream.write(line)
                stream.flush()
                if line.startswith(echo_prefix.encode()):
                    sys.stdout.write(line.decode(errors="replace"))
                    sys.stdout.flush()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            code = process.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = "timeout"
        thread.join(timeout=10)
    return {"command": command[3:], "returncode": code, "seconds": round(time.monotonic() - started, 1)}


async def run_expert_quant(study, trial, *, source, output, server_dir, base_url, deadline_epoch, log=print,
                           transport=None):
    settings = study["expert_quant"]
    phase = trial.get("phase")
    if phase not in ("bench", "calibrate"):
        raise ValueError(f"unknown expert-quant phase {phase!r}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    working = Path(server_dir).parent
    cap_root = working / "expert-capture"
    report = {"phase": phase, "settings": settings, "errors": [], "status": "error"}
    path = output / "expert-quant.json"

    def save():
        path.write_text(json.dumps(report, indent=1, default=str))

    def event(name, **extra):
        log(json.dumps({"event": f"expert_quant_{name}", **extra}, default=str), flush=True)

    usage = shutil.disk_usage(working)
    report["disk"] = {"path": str(working), "total_bytes": usage.total, "free_bytes": usage.free}
    inputs = Path(source) / INPUTS
    if phase == "bench":
        rows = json.loads((inputs / "stage1.json").read_text())
        passes = [(split, [r for r in rows if r["split"] == split], settings["layers"], settings["stride"], "bf16",
                   None) for split in ("calib", "eval")]
    else:
        passes = [("calib", json.loads((inputs / "calib.json").read_text()), None, settings["calib_stride"], "fp8",
                   None),
                  ("eval", json.loads((inputs / "eval.json").read_text()), None, settings["eval_stride"], "bf16",
                   settings["eval_max_rows"])]
    client = Client(base_url, study.get("served_model", MODEL), transport=transport)
    try:
        for split, chosen, layers, stride, fmt, max_rows in passes:
            summary = await capture(client, chosen, directory=cap_root / split, layers=layers, stride=stride, fmt=fmt,
                                    concurrency=settings["concurrency"], max_rows=max_rows,
                                    deadline=deadline_epoch - settings["gpu_reserve_seconds"])
            report[f"capture_{split}"] = summary
            event("capture", split=split, rows=len(summary.get("rows") or {}), bytes=summary.get("bytes"),
                  requests=summary["requests"], failed=len(summary["failed"]), prompt_tokens=summary["prompt_tokens"],
                  seconds=summary["wall_seconds"])
            save()
    finally:
        await client.close()
    try:
        env, identity = server_process_env(server_dir)
        report["runtime_env"] = "server process"
    except OSError as exc:
        # Fallback: the runtime keys the serving script persisted for analyzers (no cache paths).
        identity = json.loads((Path(server_dir) / "vllm-server-identity.json").read_text())
        env = {**os.environ, **json.loads((Path(server_dir) / "serving-environment.json").read_text())}
        report["runtime_env"] = f"serving-environment.json ({type(exc).__name__})"
    report["server_pid"] = identity.get("pid")
    report["stop"] = stop_server(server_dir)
    event("server_stopped", **report["stop"])
    # The small captures are persisted for reruns of the GPU program (the stage-2 calibration capture stays local).
    copier, copied = persist_async(cap_root if phase == "bench" else cap_root / "eval",
                                   "expert-capture" if phase == "bench" else "expert-capture/eval")
    remaining = deadline_epoch - time.time() - 15
    common = ["--capture", str(cap_root), "--model-dir", model_dir_of(identity), "--bf16-dir", settings["bf16_dir"],
              "--deadline-seconds", str(int(remaining - 10)), "--eval-rows", str(settings["eval_rows"])]
    if phase == "bench":
        args = [*common, "--layers", ",".join(str(x) for x in settings["layers"]), "--out", str(output / "bench.json")]
        if settings.get("rtn_search"):
            args.append("--rtn-search")
        report["program"] = run_program("arena.expert_quant_bench", args, env=env, source=source,
                                        log_path=output / "bench.log", timeout=remaining)
    else:
        args = [*common, "--calib", str(cap_root / "calib"), "--overlay-out", settings["overlay_dir"],
                "--out", str(output / "gptq.json"), "--percdamp", str(settings["percdamp"]),
                "--min-rows", str(settings["min_rows"]), "--batch", str(settings["batch"]),
                "--scale-search", *[str(x) for x in settings["scale_search"]]]
        # Phase 3 (29 September): pooled-Hessian prior, act-order with static scales, mask-aware calibration on a
        # load-time-pruned server (the capture routes in the kept experts' ranks).
        if settings.get("prior_rows"):
            args += ["--prior-rows", str(settings["prior_rows"])]
        if settings.get("act_order"):
            args.append("--act-order")
        prune = study["server"].get("expert_prune")
        if prune:
            args += ["--prune-mask", str(Path(source) / prune["mask"])]
            if settings.get("omit_pruned"):
                args.append("--omit-pruned")
        report["program"] = run_program("arena.expert_quant_gptq", args, env=env, source=source,
                                        log_path=output / "gptq.log", timeout=remaining)
    event("program_done", **report["program"])
    if copier is not None:
        copier.join(timeout=max(1, deadline_epoch - time.time() - 5))
        report["persisted"] = copied or {"error": "copy still running at the deadline"}
    result_path = output / ("bench.json" if phase == "bench" else "gptq.json")
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        report["program_errors"] = result.get("errors")
        report["layers_done"] = sorted(int(k) for k, v in result.get("layers", {}).items()
                                       if v.get("errors") or v.get("shards"))
        if phase == "calibrate":
            report["overlay_complete"] = bool((result.get("manifest") or {}).get("complete"))
    ok = report.get("overlay_complete") if phase == "calibrate" else report.get("layers_done")
    report["status"] = "complete" if ok else "error"
    save()
    return report
