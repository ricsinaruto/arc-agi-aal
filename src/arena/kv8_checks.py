"""Checks for the 8-bit QSA KV cache smoke (docs/validation-20260927/kv-cache-8bit.md), run by the serving bench.

- ``quality``: the kernel gate's greedy and teacher-forced measurements (``arena.kernel_gate``) on the 8-bit
  server, on the exact token ids of a recorded BF16-cache reference (the phase-2 A+C gate trial), compared
  with it; plus a teacher-forced repeat at another concurrency (this server's own run-to-run noise). No
  thresholds: the doc reads the drift against the recorded control noise (KL ~0.09 nats, kernels-phase2.md).
- ``microbench``: ``python -m vllm.arc_kv8 bench`` in the server's runtime environment (recorded by
  ``arena.serving.configure`` as ``kv8-runtime-env.json``) while the server idles: the QSA sparse-attention
  kernel per call, pinned BF16 against int8 / fp8 codes, errors against FP32 and a writer round trip.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path

from arc_infra.serving import MODEL

from . import kernel_gate as gate

RUNTIME_ENV = "kv8-runtime-env.json"
# Environment keys the microbenchmark subprocess may inherit from the server's runtime environment (no tokens).
ENV_PREFIXES = ("CUDA", "TRITON", "TORCH", "PYTORCH", "LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "HOME", "TMPDIR",
                "XDG_CACHE_HOME", "HF_HOME", "FLASHINFER", "NVIDIA", "VLLM_ENABLE_CUDA_COMPATIBILITY")
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def runtime_env_subset(env):
    return {k: v for k, v in env.items()
            if k.startswith(ENV_PREFIXES) and not any(m in k.upper() for m in SECRET_MARKERS)}


def record_runtime_env(working, python, env):
    """Called from the serving process when the kv8 overlay is on (``arena.serving.configure``)."""
    path = Path(working) / RUNTIME_ENV
    path.write_text(json.dumps({"python": str(python), "env": runtime_env_subset(env)}, indent=2, sort_keys=True))
    return path


async def quality(study, spec, *, source, output, base_url, log=print, transport=None):
    started = time.monotonic()
    prompts, _ = gate.load_inputs(source)
    reference = json.loads((Path(source) / spec["reference"]).read_text())
    client = gate.Client(base_url, study.get("served_model", MODEL), study.get("reasoning_effort", "xhigh"),
                         transport=transport)
    result = {"reference": spec["reference"], "reference_server": reference.get("server"),
              "reference_server_env": reference.get("server_env"), "errors": []}
    try:
        greedy_ids = {row["index"]: row["prompt_ids"] for row in reference["greedy"]["rows"] if "prompt_ids" in row}
        tokens = {row["index"]: row["tokens"] for row in reference["teacher_forced"]["rows"] if "tokens" in row}
        ids = [tokens.get(t["index"]) for t in prompts["teacher_forced"]]
        steps = [("greedy", gate.run_greedy(client, prompts["greedy"], gate.GREEDY["concurrency"], greedy_ids)),
                 ("teacher_forced", gate.run_teacher(client, prompts["teacher_forced"], gate.TEACHER["concurrency"],
                                                     ids if all(ids) else None))]
        if spec.get("repeat_teacher", True):
            steps.append(("teacher_forced_repeat", gate.run_teacher(
                client, prompts["teacher_forced"], gate.TEACHER["repeat_concurrency"], ids if all(ids) else None)))
        for name, coroutine in steps:
            t0 = time.monotonic()
            try:
                result[name] = await coroutine
            except Exception as exc:  # noqa: BLE001 - a failed measurement is recorded
                result[name] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}", "rows": []}
                result["errors"].append(f"{name}: {result[name]['error']}")
            rows = result[name].get("rows", [])
            log(json.dumps({"event": "kv8_quality", "step": name, "seconds": round(time.monotonic() - t0, 1),
                            "rows": len(rows), "failed": sum(1 for r in rows if "error" in r)}), flush=True)
    finally:
        await client.close()
    summary = {
        "vs_reference": {"teacher_forced": gate.compare_teacher(reference["teacher_forced"],
                                                                result.get("teacher_forced", {})),
                         "greedy": gate.compare_greedy(reference["greedy"], result.get("greedy", {}))},
        "self_repeat": {"teacher_forced": gate.compare_teacher(result.get("teacher_forced", {}),
                                                               result.get("teacher_forced_repeat", {}))},
        "errors": result["errors"], "seconds": round(time.monotonic() - started, 1),
    }
    result["summary"] = summary
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "kv8-quality.json").write_text(json.dumps(result))
    return summary


def microbench(server_dir, output, *, timeout=420):
    record = Path(server_dir) / RUNTIME_ENV
    if not record.is_file():
        return {"status": "skipped", "reason": f"no {RUNTIME_ENV} (kv8 overlay off?)"}
    runtime = json.loads(record.read_text())
    target = Path(output) / "kv8-microbench.json"
    started = time.monotonic()
    try:
        completed = subprocess.run([runtime["python"], "-u", "-m", "vllm.arc_kv8", "bench", str(target)],
                                   env=runtime["env"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "seconds": timeout, "stdout_tail": str(exc.stdout or "")[-3000:]}
    (Path(output) / "kv8-microbench.log").write_text(completed.stdout + "\n" + completed.stderr)
    result = {"status": "complete" if completed.returncode == 0 and target.is_file() else "error",
              "returncode": completed.returncode, "seconds": round(time.monotonic() - started, 1)}
    if target.is_file():
        result["report"] = json.loads(target.read_text())
    else:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


async def microbench_async(server_dir, output, *, timeout=420):
    return await asyncio.to_thread(microbench, server_dir, output, timeout=timeout)
