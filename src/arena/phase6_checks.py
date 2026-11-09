"""Kernel work, phase 6 (docs/validation-20260927/kernels-phase6.md): the kernel microbenchmarks
(``arc_infra.phase6_microbench``) in the server's runtime environment while the server idles, run by the serving
bench when its grid sets ``phase6_microbench``. Same subprocess contract as the kv8 microbenchmark."""

import asyncio
import json
import subprocess
import time
from pathlib import Path

from .kv8_checks import RUNTIME_ENV

IDENTITY = "vllm-server-identity.json"
PARTS = ("dense", "qsa", "w4")


def validate(spec):
    if not isinstance(spec, dict) or set(spec) - {"parts", "budget_seconds", "timeout"}:
        raise ValueError("phase6_microbench takes parts, budget_seconds and timeout")
    parts = list(spec.get("parts", PARTS))
    if not parts or set(parts) - set(PARTS) or len(set(parts)) != len(parts):
        raise ValueError(f"phase6_microbench.parts must be distinct parts from {PARTS}")
    budget, timeout = float(spec.get("budget_seconds", 400)), int(spec.get("timeout", 1500))
    if not (30 <= budget <= 1200 and budget * len(parts) < timeout <= 2400):
        raise ValueError("phase6_microbench: 30 <= budget_seconds <= 1200 and parts x budget < timeout <= 2400")
    return {"parts": parts, "budget_seconds": budget, "timeout": timeout}


def model_dir(server_dir):
    """The served checkpoint directory: the argument after ``serve`` in the server's recorded argv."""
    argv = json.loads((Path(server_dir) / IDENTITY).read_text())["argv"]
    return argv[argv.index("serve") + 1]


def command(runtime, target, spec, model):
    args = [runtime["python"], "-u", "-m", "arc_infra.phase6_microbench", str(target), "--parts",
            ",".join(spec["parts"]), "--budget-seconds", str(spec["budget_seconds"])]
    if "w4" in spec["parts"]:
        args += ["--model", str(model)]
    return args


def microbench(server_dir, output, spec, *, timeout=None):
    spec = validate(spec)
    record = Path(server_dir) / RUNTIME_ENV
    if not record.is_file():
        return {"status": "skipped", "reason": f"no {RUNTIME_ENV} (kv8 overlay off?)"}
    runtime = json.loads(record.read_text())
    target = Path(output) / "phase6-microbench.json"
    timeout = min(spec["timeout"], timeout or spec["timeout"])
    started = time.monotonic()
    try:
        model = model_dir(server_dir) if "w4" in spec["parts"] else None
        completed = subprocess.run(command(runtime, target, spec, model), env=runtime["env"], capture_output=True,
                                   text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        result = {"status": "timeout", "seconds": timeout, "stdout_tail": str(exc.stdout or "")[-3000:]}
        if target.is_file():
            result["report"] = json.loads(target.read_text())  # the parts written before the timeout
        return result
    (Path(output) / "phase6-microbench.log").write_text(completed.stdout + "\n" + completed.stderr)
    result = {"status": "complete" if completed.returncode == 0 and target.is_file() else "error",
              "returncode": completed.returncode, "seconds": round(time.monotonic() - started, 1)}
    if target.is_file():
        result["report"] = json.loads(target.read_text())
    else:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


async def microbench_async(server_dir, output, spec, *, timeout=None):
    return await asyncio.to_thread(microbench, server_dir, output, spec, timeout=timeout)
