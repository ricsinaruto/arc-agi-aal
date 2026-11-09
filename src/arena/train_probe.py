"""Training-cost probe trial (RL restart step 0, 11 October): stop the server, then run ``arena.train_probe_job`` once
per weight storage in the serving runtime's Python, on the served checkpoint. The trial ends in the time it is given;
a variant that fails costs only its own rows.

``study["train_probe"]``: ``variants`` ([{"storage", "lengths", "repeat_below", "share"}]: ``share`` of the time left
when the variant starts), ``label_fraction``, ``head_chunk`` and ``wheels`` (bundle paths of pure-Python wheels added
to the job's path: fla-core, the Triton kernels of the linear-attention layers).
"""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from .draft_head import STOP_PROGRAM, model_dir, runtime_env, wait_gpu_free

DEFAULTS = {"label_fraction": 0.5, "head_chunk": 2048, "wheels": [], "variants": []}


def emit(name, **fields):
    print(json.dumps({"event": "train_probe_" + name, **fields}), flush=True)


def settings(raw):
    merged = {**DEFAULTS, **(raw or {})}
    if not merged["variants"]:
        raise ValueError("train_probe.variants is empty")
    for variant in merged["variants"]:
        if variant.get("storage") not in ("bf16", "fp32") or not variant.get("lengths"):
            raise ValueError("a train_probe variant is {storage: bf16|fp32, lengths: [...]}")
        if not 0 < float(variant.get("share", 1)) <= 1:
            raise ValueError("train_probe share is in (0, 1]")
    if not 0 < merged["label_fraction"] <= 1:
        raise ValueError("train_probe.label_fraction is in (0, 1]")
    return merged


def runtime_python(server_dir):
    """The serving runtime's interpreter (the server's argv[0]): the bootstrap's own Python has no torch (step 0, 11 Oct)."""
    identity = Path(server_dir) / "vllm-server-identity.json"
    argv = json.loads(identity.read_text()).get("argv") if identity.is_file() else None
    return argv[0] if argv and Path(argv[0]).is_file() else sys.executable


def job_command(model, variant, cfg, out, deadline, python=None):
    return [python or sys.executable, "-u", "-m", "arena.train_probe_job", "--model-dir", str(model), "--out", str(out),
            "--storage", variant["storage"], "--lengths", *map(str, variant["lengths"]),
            "--repeat-below", str(variant.get("repeat_below", 0)), "--label-fraction", str(cfg["label_fraction"]),
            "--head-chunk", str(cfg["head_chunk"]), "--deadline-epoch", str(deadline)]


def run_job(model, variant, cfg, output, deadline, env, python=None):
    out = output / f"train-probe-{variant['storage']}.json"
    out.unlink(missing_ok=True)
    with (output / f"train-probe-{variant['storage']}.log").open("ab") as log:
        process = subprocess.Popen(job_command(model, variant, cfg, out, deadline, python), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b""):
            log.write(line)
            log.flush()
            if line.startswith(b'{"event": "train_probe'):
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
        code = process.wait()
    return {"storage": variant["storage"], "returncode": code,
            "report": json.loads(out.read_text()) if out.is_file() else None}


def run_train_probe(study, trial, *, source, output, server_dir, deadline_epoch):
    cfg = settings(study.get("train_probe"))
    output = Path(output)
    record = {"kind": "train-probe", "status": "running", "settings": cfg, "variants": [], "errors": []}
    path = output / "train-probe.json"

    def save():
        path.write_text(json.dumps(record, indent=1))

    started = time.time()
    try:
        model = model_dir(server_dir)
        if not (Path(model) / "config.json").is_file():
            raise FileNotFoundError(f"no served checkpoint at {model}")
        record["model_dir"] = str(model)
        stop = subprocess.run([sys.executable, "-c", STOP_PROGRAM, str(server_dir), "/kaggle/input"],
                              env={**os.environ, "PYTHONPATH": str(Path(source) / "src")}, capture_output=True,
                              text=True, timeout=240)
        record["server_stop"] = {"returncode": stop.returncode, "tail": (stop.stdout + stop.stderr)[-1500:],
                                 "gpu_used_mib": wait_gpu_free()}
        if record["server_stop"]["gpu_used_mib"] > 8192:
            subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
            record["server_stop"]["killed_leftovers"] = True
            record["server_stop"]["gpu_used_mib_after_kill"] = wait_gpu_free()
        emit("server_stopped", **{k: v for k, v in record["server_stop"].items() if k != "tail"})
        python = runtime_python(server_dir)
        record["python"] = python
        plain = runtime_env(server_dir, source)
        env, extra = dict(plain), output / "wheels"
        for wheel in cfg["wheels"]:
            with zipfile.ZipFile(Path(source) / wheel) as archive:
                archive.extractall(extra)
        if cfg["wheels"]:
            env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], str(extra)])
        save()
        for variant in cfg["variants"]:
            left = deadline_epoch - time.time() - 20
            if left < 60:
                record["errors"].append(f"{variant['storage']}: no time left")
                continue
            deadline = time.time() + left * float(variant.get("share", 1))
            entry = run_job(model, variant, cfg, output, deadline, env, python)
            if entry["returncode"] and not (entry["report"] or {}).get("rows") and env is not plain:
                # The added wheels broke the job before its first length: the runtime alone (reference torch path).
                record["errors"].append(f"{variant['storage']}: failed with the added wheels, rerun without them")
                env = plain
                entry = {**run_job(model, variant, cfg, output, deadline, env, python), "wheels_dropped": True}
            record["variants"].append(entry)
            if entry["returncode"]:
                record["errors"].append(f"{variant['storage']}: job exited {entry['returncode']}; see its log")
            wait_gpu_free()
            save()
        done = [v for v in record["variants"] if v["report"] and v["report"].get("rows")]
        record["status"] = "complete" if done and not record["errors"] else "complete_with_errors" if done else "error"
    except Exception as exc:  # noqa: BLE001 - recorded, the trial reports an error status
        record["status"] = "error"
        record["errors"].append(f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        record["elapsed_seconds"] = round(time.time() - started, 1)
        save()
    return record
