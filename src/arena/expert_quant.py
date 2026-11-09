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


def balanced_prefix_rows(rows, tokens_per_family):
    """Nested corpus budgets, preserving sequence prefixes and the recorded family order."""
    if type(tokens_per_family) is not int or tokens_per_family <= 0:
        raise ValueError("calib_tokens_per_family must be a positive integer")
    totals = {row["family"]: 0 for row in rows}
    chosen = []
    for row in rows:
        family = row["family"]
        count = min(len(row["ids"]), tokens_per_family - totals[family])
        if count <= 0:
            continue
        spans = [[a, min(b, count)] for a, b in row.get("assistant_spans", []) if a < count]
        chosen.append({**row, "ids": row["ids"][:count], "tokens": count, "assistant_spans": spans,
                       "assistant_tokens": sum(b - a for a, b in spans)})
        totals[family] += count
    if not totals or any(n != tokens_per_family for n in totals.values()):
        raise ValueError(f"insufficient calibration tokens per family: {totals}")
    return chosen, totals


def calibration_role_weights(rows, summary, assistant_weight):
    """Label sampled predictor positions only when a serial, uncached capture matches every input token."""
    import math

    if not math.isfinite(assistant_weight) or assistant_weight <= 0:
        raise ValueError("assistant weight must be finite and positive")
    total = sum(len(r["ids"]) for r in rows)
    if (not total or summary.get("failed") or summary.get("dropped_rows") or summary.get("write_errors")
            or summary.get("completed_indices") != [r["index"] for r in rows]
            or summary.get("prompt_tokens") != total):
        raise ValueError("role labels require complete serial capture in source order")
    counts = summary.get("rows", {})
    processed = summary.get("processed_tokens", {})
    stride = summary["stride"]
    expected = (total + stride - 1) // stride
    if not counts or any(processed.get(k) != total or n != expected for k, n in counts.items()):
        raise ValueError("role labels require exact token and sampled-row counts; no cache reuse or truncation")
    role = bytearray(total)
    offset = 0
    for row in rows:
        for start, end in row["assistant_spans"]:
            if not 0 <= start <= end <= len(row["ids"]):
                raise ValueError("assistant span outside input")
            # The hidden state at t predicts token t+1. Never label across sequence boundaries.
            a, b = max(0, start - 1), max(0, end - 1)
            role[offset + a:offset + b] = bytes([1]) * (b - a)
        offset += len(row["ids"])
    values = [assistant_weight if role[i] else 1.0 for i in range(0, total, stride)]
    mean = sum(values) / len(values)
    return [v / mean for v in values]


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
    summary.update(requests=len(done), failed=failed, prompt_tokens=prompt_tokens, completed_indices=done,
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


def mix_calibration_rows(primary, extra, destination, fraction, seed=0):
    """Replace a fixed fraction of captured FP8 rows, preserving the baseline row budget at every layer."""
    import numpy as np

    if not 0 <= fraction <= 1:
        raise ValueError("calibration mix fraction must be in [0, 1]")
    roots = [Path(primary), Path(extra)]
    metas = [json.loads((r / "meta.json").read_text()) for r in roots]
    if (any(m["format"] != "fp8" or m.get("dropped_rows") or m.get("write_errors")
            or m.get("row_importance") for m in metas)
            or metas[0]["rows"].keys() != metas[1]["rows"].keys()
            or metas[0]["shapes"] != metas[1]["shapes"]):
        raise ValueError("calibration mix requires compatible complete, unweighted FP8 captures")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    counts = {}
    for layer, n in metas[0]["rows"].items():
        added = round(n * fraction)
        if n <= 0 or added > metas[1]["rows"][layer]:
            raise ValueError(f"insufficient mixture rows in layer {layer}")
        counts[layer] = [n - added, added]
        picks = [np.sort(np.random.default_rng(seed).choice(m["rows"][layer], count, replace=False))
                 for m, count in zip(metas, counts[layer])]
        shape = metas[0]["shapes"][layer]
        for ext, dtype, width in (("x", np.uint8, shape["hidden"]), ("s", np.float32, 1),
                                  ("ids", np.int16, shape["topk"]), ("w", np.float32, shape["topk"])):
            chunks = []
            for root, meta, pick in zip(roots, metas, picks):
                path = root / f"L{int(layer):02d}.{ext}"
                rows = meta["rows"][layer]
                if path.stat().st_size != rows * width * np.dtype(dtype).itemsize:
                    raise ValueError(f"capture file size mismatch: {path}")
                data = np.memmap(path, mode="r", dtype=dtype, shape=(rows, width))
                chunks.append(data[pick])
                del data
            np.concatenate(chunks).tofile(destination / f"L{int(layer):02d}.{ext}")
    mixture = {"fraction": fraction, "seed": seed, "rows": counts, "sources": [str(r) for r in roots],
               "source_processed_tokens": [m.get("processed_tokens") for m in metas]}
    meta = {k: v for k, v in metas[0].items() if k not in ("processed_tokens", "bytes", "dir")}
    meta["calibration_row_mix"] = mixture
    (destination / "meta.json").write_text(json.dumps(meta, indent=1))
    return mixture


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
        # saturation (6 October): capture only these layers; the program then runs once per calibration fraction.
        sat_layers = (settings.get("saturation") or {}).get("layers")
        passes = [("calib", json.loads((inputs / "calib.json").read_text()), sat_layers, settings["calib_stride"],
                   "fp8", None),
                  ("eval", json.loads((inputs / "eval.json").read_text()), sat_layers, settings["eval_stride"], "bf16",
                   settings["eval_max_rows"])]
    mixture = settings.get("calibration_mix")
    if mixture:
        if phase != "calibrate" or settings.get("assistant_weight") is not None:
            raise ValueError("row mixtures require unweighted calibration captures")
        passes.insert(1, ("calib_extra", json.loads((inputs / mixture["file"]).read_text()), sat_layers,
                          mixture.get("stride", settings["calib_stride"]), "fp8", None))
    client = Client(base_url, study.get("served_model", MODEL), transport=transport)
    try:
        for split, chosen, layers, stride, fmt, max_rows in passes:
            if split == "calib_extra" or (split == "eval" and settings.get("reset_eval_prefix_cache")):
                from arena.isolation import isolate_trial

                report[f"{split}_cache_isolation"] = await isolate_trial(base_url)
            if split == "calib" and settings.get("calib_tokens_per_family") is not None:
                chosen, totals = balanced_prefix_rows(chosen, settings["calib_tokens_per_family"])
                report["corpus_budget"] = {"tokens": sum(totals.values()), "per_family": totals,
                                          "sequences": len(chosen), "selection": "nested sequence prefixes"}
                save()
            summary = await capture(client, chosen, directory=cap_root / split, layers=layers, stride=stride, fmt=fmt,
                                    concurrency=settings.get("eval_concurrency", settings["concurrency"])
                                    if split == "eval" else settings["concurrency"], max_rows=max_rows,
                                    deadline=deadline_epoch - settings["gpu_reserve_seconds"])
            if split == "calib" and settings.get("assistant_weight") is not None:
                if settings["concurrency"] != 1 or study["server"].get("prefix_caching", True):
                    raise ValueError("assistant-weighted calibration requires concurrency1 and prefix_caching=false")
                from array import array

                weights = calibration_role_weights(chosen, summary, settings["assistant_weight"])
                for layer in summary["rows"]:
                    with (cap_root / split / f"L{int(layer):02d}.importance").open("wb") as handle:
                        array("f", weights).tofile(handle)
                meta_path = cap_root / split / "meta.json"
                meta = json.loads(meta_path.read_text())
                meta["row_importance"] = {"assistant_weight": settings["assistant_weight"],
                                          "normalized_mean": 1.0, "predictor_offset": -1}
                meta_path.write_text(json.dumps(meta, indent=1))
                summary["row_importance"] = meta["row_importance"]
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
    calib_path = cap_root / "calib"
    if mixture:
        if any(report[f"capture_{split}"].get("failed") for split in ("calib", "calib_extra")):
            raise RuntimeError("calibration mixture capture had failed requests")
        report["calibration_row_mix"] = mix_calibration_rows(
            calib_path, cap_root / "calib_extra", cap_root / "calib_mixed", mixture["fraction"],
            mixture.get("seed", 0))
        calib_path = cap_root / "calib_mixed"
        save()
    # Full calibration captures are opt-in; two-layer screens can reuse them without another server startup.
    persist_both = phase == "bench" or settings.get("persist_calibration_capture") or settings.get("capture_only")
    copier, copied = persist_async(cap_root if persist_both else cap_root / "eval",
                                   "expert-capture" if persist_both else "expert-capture/eval")
    if settings.get("capture_only"):
        if copier is None:
            raise RuntimeError("capture_only requires a persistent validation output")
        copier.join(timeout=max(1, deadline_epoch - time.time() - 5))
        report["persisted"] = copied or {"error": "copy still running at the deadline"}
        valid = all(report[f"capture_{split}"].get("rows")
                    and not any(report[f"capture_{split}"].get(k)
                                for k in ("failed", "dropped_rows", "write_errors"))
                    for split in ("calib", "eval"))
        report["status"] = "complete" if valid and not report["persisted"].get("error") else "error"
        report["capture_only"] = True
        save()
        return report
    remaining = deadline_epoch - time.time() - 15
    report["capture_model_dir"] = model_dir_of(identity)
    # Quantized capture must not change the reference NVFP4 grid used by a paired W4 refit.
    report["reference_model_dir"] = settings.get("reference_model_dir", report["capture_model_dir"])
    common = ["--capture", str(cap_root), "--model-dir", report["reference_model_dir"], "--bf16-dir", settings["bf16_dir"],
              "--deadline-seconds", str(int(remaining - 10)), "--eval-rows", str(settings["eval_rows"])]
    if phase == "bench":
        args = [*common, "--layers", ",".join(str(x) for x in settings["layers"]), "--out", str(output / "bench.json")]
        if settings.get("rtn_search"):
            args.append("--rtn-search")
        report["program"] = run_program("arena.expert_quant_bench", args, env=env, source=source,
                                        log_path=output / "bench.log", timeout=remaining)
    else:
        args = [*common, "--calib", str(calib_path), "--overlay-out", settings["overlay_dir"],
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
        if settings.get("int3_overlay_dir"):  # 6 October: direct GPTQ-int3 beside the W4A16 overlay
            args += ["--int3-out", settings["int3_overlay_dir"], "--int3-group", str(settings.get("int3_group", 16))]
        if settings.get("int3_scale_sign_search"):
            args.append("--int3-scale-sign-search")
        if settings.get("int3_scale_metric"):
            args += ["--int3-scale-metric", settings["int3_scale_metric"]]
        saturation = settings.get("saturation")
        if saturation:
            # Held-out error vs calibration amount: uniform row subsets of the same capture, nothing published.
            report["saturation"] = []
            sat_args = [a for a in args if a != "--omit-pruned"]
            sat_args[sat_args.index("--overlay-out") + 1] = str(working / "saturation-overlay")
            for fraction in saturation["fractions"]:
                left = deadline_epoch - time.time() - 30
                run = run_program("arena.expert_quant_gptq",
                                  [*sat_args, "--layers", ",".join(str(x) for x in saturation["layers"]),
                                   "--calib-subsample", str(fraction), "--out", str(output / f"gptq-{fraction}.json")]
                                  + (["--int3-out", str(working / "saturation-int3")] if settings.get("int3_group")
                                     else []),
                                  env=env, source=source, log_path=output / "gptq.log", timeout=left)
                report["saturation"].append({"fraction": fraction, **run})
                save()
            report["program"] = report["saturation"][-1]
        else:
            report["program"] = run_program("arena.expert_quant_gptq", args, env=env, source=source,
                                            log_path=output / "gptq.log", timeout=remaining)
    event("program_done", **report["program"])
    if copier is not None:
        copier.join(timeout=max(1, deadline_epoch - time.time() - 5))
        report["persisted"] = copied or {"error": "copy still running at the deadline"}
    result_path = output / ("bench.json" if phase == "bench" else "gptq.json")
    if phase == "calibrate" and settings.get("saturation"):
        done = [f for f in settings["saturation"]["fractions"] if (output / f"gptq-{f}.json").is_file()]
        report["status"] = "complete" if len(done) == len(settings["saturation"]["fractions"]) else "error"
        save()
        return report
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        report["program_errors"] = result.get("errors")
        report["layers_done"] = sorted(int(k) for k, v in result.get("layers", {}).items()
                                       if v.get("errors") or v.get("shards"))
        if phase == "calibrate":
            report["overlay_complete"] = bool((result.get("manifest") or {}).get("complete"))
            if settings.get("int3_overlay_dir"):
                report["int3_overlay_complete"] = bool((result.get("manifest_int3") or {}).get("complete"))
    ok = report.get("overlay_complete") if phase == "calibrate" else report.get("layers_done")
    report["status"] = "complete" if ok else "error"
    save()
    return report
