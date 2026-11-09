"""v8.2's harness on dfranzen's exact serving: bundle entry on the Nebius VM.

1. His server, started by his own notebook cells (2, 4, 6, 8, 10, 12: patch, environment, precache, arc-agi, sources,
   SGLang Pennyroyal with his AutoRound W4A16 weights, FP8 KV, NEXTN-3 + FR-Spec and his flags), in an IPython kernel
   that then idles on one added cell until the run ends (so his server outlives no kernel it belongs to).
2. ``serving/startup.json`` and the study's ``sglang_profile`` / ``served_model`` from the live server.
3. A preflight of what the v8.2 harness needs from this engine (chat ``/tokenize`` with tools and template kwargs,
   parsed tool calls and reasoning, image tokens, ``continue_final_message``); any failure ends the job here.
4. v8.2's arena trial (``arena.run``, backend ``sglang``) on the copied v8.2 source, with live progress as usual.
"""

import argparse
import base64
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dfr"))
sys.path.insert(0, str(HERE))
import bootstrap_dfr as dfr  # noqa: E402

BASE = "http://127.0.0.1:8001"
HOLD_CELL = ("import os, time\n_p = os.environ['DFR_STOP_FILE']\nprint('[arc] server cells done; holding', flush=True)\n"
             "while not os.path.exists(_p):\n    time.sleep(5)\n")
TOOL = {"type": "function", "function": {"name": "move", "description": "Move the player one cell.",
                                         "parameters": {"type": "object", "properties": {
                                             "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
                                             "required": ["direction"]}}}
IMAGE_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


def log(event, **kw):
    print(json.dumps({"event": event, **kw}), flush=True)


def http(path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def png(size=64):
    rows = b"".join(b"\x00" + b"".join(bytes(((x * 37) % 256, (y * 53) % 256, 90)) for x in range(size))
                    for y in range(size))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def preflight(model, window):
    out, ok = {}, True
    kwargs = {"enable_thinking": True, "reasoning_effort": "xhigh"}
    messages = [{"role": "system", "content": "You play a grid game."}, {"role": "user", "content": "Start."}]
    try:
        t = http("/tokenize", {"model": model, "messages": messages, "tools": [TOOL], "chat_template_kwargs": kwargs})
        out["tokenize"] = {"count": t["count"], "max_model_len": t["max_model_len"]}
        ok &= t["count"] > 0 and t["max_model_len"] >= window
        c = http("/tokenize", {"model": model, "messages": messages + [{"role": "assistant", "content": "<think>\n\n</think>\n\n"}],
                               "continue_final_message": True, "add_generation_prompt": False,
                               "chat_template_kwargs": kwargs})
        out["continue_final_message_count"] = c["count"]
    except Exception as exc:  # noqa: BLE001
        out["tokenize_error"] = repr(exc)[:500]
        ok = False
    try:
        r = http("/v1/chat/completions", {"model": model, "messages": [
            {"role": "user", "content": "Call the tool move with direction up. Do not answer in text."}],
            "tools": [TOOL], "tool_choice": "auto", "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95,
            "top_k": 20, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}})
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        out["tool_call"] = {"calls": [c["function"]["name"] + " " + c["function"]["arguments"] for c in calls],
                            "reasoning_chars": len(msg.get("reasoning_content") or ""),
                            "content": (msg.get("content") or "")[:200], "usage": r.get("usage")}
        ok &= bool(calls) and bool(msg.get("reasoning_content"))
    except Exception as exc:  # noqa: BLE001
        out["tool_call_error"] = repr(exc)[:500]
        ok = False
    try:
        # History reasoning (12 October): his template renders an earlier turn's reasoning only from the
        # ``reasoning_content`` key; the harness must send it under that key (ADAPTED in the builder).
        def history(key):
            turn = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "move", "arguments": '{"direction": "up"}'}}]}
            if key:
                turn[key] = "I should move up because the goal is above. " * 20
            return [*messages, turn, {"role": "tool", "tool_call_id": "c1", "content": "moved"},
                    {"role": "user", "content": "Next."}]
        counts = {key or "none": http("/tokenize", {"model": model, "messages": history(key), "tools": [TOOL],
                                                    "chat_template_kwargs": kwargs})["count"]
                  for key in (None, "reasoning", "reasoning_content")}
        out["history_reasoning_tokens"] = counts
        ok &= counts["reasoning_content"] - counts["none"] > 100
    except Exception as exc:  # noqa: BLE001
        out["history_error"] = repr(exc)[:500]
        ok = False
    try:
        def prompt_tokens(content):
            return http("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}],
                                                 "max_tokens": 1, "temperature": 0,
                                                 "chat_template_kwargs": {"enable_thinking": False}})["usage"]["prompt_tokens"]
        with_image = prompt_tokens([{"type": "image_url", "image_url": {"url": png()}}])
        as_text = prompt_tokens([{"type": "text", "text": IMAGE_TEXT}])
        out["image_64px_tokens"] = with_image - as_text + 1
        ok &= with_image - as_text + 1 >= 2
    except Exception as exc:  # noqa: BLE001
        out["image_error"] = repr(exc)[:500]
        ok = False
    out["ok"] = bool(ok)
    return out


def startup_info():
    info = http("/server_info")
    states = info.get("internal_states") or [{}]
    models = http("/v1/models")["data"][0]
    total = info.get("max_total_num_tokens") or states[0].get("max_total_num_tokens")
    return {"max_total_num_tokens": int(total), "max_running_requests": int(info["max_running_requests"]),
            "max_mamba_cache_size": info.get("max_mamba_cache_size"),
            "context_len": int(info.get("context_length") or models.get("max_model_len")),
            "page_size": int(info.get("page_size") or 64), "served_model": models["id"],
            "speculative_algorithm": info.get("speculative_algorithm"), "kv_cache_dtype": info.get("kv_cache_dtype"),
            "quantization": info.get("quantization"), "version": info.get("version")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(HERE))
    ap.add_argument("--working", default="/tmp/arena")
    ap.add_argument("--output", required=True)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--max-seconds", type=int, required=True)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()
    started = time.time()
    bundle, working, output = Path(args.bundle).resolve(), Path(args.working), Path(args.output)
    from bootstrap import collect_evidence, install_agent_environment, verify
    verify(bundle)
    study = json.loads((bundle / "configs/comparison.json").read_text())
    block = study["dfr_exact"]
    working.mkdir(parents=True, exist_ok=True)
    (output / "live").mkdir(parents=True, exist_ok=True)
    for folder in (working / "serving", Path("/kaggle/working")):
        folder.mkdir(parents=True, exist_ok=True)
    result = {"run": study["name"], "started_epoch": started}
    procs, stop_file = {}, working / "dfr-hold-release"
    stop_file.unlink(missing_ok=True)
    trial = study["trials"][0]
    try:
        kernel_python = dfr.make_venv(working / "kernel-venv", bundle / "dfr/wheels", dfr.KERNEL_REQUIREMENTS)
        result["notebook"] = dfr.prepare_notebook(output / "notebook-input.ipynb", block["notebook_sha256"],
                                                  bundle / "dfr" / block["notebook_file"],
                                                  edits=block.get("notebook_edits") or ())
        env = dfr.kernel_env(kernel_python, stop_file, public=True)
        command = [str(kernel_python), "-u", str(bundle / "dfr/run_notebook.py"), str(output / "notebook-input.ipynb"),
                   str(output / "notebook-executed.ipynb"), "--cells", "2,4,6,8,10,12", "--extra", HOLD_CELL]
        nb_log = (output / "notebook.log").open("ab")
        procs["notebook"] = subprocess.Popen(command, env=env, cwd="/kaggle/working", stdout=nb_log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + block["server_start_seconds"]
        ready = False
        while time.time() < deadline and procs["notebook"].poll() is None:
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                    ready = r.status == 200
            except Exception:  # noqa: BLE001
                pass
            if ready:
                break
            time.sleep(10)
        if not ready:
            raise RuntimeError(f"his server did not come up (notebook exit {procs['notebook'].poll()})")
        result["server_ready_seconds"] = round(time.time() - started, 1)
        log("server_ready", seconds=result["server_ready_seconds"])
        startup = startup_info()
        (working / "serving/startup.json").write_text(json.dumps(startup, indent=1))
        result["startup"] = startup
        if block.get("capture"):
            # Kernel-quality capture instead of play (compared offline with arena.kernel_gate's compare_*).
            base_env = {**os.environ, "PYTHONPATH": str(bundle / "src"), "PYTHONNOUSERSITE": "1",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX": "1"}
            python = install_agent_environment(bundle, working / "agent-venv", base_env)
            with (output / "capture.log").open("ab") as stream:
                done = subprocess.run([python, str(bundle / "sgl_capture.py"), BASE, startup["served_model"],
                                       str(bundle / "configs/kernel-quality"), str(output / "capture.json")],
                                      env=base_env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.max_seconds)
            result["arena_exit"] = done.returncode
            return 0 if done.returncode == 0 else 1
        window = int(study["windows"][0])
        study["sglang_profile"] = {"max_total_tokens": startup["max_total_num_tokens"],
                                   "admitted": startup["max_running_requests"],
                                   "mamba_slots": startup["max_mamba_cache_size"],
                                   "context_length": startup["context_len"], "page_size": startup["page_size"],
                                   "base_url": BASE + "/v1", "served_model_name": startup["served_model"]}
        study["served_model"] = startup["served_model"]
        study_path = working / "study.json"
        study_path.write_text(json.dumps(study, indent=1))
        result["preflight"] = check = preflight(startup["served_model"], window)
        (output / "preflight.json").write_text(json.dumps(check, indent=1))
        log("preflight", **check)
        if not check["ok"]:
            raise RuntimeError("preflight failed: the v8.2 harness cannot run on this engine as built")
        base_env = {**os.environ, "PYTHONPATH": str(bundle / "src"), "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PIP_NO_INDEX": "1", "MPLBACKEND": "Agg", "ONLY_RESET_LEVELS": "true"}
        python = install_agent_environment(bundle, working / "agent-venv", base_env)
        import bootstrap_arena
        live = bootstrap_arena.start_live_progress(bundle, study, trial, working, output)
        (working / "trials").mkdir(exist_ok=True)
        command = [python, "-m", "arena.run", "--source", str(bundle), "--study", str(study_path),
                   "--output", str(working / "trials"), "--trial", trial["name"], "--server-dir",
                   str(working / "serving"), "--analysis-python", python, "--base-url", BASE + "/v1",
                   "--notebook-start-epoch", str(started), "--notebook-seconds", str(args.max_seconds),
                   "--games-dir", args.games_dir]
        with (working / "trials" / f"{trial['name']}-process.log").open("ab") as stream:
            procs["arena"] = subprocess.Popen(command, env=base_env, stdout=stream, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            hard_end = started + args.max_seconds - 300
            while procs["arena"].poll() is None and time.time() < hard_end:
                time.sleep(15)  # a bucket stop request reaches the trial itself (arena.early_stop, ARENA_STOP_REQUEST)
            if procs["arena"].poll() is None:
                os.killpg(procs["arena"].pid, signal.SIGTERM)
                try:
                    procs["arena"].wait(240)
                except subprocess.TimeoutExpired:
                    os.killpg(procs["arena"].pid, signal.SIGKILL)
        result["arena_exit"] = procs["arena"].wait()
        if live:
            live[0].set()
            live[1].join(timeout=90)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_file.write_text("done")
        for proc in procs.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
        for name in ("serve.log",):
            if Path("/kaggle/working", name).exists():
                subprocess.run(["cp", f"/kaggle/working/{name}", str(output / name)], check=False)
        for prof in sorted(Path("/kaggle/working").glob("profile-*")):  # --his-profile traces (prof10 v1 lost them)
            subprocess.run(["cp", "-r", str(prof), str(output / prof.name)], check=False)
        try:
            collect_evidence(working, output)
        except Exception as exc:  # noqa: BLE001
            result["collect_error"] = repr(exc)[:300]
        result["seconds"] = round(time.time() - started, 1)
        (output / "v82sgl-result.json").write_text(json.dumps(result, indent=1))
        log("v82sgl_result", **{k: v for k, v in result.items() if k not in ("preflight", "startup")})
    return 0 if result.get("arena_exit") == 0 else 1


if __name__ == "__main__":
    threading.current_thread().name = "v82sgl"
    raise SystemExit(main())
