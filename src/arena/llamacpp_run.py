"""Own one llama-server through setup, the serve check, the arena trial(s) and cleanup (as sglang_run)."""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from arc_infra.model_templates import template_kwargs

from .llamacpp import (
    BASE_URL,
    Server,
    SlotCapacity,
    check_flags,
    decode_table,
    progress_line,
    pump,
    serve_check,
    server_argv,
    study_profile,
)


def probe_template_kwargs(study, prof):
    return template_kwargs(prof["served_model"], {"enable_thinking": bool(study.get("enable_thinking", True)),
                                                  "reasoning_effort": study["reasoning_effort"]})


def metrics_text():
    try:
        return httpx.get(BASE_URL.removesuffix("/v1") + "/metrics", timeout=5, trust_env=False).text
    except httpx.HTTPError:
        return ""


def effective_trial_wall(study):
    """Budget an unconditional pilot stop without changing the agents' scheduler horizon."""
    from .early_stop import settings
    wall = study['wall_seconds']
    stop = settings(study.get('early_stop'), wall)
    return min(wall, stop['at_seconds']) if stop and stop.get('always') else wall


def run_trial(command, env, trials, name, *, timeout, every=60):
    """The trial process with its output teed to a file and stdout (live Modal logs), plus a
    ``probe_progress`` line every ``every`` seconds (levels, completion tokens, tokens/s)."""
    from arc_infra.metrics import parse_metrics

    trials.mkdir(parents=True, exist_ok=True)
    start = {"epoch": time.time(), "tokens": parse_metrics(metrics_text()).get("llamacpp:tokens_predicted_total")}
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    reader = threading.Thread(target=pump, args=(process.stdout, trials / f"{name}-process.log"), daemon=True)
    reader.start()
    deadline = time.time() + timeout
    try:
        while process.poll() is None:
            if time.time() > deadline:
                process.kill()
                break
            try:
                process.wait(timeout=every)
            except subprocess.TimeoutExpired:
                print(json.dumps(progress_line(trials / name, metrics_text(), start)), flush=True)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=10)
    print(json.dumps(progress_line(trials / name, metrics_text(), start)), flush=True)
    return process.returncode


def run(args):
    bundle, working = Path(args.bundle), Path(args.working)
    study_path = bundle / "configs/comparison.json"
    study = json.loads(study_path.read_text())
    prof = study_profile(study)
    serving = working / "serving"
    serving.mkdir(parents=True, exist_ok=True)
    setup = {"started_epoch": args.started_epoch, "profile": study["llamacpp"]["profile"],
             "build_info": Path(os.environ["LLAMACPP_BUILD_INFO"]).read_text()
             if os.environ.get("LLAMACPP_BUILD_INFO") else None}
    results = {}
    server = None

    def stage(name, **fields):
        setup.update(fields, stage=name)
        setup.setdefault("stages", []).append({"stage": name, "elapsed_seconds": round(time.time() - args.started_epoch, 1)})
        (working / "setup.json").write_text(json.dumps(setup, indent=2, default=str))

    try:
        argv = server_argv(prof, binary=args.server_bin, weights_dir=args.weights_dir,
                           slots=study["generation_slots"], window=study["windows"][0],
                           kv_unified=bool(study["llamacpp"].get("kv_unified", False)))
        stage("flag_check", argv=argv)
        check_flags(args.server_bin, argv)
        print(json.dumps({"event": "stage", "stage": "server_start", "profile": setup["profile"]}), flush=True)
        stage("server_start")
        server = Server(argv, serving)
        setup["server"] = server.start()
        stage("capacity_validation")
        SlotCapacity.load(serving, study)  # fail before gameplay on a slot or context mismatch
        stage("serve_check")
        kwargs = probe_template_kwargs(study, prof)
        # Optional study settings: depths, streams, output_tokens; "gate": false plays on after a failed
        # check (local tests with a tiny stand-in model only).
        settings = dict(study["llamacpp"].get("serve_check", {}))
        gate = settings.pop("gate", True)
        check = asyncio.run(serve_check(BASE_URL, prof["served_model"], template_kwargs=kwargs,
                                        quick_template_kwargs=prof.get("quick_template_kwargs"),
                                        sampling=prof["sampling"], output=serving, **settings))
        (serving / "serve-check.md").write_text(decode_table(check) + "\n")
        print(json.dumps({"event": "serve_check", "passed": check["passed"],
                          "tool_round_trip": check["tool_round_trip"].get("passed"),
                          "recall": check["recall"].get("passed")}), flush=True)
        print(decode_table(check), flush=True)
        setup["serve_check_passed"] = check["passed"]
        if args.serve_check_only:
            results["serve-check"] = "complete" if check["passed"] else "failed"
            # Further server configurations (dots3 screen, 27 September): e.g. more slots or a unified KV
            # cache, each a fresh server with its own decode grid and recall check (files tagged -<name>).
            for extra in study["llamacpp"].get("serve_check_extra", []):
                tag = f"-{extra['name']}"
                server.stop()
                stage(f"server_start{tag}")
                extra_argv = server_argv(prof, binary=args.server_bin, weights_dir=args.weights_dir,
                                         slots=extra["slots"], window=study["windows"][0],
                                         kv_unified=extra.get("kv_unified", False))
                check_flags(args.server_bin, extra_argv)
                server = Server(extra_argv, serving, tag=tag)
                setup[f"server{tag}"] = server.start()
                stage(f"serve_check{tag}")
                more = asyncio.run(serve_check(
                    BASE_URL, prof["served_model"], template_kwargs=kwargs, tag=tag, tool_check=False,
                    quick_template_kwargs=prof.get("quick_template_kwargs"), sampling=prof["sampling"],
                    output=serving, depths=extra.get("depths", (8192, 32768)), streams=extra["streams"],
                    output_tokens=settings.get("output_tokens", 256)))
                (serving / f"serve-check{tag}.md").write_text(decode_table(more) + "\n")
                print(json.dumps({"event": "serve_check", "config": extra, "recall": more["recall"].get("passed")}),
                      flush=True)
                print(decode_table(more), flush=True)
                setup[f"serve_check{tag}_recall"] = more["recall"].get("passed")
                results[f"serve-check{tag}"] = "complete" if more["recall"].get("passed") else "failed"
            stage("done")
            return 0 if check["passed"] else 1
        if gate and not check["passed"]:
            raise RuntimeError("Serve check failed (tool round trip or recall); refusing to play on this server")
        setup["setup_seconds"] = time.time() - args.started_epoch
        stage("ready")
        env = dict(os.environ)
        trials = [t for t in study["trials"] if not args.trial or t["name"] in args.trial]
        for trial in trials:
            stage("gameplay", trial=trial["name"])
            left = args.started_epoch + args.max_seconds - time.time() - 180
            pilot_wall = effective_trial_wall(study)
            if left < pilot_wall + 300:
                raise TimeoutError("Insufficient budget for a complete arena trial and cleanup")
            (working / "trials").mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "arena.run", "--source", str(bundle), "--study", str(study_path),
                       "--output", str(working / "trials"), "--trial", trial["name"], "--server-dir", str(serving),
                       "--games-dir", args.games_dir,
                       "--analysis-python", sys.executable, "--base-url", BASE_URL,
                       "--notebook-start-epoch", str(args.started_epoch), "--notebook-seconds", str(args.max_seconds)]
            returncode = run_trial(command, env, working / "trials", trial["name"],
                                   timeout=min(left, pilot_wall + 600))
            results[trial["name"]] = "complete" if returncode == 0 else f"exit {returncode}"
            print(json.dumps({"event": "trial_process", "trial": trial["name"], "result": results[trial["name"]]}),
                  flush=True)
        stage("done")
        return 0 if len(results) == len(trials) and all(v == "complete" for v in results.values()) else 1
    except BaseException as exc:
        stage("failed", failed_stage=setup.get("stage"), error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if server is not None:
            (serving / "stop.json").write_text(json.dumps(server.stop(), indent=2))
        (working / "bootstrap-results.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "working", "games-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--server-bin", default=os.environ.get("LLAMACPP_SERVER_BIN", "llama-server"))
    parser.add_argument("--weights-dir", default=os.environ.get("LLAMACPP_WEIGHTS_DIR"))
    parser.add_argument("--trial", action="append")
    parser.add_argument("--serve-check-only", action="store_true")
    parser.add_argument("--started-epoch", required=True, type=float)
    parser.add_argument("--max-seconds", required=True, type=int)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"llama.cpp arena termination signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    args = parser.parse_args()
    if not args.weights_dir:
        raise ValueError("Set --weights-dir or LLAMACPP_WEIGHTS_DIR")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
