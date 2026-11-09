"""Own one SGLang process through setup, independent arena trials and cleanup."""

import argparse
import asyncio
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from arc_infra.arcade import find_games
from sglang_bench import launch, runtime, tokenize_check, tokenize_patch
from sglang_bench.profile import ServerProfile


def run(args):
    bundle, working = Path(args.bundle), Path(args.working)
    study_path = bundle / "configs/comparison.json"
    study = json.loads(study_path.read_text())
    profile = ServerProfile.from_dict(study["sglang_profile"])
    serving = working / "serving"
    serving.mkdir(exist_ok=True)
    server = None
    results = {}
    setup = {"started_epoch": args.started_epoch}

    def stage(name, **fields):
        now = time.time()
        setup.update(fields)
        setup["stage"] = name
        setup.setdefault("stages", []).append({"stage": name, "epoch": now,
                                               "elapsed_seconds": now - args.started_epoch,
                                               "host": runtime.host_facts()})
        temporary = working / "setup.json.tmp"
        temporary.write_text(json.dumps(setup, indent=2, default=str))
        temporary.replace(working / "setup.json")

    try:
        stage("runtime_extraction_and_prefetch")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            prefetched = pool.submit(runtime.prefetch_model)
            mount = runtime.locate(args.input_root)
            manifest = runtime.verify_manifest(mount)
            setup["runtime_extraction"] = runtime.extract(mount, manifest)
            runtime.link_cudart()
            runtime.link_cuda_driver()
            setup["prefetch"] = prefetched.result()
        stage("runtime_check")
        env = launch.recipe_env(profile, cache_base=runtime.SCRATCH / "cache", cuda_home=runtime.CUDA_HOME,
                                repo_root=runtime.REPO_ROOT, scratch=runtime.SCRATCH / "tmp")
        checked = runtime.check(env)
        checked["model"] = runtime.verify_model()
        (serving / "runtime-check.json").write_text(json.dumps(checked, indent=2, default=str))
        if checked["problems"]:
            raise RuntimeError("SGLang runtime check failed: " + "; ".join(checked["problems"]))
        stage("tokenize_patch")
        setup["tokenize_patch"] = tokenize_patch.apply(runtime.REPO_ROOT)
        preflight = launch.preflight(profile, model_dir=runtime.MODEL_DIR, repo_root=runtime.REPO_ROOT)
        argv = launch.recipe_argv(profile, model_dir=runtime.MODEL_DIR, repo_root=runtime.REPO_ROOT,
                                  sglang_exe=runtime.REPO_ROOT / ".venv/bin/sglang")
        stage("server_start")
        server = launch.start(profile, working=serving, env=env, argv=argv, preflight_record=preflight)
        stage("server_readiness")
        startup = launch.wait_ready(server, timeout=study["startup_timeout_seconds"])
        from .sglang import TokenBudget

        (serving / "startup.json").write_text(json.dumps(startup, indent=2))
        stage("capacity_validation")
        TokenBudget(startup, study)  # fail before gameplay on clamps or missing capacity evidence
        stage("warmup")
        setup["warmup"] = launch.warmup(server, probes=False)
        stage("tokenize_generation_parity")
        setup["tokenize_generation_parity"] = asyncio.run(tokenize_check.run(
            profile.base_url, profile.served_model_name, serving / "tokenize-parity.json"))
        setup["setup_seconds"] = time.time() - args.started_epoch
        stage("ready")
        games = args.games_dir or find_games(args.input_root)
        trials = [t for t in study["trials"] if not args.trial or t["name"] in args.trial]
        for trial in trials:
            stage("gameplay", trial=trial["name"])
            left = args.started_epoch + args.max_seconds - time.time() - 180
            if left < study["wall_seconds"] + 600:
                raise TimeoutError("Insufficient budget for a complete arena trial and cleanup")
            command = [sys.executable, "-u", "-m", "arena.run", "--source", str(bundle),
                       "--study", str(study_path), "--output", str(working / "trials"),
                       "--trial", trial["name"], "--server-dir", str(serving),
                       "--games-dir", games,
                       "--analysis-python", sys.executable, "--base-url", profile.base_url,
                       "--notebook-start-epoch", str(args.started_epoch),
                       "--notebook-seconds", str(args.max_seconds)]
            result = subprocess.run(command, env=os.environ.copy(), check=False,
                                    timeout=min(left, study["wall_seconds"] + 600))
            results[trial["name"]] = "complete" if result.returncode == 0 else f"exit {result.returncode}"
            print(json.dumps({"event": "trial_process", "trial": trial["name"],
                              "result": results[trial["name"]]}), flush=True)
        return 0 if len(results) == len(trials) and all(v == "complete" for v in results.values()) else 1
    except BaseException as exc:
        stage("failed", failed_stage=setup.get("stage"),
              error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if server is not None:
            stopped = launch.stop(server)
            (serving / "stop.json").write_text(json.dumps(stopped, indent=2))
        (working / "bootstrap-results.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "working", "input-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--games-dir")
    parser.add_argument("--trial", action="append")
    parser.add_argument("--started-epoch", required=True, type=float)
    parser.add_argument("--max-seconds", required=True, type=int)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"SGLang arena termination signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
