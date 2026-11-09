"""Modal entry point for the RL arena bundle: one server (``arena.serving``, hybrid blocks), one trial.

Uses the bundle helpers (``bootstrap.py`` next to this file in the bundle, built from
``scripts/arena/bundle_bootstrap.py``: verified manifest, locked wheel install, evidence collection). The trial is its
own ``arena.run`` process and plays the bundled generated games.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import collect_evidence, install_agent_environment, verify  # noqa: E402

TRIAL_GRACE = 600
KINDS = ("level-bench", "rl-loop")


def run_echoing(command, env, process_log, *, timeout, prefix):
    """Run a trial with its output in ``process_log``, echoing lines that start with ``prefix`` to stdout
    as they arrive; raises ``subprocess.TimeoutExpired`` (after killing the process) like ``run``."""
    import threading

    with Path(process_log).open("ab") as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        marker = prefix.encode()

        def pump():
            for line in iter(process.stdout.readline, b""):
                stream.write(line)
                stream.flush()
                if line.startswith(marker):
                    sys.stdout.write(line.decode(errors="replace"))
                    sys.stdout.flush()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            thread.join(timeout=10)
    return process.returncode


def trial_lifecycle_lines(process_log, limit=20):
    """The trial's own JSON lifecycle events (starting_trial, trial_result) from its log file."""
    lines = []
    try:
        for raw in Path(process_log).read_text(errors="replace").splitlines():
            if raw.startswith('{"event": "') and ('"starting_trial"' in raw or '"trial_result"' in raw):
                lines.append(raw)
    except OSError:
        return []
    return lines[-limit:]


def server2_export(study):
    """MFU-2 (25 Oct): server 2 (rl_loop.mg.timeshare) copies server 1's runtime and applies no patches itself, so
    serving_setup must see the flags only server 2 sets (ARC_SERVER2_ENV) to put their files in place."""
    env = ((study.get("rl_loop") or {}).get("mg") or {}).get("timeshare") or {}
    env = env.get("env") if isinstance(env, dict) else None
    return {"ARC_SERVER2_ENV": json.dumps({str(k): str(v) for k, v in env.items()}, sort_keys=True)} if env else {}


def card0_env(study):
    """mg.redesign.card_env["0"] (RL 26 Oct, per-card A/B): card 0's overrides of the trial's server_env (None
    removes a key); the other cards' apply over a copy of this server's environment (arena.rl_loop.card_environment)."""
    redesign = (((study.get("rl_loop") or {}).get("mg") or {}).get("redesign")) or {}
    env = (redesign.get("card_env") or {}) if isinstance(redesign, dict) else {}
    return dict(env.get("0") or env.get(0) or {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--working", default="/tmp/arena")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--games-dir", help="unused: the trial plays the bundle's generated games")
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial", action="append", help="Trial name; default the study's only trial")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    bundle, working = Path(args.bundle).resolve(), Path(args.working).resolve()
    started = time.time()
    identity = verify(bundle)
    study_path = str(bundle / "configs/comparison.json")
    study = json.loads(Path(study_path).read_text())
    trials = [t for t in study["trials"] if not args.trial or t["name"] in args.trial]
    if args.trial and len(trials) != len(args.trial):
        raise ValueError("Unknown trial name")
    if len(trials) != 1:
        raise ValueError("A run plays exactly one trial")
    trial = trials[0]
    if args.max_seconds < 600:
        raise ValueError("The run envelope must include setup and cleanup time")
    working.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "PYTHONPATH": str(bundle / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "MPLBACKEND": "Agg",
        "ONLY_RESET_LEVELS": "true",
    }
    python = install_agent_environment(bundle, working / "agent-venv", env)
    server_env = {**server2_export(study), **{str(k): str(v) for k, v in (trial.get("server_env") or {}).items()}}
    for key, value in card0_env(study).items():  # mg.redesign.card_env "0": the first server's own A/B arm
        if value is None:
            server_env.pop(key, None)
        else:
            server_env[key] = str(value)
    dry = os.environ.get("ARC_DRY_BOOT_ONLY") == "1"  # 25 Oct: the same boot through the engine start (arena.dry_boot)
    if os.environ.get("ARC_PATCH_CHECK_ONLY") == "1" or dry:  # CPU gate (24 Oct): every boot patch on the pinned runtime
        command = [python, "-m", "arena.serving", "--study", study_path,
                   "--working", str(working / "serving"), "--input-root", args.input_root]
        flag = {"ARC_SERVING_DRY_BOOT": "1"} if dry else {"ARC_SERVING_PATCH_CHECK": "1"}
        done = subprocess.run(command, env={**env, **server_env, **flag}, check=False)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        name = "dry-boot" if dry else "patch-check"
        (Path(args.output) / f"{name}.json").write_text(json.dumps({"returncode": done.returncode, "server_env": server_env}))
        for f in ("dry-boot.json",) if dry else ():
            if (working / "serving" / f).is_file():
                shutil.copyfile(working / "serving" / f, Path(args.output) / f)
        print(json.dumps({"event": name.replace("-", "_"), "returncode": done.returncode}), flush=True)
        raise SystemExit(done.returncode)
    results = {}
    try:
        serving_command = [python, "-m", "arena.serving", "--study", study_path]
        if os.environ.get("ARC_PREFLIGHT_STUB"):  # launch-gate preflight (19 Oct): a replay stub stands in for vLLM
            serving_command = [python, "-m", "arena.replay_stub", "launch"]
        subprocess.run(
            [*serving_command, "--working", str(working / "serving"), "--input-root", args.input_root],
            env={**env, **server_env}, check=True, timeout=max(1, started + args.max_seconds - 60 - time.time()),
        )
        if study.get("cache_probe", True):  # RL redesign (H2 26 Oct): off; 87 s before play, rl_seq warms every server
            subprocess.run([python, "-m", "arena.cache_probe", "--server-dir", str(working / "serving"),
                            "--study", study_path, "--base-url", args.base_url], env=env, check=True, timeout=300)
        (working / "setup.json").write_text(
            json.dumps({"setup_seconds": time.time() - started, "bundle": identity}, indent=2))
        games = str(bundle / study["generated_games"]["dir"])
        if trial["kind"] not in KINDS:
            raise ValueError(f"Unknown trial kind {trial['kind']!r}")
        reserves = study.get("bootstrap_reserves") or {}
        budget = started + args.max_seconds - float(reserves.get("end_seconds", 120)) - time.time()
        margin = float(reserves.get("trial_seconds", 300))
        # The trial sizes its own wall from what is left (at least its min_wall_seconds).
        trial_wall = min(float(trial.get("wall_seconds", study["wall_seconds"])), budget - 300)
        if budget < trial_wall + margin or trial_wall <= 0:
            results[trial["name"]] = "skipped: notebook budget"
            print(json.dumps({"event": "trial_skipped", "trial": trial["name"]}), flush=True)
        else:
            command = [
                python, "-m", "arena.run", "--source", str(bundle), "--study", study_path,
                "--output", str(working / "trials"), "--trial", trial["name"], "--server-dir", str(working / "serving"),
                "--analysis-python", python,
                "--base-url", args.base_url, "--notebook-start-epoch", str(started),
                "--notebook-seconds", str(args.max_seconds), "--games-dir", games,
            ]
            # The trial's stdout and stderr go to a file, never to the platform pipe (a blocked print froze a trial
            # on 25 September); the bootstrap prints only the trial's lifecycle lines afterwards.
            (working / "trials").mkdir(parents=True, exist_ok=True)
            process_log = working / "trials" / f"{trial['name']}-process.log"
            timeout = max(1, min(budget, trial_wall + TRIAL_GRACE))
            try:
                if trial["kind"] == "level-bench":  # live progress: the trial's own JSON lines echoed
                    returncode = run_echoing(command, env, process_log, timeout=timeout, prefix='{"event": "level_bench')
                else:
                    with process_log.open("ab") as stream:
                        returncode = subprocess.run(command, env=env, check=False, timeout=timeout,
                                                    stdout=stream, stderr=subprocess.STDOUT).returncode
                results[trial["name"]] = "complete" if returncode == 0 else f"exit {returncode}"
            except subprocess.TimeoutExpired:
                results[trial["name"]] = "timeout"
            for line in trial_lifecycle_lines(process_log):
                print(line, flush=True)
            print(json.dumps({"event": "trial_process", "trial": trial["name"], "result": results[trial["name"]],
                              "process_log": str(process_log)}), flush=True)
            if trial.get("server_env") is not None and (working / "serving/vllm-openai-server.log").is_file():
                (working / "trials" / trial["name"]).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(working / "serving/vllm-openai-server.log",
                                working / "trials" / trial["name"] / "vllm-openai-server.log")
            collect_evidence(working, args.output)
    finally:
        collect_evidence(working, args.output)
        program = "from arc_infra.serving import load_recipe; import sys; load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()"
        try:
            subprocess.run([python, "-c", program, str(working / "serving"), args.input_root], env=env, check=False, timeout=180)
        except subprocess.TimeoutExpired:
            print(json.dumps({"event": "server_stop_timeout", "seconds": 180}), flush=True)
        collect_evidence(working, args.output)
        Path(args.output, "bootstrap-results.json").write_text(json.dumps(results, indent=2))
    if results.get(trial["name"]) != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
