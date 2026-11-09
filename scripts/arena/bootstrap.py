"""Offline Kaggle/Modal entry point for the arena bundle: one server, sequential trials.

Uses the bundle helpers (``bootstrap.py`` next to this file in the bundle, built from
``scripts/arena/bundle_bootstrap.py``: verified manifest, locked wheel install, evidence collection) and
``arena.serving`` (hybrid-block studies) or ``arena.serving_core.cli serve-native`` (the ARC fork at 262,144 with
prefix caching). Each trial is its own ``arena.run`` process.
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


def start_live_progress(bundle, study, trial, working, output):
    """Live progress (hero-v2, 29 September): a thread that copies the running arena trial's progress.json to
    ``output/live`` and prints one ``live_progress`` line every ``interval_seconds`` (arena.live_progress). The trial's
    own output stays in its file; only this thread writes to stdout. Returns ``(stop event, thread)`` or None."""
    spec = study.get("live_progress")
    if not spec or not output or not str(trial.get("kind", "")).startswith("duck"):
        return None
    import importlib.util
    import threading

    source = Path(bundle) / "src/arena/live_progress.py"
    module_spec = importlib.util.spec_from_file_location("arc_live_progress", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    stop = threading.Event()
    thread = threading.Thread(target=module.tail, daemon=True, name="live-progress",
                              args=(Path(working) / "trials" / trial["name"] / "progress.json", Path(output),
                                    trial["name"], stop, int(spec.get("interval_seconds", 60))),
                              kwargs={"server_log": Path(working) / "serving/vllm-openai-server.log"})
    thread.start()
    return stop, thread


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--working", default="/tmp/arena")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--games-dir")
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evolve", action="store_true", help="Offline harness evolution loop instead of the trials")
    parser.add_argument("--trial", action="append", help="Trial name(s); default all")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--notebook-start-epoch", type=float)
    parser.add_argument("--serve-check-only", action="store_true",
                        help="llama.cpp studies: start the server, run the serve check and stop")
    args = parser.parse_args()
    bundle, working = Path(args.bundle).resolve(), Path(args.working).resolve()
    started = args.notebook_start_epoch or time.time()
    identity = verify(bundle)
    study = json.loads((bundle / "configs/comparison.json").read_text())
    competition = study.get('competition')
    rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
    if competition and rerun:
        args.max_seconds = competition['submission_seconds']
    trials = [t for t in study["trials"] if not args.trial or t["name"] in args.trial]
    if args.trial and len(trials) != len(args.trial):
        raise ValueError("Unknown trial name")
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
    if args.serve_check_only and study.get("backend") != "llamacpp":
        raise ValueError("--serve-check-only applies to llama.cpp studies")
    if study.get("backend") == "llamacpp":
        # Model screens (27 September): the platform supplies the llama-server binary and the weights
        # (LLAMACPP_SERVER_BIN, LLAMACPP_WEIGHTS_DIR); arena.llamacpp_run owns the server.
        command = [python, "-u", "-m", "arena.llamacpp_run", "--bundle", str(bundle), "--working", str(working),
                   "--games-dir", args.games_dir,
                   "--started-epoch", str(started), "--max-seconds", str(args.max_seconds)]
        if args.serve_check_only:
            command.append("--serve-check-only")
        for name in args.trial or []:
            command += ["--trial", name]
        try:
            completed = subprocess.run(command, env=env, check=False)
        finally:
            collect_evidence(working, args.output)
            if (working / "bootstrap-results.json").is_file():
                shutil.copyfile(working / "bootstrap-results.json", Path(args.output) / "bootstrap-results.json")
        raise SystemExit(completed.returncode)
    if study.get("backend") == "sglang":
        command = [python, "-u", "-m", "arena.sglang_run", "--bundle", str(bundle),
                   "--working", str(working),
                   "--input-root", args.input_root, "--started-epoch", str(started),
                   "--max-seconds", str(args.max_seconds)]
        if args.games_dir:
            command += ["--games-dir", args.games_dir]
        for name in args.trial or []:
            command += ["--trial", name]
        try:
            completed = subprocess.run(command, env=env, check=False)
        finally:
            collect_evidence(working, args.output)
            if (working / "bootstrap-results.json").is_file():
                shutil.copyfile(working / "bootstrap-results.json", Path(args.output) / "bootstrap-results.json")
        raise SystemExit(completed.returncode)
    results = {}
    try:
        serving_command = [python, "-m", "arena.serving_core.cli", "serve-native"]
        if study.get('candidate_profile'):
            serving_command = [python, '-m', 'arena.candidate_serving', '--study', str(bundle / 'configs/comparison.json')]
        elif study.get("cache_admission") == "hybrid_blocks":
            serving_command = [python, "-m", "arena.serving", "--study", str(bundle / "configs/comparison.json")]
        def server_env(trial):
            # Serving A/B bench (26 September): a trial may carry server environment overrides
            # (TAAF_VLLM_* knobs); the server restarts in place whenever they change.
            return {str(k): str(v) for k, v in (trial.get("server_env") or {}).items()}

        def start_server(extra):
            subprocess.run(
                [*serving_command, "--working", str(working / "serving"), "--input-root", args.input_root],
                env={**env, **extra}, check=True, timeout=max(1, started + args.max_seconds - 60 - time.time()),
            )
            if study.get("cache_admission") == "hybrid_blocks":
                probe = working / "serving" / "cache-probe"
                if probe.exists():  # a restarted server is probed again; keep the earlier probe aside
                    probe.rename(probe.with_name(f"cache-probe-{int(time.time())}"))
                subprocess.run([python, "-m", "arena.cache_probe", "--server-dir", str(working / "serving"),
                                "--study", str(bundle / "configs/comparison.json"), "--base-url", args.base_url],
                               env=env, check=True, timeout=300)

        stop_program = ("from arc_infra.serving import load_recipe; import sys; "
                        "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")
        current_env = server_env(trials[0]) if trials and not args.evolve else {}
        if study.get("start_in_trial_loop") and not args.evolve:
            # Server diagnostics (expert usage, 28 September): the first trial's server may fail to start; starting
            # it inside the trial loop costs only that trial instead of the whole run.
            current_env = None
        else:
            start_server(current_env)
        (working / "setup.json").write_text(
            json.dumps({"setup_seconds": time.time() - started, "bundle": identity, "smoke": args.smoke}, indent=2)
        )
        hidden = bool(competition) and rerun and not competition.get('rehearsal')
        # Teacher runs ship their own games, which take precedence over a platform --games-dir (Modal
        # always passes the public games).
        bundled = study.get("generated_games", {}).get("dir")
        games = '' if hidden else (str(bundle / bundled) if bundled else None) or args.games_dir or subprocess.check_output(
            [python, "-c", "from arc_infra.arcade import find_games; import sys; print(find_games(sys.argv[1]))",
             args.input_root], env=env, text=True,
        ).strip()
        wall = study["smoke"]["wall_seconds"] if args.smoke else study["wall_seconds"]
        if args.evolve:
            # Offline harness evolution: one driver process runs the rounds against the server.
            command = [python, "-m", "arena.evolve", "--bundle", str(bundle), "--working", str(working),
                       "--output", args.output, "--base-url", args.base_url, "--python", python,
                       "--started-epoch", str(started),
                       "--max-seconds", str(args.max_seconds)]
            if games:
                command += ["--games-dir", games]
            (working / "trials").mkdir(parents=True, exist_ok=True)
            process_log = working / "trials" / "evolve-process.log"
            try:
                with process_log.open("ab") as stream:
                    completed = subprocess.run(command, env=env, check=False, stdout=stream, stderr=subprocess.STDOUT,
                                               timeout=max(1, started + args.max_seconds - 120 - time.time()))
                results["evolve"] = "complete" if completed.returncode == 0 else f"exit {completed.returncode}"
            except subprocess.TimeoutExpired:
                results["evolve"] = "timeout"
            for line in trial_lifecycle_lines(process_log):
                print(line, flush=True)
            print(json.dumps({"event": "evolve_process", "result": results["evolve"], "process_log": str(process_log)}), flush=True)
            trials = [{"name": "evolve"}]
        for trial in ([] if args.evolve else trials):
            trial_competition = bool(competition) and trial.get('competition', True)
            if competition and rerun and not trial_competition:
                results[trial["name"]] = "skipped: hidden rerun"
                print(json.dumps({"event": "trial_skipped", "trial": trial["name"], "reason": "hidden rerun"}), flush=True)
                continue
            if trial.get("requires") and results.get(trial["requires"]) != "complete":
                # Kernel work, phase 2: the arena trial runs only after its quality gate passed.
                results[trial["name"]] = f"skipped: {trial['requires']} {results.get(trial['requires'], 'did not run')}"
                print(json.dumps({"event": "trial_skipped", "trial": trial["name"],
                                  "reason": results[trial["name"]]}), flush=True)
                continue
            if server_env(trial) != current_env:
                subprocess.run([python, "-c", stop_program, str(working / "serving"), args.input_root],
                               env=env, check=False, timeout=180)
                restart = time.time()
                try:
                    start_server(server_env(trial))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    # A candidate server that fails to start (e.g. a larger KV allocation at warmup)
                    # costs only its own trial; the next trial restarts from scratch.
                    results[trial["name"]] = f"server_failed: {type(exc).__name__}"
                    current_env = None
                    print(json.dumps({"event": "server_restart_failed", "trial": trial["name"],
                                      "error": str(exc)[:300]}), flush=True)
                    if (working / "serving/vllm-openai-server.log").is_file():
                        (working / "trials" / trial["name"]).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(working / "serving/vllm-openai-server.log",
                                        working / "trials" / trial["name"] / "vllm-openai-server.log")
                    continue
                current_env = server_env(trial)
                print(json.dumps({"event": "server_restart", "trial": trial["name"], "server_env": current_env,
                                  "seconds": round(time.time() - restart, 1)}), flush=True)
            trial_wall = float(trial.get('wall_seconds', wall))
            stop = study.get('early_stop') or {}
            if stop.get('always') and trial['kind'].startswith('duck'):
                # An unconditional run stop (arena.early_stop ``always``; hero-v2-blocks): the trial plays at most
                # at_seconds of its longer wall, so it is budgeted by the stop.
                trial_wall = min(trial_wall, float(stop['at_seconds']))
            # Short research runs (harness arm, 3 October: <= 30-minute Modal runs) may shrink the two reserves (the
            # post-trial 120 s and the 300 s per-trial margin) by a study field; the defaults are unchanged.
            reserves = study.get('bootstrap_reserves') or {}
            budget = started + args.max_seconds - float(reserves.get('end_seconds', 120)) - time.time()
            margin = float(reserves.get('trial_seconds', 300))
            if trial["kind"] in ("level-bench", "expert-usage", "prune-quality"):
                # The bench sizes its own wall from what is left (at least its min_wall_seconds); the
                # expert-usage trial runs its passes until shortly before this timeout.
                trial_wall = min(trial_wall, budget - 300)
            skip = budget < trial_wall + margin or trial_wall <= 0
            fit = (study.get('competition') or {}).get('fit_wall_min_seconds')
            if fit and (study.get('competition') or {}).get('research_clones') and trial_wall > 0:
                # Harness arm: the trial shortens its wall to what is left (arena.competition.gameplay_seconds).
                skip = budget - margin < float(fit)
            if trial["kind"] == "expert-quant":
                # Expert quantization (28 September): capture, then a GPU program sized to what is left; it
                # needs no gameplay margin, only two minutes.
                trial_wall, skip = budget, budget < 120
            if not (trial_competition and rerun) and skip:
                results[trial["name"]] = "skipped: notebook budget"
                print(json.dumps({"event": "trial_skipped", "trial": trial["name"]}), flush=True)
                continue
            command = [
                python, "-m", "arena.run", "--source", str(bundle), "--study", str(bundle / "configs/comparison.json"),
                "--output", str(working / "trials"), "--trial", trial["name"], "--server-dir", str(working / "serving"),
                "--analysis-python", python,
                "--base-url", args.base_url, "--notebook-start-epoch", str(started),
                "--notebook-seconds", str(args.max_seconds),
            ]
            if games:
                command += ['--games-dir', games]
            if args.smoke:
                command.append("--smoke")
            # The trial's stdout and stderr go to a file, never to the notebook pipe: in the v3
            # preview (25 September) Kaggle's log capture stalled and the 110-instance trial froze
            # at 255 s on a blocked print (gate, telemetry and progress stopped in the same
            # instant, the process stayed alive until the timeout). The bootstrap prints only
            # the trial's lifecycle lines afterwards.
            (working / "trials").mkdir(parents=True, exist_ok=True)
            process_log = working / "trials" / f"{trial['name']}-process.log"
            try:
                timeout = budget if trial_competition and rerun else min(budget, trial_wall + TRIAL_GRACE)
                if trial["kind"] in ("level-bench", "kernel-quality", "draft-head", "expert-quant", "expert-usage",
                                     "server-check", "prune-quality"):
                    # Live progress (Modal only builds these trials): the trial's own JSON lines are echoed as
                    # they arrive; everything else still goes to the file alone.
                    returncode = run_echoing(command, env, process_log, timeout=max(1, timeout),
                                             prefix={"level-bench": '{"event": "level_bench',
                                                     "kernel-quality": '{"event": "kernel_quality',
                                                     "draft-head": '{"event": "draft_head',
                                                     "expert-quant": '{"event": "expert_quant',
                                                     "expert-usage": '{"event": "expert_usage',
                                                     "server-check": '{"event": "server_check',
                                                     "prune-quality": '{"event": "prune_quality'}[trial["kind"]])
                else:
                    live = start_live_progress(bundle, study, trial, working, args.output)
                    try:
                        with process_log.open("ab") as stream:
                            returncode = subprocess.run(command, env=env, check=False, timeout=max(1, timeout),
                                                        stdout=stream, stderr=subprocess.STDOUT).returncode
                    finally:
                        if live is not None:
                            live[0].set()
                            live[1].join(timeout=30)
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
            if args.output:
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
    # A trial skipped by design on the hidden rerun (the public sanity trial) is not a failure;
    # the 25 September rehearsal caught this exit code raising the notebook after gameplay.
    if any(v != "complete" and v != "skipped: hidden rerun" for v in results.values()) or len(results) != len(trials):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
