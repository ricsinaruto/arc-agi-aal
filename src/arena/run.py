"""One RL trial per invocation (its own process): ``rl-loop`` (training) or ``level-bench`` (held-out read); results
merge into study.json."""

import argparse
import asyncio
import json
import signal
import time
import traceback
from pathlib import Path

from arc_infra.arcade import find_games
from arc_infra.config import Config as DuckConfig
from arc_infra.server import hardware
from arc_infra.serving_checks import validate_api, wait_idle

from .isolation import isolate_trial


def load_record(output, study, *, hardware, api_validation, notebook_start):
    """The study record, extended when an earlier invocation (another trial) wrote one."""
    path = Path(output) / "study.json"
    if path.is_file():
        record = json.loads(path.read_text())
        record["api_validations"] = [*record.get("api_validations", []), api_validation]
        return record
    return {
        "study": study,
        "hardware": hardware,
        "setup_seconds": time.time() - notebook_start,
        "api_validation": api_validation,
        "trials": [],
    }


def merge_trial(record, execution):
    record["trials"] = [t for t in record["trials"] if t["trial"]["name"] != execution["trial"]["name"]]
    record["trials"].append(execution)
    record["comparison"] = [
        {
            "trial": t["trial"]["name"],
            "status": t.get("status"),
            "mean_score_equal_weight_25_games": (t.get("report") or {}).get("mean_score_equal_weight_25_games"),
            "levels": (t.get("report") or {}).get("levels"),
            "score_denominator_games": (t.get("report") or {}).get("score_denominator_games"),
            "actions": (t.get("report") or {}).get("actions"),
            "switch_wall_seconds": ((t.get("report") or {}).get("scheduler") or {}).get("switch_wall_seconds"),
            "preemptions": ((t.get("report") or {}).get("engine") or {}).get("preemptions"),
            "window": ((t.get("report") or {}).get("window_choice") or {}).get("window"),
        }
        for t in record["trials"]
    ]
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--server-dir", required=True)
    parser.add_argument("--games-dir")
    parser.add_argument("--analysis-python")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--notebook-start-epoch", required=True, type=float)
    parser.add_argument("--notebook-seconds", type=float)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    study = json.loads(Path(args.study).read_text())
    trial = next((t for t in study["trials"] if t["name"] == args.trial), None)
    if trial is None:
        raise ValueError(f"Unknown trial name: {args.trial}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    notebook_seconds = args.notebook_seconds or (
        study["smoke"]["notebook_seconds"] if args.smoke else study["notebook_seconds"]
    )
    hard_end = args.notebook_start_epoch + notebook_seconds
    probe = DuckConfig.load(str(Path(args.source) / "configs/duck_controls.yaml"))
    probe.base_url = args.base_url
    probe.model = study.get("served_model", probe.model)
    probe.serving_backend = study.get("backend", "vllm")
    record = load_record(
        output, study, hardware=hardware(),
        api_validation=asyncio.run(validate_api(probe)),
        notebook_start=args.notebook_start_epoch,
    )
    if trial.get('wall_seconds') is not None:
        study = {**study, 'wall_seconds': float(trial['wall_seconds'])}
    games_dir = args.games_dir or find_games("/kaggle/input")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Notebook termination signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(1, hard_end - time.time()))
    folder = output / trial["name"]
    folder.mkdir(exist_ok=True)
    execution = {"trial": trial, "started_epoch": time.time(), "smoke": args.smoke}
    try:
        if trial["kind"] not in ("rl-loop", "level-bench"):
            raise ValueError(f"trial kind {trial['kind']!r}: rl-loop or level-bench")
        if study.get("isolate_trial_cache"):
            execution["isolation"] = asyncio.run(isolate_trial(args.base_url))
        else:
            asyncio.run(wait_idle(args.base_url))
        print(json.dumps({"event": "starting_trial", "trial": trial}), flush=True)
        kwargs = dict(source=args.source, games_dir=games_dir, output=folder, server_dir=args.server_dir,
                      analysis_python=args.analysis_python, base_url=args.base_url, deadline_epoch=hard_end - 300)
        if trial["kind"] == "rl-loop":
            # RL: play, updates, held-out reads, stop rules (arena.rl_loop)
            from .rl_loop import run_rl_loop

            report = asyncio.run(run_rl_loop(study, trial, **kwargs))
            execution["report"] = {k: v for k, v in report.items() if k not in ("settings",)}
        else:
            from .level_bench import run_level_bench

            report = asyncio.run(run_level_bench(study, trial, **kwargs))
            # study.json keeps the summary; the attempts stay in the trial's level-bench.json.
            execution["report"] = {k: v for k, v in report.items() if k not in ("attempts", "config")}
        execution["status"] = report["status"]
    except Exception as exc:  # noqa: BLE001 - every trial outcome is recorded
        execution.update(status="error", error=f"{type(exc).__name__}: {exc}")
        (folder / "exception.txt").write_text(traceback.format_exc())
    finally:
        execution["elapsed_seconds"] = time.time() - execution["started_epoch"]
        (folder / "execution.json").write_text(json.dumps(execution, indent=2))
        merge_trial(record, execution)
        (output / "study.json").write_text(json.dumps(record, indent=2))
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"event": "trial_result", "trial": trial["name"], "status": execution["status"], "error": execution.get("error")}), flush=True)
    if execution["status"] not in ("complete", "complete_with_errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
