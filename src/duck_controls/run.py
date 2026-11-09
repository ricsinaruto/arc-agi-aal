"""Run the Duck controls: duck-transfer, then the public loop.

Both trials play the same public games for the same gameplay seconds with fresh
environments and scorecards. Each trial's server keeps the Duck comparison profile (MTP 3,
eight slots) with the study's per-trial ``servers`` record (context, explicit KV or a
profiled allocation on the ARC fork, prefix caching); since v5b the notebook starts one
server per trial and invokes this module once per trial (``--trial``, ``--server-dir``),
merging every run into ``output/study.json``. The public control always runs at a 32K
analyzer window. The study either selects games explicitly (one session each) or plays a
full public fleet of 25 or 110 sessions. The public trial runs the notebook's own TAAF
benchmark in a subprocess with the analyzer environment the serving script persisted.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from arc_infra import fleet
from arc_infra.arcade import find_games
from arc_infra.config import Config
from arc_infra.metrics import GpuMonitor
from arc_infra.server import hardware
from arc_infra.serving import SERVER_CONTEXT
from arc_infra.serving_checks import validate_api, validate_config, wait_idle

ANALYZER_PREFIXES = ("LOCAL_ANALYZER_", "OPENAI_", "INFERENCE_ANALYZER_", "MULTIMODAL_")
# Selected-game studies keep the planning clock of the 25-session studies they extend.
SELECTED_CLOCK_SESSIONS = 25


def analyzer_environment(server_dir, context_window=None, temperature=None):
    """The analyzer settings the pinned serving script persisted for the notebook's agent.

    ``context_window`` pins the public loop's window when the server runs a larger one;
    ``temperature`` (v5c) overrides the notebook's 0.6 so the
    control samples at the same temperature as the corrected agent (0.7).
    """
    env = json.loads((Path(server_dir) / "serving-environment.json").read_text())
    selected = {k: str(v) for k, v in env.items() if k.startswith(ANALYZER_PREFIXES)}
    if "LOCAL_ANALYZER_BASE_URL" not in selected:
        raise RuntimeError("The serving script did not persist the analyzer environment")
    if context_window is not None:
        selected["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str(int(context_window))
    if temperature is not None:
        selected["LOCAL_ANALYZER_TEMPERATURE"] = str(float(temperature))
    return selected


def scope(study):
    """``(games, sessions)`` from a study: explicit games or a 25/110-session fleet."""
    games, sessions = study.get("games"), study.get("sessions")
    if bool(games) == bool(sessions):
        raise ValueError("A study lists either games or a session count")
    if sessions is not None and sessions not in (25, 110):
        raise ValueError("Fleet studies use 25 or 110 sessions")
    return (list(games) if games else None), sessions


async def corrected_trial(cfg, games_dir, folder, interpreter, games=None, sessions=None):
    from duck_transfer import loader

    profile = loader.configure(cfg)
    from duck_transfer.transfer import TransferDuckAgent

    report = await fleet.run_replicas(
        cfg,
        games_dir,
        folder,
        interpreter,
        GpuMonitor(folder, cfg.base_url),
        agent_class=TransferDuckAgent,
        agent_profile="duck-transfer",
        profile_source=profile,
        sessions=sessions or SELECTED_CLOCK_SESSIONS,
        games=games,
        clock_sessions=SELECTED_CLOCK_SESSIONS if games else None,
    )
    if games:
        report["selected_games"] = list(games)
        report["timing_normalization"] = (
            f"competition-equivalent clock of a {SELECTED_CLOCK_SESSIONS}-session fleet"
        )
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    return report


def corrected_summary(report):
    games = [
        {
            k: g.get(k)
            for k in (
                "game_id",
                "status",
                "error",
                "score",
                "levels",
                "total_levels",
                "actions",
                "resets",
                "model_calls",
                "prompt_tokens",
                "completion_tokens",
                "seconds",
            )
        }
        for g in report["games"]
    ]
    return {
        "games": games,
        "mean_score": report["mean_score"],
        "actions": report["actions"],
        "levels": report["levels"],
        "failed_sessions": report["failed_sessions"],
        "task_errors": report["task_errors"],
        "engine": report["engine"],
        "gameplay_seconds": report["gameplay_seconds"],
        "completed_replies": report["completed_replies"],
        "completed_reply_tokens": report["completed_reply_tokens"],
    }


def public_command(source, folder, games_dir, seconds, games=None, passes=1):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("public_trial.py")),
        "--config",
        str(Path(source) / "configs/duck_public_notebook.json"),
        "--games-dir",
        str(games_dir),
        "--output",
        str(folder / "taaf"),
        "--seconds",
        str(seconds),
        "--passes",
        str(passes),
    ]
    for game in games or ():
        command += ["--game", game]
    return command


def public_trial(
    source, output, folder, games_dir, seconds, env, games=None, passes=1, server_dir=None,
    temperature=None,
):
    command = public_command(source, folder, games_dir, seconds, games, passes)
    child_env = {
        **env,
        **analyzer_environment(
            server_dir or Path(output) / "server", context_window=SERVER_CONTEXT,
            temperature=temperature,
        ),
        "PYTHONPATH": str(Path(source) / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (folder / "public-trial.log").open("w") as log:
        process = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    summary_path = folder / "taaf/summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Public trial exited {process.returncode} without a summary")
    summary = json.loads(summary_path.read_text())
    summary["exit_code"] = process.returncode
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-python", required=True)
    parser.add_argument("--notebook-start-epoch", required=True, type=float)
    parser.add_argument("--trial", action="append", help="Trial name(s) to run; default all")
    parser.add_argument("--server-dir", help="This run's server directory (default <output>/server)")
    args = parser.parse_args()
    source = Path(args.source)
    cfg = Config.load(str(source / "configs/experiment.yaml"))
    validate_config(cfg)
    study = json.loads(Path(args.study).read_text())
    games, sessions = scope(study)
    if study["gameplay_seconds"] != cfg.gameplay_seconds:
        raise ValueError("Wrong control scope")
    trials = [t for t in study["trials"] if not args.trial or t["name"] in args.trial]
    if args.trial and len(trials) != len(args.trial):
        raise ValueError("Unknown trial name")
    for trial in trials:
        # v5c: the fork may serve more context than the agent's limit (262K server, 64K agent);
        # the public control's pinned server still has to match the config.
        spec = server_spec(study, trial)
        served = spec.get("max_model_len", SERVER_CONTEXT)
        if served < cfg.max_model_len or (not spec.get("gpu_memory_utilization") and served != cfg.max_model_len):
            raise ValueError("Study server context differs from the experiment config")
    # The public loop plays whole passes over the 25 games: 25 sessions is one pass,
    # 110 sessions four passes (100 sessions; the loop cannot play a partial pass).
    passes = 1 if games else sessions // 25
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    server_dir = Path(args.server_dir) if args.server_dir else output / "server"
    hard_end = args.notebook_start_epoch + study["notebook_seconds"]
    record = load_record(output, study, cfg, args.notebook_start_epoch)
    games_dir = find_games("/kaggle/input")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Notebook termination signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(1, hard_end - time.time()))
    (output / "study.json").write_text(json.dumps(record, indent=2))
    for trial in trials:
        folder = output / trial["name"]
        folder.mkdir()
        spec = server_spec(study, trial)
        record["serving_profiles"][trial["name"]] = serving_profile(spec, cfg)
        execution = {
            "trial": trial, "config": cfg.to_dict(), "server": spec,
            "server_dir": str(server_dir), "started_epoch": time.time(),
        }
        try:
            if hard_end - time.time() < cfg.gameplay_seconds * max(1, passes) + 150:
                raise TimeoutError("Insufficient time for a full gameplay trial")
            asyncio.run(wait_idle(cfg.base_url))
            print(json.dumps({"event": "starting_trial", "trial": trial}), flush=True)
            if trial["kind"] == "corrected":
                report = asyncio.run(
                    corrected_trial(cfg, games_dir, folder, args.analysis_python, games, sessions)
                )
                execution["summary"] = corrected_summary(report)
                if (
                    report["failed_sessions"]
                    or report["task_errors"]
                    or not report["engine"]["valid"]
                ):
                    raise RuntimeError(
                        "Trial has agent or telemetry failures; do not treat it as a valid score"
                    )
            elif trial["kind"] == "public":
                summary = public_trial(
                    source, output, folder, games_dir, cfg.gameplay_seconds, dict(os.environ),
                    games, passes, server_dir=server_dir, temperature=cfg.temperature,
                )
                execution["summary"] = summary
                if summary["status"] != "complete":
                    raise RuntimeError(summary.get("error") or "Public trial did not complete")
            else:
                raise ValueError(f"Unknown trial kind: {trial['kind']}")
            execution["status"] = "complete"
        except Exception as exc:  # noqa: BLE001 - every trial outcome is recorded
            execution.update(status="error", error=f"{type(exc).__name__}: {exc}")
            (folder / "exception.txt").write_text(traceback.format_exc())
        finally:
            execution["elapsed_seconds"] = time.time() - execution["started_epoch"]
            (folder / "execution.json").write_text(json.dumps(execution, indent=2))
            record["trials"] = [t for t in record["trials"] if t["trial"]["name"] != trial["name"]]
            record["trials"].append(execution)
            (output / "study.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"event": "trial_result", **execution}), flush=True)
    signal.setitimer(signal.ITIMER_REAL, 0)
    if any(t["status"] != "complete" for t in record["trials"]):
        raise SystemExit(1)


def server_spec(study, trial):
    """The trial's server record: per-trial ``servers`` (v5b) or the shared ``server``."""
    servers = study.get("servers")
    if servers:
        return servers[trial["name"]]
    return study.get("server", {})


def serving_profile(spec, cfg):
    allocation = (
        f"profiled KV cache at utilization {spec['gpu_memory_utilization']} (ARC fork)"
        if spec.get("gpu_memory_utilization")
        else f"explicit {spec.get('kv_cache_gib', 5)} GiB KV (pinned script)"
    )
    return (
        f"flash-next-nvfp4 Duck comparison profile; {allocation}; prefix caching "
        f"{'on' if spec.get('prefix_caching') else 'off'}; server context "
        f"{spec.get('max_model_len', SERVER_CONTEXT)}, agent limit {cfg.max_model_len}, "
        f"public loop at {SERVER_CONTEXT}"
    )


def load_record(output, study, cfg, notebook_start):
    """The study record: extended when an earlier invocation (another trial) wrote one."""
    path = output / "study.json"
    if path.is_file():
        record = json.loads(path.read_text())
        record.setdefault("serving_profiles", {})
        record["api_validations"] = [*record.get("api_validations", []), asyncio.run(validate_api(cfg))]
        return record
    return {
        "study": study,
        "hardware": hardware(),
        "serving_profiles": {},
        "setup_seconds": time.time() - notebook_start,
        "api_validation": asyncio.run(validate_api(cfg)),
        "trials": [],
    }


if __name__ == "__main__":
    main()
