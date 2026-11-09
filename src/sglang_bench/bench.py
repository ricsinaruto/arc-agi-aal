"""Per-point driver: start the SGLang server, play the public Duck loop, record everything.

Runs under the agent venv (httpx, arc_agi). One point = one server profile and its games;
each point starts a fresh server and stops it afterwards, so failures never bleed across.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import launch, runtime
from .metrics import SglangMonitor
from .profile import ServerProfile

DEFAULT_CONTEXT_WINDOW = 131_072
BOOT_ALLOWANCE = launch.READY_TIMEOUT + 60
DRAIN_ALLOWANCE = 200


def analyzer_environment(base_url: str, model: str, context_window: int = DEFAULT_CONTEXT_WINDOW) -> dict[str, str]:
    """The public notebook's analyzer block (``serving_setup.persist_analyzer_environment``)
    at the raised window and pointed at the SGLang server."""
    return {
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_MODEL_ID": model,
        "INFERENCE_ANALYZER_MODEL": model,
        "OPENAI_API_KEY": "offline-kaggle-local-server",
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str(context_window),
        "LOCAL_ANALYZER_MAX_OUTPUT": "0",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6",
        "LOCAL_ANALYZER_TOP_P": "0.95",
        "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
    }


def public_command(source, folder, games_dir, seconds, games, *, python=None) -> list[str]:
    source = Path(source)
    command = [
        python or sys.executable, "-u", str(source / "src/duck_controls/public_trial.py"),
        "--config", str(source / "configs/duck_public_notebook.json"),
        "--games-dir", str(games_dir), "--output", str(Path(folder) / "taaf"),
        "--seconds", str(seconds), "--passes", "1",
    ]
    for game in games:
        command += ["--game", game]
    return command


def run_public_trial(source, folder, games_dir, seconds, games, env, *, timeout=None, python=None) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    command = public_command(source, folder, games_dir, seconds, games, python=python)
    timeout = timeout or seconds + 90 + 300
    started = time.time()
    log_path = folder / "public-trial.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            code = f"killed after {timeout} s (exit {code})"
    record = {"command": command, "exit_code": code, "seconds": round(time.time() - started, 1)}
    summary_path = folder / "taaf/summary.json"
    if summary_path.is_file():
        record["summary"] = json.loads(summary_path.read_text())
    else:
        record["error"] = f"no summary.json; log tail: {launch.tail(log_path, 40)}"
    return record



def run_transfer_trial(source, folder, games_dir, seconds, games, env, *, base_url, model, config, python=None):
    folder = Path(folder)
    target = folder / "transfer"
    command = [python or sys.executable, "-u", "-m", "sglang_bench.transfer",
               "--source", str(source), "--output", str(target), "--games-dir", str(games_dir),
               "--seconds", str(seconds), "--base-url", base_url, "--model", model, "--config", config]
    for game in games:
        command += ["--game", game]
    started = time.time()
    with (folder / "transfer-trial.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=seconds + 90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError("Duck-transfer child exceeded gameplay plus cleanup allowance") from None
    result = {"command": command, "exit_code": code, "seconds": round(time.time() - started, 1)}
    if (target / "report.json").is_file():
        result["summary"] = json.loads((target / "report.json").read_text())
    else:
        result["error"] = launch.tail(folder / "transfer-trial.log", 40)
    return result


def public_rows(summary: dict | None) -> list[dict]:
    return [
        {k: r.get(k) for k in ("game_id", "state", "final_score", "levels_completed", "number_of_levels",
                               "actions", "generated_tokens")}
        for r in (summary or {}).get("rows", [])
    ]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")


class Bench:
    def __init__(self, *, source, study, output, model_dir, notebook_start, games_dir, agent_python):
        self.source, self.study, self.output = Path(source), study, Path(output)
        self.model_dir, self.notebook_start = Path(model_dir), notebook_start
        self.games_dir, self.agent_python = games_dir, agent_python
        self.deadline = notebook_start + study["notebook_seconds"]
        self.cache_base = runtime.SCRATCH / "cache"
        self.record = {
            "study": study, "started_epoch": time.time(), "model_dir": str(self.model_dir),
            "points": [], "hardware": None, "runtime_check": None,
        }
        try:
            from arc_infra.server import hardware

            self.record["hardware"] = hardware()
        except Exception as exc:  # noqa: BLE001 - recorded
            self.record["hardware"] = {"error": f"{type(exc).__name__}: {exc}"}
        check = self.output / "runtime-check.json"
        if check.is_file():
            self.record["runtime_check"] = json.loads(check.read_text())

    def save(self):
        self.record["elapsed_seconds"] = round(time.time() - self.record["started_epoch"], 1)
        write_json(self.output / "study.json", self.record)

    def run_point(self, point: dict) -> dict:
        profile = ServerProfile.from_dict(point["profile"])
        seconds = point.get("seconds", self.study["game_seconds"])
        folder = self.output / point["name"]
        folder.mkdir(parents=True, exist_ok=True)
        record = {"point": point["name"], "profile": profile.to_dict(), "games": point["games"],
                  "seconds": seconds, "status": "error", "timings": {}, "started_epoch": time.time()}
        startup_timeout = self.study.get("startup_timeout_seconds", launch.READY_TIMEOUT)
        needed = startup_timeout + 60 + seconds + 90 + DRAIN_ALLOWANCE
        if time.time() + needed > self.deadline:
            record.update(status="skipped", error=f"{needed} s needed, {int(self.deadline - time.time())} s left")
            return record
        server = None
        monitor = None
        try:
            t0 = time.monotonic()
            record["preflight"] = launch.preflight(profile, model_dir=self.model_dir, repo_root=runtime.REPO_ROOT)
            env = launch.recipe_env(profile, cache_base=self.cache_base, cuda_home=runtime.CUDA_HOME,
                                    repo_root=runtime.REPO_ROOT, scratch=runtime.SCRATCH / "tmp")
            argv = launch.recipe_argv(profile, model_dir=self.model_dir, repo_root=runtime.REPO_ROOT,
                                      sglang_exe=runtime.REPO_ROOT / ".venv/bin/sglang")
            server = launch.start(profile, working=folder, env=env, argv=argv, preflight_record=record["preflight"])
            record["identity"] = json.loads(server.identity_path.read_text())
            record["startup"] = launch.wait_ready(server, timeout=startup_timeout)
            record["clamp"] = launch.clamp_check(profile, record["startup"])
            record["timings"]["boot_seconds"] = round(time.monotonic() - t0, 1)
            record["gpu_idle"] = launch.gpu_snapshot()
            record["warmup"] = launch.warmup(server, probes=bool(point.get("probes")))
            env_public = {**os.environ, **analyzer_environment(server.base_url, profile.served_model_name,
                                                                 self.study["analyzer_context_window"]),
                          "PYTHONPATH": str(self.source / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                          "ONLY_RESET_LEVELS": "true", "MPLBACKEND": "Agg"}
            monitor = SglangMonitor(folder, server.base_url).start()
            t1 = time.monotonic()
            harness = point.get("harness", "public")
            if harness == "duck-transfer":
                record["public"] = run_transfer_trial(
                    self.source, folder, self.games_dir, seconds, point["games"], env_public,
                    python=self.agent_python, base_url=server.base_url, model=profile.served_model_name,
                    config=self.study["agent_config"],
                )
                record["harness"] = harness
            else:
                record["public"] = run_public_trial(self.source, folder, self.games_dir, seconds, point["games"],
                                                    env_public, python=self.agent_python)
            record["timings"]["play_seconds"] = round(time.monotonic() - t1, 1)
            record["metrics"] = monitor.stop()
            monitor = None
            if harness == "duck-transfer":
                transfer = record["public"].get("summary", {})
                record["public_rows"] = [
                    {"game_id": g["game_id"], "state": g["status"], "final_score": g["score"],
                     "levels_completed": g["levels"], "actions": g["actions"],
                     "generated_tokens": g["completion_tokens"]} for g in transfer.get("games", [])
                ]
                record["gameplay_metrics"] = transfer.get("engine")
            else:
                record["public_rows"] = public_rows(record["public"].get("summary"))
            record["stop"] = launch.stop(server)
            server = None
            log_text = (folder / "sglang-server.log").read_text(errors="replace")
            stats = launch.request_time_stats(log_text)
            record["request_time_stats"] = {"rows": len(stats), "cold_vs_warm": launch.cold_vs_warm(stats)}
            record["startup"].update({k: v for k, v in launch.parse_startup(log_text).items() if k == "jit_compile_lines"})
            summary = record["public"].get("summary") or {}
            if summary.get("status") == "complete" and record["public"].get("exit_code") == 0:
                record["status"] = "complete"
            else:
                record["error"] = summary.get("error") or record["public"].get("error") or "public trial incomplete"
        except Exception as exc:  # noqa: BLE001 - recorded; the sweep continues
            record["error"] = f"{type(exc).__name__}: {exc}"
            (folder / "exception.txt").write_text(traceback.format_exc())
        finally:
            if monitor is not None:
                record["metrics"] = monitor.stop()
            if server is not None:
                try:
                    record["stop"] = launch.stop(server)
                except Exception as exc:  # noqa: BLE001
                    record["stop_error"] = f"{type(exc).__name__}: {exc}"
            record["timings"]["total_seconds"] = round(time.time() - record["started_epoch"], 1)
            write_json(folder / "point.json", record)
        return record

    def run(self, names=None) -> dict:
        points = [p for p in self.study["points"] if not names or p["name"] in names]
        for point in points:
            print(json.dumps({"event": "point_start", "point": point["name"], "games": point["games"]}), flush=True)
            result = self.run_point(point)
            self.record["points"].append(result)
            self.save()
            print(json.dumps({"event": "point_result", "point": point["name"], "status": result["status"],
                              "error": result.get("error"),
                              "tokens_per_second": (result.get("metrics") or {}).get("accepted_generation_tokens_per_second"),
                              "rows": result.get("public_rows")}), flush=True)
        self.record["status"] = "complete" if all(p["status"] == "complete" for p in self.record["points"]) else "error"
        self.save()
        return self.record


def find_games_dir(input_root="/kaggle/input"):
    from arc_infra.arcade import find_games

    return find_games(input_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=str(runtime.MODEL_DIR))
    parser.add_argument("--notebook-start-epoch", type=float, required=True)
    parser.add_argument("--point", action="append", help="Run only these points (default: all)")
    parser.add_argument("--games-dir", default=None)
    parser.add_argument("--agent-python", default=None, help="Interpreter for the public loop (default: this one)")
    parser.add_argument("--input-root", default="/kaggle/input")
    args = parser.parse_args(argv)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    study = json.loads(Path(args.study).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    remaining = args.notebook_start_epoch + study["notebook_seconds"] - time.time()
    if remaining > 0:
        signal.signal(signal.SIGALRM, interrupt)
        signal.alarm(int(remaining))
    games_dir = args.games_dir or find_games_dir(args.input_root)
    bench = Bench(source=args.source, study=study, output=output, model_dir=args.model_dir,
                  notebook_start=args.notebook_start_epoch, games_dir=games_dir,
                  agent_python=args.agent_python or sys.executable)
    try:
        record = bench.run(args.point)
    except KeyboardInterrupt as exc:
        bench.record["status"] = f"interrupted: {exc}"
        bench.save()
        return 1
    finally:
        shutil.rmtree(runtime.SCRATCH / "tmp", ignore_errors=True)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
