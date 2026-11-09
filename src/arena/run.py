"""One arena trial per invocation (its own process); results merge into study.json."""

import argparse
import asyncio
import json
import signal
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

from arc_infra.arcade import find_games
from arc_infra.config import Config as DuckConfig
from arc_infra.server import hardware
from arc_infra.serving_checks import validate_api, wait_idle

from .isolation import isolate_trial
from .report import load_record, merge_trial
from .trial import run_trial


class _Done(Exception):
    """The serving benchmark finished; skip the gameplay path."""


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
        output, study, hardware=hardware(), api_validation=asyncio.run(validate_api(probe)),
        notebook_start=args.notebook_start_epoch,
    )
    competition = study.get('competition') if trial.get('competition', True) else None
    if trial.get('wall_seconds') is not None:
        study = {**study, 'wall_seconds': float(trial['wall_seconds'])}
    if competition:
        from .competition import gameplay_seconds, is_rerun, open_arcade, resolve_blocks
    else:
        from .competition import is_rerun
    hidden = bool(competition) and is_rerun() and not competition.get('rehearsal')
    games_dir = None if hidden else args.games_dir or find_games("/kaggle/input")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Notebook termination signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(1, hard_end - time.time()))
    folder = output / trial["name"]
    folder.mkdir(exist_ok=True)
    execution = {"trial": trial, "started_epoch": time.time(), "smoke": args.smoke}
    wall = study["smoke"]["wall_seconds"] if args.smoke else study["wall_seconds"]
    if (study.get("early_stop") or {}).get("always") and not args.smoke:
        wall = min(wall, study["early_stop"]["at_seconds"])  # an unconditional run stop plays at most at_seconds
    try:
        # The level bench sizes its own wall from what is left (arena.level_bench.LevelBench.wall_seconds).
        if (not competition
                and trial["kind"] not in ("level-bench", "expert-quant", "expert-usage", "server-check",
                                          "prune-quality", "mtp-screen")
                and hard_end - time.time() < wall + 480):
            raise TimeoutError("Insufficient time for a full arena trial")
        if study.get("backend") == "sglang":
            from .sglang import isolate

            execution["isolation"] = asyncio.run(isolate(args.base_url))
        elif study.get("isolate_trial_cache"):
            execution["isolation"] = asyncio.run(isolate_trial(args.base_url))
        else:
            asyncio.run(wait_idle(args.base_url))
        print(json.dumps({"event": "starting_trial", "trial": trial}), flush=True)
        if trial["kind"] == "serving-bench":
            from .throughput_bench import run_bench

            report = asyncio.run(run_bench(study, trial, output=folder, server_dir=args.server_dir,
                                           base_url=args.base_url, deadline_epoch=hard_end - 120,
                                           source=args.source))
            execution["report"] = report
            execution["status"] = report["status"]
            if report["status"] == "error":
                raise RuntimeError("Serving benchmark failed: " + "; ".join(report["task_errors"])[:2000])
            raise _Done
        if trial["kind"] == "kernel-quality":
            # Kernel work, phase 2: control reference or patched-server gate on fixed inputs (arena.kernel_gate).
            from .kernel_gate import run_quality

            report = asyncio.run(run_quality(study, trial, source=args.source, output=folder,
                                             server_dir=args.server_dir, base_url=args.base_url,
                                             deadline_epoch=hard_end - 120))
            execution["report"] = {k: report.get(k) for k in ("role", "status", "errors", "verdict", "bench")}
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "draft-head":
            # Draft-head fine-tune (28 September): dump the MTP input rows, stop the server, train the head.
            # The bootstrap ends the trial process at its start + wall + 600 s (TRIAL_GRACE).
            from .draft_head import run_draft_head

            report = run_draft_head(study, trial, source=args.source, output=folder, server_dir=args.server_dir,
                                    base_url=args.base_url,
                                    deadline_epoch=min(hard_end - 120, execution["started_epoch"] + wall + 560))
            execution["report"] = {k: report.get(k) for k in ("status", "errors", "dumped", "tokens", "result")}
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "expert-quant":
            # Expert quantization (28 September): MoE-input capture on fixed recorded sequences, then a GPU
            # program after the trial stopped the server (arena.expert_quant); it ends before the bootstrap's
            # trial timeout (hard_end - 120).
            from .expert_quant import run_expert_quant

            report = asyncio.run(run_expert_quant(study, trial, source=args.source, output=folder,
                                                  server_dir=args.server_dir, base_url=args.base_url,
                                                  deadline_epoch=hard_end - 140))
            execution["report"] = {k: report.get(k) for k in ("phase", "status", "errors", "layers_done")}
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "prune-quality":
            # Load-time expert pruning GPU check (28 September): held-out teacher forcing on an unpruned reference
            # server and on each pruned server (arena.expert_usage.run_prune_check).
            from .expert_usage import run_prune_check

            report = asyncio.run(run_prune_check(study, trial, source=args.source, output=folder,
                                                 server_dir=args.server_dir, base_url=args.base_url,
                                                 deadline_epoch=min(hard_end - 150, time.time() + float(
                                                     trial.get("wall_seconds", 1e9)))))
            execution["report"] = {k: report.get(k) for k in ("status", "errors", "server")}
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "server-check":
            # Server-start diagnostic arm (expert usage, 28 September): reaching here means the server started.
            from .expert_usage import server_check

            report = asyncio.run(server_check(study, trial, base_url=args.base_url, output=folder))
            execution["report"] = report
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "expert-usage":
            # Expert-usage measurement (28 September): routing statistics and pruned-router quality on fixed
            # recorded sequences (arena.expert_usage); its passes run until shortly before the bootstrap's
            # trial timeout (hard_end - 120).
            from .expert_usage import run_expert_usage

            report = asyncio.run(run_expert_usage(study, trial, source=args.source, output=folder,
                                                  base_url=args.base_url,
                                                  deadline_epoch=min(hard_end - 150,
                                                                     time.time() + float(trial.get("wall_seconds",
                                                                                                   1e9)))))
            execution["report"] = {k: report.get(k) for k in ("status", "errors", "skipped")}
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "mtp-screen":
            # MTP exactness screen (19 October): play-settings sampling with the served draft vs plain (arc2.mtp_screen)
            from arc2.mtp_screen import run_mtp_screen

            report = asyncio.run(run_mtp_screen(study, trial, source=args.source, output=folder,
                                                base_url=args.base_url, deadline_epoch=hard_end - 180))
            execution["report"] = {k: report.get(k) for k in ("status", "drafts", "samples", "tokens", "errors",
                                                              "summary", "rescore", "accepted_per_position")}
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "nll-parity":
            # TTT parity reference (26 October): served log-probs of true answers (arc2.nll_parity)
            from arc2.nll_parity import run_nll_parity

            report = asyncio.run(run_nll_parity(study, trial, source=args.source, output=folder,
                                                base_url=args.base_url))
            execution["report"] = report
            execution["status"] = report["status"]
            raise _Done
        if trial["kind"] == "arc2":
            # ARC-AGI-2 arm (9 October): static tasks, verify-and-refine (src/arc2); no arcade, no scorecard.
            from arc2.trial import run_arc2

            report = asyncio.run(run_arc2(study, trial, source=args.source, output=folder, server_dir=args.server_dir,
                                          base_url=args.base_url, deadline_epoch=hard_end - 180,
                                          python=args.analysis_python))
            execution["report"] = report
            execution["status"] = report["status"]
            if report["status"] == "error":
                raise RuntimeError("arc2 trial failed: " + "; ".join(report["task_errors"])[:2000])
            raise _Done
        if trial["kind"] == "level-bench":
            from .level_bench import run_level_bench

            report = asyncio.run(run_level_bench(
                study, trial, source=args.source, games_dir=games_dir, output=folder, server_dir=args.server_dir,
                analysis_python=args.analysis_python, base_url=args.base_url, deadline_epoch=hard_end - 300))
            # study.json keeps the summary; the attempts stay in the trial's level-bench.json.
            execution["report"] = {k: v for k, v in report.items() if k not in ("attempts", "config")}
            execution["status"] = report["status"]
            raise _Done
        lifecycle = open_arcade(study, games_dir, folder) if competition else nullcontext((study, None))
        with lifecycle as (resolved, arcade):
            extra = {}
            if competition:
                resolved['wall_seconds'] = gameplay_seconds(resolved, hard_end)
                # Blocks (30 September): the scheduler's wall is the competition play time (arena.competition).
                resolve_blocks(resolved, args.notebook_start_epoch)
                record['study'] = resolved
                extra['arcade'] = arcade
                if is_rerun():
                    extra['gameplay_deadline_epoch'] = hard_end - competition['cleanup_seconds']
            report = asyncio.run(
                run_trial(
                    resolved, trial, source=args.source, games_dir=games_dir, output=folder,
                    server_dir=args.server_dir,
                    analysis_python=args.analysis_python, base_url=args.base_url, smoke=args.smoke,
                    **extra,
                )
            )
        execution["report"] = report
        execution["status"] = report["status"]
        if report["status"] == "error":
            raise RuntimeError("Trial has agent or telemetry failures: " + "; ".join(report["task_errors"])[:2000])
    except _Done:
        pass
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
    # On the hidden rerun the gateway scorecard is the submission; a trial that played with a
    # failed session or preemptions must not fail the notebook afterwards.
    acceptable = {'complete', 'complete_with_preemptions', 'error'} if competition and is_rerun() else {'complete'}
    if trial["kind"] in ("serving-bench", "level-bench", "draft-head", "expert-quant", "expert-usage",
                         "server-check", "prune-quality"):
        acceptable = {'complete', 'complete_with_errors'}
    if execution["status"] not in acceptable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
