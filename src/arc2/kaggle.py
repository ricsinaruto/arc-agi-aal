"""The packaged ARC-AGI-2 run on four L4s (the Kaggle notebook's one command; the same on a Modal L4 host).

    python -m arc2.kaggle --bundle B --working /tmp/arc2 --output /kaggle/working/arc2 --input-root /kaggle/input
        --challenges <arc-agi_test_challenges.json> --submission /kaggle/working/submission.json --end-epoch E
        [--solutions <json>] [--tasks N] [--copies K] [--play-seconds S] [--resume-state <play-state.json>]

Order: a fallback ``submission.json`` (every test input as its own answer) before anything can fail; the agent
environment from the bundle's wheels; the server of ``arc2.l4_serving`` under ``OPTIONS``; its probe (first turns at
once, the prefill slots and the streams that fit); the roster loop on the challenge tasks until ``--end-epoch`` less
``STOP_SECONDS``, rewriting ``submission.json`` every minute; the server stopped. A failed start is tried once more.
A play that ends early (the engine died) is followed by a new server, 0.03 less of each card, and a play that takes
up every task's programs, tokens and phase: up to ``RESTARTS`` times, while a start and ``MIN_PLAY_SECONDS`` fit.
Whatever fails, the file that exists is whole and names every task. ``--solutions`` (a public set) adds the score to
the report; ``--tasks`` plays the first N (the notebook's saved version: minutes, not hours). ``--resume-state``
(a host that takes its machine back, not Kaggle): an earlier process's ``play-state.json``; the first play takes up
its tasks, and the rows of the submission file that is there are kept.
"""

import argparse
import json
import os
import runpy
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

# The scored L4 run's settings (arc2-l4-score-v4): 32 server streams (the recipe takes 8, 12, 16 or 25..96), of
# which the probe's fit lets 23 play at 0.90 of each card (``slots``). 0.90 with the PLE layer's padding bounded
# (rehearsal 7: 330 tok/s for 289 at 0.85, no allocator retry in 7,150 s; unbounded, engines died at 0.85-0.87).
OPTIONS = {"streams": 32, "kv": "int4", "tasks": 48, "wall": 5400, "utilization": "0.90", "expert_parallel": True,
           "ple_pack": True, "ple_drop": True,  # ple_drop: v41 loaded weights in 563 s (arc2.l4_ple_drop)
           # the tool arm (10 October: 12 of 59 right against 4 after 20 minutes on the L4s, and the plain loop's
           # engine died at 0.85 in both rehearsals); the bundle's roster trial must carry it
           "roster": {"arm": "tool", "rates": [330]}, "prefix_patches": True, "sticky": True, "kv4_guard": True,
           # rehearsal 3: the engine died at 0.85 as 12 turns began within 18 s (two prefills at once, not four),
           # and at 0.82 nine prompt tokens in ten were computed again (the attempt keeps its blocks)
           # rehearsal 9 against 7: waiting for blocks with the attempt's own held buys no tok/s (325 both)
           "prefill_slots": 2, "hold": True, "grow_wait": 0}
SERVER_SECONDS = 3720  # the start's own limits: 3,000 s to ready, 3,600 s in all (arc2.l4_serving.SERVER_ENV)
PROBE_SECONDS = 300
EARLY_SECONDS = 180  # a play that ends this long before its wall did not end by the clock
RESTARTS = 3  # further plays after a play that ended early
STOP_SECONDS = 420  # after the play: the last reports, the server stopped, the evidence copied
MIN_PLAY_SECONDS = 600
# The server's compile caches (arc_infra.serving_setup.CACHE_ROOT / COMPILE_CACHE_ROOT): a fresh one makes the first
# start JIT-compile its Triton kernels inside CUDA-graph capture, where 6 of 7 first starts of the 1,600-token-block
# stack on Modal hung (stuck at the 4-bit cache's rotation kernel, int4_per_token_head.single_rht); every second
# start in the same container, with the caches warm, served. The bundle carries a warm copy (``CACHE_FILE``).
CACHE_ROOTS = (Path("/tmp/qwen38-flash-next-vllm-cache"), Path("/tmp/qwen38-flash-next-vllm-compile-cache"))
CACHE_FILE = "compile-cache.tar.gz"


def fallback(challenges):
    return {name: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]
            for name, task in challenges.items()}


def problems(submitted, challenges):
    """What the competition's scorer would not read: a task or an attempt missing, or a grid that is not one."""
    from .tasks import valid

    found = []
    for name, task in challenges.items():
        rows = submitted.get(name)
        if not isinstance(rows, list) or len(rows) != len(task["test"]):
            found.append(f"{name}: {len(rows) if isinstance(rows, list) else 'no'} answers for {len(task['test'])}")
        elif not all(isinstance(r, dict) and valid(r.get("attempt_1")) and valid(r.get("attempt_2")) for r in rows):
            found.append(f"{name}: an attempt is not a grid")
    return found


def health(base_url):
    """The server's /health status, or the error's name."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.removesuffix("/v1") + "/health", timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


def live_row(record, row, elapsed, epoch, history=()):
    """A row of the live record in the shape ``tools/ops`` reads (``mon.py``, ``compare_runs.py``), from the
    driver's ``run.json`` and the play's last progress row: the job's clock, the tokens charged to the tasks over
    all plays, the gate, and, where the task files carry the answers, the score so far. ``history``: earlier
    ``(elapsed, tokens)`` for the rate of the last half hour."""
    plays, stage = record.get("plays") or [], str(record.get("stage") or "")
    playing = stage == f"play-{len(plays) + 1}" and bool(row)
    played = sum(p.get("seconds") or 0 for p in plays) + (row.get("elapsed_seconds") or 0 if playing else 0)
    tokens, gate = (row or {}).get("completion_tokens") or 0, (row or {}).get("gate") or {}
    tokens = max([tokens, *(point[1] for point in history[-1:])])  # a restarted play's first row may trail the last
    engine = {"generated_tokens": tokens, "tokens_per_second_mean": tokens / played if played else None,
              "running": gate.get("inflight") if playing else 0,
              "kv_cache_fraction": gate["reserved"] / gate["capacity"] if playing and gate.get("capacity") else None}
    old = [point for point in history if point[0] <= elapsed - 1800]
    if old and elapsed > old[-1][0]:
        engine["tokens_per_second_30m"] = (tokens - old[-1][1]) / (elapsed - old[-1][0])
    out = {"elapsed_seconds": round(elapsed, 1), "epoch": epoch, "phase": stage, "play_seconds": round(played, 1),
           "restarts": max(0, len(record.get("attempts") or []) - 1), "finished": stage in ("done", "failed"),
           "gate": {"waiting": gate.get("waiting")}, "priority": {"slots": gate.get("slots")},
           "utilization": (record.get("choice") or {}).get("utilization"),
           "live": {"engine": engine, "errors": (row or {}).get("request_errors")}}
    final = (row or {}).get("final") or {}
    if final.get("score_outputs") is not None:  # outputs right of all of them: an undecided one counts as wrong
        by = final.get("by_source") or {}
        out["live"]["score"] = {
            "score": final["score_outputs"], "at_elapsed_seconds": out["elapsed_seconds"], "at_epoch": epoch,
            "levels": sum(v["correct"] for v in by.values()), "outputs": row.get("test_outputs"),
            "decided": sum(v["outputs"] for k, v in by.items() if k != "fallback"),
            "verified_outputs": (by.get("verified") or {}).get("outputs", 0),
            "verified_right": (by.get("verified") or {}).get("correct", 0),
            "verified_tasks_share": final.get("verified_tasks")}
    return out


PROGRESS_SECONDS = 300
PROGRESS_URL_ENV, PROGRESS_TOKEN_ENV = "ARC2_PROGRESS_URL", "ARC2_PROGRESS_TOKEN"  # both set: each row is also POSTed


def progress_line(row):
    """One compact line of a live row: elapsed, play, phase, restarts, tokens, tok/s, right so far."""
    engine, score = row["live"]["engine"], row["live"].get("score") or {}
    rate = engine.get("tokens_per_second_30m") or engine.get("tokens_per_second_mean")
    right = f"{score['levels']}/{score['outputs']}" if score.get("outputs") else "-"
    return (f"ARC2_PROGRESS elapsed {row['elapsed_seconds'] / 60:.0f} min, play {row['play_seconds'] / 60:.0f} min, "
            f"{row['phase']}, restarts {row['restarts']}, tokens {engine['generated_tokens'] / 1e6:.2f}M, "
            f"{rate or 0:.0f} tok/s, right {right}, verified outputs {score.get('verified_outputs', '-')}")


def progress(out, started, path, every=PROGRESS_SECONDS):
    """Every ``every`` seconds, in a daemon thread: the live row (``live_row``) to ``path`` (whole or not at all),
    one line to stdout, and with ``PROGRESS_URL_ENV`` and ``PROGRESS_TOKEN_ENV`` set, the row POSTed there (a save
    run with internet; a competition run has none and only writes the file)."""
    import threading
    import urllib.request

    history, url, token = [], os.environ.get(PROGRESS_URL_ENV), os.environ.get(PROGRESS_TOKEN_ENV)

    def read(p):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}

    def tick():
        while True:
            time.sleep(every)
            try:
                now = time.time()
                row = live_row(read(out / "run.json"), read(out / "arena/live/arc2-submit.json"), now - started, now,
                               history)
                history.append((row["elapsed_seconds"], row["live"]["engine"]["generated_tokens"]))
                path.with_name(path.name + ".part").write_text(json.dumps(row))
                os.replace(path.with_name(path.name + ".part"), path)
                print(progress_line(row), flush=True)
                if url and token:
                    request = urllib.request.Request(url, data=json.dumps(row).encode(), method="POST", headers={
                        "Content-Type": "application/json", "Authorization": f"Bearer {token}"})
                    urllib.request.urlopen(request, timeout=20).close()
            except Exception as exc:  # noqa: BLE001 - progress never stops the play
                print(f"ARC2_PROGRESS error {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    threading.Thread(target=tick, daemon=True).start()


DRIVER_KEYS = ("drill_restart", "cache_save", "engine_watchdog", "prefetch_threads", "model")


def check_options(options):
    """Refuse option keys no code reads (19 Oct: PTH lost a play check to a study key its loader ignored): the
    server's (``l4_serving.l4_study``'s keywords), this driver's own and ``OPTIONS``'."""
    import inspect

    from . import l4_serving

    known = (set(inspect.signature(l4_serving.l4_study).parameters) - {"study"}) | set(DRIVER_KEYS) | set(OPTIONS)
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"options no code reads: {unknown}")
    return options


def cgroup_memory(root=Path("/sys/fs/cgroup")):
    """The container's memory limit and use in GiB (cgroup v2, else v1; {} without either). The page cache counts
    against the limit: v41's PLE copy on top of the prefetched checkpoint made the kernel evict unread shards."""
    for folder, limit_name, used_name in ((root, "memory.max", "memory.current"),
                                          (root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes")):
        try:
            limit, used = (folder / limit_name).read_text().strip(), int((folder / used_name).read_text())
        except (OSError, ValueError):
            continue
        out = {"limit_gib": None if limit == "max" or int(limit) >= 1 << 60 else round(int(limit) / 2**30, 1),
               "used_gib": round(used / 2**30, 1)}
        try:
            for line in (folder / "memory.stat").read_text().splitlines():
                key, value = (line.split() + ["", ""])[:2]
                if key in ("file", "anon", "shmem", "cache", "rss", "total_cache", "total_rss"):
                    out[key + "_gib"] = round(int(value) / 2**30, 1)
        except (OSError, ValueError):
            pass
        return out
    return {}


def sample_memory(record, started, every=30, count=80):
    """``cgroup_memory`` every ``every`` s, ``count`` times (the start), into ``record["memory"]`` (a daemon
    thread; the list exists before it starts, so a concurrent ``json.dumps`` of the record never sees a new key)."""
    import threading

    samples = record.setdefault("memory", [])

    def run():
        for _ in range(count):
            try:
                samples.append({"elapsed": round(time.time() - started), **cgroup_memory()})
            except Exception:  # noqa: BLE001 - a diagnostic
                return
            time.sleep(every)

    threading.Thread(target=run, daemon=True, name="arc2-memory").start()


def prefetch(bundle, input_root, threads, record, cap_gib=None):
    """Read the checkpoint's safetensors (the expert overlay's shards in place of the base's) in ``threads`` daemon
    threads, all of them unless ``cap_gib``, so the server's load finds them in the page cache (v8.5p kernel, 17 Oct).
    No cap by default (19 Oct, v39's root cause): Kaggle serves inputs over GCS FUSE, where vLLM turns its own
    prefetch off and an mmap load reads ~1 shard per 450-500 s, so a file left out stalls the start. Progress and the
    end in ``record["prefetch"]``; never raises into the run."""
    import threading

    started, state = time.time(), {"threads": threads, "files": 0, "gib": 0.0}
    record["prefetch"] = state
    try:
        server = json.loads((Path(bundle) / "configs/comparison.json").read_text())["server"]
        model = Path(os.environ.get("ARC_MODEL_DIR") or Path(input_root) /
                     "models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1")
        dataset = (server.get("expert_overlay") or {}).get("dataset")
        overlay = Path(input_root) / "datasets" / dataset if dataset else None
        shards = (set(json.loads((overlay / "overlay-manifest.json").read_text())["shards"])
                  if overlay and (overlay / "overlay-manifest.json").is_file() else set())
        files = [overlay / f.name if f.name in shards else f.resolve()
                 for f in sorted(model.iterdir()) if f.name.endswith(".safetensors")]
        state.update(of=len(files), overlay_shards=len(shards))
    except Exception as exc:  # noqa: BLE001 - a convenience
        state["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return
    queue, lock = iter(files), threading.Lock()

    def work():
        buffer = bytearray(16 << 20)
        while True:
            with lock:
                item = next(queue, None) if cap_gib is None or state["gib"] < cap_gib else None
                if item is None:
                    return
                state["files"] += 1
                state["gib"] = round(state["gib"] + item.stat().st_size / 2**30, 2)
            try:
                with open(item, "rb", buffering=0) as stream:
                    while stream.readinto(buffer):
                        pass
            except OSError as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"[:300]

    workers = [threading.Thread(target=work, daemon=True, name=f"arc2-prefetch-{i}") for i in range(threads)]
    for worker in workers:
        worker.start()

    def finish():
        for worker in workers:
            worker.join()
        state["seconds"] = round(time.time() - started, 1)
        print(json.dumps({"event": "arc2_prefetch", **state}), flush=True)

    threading.Thread(target=finish, daemon=True).start()


def run_logged(command, env, log_path, timeout):
    """Output to a file (a pipe to the notebook froze a preview); the code or "timeout"."""
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return "timeout"


def main():
    parser = argparse.ArgumentParser()
    for name in ("bundle", "working", "output", "input-root", "challenges", "submission"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--end-epoch", type=float, required=True)
    parser.add_argument("--solutions")
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--play-seconds", type=float)
    parser.add_argument("--options", help="JSON over OPTIONS")
    parser.add_argument("--resume-state")
    parser.add_argument("--progress", help="live progress file (default: progress.json beside the output folder)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    started = time.time()
    bundle, working, out = Path(args.bundle), Path(args.working), Path(args.output)
    from . import (
        l4_serving,  # standard library only, as everything this process imports: the notebook's Python
    )
    from .tasks import (
        write_json,  # has no package of ours (the agent environment is for the children)
    )

    challenges = json.loads(Path(args.challenges).read_text())
    solutions = json.loads(Path(args.solutions).read_text()) if args.solutions else {}
    kept = {}
    if args.resume_state:
        try:
            kept = {name: rows for name, rows in json.loads(Path(args.submission).read_text()).items()
                    if name in challenges and not problems({name: rows}, {name: challenges[name]})}
        except (OSError, ValueError, AttributeError):
            pass
    write_json(args.submission, {**fallback(challenges), **kept})
    check_options(json.loads(args.options or "{}"))  # after the fallback file: a refusal still leaves it whole
    out.mkdir(parents=True, exist_ok=True)
    working.mkdir(parents=True)
    record = {"kind": "arc2-kaggle", "started_epoch": started, "end_epoch": args.end_epoch, "stage": "host",
              "tasks_in_file": len(challenges), "rerun": bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))}
    if args.resume_state:
        record["resumed"] = {"state": Path(args.resume_state).is_file(), "submission_rows_kept": len(kept)}
        if record["resumed"]["state"]:
            shutil.copyfile(args.resume_state, working / "play-state.json")

    def stage(name):
        record.update(stage=name, elapsed=round(time.time() - started))
        (out / "run.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({"event": "arc2_kaggle", "stage": name, "elapsed": record["elapsed"]}), flush=True)

    def left():
        return args.end_epoch - time.time()

    progress(out, started, Path(args.progress) if args.progress else out.parent / "progress.json")
    sample_memory(record, started)

    threads = int(json.loads(args.options or "{}").get("prefetch_threads") or 0)
    if threads:  # the checkpoint's files into the page cache while the venv installs (v8.5p kernel, 17 Oct)
        prefetch(bundle, args.input_root, threads, record)
    ids = sorted(challenges)[:args.tasks] if args.tasks else sorted(challenges)
    # ``attempts[].options`` below are the server's and the probe's study (OPTIONS: its 48 training tasks are the
    # probe's first turns); the play's own tasks and wall are these and ``plays[].wall``.
    record.update(tasks_played=len(ids), play_seconds_asked=args.play_seconds)
    folder = working / "challenge"
    folder.mkdir()
    for name in ids:
        answers = solutions.get(name) or [None] * len(challenges[name]["test"])
        (folder / f"{name}.json").write_text(json.dumps({
            "train": challenges[name]["train"],
            "test": [{"input": t["input"], "output": answer} for t, answer in zip(challenges[name]["test"], answers)]}))
    env = {**os.environ, "PYTHONPATH": str(bundle / "src"), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX": "1", "MPLBACKEND": "Agg",
           "ARC_VALIDATION_OUTPUT": str(out)}
    serving, python, error, bootstrap = working / "serving", None, None, None

    def stop_server():
        program = ("from arc_infra.serving import load_recipe; import sys; "
                   "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")
        try:
            subprocess.run([python, "-c", program, str(serving), args.input_root], env=env, check=False, timeout=150,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass

    try:
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
            record["gpus"] = smi.stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired) as exc:
            record["gpus"] = [f"{type(exc).__name__}: {exc}"[:200]]
        stage("venv")
        bootstrap = runpy.run_path(str(bundle / "bootstrap.py"))
        python = bootstrap["install_agent_environment"](bundle, working / "agent-venv", env)
        base = json.loads((bundle / "configs/comparison.json").read_text())
        # The study sits in the bundle's configs, as in every scored L4 run: ``arena.serving`` takes the bundle
        # (the pruning mask, the draft head) to be two levels above the study file. The bundle is the notebook's
        # extracted copy, so it can be written. (Rehearsal 2 had the study in the working folder: no mask.)
        choice, study_path = {**OPTIONS, **json.loads(args.options or "{}")}, bundle / "configs/comparison-l4.json"
        record["attempts"], found = [], None
        watchdog = choice.pop("engine_watchdog", False)  # restart a dead engine inside the play (arena.engine_watchdog)
        choice.pop("prefetch_threads", None)
        drill = choice.pop("drill_restart", False)  # test only: the first start that came up is taken as failed
        save_cache = choice.pop("cache_save", False)  # a served start's caches into the evidence, for the bundle
        if (bundle / CACHE_FILE).is_file():
            begun = time.time()
            with tarfile.open(bundle / CACHE_FILE, "r:gz") as archive:
                names = archive.getnames()
                archive.extractall("/tmp", filter="data")
            record["cache_seeded"] = {"files": len(names), "seconds": round(time.time() - begun, 1)}

        def triton_entries():  # a seeded cache whose keys miss (another Triton, ptxas or arch) shows as new entries
            folder = CACHE_ROOTS[1] / "triton"
            return sum(1 for _ in folder.iterdir()) if folder.is_dir() else 0

        record["triton_entries_before"] = triton_entries()
        for number in (1, 2):
            study = l4_serving.study_for(base, choice)
            study_path.write_text(json.dumps(study, indent=1))
            stage(f"server-{number}")
            begun = time.time()
            code = run_logged([python, "-u", "-m", "arc2.l4_serving", "serve", "--study", str(study_path),
                               "--working", str(serving), "--input-root", args.input_root],
                              {**env, **study["trials"][0]["server_env"]}, out / f"serving-process-{number}.log",
                              min(SERVER_SECONDS, left() - STOP_SECONDS - MIN_PLAY_SECONDS))
            record["attempts"].append({"options": choice, "code": code, "seconds": round(time.time() - begun)})
            problem, following = f"the server start ended with {code}", l4_serving.start_fallback(choice)
            if code == 0 and drill and number == 1:
                problem = "restart drill: the first start is taken as failed"
            elif code == 0:
                stage(f"probe-{number}")
                probe_file = out / f"probe-{number}.json"
                run_logged([python, "-u", "-m", "arc2.l4_serving", "probe", "--study", str(study_path), "--source",
                            str(bundle), "--server-dir", str(serving), "--output", str(probe_file),
                            "--budget-seconds", str(PROBE_SECONDS)], env, out / f"probe-process-{number}.log",
                           PROBE_SECONDS + 560)
                found = json.loads(probe_file.read_text()) if probe_file.is_file() else None
                if found is None:  # the probe itself broke: the play still runs
                    break
                if found["health"] == 200 and found["passed"] >= 1:  # as scripts/modal_arc2_l4.py reads it
                    if found["passed"] < found["top"]:
                        choice = {**choice, "prefill_slots": found["passed"]}
                    cap = study["trials"][0]["arms"][0].get("cap") or study["completion_cap"]
                    entry = found["gate"]["real_loop_streams"][str(cap)]
                    fit = entry.get("second_turn_reserved", entry["second_turn"])
                    choice = {**choice, "slots": max(8, min(int(choice["streams"]), fit - 2))}
                    study = l4_serving.study_for(base, choice)
                    record["triton_entries_after_start"] = triton_entries()
                    if save_cache:
                        begun = time.time()
                        with tarfile.open(out / CACHE_FILE, "w:gz") as archive:
                            for root in CACHE_ROOTS:
                                for item in sorted(root.iterdir()) if root.is_dir() else []:
                                    if item.name != "huggingface":
                                        archive.add(item, arcname=str(item.relative_to("/tmp")))
                        record["cache_saved"] = {"bytes": (out / CACHE_FILE).stat().st_size,
                                                 "seconds": round(time.time() - begun, 1), "after_start": number}
                    break
                problem = f"the probe failed (health {found['health']}): {found['stopped']}"
                following = l4_serving.fallback(choice, found["passed"])
            report = l4_serving.failure(serving, record["stage"], problem, out / f"serving-process-{number}.log")
            (out / f"failure-{number}.json").write_text(json.dumps(report, indent=1))
            if not (drill and number == 1):  # the drill leaves the live server to free_cards, as a hung one is
                stop_server()
            if number == 2 or following is None or left() < SERVER_SECONDS + STOP_SECONDS + MIN_PLAY_SECONDS:
                raise RuntimeError(problem)
            record.setdefault("killed_after_stop", []).append(l4_serving.free_cards())
            if serving.exists():
                os.rename(serving, working / "serving-failed")
                log = working / "serving-failed/vllm-openai-server.log"  # the given-up start's stacks, if dumped
                if log.is_file():
                    (out / f"server-failed-{number}.log").write_text(
                        "\n".join(log.read_text(errors="replace").splitlines()[-20000:]) + "\n")
            choice = following
        record["probe"] = found and {key: found.get(key) for key in ("health", "passed", "top", "stopped", "seconds")}
        budget = args.play_seconds or float("inf")

        def watchdog_env(number):
            """The play's ARENA_ENGINE_WATCHDOG: the server's own start and a stop that frees the cards (vLLM
            processes only: the play's sandboxes run from the agent venv). ``test_kill_seconds`` only in play 1."""
            if not watchdog:
                return {}
            kill = watchdog.get("test_kill_seconds") if isinstance(watchdog, dict) and number == 1 else None
            stop = ("from arc_infra.serving import load_recipe; import sys; from arc2 import l4_serving; "
                    "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server(); "
                    "print(l4_serving.free_cards(marks=('vllm', 'pleoffload')))")
            return {"ARENA_ENGINE_WATCHDOG_WAIT": str(SERVER_SECONDS), "ARENA_ENGINE_WATCHDOG": json.dumps({
                "start": [python, "-u", "-m", "arc2.l4_serving", "serve", "--study", str(study_path), "--working",
                          str(serving), "--input-root", args.input_root],
                "stop": [python, "-c", stop, str(serving), args.input_root],
                "env": study["trials"][0]["server_env"], "start_timeout": SERVER_SECONDS,
                **({"test_kill_seconds": kill} if kill else {})})}

        (working / "trials").mkdir()
        record["plays"] = []
        while True:  # one play, and another after each death of the engine while the time is worth a start
            wall = min(left() - STOP_SECONDS, budget)
            if wall < 120:
                if record["plays"]:
                    break
                raise RuntimeError(f"no time left to play ({int(wall)} s)")
            number = len(record["plays"]) + 1
            trial = study["trials"][0]
            trial.update(name="submit", split=str(folder), tasks=ids, copies=args.copies, wall_seconds=int(wall),
                         marks={"end": 1e15}, submission=str(Path(args.submission).resolve()),
                         state=str(working / "play-state.json"), min_verified_tasks=0)
            trial.pop("curve_step", None)
            study_path.write_text(json.dumps(study, indent=1))
            (out / "comparison-l4.json").write_text(json.dumps(study, indent=1))
            record["choice"] = choice
            stage(f"play-{number}")
            folder_out, begun = working / f"trials/submit-{number}", time.time()
            code = run_logged([python, "-u", "-m", "arc2.l4_serving", "load", "--study", str(study_path), "--source",
                               str(bundle), "--output", str(folder_out), "--server-dir", str(serving)], {**env, **watchdog_env(number)},
                              out / f"trial-process-{number}.log", wall + 120)
            played = time.time() - begun
            budget -= played
            entry = {"code": code, "seconds": round(played), "wall": int(wall), "utilization": choice["utilization"]}
            record["plays"].append(entry)
            for name in ("report.json", "progress.json"):
                if (folder_out / name).is_file():
                    data = json.loads((folder_out / name).read_text())
                    entry.update({key: data.get(key) for key in (
                        "status", "play_seconds", "elapsed_seconds", "completion_tokens", "replies", "request_errors",
                        "server_tokens_per_second", "task_errors", "ended", "final")})
                    (out / f"play-{number}-{name}").write_text(json.dumps(data, indent=1))
                    break
            record["play"] = entry
            if played >= wall - EARLY_SECONDS or number > RESTARTS:
                break
            # Ended early: the engine died (every request fails, the tasks give up) or the trial's process did.
            # The next play takes up the tasks' programs, tokens and phases (``play-state.json``, a minute old at
            # most); open conversations are lost.
            entry["health"] = health(args.base_url)
            entry["server_errors"] = l4_serving.log_first(serving)[:4]
            if entry["health"] != 200:
                if min(left() - STOP_SECONDS, budget) < SERVER_SECONDS + MIN_PLAY_SECONDS:
                    break
                stage(f"restart-{number}")
                stop_server()
                try:  # the dead server's last words: the evidence copy takes the last server's folder only
                    with (serving / "vllm-openai-server.log").open("rb") as stream:
                        stream.seek(max(0, stream.seek(0, 2) - 3 * 2**20))
                        (out / f"serving-died-{number}.log").write_bytes(stream.read())
                    with (serving / "vllm-openai-server.log").open("rb") as stream:  # the cause: the engine's own
                        core = [line for line in stream if b"(EngineCore" in line and b" ERROR " in line]  # lines,
                    (out / f"serving-died-{number}-core.log").write_bytes(b"".join(core[:400]))  # before the API's
                except OSError:
                    pass
                os.rename(serving, working / f"serving-died-{number}")
                choice = l4_serving.fallback(choice, choice.get("prefill_slots") or 4) or choice  # 0.03 less a card
                study = l4_serving.study_for(base, choice)
                study_path.write_text(json.dumps(study, indent=1))
                begun = time.time()
                code = run_logged([python, "-u", "-m", "arc2.l4_serving", "serve", "--study", str(study_path),
                                   "--working", str(serving), "--input-root", args.input_root],
                                  {**env, **study["trials"][0]["server_env"]}, out / f"serving-restart-{number}.log",
                                  min(SERVER_SECONDS, left() - STOP_SECONDS - MIN_PLAY_SECONDS))
                record["attempts"].append({"options": choice, "code": code, "seconds": round(time.time() - begun),
                                           "after_play": number})
                if code != 0:
                    break
        stage("report")
        record["server_errors_after_ready"] = l4_serving.log_first(serving)
    except Exception as exc:  # noqa: BLE001 - the submission stands, whatever broke
        error = record["error"] = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        if python:
            stop_server()
        try:  # the trial's records and the server's logs, not its installations
            if bootstrap:
                bootstrap["collect_evidence"](working, out / "evidence")
        except Exception as exc:  # noqa: BLE001
            record["evidence_error"] = f"{type(exc).__name__}: {exc}"[:300]
        submitted = json.loads(Path(args.submission).read_text())
        bad = problems(submitted, challenges)
        if bad:  # never leave a file the scorer would refuse
            write_json(args.submission, {**fallback(challenges), **{k: v for k, v in submitted.items()
                                                                    if not problems({k: v}, {k: challenges[k]})}})
        own = fallback(challenges)
        record.update(submission_problems=bad[:5], seconds=round(time.time() - started),
                      answered=sum(submitted.get(name) != own[name] for name in challenges))
        stage("done" if error is None else "failed")
    raise SystemExit(0 if error is None else 1)


if __name__ == "__main__":
    main()
