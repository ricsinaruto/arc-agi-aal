"""Watch a running Modal arena run's live progress (hero-v2, "Check intermediate score every 30
minutes").

Every ``--interval`` seconds (default 1,800) it reads ``validation/<RUN_ID>/arena/live/progress.json`` from the runs
volume (written every 60 s by the bootstrap's live-progress thread, committed every 120 s by
scripts/modal_validation.py; arena.live_progress) and prints one summary: elapsed, phase, score (read from the open
scorecard every 5 minutes), levels, the games with levels, completion tokens, generation throughput, running /
waiting requests, KV use and preemptions. Every ``--poll`` seconds (default 60) it checks the run's function call (the
id in the launch record) and exits, after one last summary, when the call has finished or failed. With ``--tail`` it
also prints the last lines of the server log (``live/server-tail.log``), e.g. a stall dump.

Usage:
    uv run --extra modal python scripts/arena/watch_live.py RUN_ID [--interval 1800] [--poll 60] [--once] [--tail]

It never stops, cancels or relaunches anything.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[2]).resolve()
VOLUME = "arc-prolong-runs"


def stamp():
    return datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S")


def launch_record(run_id, root=ROOT):
    matches = sorted(root.glob(f"docs/validation-*/modal-{run_id}.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one launch record docs/validation-*/modal-{run_id}.json under {root}, "
                         f"found {len(matches)}")
    record = json.loads(matches[0].read_text())
    if not record.get("function_call_id"):
        raise SystemExit(f"{matches[0]}: no function_call_id (not submitted)")
    return matches[0], record


def read_volume(volume, path):
    try:
        return b"".join(volume.read_file(path))
    except Exception as exc:  # noqa: BLE001 - not written yet, or a transient error
        print(stamp(), "no", path, f"({type(exc).__name__})", flush=True)
        return None


def summary(progress):
    live = progress.get("live") or {}
    score = live.get("score") or {}
    engine = live.get("engine") or {}
    value = score.get("score")
    lines = [
        f"elapsed {progress.get('elapsed_seconds')} s  phase {progress.get('phase')}  "
        # The mean of the scorecard's per-game scores (0-100 each) is already the arena's percent score.
        f"score {'-' if value is None else f'{value:.3f}%'} (at {score.get('at_elapsed_seconds')} s)  "
        f"levels {progress.get('levels')} on {live.get('games_with_levels')} games  "
        f"active {len(progress.get('active') or [])}  parked {progress.get('parked')}",
        f"completion tokens {live.get('completion_tokens')}  tok/s recent {engine.get('tokens_per_second_recent')} "
        f"mean {engine.get('tokens_per_second_mean')}  running {engine.get('running')} waiting "
        f"{engine.get('waiting')}  kv {engine.get('kv_cache_fraction')}  preemptions {engine.get('preemptions')}",
        "levels by game: " + ", ".join(f"{k[:9]} {v}" for k, v in list((live.get("levels_by_game") or {}).items())[:12]),
    ]
    if live.get("score_error"):
        lines.append(f"score error: {live['score_error']}")
    return "\n".join(lines)


def show(volume, run_id, *, tail=False):
    raw = read_volume(volume, f"validation/{run_id}/arena/live/progress.json")
    if raw is not None:
        try:
            progress = json.loads(raw)
            print(stamp(), f"[{run_id}]", "\n" + summary(progress), flush=True)
        except ValueError:
            print(stamp(), "progress.json not parseable yet", flush=True)
    if tail:
        text = read_volume(volume, f"validation/{run_id}/arena/live/server-tail.log")
        if text:
            print("\n".join(text.decode(errors="replace").splitlines()[-40:]), flush=True)


def call_state(call):
    """``running``, ``done`` or ``error: ...``; transient transport errors count as running."""
    import socket
    from modal.exception import ConnectionError as ModalConnectionError
    from modal.exception import FunctionTimeoutError

    try:
        call.get(timeout=0)
        return "done"
    except FunctionTimeoutError as exc:
        # A remote execution limit is terminal, unlike our nonblocking polling timeout.
        return f"error: {type(exc).__name__}: {str(exc)[:300]}"
    except TimeoutError:
        return "running"
    except (ModalConnectionError, ConnectionError, socket.gaierror):
        return "running"
    except Exception as exc:  # noqa: BLE001 - the remote exception type is not importable locally
        text = f"{type(exc).__name__}: {str(exc)[:300]}"
        return f"error: {text}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--interval", type=int, default=1800, help="seconds between summaries")
    parser.add_argument("--poll", type=int, default=60, help="seconds between run-state checks")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--tail", action="store_true", help="also print the server log's last lines")
    args = parser.parse_args(argv)
    path, record = launch_record(args.run_id)
    import modal

    volume = modal.Volume.from_name(VOLUME)
    call = modal.FunctionCall.from_id(record["function_call_id"])
    print(stamp(), "watching", args.run_id, record["function_call_id"], "record", path, flush=True)
    last_show = 0.0
    while True:
        state = call_state(call)
        if time.monotonic() - last_show >= args.interval or state != "running" or args.once:
            show(volume, args.run_id, tail=args.tail)
            last_show = time.monotonic()
        if state != "running":
            print(stamp(), "run ended:", state, flush=True)
            return 0 if state == "done" else 1
        if args.once:
            print(stamp(), "state:", state, flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
