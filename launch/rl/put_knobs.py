"""Live knobs for an RL run, loud on failure (v6h 24 Oct: the launcher's put read a missing file, its error went to
/dev/null and the run played 30 min on bundle defaults).

  put_knobs.py put RUN KNOBS.json      put, read back, compare; exit 1 on any failure
  put_knobs.py verify RUN KNOBS.json   after launch: the run's knobs_file feed row must equal the file within
                                       --within seconds of v7_start; prints ALERT and exits 1 otherwise
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODAL = "./.venv/bin/modal"
VOLUME = "arc-prolong-runs"
KNOBS = "validation/{run}/arena/live/knobs.json"
FEED = "validation/{run}/arena/trials/rl-loop/rl/feed.jsonl"


def out_of(run):
    """The run's current attempt folder: <run>, or <run>-attempt<N> after a re-reserve (v6k 25 Oct: knobs put to
    validation/<run>/ went to an empty folder), from the main checkout's records (`records.py out`). An attempt name
    or an unknown run is used as given."""
    try:
        r = subprocess.run([str(Path(MODAL).parent / "python"), str(Path(MODAL).parents[2] / "tools/ops/records.py"), "out", run],
                           capture_output=True, text=True)
    except OSError:  # no records here: the name as given
        return run
    out = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    return out or run


def started(run):
    """True when the run's record says its current attempt is up (state running); a reserved, not yet launched
    attempt (launchers put knobs before `modal run`) or an unknown run is False."""
    try:
        r = subprocess.run([str(Path(MODAL).parent / "python"), str(Path(MODAL).parents[2] / "tools/ops/records.py"), "show",
                            run.split("-attempt")[0]], capture_output=True, text=True)
        return (json.loads(r.stdout) or {}).get("state") == "running"
    except (OSError, ValueError):  # no records here (CI container): not known to be started
        return False


def live_exists(run):
    return subprocess.run([MODAL, "volume", "ls", VOLUME, f"validation/{run}/arena"], capture_output=True).returncode == 0


def volume_get(path, dest):
    return subprocess.run([MODAL, "volume", "get", "--force", VOLUME, path, str(dest)], capture_output=True).returncode == 0


def put(run, knobs, sh=subprocess.run, get=volume_get, exists=live_exists, is_started=started):
    want = json.loads(Path(knobs).read_text())  # a missing or bad file fails here, loudly
    # A started run whose folder has no arena/ would never read the put (v6k: the attempt-2 folder); before launch
    # the folder does not exist yet and the resolved name is the one the launch will use (v6l, 25 Oct).
    if is_started(run) and not exists(run):
        raise SystemExit(f"KNOBS TARGET MISSING {run}: no validation/{run}/arena on {VOLUME} (wrong attempt folder? "
                         f"`records.py out RUN`)")
    r = sh([MODAL, "volume", "put", "-f", VOLUME, str(knobs), KNOBS.format(run=run)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"KNOBS PUT FAILED {run}: {(r.stderr or r.stdout)[-300:]}")
    with tempfile.TemporaryDirectory() as d:
        back = Path(d) / "knobs.json"
        if not get(KNOBS.format(run=run), back) or json.loads(back.read_text()) != want:
            raise SystemExit(f"KNOBS READ-BACK MISMATCH {run}")
    print(f"knobs put {run}: {want}")
    return want


def check(rows, want, within, now):
    """None when the run read ``want``; else why not (yet). ``rows``: feed rows."""
    start = next((r for r in rows if r.get("kind") == "v7_start"), None)
    files = [r for r in rows if r.get("kind") == "knobs_file"]
    if files and files[-1].get("keys") == want:
        return None
    if start is None or now - float(start["epoch"]) < within:
        return "pending"
    got = files[-1] if files else None
    return f"run read {'nothing' if got is None else ('no file' if got.get('missing') else got.get('keys'))}"


def verify(run, knobs, within=120.0, poll=20.0, timeout=3600.0, get=volume_get, clock=time.time, sleep=time.sleep):
    want = json.loads(Path(knobs).read_text())
    began = clock()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "feed.jsonl"
        while clock() - began < timeout:
            rows = []
            if get(FEED.format(run=run), f):
                for line in f.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
            why = check(rows, want, within, clock())
            if why is None:
                print(f"knobs applied {run}")
                return 0
            if why != "pending":
                print(f"ALERT {run} [knobs] applied knobs != requested {want}: {why}")
                return 1
            sleep(poll)
    print(f"ALERT {run} [knobs] no v7_start within {timeout:.0f} s")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["put", "verify"])
    ap.add_argument("run")
    ap.add_argument("knobs")
    ap.add_argument("--within", type=float, default=120.0)
    a = ap.parse_args()
    out = out_of(a.run)
    if out != a.run:
        print(f"{a.run}: current attempt folder {out}")
        a.run = out
    if a.op == "put":
        put(a.run, a.knobs)
        return 0
    return verify(a.run, a.knobs, within=a.within)


if __name__ == "__main__":
    sys.exit(main())
