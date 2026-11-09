"""Launch queued Nebius runs in order as machines free (the limit: 8 RTX PRO 6000 at once, 24 h long runs (run id contains "-24h-") exempt; the account's address quota is 10). Usage: chain.py <queue.json> [start index]
queue.json: [[worktree name under .claude/worktrees, test kind, bundle dir, run id, planned minutes incl. boot, extra launcher args...], ...]
Each run needs its row in docs/modal-queue.json first; its start and end times are stamped at launch. Skips while <scratch>/chain.hold exists.
The queue file is re-read every minute, so rows can be added or reordered while it runs; tried run ids are kept in <queue.json>.done."""
import json, subprocess, time, datetime, fcntl, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import paths
ROOT = paths.ROOT_PATH; HERE = pathlib.Path(__file__).parent; W = ROOT / ".claude/worktrees"; QUOTA = 20; SHORT = 16
Q = json.loads(pathlib.Path(sys.argv[1]).read_text()); i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
def used():
    try: r = subprocess.run([str(ROOT / ".venv/bin/python"), str(paths.NEBIUS_SCRIPTS / "nebius_validation.py"), "list"], capture_output=True, text=True, timeout=300,
                            env={**os.environ, "ARC_NEBIUS_CONFIG": str(paths.NEBIUS_CONFIG)})
    except subprocess.TimeoutExpired: return 99, 99
    try: runs = [m.get("run") or "" for m in json.loads(r.stdout)] if r.returncode == 0 else None
    except ValueError: runs = None
    return (len(runs), sum("-24h-" not in x and not x.startswith("hx8-") for x in runs)) if runs is not None else (99, 99)  # total, short runs; hx8- is a collaborator's, outside the 8
def stamp(run, minutes):
    now = datetime.datetime.now(datetime.timezone.utc); end = now + datetime.timedelta(minutes=minutes)
    with open(ROOT / "docs/modal-queue.json", "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX); q = json.load(f)
        for e in q["live"]:
            if e["run"] == run: e["start"], e["end"] = f"{now:%H:%M}", f"{end:%H:%M}"
        f.seek(0); f.truncate(); json.dump(q, f, indent=1)
DONE = pathlib.Path(sys.argv[1] + ".done"); done = set(json.loads(DONE.read_text())) if DONE.exists() else {r[3] for r in Q[:i]}
for _ in range(1500):
    try: Q = json.loads(pathlib.Path(sys.argv[1]).read_text())  # re-read: rows may be added or reordered while it runs
    except Exception: pass
    todo = [r for r in Q if r[3] not in done]
    if not todo: break
    total, short = used(); long_run = "-24h-" in todo[0][3]
    if not (paths.SCRATCH / "chain.hold").exists() and total < QUOTA and (long_run or short < SHORT):
        wt, test, bundle, run, minutes, *extra = todo[0]
        r = subprocess.run([str(HERE / "nlaunch.sh"), str(W / wt / "scripts/modal_validation.py"), test, bundle, run, *extra], capture_output=True, text=True, timeout=900)
        print(f"{datetime.datetime.now(datetime.timezone.utc):%H:%M}", r.stdout.strip()[:220], flush=True)
        if "exit=0" in r.stdout: stamp(run, minutes)
        elif "spend cap" in r.stdout: time.sleep(600); continue  # the cap refused it: keep the row, try again later
        done.add(run); DONE.write_text(json.dumps(sorted(done))); time.sleep(90); continue
    time.sleep(60)
print("chain done:", len(done), "launched or tried", flush=True)
