"""heal_watch.py RUN APP_ID: live watcher of a heal_round (every 5 min): reads gpu.jsonl and train/events.jsonl from the
runs volume; stops the app (verified) on a non-finite step, KD on trained positions not 25% below steps 1-2 by step 10,
no new telemetry for 15 min, or step util (between optimizer steps, after step 2) < 85% on 3 reads in a row. Prints one line per read."""
import json, subprocess, sys, time
from pathlib import Path
MV = "./.venv/bin/modal"
run, app = sys.argv[1], sys.argv[2]
D = Path(f"~/arc-sft-data/mon/{run}"); D.mkdir(parents=True, exist_ok=True)
def get(rel):
    out = D / rel.replace("/", "_")
    ok = subprocess.run([MV, "volume", "get", "--force", "arc-prolong-runs", f"/validation/{run}/{rel}", str(out)], capture_output=True).returncode == 0
    return [json.loads(l) for l in open(out) if l.strip()] if ok else []
def stop(why):
    print(time.strftime("%H:%M", time.gmtime()), "STOP", why, flush=True)
    subprocess.run([MV, "app", "stop", "-y", app], capture_output=True)
    for _ in range(20):
        live = subprocess.run([MV, "container", "list", "--json"], capture_output=True, text=True).stdout
        if app not in live:
            print("verified stopped", flush=True); sys.exit(1)
        time.sleep(15)
    print("WARNING: container still listed", flush=True); sys.exit(2)
last_len, last_change, low = 0, time.time(), 0
while True:
    ident = D / "id.json"
    if subprocess.run([MV, "volume", "get", "--force", "arc-prolong-runs", f"/validation/{run}/modal-identity.json", str(ident)], capture_output=True).returncode == 0:
        print(time.strftime("%H:%M", time.gmtime()), "ended:", ident.read_text()[:400], flush=True); sys.exit(0)
    gpu, ev = get("gpu.jsonl"), get("train/events.jsonl")
    n = len(gpu) + len(ev)
    if n > last_len:
        last_len, last_change = n, time.time()
    steps = [e for e in ev if e.get("event") == "step"]
    # util during optimizer steps only: rows between the 2nd step event and the latest one (load, routing, probes
    # and the final save excluded; stop if step util stays under ~85% after warm-up)
    recent = []
    if len(steps) >= 3:
        lo, hi = steps[1]["t"], steps[-1]["t"]
        recent = [g for g in gpu if g.get("epoch") and lo <= g["epoch"] <= hi][-60:]
    util = round(sum(sum(g["util"]) / len(g["util"]) for g in recent) / len(recent), 1) if recent else None
    mem = max((max(g["mem_gib"]) for g in recent), default=None)
    phase = gpu[-1].get("phase") if gpu else None
    s = steps[-1] if steps else {}
    print(time.strftime("%H:%M", time.gmtime()), f"phase {phase} step {s.get('step')} kd {s.get('kd')} reply {s.get('kd_reply')} "
          f"mtp {s.get('kd_mtp')} gn {s.get('grad_norm')} util5m {util} mem {mem}", flush=True)
    if any(not e.get("finite", True) for e in steps):
        stop("non-finite step")
    if len(steps) >= 10:
        base = sum(e["kd"] for e in steps[:2]) / 2
        if sum(e["kd"] for e in steps[8:10]) / 2 > 0.75 * base:
            stop(f"KD not 25% below steps 1-2 by step 10 ({base:.3f})")
    if time.time() - last_change > 900:
        stop("no new telemetry for 15 min")
    low = low + 1 if (util is not None and util < 80) else 0
    if low >= 3:
        stop(f"step util < 80% on 3 reads ({util})")
    time.sleep(300)
