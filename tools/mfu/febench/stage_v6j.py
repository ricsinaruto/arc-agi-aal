"""Stage long multi-image v6j/v6j3/v6k tries (rollout.jsonl) as a febench driver corpus: OUT/v6j-<run>-<set>/<attempt>.jsonl.
Keeps tries with >= MIN_IMG images and >= MIN_ROWS rows. usage: stage.py OUT"""
import concurrent.futures as cf, json, subprocess, sys
from pathlib import Path
M = "./.venv/bin/modal"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MIN_IMG, MIN_ROWS = 15, 15
def ls(p):
    r = subprocess.run([M, "volume", "ls", "arc-prolong-runs", p], capture_output=True, text=True)
    return [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("validation/")]
jobs = []
for run in ("rl1-loop-hero-v6j", "rl1-loop-hero-v6j3", "rl1-loop-hero-v6k-attempt2"):
    for s in ls(f"validation/{run}/partial/v7"):
        if "/g0" not in s:
            continue
        for a in ls(f"{s}/attempts"):
            jobs.append((run.split("-hero-")[1], s.rsplit("/", 1)[1], a))
print("attempts", len(jobs), flush=True)
def get(job):
    run, sset, a = job
    d = OUT / f"v6j-{run}-{sset}"; d.mkdir(exist_ok=True)
    f = d / (a.rsplit("/", 1)[1] + ".jsonl")
    subprocess.run([M, "volume", "get", "--force", "arc-prolong-runs", f"{a}/rollout.jsonl", str(f)], capture_output=True)
    if not f.is_file():
        return None
    s = f.read_text(errors="replace"); rows = s.count("\n"); img = s.count("data:image/png")
    if img < MIN_IMG or rows < MIN_ROWS:
        f.unlink(); return None
    return (str(f.relative_to(OUT)), rows, img, len(s))
with cf.ThreadPoolExecutor(12) as ex:
    kept = [k for k in ex.map(get, jobs) if k]
for d in OUT.iterdir():
    if d.is_dir() and not any(d.iterdir()):
        d.rmdir()
json.dump(kept, open(OUT / "kept.json", "w"), indent=0)
print("kept", len(kept), "images", sum(k[2] for k in kept), "MB", round(sum(k[3] for k in kept) / 1e6), flush=True)
