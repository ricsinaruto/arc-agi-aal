"""Eval bundles for the replan: build_eval.py ADAPTER_PATH [TAG]
Fresh starts from the 24-game eval recording, 10 starts x 2 tries per level and arm, seeds 9900+level.
Jobs: L1 and L2 with both arms; L3 and L4 one arm per job (worst case at the cap fits the wall at 20 streams)."""
import hashlib, json, shutil, sys
from pathlib import Path
WT = Path("./.claude/worktrees/sft-heal")
adapter = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "v1"
REC = "/runs/validation/sft-heldout-evalrec-re86-v1/arena/trials/rl-loop/rl/gate-rec-re86"
CAPS = {"1": 22000, "2": 37000, "3": 38000, "4": 66000, "5": 70000}  # L5: held-out level (3x the v3 L5 base median 23.3K)
JOBS = [("1", "both"), ("2", "both"), ("3", "base"), ("3", "sft"), ("4", "base"), ("4", "sft"), ("5", "base"), ("5", "sft")]
if len(sys.argv) > 3:  # e.g. "34:both,5:both": several levels per job (redesign, fewer B300 jobs)
    JOBS = [tuple(x.split(":")) for x in sys.argv[3].split(",")]  # LEVELS:ARM[:STARTS]
out_names = []
for lv, arm, *more in JOBS:
    name = f"sft-eval-re86c-L{lv}-{arm}-{tag}"
    src, out = WT / "dist/sft-collect-re86-L1", WT / "dist" / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    for f in ("src/arena/rl_sft.py", "src/arena/rl_loop.py", "src/arena/run.py", "src/arena/level_bench.py"):
        shutil.copyfile(WT / f, out / f)
    shutil.copyfile(WT / "scripts/arena/bootstrap.py", out / "bootstrap_arena.py")
    p = out / "configs/comparison.json"
    c = json.loads(p.read_text())
    c["notebook_seconds"] = c["wall_seconds"] = 3300
    c["trials"][0]["wall_seconds"] = 3300
    g = c["rl_loop"]["sft"]["gate"]
    g.update(levels={x: CAPS[x] for x in lv}, control="none", starts=int(more[0]) if more else 10, tries=2, tries_per_start=2, seed=9900 + int(lv[0]), seeds={x: 9900 + int(x) for x in lv},
             concurrency=None, pin_restore=True, skip_base=(arm == "sft"),
             reuse={"recording": REC, "adapters": {} if arm == "base" else {"sft": adapter}})
    p.write_text(json.dumps(c, indent=2) + "\n")
    sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
                      if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
    out_names.append(name)
print(json.dumps(out_names))
