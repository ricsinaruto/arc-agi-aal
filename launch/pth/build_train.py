"""Build the replan's training bundle: build_train.py CORPUS_DIR WINDOW [NAME]
CORPUS_DIR holds train/ and val/ from select_cheapest.py. Base: dist/sft-util-check-v3 (train-only, no server)."""
import hashlib, json, os, shutil, sys
from pathlib import Path
WT = Path("./.claude/worktrees/sft-heal")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_gate  # noqa: E402
ci_gate.require(WT, "pth-train")  # a red CI blocks the build
corpus, window = Path(sys.argv[1]), int(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else "sft-train-re86c-v1"
src, out = WT / "dist/sft-util-check-v3", WT / "dist" / name
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)
for f in ("src/arena/rl_sft.py", "src/arena/rl_loop.py", "src/arena/run.py"):
    shutil.copyfile(WT / f, out / f)
shutil.copyfile(WT / "scripts/posttrain/rl1_trainer.py", out / "rl1/rl1_trainer.py")
shutil.copyfile(WT / "scripts/arena/bootstrap.py", out / "bootstrap_arena.py")
for split in ("train", "val"):
    shutil.copytree(corpus / split, out / f"sft/{split}-re86c")
p = out / "configs/comparison.json"
c = json.loads(p.read_text())
c["notebook_seconds"] = c["wall_seconds"] = 3300
c["trials"][0]["wall_seconds"] = 3300
s = c["rl_loop"]["sft"]
s.update(corpus="sft/train-re86c", val_corpus="sft/val-re86c", val_patience=2, window=window, passes=3, probe_every=int(os.environ.get('PROBE_EVERY', '4')),
         probe_base=True, fit_probes=0, probe_count=2, save_every=100000, save_passes=False, schedule="wsd", warmup=0.03, stable=0.8,
         lr=float(os.environ.get('LR', '1.5e-4')), kl_stop=0.15, resume=None, fit_gate=None, batch_windows=int(os.environ.get('BATCH', '8')))
s["gate"].update(train_only=True, drift={"dir": "/runs/validation/sft-heal-speed-v4/general", "count": 16}, play_reserve=300)
p.write_text(json.dumps(c, indent=2) + "\n")
sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()
ident = json.loads((out / "MANIFEST.json").read_text())
ident["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
                  if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
n = {k: len(list((out / f"sft/{k}-re86c").glob("*/meta.json"))) for k in ("train", "val")}
print(json.dumps({"bundle": str(out), "segments": n, "window": window, "probe_corpus": s["probe_corpus"]}))
