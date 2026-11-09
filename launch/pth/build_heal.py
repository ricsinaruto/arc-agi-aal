"""build_heal.py NAME: heal_round bundle = dist/sft-heal-v4 with the current heal/teacher code of the sft-heal worktree."""
import hashlib, json, shutil, sys
from pathlib import Path
WT = Path("./.claude/worktrees/sft-heal")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_gate  # noqa: E402
ci_gate.require(WT, "pth-heal")  # a red CI blocks the build
name = sys.argv[1]
src, out = WT / "dist/sft-heal-v4", WT / "dist" / name
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)
for f in ("heal_kd.py", "rl1_trainer.py", "rl1_sparse.py", "flash_qlora_job.py", "rl_job.py", "flash_config.json",
          "heal_delta.py", "heal_stream.py", "heal_prune-t75-excl5-v1.json", "fla_pin.py", "rl1_hyper.py", "rl1_sparse2.py"):  # stream OPD (22 Oct); fla pin (23 Oct)
    own = WT / "scripts/posttrain/pth" / f  # trunk 24 Oct: PTH's copies of the forked trainer files
    if own.exists() or (WT / "scripts/posttrain" / f).exists():
        shutil.copyfile(own if own.exists() else WT / "scripts/posttrain" / f, out / "rl1" / f)
(out / "rl1" / "trainer_profile").write_text("pth\n")  # the shared rl1_trainer runs PTH's profile (trunk dedup)
shutil.copyfile(WT / "src/posttrain/rl_rollouts.py", out / "rl1/rl_rollouts.py")
for f in ("src/arena/rl_loop.py", "src/arena/rl_sft.py", "src/arena/run.py", "src/posttrain/rl_rollouts.py"):
    if (out / f).exists() or f.startswith("src/arena/rl_loop"):
        shutil.copyfile(WT / f, out / f)
shutil.copytree(WT / "launch/pth", out / "launch/pth", dirs_exist_ok=True,  # the launchers are part of the bundle id
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()
ident = json.loads((out / "MANIFEST.json").read_text())
ident["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
                  if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
print(json.dumps({"bundle": str(out), "files": len(ident["files"])}))
