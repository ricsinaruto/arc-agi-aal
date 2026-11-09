"""Bundle for an SFT adapter's reads (SFT arm, 16 Oct): rl1 mode ``final`` with ``resume_adapter`` (no trainer, the
server stays awake with the base bundle's KV pool) and ``paired``: whole games from level 1 of the held-out half's
families, adapter and base on the same seeds, each game up to ``BUDGET`` completion tokens.

usage (this worktree): python scripts/posttrain/build_sft_reads.py BASE_BUNDLE OUT ADAPTER NOTEBOOK_S POINTS GAMES
    ATTEMPTS BUDGET SEED   (ADAPTER: the adapter folder on the runs volume, e.g. /runs/validation/<t3>/.../sft-adapter;
    GAMES: comma-separated 4-letter families)
"""
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from arena import rl_loop  # noqa: E402
from build_arena import manifest  # noqa: E402

ARENA_FILES = ("rl_loop.py", "rl_single.py", "rl_sft.py", "rl_ladder.py", "level_bench.py", "restore.py")

if __name__ == "__main__":
    base, out, adapter, notebook, points = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    games, attempts, budget, seed = sys.argv[6].split(","), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(base, out)
    for name in ARENA_FILES:
        shutil.copyfile(ROOT / "src/arena" / name, out / "src/arena" / name)
    shutil.copyfile(points, out / "configs/sft_points.json")
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    rl = dict(study["rl_loop"])
    for key in ("mg", "transfer", "resume_state", "resume_round", "equality_source", "round0_from", "round0_play_from",
                "lvl_resume", "l2_record", "sft"):
        rl[key] = None
    rl.update(mode="final", resume_adapter=adapter, paired={"games": games, "attempts": attempts, "seed_base": seed},
              train_budget=budget, eval_budget=budget, dual=False, lvl=False, l2=False, f_small=False, opd=False)
    study["rl_loop"] = rl_loop.settings(rl)
    study["level_bench"] = {**study["level_bench"], "start_points": "configs/sft_points.json", "games": games}
    study["trials"] = [{**study["trials"][0], "wall_seconds": notebook}]
    study["wall_seconds"] = study["notebook_seconds"] = notebook
    study["name"] = out.name
    study["sft_build"] = {"source": "scripts/posttrain/build_sft_reads.py", "base_bundle": base.name, "adapter": adapter,
                          "games": games, "attempts": attempts, "budget": budget, "seed": seed}
    path.write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(json.dumps({"output": str(out), "manifest": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()[:8],
                      "notebook_seconds": notebook, **study["sft_build"]}))
