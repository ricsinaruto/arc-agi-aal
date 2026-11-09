"""Competition build: the Run B fixed-2x policy (comp-prune25 arm: fixed-2x first-clear grants,
progress-fruitless backlog, no plans, no level allowance, newer cache improvements) with prune12 + int3 g16 experts on
Humming, a 130,816 window, and 50 streams on a 1,700-block floor; 300 s preview. Run from the worktree root.

argv: streams floor output [--dependencies]
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from arena.campaign import components  # noqa: E402
from build_arena import build, kaggle_files  # noqa: E402

streams, floor, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
NAME = "arena-flash-competition-fixed2x-p12-int3-v1"
RUNTIME = "richardcsaky/arc-arena-hero-v2-fixed2x-p12-int3-runtime"
# int8-unit projection of the served pool: the Kaggle-measured p37-int3 pool (2,342 int4 blocks at 64 sequences / 64
# graphs) minus the 6,144 further kept experts at int3 (6,144 x 2,150,400 B / 22,278,144 B = 593) = 1,749 int4 blocks.
SERVED_INT8 = 1839

hero = components("comp-prune25")
hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=streams, density_blocks=floor,
            graph_capture=-(-(streams + 4) // 8) * 8)
output, study = build(NAME, output=Path(out), dependencies="--dependencies" in sys.argv, paired_30m=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero, hero_v2_served_blocks=SERVED_INT8,
                      exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], opening_burst_requests=3,
                      competition=True, preview_seconds=300, preview_instances=110, sanity_trial=False)
digest = kaggle_files(output, study, smoke=False, runtime_dataset=RUNTIME)
s = study["server"]
print(json.dumps({"output": str(output), "sha256": digest, "streams": study["active_limit"],
                  "window": study["windows"][0], "max_num_seqs": s["max_num_seqs"],
                  "graphs": s.get("max_cudagraph_capture_size"), "kernels": s["serving_kernels"],
                  "overlay": s.get("expert_overlay"), "prune": s.get("expert_prune"),
                  "requested": study["hero_v2"]["requested"], "model_datasets": study.get("model_datasets")}))
