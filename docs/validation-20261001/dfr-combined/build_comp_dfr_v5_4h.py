"""dfr-combined v5, four hours on Modal ("implement a v5 and run it for 4 hours with better matching to
his harness"): v4 + ``dfr_loop`` (the drain system line, the 2,048-token yield, no world model, his action feedback,
the inventory's lazy-animation fix) + his 12,288-token reply cap (``completion_cap``). Everything else is
build_comp_dfr_v3_4h.py / v4. Run from the worktree root.

argv: streams floor output [--dependencies]
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import build, kaggle_files  # noqa: E402

from arena.campaign import components  # noqa: E402

streams, floor, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
NAME = "arena-flash-competition-dfr-combined-v5-4h"
RUNTIME = "richardcsaky/arc-arena-hero-v2-dfr-combined-v5-4h-runtime"
UTILIZATION = "0.92"
# v1's int8-unit projection (1,839) minus the headroom: 0.04 x 101,974,081,536 B / 21,196,800 B per int8 block = 192.
SERVED_INT8 = 1839 - 192

hero = components("comp-prune25")
hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=streams, density_blocks=floor,
            graph_capture=-(-(streams + 4) // 8) * 8, backlog_fix=False, budget_policy="priority",
            dfr_harness=True, drain_half=True, dfr_loop=True, completion_cap=12288)
output, study = build(NAME, output=Path(out), dependencies="--dependencies" in sys.argv, paired_30m=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero, hero_v2_served_blocks=SERVED_INT8,
                      exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], opening_burst_requests=3,
                      competition=True, preview_seconds=14400, preview_instances=110, sanity_trial=False,
                      no_preview=False, gpu_memory_utilization=UTILIZATION)
digest = kaggle_files(output, study, smoke=False, runtime_dataset=RUNTIME)
s = study["server"]
print(json.dumps({"output": str(output), "sha256": digest, "streams": study["active_limit"],
                  "window": study["windows"][0], "max_num_seqs": s["max_num_seqs"],
                  "graphs": s.get("max_cudagraph_capture_size"), "utilization": s["gpu_memory_utilization"],
                  "server_env": study["trials"][0]["server_env"], "completion_cap": study["completion_cap"],
                  "dfr_loop": study.get("dfr_loop"),
                  "density": study.get("focus_oversubscription", {}).get("gate_blocks_per_stream")}))
