"""Four-hour Modal variant ("do a 4-hour run of our current v3 version too"). Competition build: dfr-combined v3 = v2 with the append-only vision history (extra images stubbed only at drains; v2 stubbed the previous turn on every request and lost the prefix cache on 37% of new-observation requests). v2: The v1 bundle (40 x 130,816, priority, drain_half, dfr_harness,
one-hour 110-clone preview) died on Modal ~70 s into gameplay on a CUDA OOM after the first 640 px vision-encoder
requests (KV pool profiled at the fork's 0.96). User: "reserve GPU headroom": the server profiles at 0.92 (about 4 GiB
outside the pool), the pool shrinks from the measured 1,851 to ~1,668 blocks, so 38 streams (the same 42-block mean
request at drain-half's ~85K) on a 1,600-block floor; the projected int8 pool drops by the same 4.08 GB (192 int8 blocks).
Everything else is v1. Kaggle start-up limits unchanged (user). Run from the worktree root.

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
NAME = "arena-flash-competition-dfr-combined-v3-4h"
RUNTIME = "richardcsaky/arc-arena-hero-v2-dfr-combined-v3-4h-runtime"
UTILIZATION = "0.92"
# v1's int8-unit projection (1,839) minus the headroom: 0.04 x 101,974,081,536 B / 21,196,800 B per int8 block = 192.
SERVED_INT8 = 1839 - 192

hero = components("comp-prune25")
hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=streams, density_blocks=floor,
            graph_capture=-(-(streams + 4) // 8) * 8, backlog_fix=False, budget_policy="priority",
            dfr_harness=True, drain_half=True)
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
                  "server_env": study["trials"][0]["server_env"],
                  "density": study.get("focus_oversubscription", {}).get("gate_blocks_per_stream")}))
