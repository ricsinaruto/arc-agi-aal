"""Long-context test ("rerun with 192k context (streams scaled accordingly), and compaction hands over the slot
(similar to drain). 4 hours. my aim is to test whether long context helps harder games. do not add the throughput arm's
improvements so that it's a like for like comparison"). dfr-combined v4 code (v3 gameplay + the deadline-drain fix) at
window 196,608 with the model-written summary compaction in place of drain_half (prompt budget 179,712). Compaction is
already a priority handover point (GameRecord.note_compaction -> on_context_eviction, the same path as a drain); the
256K run never compacted, so it never handed over. Streams sized so every stream can hold a full window at once (the
256K lesson: streams x context must fit the pool): ceil(196,864 / 3,136) + 9 = 72 blocks per stream, ~1,670-block
pool -> 23 streams worst case; 24 chosen (replies rarely use the full 16K reserve). Like-for-like with v3 4 h (38 x 130,816, drain_half): same model, serving, harness, scheduler,
utilization, roster; no throughput-arm components. Run from the worktree root.

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
NAME = "arena-flash-competition-dfr-192k-compact-4h"
RUNTIME = "richardcsaky/arc-arena-hero-v2-dfr-192k-compact-4h-runtime"
UTILIZATION = "0.92"
# v1's int8-unit projection (1,839) minus the headroom: 0.04 x 101,974,081,536 B / 21,196,800 B per int8 block = 192.
SERVED_INT8 = 1839 - 192

hero = components("comp-prune25")
hero.update(expert_prune="prune12", expert_int3=16, window=196608, focus_slots=streams, density_blocks=floor,
            graph_capture=(-(-(streams + 4) // 8) * 8 if streams + 4 > 32 else None),  # <= 32 seqs: default graphs
            backlog_fix=False, budget_policy="priority",
            dfr_harness=True, drain_half=False)
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
