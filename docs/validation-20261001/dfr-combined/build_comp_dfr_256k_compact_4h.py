"""Four-hour variant ("run the 256k run for 4 hours"). Modal research build: dfr-combined v3 with "256k context (scale streams accordingly) + context
compaction": window 261,120 (the largest multiple of 1,024 whose request, window + 256, fits max_model_len 262,144) and
the model-written summary compaction in place of drain_half (prompt budget 244,224; compaction leaves ~12K, so the mean
context is ~128K, 56 int4 blocks per request with the 16K reply: 28 streams on the expected ~1,668-block pool at 0.92,
27 on the 1,600 floor). Everything else is v3 (priority gate: handovers at compactions; dfr_harness; 0.92). Run from the
worktree root.

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
NAME = "arena-flash-competition-dfr-256k-compact-4h"
RUNTIME = "richardcsaky/arc-arena-hero-v2-dfr-256k-compact-4h-runtime"
UTILIZATION = "0.92"
# v1's int8-unit projection (1,839) minus the headroom: 0.04 x 101,974,081,536 B / 21,196,800 B per int8 block = 192.
SERVED_INT8 = 1839 - 192

hero = components("comp-prune25")
hero.update(expert_prune="prune12", expert_int3=16, window=261120, focus_slots=streams, density_blocks=floor,
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
