"""v6, 8 h Modal preview on the 110 clones. The throughput arm's MTP-1 build
(throughput-arm:docs/validation-20261003/throughput/build_tp_mtp1.py 32 1470 <out> --dependencies --wide-tiles
--indexer-fp8; no request-end tail) plus:
- dfr_loop prompt_fix + inventory_animation (v5's drain system line and interpreter-reset fix; other v5 flags off);
- hx_harness economy + autoexplore + memo (harness arm);
- preview_live_score: live score from the local gateway's open scorecard (Modal preview only).

v6b: autoexplore off; toolbox + reflect style 2 (the harness arm's
hx-meta1h-v1) on, economy and memo kept.

argv: output [--dependencies] [--preview-seconds=N] (default 28,800) [--v6b] [--request-end-tail] [--memo-once]
[--kaggle] (own runtime dataset, no live score; v6d to Kaggle with a 10-minute preview)
[--no-preview] (Kaggle preview queue too long; tested offline on Modal, submitted without a preview)
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import build, kaggle_files  # noqa: E402

from arena.campaign import components  # noqa: E402

out = sys.argv[1]
NAME = Path(out).name
KAGGLE = "--kaggle" in sys.argv
RUNTIME = (f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-competition-')}-runtime" if KAGGLE
           else "richardcsaky/arc-arena-hero-v2-dfr-combined-v3-runtime")  # unused on Modal
PREVIEW = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--preview-seconds=")), 28800))
STREAMS, FLOOR, SPARE, MTP = 32, 1470, 4, 1
SERVED_INT8 = 1839 - 192

hero = components("comp-prune25")
hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=STREAMS, density_blocks=FLOOR,
            graph_capture=(MTP + 1) * (STREAMS + SPARE), backlog_fix=False, budget_policy="priority",
            dfr_harness=True, drain_half=True, mtp_tokens=MTP, mtp_prefix_cache=True, wide_tiles=True,
            indexer_fp8=True,
            dfr_loop={"prompt_fix": True, "inventory_animation": True},
            hx_harness={"economy": True, "autoexplore": True, "memo": True})
if "--request-end-tail" in sys.argv:
    hero["request_end_tail"] = True  # MTP request-end tails (throughput arm phase 3)
if "--v6b" in sys.argv:
    hero["hx_harness"] = {"economy": True, "memo": True, "toolbox": True, "reflect": True, "reflect_style": 2}
if "--memo-once" in sys.argv:
    hero["hx_harness"]["memo_once"] = 1  # v6d: notes shown on change / after a drain / on a new level only
output, study = build(NAME, output=Path(out), dependencies="--dependencies" in sys.argv, paired_30m=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero, hero_v2_served_blocks=SERVED_INT8,
                      exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], opening_burst_requests=3,
                      competition=True, preview_seconds=PREVIEW, preview_instances=110, sanity_trial=False,
                      gpu_memory_utilization="0.92",
                      preview_live_score=not KAGGLE, **({'no_preview': True} if '--no-preview' in sys.argv else {}))
digest = kaggle_files(output, study, smoke=False, runtime_dataset=RUNTIME)
s = study["server"]
print(json.dumps({"output": str(output), "sha256": digest, "streams": study["active_limit"],
                  "window": study["windows"][0], "max_num_seqs": s["max_num_seqs"], "mtp_tokens": s.get("mtp_tokens"),
                  "graphs": s.get("max_cudagraph_capture_size"), "wall": study["wall_seconds"],
                  "notebook_seconds": study["notebook_seconds"], "dfr_loop": study.get("dfr_loop"),
                  "hx_harness": study.get("hx_harness"),
                  "density": study.get("focus_oversubscription", {}).get("gate_blocks_per_stream")}))
