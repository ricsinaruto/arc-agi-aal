"""Throughput arm (3 October): dfr-combined v3 with MTP-1 speculative decoding and the spec-aware prefix cache.

The study is dfr-combined v3 (docs/validation-20261001/dfr-combined/build_comp_dfr_v3.py) with only:
- ``mtp_tokens`` 1 (the 30 September pilot-validated draft stack: BF16 GDN state, NvFp4 draft experts, 16K draft
  vocabulary; 13 fixed blocks per request, blocks of 13 QSA-layer pages);
- ``mtp_prefix_cache`` (arena.prefix_cache_patch ``spec_decode``: keep/demote under MTP-k, no EAGLE last-unit drop);
- streams / floor / sequences / graphs re-sized for the MTP-1 pool (argv), graph_capture = 2 x max_num_seqs.
Same harness, scheduler, window (130,816), utilization (0.92), 1 h preview on the 110 public clones. Not for Kaggle.

argv: streams floor output [--dependencies] [--memory-watch] [--wide-tiles]
``--mtp=2`` serves MTP-2 (graphs 3 x sequences; wide buckets to 96 with --wide-tiles).
``--utilization=X`` replaces the 0.92 GPU-memory utilization (headroom test).
``--indexer-fp8`` adds the hero_v2 ``indexer_fp8`` component (float8 QSA indexer keys).
``--wide-tiles`` adds the hero_v2 ``wide_tiles`` component (wide decode tiles, buckets 40-72).
``--request-end-tail`` adds the hero_v2 ``request_end_tail`` component (MTP-k tails via ``spec_decode``, phase 3).
``--tail-keep-boundary`` adds the hero_v2 ``tail_keep_boundary`` component (phase 4).
``--collapse-log`` adds ARC_TAIL_COLLAPSE_LOG=1 (needs --request-end-tail; ARC_TAIL_COLLAPSE lines in the server log).
``--prune=prune-t37`` etc. serves a tuning-family mask (phase 5); ``--heldout15`` plays the 15 held-out games only.
``--memory-watch`` adds ARENA_MEMORY_WATCH=1 to the trial's server_env (worker GPU-memory growth log; diagnostics).
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import build, kaggle_files  # noqa: E402

from arena.campaign import components  # noqa: E402

streams, floor, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
NAME = Path(out).name
RUNTIME = "richardcsaky/arc-arena-hero-v2-dfr-combined-v3-runtime"  # unused on Modal (offline preview test only)
UTILIZATION = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--utilization=")), "0.92")
SERVED_INT8 = 1839 - 192  # as v3 (the projection only; the startup floor and the gate use the served pool)
PRUNE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--prune=")), "prune12")
AUTOROUND = "--checkpoint=autoround" in sys.argv  # matrix row 1: Intel's AutoRound W4A16 (arena.research_checkpoint)
if AUTOROUND:
    PRUNE = "none"
INT3_ALL = PRUNE == "none-int3"  # matrix row 4: int3 g16 on all 512 experts (overlay via --int3-overlay)
PLAIN = PRUNE in ("none", "none-int3")  # matrix row 2: the plain NVFP4 checkpoint (no prune, no GPTQ-ARC, no int3)
if INT3_ALL:
    SERVED_INT8 -= 48 * 64 * 2_150_400 // 21_196_800
elif PLAIN:
    # all 512 experts at NVFP4 (2,764,800 B) instead of 448 at int3 g16 (2,150,400 B)
    SERVED_INT8 -= 48 * (512 * 2_764_800 - 448 * 2_150_400) // 21_196_800
elif PRUNE != "prune12":
    # Phase 5: the further pruned int3 g16 experts' bytes (2,150,400 B each) as int8 blocks (21,196,800 B).
    from arena.hero_v2 import _prune_spec

    _extra = 48 * (int(512 * _prune_spec(PRUNE)["fraction"]) - 64)
    SERVED_INT8 += _extra * 2_150_400 // 21_196_800
TUNING = ["cd82", "cn04", "ft09", "lp85", "ls20", "re86", "sb26", "sk48", "vc33", "wa30"]
EXCLUDE = TUNING if "--heldout15" in sys.argv else ["ft09", "lp85", "ar25", "vc33", "tu93"]
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
if ONLY:  # v7 refit traffic (5 Oct): only these public games (e.g. ft09,lp85, never recorded before)
    HELDOUT = ["ar25", "bp35", "dc22", "g50t", "ka59", "lf52", "m0r0", "r11l", "s5i5", "sc25", "sp80", "su15", "tn36",
               "tr87", "tu93"]
    EXCLUDE = [g for g in TUNING + HELDOUT if g not in ONLY]
SPARE = 4

hero = components("comp-prune25")
hero.update(expert_prune=None if PLAIN else PRUNE, expert_int3=None if PLAIN and not INT3_ALL else 16, window=130816, focus_slots=streams, density_blocks=floor,
            graph_capture=2 * (streams + SPARE), backlog_fix=False, budget_policy="priority",
            dfr_harness=True, drain_half=True, mtp_tokens=1, mtp_prefix_cache=True)
if PLAIN:
    hero.update(gptq_arc=INT3_ALL, stall_dump=None)  # the stall dump rides on the prune overlay
MTP = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--mtp=")), 1)
hero.update(mtp_tokens=MTP, graph_capture=max(50, (MTP + 1) * (streams + SPARE)))
if "--wide-tiles" in sys.argv:
    hero["wide_tiles"] = True
if "--indexer-fp8" in sys.argv:
    hero["indexer_fp8"] = True
if "--request-end-tail" in sys.argv:
    hero["request_end_tail"] = True  # prefix cache v3 tails; with mtp_prefix_cache the helper's SPEC_TAIL is on
if "--compaction" in sys.argv:
    hero["drain_half"] = None  # phase 5: the summary compaction (arena.compaction) in place of the drop-half drain
if "--tail-spec-acceptance" in sys.argv:
    hero["tail_spec_acceptance"] = True  # tails also for finishes without an in-flight verify step
if "--keep-reply-states" in sys.argv:
    hero["keep_reply_states"] = True  # cause 2: hold superseded decode-crossing states until the request ends
if "--tail-keep-boundary" in sys.argv:
    hero["tail_keep_boundary"] = True  # phase 4: keep the boundary state as the fallback (no swap to the front)
output, study = build(NAME, output=Path(out), dependencies="--dependencies" in sys.argv, paired_30m=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero, hero_v2_served_blocks=SERVED_INT8,
                      exclude_games=EXCLUDE, opening_burst_requests=3,
                      competition=True, preview_seconds=3600, preview_instances=110, sanity_trial=False,
                      no_preview=False, gpu_memory_utilization=UTILIZATION)
OVERLAY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--int3-overlay=")), None)
if "--memory-watch" in sys.argv or "--collapse-log" in sys.argv or OVERLAY or PLAIN:
    import runpy

    from build_arena import manifest

    if "--memory-watch" in sys.argv:
        study["trials"][0]["server_env"]["ARENA_MEMORY_WATCH"] = "1"
    if "--collapse-log" in sys.argv:
        study["trials"][0]["server_env"]["ARC_TAIL_COLLAPSE_LOG"] = "1"
    if OVERLAY:
        # phase 5: another int3 overlay (dataset:manifest_sha256), e.g. the dev-calibrated prune-t12 one
        dataset, digest = OVERLAY.split(":")
        study["trials"][0]["server_env"]["ARENA_EXPERT_OVERLAY"] = json.dumps(
            {"dataset": dataset, "manifest_sha256": digest, "check": "size"})
    if PLAIN:
        # all experts resident: ~13 GiB of KV (10 full windows), below the hero's 14-window sanity floor
        study["focus_oversubscription"]["min_full_windows"] = 8
    if AUTOROUND:
        from arena.research_checkpoint import AUTOROUND_REPO, AUTOROUND_REVISION

        study["research_checkpoint"] = {"repo": AUTOROUND_REPO, "revision": AUTOROUND_REVISION,
                                        "path": "/assets/research-20260930/intel-autoround-w4a16"}
    (output / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((output / "MANIFEST.json").read_text())
    identity["files"] = manifest(output)
    (output / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(output / "bootstrap.py"))["verify"](output)
digest = kaggle_files(output, study, smoke=False, runtime_dataset=RUNTIME)
s = study["server"]
print(json.dumps({"output": str(output), "sha256": digest, "streams": study["active_limit"],
                  "window": study["windows"][0], "max_num_seqs": s["max_num_seqs"], "mtp_tokens": s.get("mtp_tokens"),
                  "graphs": s.get("max_cudagraph_capture_size"), "capture_sizes": s.get("cudagraph_capture_sizes"),
                  "utilization": s["gpu_memory_utilization"], "prefix_cache": study["cache_experiment"]["prefix_cache"],
                  "server_env": study["trials"][0]["server_env"],
                  "density": study.get("focus_oversubscription", {}).get("gate_blocks_per_stream")}))
