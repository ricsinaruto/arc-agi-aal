"""v7.1: v7 (below) + the drain fix, the python-time fixes and the strict loop rule
(``loop_rule`` fresh_context); ``--keep-tokens=N`` drains to N kept tokens (keep-target A/B); ``--heldout15`` plays the
15 held-out games (standard held-out matrix).

v7: v6d (docs/validation-20261004/build_v6_8h.py --v6b --request-end-tail --memo-once) with:
1. prefix-cache fixes: tail_spec_acceptance v3, keep_reply_states, tail_keep_boundary (throughput arm LOG 17-18);
2. harness_fixes text_count + API server --renderer-num-workers 4 (b200 9ceda19);
3. no hx toolbox, memo, reflect, economy, and no stuck notice (harness_fixes without stuck_notice, hx no_stuck_notice):
   B200 run 1's harness;
4. NVFP4 draft experts, arena16k draft vocab, wide tiles kept (router shape and buckets follow the mask / streams);
5. experts: PRUNE mask + its int3 g16 overlay (arena.variants EXPERT_OVERLAY_INT3_PRUNED), Humming;
6. streams 44 at v6d's gate density (44.093 blocks per stream), scaled down at startup by the exact KV gate.

argv: output [--dependencies] [--preview-seconds=N] (default 3,600) [--kaggle] [--no-preview]
[--mask=data/expert_prune/<level>.json --overlay=<owner/dataset>:<manifest sha256>] (default: the dev prune-t37 mask and
its overlay in arena.variants; another mask is registered as level <stem> for this build; uniform masks only)
[--streams=N] (default 44; 32 with --p12) [--p12] (v6d experts: prune12 + arc-expert-int3-g16-v1, v6d streams)
[--compact] (fraction compaction instead of drains: summary of the oldest 90%, newest 10% verbatim)
[--stagger=F] (first drain / compaction of each game at F..1 x the budget, to keep clones out of phase)
[--window=N] (default 130,816; a smaller window lets more streams fit the same pool: about 130,816 x 44 / streams)
[--draft-head-overlay=DIR] (a fine-tuned MTP draft head from arena.draft_head_train, copied into the bundle and pinned;
fit on DEV10 for tests, refit on all 25 games for a submission)
[--priority-floor=off] (arena.priority ``floor`` false: float priorities) [--dfr=FLAG ...] (arena.dfr_harness opt-in
flags: transient, reset_offer, border_moves, sweep) [--version=NAME] (the b200 block's version, default v7.1)
[--all-experts --overlay=owner/dataset:sha256] [--restore-v6d] [--wall=N] [--mm-cache-gb=X|default] (default 1: --mm-processor-cache-gb 1; "default" leaves vLLM's 4)
"""
import json
import math
import runpy
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import build, kaggle_files, manifest  # noqa: E402

from arena import expert_overlay, hero_v2, variants  # noqa: E402
from arena.campaign import FIXES, components  # noqa: E402
from arena.hero_v2 import _prune_spec  # noqa: E402


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)

out = sys.argv[1]
NAME = Path(out).name
KAGGLE = "--kaggle" in sys.argv
RUNTIME = (f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-competition-')}-runtime" if KAGGLE
           else "richardcsaky/arc-arena-hero-v2-dfr-combined-v3-runtime")  # unused on Modal
PREVIEW = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--preview-seconds=")), 3600))
PRUNE = "prune-t37"
MASK, OVERLAY = arg("mask"), arg("overlay")
ALL_EXPERTS = "--all-experts" in sys.argv
RESTORE_V6D = "--restore-v6d" in sys.argv
WALL = int(arg("wall", "0"))
if ALL_EXPERTS and (MASK or not OVERLAY or "--p12" in sys.argv):
    raise SystemExit("--all-experts needs --overlay, without --mask or --p12")
if not ALL_EXPERTS and bool(MASK) != bool(OVERLAY):
    raise SystemExit("--mask and --overlay go together")
if MASK:
    # The final all-games refit: the mask and its int3 overlay registered under the mask's stem (build-time tables
    # only; the bundle ships the mask and pins it and the overlay manifest by sha256).
    import hashlib

    mask = json.loads(Path(MASK).read_text())
    per_layer = {len(v) for v in mask["pruned"].values()}
    if mask.get("format") != "arc-expert-prune-v1" or mask["experts"] != 512 or len(per_layer) != 1:
        raise SystemExit(f"{MASK}: not a uniform arc-expert-prune-v1 mask over 512 experts")
    PRUNE = Path(MASK).stem
    fraction = per_layer.pop() / 512
    if fraction != mask["fraction"]:
        raise SystemExit(f"{MASK}: fraction {mask['fraction']} vs {fraction} pruned per layer")
    dataset, digest = OVERLAY.split(":")
    variant = f"fixed-2x-mtp0-wide-kv8-hero-{PRUNE}"
    variants.EXPERT_PRUNE_VARIANTS[variant] = {
        "fraction": fraction, "source": MASK, "mask": f"configs/expert-prune/{PRUNE}.json",
        "sha256": hashlib.sha256(Path(MASK).read_bytes()).hexdigest(),
        "served_blocks": variants.EXPERT_PRUNE_VARIANTS["fixed-2x-mtp0-wide-kv8-hero-prune-t37"]["served_blocks"]}
    hero_v2.PRUNE_LEVELS[PRUNE] = variant
    variants.EXPERT_OVERLAY_INT3_PRUNED[(PRUNE, 16)] = expert_overlay.validate(
        {"dataset": dataset, "manifest_sha256": digest, "check": "size"})
P12 = "--p12" in sys.argv  # v7p12 (5 October): v6d's experts (prune12 + int3-g16-v1, on Kaggle) and v6d's 32 streams
if P12 and MASK:
    raise SystemExit("--p12 serves v6d's mask and overlay")
if P12:
    PRUNE = "prune12"
STREAMS, SPARE, MTP = int(arg("streams", 32 if P12 else 44)), 4, 1
V6D_GATE_BLOCKS_PER_STREAM = 44.093  # v6d: floor(0.96 x 1,470) / 32
# The smallest served pool whose gate density (floor(0.96 x FLOOR) / STREAMS, in milli-blocks) is v6d's or more.
FLOOR = next(f for f in range(math.ceil(V6D_GATE_BLOCKS_PER_STREAM * STREAMS / 0.96), 4000)
             if int(0.96 * f) * 1000 // STREAMS >= V6D_GATE_BLOCKS_PER_STREAM * 1000)
FLOOR = int(arg("density-blocks", FLOOR))  # stream sweep (7 October): more streams on the same pool
RENDERER_WORKERS = 4
# v7 host memory (5 October): vLLM's processed-image LRU (4 GiB default, in the API server and the engine core).
_mm = arg("mm-cache-gb", "1")
MM_CACHE_GB = None if _mm == "default" else float(_mm)
# v6d's projection (prune12 int8-equivalent blocks) + the further pruned int3 g16 experts' bytes as int8 blocks.
SERVED_INT8 = 1839 - 192 + 48 * (int(512 * _prune_spec(PRUNE)["fraction"]) - 64) * 2_150_400 // 21_196_800

TUNING = ["cd82", "cn04", "ft09", "lp85", "ls20", "re86", "sb26", "sk48", "vc33", "wa30"]
HELDOUT = "ar25 bp35 dc22 g50t ka59 lf52 m0r0 r11l s5i5 sc25 sp80 su15 tn36 tr87 tu93".split()
EXCLUDE = HELDOUT if "--dev10" in sys.argv else TUNING if "--heldout15" in sys.argv else ["ft09", "lp85", "ar25", "vc33", "tu93"]
KEEP = arg("keep-tokens")
LOOP_RULE = {"action": arg("loop-rule", "fresh_context")}

hero = components("comp-prune25")
hero.update(expert_prune=PRUNE, expert_int3=16, window=int(arg("window", 130816)), focus_slots=STREAMS, density_blocks=FLOOR,
            graph_capture=(MTP + 1) * (STREAMS + SPARE), backlog_fix=False, budget_policy="priority",
            dfr_harness=True, drain_half={"keep_tokens": int(KEEP)} if KEEP else True, mtp_tokens=MTP, mtp_prefix_cache=True, wide_tiles=True,
            indexer_fp8=True, request_end_tail=True,
            tail_spec_acceptance=True, keep_reply_states=True, tail_keep_boundary=True,
            dfr_loop={"prompt_fix": True, "inventory_animation": True},
            hx_harness={"no_stuck_notice": True},
            harness_fixes=[f for f in FIXES + ["summary_headroom", "text_count"] if f != "stuck_notice"])
if ALL_EXPERTS:
    hero.update(expert_prune=None, expert_int3=None, gptq_arc=False, stall_dump=None)
DFR = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--dfr=")]
if DFR:  # hx4 opt-in flags on top of the six v7.1 ones (arena.dfr_harness validates the names)
    from arena.dfr_harness import settings as dfr_settings

    hero["dfr_harness"] = {**dfr_settings(True), **{flag: True for flag in DFR}}
if arg("priority-floor", "on") not in ("on", "off"):
    raise SystemExit("--priority-floor=on|off")
if arg("priority-floor") == "off":
    hero["priority"] = {"floor": False}
if RESTORE_V6D:
    hero.update(hx_harness={"economy": True, "memo": True, "toolbox": True, "reflect": True,
                            "reflect_style": 2, "memo_once": 1},
                harness_fixes=FIXES + ["summary_headroom", "text_count"])
output, study = build(NAME, output=Path(out), dependencies="--dependencies" in sys.argv, paired_30m=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero, hero_v2_served_blocks=None if ALL_EXPERTS else SERVED_INT8,
                      exclude_games=EXCLUDE, opening_burst_requests=3,
                      competition=True, preview_seconds=min(PREVIEW, 28800), preview_instances=int(arg("instances", 110)), sanity_trial=False,
                      gpu_memory_utilization="0.92",
                      preview_live_score=not KAGGLE, draft_head_overlay=arg("draft-head-overlay"),
                      **({'no_preview': True} if '--no-preview' in sys.argv else {}))
study["server"]["renderer_num_workers"] = RENDERER_WORKERS
if "--compact" in sys.argv:
    # Fraction compaction (codex/sft-pilot, ported 7 October): at a full context the oldest 90% becomes a summary of at
    # most 4,096 tokens and the newest 10% stays verbatim; drain_half stays configured but is not reached.
    study.update(compaction_keep_fraction=.1, compaction_summary_cap=4096, compaction_thinking="gameplay")
    if "--compact-full-prefix" in sys.argv:  # the summary request on the whole cached context (arena.compaction)
        study["compaction_full_prefix"] = True
if "--collapse-log" in sys.argv:  # diagnostics: one ARC_TAIL_COLLAPSE line per full-prefix miss in the server log
    study["cache_experiment"]["prefix_cache"]["collapse_log"] = True
if "--reply-ids" in sys.argv:  # API server: a re-sent reply keeps its generated token ids (arena.reply_ids_runtime)
    study["cache_experiment"]["prefix_cache"]["reply_ids"] = True
if "--keepalive" in sys.argv:  # a game idle past 5 s keeps its cached blocks fresh in the eviction queue
    study["cache_experiment"]["prefix_cache"]["tail_keepalive"] = True
if "--state-cotenant" in sys.argv:
    # PLE short-conv state as a co-tenant of GDN group 0's blocks (arena.cache_patch): 13 -> 10 fixed blocks per request.
    study["cache_experiment"]["prefix_cache"]["state_cotenant"] = True
if arg("stagger"):
    # Per-game first drain / compaction trigger spread over [N, 1] x the budget (arena.compaction).
    study["first_trigger_stagger"] = {"min_fraction": float(arg("stagger"))}
if LOOP_RULE["action"] != "off":
    from arena.loop_rule import settings as loop_rule_settings

    study["loop_rule"] = loop_rule_settings(LOOP_RULE)
if MM_CACHE_GB is not None:
    study["server"]["mm_processor_cache_gb"] = MM_CACHE_GB
if ALL_EXPERTS:
    dataset, digest = OVERLAY.split(":")
    study["server"]["expert_overlay"] = expert_overlay.validate(
        {"dataset": dataset, "manifest_sha256": digest, "check": "size"})
    study["focus_oversubscription"].update(min_full_windows=8, min_slots=8)
if WALL:
    if not 0 < WALL <= 82200 or KAGGLE:
        raise SystemExit("--wall is a Modal run of at most 82,200 gameplay seconds")
    study["wall_seconds"] = study["competition"]["preview_seconds"] = WALL
    study["notebook_seconds"] = int(arg("notebook", WALL + 3900))
    study["b200"] = {"test": "b200-run", "harness": "arena", "gpu": "RTX-PRO-6000",
                     "stop": {"score_target": 95.0, "wall_seconds": WALL, "usd_limit": None},
                     "version": arg("version", "v7.1")}
(output / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
identity = json.loads((output / "MANIFEST.json").read_text())
identity["files"] = manifest(output)
(output / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
runpy.run_path(str(output / "bootstrap.py"))["verify"](output)
digest = kaggle_files(output, study, smoke=False, runtime_dataset=RUNTIME)
if KAGGLE:
    # the pinned Python 3.12 image, as on v6d-kaggle / v6d-sub (the builder omits it).
    meta_path = output / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta.update(docker_image="gcr.io/kaggle-private-byod/python@sha256:"
                             "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                docker_image_pinning_type="original")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
s = study["server"]
print(json.dumps({"output": str(output), "sha256": digest, "streams": study["active_limit"],
                  "window": study["windows"][0], "max_num_seqs": s["max_num_seqs"], "mtp_tokens": s.get("mtp_tokens"),
                  "graphs": s.get("cudagraph_capture_sizes"), "wall": study["wall_seconds"],
                  "notebook_seconds": study["notebook_seconds"], "prefix_cache": study["cache_experiment"]["prefix_cache"],
                  "expert_prune": s.get("expert_prune"), "expert_overlay": s.get("expert_overlay"),
                  "model_datasets": study.get("model_datasets"), "renderer_num_workers": s.get("renderer_num_workers"),
                  "mm_processor_cache_gb": s.get("mm_processor_cache_gb"),
                  "hx_harness": study.get("hx_harness"), "harness_fixes": (study.get("harness_fixes") or {}).get("fixes"),
                  "density": study.get("focus_oversubscription", {}).get("gate_blocks_per_stream"),
                  "drain_half": study.get("drain_half"), "loop_rule": study.get("loop_rule"), "games": len(study["games"]),
                  "preview_seconds": study["competition"].get("preview_seconds"),
                  "dfr_harness": study.get("dfr_harness"), "priority_floor": study["priority"].get("floor"),
                  "draft_head": s.get("draft_head"), "version": (study.get("b200") or {}).get("version")}))
