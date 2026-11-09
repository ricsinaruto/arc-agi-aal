#!/bin/sh
# Runs A and B: two parallel 2 h hero-v2 follow-ups, no early stop; run from the worktree root.
set -e
REASONS='"reasons": {"kernels_phase6": "phase-6 GPU test: no speed gain (fused split-K saves nothing; UVA PLE already in serving_tweaks)", "topk_sampler": "kept: the sampler behavior check found the top-k-first path never runs on gameplay batches", "prefix_match_unit": "off: the pre-check warm/cold check was inconclusive (narrowly fails its bar, CIs include 0); user rule: only if it passes"}'
COMMON='"kernels_phase6": false, "prefix_cache": true, "kv_cache": "int4", "focus_oversubscription": true, "prompt_tokens_details": true, "backlog_fix": true, "harness_fixes": true'
ARGS="--runtime-dataset richardcsaky/arc-arena-experiments-runtime --paired-30m --arena-variant fixed-2x-mtp0-wide-kv8-hero --hero-v2-served-blocks 1335 --opening-burst-requests 3 --exclude-games ft09,lp85,ar25,vc33,tu93 --wall-seconds 7200 --clone-instances 110"
uv run --extra dev python scripts/build_arena.py --name arena-hero-v2-int4-streams-v1 --dependencies $ARGS \
  --hero-v2 "{$COMMON, \"focus_slots\": 38, $REASONS}"
uv run --extra dev python scripts/build_arena.py --name arena-hero-v2-128k-streams-v1 --dependencies $ARGS \
  --hero-v2 "{$COMMON, \"window\": 130816, \"focus_slots\": 24, $REASONS}"
