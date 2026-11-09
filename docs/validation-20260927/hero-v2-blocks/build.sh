#!/bin/sh
# hero-v2-blocks: the three-block scheduler on the c110 roster; run B's harness and hero-v2's
# serving on int4 KV with --prefix-match-unit 16; 37 streams (a block), max_num_seqs 41, graphs to 41, the window from
# the gate replay (window-replay*.json); the scheduler's wall is 30,600 s and the run stops at 7,200 s. From the
# worktree root.
set -e
REASONS='"reasons": {"kernels_phase6": "phase-6 GPU test: no speed gain (fused split-K saves nothing; UVA PLE already in serving_tweaks)", "topk_sampler": "kept: the sampler behavior check found the top-k-first path never runs on gameplay batches", "prefix_match_unit": "on: prefix-unit-int4 v2 passed on the int4 layout (docs/validation-20260927/prefix-unit-int4.md; partial-tail hits 96/96 exact, R_P 0.993 [0.974, 1.011])", "window": "93,184: the largest window at which 37 steady streams pass the gate-replay bar on both run A and run B traces (hero-v2-blocks.md)", "harness_fixes": "run B eight sub-fixes plus summary_headroom; inventory_imports off"}'
FIXES='["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox", "timeout_interrupt", "prompt_text", "summary_headroom"]'
COMMON='"kernels_phase6": false, "prefix_cache": true, "kv_cache": "int4", "focus_oversubscription": true, "prompt_tokens_details": true, "prefix_match_unit": 16'
BLOCKS='"budget_policy": "blocks", "blocks": {"stagger_seconds": 0}, "focus_slots": 37, "spare_sequences": 4, "density_blocks": 1260, "graph_capture": 41, "window": 93184'
ARGS="--runtime-dataset richardcsaky/arc-arena-experiments-runtime --paired-30m --arena-variant fixed-2x-mtp0-wide-kv8-hero --hero-v2-served-blocks 1335 --opening-burst-requests 3 --exclude-games ft09,lp85,ar25,vc33,tu93 --wall-seconds 30600 --clone-instances 110"
uv run --extra dev python scripts/build_arena.py --name arena-hero-v2-blocks-v1 --dependencies $ARGS \
  --hero-v2 "{$COMMON, $BLOCKS, \"harness_fixes\": $FIXES, $REASONS}" \
  --early-stop '{"at_seconds": 7200, "always": true}'
