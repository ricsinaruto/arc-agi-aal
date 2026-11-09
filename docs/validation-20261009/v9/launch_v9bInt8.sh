#!/bin/bash
# v9b + INT8 dense load test; 
cd .
RUN=b2-v9b-p75c613-int8-check-v1
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="next p75 leg is INT8 dense + AR W4A16 experts + int4 KV; confirm it loads and serves in v9b's stack before SFT's 14:30 UTC run"
export APPROVAL=" 1 RTX, ~30 min, inside the arc3sub cap"
export V87B_STACK=0 V87B_WHY="v9b = v9 tree (window 151,040, cpu_cache_gib 0, gate_blocks_per_stream 47.9) serving p75 ckpt-613 on the AR W4A16 heal-overlay path by this run adds only dense_kernels.int8_weights"
WHAT="v9b INT8 load test: the v9b 9 h stack with c613's AR overlay and int4 KV, adding INT8 dense (lm_head, GDN qkvz, QSA qkv, attn out) quantized at load, plays 10 min at the usual density. Judged as a load/serve check: ARC_DENSE_CHECK passes, INT8 layers counted, KV blocks, tok/s and 0 Tracebacks; no quality read."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-280
date -u +%H:%M:%S
