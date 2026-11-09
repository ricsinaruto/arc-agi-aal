#!/bin/bash
# p50d7 9 h held-out read (9 h reads outside the arm's 4 RTX). Usage: launch_p50d7.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export ARC_ALLOWANCE_EXEMPT="9 h reads sit outside arc3sub's 4 RTX, within the global cap"
export REASON="9 h 11x10 held-out read of the latest sft-p50d7 checkpoint, pairs with b2-k256c268-kf-ho10x11-9h-v3 (63.72 / 656 / 46.4M), no pass bar"
export APPROVAL="approval past 2 h for this 9 h read"
export V87B_STACK=0 V87B_WHY="p50 v3 tree: window 151,040, cpu_cache_gib 0, gate_blocks_per_stream as the bundle; p50d7 overlay, drop_layers mask, BF16 dense by choice; KV gate, prefix cache and block budget code as v87b"
WHAT="p50d7 held-out read: the 41-layer p50 student from sft-p50d7 (256 experts, INT3 experts, BF16 dense, int4 KV, own MTP) on p50 v3's harness: 10 held-out games x 11 for 9 h Kaggle-faithfully with live score. Why: does the 7-layer cut plus one more pass hold p50's score. Judged by score, levels, tokens and tokens per action at matched tokens and at 9 h vs p50 c268 63.72 / 656 / 46.4M, V9C 54.78 / 559 / 37.4M, v9 58.28, paired by clone."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-260
date -u; date +%s
