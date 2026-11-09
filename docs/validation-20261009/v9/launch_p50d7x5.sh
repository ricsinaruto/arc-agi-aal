#!/bin/bash
# p50d7 10 x 5 held-out read (9 h reads outside the arm's 4 RTX). Usage: launch_p50d7x5.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export ARC_ALLOWANCE_EXEMPT="9 h reads sit outside arc3sub's 4 RTX, within the global cap"
export REASON="${X5_REASON:-10 x 5 excl5 held-out read of the last sft-p50d7-top20-c731-v1 checkpoint vs b2-p50d7c294-kf-ho10x11-9h-v2, bar +.3 levels a clone or +3 score, interval above 0; SFT resumes training on a pass}"
export APPROVAL="approval past 2 h for this read"
export V87B_STACK=0 V87B_WHY="p50d7 v2 tree: window 151,040, cpu_cache_gib 0, gate_blocks_per_stream as the bundle; weights and roster by choice; KV gate, prefix cache and block budget code as v87b"
WHAT="${X5_WHAT:-p50d7 top-20 held-out read: the 41-layer p50 student at the end of sft-p50d7-top20-c731-v1 on the p50d7 v2 stack: 10 clones each of ft09 vc33 lp85 tu93 ar25, stopped at ~20M tokens. Why: is the top-20% resume meaningfully better. Judged paired by clone at matched tokens vs the prior p50d7 read: levels a clone, score, tokens per action; bar +.3 levels or +3 score, interval above 0, resampling clones.}"
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-260
date -u; date +%s
