#!/bin/bash
# v9b Nebius boot + ~1 h play check. Usage: launch_v9b.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="parallel ~1 h Nebius boot + play check of the v9b bundle (router fix) beside the Kaggle 9 h preview v2"
export APPROVAL="YES to the ~1 h Nebius boot + play check, 1 RTX, "
export V87B_STACK=0 V87B_WHY="v9b = v9 tree (window 151,040, cpu_cache_gib 0, gate_blocks_per_stream 47.9) serving p75 ckpt-613 on the AR W4A16 heal-overlay path by "
WHAT="v9b boot + play check: the v9b bundle (p75 ckpt-613 AR INT4 g128 + BF16 dense + own MTP, int4 KV, qsa_fast + gdn_defer, drain jitter, no tier, density streams) boots Kaggle-faithfully and plays the 11x10 held-out roster ~1 h; judged on blocks/streams, KV, prefix hit after ramp, tok/s, MTP acceptance, no Traceback; catches post-load failures before the Kaggle preview's end."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6
date -u
date +%s
