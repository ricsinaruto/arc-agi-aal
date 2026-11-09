#!/bin/bash
# c893 INT3 dn4g8 100K held-out read, on SFT's idle RTX by word.
cd .
RUN=b2-int3dn4g8-p75c893-ho10x6-100k-v1
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=${ARM:-sft}
export REASON="one 100K held-out read of p75 ckpt-893 with a v8.7b-style INT3 dn4g8 refit on the 15 train games; pairs by game with b2-int8-p75c893-ho10x6-100k-v1"
export APPROVAL="SFT's idle RTX allowed by "
export V87B_STACK=0 V87B_WHY="v9b tree as the INT8 read (window 151,040, no CPU KV, density 47.9, MTP 2, KV gate and prefix caching as v9b), held at the anchor's 53 streams so ms/step and tok/s pair; token_stop ported"
WHAT="Held-out play read: p75 c893 with experts INT3 g16 dn4g8 refit on the 15 train games only, INT8 dense, int4 KV: 10 held-out games x 6, 100K tokens per try, 53 streams. Why: the V9C ingredient, on the 15 so the read is clean held-out. Judged by score at 25K/50K/100K paired by game against the INT8 read (22.86, 10.33, 4.68), tok/s and ms/step; KV and prefix hit reported."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
