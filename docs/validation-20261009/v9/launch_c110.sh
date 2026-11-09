#!/bin/bash
# 9 h c110 pair (9 h reads outside the arm's 4 RTX). Usage: launch_c110.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export ARC_ALLOWANCE_EXEMPT="9 h reads sit outside arc3sub's 4 RTX, within the global cap"
export REASON="9 h c110 pair V9C vs p50 ckpt-647, same roster and seeds, gap at 300K; V9C scored 37 hidden vs v9 39 at 6 h 44, both under v8.5p 45.4"
[ "$RUN" = b2-k256c268-kf-c110-9h-v1 ] && REASON="9 h c110 read of p50-sub, completes the same-roster comparison with V9C c893 and c647; hidden p50 44.41, V9C 36.88"
export APPROVAL="approval past 2 h for both 9 h reads"
if [ "$RUN" = b2-p50d7tc942-kf-c110-9h-v1 ]; then
  export REASON="9 h c110 read of the p50d7-sub kernel tree (p50d7 ckpt-942) in full mode, anchors p50-sub / v9 / v87b c110 at matched hours and tokens"
  export APPROVAL="approval past 2 h for this 9 h read"
  export ARC_ALLOWANCE_EXEMPT="9 h reads sit outside arc3sub's 4 RTX, within the global cap"
fi
export V87B_STACK=0 V87B_WHY="submitted v9 tree: window 151,040, cpu_cache_gib 0, density stream formula from the bundle; KV gate, prefix cache and block budget code as v87b"
case $RUN in *v9c*) M="V9C as submitted: p75 c893 INT3 dn4g8 refit, INT8 dense, own MTP, GDN + QSA";; *p50d7tc942*) M="p50d7-sub kernel tree as pushed: p50d7 ckpt-942 (41 layers), INT3 overlay 3d689abf, d7 mask, BF16 dense, boot stream formula";; *k256c268*) M="p50-sub / p50 v3 weights: k256 ckpt-268, INT3 direct export, INT8 dense, kv4, own MTP";; *) M="p50 at ckpt-647 of sft-k256-kd: k256 mask, INT3 direct export, INT8 dense, kv4, own MTP, p50 v3's recipe";; esac
WHAT="9 h c110 read: $M, plays 110 cyclic clones of the 20 public games, park 300K per game, no level budget. Why: a same-roster pair of V9C and p50 ckpt-647, the gap at 300K tokens that 100K reads cannot see. Judged by score, levels and tokens at hours 1-9 and matched tokens, paired by clone and by game, late clears 200-300K per 100 instances; KV ~.8, hit >= .95, 0 preemptions."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-260
date -u; date +%s
