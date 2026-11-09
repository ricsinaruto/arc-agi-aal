#!/bin/bash
# v9 v2 = v1 + 48 GiB CPU KV tier, streams by the density rule on measured gate reserve. Usage: launch_v9v2.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=throughput
export REASON="+ ~22:10 UTC 29 Oct: v9 v3 (v2 rebuilt at 60 streams) = exactly v1 (same bundle code, 10 held-out x 11, 9 h, kaggle_faithful) with one change: a 48 GiB CPU KV tier (ArcOffloadingConnector, offload_prompt_only false) and streams rescaled by the density rule"
export APPROVAL="launch v2 (rebuilt at 60 streams as v3 by ); explicit approval = the 9 h approval."
export V87B_STACK=0 V87B_WHY="v9 by owner spec (docs/v9-submission.md #8-9) + 48 GiB tier: window 151,040 for the drain jitter, cpu_cache_gib 48, gate_blocks_per_stream 41.666 (v8.7b formula), 110 held-out instances; KV gate, prefix cache and block-budget code unchanged from v87b"
WHAT="v9 with a 48 GiB CPU KV tier: identical to the live v9 read b2-v9-...-v1 (p75 ckpt-268, INT3 dn4g8 all-25 refit, FP8 calib refit, own MTP, qsa_fast, GDN deferred commit, drain jitter) plus the CPU tier, 60 streams (v8.7b density 41.666); plays the 10 held-out games x 11 for 9 h Kaggle-faithfully with live score; judged vs v1 and v8.7b on score and score/Mtok at matched tokens and at 9 h, tok/s, prefix hit, 0 preemptions."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6
date -u
date +%s
