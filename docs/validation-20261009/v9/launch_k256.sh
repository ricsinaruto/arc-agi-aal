#!/bin/bash
# p50 (k256 ckpt-268) 9 h held-out kf read. Usage: launch_k256.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="9 h 11x10 held-out read of p50 = k256 student sft-k256-qad-v1b ckpt-268 in its trained serving format, same harness/roster/seeds/envelope as v9 v1"
export APPROVAL="9 h held-out read of p50 ckpt-268 (approved); launch on the 4th RTX after v9b; v3 relaunch at 46 streams: "
export V87B_STACK=0 V87B_WHY="v9 v1 tree (owner spec docs/v9-submission.md): window 151,040, cpu_cache_gib 0, gate_blocks_per_stream as the bundle (owner density); k256 mask/overlay/INT8 dense by "
WHAT="p50 held-out read: the k256 student (256 of 512 experts, mask t50-r15b) at ckpt-268 served as QAD trained (experts INT3 g16 RTN, dense INT8 W8A16, int4 KV, own MTP) on the v9 v1 stack; plays the 10 held-out games x 11 for 9 h Kaggle-faithfully with live score; judged vs v9 v1 (p75 ckpt-268) on score and score/Mtok at matched tokens and at 9 h, tok/s, KV/hit, broken tool calls (stop > 3% at 10 min)."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6
date -u
date +%s
