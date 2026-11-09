#!/bin/bash
# v9 9 h held-out kf read. Usage: launch_v9.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=throughput
export REASON="v9 ARC-3 candidate (p75 ckpt-268, all-25 INT3 dn4g8 refit, FP8 dense calib refit, own MTP, qsa_fast, drain jitter, no CPU tier) vs v8.7b, 9 h each on the 11x10 held-out roster, same harness and seeds"
export APPROVAL="build v9 ... then 9 h reads on 11x10 held-out roster: v9 vs v8.7b (approved)."
export V87B_STACK=0 V87B_WHY="v9 by owner spec (docs/v9-submission.md #8-9): window 151,040 for the drain jitter, cpu_cache_gib 0, gate_blocks_per_stream 47.9 (KV 0.8 from A' per-stream KV), 110 held-out instances; KV gate, prefix cache and block-budget code unchanged from v87b"
WHAT="v9 ARC-3 candidate: p75 ckpt-268 (t75), experts INT3 g16 dn4g8 GPTQ-refit on its native all-25 capture, FP8 dense calib refit on ckpt-268, p75's own MTP, qsa_fast decode, per-drain jitter T~U[90K,134K] keep 64K, no CPU KV tier, streams by density rule for KV ~0.8; plays the 10 held-out games x 11 for 9 h Kaggle-faithfully with live score; judged vs v8.7b (b2-v9ref) on score and score/Mtok at matched tokens and at 9 h, tok/s, KV/hit, 0 preemptions."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6
date -u
date +%s
