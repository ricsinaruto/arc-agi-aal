#!/bin/bash
# c893 native capture on the 15 train games: MoE + dense inputs for the INT3 dn4g8 refit
cd .
B=./.claude/worktrees/sft-prune/dist/cap-v85p-p75c893-tr15-v1
export ARC_ARM=arc3sub
export REASON="INT3 dn4g8 refit of p75 ckpt-893 on the 15 train games; this is its native capture, inputs-r15-v1"
export APPROVAL="arc3sub 2 RTX; capture 32 min "
export V87B_STACK=0 V87B_WHY="calibration capture (prefill-only expert_quant capture_only, no play): A's capture bundle with c893's NVFP4 RTN + dense overlay; the v87b KV/stream rules govern play runs"
WHAT="c893 calibration capture: the v8.5p served stack serving p75 ckpt-893 (t75 nodrop mask, overlay = NVFP4 RTN of its 128 kept experts + its dense/MTP shards) prefills inputs-r15-v1 (the 15 RL training games only) and saves the MoE inputs of all 48 layers. Why: input to the GPTQ W4, INT3 g16 and dn4g8 refit the ed for the 100K held-out read. Judged by a complete capture, 48 layers."
bash <scratch>/limits/launch_rtx_appr.sh b2-tp-cap-v85p-p75c893-tr15-v1 $B "$WHAT" 2>&1 | tail -5
date -u
date +%s
