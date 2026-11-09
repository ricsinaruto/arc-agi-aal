#!/bin/bash
# v9 native capture (docs/v9-submission.md #5): ckpt-268 on all 25 games, MoE + dense inputs
cd .
B=./.claude/worktrees/sft-prune/dist/cap-v85p-p75c268-all25-v1
export ARC_ARM=throughput
export REASON="v9 = p75 ckpt-268 with all-25 INT3 dn4g8 refit + v8.7b FP8 dense calib refit; this is its native capture (docs/v9-submission.md)"
export APPROVAL="build v9 ARC-AGI-3 candidate ... v8.7b-style INT3 refit (dn4g8) on all 25 games, v8.7b dense FP8 calibration"
export V87B_STACK=0 V87B_WHY="calibration capture (prefill-only expert_quant capture_only run, no play): A's capture bundle with the all-25 inputs and ckpt-268's NVFP4 RTN + dense overlay; the v87b KV/stream rules govern play runs"
WHAT="v9 calibration capture: the v8.5p served stack serving p75 ckpt-268 (t75 nodrop mask; overlay = NVFP4 RTN of its 128 kept experts + its trained dense/MTP shards) prefills inputs-all25-v1 (v7.1 transcripts of all 25 games, the v8.7b fit's inputs) and persists the MoE inputs of all 48 layers and the dense inputs (attn_out, gdn in_proj, qsa_qkv); input to v9's GPTQ W4 + INT3 g16 dn4g8 refit and FP8 dense calibration. = A's capture bundle except inputs and overlay."
bash <scratch>/limits/launch_rtx_appr.sh b2-tp-cap-v85p-p75c268-all25-v1 $B "$WHAT" 2>&1 | tail -5
date -u
date +%s
