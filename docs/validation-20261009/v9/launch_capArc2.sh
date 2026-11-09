#!/bin/bash
# Native c240 capture of SFT's ARC-2 p75 finetune (v55b plain INT3 refit). Usage: launch_capArc2.sh CKPT [RUN]
cd .
CK=$1; RUN=${2:-b2-tp-cap-p75arc2-c240-v1}; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="plain INT3 refit of SFT's ARC-2 p75 $CK for v55b, fit on c240; this is its native capture"
export APPROVAL="inside arc3sub's RTX, under 2 h, "
export V87B_STACK=0 V87B_WHY="calibration capture (prefill-only expert_quant capture_only, no play): the c893 capture bundle with ARC-2 c240 inputs; the v87b KV/stream rules govern play runs"
WHAT="ARC-2 calibration capture: serves SFT's ARC-2 p75 $CK (t75 nodrop mask, overlay = NVFP4 RTN of its 128 kept experts + its dense/MTP) and prefills the 365 c240 attempt prefixes (9.5M tokens, all 120 tasks), saving the MoE inputs of all 48 layers. Why: input to the plain INT3 g16 refit for v55b. Judged by a complete capture: 365 prefixes, 48 layers."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -5 | cut -c1-260
date -u; date +%s
