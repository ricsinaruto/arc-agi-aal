#!/bin/bash
# v9 expert refit (docs/v9-submission.md #3): masked W4 GPTQ of ckpt-268 masters (t75 nodrop, native routing) on the
# all-25 ckpt-268 capture, INT3 g16 requant, ckpt-268 dense/MTP merged; throughput's tp-heldout pipeline (2aec47b59)
cd ./.claude/worktrees/agent-aad6619d37e5f5311/scripts/dfr_exact/int3 || exit 1
export HK_RUN=tp-v9-refit-c268-v1
./.venv/bin/modal run --detach modal_heldout_refit.py --stage all --capture-run b2-tp-cap-v85p-p75c268-all25-v1 \
  --mask prune-t75-nodrop-excl5-v1.json --masters /prolong/validation/sft-p75-cur20-c231-v1/train/ckpt-268/student \
  --dense sft-heal/nvfp4-dense/p75-c268 \
  --w4 arc-expert-gptq-p75c268-all25-v1 --out arc-expert-int3-requant-p75c268-all25-v1 2>&1 | grep -E "function_call_id|Error|ap-" | head -5
